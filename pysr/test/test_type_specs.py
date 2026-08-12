import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pysr import PySRRegressor, TypeSpec, jl
from pysr.expression_specs import ExpressionSpec
from pysr.type_specs import (
    build_type_spec_module_source,
    load_type_spec_runtime,
    object_array_1d,
    object_array_2d,
    type_spec_to_julia_array,
    type_spec_to_python_array,
)


def string_spec(**overrides):
    parameters = {
        "fields": {"data": "String"},
        "init_value": '() -> Value("")',
        "sample_value": '(rng, options) -> Value(rand(rng, ("a", "b")))',
        "mutate_value": (
            '(rng, value, temperature, options) -> Value(rand(rng, ("a", "b")))'
        ),
        "count_scalar_constants": 1,
        "is_valid": "value -> true",
        "can_optimize": False,
    }
    parameters.update(overrides)
    return TypeSpec(**parameters)


def vector_spec(**overrides):
    parameters = {
        "fields": {"data": "Vector{Float64}"},
        "init_value": "() -> Value([1.0, 2.0])",
        "sample_value": "(rng, options) -> Value([3.0, 4.0])",
        "mutate_value": "(rng, value, temperature, options) -> value",
        "count_scalar_constants": "value -> length(value.data)",
        "is_valid": "value -> all(isfinite, value.data)",
        "can_optimize": True,
        "pack_scalar_constants": (
            "(buffer, idx, value) -> " "(buffer[idx:idx+1] .= value.data; idx + 2)"
        ),
        "unpack_scalar_constants": (
            "(buffer, idx, value) -> " "(idx + 2, Value(copy(buffer[idx:idx+1])))"
        ),
        "number_type": "Float64",
    }
    parameters.update(overrides)
    return TypeSpec(**parameters)


def module_source(
    spec,
    operators=None,
    *,
    elementwise_loss=("value_loss(x::Value, y::Value)::Float64 = x == y ? 0.0 : 1.0"),
    loss_function=None,
    loss_function_expression=None,
):
    return build_type_spec_module_source(
        spec,
        operators or {1: ["identity_value(x::Value) = x"]},
        elementwise_loss=elementwise_loss,
        loss_function=loss_function,
        loss_function_expression=loss_function_expression,
    )


def tiny_model(spec, *, parallelism="serial", procs=None, **overrides):
    parameters = {
        "type_spec": spec,
        "operators": {1: ["identity_value(x::Value) = x"]},
        "elementwise_loss": (
            "value_loss(x::Value, y::Value)::Float64 = " "x == y ? 0.0 : 1.0"
        ),
        "niterations": 1,
        "ncycles_per_iteration": 2,
        "populations": 1,
        "population_size": 8,
        "tournament_selection_n": 3,
        "maxsize": 7,
        "parallelism": parallelism,
        "procs": procs,
        "deterministic": parallelism == "serial",
        "random_state": 0 if parallelism == "serial" else None,
        "progress": False,
        "verbosity": 0,
        "temp_equation_file": True,
        "should_optimize_constants": False,
    }
    parameters.update(overrides)
    return PySRRegressor(**parameters)


class _TypeSpecContractTests:
    def test_requires_explicit_lifecycle(self):
        required = {
            "fields": {"data": "String"},
            "init_value": '() -> Value("")',
            "sample_value": '(rng, options) -> Value("")',
            "mutate_value": "(rng, value, temperature, options) -> value",
            "count_scalar_constants": 1,
            "is_valid": "value -> true",
            "can_optimize": False,
        }
        for field in required:
            with self.subTest(field=field):
                parameters = required.copy()
                del parameters[field]
                with self.assertRaises(TypeError):
                    TypeSpec(**parameters)

    def test_rejects_invalid_fields_and_optimization_combinations(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            string_spec(fields={})
        with self.assertRaisesRegex(ValueError, "not an identifier"):
            string_spec(fields={"not valid": "String"})
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            string_spec(count_scalar_constants=-1)
        with self.assertRaisesRegex(ValueError, "requires"):
            string_spec(can_optimize=True)
        with self.assertRaisesRegex(ValueError, "require `can_optimize=True`"):
            string_spec(pack_scalar_constants="(buffer, idx, value) -> idx")
        for overrides, message in (
            ({"fields": {"data": ""}}, "requires a Julia type"),
            ({"init_value": ""}, "must contain Julia source"),
            ({"count_scalar_constants": None}, "nonnegative integer"),
            ({"can_optimize": 1}, "explicitly set"),
            ({"preamble": ""}, "cannot be empty"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    string_spec(**overrides)

    def test_rejects_invalid_operator_and_loss_declarations(self):
        def build(operators, **losses):
            return build_type_spec_module_source(
                string_spec(),
                operators,
                elementwise_loss=losses.get("elementwise_loss"),
                loss_function=losses.get("loss_function"),
                loss_function_expression=losses.get("loss_function_expression"),
            )

        for operators, message in (
            (None, "explicit .*operators"),
            ({0: ["identity_value(x::Value) = x"]}, "positive integers"),
            ({1: []}, "cannot be empty"),
            ({1: [""]}, "must contain Julia source"),
        ):
            with self.subTest(operators=operators):
                with self.assertRaisesRegex(ValueError, message):
                    build(operators, elementwise_loss="value_loss(x, y) = 0.0")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build({1: ["identity_value(x::Value) = x"]})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build(
                {1: ["identity_value(x::Value) = x"]},
                elementwise_loss="value_loss(x, y) = 0.0",
                loss_function="full_loss(tree, dataset, options) = 0.0",
            )

    def test_python_object_array_helpers(self):
        one_dimensional = object_array_1d(value for value in ([1, 2], [3, 4]))
        self.assertEqual(one_dimensional.dtype, object)
        self.assertEqual(one_dimensional.tolist(), [[1, 2], [3, 4]])
        np.testing.assert_array_equal(
            object_array_1d(np.array([1, 2])), np.array([1, 2], dtype=object)
        )

        two_dimensional = object_array_2d(
            row for row in (([1, 2], [3, 4]), ([5, 6], [7, 8]))
        )
        self.assertEqual(two_dimensional.shape, (2, 2))
        self.assertEqual(two_dimensional[0, 0], [1, 2])
        with self.assertRaisesRegex(ValueError, "2D array"):
            object_array_2d(["a", "b"])
        with self.assertRaisesRegex(ValueError, "same number"):
            object_array_2d([[1], [2, 3]])

    def test_optimization_contract_uses_packed_round_trip(self):
        spec = vector_spec()
        load_type_spec_runtime(spec, module_source(spec))

        broken = vector_spec(
            unpack_scalar_constants=(
                "(buffer, idx, value) -> "
                "(idx + 2, Value(copy(buffer[idx:idx+1]) .+ 1.0))"
            )
        )
        with self.assertRaisesRegex(ValueError, "packed scalar representation"):
            load_type_spec_runtime(broken, module_source(broken))

    def test_optimization_contract_rejects_each_invalid_hook(self):
        broken_specs = (
            (
                vector_spec(number_type="Int"),
                "concrete `AbstractFloat`",
            ),
            (
                vector_spec(pack_scalar_constants="(buffer, idx, value) -> idx + 1"),
                "pack_scalar_constants.*wrong next index",
            ),
            (
                vector_spec(unpack_scalar_constants="(buffer, idx, value) -> value"),
                "must return `\\(next_idx, Value\\)`",
            ),
            (
                vector_spec(
                    unpack_scalar_constants=("(buffer, idx, value) -> (idx + 1, value)")
                ),
                "unpack_scalar_constants.*wrong next index",
            ),
            (
                vector_spec(
                    unpack_scalar_constants=(
                        "(buffer, idx, value) -> (idx + 2, value.data)"
                    )
                ),
                "unpack_scalar_constants.*return `Value`",
            ),
        )
        for spec, message in broken_specs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    load_type_spec_runtime(spec, module_source(spec))

    def test_hook_and_loss_validation(self):
        with self.assertRaisesRegex(ValueError, "sample_value"):
            spec = string_spec(sample_value='() -> Value("a")')
            load_type_spec_runtime(spec, module_source(spec))
        with self.assertRaisesRegex(ValueError, "is_valid.*Bool"):
            spec = string_spec(is_valid="value -> 1")
            load_type_spec_runtime(spec, module_source(spec))
        with self.assertRaisesRegex(ValueError, "AbstractFloat"):
            source = module_source(
                string_spec(),
                elementwise_loss=(
                    "value_loss(x::Value, y::Value) = rand(Bool) ? 1.0 : 1"
                ),
            )
            load_type_spec_runtime(string_spec(), source)
        with self.assertRaisesRegex(ValueError, "explicit `loss_type`"):
            module_source(
                string_spec(),
                elementwise_loss=None,
                loss_function="full_loss(tree, dataset, options) = 0.0",
            )
        with self.assertRaisesRegex(ValueError, "return type is inferred"):
            module_source(string_spec(loss_type="Float64"))
        with self.assertRaisesRegex(ValueError, "named Julia functions"):
            source = module_source(string_spec(), {1: ["x -> x"]})
            load_type_spec_runtime(string_spec(), source)
        with self.assertRaisesRegex(ValueError, "Julia function"):
            source = module_source(string_spec(), {1: ["1"]})
            load_type_spec_runtime(string_spec(), source)
        with self.assertRaisesRegex(ValueError, "init_value.*return `Value`"):
            spec = string_spec(init_value='() -> ""')
            load_type_spec_runtime(spec, module_source(spec))
        with self.assertRaisesRegex(ValueError, "returned an invalid value"):
            spec = string_spec(is_valid="value -> false")
            load_type_spec_runtime(spec, module_source(spec))
        with self.assertRaisesRegex(ValueError, "nonnegative `Int`"):
            spec = string_spec(count_scalar_constants="value -> -1")
            load_type_spec_runtime(spec, module_source(spec))
        with self.assertRaisesRegex(ValueError, "operator.*return `Value`"):
            source = module_source(
                string_spec(), {1: ['bad_operator(x::Value) = "bad"']}
            )
            load_type_spec_runtime(string_spec(), source)

    def test_preamble_callable_count_and_full_objectives(self):
        spec = string_spec(
            preamble="const TYPE_SPEC_TEST_VALUE = 1",
            count_scalar_constants="value -> TYPE_SPEC_TEST_VALUE",
        )
        runtime = load_type_spec_runtime(spec, module_source(spec))
        self.assertEqual(runtime.module.TYPE_SPEC_TEST_VALUE, 1)

        full_spec = string_spec(loss_type="Float64")
        for mode, source in (
            (
                "loss_function",
                "full_loss(tree, dataset, options)::Float64 = 0.0",
            ),
            (
                "loss_function_expression",
                "full_expression_loss(expression, dataset, options)::Float64 = 0.0",
            ),
        ):
            with self.subTest(mode=mode):
                losses = {
                    "elementwise_loss": None,
                    "loss_function": None,
                    "loss_function_expression": None,
                }
                losses[mode] = source
                generated = module_source(full_spec, **losses)
                runtime = load_type_spec_runtime(full_spec, generated, validate=False)
                self.assertIsNotNone(getattr(runtime, mode))

    def test_conversion_and_unwrapping(self):
        spec = TypeSpec(
            fields={"number": "Float64", "label": "String"},
            init_value='() -> Value(0.0, "")',
            sample_value='(rng, options) -> Value(0.0, "")',
            mutate_value="(rng, value, temperature, options) -> value",
            count_scalar_constants=1,
            is_valid="value -> true",
            can_optimize=False,
        )
        source = module_source(
            spec,
            {1: ["identity_value(x::Value) = x"]},
            elementwise_loss=(
                "value_loss(x::Value, y::Value)::Float64 = " "x == y ? 0.0 : 1.0"
            ),
        )
        runtime = load_type_spec_runtime(spec, source)
        values = np.empty(2, dtype=object)
        values[:] = [(1.0, "one"), (2.0, "two")]
        converted = type_spec_to_julia_array(runtime, values)
        self.assertEqual(
            type_spec_to_python_array(runtime, converted).tolist(), values.tolist()
        )
        for value in ((1.0,), (1.0, "one", "extra")):
            wrong_size = np.empty(1, dtype=object)
            wrong_size[0] = value
            with self.assertRaisesRegex(ValueError, "exactly 2 fields"):
                type_spec_to_julia_array(runtime, wrong_size)
        with self.assertRaisesRegex(ValueError, "exactly 2 fields"):
            type_spec_to_julia_array(runtime, ["one"])
        with self.assertRaisesRegex(ValueError, "exactly 2 fields"):
            type_spec_to_julia_array(runtime, [1.0])
        with self.assertRaisesRegex(ValueError, "1D or 2D"):
            type_spec_to_julia_array(runtime, np.empty((1, 1, 1), dtype=object))
        non_object = type_spec_to_julia_array(
            load_type_spec_runtime(string_spec(), module_source(string_spec())),
            np.array(["a", "b"]),
        )
        self.assertEqual(len(non_object), 2)


class TestTypeSpecs(_TypeSpecContractTests, unittest.TestCase):
    def test_private_modules_isolate_identical_local_names(self):
        spec = string_spec(
            sample_value='(rng, options) -> Value("Ab")',
            mutate_value="(rng, value, temperature, options) -> value",
        )
        lower_source = module_source(
            spec,
            {1: ["same_operator(x::Value) = Value(lowercase(x.data))"]},
        )
        upper_source = module_source(
            spec,
            {1: ["same_operator(x::Value) = Value(uppercase(x.data))"]},
        )
        lower = load_type_spec_runtime(spec, lower_source)
        upper = load_type_spec_runtime(spec, upper_source)
        self.assertNotEqual(lower_source.module_name, upper_source.module_name)
        value = lower.module._convert_value("Ab")
        self.assertEqual(lower.operators[1][0](value).data, "ab")
        value = upper.module._convert_value("Ab")
        self.assertEqual(upper.operators[1][0](value).data, "AB")
        value = lower.module._convert_value("Ab")
        self.assertEqual(lower.operators[1][0](value).data, "ab")

    def test_serial_fit_predicts_logical_payloads(self):
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.array(["a", "b", "a", "b"], dtype=object)
        model = tiny_model(string_spec())
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        self.assertTrue(model._type_spec_module_name_.startswith("_PySRTypeSpec_"))
        self.assertTrue(
            bool(
                jl.seval("(a, b) -> a === b")(
                    model.julia_type_spec_module_, model.julia_type_spec_module_
                )
            )
        )
        model.set_params(warm_start=True)
        model.fit(X, y)
        model.set_params(warm_start=False)
        model.fit(X, y)
        exports = ExpressionSpec().create_exports(model, model.equations_, None)
        self.assertIn("lambda_format", exports)

    def test_multi_field_fit_predicts_tuples(self):
        spec = TypeSpec(
            fields={"number": "Float64", "label": "String"},
            init_value='() -> Value(0.0, "")',
            sample_value='(rng, options) -> Value(0.0, "")',
            mutate_value="(rng, value, temperature, options) -> value",
            count_scalar_constants=1,
            is_valid="value -> true",
            can_optimize=False,
        )
        pairs = [(1.0, "one"), (2.0, "two"), (3.0, "three"), (4.0, "four")]
        X = pd.DataFrame({"x": pairs})
        y = pd.Series(pairs, dtype=object)
        model = tiny_model(spec)
        model.fit(X, y)
        self.assertEqual(model.predict(X).tolist(), pairs)

    def test_multithreading(self):
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.array(["a", "b", "a", "b"], dtype=object)
        model = tiny_model(string_spec(), parallelism="multithreading")
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_multiprocessing(self):
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.array(["a", "b", "a", "b"], dtype=object)
        model = tiny_model(string_spec(), parallelism="multiprocessing", procs=2)
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_fresh_process_checkpoint(self):
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.array(["a", "b", "a", "b"], dtype=object)
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model(
                string_spec(),
                temp_equation_file=False,
                output_directory=directory,
                run_id="typespec-checkpoint",
                delete_tempfiles=False,
            )
            model.fit(X, y)
            run_directory = Path(directory) / "typespec-checkpoint"
            code = f"""
import json
import numpy as np
from pysr import PySRRegressor, jl
name = {model._type_spec_module_name_!r}
assert not bool(jl.isdefined(jl.Main, jl.Symbol(name)))
model = PySRRegressor.from_file(run_directory={str(run_directory)!r})
X = np.array([[\"a\"], [\"b\"], [\"a\"], [\"b\"]], dtype=object)
print(json.dumps(model.predict(X).tolist()))
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(result.stdout.strip().splitlines()[-1]), y.tolist()
            )

    def test_warm_start_rejects_runtime_changes(self):
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.array(["a", "b", "a", "b"], dtype=object)
        model = tiny_model(string_spec())
        model.fit(X, y)
        model.set_params(
            warm_start=True,
            operators={1: ["changed_operator(x::Value) = x"]},
        )
        with self.assertRaisesRegex(ValueError, "Cannot warm-start"):
            model.fit(X, y)
        model.set_params(
            type_spec=None,
            operators={1: ["identity_value(x::Value) = x"]},
        )
        with self.assertRaisesRegex(ValueError, "enabling or disabling TypeSpec"):
            model._validate_and_modify_params()

    def test_rejects_unsupported_options(self):
        for parameter, value in (
            ("guesses", ["x0"]),
            ("turbo", True),
            ("bumper", True),
            ("autodiff_backend", "Zygote"),
            ("output_jax_format", True),
        ):
            with self.subTest(parameter=parameter):
                with self.assertRaisesRegex(ValueError, parameter):
                    tiny_model(
                        string_spec(), **{parameter: value}
                    )._validate_and_modify_params()
        with self.assertRaisesRegex(ValueError, "binary_operators"):
            tiny_model(
                string_spec(), binary_operators=["+"]
            )._validate_and_modify_params()
        model = tiny_model(string_spec())
        model.expression_spec = object()
        with self.assertRaisesRegex(ValueError, "default `ExpressionSpec`"):
            model._validate_and_modify_params()
        with self.assertWarnsRegex(UserWarning, "large maxsize"):
            tiny_model(string_spec(), maxsize=41)._validate_and_modify_params()

    def test_rejects_empty_feature_axis(self):
        model = tiny_model(string_spec())
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            model.fit(np.empty((2, 0), dtype=object), np.array(["a", "b"]))

    def test_expression_export_requires_checkpoint_state(self):
        model = tiny_model(string_spec())
        with self.assertRaisesRegex(ValueError, "serialized Julia state"):
            ExpressionSpec().create_exports(
                model,
                pd.DataFrame({"equation": ["x0"]}),
                search_output=None,
            )

    def test_csv_only_loading_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hall_of_fame.csv").write_text(
                "Complexity,Loss,Equation\n1,0.0,x0\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checkpoint.pkl"):
                PySRRegressor.from_file(
                    run_directory=directory,
                    type_spec=string_spec(),
                    operators={1: ["identity_value(x::Value) = x"]},
                    n_features_in=1,
                )

    def test_numeric_path(self):
        X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
        y = X[:, 0]
        model = PySRRegressor(
            unary_operators=[],
            binary_operators=["+"],
            niterations=1,
            ncycles_per_iteration=2,
            populations=1,
            population_size=8,
            tournament_selection_n=3,
            maxsize=7,
            parallelism="serial",
            deterministic=True,
            random_state=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            should_optimize_constants=False,
        )
        model.fit(X, y)
        prediction = model.predict(X)
        self.assertEqual(prediction.shape, y.shape)
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
