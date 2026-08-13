import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pysr import PySRRegressor, TemplateExpressionSpec, TypeSpec, jl
from pysr.expression_specs import ExpressionSpec
from pysr.type_specs import (
    compile_type_spec,
    load_type_spec_runtime,
    object_array_1d,
    object_array_2d,
    type_spec_to_julia_array,
    type_spec_to_python_array,
    wrap_type_spec_addprocs_function,
)


def string_spec(**overrides):
    name = overrides.pop("name", "StringValue")
    return TypeSpec(
        name,
        fields=overrides.pop("fields", {"data": "String"}),
        sample=overrides.pop("sample", 'rng -> StringValue(rand(rng, ("a", "b")))'),
        mutate=overrides.pop(
            "mutate",
            '(rng, value, temperature) -> StringValue(rand(rng, ("a", "b")))',
        ),
        **overrides,
    )


def vector_spec(**overrides):
    name = overrides.pop("name", "VectorValue")
    return TypeSpec(
        name,
        fields=overrides.pop("fields", {"data": "Vector{Float64}"}),
        sample=overrides.pop("sample", "rng -> VectorValue([3.0, 4.0])"),
        parameters=overrides.pop("parameters", "value -> value.data"),
        with_parameters=overrides.pop(
            "with_parameters", "(value, parameters) -> VectorValue(parameters)"
        ),
        **overrides,
    )


def low_level_spec(**overrides):
    return vector_spec(
        count_parameters=overrides.pop("count_parameters", 2),
        pack_parameters=overrides.pop(
            "pack_parameters",
            "(buffer, idx, value) -> (buffer[idx:idx+1] .= value.data; idx + 2)",
        ),
        unpack_parameters=overrides.pop(
            "unpack_parameters",
            "(buffer, idx, value) -> (idx + 2, VectorValue(copy(buffer[idx:idx+1])))",
        ),
        **overrides,
    )


def module_source(
    spec,
    operators=None,
    *,
    elementwise_loss=None,
    loss_function=None,
    loss_function_expression=None,
):
    type_name = spec.name
    if (
        elementwise_loss is None
        and loss_function is None
        and loss_function_expression is None
    ):
        elementwise_loss = (
            f"value_loss(x::{type_name}, y::{type_name})::Float64 = "
            "x == y ? 0.0 : 1.0"
        )
    return compile_type_spec(
        spec,
        operators or {1: [f"identity_value(x::{type_name}) = x"]},
        elementwise_loss=elementwise_loss,
        loss_function=loss_function,
        loss_function_expression=loss_function_expression,
    )


def tiny_model(spec, *, parallelism="serial", procs=None, **overrides):
    type_name = spec.name
    parameters = {
        "type_spec": spec,
        "operators": {1: [f"identity_value(x::{type_name}) = x"]},
        "elementwise_loss": (
            f"value_loss(x::{type_name}, y::{type_name})::Float64 = "
            "x == y ? 0.0 : 1.0"
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


def identity_template():
    return TemplateExpressionSpec(
        combine="f(x)",
        expressions=["f"],
        variable_names=["x"],
    )


def string_data(*, constant=False):
    X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
    y = np.full(4, "a", dtype=object) if constant else X[:, 0].copy()
    return X, y


class TestTypeSpecs(unittest.TestCase):
    def _assert_invalid_runtime(self, message, spec=None, operators=None, **losses):
        with self.assertRaisesRegex(ValueError, message):
            load_type_spec_runtime(
                module_source(spec or string_spec(), operators, **losses)
            )

    def test_requires_name_fields_sample_and_keyword_only_configuration(self):
        with self.assertRaises(TypeError):
            TypeSpec(fields={"data": "String"}, sample="rng -> nothing")
        with self.assertRaises(TypeError):
            TypeSpec("StringValue", {"data": "String"}, "rng -> nothing")
        with self.assertRaises(TypeError):
            TypeSpec("StringValue", fields={"data": "String"})

    def test_rejects_invalid_fields_and_optimization_combinations(self):
        low_level = {
            "pack_parameters": "(buffer, idx, value) -> idx",
            "unpack_parameters": "(buffer, idx, value) -> (idx, value)",
        }
        cases = (
            ({"name": "not valid"}, "not an identifier"),
            ({"fields": {}}, "non-empty"),
            ({"fields": {"not valid": "String"}}, "not an identifier"),
            ({"parameters": "value -> Float64[]"}, "provided together"),
            ({"mutate": None}, "requires an explicit `mutate`"),
            ({"count_parameters": 0, **low_level}, "require `parameters`"),
            ({"fields": {"data": ""}}, "requires a Julia type"),
            ({"sample": ""}, "must contain Julia source"),
            ({"init": ""}, "cannot be empty"),
            ({"preamble": ""}, "cannot be empty"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    string_spec(**overrides)
        for overrides, message in (
            ({"pack_parameters": low_level["pack_parameters"]}, "provided together"),
            ({"count_parameters": -1, **low_level}, "nonnegative"),
            ({"count_parameters": "", **low_level}, "cannot be empty"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    vector_spec(**overrides)
        if not os.environ.get("PYSR_USE_BEARTYPE"):
            with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                vector_spec(
                    count_parameters=object(),
                    pack_parameters="identity",
                    unpack_parameters="identity",
                )

    def test_rejects_invalid_operator_and_loss_declarations(self):
        def build(operators, **losses):
            return compile_type_spec(
                string_spec(),
                operators,
                elementwise_loss=losses.get("elementwise_loss"),
                loss_function=losses.get("loss_function"),
                loss_function_expression=losses.get("loss_function_expression"),
            )

        for operators, message in (
            (None, "explicit .*operators"),
            ({0: ["identity_value(x::StringValue) = x"]}, "positive integers"),
            ({1: []}, "cannot be empty"),
            ({1: [""]}, "must contain Julia source"),
        ):
            with self.subTest(operators=operators):
                with self.assertRaisesRegex(ValueError, message):
                    build(operators, elementwise_loss="value_loss(x, y) = 0.0")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build({1: ["identity_value(x::StringValue) = x"]})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build(
                {1: ["identity_value(x::StringValue) = x"]},
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
        runtime = load_type_spec_runtime(module_source(spec))
        self.assertEqual(str(jl.nameof(runtime.value_type)), "VectorValue")
        self.assertEqual(
            str(runtime.module._string(runtime.module._convert_value([1, 2]))),
            "[1.0, 2.0]",
        )

        immutable_parameters = vector_spec(
            parameters=(
                "value -> range(value.data[1], value.data[end]; "
                "length=length(value.data))"
            ),
            with_parameters=("(value, parameters) -> VectorValue(collect(parameters))"),
        )
        load_type_spec_runtime(module_source(immutable_parameters))

    def test_custom_init_mutation_validity_and_string(self):
        spec = vector_spec(
            init="() -> VectorValue([1.0, 2.0])",
            mutate="(rng, value, temperature) -> VectorValue(value.data .+ temperature)",
            is_valid="value -> all(>(0), value.data)",
            string='value -> "vec($(join(value.data, ", ")))"',
        )
        runtime = load_type_spec_runtime(module_source(spec))
        value = runtime.module._convert_value([1.0, 2.0])
        self.assertEqual(str(runtime.module._string(value)), "vec(1.0, 2.0)")
        self.assertEqual(str(jl.sprint(jl.show, value)), "vec(1.0, 2.0)")

    def test_union_payload_parameterization(self):
        spec = TypeSpec(
            "TensorValue",
            fields={"data": "Union{Float64, Vector{Float64}, Matrix{Float64}}"},
            sample="rng -> TensorValue(randn(rng, 2, 2))",
            parameters="""
            function parameters(value)
                return value.data isa Float64 ? [value.data] : vec(value.data)
            end
            """,
            with_parameters="""
            function with_parameters(value, parameters)
                data = value.data isa Float64 ? parameters[1] :
                    reshape(collect(parameters), size(value.data))
                return TensorValue(data)
            end
            """,
        )
        runtime = load_type_spec_runtime(module_source(spec))
        value = runtime.module._convert_value(np.eye(2))
        self.assertEqual(list(runtime.module._parameters(value)), [1.0, 0.0, 0.0, 1.0])

    def test_variable_length_vector_constants(self):
        spec = TypeSpec(
            "VariableVector",
            fields={"data": "Vector{Float64}"},
            sample="rng -> VariableVector(randn(rng, rand(rng, 1:5)))",
            mutate="""
            function mutate_vector(rng, value, temperature)
                if rand(rng) < 0.2
                    return VariableVector(randn(rng, rand(rng, 1:5)))
                end
                data = copy(value.data)
                data[rand(rng, eachindex(data))] += temperature * randn(rng)
                return VariableVector(data)
            end
            """,
            parameters="value -> value.data",
            with_parameters="(value, parameters) -> VariableVector(parameters)",
        )
        rng = np.random.default_rng(0)
        values = [rng.normal(size=2) for _ in range(64)]
        X = pd.DataFrame({"x": values})
        prefix = np.array([1.5])
        suffix = np.array([-0.2, -3.0, 0.1])
        y = np.empty(len(values), dtype=object)
        y[:] = [np.concatenate((prefix, value, suffix)) for value in values]

        model = tiny_model(
            spec,
            expression_spec=identity_template(),
            operators={
                2: ["concat_vectors(a, b) = VariableVector(vcat(a.data, b.data))"]
            },
            elementwise_loss="""
            function vector_loss(a, b)::Float64
                return length(a.data) == length(b.data) ?
                    sum(abs2, a.data - b.data) : 1.0e6
            end
            """,
            niterations=60,
            ncycles_per_iteration=100,
            populations=4,
            population_size=50,
            tournament_selection_n=10,
            maxsize=7,
            early_stop_condition="(loss, complexity) -> loss < 1.0e-8 && complexity == 5",
            should_optimize_constants=True,
        )
        model.fit(X, y)

        exact = model.equations_.query("complexity == 5").sort_values("loss").iloc[0]
        self.assertLess(exact.loss, 1.0e-8)
        constants = jl.seval("""
        function (expression)
            DE = SymbolicRegression.InterfaceDynamicExpressionsModule.DE
            tree = DE.get_tree(expression)
            nodes = DE.filter_map(
                node -> node.degree == 0 && node.constant,
                identity,
                tree,
                typeof(tree),
            )
            return map(node -> node.val.data, nodes)
        end
        """)(exact.julia_expression)
        self.assertEqual([len(value) for value in constants], [1, 3])
        np.testing.assert_allclose(constants[0], prefix, atol=1.0e-6)
        np.testing.assert_allclose(constants[1], suffix, atol=1.0e-6)

    def test_optimization_contract_rejects_each_invalid_hook(self):
        broken_specs = (
            (
                vector_spec(parameters="value -> 1.0"),
                "must return an `AbstractVector`",
            ),
            (
                vector_spec(
                    parameters="value -> [1, 2]",
                    with_parameters="(value, p) -> VectorValue(collect(p))",
                ),
                "concrete `AbstractFloat`",
            ),
            (
                vector_spec(with_parameters="(value, parameters) -> parameters"),
                "must return `VectorValue`",
            ),
            (
                low_level_spec(
                    pack_parameters="(buffer, idx, value) -> idx + 1",
                ),
                "pack_parameters.*wrong next index",
            ),
            (
                low_level_spec(
                    unpack_parameters="(buffer, idx, value) -> value",
                ),
                "must return `\\(next_idx, VectorValue\\)`",
            ),
            (
                low_level_spec(
                    unpack_parameters="(buffer, idx, value) -> (idx + 1, value)",
                ),
                "unpack_parameters.*wrong next index",
            ),
            (
                low_level_spec(
                    unpack_parameters="(buffer, idx, value) -> (idx + 2, value.data)",
                ),
                "unpack_parameters.*return `VectorValue`",
            ),
            (
                low_level_spec(
                    mutate="(rng, value, temperature) -> VectorValue([1.0])",
                    count_parameters="value -> length(value.data) == 1 ? -1 : 2",
                ),
                "count_parameters.*nonnegative",
            ),
            (
                low_level_spec(
                    unpack_parameters=(
                        "(buffer, idx, value) -> (idx + 2, "
                        "VectorValue(copy(buffer[idx:idx+1]) .+ 1.0))"
                    )
                ),
                "packed scalar representation",
            ),
            (
                low_level_spec(
                    count_parameters=1,
                    pack_parameters="(buffer, idx, value) -> (buffer[idx] = value.data[1]; idx + 1)",
                    unpack_parameters="(buffer, idx, value) -> (idx + 1, VectorValue([buffer[idx], value.data[2]]))",
                ),
                "count_parameters.*disagrees",
            ),
            (
                low_level_spec(
                    pack_parameters="(buffer, idx, value) -> (buffer[idx:idx+1] .= reverse(value.data); idx + 2)"
                ),
                "pack_parameters.*disagrees",
            ),
            (
                vector_spec(
                    with_parameters="(value, parameters) -> VectorValue(reverse(collect(parameters)))"
                ),
                "must preserve",
            ),
            (
                low_level_spec(
                    mutate="(rng, value, temperature) -> VectorValue([1.0, 2.0, 3.0])",
                    pack_parameters="(buffer, idx, value) -> (buffer[idx:idx+1] .= value.data[1:2]; idx + 2)",
                ),
                "count_parameters.*disagrees",
            ),
            (vector_spec(string="value -> 1"), "string.*AbstractString"),
            (string_spec(sample='() -> StringValue("a")'), "sample"),
            (string_spec(is_valid="value -> 1"), "is_valid.*Bool"),
            (string_spec(init='() -> ""'), "init.*return `StringValue`"),
            (string_spec(is_valid="value -> false"), "returned an invalid value"),
            (
                vector_spec(
                    count_parameters="value -> -1",
                    pack_parameters="(buffer, idx, value) -> idx",
                    unpack_parameters="(buffer, idx, value) -> (idx, value)",
                ),
                "nonnegative `Int`",
            ),
        )
        for spec, message in broken_specs:
            with self.subTest(message=message):
                self._assert_invalid_runtime(message, spec)

    def test_hook_and_loss_validation(self):
        self._assert_invalid_runtime(
            "AbstractFloat",
            elementwise_loss=(
                "value_loss(x::StringValue, y::StringValue) = rand(Bool) ? 1.0 : 1"
            ),
        )
        with self.assertRaisesRegex(ValueError, "explicit `loss_type`"):
            module_source(
                string_spec(),
                elementwise_loss=None,
                loss_function="full_loss(tree, dataset, options) = 0.0",
            )
        with self.assertRaisesRegex(ValueError, "return type is inferred"):
            module_source(string_spec(loss_type="Float64"))
        for operators, message in (
            ({1: ["x -> x"]}, "named Julia functions"),
            ({1: ["1"]}, "Julia function"),
            (
                {1: ['bad_operator(x::StringValue) = "bad"']},
                "operator.*infer `StringValue`",
            ),
            (
                {1: ['unstable_operator(x::StringValue) = rand(Bool) ? x : "bad"']},
                "operator.*infer `StringValue`",
            ),
        ):
            with self.subTest(message=message):
                self._assert_invalid_runtime(message, operators=operators)

    def test_preamble_low_level_overrides_and_full_objectives(self):
        spec = low_level_spec(
            fields={"data": "TypeSpecPayload"},
            preamble=(
                "const TypeSpecPayload = Vector{Float64}\n"
                "const TYPE_SPEC_TEST_VALUE = 1"
            ),
            count_parameters="value -> 2 * TYPE_SPEC_TEST_VALUE",
        )
        runtime = load_type_spec_runtime(module_source(spec))
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
                runtime = load_type_spec_runtime(generated)
                self.assertIsNotNone(getattr(runtime, mode))

    def test_default_addprocs_wrapper(self):
        source = module_source(string_spec())
        default_wrapper = wrap_type_spec_addprocs_function(source, None, None)
        self.assertTrue(bool(jl.seval("x -> x isa Function")(default_wrapper)))
        explicit_wrapper = wrap_type_spec_addprocs_function(
            source, jl.Distributed.addprocs, None
        )
        self.assertTrue(bool(jl.seval("x -> x isa Function")(explicit_wrapper)))

    def test_conversion_and_unwrapping(self):
        spec = TypeSpec(
            "PairValue",
            fields={"number": "Float64", "label": "String"},
            sample='rng -> PairValue(0.0, "")',
            mutate="(rng, value, temperature) -> value",
        )
        source = module_source(
            spec,
            {1: ["identity_value(x::PairValue) = x"]},
            elementwise_loss=(
                "value_loss(x::PairValue, y::PairValue)::Float64 = "
                "x == y ? 0.0 : 1.0"
            ),
        )
        runtime = load_type_spec_runtime(source)
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
            load_type_spec_runtime(module_source(string_spec())),
            np.array(["a", "b"]),
        )
        self.assertEqual(len(non_object), 2)

    def test_template_serial_fit_predicts_logical_payloads(self):
        X, y = string_data()
        model = tiny_model(string_spec(), expression_spec=identity_template())
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        self.assertTrue(
            bool(
                jl.seval("x -> x isa SymbolicRegression.TemplateExpression")(
                    model.equations_.iloc[0].julia_expression
                )
            )
        )

    def test_template_custom_combiner_infers_num_features(self):
        model = tiny_model(
            vector_spec(),
            expression_spec=TemplateExpressionSpec(
                combine="add_vectors(f(x1), g(x2))",
                expressions=["f", "g"],
                variable_names=["x1", "x2"],
            ),
            operators={
                2: [
                    """
                    add_vectors(a::VectorValue, b::VectorValue) =
                        VectorValue(a.data + b.data)
                    add_vectors(a::ValidVector, b::ValidVector) =
                        ValidVector(map(add_vectors, a.x, b.x), a.valid && b.valid)
                    """
                ]
            },
            elementwise_loss=(
                "vector_loss(a::VectorValue, b::VectorValue)::Float64 = "
                "sum(abs2, a.data - b.data)"
            ),
        )
        runtime = model._load_type_spec_runtime(model._operators_from_params())
        assert runtime.expression_spec is not None
        self.assertEqual(int(runtime.expression_spec.structure.num_features.f), 1)
        self.assertEqual(int(runtime.expression_spec.structure.num_features.g), 1)

    def test_template_type_spec_rejects_unsupported_shapes(self):
        X, y = string_data()
        cases = (
            (
                TemplateExpressionSpec(
                    combine="p[1] * f(x)",
                    expressions=["f"],
                    variable_names=["x"],
                    parameters={"p": 1},
                    num_features={"f": 1},
                ),
                "parameters",
            ),
            (
                TemplateExpressionSpec(["f"], "(fs, x) -> fs.f(x[1])", {"f": 1}),
                "legacy",
            ),
        )
        for expression_spec, message in cases:
            with self.subTest(message=message):
                model = tiny_model(string_spec(), expression_spec=expression_spec)
                with self.assertRaisesRegex(ValueError, message):
                    model.fit(X, y)

    def test_private_modules_isolate_identical_local_names(self):
        spec = string_spec(
            sample='rng -> StringValue("Ab")',
            mutate="(rng, value, temperature) -> value",
        )
        lower_source = module_source(
            spec,
            {1: ["same_operator(x::StringValue) = " "StringValue(lowercase(x.data))"]},
        )
        upper_source = module_source(
            spec,
            {1: ["same_operator(x::StringValue) = " "StringValue(uppercase(x.data))"]},
        )
        lower = load_type_spec_runtime(lower_source)
        upper = load_type_spec_runtime(upper_source)
        self.assertNotEqual(lower_source.module_name, upper_source.module_name)
        value = lower.module._convert_value("Ab")
        self.assertEqual(lower.operators[1][0](value).data, "ab")
        value = upper.module._convert_value("Ab")
        self.assertEqual(upper.operators[1][0](value).data, "AB")
        value = lower.module._convert_value("Ab")
        self.assertEqual(lower.operators[1][0](value).data, "ab")

    def test_invalid_runtime_is_not_cached(self):
        model = tiny_model(
            string_spec(),
            operators={1: ['bad_operator(x::StringValue) = "bad"']},
        )
        with self.assertRaisesRegex(ValueError, "operator"):
            model._load_type_spec_runtime(model._operators_from_params())
        self.assertFalse(model._has_fitted_type_spec())

    def test_serial_fit_predicts_logical_payloads(self):
        X, y = string_data()
        model = tiny_model(string_spec())
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        self.assertTrue(
            model._type_spec_definition_.module_name.startswith("_PySRTypeSpec_")
        )
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

        equations = model.equations_
        model.equations_ = equations.copy()

        def fail_prediction(_):
            raise RuntimeError("TypeSpec evaluation failed")

        model.equations_["lambda_format"] = [fail_prediction] * len(model.equations_)
        with self.assertRaisesRegex(RuntimeError, "TypeSpec evaluation failed"):
            model.predict(X)
        model.equations_ = equations

        model.set_params(type_spec=None)
        np.testing.assert_array_equal(model.predict(X), y)
        with self.assertRaisesRegex(NotImplementedError, "score"):
            model.score(X, y)
        pickled_equations = model.__getstate__()["equations_"]
        self.assertNotIn("lambda_format", pickled_equations.columns)
        exports = ExpressionSpec().create_exports(model, model.equations_, None)
        self.assertIn("lambda_format", exports)

    def test_numeric_state_access_does_not_install_future_type_spec(self):
        X = np.arange(8.0).reshape(-1, 1)
        model = PySRRegressor(
            niterations=0,
            populations=1,
            population_size=8,
            tournament_selection_n=3,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
        )
        model.fit(X, X[:, 0])
        model.set_params(
            type_spec=string_spec(),
            operators={1: ["identity_value(x::StringValue) = x"]},
            elementwise_loss=(
                "value_loss(x::StringValue, y::StringValue)::Float64 = "
                "x == y ? 0.0 : 1.0"
            ),
        )
        _ = model.julia_state_
        _ = model.julia_options_
        self.assertFalse(model._has_fitted_type_spec())

    def test_private_operator_options(self):
        operator = "private_identity_value(x::StringValue) = x"
        X, _ = string_data()
        model = tiny_model(
            string_spec(),
            operators={1: [operator]},
            complexity_of_operators={"private_identity_value": 2},
            nested_constraints={
                "private_identity_value": {"private_identity_value": 0}
            },
        )
        model.fit(X, X[:, 0])
        self.assertEqual(model.predict(X).shape, (4,))

        model = tiny_model(
            string_spec(),
            operators={1: [operator]},
            complexity_of_operators={"missing_operator": 2},
        )
        with self.assertRaisesRegex(ValueError, "Unknown TypeSpec operator"):
            model.fit(X, X[:, 0])

    def test_multi_field_fit_predicts_tuples(self):
        spec = TypeSpec(
            "PairValue",
            fields={"number": "Float64", "label": "String"},
            sample='rng -> PairValue(0.0, "")',
            mutate="(rng, value, temperature) -> value",
        )
        pairs = [(1.0, "one"), (2.0, "two"), (3.0, "three"), (4.0, "four")]
        X = pd.DataFrame({"x": pairs})
        y = pd.Series(pairs, dtype=object)
        model = tiny_model(spec)
        model.fit(X, y)
        self.assertEqual(model.predict(X).tolist(), pairs)

    def test_multithreading(self):
        X, y = string_data()
        model = tiny_model(string_spec(), parallelism="multithreading")
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_multiprocessing(self):
        X, y = string_data(constant=True)
        model = tiny_model(
            string_spec(
                sample='rng -> StringValue("a")',
                mutate="(rng, value, temperature) -> value",
            ),
            expression_spec=identity_template(),
            parallelism="multiprocessing",
            procs=2,
        )
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_fresh_process_checkpoint(self):
        X, y = string_data()
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model(
                string_spec(),
                expression_spec=identity_template(),
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
name = {model._type_spec_definition_.module_name!r}
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
        X, y = string_data()
        model = tiny_model(string_spec())
        model.fit(X, y)
        model.set_params(
            warm_start=True,
            expression_spec=TemplateExpressionSpec(
                combine="f(x)",
                expressions=["f"],
                variable_names=["x"],
                num_features={"f": 2},
            ),
        )
        with self.assertRaisesRegex(ValueError, "Cannot warm-start"):
            model.fit(X, y)
        model.set_params(
            type_spec=None,
            operators={1: ["identity_value(x::StringValue) = x"]},
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
        with self.assertRaisesRegex(
            ValueError, "ExpressionSpec.*TemplateExpressionSpec"
        ):
            model._validate_and_modify_params()
        with self.assertWarnsRegex(UserWarning, "large maxsize"):
            tiny_model(string_spec(), maxsize=41)._validate_and_modify_params()

    def test_rejects_empty_feature_axis(self):
        model = tiny_model(string_spec())
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            model.fit(np.empty((2, 0), dtype=object), np.array(["a", "b"]))

    def test_expression_export_requires_checkpoint_state(self):
        model = tiny_model(string_spec())
        model._load_type_spec_runtime(model._operators_from_params())
        for expression_spec in (ExpressionSpec(), identity_template()):
            with self.subTest(expression_spec=type(expression_spec).__name__):
                with self.assertRaisesRegex(ValueError, "serialized Julia state"):
                    expression_spec.create_exports(
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
                    operators={1: ["identity_value(x::StringValue) = x"]},
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
        model.set_params(
            type_spec=string_spec(),
            operators={1: ["identity_value(x::StringValue) = x"]},
        )
        np.testing.assert_array_equal(model.predict(X), prediction)
        self.assertTrue(model._supports_export("sympy"))


if __name__ == "__main__":
    unittest.main()
