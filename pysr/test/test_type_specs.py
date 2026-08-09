import pickle
import unittest
import uuid

import numpy as np
import pandas as pd

from pysr import PySRRegressor, TypeSpec, jl
from pysr.expression_specs import ExpressionSpec, TemplateExpressionSpec


class TestTypeSpecs(unittest.TestCase):
    @staticmethod
    def _validate_fit_params(model, X, y, **kwargs):
        params = dict(
            Xresampled=None,
            weights=None,
            variable_names=None,
            complexity_of_variables=None,
            X_units=None,
            y_units=None,
        )
        params.update(kwargs)
        return model._validate_and_set_fit_params(X, y, **params)

    def test_type_spec_fit_validation(self):
        spec = TypeSpec("String", loss_type="Float64")
        model = PySRRegressor(type_spec=spec)

        X_lists = [[(1.0, "one")], [(2.0, "two")]]
        X, y, *_ = self._validate_fit_params(model, X_lists, [(1, 2), (3, 4)])
        self.assertEqual(X.shape, (2, 1))
        self.assertEqual(X[0, 0], (1.0, "one"))
        self.assertEqual(y.shape, (2,))
        self.assertEqual(y[1], (3, 4))

        for column_y in (
            np.ones((2, 1)),
            np.ones((2, 1)).astype(object),
            pd.DataFrame({"y": [1.0, 1.0]}),
        ):
            _, y, *_ = self._validate_fit_params(model, X_lists, column_y)
            self.assertEqual(y.shape, (2,))
            self.assertEqual(y[0], 1.0)

        with self.assertRaisesRegex(ValueError, "at least one sample"):
            self._validate_fit_params(model, np.empty((0, 2), dtype=object), [])
        with self.assertRaisesRegex(NotImplementedError, "units"):
            self._validate_fit_params(model, X_lists, [1.0, 2.0], X_units=["m"])
        with self.assertRaisesRegex(NotImplementedError, "one output"):
            self._validate_fit_params(model, X_lists, np.ones((2, 2)))

        default_spec_model = PySRRegressor(
            type_spec=spec, expression_spec=ExpressionSpec()
        )
        self._validate_fit_params(default_spec_model, X_lists, [1.0, 2.0])
        template_model = PySRRegressor(
            type_spec=spec,
            expression_spec=TemplateExpressionSpec(
                "f(x)", expressions=["f"], variable_names=["x"]
            ),
        )
        with self.assertRaisesRegex(NotImplementedError, "expression shape"):
            self._validate_fit_params(template_model, X_lists, [1.0, 2.0])

    def test_type_spec_complexity_of_variables(self):
        spec = TypeSpec("String", loss_type="Float64")
        model = PySRRegressor(type_spec=spec, complexity_of_variables=[3])
        self._validate_fit_params(model, [["a"], ["b"]], ["a", "b"])
        self.assertEqual(model.complexity_of_variables_, [3])
        with self.assertRaisesRegex(ValueError, "at both `fit` and `__init__`"):
            self._validate_fit_params(
                model, [["a"], ["b"]], ["a", "b"], complexity_of_variables=[5]
            )

    def test_type_spec_loss_type_validation(self):
        with self.assertRaisesRegex(ValueError, "type_spec.loss_type"):
            PySRRegressor(
                type_spec=TypeSpec("String", loss_type=""),
                elementwise_loss="loss(x, y) = x == y ? 0.0 : 1.0",
            )._validate_and_modify_params()
        with self.assertRaisesRegex(ValueError, "must evaluate to a Julia type"):
            TypeSpec("String", loss_type="1").julia_loss_type()

    def test_type_spec_accepts_multi_method_callbacks(self):
        suffix = uuid.uuid4().hex
        jl.seval(f"""
            _pysr_multi_sample_{suffix}(rng) = "a"
            _pysr_multi_sample_{suffix}(rng, options) = "b"
            """)
        spec = TypeSpec("String", sample_value=f"_pysr_multi_sample_{suffix}")
        value_type = spec.instantiate()
        jl.seval("using Random")
        self.assertEqual(
            jl.SymbolicRegression.sample_value(
                jl.Random.Xoshiro(0), value_type, jl.nothing
            ),
            "b",
        )

    def test_type_spec_can_optimize_defaults(self):
        name = f"NoOptValue_{uuid.uuid4().hex}"
        value_type = TypeSpec(name, fields={"data": "Float64"}).instantiate()
        self.assertFalse(
            jl.SymbolicRegression.ConstantOptimizationModule.can_optimize(
                value_type, jl.nothing
            )
        )
        float_type = TypeSpec("Float64").instantiate()
        self.assertTrue(
            jl.SymbolicRegression.ConstantOptimizationModule.can_optimize(
                float_type, jl.nothing
            )
        )

    def test_type_spec_old_checkpoint_state(self):
        state = PySRRegressor().__dict__.copy()
        del state["type_spec"]
        model = PySRRegressor.__new__(PySRRegressor)
        model.__setstate__(state)
        self.assertIsNone(model.type_spec)

    def test_type_spec_score_not_implemented(self):
        model = PySRRegressor(type_spec=TypeSpec("String", loss_type="Float64"))
        with self.assertRaises(NotImplementedError):
            model.score([["a"]], ["a"])

    def test_type_spec_instantiates_compact_global_interface(self):
        name = f"PySRTestValue_{uuid.uuid4().hex}"
        spec = TypeSpec(
            name,
            fields={"data": "Float64"},
            init_value=f"() -> {name}(0.0)",
            sample_value=f"rng -> {name}(1.0)",
            mutate_value=f"(rng, value, temperature) -> {name}(value.data + temperature)",
            count_scalar_constants=1,
            can_optimize=False,
        )

        value_type = spec.instantiate()
        options = jl.nothing
        jl.seval("using Random")
        rng = jl.Random.Xoshiro(0)

        self.assertEqual(jl.SymbolicRegression.init_value(value_type).data, 0.0)
        self.assertEqual(
            jl.SymbolicRegression.sample_value(rng, value_type, options).data, 1.0
        )
        self.assertEqual(
            jl.SymbolicRegression.mutate_value(
                rng, jl.SymbolicRegression.init_value(value_type), 0.5, options
            ).data,
            0.5,
        )
        self.assertEqual(
            jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE.count_scalar_constants(
                jl.SymbolicRegression.init_value(value_type)
            ),
            1,
        )
        self.assertFalse(
            jl.SymbolicRegression.ConstantOptimizationModule.can_optimize(
                value_type, options
            )
        )

    def test_type_spec_rejects_wrong_callback_arity(self):
        name = f"InvalidTypeSpec_{uuid.uuid4().hex}"
        with self.assertRaisesRegex(ValueError, "sample_value must accept"):
            TypeSpec(
                name, fields={"data": "String"}, sample_value='() -> ""'
            ).instantiate()

    def test_type_spec_rejects_incompatible_or_invalid_definitions(self):
        with self.assertRaisesRegex(ValueError, "simple type name"):
            TypeSpec("Base.Invalid", fields={"data": "Float64"}).instantiate()
        with self.assertRaisesRegex(ValueError, "concrete Julia type"):
            TypeSpec("nothing").instantiate()

    def test_type_spec_requires_loss_and_loss_type(self):
        with self.assertRaisesRegex(ValueError, "type_spec.loss_type"):
            PySRRegressor(
                type_spec=TypeSpec("String"),
                elementwise_loss="loss(x, y) = x == y ? 0.0 : 1.0",
            )._validate_and_modify_params()
        with self.assertRaisesRegex(ValueError, "requires a loss"):
            PySRRegressor(
                type_spec=TypeSpec("String", loss_type="Float64")
            )._validate_and_modify_params()

    def test_type_spec_converts_values_and_callback_constants(self):
        float_spec = TypeSpec("Float64")
        values = float_spec.to_julia_array([1.0, 2.0])
        np.testing.assert_array_equal(np.asarray(list(values)), [1.0, 2.0])
        transposed = float_spec.to_julia_array([[1.0, 2.0]], transpose=True)
        self.assertEqual(tuple(transposed.shape), (2, 1))
        with self.assertRaisesRegex(ValueError, "1D or 2D"):
            TypeSpec("Float64").to_julia_array(np.zeros((1, 1, 1)))

        name = f"CountingTypeSpec_{uuid.uuid4().hex}"
        spec = TypeSpec(
            name,
            fields={"data": "Float64"},
            count_scalar_constants="value -> 2",
        )
        spec.instantiate()
        self.assertEqual(spec.instantiate(), jl.seval(name))
        self.assertEqual(spec.to_julia_array([1.0])[0].data, 1.0)
        value = jl.seval(f"{name}(1.0)")
        self.assertEqual(
            jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE.count_scalar_constants(
                value
            ),
            2,
        )
        name = f"TwoFieldTypeSpec_{uuid.uuid4().hex}"
        pairs = np.empty(2, dtype=object)
        pairs[:] = [(1.0, "one"), [2.0, "two"]]
        converted = TypeSpec(
            name,
            fields={"x": "Float64", "label": "String"},
        ).to_julia_array(pairs)
        self.assertEqual(
            [(value.x, value.label) for value in converted],
            [(1.0, "one"), (2.0, "two")],
        )

    @staticmethod
    def _tiny_model(type_spec, operator, loss, **kwargs):
        params = dict(
            type_spec=type_spec,
            operators={1: [operator]},
            elementwise_loss=loss,
            niterations=1,
            ncycles_per_iteration=5,
            populations=1,
            population_size=10,
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
        params.update(kwargs)
        return PySRRegressor(
            **params,
        )

    def test_string_type_spec_fit_and_predict(self):
        spec = TypeSpec(
            "String",
            init_value='() -> ""',
            sample_value='rng -> rand(rng, ("a", "b"))',
            mutate_value='(rng, value, temperature) -> rand(rng, ("a", "b"))',
            count_scalar_constants=1,
            can_optimize=False,
            loss_type="Float64",
        )
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.array(["a", "b", "a", "b"], dtype=object)
        model = self._tiny_model(
            spec,
            "identity_string(x::String) = x",
            "string_loss(x::String, y::String) = x == y ? 0.0 : 1.0",
        )

        model.fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)

    def test_type_spec_supports_full_loss_function(self):
        spec = TypeSpec(
            "String",
            init_value='() -> ""',
            sample_value='rng -> rand(rng, ("a", "b"))',
            mutate_value='(rng, value, temperature) -> rand(rng, ("a", "b"))',
            count_scalar_constants=1,
            can_optimize=False,
            loss_type="Float64",
        )
        X = np.array([["a"], ["b"]], dtype=object)
        y = np.array(["a", "b"], dtype=object)
        for loss_kwarg, loss in (
            ("loss_function", "full_string_loss(tree, dataset, options) = 0.0"),
            (
                "loss_function_expression",
                "full_string_expression_loss(expression, dataset, options) = 0.0",
            ),
        ):
            with self.subTest(loss_kwarg=loss_kwarg):
                model = PySRRegressor(
                    type_spec=spec,
                    operators={1: ["identity_string_full_loss(x::String) = x"]},
                    niterations=1,
                    ncycles_per_iteration=1,
                    populations=1,
                    population_size=5,
                    tournament_selection_n=3,
                    maxsize=7,
                    parallelism="serial",
                    deterministic=True,
                    random_state=0,
                    progress=False,
                    verbosity=0,
                    temp_equation_file=True,
                    should_optimize_constants=False,
                    **{loss_kwarg: loss},
                )

                model.fit(X, y)

    def test_struct_type_spec_fit_and_predict(self):
        name = f"RASPValue_{uuid.uuid4().hex}"
        spec = TypeSpec(
            name,
            fields={"data": "Union{Float64, Vector{Float64}}"},
            init_value=f"() -> {name}(0.0)",
            sample_value=f"rng -> {name}(randn(rng))",
            mutate_value=(
                f"(rng, value, temperature) -> {name}(value.data isa Vector "
                "? value.data : value.data + temperature * randn(rng))"
            ),
            count_scalar_constants=1,
            can_optimize=False,
            loss_type="Float64",
        )
        sequences = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        X = pd.DataFrame({"x": sequences})
        y = pd.Series(sequences, dtype=object)
        operator = f"identity_rasp(x::{name}) = x"
        loss = f"rasp_loss(x::{name}, y::{name}) = x.data == y.data ? 0.0 : 1.0"
        model = self._tiny_model(spec, operator, loss)

        model.fit(X, y)

        prediction = model.predict(X, index=model.equations_["loss"].idxmin())
        self.assertEqual([list(value.data) for value in prediction], y.tolist())

        model = self._tiny_model(
            spec,
            operator,
            loss,
            parallelism="multiprocessing",
            procs=2,
            deterministic=False,
            random_state=None,
            worker_imports=["Random"],
        )
        model.fit(X, y)

        prediction = model.predict(X, index=model.equations_["loss"].idxmin())
        self.assertEqual([list(value.data) for value in prediction], y.tolist())

    def test_tensor_type_spec_uses_multiple_dispatch(self):
        suffix = uuid.uuid4().hex
        name = f"TensorValue_{suffix}"
        payload_mul = f"payload_mul_{suffix}"
        payload_mse = f"payload_mse_{suffix}"
        operator = f"tensor_mul_{suffix}"
        loss = f"tensor_mse_{suffix}"
        random_payload = f"random_payload_{suffix}"
        payload_type = "Union{Float64, Vector{Float64}, Matrix{Float64}}"
        jl.seval(f"""
            using LinearAlgebra: dot

            {payload_mul}(a::Float64, b::Float64) = a * b
            {payload_mul}(a::Float64, b::Union{{Vector{{Float64}}, Matrix{{Float64}}}}) = a * b
            {payload_mul}(a::Union{{Vector{{Float64}}, Matrix{{Float64}}}}, b::Float64) = a * b
            {payload_mul}(a::Vector{{Float64}}, b::Vector{{Float64}}) = dot(a, b)
            {payload_mul}(a::Matrix{{Float64}}, b::Vector{{Float64}}) = a * b
            {payload_mul}(a::Matrix{{Float64}}, b::Matrix{{Float64}}) = a * b
            {payload_mul}(::{payload_type}, ::{payload_type}) = NaN

            {payload_mse}(a::Float64, b::Float64) = isfinite(a) ? abs2(a - b) : Inf
            {payload_mse}(a::T, b::T) where {{T<:Union{{Vector{{Float64}}, Matrix{{Float64}}}}}} =
                size(a) == size(b) && all(isfinite, a) ? sum(abs2, a .- b) / length(a) : Inf
            {payload_mse}(::{payload_type}, ::{payload_type}) = Inf

            {random_payload}(rng) = rand(rng, (randn(rng), randn(rng, 3), randn(rng, 3, 3)))
            """)
        spec = TypeSpec(
            name,
            fields={"data": payload_type},
            init_value=f"() -> {name}(0.0)",
            sample_value=f"rng -> {name}({random_payload}(rng))",
            mutate_value=(
                f"(rng, value, temperature) -> rand(rng) < 0.1 "
                f"? {name}({random_payload}(rng)) "
                f": {name}(value.data .+ temperature .* randn(rng, size(value.data)...))"
            ),
            count_scalar_constants="value -> length(value.data)",
            loss_type="Float64",
        )

        rng = np.random.default_rng(0)
        scalar_a, scalar_b = rng.normal(size=2)
        vector_a, vector_b = rng.normal(size=(2, 3))
        matrix_a, matrix_b = rng.normal(size=(2, 3, 3))
        cases = (
            (scalar_a, scalar_b, scalar_a * scalar_b),
            (scalar_a, vector_b, scalar_a * vector_b),
            (vector_a, scalar_b, vector_a * scalar_b),
            (vector_a, vector_b, vector_a @ vector_b),
            (matrix_a, vector_b, matrix_a @ vector_b),
            (matrix_a, matrix_b, matrix_a @ matrix_b),
        )
        left, right, expected = zip(*cases)

        X = pd.DataFrame({"left": left, "right": right})
        y = pd.Series(expected, dtype=object)
        converted = spec.to_julia_array(X.to_numpy(dtype=object), transpose=True)
        converted_y = spec.to_julia_array(y.to_numpy(dtype=object))
        self.assertEqual(np.shape(converted[0, 0].data), ())
        self.assertEqual(np.shape(converted[1, 1].data), (3,))
        self.assertEqual(np.shape(converted[0, 5].data), (3, 3))
        count = (
            jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE.count_scalar_constants
        )
        self.assertEqual(count(converted[0, 0]), 1)
        self.assertEqual(count(converted[1, 1]), 3)
        self.assertEqual(count(converted[0, 5]), 9)
        jl.seval(f"""
            {operator}(a::{name}, b::{name}) = {name}({payload_mul}(a.data, b.data))
            {loss}(a::{name}, b::{name}) = {payload_mse}(a.data, b.data)
            """)
        self.assertTrue(
            jl.seval(
                f"Core.Compiler.return_type({operator}, Tuple{{{name}, {name}}}) === {name}"
            )
        )
        julia_operator = jl.seval(operator)
        julia_loss = jl.seval(loss)
        for i, target in enumerate(expected):
            actual = julia_operator(converted[0, i], converted[1, i])
            np.testing.assert_allclose(actual.data, target, atol=1e-12, rtol=1e-12)
            self.assertLess(julia_loss(actual, converted_y[i]), 1e-28)

        vector = jl.seval("[1.0, 2.0, 3.0]")
        matrix = jl.seval("zeros(3, 3)")
        self.assertTrue(np.isnan(jl.seval(payload_mul)(vector, matrix)))
        self.assertTrue(np.isinf(jl.seval(payload_mse)(vector, matrix)))

        fit_values = (scalar_a, scalar_b, vector_a, vector_b, matrix_a, matrix_b)
        fit_expected = (
            scalar_a * scalar_a,
            scalar_b * scalar_b,
            vector_a @ vector_a,
            vector_b @ vector_b,
            matrix_a @ matrix_a,
            matrix_b @ matrix_b,
        )
        fit_X = pd.DataFrame({"value": fit_values})
        fit_y = pd.Series(fit_expected, dtype=object)

        model = PySRRegressor(
            type_spec=spec,
            operators={
                2: [
                    f"{operator}(a::{name}, b::{name}) = "
                    f"{name}({payload_mul}(a.data, b.data))"
                ]
            },
            elementwise_loss=(
                f"{loss}(a::{name}, b::{name}) = {payload_mse}(a.data, b.data)"
            ),
            nested_constraints={operator: {operator: 0}},
            niterations=2,
            ncycles_per_iteration=5,
            populations=1,
            population_size=10,
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

        model.fit(fit_X, fit_y)

        best = model.equations_["loss"].idxmin()
        self.assertEqual(
            model.equations_.loc[best, "equation"], f"{operator}(value, value)"
        )
        prediction = [value.data for value in model.predict(fit_X, index=best)]
        for actual, target in zip(prediction, fit_expected):
            np.testing.assert_allclose(actual, target, atol=1e-12, rtol=1e-12)

    def test_multi_field_struct_type_spec_fit_and_predict(self):
        name = f"PairValue_{uuid.uuid4().hex}"
        spec = TypeSpec(
            name,
            fields={"number": "Float64", "label": "String"},
            init_value=f'() -> {name}(0.0, "")',
            sample_value=f'rng -> {name}(randn(rng), "")',
            mutate_value=(
                f"(rng, value, temperature) -> {name}(value.number + "
                "temperature * randn(rng), value.label)"
            ),
            count_scalar_constants=1,
            can_optimize=False,
            loss_type="Float64",
        )
        pairs = [(1.0, "one"), (2.0, "two"), (3.0, "three"), (4.0, "four")]
        X = pd.DataFrame({"x": pairs})
        y = pd.Series(pairs, dtype=object)
        model = self._tiny_model(
            spec,
            f"identity_pair(x::{name}) = x",
            f"pair_loss(x::{name}, y::{name}) = x == y ? 0.0 : 1.0",
        )

        model.fit(X, y)

        prediction = model.predict(X, index=model.equations_["loss"].idxmin())
        self.assertEqual([(value.number, value.label) for value in prediction], pairs)

    def test_type_spec_fit_and_predict_with_plain_lists(self):
        name = f"ListPair_{uuid.uuid4().hex}"
        spec = TypeSpec(
            name,
            fields={"number": "Float64", "label": "String"},
            init_value=f'() -> {name}(0.0, "")',
            sample_value=f'rng -> {name}(randn(rng), "")',
            mutate_value=(
                f"(rng, value, temperature) -> {name}(value.number + "
                "temperature * randn(rng), value.label)"
            ),
            count_scalar_constants=1,
            can_optimize=False,
            loss_type="Float64",
        )
        pairs = [(1.0, "one"), (2.0, "two"), (3.0, "three"), (4.0, "four")]
        X = [[pair] for pair in pairs]
        model = self._tiny_model(
            spec,
            f"identity_list_pair(x::{name}) = x",
            f"list_pair_loss(x::{name}, y::{name}) = x == y ? 0.0 : 1.0",
        )

        model.fit(X, pairs)

        prediction = model.predict(X, index=model.equations_["loss"].idxmin())
        self.assertEqual(prediction.shape, (4,))
        self.assertEqual(prediction.dtype, object)
        self.assertEqual([(value.number, value.label) for value in prediction], pairs)

    def test_type_spec_dataframe_columns_and_pickle_round_trip(self):
        spec = TypeSpec(
            "String",
            init_value='() -> ""',
            sample_value='rng -> rand(rng, ("a", "b"))',
            mutate_value='(rng, value, temperature) -> rand(rng, ("a", "b"))',
            count_scalar_constants=1,
            can_optimize=False,
            loss_type="Float64",
        )
        X = pd.DataFrame({10: ["a", "b", "a", "b"]})
        y = np.array(["a", "b", "a", "b"], dtype=object)
        model = self._tiny_model(
            spec,
            "identity_string_pickled(x::String) = x",
            "string_loss_pickled(x::String, y::String) = x == y ? 0.0 : 1.0",
        )

        model.fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)

        loaded = pickle.loads(pickle.dumps(model))
        self.assertNotIn("lambda_format", loaded.equations_.columns)
        self.assertNotIn("julia_expression", loaded.equations_.columns)
        np.testing.assert_array_equal(loaded.predict(X), y)

        loaded.julia_state_stream_ = None
        loaded.equations_ = loaded.equations_.drop(
            columns=["julia_expression", "lambda_format"]
        )
        with self.assertRaisesRegex(ValueError, "checkpoint.pkl"):
            loaded.predict(X)

    def test_type_spec_supports_multithreading(self):
        spec = TypeSpec(
            "String",
            init_value='() -> ""',
            sample_value='rng -> rand(rng, ("a", "b"))',
            mutate_value='(rng, value, temperature) -> rand(rng, ("a", "b"))',
            count_scalar_constants=1,
            can_optimize=False,
            loss_type="Float64",
        )
        X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
        y = np.array(["a", "b", "a", "b"], dtype=object)
        model = self._tiny_model(
            spec,
            "identity_string_threaded(x::String) = x",
            "string_loss_threaded(x::String, y::String) = x == y ? 0.0 : 1.0",
            parallelism="multithreading",
            deterministic=False,
            random_state=None,
        )

        model.fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)
