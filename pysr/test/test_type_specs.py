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
        with self.assertRaisesRegex(ValueError, "2D array"):
            self._validate_fit_params(model, "invalid", [1.0])
        with self.assertRaisesRegex(ValueError, "same number of features"):
            self._validate_fit_params(model, [["a"], ["b", "c"]], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "2D array"):
            self._validate_fit_params(
                model, np.array(["a", "b"], dtype=object), [1.0, 2.0]
            )
        with self.assertRaisesRegex(ValueError, "inconsistent numbers of samples"):
            self._validate_fit_params(model, [["a"], ["b"]], [1.0])
        with self.assertRaisesRegex(NotImplementedError, "weights"):
            self._validate_fit_params(
                model, [["a"], ["b"]], [1.0, 2.0], weights=np.ones(2)
            )
        with self.assertRaisesRegex(NotImplementedError, "resampling"):
            self._validate_fit_params(
                model,
                [["a"], ["b"]],
                [1.0, 2.0],
                Xresampled=np.array([["a"], ["b"]], dtype=object),
            )

        with self.assertWarnsRegex(UserWarning, "variable_names"):
            *_, feature_names, _, _, _ = self._validate_fit_params(
                model,
                pd.DataFrame({"dataframe_name": ["a", "b"]}),
                [1.0, 2.0],
                variable_names=["ignored_name"],
            )
        self.assertEqual(feature_names.tolist(), ["dataframe_name"])

        with self.assertWarnsRegex(UserWarning, "Spaces in variable names"):
            *_, feature_names, _, _, _ = self._validate_fit_params(
                model,
                [["a"], ["b"]],
                [1.0, 2.0],
                variable_names=["spaced name"],
            )
        self.assertEqual(feature_names.tolist(), ["spaced_name"])

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
        with self.assertRaisesRegex(ValueError, "concrete subtype of `Real`"):
            TypeSpec("String", loss_type="1").julia_loss_type()
        for loss_type in ("String", "Real"):
            with self.subTest(loss_type=loss_type):
                with self.assertRaisesRegex(
                    ValueError, f"`loss_type` \\(`{loss_type}`\\)"
                ):
                    TypeSpec("String", loss_type=loss_type).julia_loss_type()

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

    def test_type_spec_predict_validation(self):
        model = PySRRegressor(type_spec=TypeSpec("String", loss_type="Float64"))
        model.selection_mask_ = None
        model.feature_names_in_ = np.array(["x0"])
        model.n_features_in_ = 1
        model.nout_ = 1
        model.equations_ = pd.DataFrame(
            {"lambda_format": [lambda X: X[:, 0]], "loss": [0.0], "score": [0.0]}
        )

        with self.assertRaisesRegex(ValueError, "2D array"):
            model.predict(np.array(["a"], dtype=object), index=0)
        with self.assertRaisesRegex(ValueError, "different number of features"):
            model.predict(np.array([["a", "b"]], dtype=object), index=0)
        with self.assertRaisesRegex(ValueError, "missing features"):
            model.predict(pd.DataFrame({"x1": ["a"]}), index=0)

    def test_type_spec_instantiates_compact_global_interface(self):
        name = f"PySRTestValue_{uuid.uuid4().hex}"
        spec = TypeSpec(
            name,
            fields={"data": "Float64"},
            init_value=f"() -> {name}(0.0)",
            sample_value=f"rng -> {name}(1.0)",
            mutate_value=f"(rng, value, temperature) -> {name}(value.data + temperature)",
            count_scalar_constants=1,
            pack_scalar_constants=(
                "(nvals, idx, value) -> (nvals[idx] = value.data; idx + 1)"
            ),
            unpack_scalar_constants=(
                f"(nvals, idx, value) -> (idx + 1, {name}(nvals[idx]))"
            ),
            get_number_type="T -> Float64",
            is_valid="value -> isfinite(value.data)",
            can_optimize=True,
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
        dynamic_expressions = jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE
        packed = jl.seval("zeros(1)")
        value = jl.seval(f"{name}(2.0)")
        self.assertEqual(
            dynamic_expressions.pack_scalar_constants_b(packed, 1, value), 2
        )
        self.assertEqual(packed[0], 2.0)
        next_idx, unpacked = dynamic_expressions.unpack_scalar_constants(
            packed, 1, value
        )
        self.assertEqual(next_idx, 2)
        self.assertEqual(unpacked.data, 2.0)
        self.assertEqual(dynamic_expressions.get_number_type(value_type), jl.Float64)
        self.assertTrue(dynamic_expressions.is_valid(value))
        self.assertFalse(dynamic_expressions.is_valid(jl.seval(f"{name}(NaN)")))
        self.assertTrue(
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
        for field in (
            "pack_scalar_constants",
            "unpack_scalar_constants",
            "get_number_type",
            "is_valid",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"{field} must accept"):
                    TypeSpec("String", **{field: "() -> nothing"}).instantiate()

    def test_type_spec_rejects_broken_optimization_round_trip(self):
        cases = (
            (
                "pack_scalar_constants",
                "(nvals, idx, value) -> (nvals[idx] = value.data; idx)",
                "(nvals, idx, value) -> (idx + 1, value)",
                "first unused index after packing",
            ),
            (
                "unpack_scalar_constants index",
                "(nvals, idx, value) -> (nvals[idx] = value.data; idx + 1)",
                "(nvals, idx, value) -> (idx, value)",
                "first unused index after unpacking",
            ),
            (
                "unpack_scalar_constants value",
                "(nvals, idx, value) -> (nvals[idx] = value.data; idx + 1)",
                None,
                "round-trip `init_value`",
            ),
        )
        for label, pack, unpack, message in cases:
            with self.subTest(label=label):
                name = f"BrokenRoundTrip_{uuid.uuid4().hex}"
                unpack = unpack or (
                    f"(nvals, idx, value) -> (idx + 1, {name}(value.data + 1))"
                )
                with self.assertRaisesRegex(ValueError, message):
                    TypeSpec(
                        name,
                        fields={"data": "Float64"},
                        init_value=f"() -> {name}(0.0)",
                        count_scalar_constants=1,
                        pack_scalar_constants=pack,
                        unpack_scalar_constants=unpack,
                        can_optimize=True,
                    ).instantiate()

    def test_type_spec_rejects_incompatible_or_invalid_definitions(self):
        with self.assertRaisesRegex(ValueError, "simple type name"):
            TypeSpec("Base.Invalid", fields={"data": "Float64"}).instantiate()
        with self.assertRaisesRegex(ValueError, "does not evaluate to a Julia type"):
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

    def test_type_spec_optimizes_two_layer_neural_network_constants(self):
        suffix = uuid.uuid4().hex
        name = f"NNValue_{suffix}"
        matmul = f"nn_matmul_{suffix}"
        add = f"nn_add_{suffix}"
        relu = f"nn_relu_{suffix}"
        loss = f"nn_mse_{suffix}"
        payload_matmul = f"nn_payload_matmul_{suffix}"
        payload_add = f"nn_payload_add_{suffix}"
        random_payload = f"nn_random_payload_{suffix}"
        payload_type = "Union{Float64, Vector{Float64}, Matrix{Float64}}"
        jl.seval(f"""
            {payload_matmul}(a::Matrix{{Float64}}, b::Vector{{Float64}}) =
                size(a, 2) == length(b) ? a * b : NaN
            {payload_matmul}(::{payload_type}, ::{payload_type}) = NaN
            {payload_add}(a::Float64, b::Float64) = a + b
            {payload_add}(a::T, b::T) where {{
                T<:Union{{Vector{{Float64}}, Matrix{{Float64}}}}
            }} = size(a) == size(b) ? a + b : NaN
            {payload_add}(::{payload_type}, ::{payload_type}) = NaN
            function {random_payload}(rng)
                rank = rand(rng, 0:2)
                rank == 0 ? randn(rng) : rank == 1 ? randn(rng, 2) : randn(rng, 2, 2)
            end
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
            pack_scalar_constants="""
                (nvals, idx, value) -> begin
                    n = length(value.data)
                    nvals[idx:idx+n-1] .= value.data isa Float64 ?
                        value.data : vec(value.data)
                    idx + n
                end
            """,
            unpack_scalar_constants=f"""
                (nvals, idx, value) -> begin
                    n = length(value.data)
                    data = value.data isa Float64 ? nvals[idx] :
                        reshape(copy(nvals[idx:idx+n-1]), size(value.data))
                    (idx + n, {name}(data))
                end
            """,
            get_number_type="T -> Float64",
            is_valid="value -> all(isfinite, value.data)",
            can_optimize=True,
            loss_type="Float64",
        )

        rng = np.random.default_rng(0)
        x_values = rng.normal(size=(64, 2))
        W1 = np.array([[1.2, -0.7], [0.5, 1.1]])
        b1 = np.array([0.3, -0.2])
        W2 = np.array([[0.8, -1.0], [1.3, 0.4]])
        b2 = np.array([-0.4, 0.2])
        y_values = (W2 @ np.maximum(x_values @ W1.T + b1, 0).T).T + b2
        X = pd.DataFrame(
            {
                "x": pd.Series(list(x_values), dtype=object),
                "W1": pd.Series([W1] * len(x_values), dtype=object),
                "b1": pd.Series([b1] * len(x_values), dtype=object),
            }
        )
        y = pd.Series(list(y_values), dtype=object)

        guess_W2 = W2 + np.array([[-0.3, 0.25], [0.2, -0.25]])
        guess_b2 = b2 + np.array([0.2, -0.15])
        initial_prediction = (
            guess_W2 @ np.maximum(x_values @ W1.T + b1, 0).T
        ).T + guess_b2
        initial_loss = np.mean((initial_prediction - y_values) ** 2)

        guess_constants = np.empty(2, dtype=object)
        guess_constants[:] = [guess_W2, guess_b2]
        julia_constants = spec.to_julia_array(guess_constants)
        call = jl.Symbol("call")
        guess = jl.Expr(
            call,
            jl.Symbol(add),
            jl.Expr(
                call,
                jl.Symbol(matmul),
                julia_constants[0],
                jl.Expr(
                    call,
                    jl.Symbol(relu),
                    jl.Expr(
                        call,
                        jl.Symbol(add),
                        jl.Expr(
                            call,
                            jl.Symbol(matmul),
                            jl.Symbol("W1"),
                            jl.Symbol("x"),
                        ),
                        jl.Symbol("b1"),
                    ),
                ),
            ),
            julia_constants[1],
        )

        model = PySRRegressor(
            type_spec=spec,
            operators={
                1: [f"{relu}(a::{name}) = {name}(max.(a.data, 0.0))"],
                2: [
                    f"{matmul}(a::{name}, b::{name}) = "
                    f"{name}({payload_matmul}(a.data, b.data))",
                    f"{add}(a::{name}, b::{name}) = "
                    f"{name}({payload_add}(a.data, b.data))",
                ],
            },
            elementwise_loss=(
                f"{loss}(a::{name}, b::{name}) = "
                "a.data isa Vector && b.data isa Vector && "
                "size(a.data) == size(b.data) ? "
                "sum(abs2, a.data .- b.data) / length(a.data) : 1.0e6"
            ),
            niterations=0,
            parallelism="serial",
            deterministic=True,
            random_state=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            guesses=[guess],
            should_optimize_constants=True,
            optimizer_nrestarts=0,
        )

        model.fit(X, y)

        best = model.equations_["loss"].idxmin()
        prediction = np.stack(
            [np.asarray(value.data) for value in model.predict(X, index=best)]
        )
        prediction_loss = np.mean((prediction - y_values) ** 2)
        self.assertEqual(prediction.shape, y_values.shape)
        self.assertGreater(initial_loss, 0.01)
        self.assertLess(prediction_loss, initial_loss / 1000)
        self.assertLess(prediction_loss, 1e-8)
        self.assertAlmostEqual(prediction_loss, model.equations_.loc[best, "loss"])

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

        def fail_prediction(_):
            raise RuntimeError("TypeSpec prediction failure")

        loaded.equations_["lambda_format"] = [fail_prediction] * len(loaded.equations_)
        with self.assertRaisesRegex(RuntimeError, "TypeSpec prediction failure"):
            loaded.predict(X)

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
