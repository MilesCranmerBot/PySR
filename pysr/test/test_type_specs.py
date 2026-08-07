import uuid

import numpy as np
import pandas as pd
import pytest

from pysr import PySRRegressor, TypeSpec, jl


def test_type_spec_installs_compact_global_interface():
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

    value_type = spec.install()
    options = jl.nothing
    jl.seval("using Random")
    rng = jl.Random.Xoshiro(0)

    assert jl.SymbolicRegression.init_value(value_type).data == 0.0
    assert jl.SymbolicRegression.sample_value(rng, value_type, options).data == 1.0
    assert (
        jl.SymbolicRegression.mutate_value(
            rng, jl.SymbolicRegression.init_value(value_type), 0.5, options
        ).data
        == 0.5
    )
    assert (
        jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE.count_scalar_constants(
            jl.SymbolicRegression.init_value(value_type)
        )
        == 1
    )
    assert not jl.SymbolicRegression.ConstantOptimizationModule.can_optimize(
        value_type, options
    )


def test_type_spec_rejects_wrong_callback_arity():
    with pytest.raises(ValueError, match="sample_value must accept"):
        TypeSpec("String", sample_value='() -> ""').install()


def _tiny_model(type_spec, operator, loss):
    return PySRRegressor(
        type_spec=type_spec,
        operators={1: [operator]},
        elementwise_loss=loss,
        loss_type="Float64",
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


def test_string_type_spec_fit_and_predict():
    spec = TypeSpec(
        "String",
        init_value='() -> ""',
        sample_value='rng -> rand(rng, ("a", "b"))',
        mutate_value='(rng, value, temperature) -> rand(rng, ("a", "b"))',
        count_scalar_constants=1,
        can_optimize=False,
    )
    X = np.array([["a"], ["b"], ["a"], ["b"]], dtype=object)
    y = np.array(["a", "b", "a", "b"], dtype=object)
    model = _tiny_model(
        spec,
        "identity_string(x::String) = x",
        "string_loss(x::String, y::String) = x == y ? 0.0 : 1.0",
    )

    model.fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)


def test_struct_type_spec_fit_and_predict():
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
    )
    sequences = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
    X = pd.DataFrame({"x": sequences})
    y = pd.Series(sequences, dtype=object)
    model = _tiny_model(
        spec,
        f"identity_rasp(x::{name}) = x",
        f"rasp_loss(x::{name}, y::{name}) = x.data == y.data ? 0.0 : 1.0",
    )

    model.fit(X, y)

    prediction = model.predict(X, index=model.equations_["loss"].idxmin())
    assert [list(value.data) for value in prediction] == y.tolist()
