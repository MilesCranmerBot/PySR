import uuid

import pytest

from pysr import TypeSpec, jl


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
