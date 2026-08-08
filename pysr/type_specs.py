from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .julia_helpers import jl_array
from .julia_import import AnyValue, jl


@dataclass(frozen=True)
class TypeSpec:
    """Runtime definition of a value type used by SymbolicRegression.jl."""

    julia_type: str
    fields: dict[str, str] | None = None
    init_value: str | None = None
    sample_value: str | None = None
    mutate_value: str | None = None
    count_scalar_constants: int | str | None = None
    can_optimize: bool | None = None
    loss_type: str | None = None

    def instantiate(self) -> AnyValue:
        """Define the type and its global SymbolicRegression.jl interface methods."""
        jl.seval("using Random: AbstractRNG")
        if self.fields is not None:
            if not self.julia_type.isidentifier():
                raise ValueError("A TypeSpec with fields requires a simple type name.")
            fields = "\n".join(
                f"    {name}::{type_}" for name, type_ in self.fields.items()
            )
            jl.seval(f"struct {self.julia_type}\n{fields}\nend")

        value_type = jl.seval(self.julia_type)
        if not jl.seval("T -> T isa Type")(value_type):
            raise ValueError(f"`{self.julia_type}` is not a concrete Julia type.")

        if self.init_value is not None:
            self._instantiate("init", self.init_value)
        if self.sample_value is not None:
            self._instantiate("sample", self.sample_value)
        if self.mutate_value is not None:
            self._instantiate("mutate", self.mutate_value)
        if self.count_scalar_constants is not None:
            if isinstance(self.count_scalar_constants, int):
                jl.seval(
                    "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                    f"count_scalar_constants(value::{self.julia_type}) = {self.count_scalar_constants}"
                )
            else:
                self._instantiate("count", self.count_scalar_constants)
        if self.can_optimize is not None:
            jl.seval(
                "SymbolicRegression.ConstantOptimizationModule."
                f"can_optimize(::Type{{{self.julia_type}}}, _) = {str(self.can_optimize).lower()}"
            )
        return value_type

    def validate_loss(
        self,
        elementwise_loss: str | None,
        loss_function: str | None,
        loss_function_expression: str | None,
    ) -> None:
        if (
            elementwise_loss is None
            and loss_function is None
            and loss_function_expression is None
        ) or self.loss_type is None:
            raise ValueError(
                "TypeSpec requires a loss (`elementwise_loss`, `loss_function`, or "
                "`loss_function_expression`) and `type_spec.loss_type`."
            )

    @staticmethod
    def supports_export() -> bool:
        return False

    @staticmethod
    def uses_generic_operators(value_type: AnyValue | None) -> bool:
        return not bool(jl.seval("T -> T <: Number")(value_type))

    @staticmethod
    def numpy_dtype(
        values: Any, precision_mapper: Callable[[np.ndarray], type]
    ) -> None:
        return None

    @staticmethod
    def elementwise_loss_probe(value_type: AnyValue | None, np_dtype: type | None):
        return jl.SymbolicRegression.init_value(value_type)

    def to_julia_array(
        self,
        values: Any,
        *,
        transpose: bool = False,
        dtype: type | None = None,
    ) -> AnyValue:
        """Convert Python logical values into a concrete Julia array."""
        array = np.asarray(values, dtype=object)
        if transpose:
            array = array.T
        if array.ndim not in (1, 2):
            raise ValueError("TypeSpec data must be a 1D or 2D array.")

        value_type = self.instantiate()
        if self.fields is None:
            convert = jl.seval("(T, x) -> PythonCall.pyconvert(T, x)")
        elif len(self.fields) == 1:
            convert = jl.seval("(T, x) -> T(PythonCall.pyconvert(fieldtype(T, 1), x))")
        else:
            arguments = ", ".join(
                f"PythonCall.pyconvert(fieldtype(T, {i + 1}), x[{i}])"
                for i in range(len(self.fields))
            )
            convert = jl.seval(f"(T, x) -> (x = PythonCall.Py(x); T({arguments}))")

        converted = [convert(value_type, value) for value in array.ravel(order="F")]
        return jl.seval("(T, xs, dims) -> reshape(T[x for x in xs], Tuple(dims))")(
            value_type, converted, array.shape
        )

    def _instantiate(self, kind: str, source: str) -> None:
        function = jl.seval(source)
        arity = jl.seval("f -> only(methods(f)).nargs - 1")(function)
        expected = {"init": (0,), "sample": (1, 2), "mutate": (3, 4), "count": (1,)}[
            kind
        ]
        if arity not in expected:
            raise ValueError(
                f"{kind}_value must accept {expected}; got {arity} arguments."
            )

        arguments = {
            "sample": {1: "rng", 2: "rng, options"},
            "mutate": {
                3: "rng, value, temperature",
                4: "rng, value, temperature, options",
            },
        }
        if kind == "init":
            definition = f"SymbolicRegression.init_value(::Type{{{self.julia_type}}}) = ({source})()"
        elif kind == "sample":
            definition = (
                "SymbolicRegression.sample_value("
                f"rng::AbstractRNG, ::Type{{{self.julia_type}}}, options) = "
                f"({source})({arguments['sample'][arity]})"
            )
        elif kind == "mutate":
            definition = (
                "SymbolicRegression.mutate_value("
                f"rng::AbstractRNG, value::{self.julia_type}, temperature, options) = "
                f"({source})({arguments['mutate'][arity]})"
            )
        else:
            definition = (
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                f"count_scalar_constants(value::{self.julia_type}) = ({source})(value)"
            )
        jl.seval(definition)


class _DefaultTypeSpec:
    loss_type = None

    @staticmethod
    def instantiate() -> None:
        return None

    @staticmethod
    def validate_loss(
        elementwise_loss: str | None,
        loss_function: str | None,
        loss_function_expression: str | None,
    ) -> None:
        return None

    @staticmethod
    def supports_export() -> bool:
        return True

    @staticmethod
    def uses_generic_operators(value_type: AnyValue | None) -> bool:
        return False

    @staticmethod
    def numpy_dtype(
        values: Any, precision_mapper: Callable[[np.ndarray], type]
    ) -> type:
        return precision_mapper(np.array(values))

    @staticmethod
    def elementwise_loss_probe(value_type: AnyValue | None, np_dtype: type | None):
        assert np_dtype is not None
        return np_dtype(1.0)

    @staticmethod
    def to_julia_array(
        values: Any,
        *,
        transpose: bool = False,
        dtype: type | None = None,
    ) -> AnyValue:
        array = np.array(values, dtype=dtype)
        if transpose:
            array = array.T
        return jl_array(array)


_DEFAULT_TYPE_SPEC = _DefaultTypeSpec()
