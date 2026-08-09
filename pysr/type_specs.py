from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .julia_helpers import jl_array
from .julia_import import AnyValue, jl


def object_array_1d(values: Any) -> np.ndarray:
    """Build a 1D object array whose cells are the logical values themselves.

    `np.asarray(values, dtype=object)` would expand nested tuples/lists into
    extra axes, so cells are assigned individually.
    """
    if isinstance(values, np.ndarray):
        return values if values.dtype == object else values.astype(object)
    values = list(values)
    array = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        array[i] = value
    return array


def object_array_2d(values: Any) -> np.ndarray:
    """Build a 2D object array whose cells are the logical values themselves."""
    if isinstance(values, np.ndarray):
        return values if values.dtype == object else values.astype(object)
    try:
        if any(isinstance(row, (str, bytes)) for row in values):
            raise TypeError
        rows = [list(row) for row in values]
    except TypeError:
        raise ValueError("TypeSpec X must be a 2D array of logical values.")
    n_columns = len(rows[0]) if rows else 0
    array = np.empty((len(rows), n_columns), dtype=object)
    for i, row in enumerate(rows):
        if len(row) != n_columns:
            raise ValueError("All rows of X must have the same number of features.")
        for j, value in enumerate(row):
            array[i, j] = value
    return array


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
            jl.seval(self._struct_definition())

        value_type = jl.seval(self.julia_type)
        if not jl.seval("T -> T isa Type")(value_type):
            raise ValueError(f"`{self.julia_type}` is not a concrete Julia type.")

        for definition in self._interface_definitions():
            jl.seval(definition)
        return value_type

    def _wrap_addprocs_function(
        self, addprocs_function: AnyValue | None, worker_imports: AnyValue | None
    ) -> AnyValue:
        """Initialize TypeSpec definitions before Julia serializes work to workers."""
        jl.seval("using Distributed: Distributed")
        if addprocs_function is None:
            addprocs_function = jl.Distributed.addprocs
        definitions = ["using Random: AbstractRNG", *self._interface_definitions()]
        if self.fields is not None:
            definitions.insert(1, self._struct_definition())
        definition = jl.Meta.parse("begin\n" + "\n".join(definitions) + "\nend")
        imports = worker_imports if worker_imports is not None else jl.nothing
        return jl.seval("""
            function (addprocs_function, definition, imports)
                return function (numprocs; kws...)
                    procs = addprocs_function(numprocs; kws...)
                    try
                        SymbolicRegression.import_module_on_workers(
                            procs, pathof(SymbolicRegression), imports, 0
                        )
                        Distributed.remotecall_eval(Main, procs, definition)
                    catch
                        Distributed.rmprocs(procs)
                        rethrow()
                    end
                    return procs
                end
            end
            """)(addprocs_function, definition, imports)

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
        ) or not self.loss_type:
            raise ValueError(
                "TypeSpec requires a loss (`elementwise_loss`, `loss_function`, or "
                "`loss_function_expression`) and `type_spec.loss_type`."
            )

    def julia_loss_type(self) -> AnyValue:
        assert self.loss_type
        loss_type = jl.seval(self.loss_type)
        if not jl.seval("T -> T isa Type")(loss_type):
            raise ValueError(
                f"`loss_type` (`{self.loss_type}`) must evaluate to a Julia type."
            )
        return loss_type

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
        if isinstance(values, np.ndarray):
            array = values if values.dtype == object else values.astype(object)
        else:
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

    def _struct_definition(self) -> str:
        assert self.fields is not None
        fields = "\n".join(
            f"    {name}::{type_}" for name, type_ in self.fields.items()
        )
        return f"struct {self.julia_type}\n{fields}\nend"

    def _interface_definitions(self) -> list[str]:
        definitions = []
        if self.init_value is not None:
            definitions.append(self._interface_definition("init", self.init_value))
        if self.sample_value is not None:
            definitions.append(self._interface_definition("sample", self.sample_value))
        if self.mutate_value is not None:
            definitions.append(self._interface_definition("mutate", self.mutate_value))
        if self.count_scalar_constants is not None:
            if isinstance(self.count_scalar_constants, int):
                definitions.append(
                    "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                    f"count_scalar_constants(value::{self.julia_type}) = {self.count_scalar_constants}"
                )
            else:
                definitions.append(
                    self._interface_definition("count", self.count_scalar_constants)
                )
        if self.can_optimize is not None:
            definitions.append(
                "SymbolicRegression.ConstantOptimizationModule."
                f"can_optimize(::Type{{{self.julia_type}}}, _) = {str(self.can_optimize).lower()}"
            )
        return definitions

    def _interface_definition(self, kind: str, source: str) -> str:
        function = jl.seval(source)
        arities = {
            int(nargs) - 1
            for nargs in jl.seval("f -> Int[m.nargs for m in methods(f)]")(function)
        }
        expected = {"init": (0,), "sample": (1, 2), "mutate": (3, 4), "count": (1,)}[
            kind
        ]
        matching = arities.intersection(expected)
        if not matching:
            raise ValueError(
                f"{kind}_value must accept {expected}; got {sorted(arities)} arguments."
            )
        arity = max(matching)

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
        return definition


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
