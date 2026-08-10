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
    """Runtime definition of a value type used by SymbolicRegression.jl.

    Parameters
    ----------
    julia_type : str
        Julia expression that evaluates to the value type used for features,
        targets, constants, and expression evaluation.
    fields : dict[str, str], optional
        Ordered mapping from field names to Julia field types. When provided,
        ``julia_type`` must be a simple name and a Julia ``struct`` with these
        fields is defined. A one-field value is converted from one Python value;
        a multi-field value is converted from a sequence in field order.
    init_value : str, optional
        Julia function with signature ``() -> value`` used to define
        ``SymbolicRegression.init_value``.
    sample_value : str, optional
        Julia function with signature ``(rng) -> value`` or
        ``(rng, options) -> value`` used to sample constants.
    mutate_value : str, optional
        Julia function with signature ``(rng, value, temperature) -> value`` or
        ``(rng, value, temperature, options) -> value`` used to mutate constants.
    count_scalar_constants : int or str, optional
        Fixed scalar count or Julia function with signature ``(value) -> Int``.
        The count must match the number of entries consumed by the pack and
        unpack hooks.
    pack_scalar_constants : str, optional
        Julia function with signature ``(buffer, idx, value) -> next_idx``. It
        must write the scalar representation of ``value`` into ``buffer``
        starting at Julia's one-based ``idx`` and return the first unused index.
    unpack_scalar_constants : str, optional
        Julia function with signature
        ``(buffer, idx, value) -> (next_idx, unpacked_value)``. It must rebuild
        a value with the same shape or structure as ``value`` and return the
        first unused index.
    get_number_type : str, optional
        Julia function with signature ``(value_type) -> scalar_type``. The
        returned type must be a subtype of ``Number`` used by optimization.
    is_valid : str, optional
        Julia function with signature ``(value) -> Bool`` indicating whether a
        value is valid for evaluation and optimization.
    can_optimize : bool, optional
        Whether constants of ``julia_type`` can be optimized. Custom non-number
        types require ``count_scalar_constants``, ``pack_scalar_constants``,
        ``unpack_scalar_constants``, ``get_number_type``, and ``is_valid`` when
        this is ``True``. The pack and unpack hooks are checked against
        ``init_value`` when both are provided.
    loss_type : str, optional
        Julia expression for the concrete ``Real`` type returned by the custom
        loss, such as ``"Float64"``.

    Notes
    -----
    Hook methods are installed in Julia's global method table. Instantiating
    another specification for the same ``julia_type`` in the same process
    replaces methods with matching signatures, so the last definition wins.

    TypeSpec training accepts a two-dimensional ``X`` with shape
    ``(n_samples, n_features)`` and a one-dimensional, single-output ``y`` with
    shape ``(n_samples,)``. Prediction accepts the same two-dimensional feature
    shape. Each array cell is one logical Julia value; nested Python values are
    preserved when passed through object arrays.

    Every search operator must accept ``julia_type`` for each argument and
    infer ``julia_type`` as its return type. For heterogeneous inner payloads,
    define a typed wrapper such as
    ``op(x::T, y::T)::T = ...`` and dispatch on the payloads inside it.
    """

    julia_type: str
    fields: dict[str, str] | None = None
    init_value: str | None = None
    sample_value: str | None = None
    mutate_value: str | None = None
    count_scalar_constants: int | str | None = None
    pack_scalar_constants: str | None = None
    unpack_scalar_constants: str | None = None
    get_number_type: str | None = None
    is_valid: str | None = None
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
            raise ValueError(f"`{self.julia_type}` does not evaluate to a Julia type.")

        for definition in self._interface_definitions():
            jl.seval(definition)
        if (
            self.can_optimize is True
            and self.pack_scalar_constants is not None
            and self.unpack_scalar_constants is not None
        ):
            self._validate_optimization_round_trip(value_type)
        return value_type

    def _validate_optimization_round_trip(self, value_type: AnyValue) -> None:
        interface = jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE
        value = jl.SymbolicRegression.init_value(value_type)
        n = int(interface.count_scalar_constants(value))
        idx = 1
        packed = jl.seval("n -> Vector{Float64}(undef, n)")(n)
        packed_idx = int(interface.pack_scalar_constants_b(packed, idx, value))
        if packed_idx != idx + n:
            raise ValueError(
                "`pack_scalar_constants` must return the first unused index "
                "after packing `init_value`."
            )
        unpacked_idx, unpacked = interface.unpack_scalar_constants(packed, idx, value)
        if int(unpacked_idx) != idx + n:
            raise ValueError(
                "`unpack_scalar_constants` must return the first unused index "
                "after unpacking `init_value`."
            )
        if not bool(jl.isequal(unpacked, value)):
            raise ValueError(
                "`pack_scalar_constants` and `unpack_scalar_constants` must "
                "round-trip `init_value`."
            )

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
        if not jl.seval("T -> isconcretetype(T) && T <: Real")(loss_type):
            raise ValueError(
                f"`loss_type` (`{self.loss_type}`) must evaluate to a concrete "
                "subtype of `Real`."
            )
        return loss_type

    @staticmethod
    def supports_export() -> bool:
        return False

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
        if self.pack_scalar_constants is not None:
            definitions.append(
                self._interface_definition("pack", self.pack_scalar_constants)
            )
        if self.unpack_scalar_constants is not None:
            definitions.append(
                self._interface_definition("unpack", self.unpack_scalar_constants)
            )
        if self.get_number_type is not None:
            definitions.append(
                self._interface_definition("number_type", self.get_number_type)
            )
        if self.is_valid is not None:
            definitions.append(self._interface_definition("valid", self.is_valid))
        can_optimize = self.can_optimize
        if can_optimize is None:
            # The backend only defines `can_optimize` for `Number` types, so
            # non-numeric value types default to `false` rather than a
            # `MethodError` when the optimizer runs.
            is_number = self.fields is None and bool(
                jl.seval(f"({self.julia_type}) <: Number")
            )
            can_optimize = None if is_number else False
        if can_optimize is not None:
            definitions.append(
                "SymbolicRegression.ConstantOptimizationModule."
                f"can_optimize(::Type{{{self.julia_type}}}, _) = {str(can_optimize).lower()}"
            )
        return definitions

    def _interface_definition(self, kind: str, source: str) -> str:
        function = jl.seval(source)
        arities = {
            int(nargs) - 1
            for nargs in jl.seval("f -> Int[m.nargs for m in methods(f)]")(function)
        }
        expected = {
            "init": (0,),
            "sample": (1, 2),
            "mutate": (3, 4),
            "count": (1,),
            "pack": (3,),
            "unpack": (3,),
            "number_type": (1,),
            "valid": (1,),
        }[kind]
        matching = arities.intersection(expected)
        if not matching:
            field = {
                "init": "init_value",
                "sample": "sample_value",
                "mutate": "mutate_value",
                "count": "count_scalar_constants",
                "pack": "pack_scalar_constants",
                "unpack": "unpack_scalar_constants",
                "number_type": "get_number_type",
                "valid": "is_valid",
            }[kind]
            raise ValueError(
                f"{field} must accept {expected}; got {sorted(arities)} arguments."
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
        elif kind == "count":
            definition = (
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                f"count_scalar_constants(value::{self.julia_type}) = ({source})(value)"
            )
        elif kind == "pack":
            definition = (
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                "pack_scalar_constants!(nvals::AbstractVector{<:Number}, "
                f"idx::Int64, value::{self.julia_type}) = ({source})(nvals, idx, value)"
            )
        elif kind == "unpack":
            definition = (
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                "unpack_scalar_constants(nvals::AbstractVector{<:Number}, "
                f"idx::Int64, value::{self.julia_type}) = ({source})(nvals, idx, value)"
            )
        elif kind == "number_type":
            definition = (
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                f"get_number_type(::Type{{{self.julia_type}}}) = ({source})({self.julia_type})"
            )
        else:
            definition = (
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                f"is_valid(value::{self.julia_type}) = ({source})(value)"
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
