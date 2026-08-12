from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .julia_import import AnyValue, jl

_CODEGEN_VERSION = 1


def object_array_1d(values: Any) -> np.ndarray:
    """Build a 1D object array whose cells are the logical values themselves."""
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
        values = list(values)
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
    """Declarative definition of a custom symbolic-regression value type.

    Parameters
    ----------
    fields : dict[str, str]
        Ordered mapping from field names to Julia field types. PySR generates a
        private Julia ``Value`` wrapper containing these fields.
    init_value : str
        Julia callable with signature ``() -> Value``.
    sample_value : str
        Julia callable with signature ``(rng, options) -> Value``.
    mutate_value : str
        Julia callable with signature
        ``(rng, value, temperature, options) -> Value``.
    count_scalar_constants : int or str
        Nonnegative fixed count or a Julia callable with signature
        ``(value) -> Int``.
    is_valid : str
        Julia callable with signature ``(value) -> Bool``.
    can_optimize : bool
        Explicitly enables or disables continuous constant optimization.
    preamble : str, optional
        Julia source evaluated before the generated ``Value`` definition.
    pack_scalar_constants : str, optional
        Julia callable with signature ``(buffer, idx, value) -> next_idx``.
        Required when ``can_optimize=True``.
    unpack_scalar_constants : str, optional
        Julia callable with signature
        ``(buffer, idx, value) -> (next_idx, value)``. Required when
        ``can_optimize=True``.
    number_type : str, optional
        Concrete Julia ``AbstractFloat`` type used by constant optimization.
        Required when ``can_optimize=True``.
    loss_type : str, optional
        Concrete Julia ``AbstractFloat`` type returned by a custom full
        objective. Elementwise loss return types are inferred.
    """

    fields: dict[str, str]
    init_value: str
    sample_value: str
    mutate_value: str
    count_scalar_constants: int | str
    is_valid: str
    can_optimize: bool
    preamble: str | None = None
    pack_scalar_constants: str | None = None
    unpack_scalar_constants: str | None = None
    number_type: str | None = None
    loss_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fields, dict) or not self.fields:
            raise ValueError("`fields` must be a non-empty ordered mapping.")
        for name, field_type in self.fields.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"TypeSpec field name {name!r} is not an identifier.")
            if not isinstance(field_type, str) or not field_type.strip():
                raise ValueError(f"TypeSpec field `{name}` requires a Julia type.")
        for name in ("init_value", "sample_value", "mutate_value", "is_valid"):
            source = getattr(self, name)
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"`{name}` must contain Julia source.")
        if isinstance(self.count_scalar_constants, int):
            if self.count_scalar_constants < 0:
                raise ValueError("`count_scalar_constants` must be nonnegative.")
        elif not (
            isinstance(self.count_scalar_constants, str)
            and self.count_scalar_constants.strip()
        ):
            raise ValueError(
                "`count_scalar_constants` must be a nonnegative integer or Julia source."
            )
        if not isinstance(self.can_optimize, bool):
            raise ValueError("`can_optimize` must be explicitly set to true or false.")

        optimization_fields = (
            "pack_scalar_constants",
            "unpack_scalar_constants",
            "number_type",
        )
        configured = [getattr(self, name) is not None for name in optimization_fields]
        if self.can_optimize and not all(configured):
            missing = [
                name
                for name, is_configured in zip(optimization_fields, configured)
                if not is_configured
            ]
            raise ValueError(
                "`can_optimize=True` requires "
                + ", ".join(f"`{x}`" for x in missing)
                + "."
            )
        if not self.can_optimize and any(configured):
            raise ValueError(
                "Optimization hooks require `can_optimize=True`; remove them or enable optimization."
            )
        for name in (*optimization_fields, "preamble", "loss_type"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` cannot be empty.")


@dataclass(frozen=True)
class _TypeSpecModuleSource:
    module_name: str
    fingerprint: str
    source: str
    operator_counts: tuple[tuple[int, int], ...]
    loss_mode: str


@dataclass(frozen=True)
class _TypeSpecRuntime:
    spec: TypeSpec
    module_source: _TypeSpecModuleSource
    module: AnyValue
    value_type: AnyValue
    operators: dict[int, tuple[AnyValue, ...]]
    operator_names: dict[int, list[str]]
    elementwise_loss: AnyValue | None
    loss_function: AnyValue | None
    loss_function_expression: AnyValue | None
    loss_type: AnyValue


def _quoted(source: str) -> str:
    return json.dumps(source, ensure_ascii=False)


def _include(source: str, label: str) -> str:
    return f"Base.include_string(@__MODULE__, {_quoted(source)}, {_quoted(label)})"


def _loss_configuration(
    elementwise_loss: str | None,
    loss_function: str | None,
    loss_function_expression: str | None,
) -> tuple[str, str]:
    configured = [
        ("elementwise_loss", elementwise_loss),
        ("loss_function", loss_function),
        ("loss_function_expression", loss_function_expression),
    ]
    selected = [(mode, source) for mode, source in configured if source is not None]
    if len(selected) != 1:
        raise ValueError(
            "TypeSpec requires exactly one of `elementwise_loss`, `loss_function`, "
            "and `loss_function_expression`."
        )
    mode, source = selected[0]
    assert source is not None
    return mode, source


def _normalize_operators(
    operators: dict[int, list[str]] | None,
) -> list[tuple[int, list[str]]]:
    if not operators:
        raise ValueError("TypeSpec requires explicit `operators={...}`.")
    normalized = []
    for arity in sorted(operators):
        sources = operators[arity]
        if not isinstance(arity, int) or arity < 1:
            raise ValueError("TypeSpec operator arities must be positive integers.")
        if not sources:
            raise ValueError(f"TypeSpec operator arity {arity} cannot be empty.")
        if any(not isinstance(source, str) or not source.strip() for source in sources):
            raise ValueError("Every TypeSpec operator must contain Julia source.")
        normalized.append((arity, list(sources)))
    return normalized


def build_type_spec_module_source(
    spec: TypeSpec,
    operators: dict[int, list[str]] | None,
    *,
    elementwise_loss: str | None,
    loss_function: str | None,
    loss_function_expression: str | None,
) -> _TypeSpecModuleSource:
    """Create deterministic Julia source without evaluating Julia code."""
    normalized_operators = _normalize_operators(operators)
    loss_mode, loss_source = _loss_configuration(
        elementwise_loss, loss_function, loss_function_expression
    )
    if loss_mode == "elementwise_loss" and spec.loss_type is not None:
        raise ValueError(
            "Do not set `loss_type` with `elementwise_loss`; its return type is inferred."
        )
    if loss_mode != "elementwise_loss" and spec.loss_type is None:
        raise ValueError("TypeSpec full objectives require an explicit `loss_type`.")
    payload = {
        "codegen_version": _CODEGEN_VERSION,
        "fields": list(spec.fields.items()),
        "init_value": spec.init_value,
        "sample_value": spec.sample_value,
        "mutate_value": spec.mutate_value,
        "count_scalar_constants": spec.count_scalar_constants,
        "is_valid": spec.is_valid,
        "can_optimize": spec.can_optimize,
        "preamble": spec.preamble,
        "pack_scalar_constants": spec.pack_scalar_constants,
        "unpack_scalar_constants": spec.unpack_scalar_constants,
        "number_type": spec.number_type,
        "loss_type": spec.loss_type,
        "operators": normalized_operators,
        "loss_mode": loss_mode,
        "loss_source": loss_source,
    }
    payload_bytes = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode()
    fingerprint = hashlib.sha256(payload_bytes).hexdigest()
    module_name = f"_PySRTypeSpec_{fingerprint[:20]}"

    lines = [
        f"module {module_name}",
        "using Random",
        "using SymbolicRegression",
        "using PythonCall",
    ]
    if spec.preamble is not None:
        lines.append(_include(spec.preamble, "TypeSpec.preamble"))

    fields = "\n".join(
        f"    {name}::{field_type}" for name, field_type in spec.fields.items()
    )
    lines.extend(("struct Value", fields, "end"))
    field_values_a = ", ".join(f"a.{name}" for name in spec.fields)
    field_values_b = ", ".join(f"b.{name}" for name in spec.fields)
    field_values_x = ", ".join(f"x.{name}" for name in spec.fields)
    if len(spec.fields) == 1:
        field_values_a += ","
        field_values_b += ","
        field_values_x += ","
    lines.extend(
        (
            f"Base.:(==)(a::Value, b::Value) = ({field_values_a}) == ({field_values_b})",
            f"Base.isequal(a::Value, b::Value) = isequal(({field_values_a}), ({field_values_b}))",
            f"Base.hash(x::Value, h::UInt) = hash(({field_values_x}), h)",
            f"const _init_value = {_include(spec.init_value, 'TypeSpec.init_value')}",
            f"const _sample_value = {_include(spec.sample_value, 'TypeSpec.sample_value')}",
            f"const _mutate_value = {_include(spec.mutate_value, 'TypeSpec.mutate_value')}",
            f"const _is_valid = {_include(spec.is_valid, 'TypeSpec.is_valid')}",
            "SymbolicRegression.init_value(::Type{Value}) = _init_value()",
            "SymbolicRegression.sample_value(rng::AbstractRNG, ::Type{Value}, options) = _sample_value(rng, options)",
            "SymbolicRegression.mutate_value(rng::AbstractRNG, value::Value, temperature, options) = _mutate_value(rng, value, temperature, options)",
            "SymbolicRegression.InterfaceDynamicExpressionsModule.DE.is_valid(value::Value) = _is_valid(value)",
        )
    )

    count_interface = (
        "SymbolicRegression.InterfaceDynamicExpressionsModule.DE.count_scalar_constants"
    )
    if isinstance(spec.count_scalar_constants, int):
        lines.append(f"{count_interface}(::Value) = {spec.count_scalar_constants}")
    else:
        lines.extend(
            (
                f"const _count_scalar_constants = {_include(spec.count_scalar_constants, 'TypeSpec.count_scalar_constants')}",
                f"{count_interface}(value::Value) = _count_scalar_constants(value)",
            )
        )

    can_optimize = str(spec.can_optimize).lower()
    lines.append(
        "SymbolicRegression.ConstantOptimizationModule."
        f"can_optimize(::Type{{Value}}, _) = {can_optimize}"
    )
    if spec.can_optimize:
        assert spec.pack_scalar_constants is not None
        assert spec.unpack_scalar_constants is not None
        assert spec.number_type is not None
        lines.extend(
            (
                f"const _pack_scalar_constants = {_include(spec.pack_scalar_constants, 'TypeSpec.pack_scalar_constants')}",
                f"const _unpack_scalar_constants = {_include(spec.unpack_scalar_constants, 'TypeSpec.unpack_scalar_constants')}",
                f"const _number_type = {_include(spec.number_type, 'TypeSpec.number_type')}",
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE.pack_scalar_constants!(buffer::AbstractVector{<:Number}, idx::Int, value::Value) = _pack_scalar_constants(buffer, idx, value)",
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE.unpack_scalar_constants(buffer::AbstractVector{<:Number}, idx::Int, value::Value) = _unpack_scalar_constants(buffer, idx, value)",
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE.get_number_type(::Type{Value}) = _number_type",
            )
        )

    operator_counts = []
    for arity, sources in normalized_operators:
        operator_counts.append((arity, len(sources)))
        for index, source in enumerate(sources, start=1):
            lines.append(
                f"const _operator_{arity}_{index} = "
                + _include(source, f"TypeSpec.operator[{arity}][{index}]")
            )

    lines.append(
        f"const _{loss_mode}_impl = " + _include(loss_source, f"TypeSpec.{loss_mode}")
    )
    if loss_mode == "elementwise_loss":
        lines.append(
            "_elementwise_loss(a::Value, b::Value) = _elementwise_loss_impl(a, b)"
        )
    elif loss_mode == "loss_function":
        lines.append(
            "_loss_function(tree, dataset, options) = "
            "_loss_function_impl(tree, dataset, options)"
        )
    else:
        lines.append(
            "_loss_function_expression(expression, dataset, options) = "
            "_loss_function_expression_impl(expression, dataset, options)"
        )
    if spec.loss_type is not None:
        lines.append(
            f"const _loss_type = {_include(spec.loss_type, 'TypeSpec.loss_type')}"
        )

    lines.extend(
        (
            "function _convert_value(x)",
            "    x isa Value && return x",
            "    x = PythonCall.Py(x)",
        )
    )
    if len(spec.fields) == 1:
        lines.append("    return Value(PythonCall.pyconvert(fieldtype(Value, 1), x))")
    else:
        arguments = ", ".join(
            f"PythonCall.pyconvert(fieldtype(Value, {i}), x[{i - 1}])"
            for i in range(1, len(spec.fields) + 1)
        )
        lines.append(f"    return Value({arguments})")
    lines.extend(("end", "end"))

    return _TypeSpecModuleSource(
        module_name=module_name,
        fingerprint=fingerprint,
        source="\n".join(lines) + "\n",
        operator_counts=tuple(operator_counts),
        loss_mode=loss_mode,
    )


def load_type_spec_runtime(
    spec: TypeSpec, module_source: _TypeSpecModuleSource, *, validate: bool = True
) -> _TypeSpecRuntime:
    """Load a generated module and return its ephemeral Julia objects."""
    module = jl.seval(f"""
        if !isdefined(Main, :{module_source.module_name})
            Base.include_string(
                Main,
                {_quoted(module_source.source)},
                {_quoted('PySR.' + module_source.module_name)},
            )
        end
        getfield(Main, :{module_source.module_name})
        """)
    value_type = module.Value

    operators = {}
    operator_names = {}
    for arity, count in module_source.operator_counts:
        functions = tuple(
            jl.getproperty(module, jl.Symbol(f"_operator_{arity}_{index}"))
            for index in range(1, count + 1)
        )
        if any(not bool(jl.seval("x -> x isa Function")(f)) for f in functions):
            raise ValueError(
                "Every TypeSpec operator must evaluate to a Julia function."
            )
        names = [str(jl.nameof(function)) for function in functions]
        if any(name.startswith("#") for name in names):
            raise ValueError("TypeSpec operators must be named Julia functions.")
        operators[arity] = functions
        operator_names[arity] = names

    elementwise_loss = None
    loss_function = None
    loss_function_expression = None
    if module_source.loss_mode == "elementwise_loss":
        elementwise_loss = module._elementwise_loss
        loss_type = jl.Base.promote_op(elementwise_loss, value_type, value_type)
    else:
        loss = jl.getproperty(module, jl.Symbol(f"_{module_source.loss_mode}"))
        if module_source.loss_mode == "loss_function":
            loss_function = loss
        else:
            loss_function_expression = loss
        loss_type = module._loss_type
    if not bool(jl.seval("T -> isconcretetype(T) && T <: AbstractFloat")(loss_type)):
        raise ValueError(
            "The TypeSpec loss must return a concrete subtype of `AbstractFloat`; "
            f"got `{loss_type}`. Add a concrete Julia return type annotation."
        )

    runtime = _TypeSpecRuntime(
        spec=copy.deepcopy(spec),
        module_source=module_source,
        module=module,
        value_type=value_type,
        operators=operators,
        operator_names=operator_names,
        elementwise_loss=elementwise_loss,
        loss_function=loss_function,
        loss_function_expression=loss_function_expression,
        loss_type=loss_type,
    )
    if validate:
        _validate_type_spec_runtime(runtime)
    return runtime


def _call_hook(name: str, function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except Exception as error:
        raise ValueError(f"TypeSpec `{name}` failed its required contract.") from error


def _validate_value(runtime: _TypeSpecRuntime, value: AnyValue, hook: str) -> None:
    if not bool(jl.isa(value, runtime.value_type)):
        raise ValueError(f"TypeSpec `{hook}` must return `Value`.")
    is_valid = _call_hook(
        "is_valid",
        jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE.is_valid,
        value,
    )
    if not isinstance(is_valid, bool):
        raise ValueError("TypeSpec `is_valid` must return `Bool`.")
    if not is_valid:
        raise ValueError(f"TypeSpec `{hook}` returned an invalid value.")


def _validate_optimization_value(
    runtime: _TypeSpecRuntime, value: AnyValue, count: int
) -> None:
    interface = jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE
    number_type = interface.get_number_type(runtime.value_type)
    if not bool(jl.seval("T -> isconcretetype(T) && T <: AbstractFloat")(number_type)):
        raise ValueError(
            "TypeSpec `number_type` must be a concrete `AbstractFloat` type."
        )
    packed = jl.seval("(T, n) -> Vector{T}(undef, n)")(number_type, count)
    packed_idx = _call_hook(
        "pack_scalar_constants", interface.pack_scalar_constants_b, packed, 1, value
    )
    if packed_idx != count + 1:
        raise ValueError(
            "TypeSpec `pack_scalar_constants` returned the wrong next index."
        )
    unpacked_result = _call_hook(
        "unpack_scalar_constants", interface.unpack_scalar_constants, packed, 1, value
    )
    try:
        unpacked_idx, unpacked = unpacked_result
    except Exception as error:
        raise ValueError(
            "TypeSpec `unpack_scalar_constants` must return `(next_idx, Value)`."
        ) from error
    if unpacked_idx != count + 1:
        raise ValueError(
            "TypeSpec `unpack_scalar_constants` returned the wrong next index."
        )
    if not bool(jl.isa(unpacked, runtime.value_type)):
        raise ValueError("TypeSpec `unpack_scalar_constants` must return `Value`.")
    repacked = jl.seval("similar")(packed)
    repacked_idx = _call_hook(
        "pack_scalar_constants",
        interface.pack_scalar_constants_b,
        repacked,
        1,
        unpacked,
    )
    if repacked_idx != count + 1 or not bool(jl.isequal(packed, repacked)):
        raise ValueError(
            "TypeSpec optimization hooks must preserve the packed scalar representation."
        )


def _validate_type_spec_runtime(runtime: _TypeSpecRuntime) -> None:
    sr = jl.SymbolicRegression
    rng = runtime.module.Xoshiro(0)
    initial = _call_hook("init_value", sr.init_value, runtime.value_type)
    _validate_value(runtime, initial, "init_value")
    sampled = _call_hook(
        "sample_value", sr.sample_value, rng, runtime.value_type, jl.nothing
    )
    _validate_value(runtime, sampled, "sample_value")
    mutated = _call_hook("mutate_value", sr.mutate_value, rng, sampled, 1.0, jl.nothing)
    _validate_value(runtime, mutated, "mutate_value")

    interface = sr.InterfaceDynamicExpressionsModule.DE
    counts = []
    for value in (initial, sampled, mutated):
        count = _call_hook(
            "count_scalar_constants", interface.count_scalar_constants, value
        )
        if not isinstance(count, int) or count < 0:
            raise ValueError(
                "TypeSpec `count_scalar_constants` must return a nonnegative `Int`."
            )
        counts.append(count)
    if runtime.spec.can_optimize:
        _validate_optimization_value(runtime, initial, counts[0])
        _validate_optimization_value(runtime, sampled, counts[1])

    for arity, functions in runtime.operators.items():
        for function in functions:
            result = _call_hook(
                str(jl.nameof(function)), function, *([sampled] * arity)
            )
            if not bool(jl.isa(result, runtime.value_type)):
                raise ValueError(
                    f"TypeSpec operator `{jl.nameof(function)}` must return `Value`."
                )
    if runtime.elementwise_loss is not None:
        _call_hook("elementwise_loss", runtime.elementwise_loss, sampled, sampled)


def type_spec_to_julia_array(
    runtime: _TypeSpecRuntime, values: Any, *, transpose: bool = False
) -> AnyValue:
    """Convert logical Python values to an array of the generated Julia type."""
    array = (
        values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    )
    if array.dtype != object:
        array = array.astype(object)
    if transpose:
        array = array.T
    if array.ndim not in (1, 2):
        raise ValueError("TypeSpec data must be a 1D or 2D array.")

    converted = []
    nfields = len(runtime.spec.fields)
    for value in array.ravel(order="F"):
        if not bool(jl.isa(value, runtime.value_type)) and nfields > 1:
            if isinstance(value, (str, bytes)):
                raise ValueError(f"TypeSpec values require exactly {nfields} fields.")
            try:
                size = len(value)
            except TypeError as error:
                raise ValueError(
                    f"TypeSpec values require exactly {nfields} fields."
                ) from error
            if size != nfields:
                raise ValueError(f"TypeSpec values require exactly {nfields} fields.")
        converted.append(runtime.module._convert_value(value))
    return jl.seval("(T, xs, dims) -> reshape(T[x for x in xs], Tuple(dims))")(
        runtime.value_type, converted, array.shape
    )


def type_spec_to_python_array(runtime: _TypeSpecRuntime, values: Any) -> np.ndarray:
    """Unwrap generated values into their logical one-field or tuple payloads."""
    field_names = tuple(runtime.spec.fields)
    output = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        if len(field_names) == 1:
            output[i] = getattr(value, field_names[0])
        else:
            output[i] = tuple(getattr(value, name) for name in field_names)
    return output


def wrap_type_spec_addprocs_function(
    module_source: _TypeSpecModuleSource,
    addprocs_function: AnyValue | None,
    worker_imports: AnyValue | None,
) -> AnyValue:
    """Create workers and install the private module before function transfer."""
    jl.seval("using Distributed: Distributed")
    if addprocs_function is None:
        addprocs_function = jl.Distributed.addprocs
    definition = jl.Meta.parse(module_source.source)
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
