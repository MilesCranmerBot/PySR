from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .julia_import import AnyValue, jl

_CODEGEN_VERSION = 2


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


@dataclass(frozen=True, init=False)
class TypeSpec:
    """Declarative definition of a custom symbolic-regression value type.

    Parameters
    ----------
    name : str
        Name of the generated Julia type. Use this name in operator and hook
        definitions.
    fields : dict[str, str]
        Ordered mapping from field names to Julia field types.
    sample : str
        Julia callable with signature ``rng -> value``.
    parameters : str, optional
        Julia callable with signature ``value -> parameters``. Provide together
        with ``with_parameters`` to enable continuous constant optimization.
    with_parameters : str, optional
        Julia callable with signature ``(value, parameters) -> value``.
    init : str, optional
        Julia callable with signature ``() -> value``. The default samples a
        value using a deterministic local random-number generator.
    mutate : str, optional
        Julia callable with signature ``(rng, value, temperature) -> value``.
        When parameters are configured, the default mutates one parameter using
        SymbolicRegression.jl's scalar constant mutation.
    is_valid : str, optional
        Julia callable with signature ``value -> Bool``. The default checks that
        every optimization parameter is finite, or accepts every value for a
        non-optimizable type.
    count_parameters : int or str, optional
        Nonnegative fixed count or Julia callable with signature
        ``value -> Int``. This is a performance override and must be provided
        together with ``pack_parameters`` and ``unpack_parameters``.
    pack_parameters : str, optional
        Julia callable with signature ``(buffer, idx, value) -> next_idx``.
    unpack_parameters : str, optional
        Julia callable with signature
        ``(buffer, idx, value) -> (next_idx, value)``.
    string : str, optional
        Julia callable with signature ``value -> AbstractString`` used to print
        constants in equations.
    preamble : str, optional
        Julia source evaluated before the generated type definition.
    loss_type : str, optional
        Concrete Julia ``AbstractFloat`` type returned by a custom full
        objective. Elementwise loss return types are inferred.
    """

    name: str
    fields: dict[str, str]
    sample: str
    parameters: str | None = None
    with_parameters: str | None = None
    init: str | None = None
    mutate: str | None = None
    is_valid: str | None = None
    count_parameters: int | str | None = None
    pack_parameters: str | None = None
    unpack_parameters: str | None = None
    string: str | None = None
    preamble: str | None = None
    loss_type: str | None = None

    def __init__(
        self,
        name: str,
        *,
        fields: dict[str, str],
        sample: str,
        parameters: str | None = None,
        with_parameters: str | None = None,
        init: str | None = None,
        mutate: str | None = None,
        is_valid: str | None = None,
        count_parameters: int | str | None = None,
        pack_parameters: str | None = None,
        unpack_parameters: str | None = None,
        string: str | None = None,
        preamble: str | None = None,
        loss_type: str | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "sample", sample)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "with_parameters", with_parameters)
        object.__setattr__(self, "init", init)
        object.__setattr__(self, "mutate", mutate)
        object.__setattr__(self, "is_valid", is_valid)
        object.__setattr__(self, "count_parameters", count_parameters)
        object.__setattr__(self, "pack_parameters", pack_parameters)
        object.__setattr__(self, "unpack_parameters", unpack_parameters)
        object.__setattr__(self, "string", string)
        object.__setattr__(self, "preamble", preamble)
        object.__setattr__(self, "loss_type", loss_type)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(f"TypeSpec name {self.name!r} is not an identifier.")
        if not isinstance(self.fields, dict) or not self.fields:
            raise ValueError("`fields` must be a non-empty ordered mapping.")
        for name, field_type in self.fields.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"TypeSpec field name {name!r} is not an identifier.")
            if not isinstance(field_type, str) or not field_type.strip():
                raise ValueError(f"TypeSpec field `{name}` requires a Julia type.")
        if not isinstance(self.sample, str) or not self.sample.strip():
            raise ValueError("`sample` must contain Julia source.")

        parameterization = (self.parameters, self.with_parameters)
        if any(value is not None for value in parameterization) and not all(
            value is not None for value in parameterization
        ):
            raise ValueError(
                "`parameters` and `with_parameters` must be provided together."
            )
        if not self.can_optimize and self.mutate is None:
            raise ValueError(
                "A non-optimizable TypeSpec requires an explicit `mutate` callable."
            )

        low_level = ("count_parameters", "pack_parameters", "unpack_parameters")
        configured = [getattr(self, name) is not None for name in low_level]
        if any(configured) and not all(configured):
            raise ValueError(
                "`count_parameters`, `pack_parameters`, and `unpack_parameters` "
                "must be provided together."
            )
        if all(configured) and not self.can_optimize:
            raise ValueError(
                "Low-level optimization overrides require `parameters` and "
                "`with_parameters`."
            )
        if isinstance(self.count_parameters, int) and self.count_parameters < 0:
            raise ValueError("`count_parameters` must be nonnegative.")

        for name in (
            "parameters",
            "with_parameters",
            "init",
            "mutate",
            "is_valid",
            "pack_parameters",
            "unpack_parameters",
            "string",
            "preamble",
            "loss_type",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` cannot be empty.")
        if self.count_parameters is not None and not isinstance(
            self.count_parameters, (int, str)
        ):
            raise ValueError(
                "`count_parameters` must be a nonnegative integer or Julia source."
            )
        if isinstance(self.count_parameters, str) and not self.count_parameters.strip():
            raise ValueError("`count_parameters` cannot be empty.")

    @property
    def can_optimize(self) -> bool:
        return self.parameters is not None


@dataclass(frozen=True)
class _TypeSpecModuleSource:
    module_name: str
    fingerprint: str
    source: str
    operator_counts: tuple[tuple[int, int], ...]
    loss_mode: str
    has_template: bool = False


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
    expression_spec: AnyValue | None


def _quoted(source: str) -> str:
    return json.dumps(source, ensure_ascii=False).replace("$", r"\$")


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
    template: str | None = None,
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
        "name": spec.name,
        "fields": list(spec.fields.items()),
        "sample": spec.sample,
        "parameters": spec.parameters,
        "with_parameters": spec.with_parameters,
        "init": spec.init,
        "mutate": spec.mutate,
        "is_valid": spec.is_valid,
        "count_parameters": spec.count_parameters,
        "pack_parameters": spec.pack_parameters,
        "unpack_parameters": spec.unpack_parameters,
        "string": spec.string,
        "preamble": spec.preamble,
        "loss_type": spec.loss_type,
        "operators": normalized_operators,
        "loss_mode": loss_mode,
        "loss_source": loss_source,
        "template": template,
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
    type_name = spec.name
    lines.extend((f"struct {type_name}", fields, "end"))
    field_values_a = ", ".join(f"a.{name}" for name in spec.fields)
    field_values_b = ", ".join(f"b.{name}" for name in spec.fields)
    field_values_x = ", ".join(f"x.{name}" for name in spec.fields)
    if len(spec.fields) == 1:
        field_values_a += ","
        field_values_b += ","
        field_values_x += ","
    lines.extend(
        (
            f"Base.:(==)(a::{type_name}, b::{type_name}) = ({field_values_a}) == ({field_values_b})",
            f"Base.isequal(a::{type_name}, b::{type_name}) = isequal(({field_values_a}), ({field_values_b}))",
            f"Base.hash(x::{type_name}, h::UInt) = hash(({field_values_x}), h)",
            f"const _sample = {_include(spec.sample, 'TypeSpec.sample')}",
        )
    )

    if spec.init is None:
        lines.append(
            f"SymbolicRegression.init_value(::Type{{{type_name}}}) = _sample(Random.Xoshiro(0))"
        )
    else:
        lines.extend(
            (
                f"const _init = {_include(spec.init, 'TypeSpec.init')}",
                f"SymbolicRegression.init_value(::Type{{{type_name}}}) = _init()",
            )
        )
    lines.append(
        f"SymbolicRegression.sample_value(rng::AbstractRNG, ::Type{{{type_name}}}, options) = _sample(rng)"
    )

    can_optimize = str(spec.can_optimize).lower()
    lines.append(
        "SymbolicRegression.ConstantOptimizationModule."
        f"can_optimize(::Type{{{type_name}}}, _) = {can_optimize}"
    )
    if spec.can_optimize:
        assert spec.parameters is not None
        assert spec.with_parameters is not None
        lines.extend(
            (
                f"const _parameters = {_include(spec.parameters, 'TypeSpec.parameters')}",
                f"const _with_parameters = {_include(spec.with_parameters, 'TypeSpec.with_parameters')}",
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                f"get_number_type(::Type{{{type_name}}}) = eltype(_parameters(SymbolicRegression.init_value({type_name})))",
            )
        )

        count_interface = (
            "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
            "count_scalar_constants"
        )
        if isinstance(spec.count_parameters, int):
            lines.append(f"{count_interface}(::{type_name}) = {spec.count_parameters}")
        elif isinstance(spec.count_parameters, str):
            lines.extend(
                (
                    f"const _count_parameters = {_include(spec.count_parameters, 'TypeSpec.count_parameters')}",
                    f"{count_interface}(value::{type_name}) = _count_parameters(value)",
                )
            )
        else:
            lines.append(
                f"{count_interface}(value::{type_name}) = length(_parameters(value))"
            )

        if spec.pack_parameters is not None:
            assert spec.unpack_parameters is not None
            lines.extend(
                (
                    f"const _pack_parameters = {_include(spec.pack_parameters, 'TypeSpec.pack_parameters')}",
                    f"const _unpack_parameters = {_include(spec.unpack_parameters, 'TypeSpec.unpack_parameters')}",
                    "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                    f"pack_scalar_constants!(buffer::AbstractVector{{<:Number}}, idx::Int, value::{type_name}) = _pack_parameters(buffer, idx, value)",
                    "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                    f"unpack_scalar_constants(buffer::AbstractVector{{<:Number}}, idx::Int, value::{type_name}) = _unpack_parameters(buffer, idx, value)",
                )
            )
        else:
            lines.extend(
                (
                    "function SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                    f"pack_scalar_constants!(buffer::AbstractVector{{<:Number}}, idx::Int, value::{type_name})",
                    "    parameters = _parameters(value)",
                    "    copyto!(buffer, idx, parameters, firstindex(parameters), length(parameters))",
                    "    return idx + length(parameters)",
                    "end",
                    "function SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                    f"unpack_scalar_constants(buffer::AbstractVector{{<:Number}}, idx::Int, value::{type_name})",
                    "    n = length(_parameters(value))",
                    "    return idx + n, _with_parameters(value, buffer[idx:(idx + n - 1)])",
                    "end",
                )
            )
    else:
        lines.append(
            "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
            f"count_scalar_constants(::{type_name}) = 0"
        )

    if spec.mutate is not None:
        lines.extend(
            (
                f"const _mutate = {_include(spec.mutate, 'TypeSpec.mutate')}",
                f"SymbolicRegression.mutate_value(rng::AbstractRNG, value::{type_name}, temperature, options) = _mutate(rng, value, temperature)",
            )
        )
    else:
        lines.extend(
            (
                f"function SymbolicRegression.mutate_value(rng::AbstractRNG, value::{type_name}, temperature, options)",
                "    parameters = copy(_parameters(value))",
                "    isempty(parameters) && return _sample(rng)",
                "    i = rand(rng, eachindex(parameters))",
                "    mutation = options === nothing ? SymbolicRegression.ConstantMutation() :",
                "        SymbolicRegression.ConstantMutation(;",
                "            perturbation_factor=options.perturbation_factor,",
                "            probability_negate=options.probability_negate_constant,",
                "        )",
                "    parameters[i] = SymbolicRegression.MutationFunctionsModule."
                "mutate_value(rng, parameters[i], temperature, mutation)",
                "    return _with_parameters(value, parameters)",
                "end",
            )
        )

    if spec.is_valid is not None:
        lines.extend(
            (
                f"const _is_valid = {_include(spec.is_valid, 'TypeSpec.is_valid')}",
                "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
                f"is_valid(value::{type_name}) = _is_valid(value)",
            )
        )
    elif spec.can_optimize:
        lines.append(
            "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
            f"is_valid(value::{type_name}) = all(isfinite, _parameters(value))"
        )
    else:
        lines.append(
            "SymbolicRegression.InterfaceDynamicExpressionsModule.DE."
            f"is_valid(::{type_name}) = true"
        )

    if spec.string is not None:
        lines.append(f"const _string = {_include(spec.string, 'TypeSpec.string')}")
    elif len(spec.fields) == 1:
        only_field = next(iter(spec.fields))
        lines.append(
            f"_string(value::{type_name}) = sprint(show, value.{only_field}; context=:compact => true)"
        )
    else:
        formatted_fields = ", ".join(
            f"sprint(show, value.{name}; context=:compact => true)"
            for name in spec.fields
        )
        lines.append(
            f'_string(value::{type_name}) = Base.string("{type_name}(", join(({formatted_fields}), ", "), ")")'
        )
    lines.extend(
        (
            f"Base.show(io::IO, value::{type_name}) = print(io, _string(value))",
            "SymbolicRegression.InterfaceDynamicExpressionsModule.DE.StringsModule."
            f"needs_brackets(::{type_name}) = false",
            "SymbolicRegression.InterfaceDynamicExpressionsModule."
            f"string_constant(value::{type_name}, ::Val{{precision}}, unit) where {{precision}} = _string(value) * unit",
            "SymbolicRegression.InterfaceDynamicExpressionsModule."
            f"string_constant(value::{type_name}, bracketed, ::Val{{precision}}, unit) where {{precision}} = _string(value) * unit",
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
            f"_elementwise_loss(a::{type_name}, b::{type_name}) = _elementwise_loss_impl(a, b)"
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
    if template is not None:
        lines.append(
            "const _template_expression_spec = "
            + _include(template, "TypeSpec.template")
        )

    lines.extend(
        (
            "function _convert_value(x)",
            f"    x isa {type_name} && return x",
            "    x = PythonCall.Py(x)",
        )
    )
    if len(spec.fields) == 1:
        lines.append(
            f"    return {type_name}(PythonCall.pyconvert(fieldtype({type_name}, 1), x))"
        )
    else:
        arguments = ", ".join(
            f"PythonCall.pyconvert(fieldtype({type_name}, {i}), x[{i - 1}])"
            for i in range(1, len(spec.fields) + 1)
        )
        lines.append(f"    return {type_name}({arguments})")
    lines.extend(("end", "end"))

    return _TypeSpecModuleSource(
        module_name=module_name,
        fingerprint=fingerprint,
        source="\n".join(lines) + "\n",
        operator_counts=tuple(operator_counts),
        loss_mode=loss_mode,
        has_template=template is not None,
    )


def load_type_spec_runtime(
    spec: TypeSpec, module_source: _TypeSpecModuleSource, *, validate: bool = True
) -> _TypeSpecRuntime:
    """Load a generated module and return its ephemeral Julia objects."""
    module_symbol = jl.Symbol(module_source.module_name)
    if not bool(jl.isdefined(jl.Main, module_symbol)):
        jl.Base.include_string(
            jl.Main,
            module_source.source,
            "PySR." + module_source.module_name,
        )
    module = jl.getproperty(jl.Main, module_symbol)
    value_type = jl.getproperty(module, jl.Symbol(spec.name))

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
    expression_spec = (
        module._template_expression_spec if module_source.has_template else None
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
        expression_spec=expression_spec,
    )
    if validate:
        _validate_type_spec_runtime(runtime)
    return runtime


def _call_hook(name: str, function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except Exception as error:
        raise ValueError(f"TypeSpec `{name}` failed its required contract.") from error


def _validate_value(runtime: _TypeSpecRuntime, value: Any, hook: str) -> None:
    if not bool(jl.isa(value, runtime.value_type)):
        raise ValueError(f"TypeSpec `{hook}` must return `{runtime.spec.name}`.")
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
    runtime: _TypeSpecRuntime, value: Any, count: int
) -> None:
    interface = jl.SymbolicRegression.InterfaceDynamicExpressionsModule.DE
    parameters = _call_hook("parameters", runtime.module._parameters, value)
    if not bool(jl.seval("x -> x isa AbstractVector")(parameters)):
        raise ValueError("TypeSpec `parameters` must return an `AbstractVector`.")
    if len(parameters) != count:
        raise ValueError("TypeSpec `count_parameters` disagrees with `parameters`.")
    number_type = interface.get_number_type(runtime.value_type)
    if not bool(jl.seval("T -> isconcretetype(T) && T <: AbstractFloat")(number_type)):
        raise ValueError(
            "TypeSpec `parameters` must return a vector with a concrete "
            "`AbstractFloat` element type."
        )
    packed = jl.seval("(T, n) -> Vector{T}(undef, n)")(number_type, count)
    packed_idx = _call_hook(
        "pack_parameters", interface.pack_scalar_constants_b, packed, 1, value
    )
    if packed_idx != count + 1:
        raise ValueError("TypeSpec `pack_parameters` returned the wrong next index.")
    if not bool(jl.isequal(packed, parameters)):
        raise ValueError("TypeSpec `pack_parameters` disagrees with `parameters`.")

    rebuilt = _call_hook(
        "with_parameters", runtime.module._with_parameters, value, parameters
    )
    if not bool(jl.isa(rebuilt, runtime.value_type)):
        raise ValueError(
            f"TypeSpec `with_parameters` must return `{runtime.spec.name}`."
        )
    if not bool(jl.isequal(rebuilt, value)):
        raise ValueError(
            "TypeSpec `with_parameters(value, parameters(value))` must preserve "
            "the value."
        )
    unpacked_result = _call_hook(
        "unpack_parameters", interface.unpack_scalar_constants, packed, 1, value
    )
    try:
        unpacked_idx, unpacked = unpacked_result
    except Exception as error:
        raise ValueError(
            "TypeSpec `unpack_parameters` must return "
            f"`(next_idx, {runtime.spec.name})`."
        ) from error
    if unpacked_idx != count + 1:
        raise ValueError("TypeSpec `unpack_parameters` returned the wrong next index.")
    if not bool(jl.isa(unpacked, runtime.value_type)):
        raise ValueError(
            f"TypeSpec `unpack_parameters` must return `{runtime.spec.name}`."
        )
    repacked = jl.seval("similar")(packed)
    repacked_idx = _call_hook(
        "pack_parameters",
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
    sampled = _call_hook("sample", sr.sample_value, rng, runtime.value_type, jl.nothing)
    _validate_value(runtime, sampled, "sample")
    initial = _call_hook("init", sr.init_value, runtime.value_type)
    _validate_value(runtime, initial, "init")

    interface = sr.InterfaceDynamicExpressionsModule.DE
    counts = []
    for value in (initial, sampled):
        count = _call_hook("count_parameters", interface.count_scalar_constants, value)
        if not isinstance(count, int) or count < 0:
            raise ValueError(
                "TypeSpec `count_parameters` must return a nonnegative `Int`."
            )
        counts.append(count)
    if runtime.spec.can_optimize:
        _validate_optimization_value(runtime, initial, counts[0])
        _validate_optimization_value(runtime, sampled, counts[1])

    mutated = _call_hook("mutate", sr.mutate_value, rng, sampled, 1.0, jl.nothing)
    _validate_value(runtime, mutated, "mutate")
    mutated_count = _call_hook(
        "count_parameters", interface.count_scalar_constants, mutated
    )
    if not isinstance(mutated_count, int) or mutated_count < 0:
        raise ValueError("TypeSpec `count_parameters` must return a nonnegative `Int`.")

    formatted = _call_hook("string", runtime.module._string, sampled)
    if not bool(jl.seval("x -> x isa AbstractString")(formatted)):
        raise ValueError("TypeSpec `string` must return an `AbstractString`.")

    for arity, functions in runtime.operators.items():
        for function in functions:
            result = _call_hook(
                str(jl.nameof(function)), function, *([sampled] * arity)
            )
            if not bool(jl.isa(result, runtime.value_type)):
                raise ValueError(
                    f"TypeSpec operator `{jl.nameof(function)}` must return "
                    f"`{runtime.spec.name}`."
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
