from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from functools import cache
from textwrap import dedent
from typing import Any

import numpy as np
from juliacall import JuliaError  # type: ignore

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

        if (self.parameters is None) != (self.with_parameters is None):
            raise ValueError(
                "`parameters` and `with_parameters` must be provided together."
            )
        if not self.can_optimize and self.mutate is None:
            raise ValueError(
                "A non-optimizable TypeSpec requires an explicit `mutate` callable."
            )

        low_level = ("count_parameters", "pack_parameters", "unpack_parameters")
        configured = tuple(getattr(self, name) is not None for name in low_level)
        if any(configured) != all(configured):
            raise ValueError(
                "`count_parameters`, `pack_parameters`, and `unpack_parameters` "
                "must be provided together."
            )
        if all(configured) and not self.can_optimize:
            raise ValueError(
                "Low-level optimization overrides require `parameters` and "
                "`with_parameters`."
            )
        if type(self.count_parameters) is int and self.count_parameters < 0:
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
        if self.count_parameters is not None and not (
            type(self.count_parameters) is int or isinstance(self.count_parameters, str)
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
class _TypeSpecDefinition:
    spec: TypeSpec
    fingerprint: str
    source: str

    @property
    def module_name(self) -> str:
        return f"_PySRTypeSpec_{self.fingerprint[:20]}"


@dataclass(frozen=True)
class _TypeSpecRuntime:
    definition: _TypeSpecDefinition
    module: AnyValue
    value_type: AnyValue
    operators: dict[int, tuple[AnyValue, ...]]
    operator_names: dict[int, list[str]]
    elementwise_loss: AnyValue | None
    loss_function: AnyValue | None
    loss_function_expression: AnyValue | None
    loss_type: AnyValue
    expression_spec: AnyValue | None

    @property
    def spec(self) -> TypeSpec:
        return self.definition.spec

    def map_operator_keys(self, values: dict[str, Any], option: str) -> AnyValue:
        operators = {
            name: function
            for arity, names in self.operator_names.items()
            for name, function in zip(names, self.operators[arity])
        }

        def convert(mapping: dict[str, Any]) -> AnyValue:
            try:
                return jl.Dict(
                    [
                        jl.Pair(
                            operators[name],
                            convert(value) if isinstance(value, dict) else value,
                        )
                        for name, value in mapping.items()
                    ]
                )
            except KeyError as error:
                raise ValueError(
                    f"Unknown TypeSpec operator in `{option}`: {error.args[0]!r}."
                ) from error

        return convert(values)


def _quoted(source: str) -> str:
    return json.dumps(source, ensure_ascii=False).replace("$", r"\$")


def _block(source: str) -> str:
    return dedent(source).strip()


_TYPE_SPEC_MODULE = _block(r"""
    import SymbolicRegression: init_value, mutate_value, sample_value
    import SymbolicRegression.ConstantOptimizationModule: can_optimize
    import SymbolicRegression.InterfaceDynamicExpressionsModule: string_constant
    import SymbolicRegression.InterfaceDynamicExpressionsModule.DE:
        count_scalar_constants, get_number_type, is_valid,
        pack_scalar_constants!, unpack_scalar_constants
    import SymbolicRegression.InterfaceDynamicExpressionsModule.DE.StringsModule:
        needs_brackets

    const _config = __TYPE_SPEC_CONFIG__
    abstract type _TypeSpecValue end

    macro _define_type_spec(config_expression)
        config = Core.eval(__module__, config_expression)
        fields = map(config.fields) do (name, type)
            :($(Symbol(name))::$(Meta.parse(type)))
        end
        type_definition = Expr(
            :struct,
            false,
            Expr(:<:, Symbol(config.name), :_TypeSpecValue),
            Expr(:block, fields...),
        )
        return esc(type_definition)
    end

    _include(source, label) = Base.include_string(@__MODULE__, source, label)
    _config.preamble === nothing ||
        _include(_config.preamble, "TypeSpec.preamble")
    @_define_type_spec _config
    const _value_type = getfield(@__MODULE__, Symbol(_config.name))

    _fields(value) = ntuple(i -> getfield(value, i), fieldcount(_value_type))
    Base.:(==)(a::_TypeSpecValue, b::_TypeSpecValue) = _fields(a) == _fields(b)
    Base.isequal(a::_TypeSpecValue, b::_TypeSpecValue) =
        isequal(_fields(a), _fields(b))
    Base.hash(value::_TypeSpecValue, h::UInt) = hash(_fields(value), h)

    const _sample = _include(_config.sample, "TypeSpec.sample")
    const _init = _config.init === nothing ?
        () -> _sample(Random.Xoshiro(0)) :
        _include(_config.init, "TypeSpec.init")
    init_value(::Type{<:_TypeSpecValue}) = _init()
    sample_value(rng::AbstractRNG, ::Type{<:_TypeSpecValue}, options) = _sample(rng)
    can_optimize(::Type{<:_TypeSpecValue}, _) = _config.optimizable

    if _config.optimizable
        const _parameters = _include(_config.parameters, "TypeSpec.parameters")
        const _with_parameters =
            _include(_config.with_parameters, "TypeSpec.with_parameters")
        get_number_type(::Type{<:_TypeSpecValue}) =
            eltype(_parameters(init_value(_value_type)))
    end

    const _count_parameters = if !_config.optimizable
        _ -> 0
    elseif _config.count_parameters isa Int
        _ -> _config.count_parameters
    elseif _config.count_parameters isa String
        _include(_config.count_parameters, "TypeSpec.count_parameters")
    else
        value -> length(_parameters(value))
    end
    count_scalar_constants(value::_TypeSpecValue) = _count_parameters(value)

    if _config.optimizable
        const _pack_parameters = if _config.pack_parameters === nothing
            function (buffer, idx, value)
                parameters = _parameters(value)
                copyto!(
                    buffer,
                    idx,
                    parameters,
                    firstindex(parameters),
                    length(parameters),
                )
                return idx + length(parameters)
            end
        else
            _include(_config.pack_parameters, "TypeSpec.pack_parameters")
        end
        const _unpack_parameters = if _config.unpack_parameters === nothing
            function (buffer, idx, value)
                n = length(_parameters(value))
                return idx + n,
                _with_parameters(value, buffer[idx:(idx + n - 1)])
            end
        else
            _include(_config.unpack_parameters, "TypeSpec.unpack_parameters")
        end
        pack_scalar_constants!(
            buffer::AbstractVector{<:Number}, idx::Int, value::_TypeSpecValue
        ) = _pack_parameters(buffer, idx, value)
        unpack_scalar_constants(
            buffer::AbstractVector{<:Number}, idx::Int, value::_TypeSpecValue
        ) = _unpack_parameters(buffer, idx, value)
    end

    const _mutate = if _config.mutate === nothing
        function (rng, value, temperature, mutation)
            parameters = collect(_parameters(value))
            isempty(parameters) && return _sample(rng)
            i = rand(rng, eachindex(parameters))
            parameters[i] = SymbolicRegression.MutationFunctionsModule.mutate_value(
                rng, parameters[i], temperature, mutation
            )
            return _with_parameters(value, parameters)
        end
    else
        mutate = _include(_config.mutate, "TypeSpec.mutate")
        (rng, value, temperature, _) -> mutate(rng, value, temperature)
    end
    mutate_value(
        rng::AbstractRNG,
        value::_TypeSpecValue,
        temperature,
        mutation::SymbolicRegression.ConstantMutation,
    ) = _mutate(rng, value, temperature, mutation)

    const _is_valid = if _config.is_valid !== nothing
        _include(_config.is_valid, "TypeSpec.is_valid")
    elseif _config.optimizable
        value -> all(isfinite, _parameters(value))
    else
        _ -> true
    end
    is_valid(value::_TypeSpecValue) = _is_valid(value)

    const _string = if _config.string === nothing
        function (value)
            fields = map(1:fieldcount(_value_type)) do i
                sprint(show, getfield(value, i); context=:compact => true)
            end
            return fieldcount(_value_type) == 1 ? only(fields) :
                string(_config.name, "(", join(fields, ", "), ")")
        end
    else
        _include(_config.string, "TypeSpec.string")
    end
    Base.show(io::IO, value::_TypeSpecValue) = print(io, _string(value))
    needs_brackets(::_TypeSpecValue) = false
    string_constant(
        value::_TypeSpecValue, ::Val{precision}, unit
    ) where {precision} = _string(value) * unit
    string_constant(
        value::_TypeSpecValue, bracketed, ::Val{precision}, unit
    ) where {precision} = _string(value) * unit
    const _operators = map(_config.operators) do (arity, sources)
        arity => Tuple(
            _include(source, "TypeSpec.operator[$arity][$index]")
            for (index, source) in enumerate(sources)
        )
    end
    const _loss_mode = _config.loss_mode
    const _loss_impl = _include(_config.loss, "TypeSpec.$(_loss_mode)")
    if _loss_mode === :elementwise_loss
        _loss(a::_TypeSpecValue, b::_TypeSpecValue) = _loss_impl(a, b)
    else
        _loss(expression, dataset, options) =
            _loss_impl(expression, dataset, options)
        const _loss_type = _include(_config.loss_type, "TypeSpec.loss_type")
    end
    if _config.template !== nothing
        const _template_expression_spec =
            _include(_config.template, "TypeSpec.template")
    end

    function _convert_value(x)
        x isa _value_type && return x
        x = PythonCall.Py(x)
        fieldcount(_value_type) == 1 ||
            PythonCall.pyhasattr(x, "__len__") &&
                PythonCall.pylen(x) == fieldcount(_value_type) ||
            throw(ArgumentError(
                "TypeSpec values require exactly $(fieldcount(_value_type)) fields."
            ))
        values = ntuple(fieldcount(_value_type)) do i
            source = fieldcount(_value_type) == 1 ? x : x[i - 1]
            PythonCall.pyconvert(fieldtype(_value_type, i), source)
        end
        return _value_type(values...)
    end
    _convert_array(values, dims) = reshape(
        _value_type[_convert_value(value) for value in values], Tuple(dims)
    )
    """)


def _optional_source(source: str | None) -> str:
    return "nothing" if source is None else _quoted(source)


def _count_parameters_config(count: int | str | None) -> str:
    if type(count) is int:
        return str(count)
    return _optional_source(count)


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
    for arity, sources in operators.items():
        if type(arity) is not int or arity < 1:
            raise ValueError("TypeSpec operator arities must be positive integers.")
        if not sources:
            raise ValueError(f"TypeSpec operator arity {arity} cannot be empty.")
        if any(not isinstance(source, str) or not source.strip() for source in sources):
            raise ValueError("Every TypeSpec operator must contain Julia source.")
        normalized.append((arity, list(sources)))
    return sorted(normalized)


def compile_type_spec(
    spec: TypeSpec,
    operators: dict[int, list[str]] | None,
    *,
    elementwise_loss: str | None,
    loss_function: str | None,
    loss_function_expression: str | None,
    template: str | None = None,
) -> _TypeSpecDefinition:
    """Create deterministic Julia source without evaluating user code."""
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

    field_sources = ",\n".join(
        f"    {_quoted(name)} => {_quoted(field_type)}"
        for name, field_type in spec.fields.items()
    )
    operator_sources = ",\n".join(
        f"    {arity} => ({', '.join(map(_quoted, sources))},)"
        for arity, sources in normalized_operators
    )
    config = _block(f"""
        (
            name = {_quoted(spec.name)},
            fields = (
            {field_sources},
            ),
            sample = {_quoted(spec.sample)},
            parameters = {_optional_source(spec.parameters)},
            with_parameters = {_optional_source(spec.with_parameters)},
            init = {_optional_source(spec.init)},
            mutate = {_optional_source(spec.mutate)},
            is_valid = {_optional_source(spec.is_valid)},
            count_parameters = {_count_parameters_config(spec.count_parameters)},
            pack_parameters = {_optional_source(spec.pack_parameters)},
            unpack_parameters = {_optional_source(spec.unpack_parameters)},
            string = {_optional_source(spec.string)},
            preamble = {_optional_source(spec.preamble)},
            optimizable = {str(spec.can_optimize).lower()},
            operators = (
            {operator_sources},
            ),
            loss_mode = :{loss_mode},
            loss = {_quoted(loss_source)},
            loss_type = {_optional_source(spec.loss_type)},
            template = {_optional_source(template)},
        )
        """)
    body = _TYPE_SPEC_MODULE.replace("__TYPE_SPEC_CONFIG__", config, 1)
    fingerprint = hashlib.sha256(f"{_CODEGEN_VERSION}\0{body}".encode()).hexdigest()
    source = _block(f"""
        module _PySRTypeSpec_{fingerprint[:20]}
        using Random
        using SymbolicRegression
        using PythonCall

        {body}
        end
        """) + "\n"
    return _TypeSpecDefinition(
        spec=copy.deepcopy(spec),
        fingerprint=fingerprint,
        source=source,
    )


def load_type_spec_runtime(
    definition: _TypeSpecDefinition, *, validate: bool = True
) -> _TypeSpecRuntime:
    """Load a generated module and return its ephemeral Julia objects."""
    module_symbol = jl.Symbol(definition.module_name)
    if not bool(jl.isdefined(jl.Main, module_symbol)):
        jl.Base.include_string(
            jl.Main,
            definition.source,
            "PySR." + definition.module_name,
        )
    module = jl.getproperty(jl.Main, module_symbol)
    value_type = jl.getproperty(module, jl.Symbol(definition.spec.name))

    operators = {}
    operator_names = {}
    for group in module._operators:
        arity, functions = group
        arity = int(arity)
        functions = tuple(functions)
        if any(not bool(jl.isa(function, jl.Function)) for function in functions):
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
    loss_mode = str(module._loss_mode)
    if loss_mode == "elementwise_loss":
        elementwise_loss = module._loss
        loss_type = jl.Base.promote_op(elementwise_loss, value_type, value_type)
    else:
        if loss_mode == "loss_function":
            loss_function = module._loss
        else:
            loss_function_expression = module._loss
        loss_type = module._loss_type
    if not bool(jl.seval("T -> isconcretetype(T) && T <: AbstractFloat")(loss_type)):
        raise ValueError(
            "The TypeSpec loss must return a concrete subtype of `AbstractFloat`; "
            f"got `{loss_type}`. Add a concrete Julia return type annotation."
        )
    expression_spec = (
        module._template_expression_spec
        if bool(jl.isdefined(module, jl.Symbol("_template_expression_spec")))
        else None
    )

    runtime = _TypeSpecRuntime(
        definition=definition,
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


@cache
def _type_spec_validator() -> AnyValue:
    return jl.seval(r"""
        function (module_, T, type_name, optimizable)
            DE = SymbolicRegression.InterfaceDynamicExpressionsModule.DE
            fail(hook, message) = throw(ArgumentError("TypeSpec `$hook` $message"))
            function call(hook, f, args...)
                try
                    return f(args...)
                catch error
                    fail(hook, "failed: $(sprint(showerror, error))")
                end
            end
            function check_value(hook, value)
                value isa T || fail(hook, "must return `$type_name`.")
                valid = call("is_valid", DE.is_valid, value)
                valid isa Bool || fail("is_valid", "must return `Bool`.")
                valid || fail(hook, "returned an invalid value.")
                return value
            end
            function parameter_count(value)
                count = call("count_parameters", DE.count_scalar_constants, value)
                count isa Int && count >= 0 ||
                    fail("count_parameters", "must return a nonnegative `Int`.")
                return count
            end
            function check_optimization(value, count)
                parameters = call("parameters", module_._parameters, value)
                parameters isa AbstractVector ||
                    fail("parameters", "must return an `AbstractVector`.")
                length(parameters) == count ||
                    fail("count_parameters", "disagrees with `parameters`.")
                number_type = DE.get_number_type(T)
                isconcretetype(number_type) && number_type <: AbstractFloat ||
                    fail("parameters", "must return a vector with a concrete `AbstractFloat` element type.")

                packed = Vector{number_type}(undef, count)
                next_idx = call("pack_parameters", DE.pack_scalar_constants!, packed, 1, value)
                next_idx == count + 1 ||
                    fail("pack_parameters", "returned the wrong next index.")
                isequal(packed, parameters) ||
                    fail("pack_parameters", "disagrees with `parameters`.")

                rebuilt = call("with_parameters", module_._with_parameters, value, parameters)
                rebuilt isa T || fail("with_parameters", "must return `$type_name`.")
                isequal(rebuilt, value) ||
                    fail("with_parameters(value, parameters(value))", "must preserve the value.")

                result = call("unpack_parameters", DE.unpack_scalar_constants, packed, 1, value)
                result isa Tuple && length(result) == 2 ||
                    fail("unpack_parameters", "must return `(next_idx, $type_name)`.")
                unpacked_idx, unpacked = result
                unpacked_idx == count + 1 ||
                    fail("unpack_parameters", "returned the wrong next index.")
                unpacked isa T || fail("unpack_parameters", "must return `$type_name`.")

                repacked = similar(packed)
                repacked_idx = call("pack_parameters", DE.pack_scalar_constants!, repacked, 1, unpacked)
                repacked_idx == count + 1 && isequal(packed, repacked) ||
                    fail("optimization hooks", "must preserve the packed scalar representation.")
            end

            rng = module_.Random.Xoshiro(0)
            sampled = check_value("sample", call("sample", SymbolicRegression.sample_value, rng, T, nothing))
            initial = check_value("init", call("init", SymbolicRegression.init_value, T))
            for value in (initial, sampled)
                count = parameter_count(value)
                optimizable && check_optimization(value, count)
            end

            mutated = check_value(
                "mutate",
                call("mutate", SymbolicRegression.mutate_value, rng, sampled, 1.0, SymbolicRegression.ConstantMutation()),
            )
            count = parameter_count(mutated)
            optimizable && check_optimization(mutated, count)

            call("string", module_._string, sampled) isa AbstractString ||
                fail("string", "must return an `AbstractString`.")
            for (arity, functions) in module_._operators, f in functions
                Base.promote_op(f, ntuple(_ -> T, arity)...) === T ||
                    fail("operator `$(nameof(f))`", "must infer `$type_name` as its return type.")
                result = call(string(nameof(f)), f, ntuple(_ -> sampled, arity)...)
                result isa T || fail("operator `$(nameof(f))`", "must return `$type_name`.")
            end
            module_._loss_mode === :elementwise_loss &&
                call("elementwise_loss", module_._loss, sampled, sampled)
            return nothing
        end
        """)


def _validate_type_spec_runtime(runtime: _TypeSpecRuntime) -> None:
    try:
        _type_spec_validator()(
            runtime.module,
            runtime.value_type,
            runtime.spec.name,
            runtime.spec.can_optimize,
        )
    except JuliaError as error:
        raise ValueError(str(jl.sprint(jl.showerror, error.args[0]))) from error


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

    try:
        return runtime.module._convert_array(array.ravel(order="F"), array.shape)
    except JuliaError as error:
        raise ValueError(str(jl.sprint(jl.showerror, error.args[0]))) from error


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
    definition: _TypeSpecDefinition,
    addprocs_function: AnyValue | None,
    worker_imports: AnyValue | None,
) -> AnyValue:
    """Create workers and install the private module before function transfer."""
    jl.seval("using Distributed: Distributed")
    if addprocs_function is None:
        addprocs_function = jl.Distributed.addprocs
    module_expression = jl.Meta.parse(definition.source)
    imports = worker_imports if worker_imports is not None else jl.nothing
    return jl.seval("""
        function (addprocs_function, module_expression, imports)
            return function (numprocs; kws...)
                procs = addprocs_function(numprocs; kws...)
                try
                    SymbolicRegression.import_module_on_workers(
                        procs, pathof(SymbolicRegression), imports, 0
                    )
                    Distributed.remotecall_eval(Main, procs, module_expression)
                catch
                    Distributed.rmprocs(procs)
                    rethrow()
                end
                return procs
            end
        end
        """)(addprocs_function, module_expression, imports)
