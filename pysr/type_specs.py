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

_CODEGEN_VERSION = 3


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


@dataclass
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
    string: str | None = None
    preamble: str | None = None
    loss_type: str | None = None

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

        for name in (
            "parameters",
            "with_parameters",
            "init",
            "mutate",
            "is_valid",
            "string",
            "preamble",
            "loss_type",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` cannot be empty.")

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

    @property
    def spec(self) -> TypeSpec:
        return self.definition.spec


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

    count_scalar_constants(value::_TypeSpecValue) =
        _config.optimizable ? length(_parameters(value)) : 0

    if _config.optimizable
        function pack_scalar_constants!(
            buffer::AbstractVector{<:Number}, idx::Int, value::_TypeSpecValue
        )
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
        function unpack_scalar_constants(
            buffer::AbstractVector{<:Number}, idx::Int, value::_TypeSpecValue
        )
            count = length(_parameters(value))
            parameters = @view buffer[idx:(idx + count - 1)]
            return idx + count, _with_parameters(value, parameters)
        end
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


def validate_type_spec_loss_configuration(
    spec: TypeSpec,
    elementwise_loss: str | None,
    loss_function: str | None,
    loss_function_expression: str | None,
) -> str:
    configured = [
        ("elementwise_loss", elementwise_loss),
        ("loss_function", loss_function),
        ("loss_function_expression", loss_function_expression),
    ]
    selected = [mode for mode, source in configured if source is not None]
    if len(selected) != 1:
        raise ValueError(
            "TypeSpec requires exactly one of `elementwise_loss`, `loss_function`, "
            "and `loss_function_expression`."
        )
    mode = selected[0]
    if mode == "elementwise_loss" and spec.loss_type is not None:
        raise ValueError(
            "Do not set `loss_type` with `elementwise_loss`; its return type is inferred."
        )
    if mode != "elementwise_loss" and spec.loss_type is None:
        raise ValueError("TypeSpec full objectives require an explicit `loss_type`.")
    return mode


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


def validate_type_spec_configuration(
    spec: TypeSpec,
    operators: dict[int, list[str]] | None,
    *,
    elementwise_loss: str | None,
    loss_function: str | None,
    loss_function_expression: str | None,
) -> None:
    _normalize_operators(operators)
    validate_type_spec_loss_configuration(
        spec, elementwise_loss, loss_function, loss_function_expression
    )


def compile_type_spec(spec: TypeSpec) -> _TypeSpecDefinition:
    """Create deterministic Julia source without evaluating user code."""
    field_sources = ",\n".join(
        f"    {_quoted(name)} => {_quoted(field_type)}"
        for name, field_type in spec.fields.items()
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
            string = {_optional_source(spec.string)},
            preamble = {_optional_source(spec.preamble)},
            optimizable = {str(spec.can_optimize).lower()},
        )
        """)
    body = _TYPE_SPEC_MODULE.replace("__TYPE_SPEC_CONFIG__", config, 1)
    fingerprint = hashlib.sha256(f"{_CODEGEN_VERSION}\0{body}".encode()).hexdigest()
    module_name = f"_PySRTypeSpec_{fingerprint[:20]}"
    source = _block(f"""
        module {module_name}
        using Random
        using SymbolicRegression
        using PythonCall

        {body}
        end
        import .{module_name}: {spec.name}
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

    runtime = _TypeSpecRuntime(
        definition=definition,
        module=module,
        value_type=value_type,
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

                offset = 3
                packed = fill(number_type(NaN), count + 4)
                next_idx = call(
                    "pack_parameters",
                    DE.pack_scalar_constants!,
                    packed,
                    offset,
                    value,
                )
                next_idx == offset + count ||
                    fail("pack_parameters", "returned the wrong next index.")
                all(isnan, packed[1:(offset - 1)]) &&
                    all(isnan, packed[(offset + count):end]) ||
                    fail("pack_parameters", "wrote outside its parameter range.")
                isequal(packed[offset:(offset + count - 1)], parameters) ||
                    fail("pack_parameters", "disagrees with `parameters`.")

                rebuilt = call("with_parameters", module_._with_parameters, value, parameters)
                rebuilt isa T || fail("with_parameters", "must return `$type_name`.")
                isequal(rebuilt, value) ||
                    fail("with_parameters(value, parameters(value))", "must preserve the value.")

                result = call(
                    "unpack_parameters",
                    DE.unpack_scalar_constants,
                    packed,
                    offset,
                    value,
                )
                result isa Tuple && length(result) == 2 ||
                    fail("unpack_parameters", "must return `(next_idx, $type_name)`.")
                unpacked_idx, unpacked = result
                unpacked_idx == offset + count ||
                    fail("unpack_parameters", "returned the wrong next index.")
                unpacked isa T || fail("unpack_parameters", "must return `$type_name`.")

                repacked = fill(number_type(NaN), count + 4)
                repacked_idx = call(
                    "pack_parameters",
                    DE.pack_scalar_constants!,
                    repacked,
                    offset,
                    unpacked,
                )
                repacked_idx == offset + count && isequal(packed, repacked) ||
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


@cache
def _type_spec_operator_validator() -> AnyValue:
    return jl.seval(r"""
        function (module_, T, type_name, operator, arity)
            inferred = Base.promote_op(operator, ntuple(_ -> T, arity)...)
            inferred === T || throw(ArgumentError(
                "TypeSpec operator `$(nameof(operator))` must be type-stable and " *
                "infer `$type_name` as its return type; inferred `$inferred`. " *
                "Add an explicit `::$type_name` return annotation if needed."
            ))
            sampled = module_._init()
            result = operator(ntuple(_ -> sampled, arity)...)
            result isa T || throw(ArgumentError(
                "TypeSpec operator `$(nameof(operator))` must return `$type_name`."
            ))
            return nothing
        end
        """)


@cache
def _type_spec_loss_validator() -> AnyValue:
    return jl.seval(r"""
        function (module_, T, elementwise_loss, configured_loss_type)
            loss_type = if elementwise_loss === nothing
                configured_loss_type
            else
                inferred = Base.promote_op(elementwise_loss, T, T)
                sample = module_._init()
                elementwise_loss(sample, sample)
                inferred
            end
            isconcretetype(loss_type) && loss_type <: AbstractFloat ||
                throw(ArgumentError(
                    "The TypeSpec loss must return a concrete subtype of " *
                    "`AbstractFloat`; got `$loss_type`. Add a concrete Julia " *
                    "return type annotation."
                ))
            return loss_type
        end
        """)


def validate_type_spec_options(
    runtime: _TypeSpecRuntime,
    operators: dict[int, tuple[AnyValue, ...]],
    elementwise_loss: AnyValue | None,
) -> AnyValue:
    """Validate ordinary Julia options against a loaded TypeSpec."""
    try:
        for arity, functions in operators.items():
            for function in functions:
                _type_spec_operator_validator()(
                    runtime.module,
                    runtime.value_type,
                    runtime.spec.name,
                    function,
                    arity,
                )
        configured_loss_type = (
            None if runtime.spec.loss_type is None else jl.seval(runtime.spec.loss_type)
        )
        return _type_spec_loss_validator()(
            runtime.module,
            runtime.value_type,
            elementwise_loss,
            configured_loss_type,
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
    operator_sources: AnyValue | None = None,
    objective_sources: AnyValue | None = None,
    template_source: str | None = None,
    template_function_name: AnyValue | None = None,
) -> AnyValue:
    """Create workers and install the private module before function transfer."""
    jl.seval("using Distributed: Distributed")
    if addprocs_function is None:
        addprocs_function = jl.Distributed.addprocs
    module_expression = jl.Meta.parseall(definition.source)
    module_symbol = jl.Symbol(definition.module_name)
    operators = (
        operator_sources if operator_sources is not None else jl.seval("String[]")
    )
    objectives = (
        objective_sources if objective_sources is not None else jl.seval("String[]")
    )
    template = template_source if template_source is not None else jl.nothing
    template_name = (
        template_function_name if template_function_name is not None else jl.nothing
    )
    imports = worker_imports if worker_imports is not None else jl.nothing
    return jl.seval("""
        function (
            addprocs_function,
            module_expression,
            module_symbol,
            operator_sources,
            objective_sources,
            template_source,
            template_function_name,
            imports,
        )
            return function (numprocs; kws...)
                procs = addprocs_function(numprocs; kws...)
                try
                    SymbolicRegression.import_module_on_workers(
                        procs, pathof(SymbolicRegression), imports, 0
                    )
                    Distributed.remotecall_eval(Main, procs, module_expression)
                    for (index, source) in pairs(operator_sources)
                        expression = quote
                            module_ = getproperty(
                                Main, $(QuoteNode(module_symbol))
                            )
                            Base.include_string(
                                module_,
                                $source,
                                $("PySR TypeSpec operator $index"),
                            )
                        end
                        Distributed.remotecall_eval(Main, procs, expression)
                    end
                    for (index, source) in pairs(objective_sources)
                        expression = quote
                            module_ = getproperty(
                                Main, $(QuoteNode(module_symbol))
                            )
                            Base.include_string(
                                module_,
                                $source,
                                $("PySR TypeSpec objective $index"),
                            )
                        end
                        Distributed.remotecall_eval(Main, procs, expression)
                    end
                    if template_source !== nothing
                        expression = quote
                            module_ = getproperty(
                                Main, $(QuoteNode(module_symbol))
                            )
                            target_function_name = $(
                                QuoteNode(template_function_name)
                            )
                            worker_spec = Base.include_string(
                                module_,
                                $template_source,
                                "PySR TypeSpec template",
                            )
                            worker_function_name = nameof(
                                worker_spec.structure.combine
                            )
                            if worker_function_name != target_function_name
                                Base.include_string(
                                    module_,
                                    "$(target_function_name)(args...) = " *
                                    "$(worker_function_name)(args...)",
                                    "PySR TypeSpec template alias",
                                )
                            end
                        end
                        Distributed.remotecall_eval(Main, procs, expression)
                    end
                catch
                    Distributed.rmprocs(procs)
                    rethrow()
                end
                return procs
            end
        end
        """)(
        addprocs_function,
        module_expression,
        module_symbol,
        operators,
        objectives,
        template,
        template_name,
        imports,
    )
