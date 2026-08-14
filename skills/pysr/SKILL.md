---
name: pysr
description: Use when fitting equations or symbolic programs with PySR or SymbolicRegression.jl, including scalar, vector, tensor, string, or custom Julia value data, or when a symbolic-regression search is slow, stuck, or producing poor equations.
---

# Using PySR Effectively

PySR evolves expression trees and returns a Pareto front trading accuracy against complexity. PySR 2.x searches over ordinary scalars or custom Julia value types, so the result can be an equation, a structured symbolic model, or a small program over vectors, tensors, strings, or records.

This guide is distilled from the PySR documentation and several hundred real user threads, checked against PySR 2.x and its SymbolicRegression.jl 2.x backend. A PySR 1.x migration table is at the end. Full docs: https://ai.damtp.cam.ac.uk/pysr/

## Quick start

```python
# pip install pysr  (Julia is downloaded automatically on first import; no separate install)
import numpy as np
from pysr import PySRRegressor

X = 2 * np.random.randn(100, 5)
y = 2 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2

model = PySRRegressor(
    operators={1: ["cos"], 2: ["+", "-", "*", "/"]},
    niterations=100,
)
model.fit(X, y)
print(model)            # Pareto front: complexity, loss, equation
model.predict(X)        # uses the auto-selected "best" equation
model.sympy()           # SymPy expression
model.latex()           # LaTeX string
```

`model.equations_` is a pandas DataFrame with `complexity`, `loss`, `score`, `equation`, `sympy_format`, `lambda_format`. Pass a row index to `predict`, `sympy`, `latex`, `jax`, `pytorch` to use any equation on the front, not just the selected one.

## Critical performance fact: keep the process alive

Julia is JIT-compiled. The first `.fit()` in a Python process pays Julia startup plus compilation (roughly a minute or two); every later fit in the *same process* starts almost instantly. The single most common agent mistake is running each experiment as a fresh `python script.py`, paying the full compile cost every time.

Instead:

- Keep one long-lived Python session (a background IPython/Jupyter kernel, a REPL tool, or any mechanism you already have) and run successive experiments in it. Creating a new `PySRRegressor` per experiment is cheap; the expensive state is the Julia runtime, which lives per-process.
- If you must use scripts, structure iteration so parameter tweaks happen inside one process rather than via repeated relaunches.
- Reuse an existing Python environment that already has `pysr` installed rather than creating a fresh environment per task (a new environment re-resolves and precompiles the Julia packages).
- Do not misdiagnose the first-run compilation pause as a hang. "Compiling Julia backend..." taking a minute or two is normal.

Running plain `python` scripts works fine; this is an optimization, not a requirement.

## Recommended workflow

1. **Subsample when representative rows are abundant.** Symbolic regression rarely needs more than a few thousand rows; ~1,000 to 5,000 representative rows often suffice even when millions are available. More rows help with many features, heavy noise, rare regimes, or a custom objective that depends on population-level statistics.
2. **Choose the minimal operator set.** Only include operators plausible for the domain. Redundant operators (for example `pow` alongside `square` and `cube`, or `-` alongside `neg`) enlarge the search space. If the target is a polynomial, use `operators={2: ["+", "-", "*"]}` and skip `/` and `^`.
3. **Seed strong prior forms with `guesses`.** Ordinary search accepts a list such as `guesses=["x0 * 3.0 + x2"]`. Multi-output search accepts one list per output. A guess is an initial candidate, not a constraint; evolution can alter or discard it. Template guesses use component dictionaries and are covered below.
4. **Start from current defaults otherwise.** Population settings, parsimony, mutation weights, `ncycles_per_iteration`, and related parameters were tuned by large-scale search. PySR 2.x enables annealing and automatic batching by default. Custom mutations and plugins are expert extension points for changing search moves, not routine hyperparameters; leave `mutations`, `default_mutations`, `plugins`, `default_plugins`, and `crossover_probability` at their defaults unless a measured search pathology calls for a custom mechanism. Old forum and paper recipes often predate current tuning.
5. **Use short calibration runs.** Run for a few minutes to check operators, loss, data layout, constraints, and whether sensible equations appear. Then start one long search.
6. **Budget the final search explicitly.** Evolution does not converge in the usual sense; a stalled search can jump to a new expression family hours later. Set `niterations` very large and use `timeout_in_seconds` for wall clock, `max_evals` for compute-matched comparisons, or `early_stop_condition` such as `"stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"`. `niterations * populations * population_size` is not an evaluation count. Results stream to `outputs/<run_id>/hall_of_fame.csv`. In IPython, `q` then Enter stops gracefully.
7. **Inspect the whole Pareto front.** The auto-selected row is one heuristic choice.

`batching="auto"` is the default. When explaining it, state all three material facts: batch-capable evolution switches to batches above 1,000 samples; hall-of-fame comparisons still use the full dataset; and, with `batch_size=None`, the backend uses all samples through 1,000, then batches of 128 below 5,000, 256 below 50,000, and 512 from 50,000 upward. Set an explicit `batch_size` to override this policy, `batching=True` to force batching, or `batching=False` when a full objective must see all samples together.

`model_selection="best"` picks the equation with highest `score` among those with loss within 1.5x of the best loss; `"accuracy"` picks the lowest-loss row. The `score` column is the negative log-loss slope per unit complexity: a large score marks the "kink" where accuracy jumps for little added complexity. These are heuristics for `predict`'s default; report and examine the full table.

## Choosing an equation

Print `model.equations_` and look at where loss drops sharply as complexity increases; that kink is usually the interesting equation. Evaluate the top few candidates on held-out data (`model.predict(X_test, index=i)`) when overfitting is plausible. Check limiting behavior (x -> 0, x -> inf) against domain expectations. Prefer presenting 2-3 candidates with the tradeoff to silently picking one.

## Losses and weights

Default is MSE. Common expert moves:

- Robust to outliers: `elementwise_loss="L1DistLoss()"` (median-seeking rather than mean-seeking).
- Known per-point uncertainty: `model.fit(X, y, weights=1/sigma**2)`; the built-in losses apply weights automatically. Custom weighted form: `elementwise_loss="myloss(x, y, w) = w * abs(x - y)^2"`.
- Target spans many orders of magnitude: MSE is dominated by the largest values. Use a log-space loss, e.g.

```python
elementwise_loss = """function loss_fnc(prediction, target)
    scatter_loss = abs(log((abs(prediction)+1f-20) / (abs(target)+1f-20)))
    sign_loss = 10f0 * (sign(prediction) - sign(target))^2
    return scatter_loss + sign_loss
end"""
```

- Percentage/relative error: divide by the *target*, never the prediction (dividing by the prediction lets evolution win by sending predictions to infinity), and beware targets near zero.
- Any loss from LossFunctions.jl works as a string: `"HuberLoss(1.0)"`, `"LPDistLoss{3}()"`, etc.
- Binary classification: encode targets as +1/-1 and use a margin loss such as `"L2MarginLoss()"`. Treat the result as a decision score. For probabilities, calibrate that score on held-out data with Platt scaling or isotonic regression, or use a probabilistic loss with a documented inverse link.
- Known asymptotic or boundary behavior (y -> 0 as x -> inf, exact value at a boundary, a known limit): add a few synthetic data points along the asymptote or at the boundary with very large `weights`. This is the standard trick and usually beats a custom objective. For strict enforcement, numerically estimate the limit inside a `loss_function` and add a graded penalty.
- Losses must be deterministic (results are cached) and non-negative unless `loss_scale="linear"`, which permits negative losses (e.g. log-likelihoods).

`elementwise_loss` receives scalars `(prediction, target)` or `(prediction, target, weight)`. Never sum or broadcast inside it. Objectives needing the whole prediction vector or the expression tree use `loss_function` instead (see below).

## Scaling and feature count

- Normalization is optional. Constants are sampled near N(0,1) and mutated multiplicatively, so wildly scaled features can slow the search, but normalizing inserts nuisance constants into the final equation and hides physical meaning. Prefer natural units; rescale only if the search visibly struggles.
- Up to roughly 10 features: no special handling.
- Tens of features: raise `maxsize`, provide more rows, and let the search select features itself; it is reasonably good at this. Automatic batching handles large supported datasets by default. An equation forced to contain all of 30+ features would need `maxsize` well above 100.
- More than ~50 features: the primary fix is smarter features or structure. Engineer aggregate features from domain knowledge, or use a template expression over a sensible decomposition. For structured data (fields, images, sequences, graphs), a naive one-column-per-pixel tabular encoding is usually the wrong move; build physically meaningful features, or train a neural network with the right inductive bias and symbolically distill its components (see arXiv:2006.11287). If there is no smarter representation available, `select_k_features=k` (gradient-boosting pre-selection) is the fallback.
- If the search omits a variable the user expected: PySR only optimizes accuracy and simplicity, so omission means the variable did not reduce loss enough to pay for its complexity. Forcing inclusion requires a custom loss that penalizes its absence.

## Template expressions: use when structure is known

Plain search is the default. Use `TemplateExpressionSpec` when a free-form tree would have to rediscover known structure: a fixed outer formula, shared learned components, category-specific coefficients, coupled outputs, derivatives, or hard rules about which variables feed which component.

```python
from pysr import PySRRegressor, TemplateExpressionSpec

spec = TemplateExpressionSpec(
    combine="sin(f(x1, x2)) + g(x3)^2",
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
)
model = PySRRegressor(
    expression_spec=spec,
    operators={1: ["sin"], 2: ["+", "-", "*", "/"]},
    guesses=[{"f": "#1 + #2", "g": "#1"}],
)
model.fit(X, y)
```

Each template guess is a dictionary from component name to an ordinary equation string. `#1`, `#2`, and so on mean the component's first, second, and later argument positions; they are not global feature names because one component may be called on different variables.

`TemplateExpressionSpec(parameters=...)` defines named vectors in the **outer template**, commonly one entry per category:

```python
spec = TemplateExpressionSpec(
    combine="scale[group] * f(x) + offset[group]",
    expressions=["f"],
    variable_names=["x", "group"],
    parameters={"scale": 3, "offset": 3},
)
```

Pass `group + 1` as a column of `X` when Python group labels start at zero because Julia indexing starts at one. These parameter vectors are separate from both estimator hyperparameters and the `TypeSpec.scalar_constants` hook described below.

The combine string is arbitrary Julia. It may contain statements, reuse a component (`fx = f(x); fx + fx^2`), call one component with different arguments (`f(x1) - f(x2)`), or differentiate it (`df = D(f, 1); df(x)`). Calls such as `f(x)` return `ValidVector` wrappers: `.x` holds the raw batch array and `.valid` records whether upstream evaluation succeeded. Supported arithmetic propagates validity automatically. Custom evaluation must unpack, operate, and repack:

```python
spec = TemplateExpressionSpec(
    combine="""
    fv = f(x1)
    gv = g(x2)
    raw = map((a, b) -> clamp(a, b, b + one(b)), fv.x, gv.x)
    ValidVector(raw, fv.valid && gv.valid)
    """,
    expressions=["f", "g"],
    variable_names=["x1", "x2"],
)
```

Returning a regular `Vector` is invalid. With a `TypeSpec` custom value, each element of `.x` is that generated value. Unwrap its field, rebuild each result as that value type, then wrap the batch:

```julia
fv = f(x1)
gv = g(x2)
raw = map(fv.x, gv.x) do a, b
    Vec2(a.data .+ 2 .* b.data)
end
ValidVector(raw, fv.valid && gv.valid)
```

This is the supported bridge between known template structure and custom values throughout the tree. `TypeSpec` scalar-constant hooks still optimize the contents of each custom constant. The narrow restriction is that outer `TemplateExpressionSpec(parameters=...)` vectors are unavailable while `TypeSpec` is active.

Other template rules:

- Multi-output or residual templates can place extra targets in `X`, return a per-row loss, and fit dummy `y` with `elementwise_loss="(prediction, target) -> prediction"`.
- `loss_function_expression`, rather than `loss_function`, receives the complete template expression and can inspect component trees through `get_tree`.
- Standard SymPy, LaTeX, JAX, and PyTorch export is unavailable because `combine` can be arbitrary Julia. Keep `checkpoint.pkl` and the exact spec-construction code; the CSV alone cannot reconstruct the callable template reliably.
- Type-stable literals still matter: use `0.5f0` in a Float32 template or convert explicitly.
- Dimensional constraints are unavailable with templates.
- Translate old positional `TemplateExpressionSpec(function_symbols, combine, num_features)` calls to explicit `combine=`, `expressions=`, and `variable_names=` keywords. Do not copy the positional form into new code.

## Custom operators

Use the arity-keyed `operators` dictionary. This supports unary, binary, and n-ary operators in one API:

```python
import sympy

model = PySRRegressor(
    operators={
        1: ["inv(x) = 1 / x", "gauss(x) = exp(-x^2)"],
        2: ["+", "*"],
        3: ["clamp3(x, lo, hi) = min(max(x, lo), hi)"],
    },
    extra_sympy_mappings={
        "inv": lambda x: 1 / x,
        "gauss": lambda x: sympy.exp(-x**2),
        "clamp3": lambda x, lo, hi: sympy.Min(sympy.Max(x, lo), hi),
    },
)
```

Rules that prevent most operator bugs:

- Operators must accept every value in the search domain. Scalar search probes far outside the training range. Return a typed NaN for invalid inputs: `my_sqrt(x) = x >= 0 ? sqrt(x) : convert(typeof(x), NaN)`. Candidates producing NaN are assigned infinite loss. Built-ins such as `sqrt`, `log`, and `acosh` are already protected.
- Preserve the input type. Decimal literals such as `2.5` are Float64 and may promote Float32 calculations; use `T(2.5)` with a `where {T}` signature, `2.5f0`, or `oftype(x, 2.5)`. Integer literals are normally safe for standard numeric types, but custom types may require `one(x)`, `zero(x)`, or explicit conversion.
- Provide `extra_sympy_mappings` when symbolic export or ordinary Python prediction needs a custom operator. Use SymPy functions, never NumPy or SciPy. JAX and PyTorch need their corresponding mappings.
- When no closed SymPy form exists, map the operator to a `sympy.Function` subclass as a symbolic placeholder. A placeholder preserves symbolic representation only; numeric Python prediction still needs a lambdify implementation for that function. Call `model.refresh()` after changing mappings on a fitted model.
- Julia packages can back an operator. Install them outside the search, import them before constructing the model, and list packages needed by multiprocessing workers in `worker_imports`.

## Custom objectives (`loss_function`)

Use a full Julia objective when scoring needs the whole prediction vector, the expression tree, derivatives, or auxiliary sample-aligned data. Under automatic batching, use the optional four-argument form so the backend supplies the full dataset plus active sample indices:

```python
objective = """
function my_objective(
    tree, dataset::Dataset{T,L}, options, idx=nothing
)::L where {T,L}
    sample_idx = isnothing(idx) ? axes(dataset.X, 2) : idx
    X = @view dataset.X[:, sample_idx]
    y = @view dataset.y[sample_idx]
    prediction, completed = eval_tree_array(tree, X, options)
    !completed && return L(Inf)
    return sum(i -> abs2(prediction[i] - y[i]), eachindex(y)) / size(X, 2)
end
"""
model = PySRRegressor(loss_function=objective, operators={2: ["+", "*", "-"]})
```

The three-argument form still works, but receives the current `Dataset` or `SubDataset`; it does not guarantee all samples during batched evolution. Hall-of-fame scoring uses the full dataset. Set `batching=False` when a statistic truly requires the complete population every call.

Hard-won rules:

- Always check `completed` before reading predictions; a failed evaluation leaves unusable output.
- `dataset.X` is features by samples, transposed relative to Python. In the four-argument full-dataset-plus-`idx` path, `dataset.n` is the full sample count; in a three-argument callback receiving a `SubDataset`, it is the current batch size. After slicing, use `size(X, 2)` as the active sample count; never `size(dataset.X, 1)`, which counts features. With `idx`, slice the sample axis of every sample-aligned auxiliary array in the identical order. Append auxiliary values as rows of `dataset.X` or interpolate fixed arrays into the Julia source; appended rows are searchable features unless a template or structural penalty prevents their use.
- Return the declared loss type `L`, including `L(Inf)` for failed numerical evaluation. Structural preferences should use large, finite, typed penalties such as `L(1e6 * n_violations)`, so intermediate mutations retain evolutionary guidance.
- This path runs millions of times: keep it type-stable, allocation-conscious, deterministic, and free of printing, Python callbacks, and untyped globals. Do not call it `eval_loss`, which collides with an internal name.
- If a custom objective is unexpectedly slow, reproduce its hot call in Julia and inspect it with `@code_warntype`; benchmark representative trees with BenchmarkTools before spending a long search budget.
- `eval_diff_tree_array` evaluates a derivative with respect to one feature; `eval_grad_tree_array` evaluates all feature derivatives. Templates normally use `D(f, i)` inside `combine`.
- A custom objective that interprets the raw tree differently from its printed form must decode and export that interpretation itself. Symbolic exporters and ordinary `predict` only understand the configured expression specification.

### Symbolic constraints by walking the tree

Rules that `constraints` and `nested_constraints` cannot express, such as required features or operator-specific constant ranges, belong in the objective. `any`, `all`, `count`, `sum`, and `foreach` traverse trees without the explicit stack allocated by `for node in tree`. Leaves expose `constant`, `val`, and one-based `feature`; branches expose `degree` and a one-based `op` index within that arity's configured operator list. Access n-ary children through `SymbolicRegression.InterfaceDynamicExpressionsModule.DE.get_child(node, i)`. DynamicExpressions traversal and construction APIs: https://ai.damtp.cam.ac.uk/dynamicexpressions/stable/examples/base_operations/

Example: penalize `^` unless its exponent is a lone constant in `[0, 1]`:

```python
objective = """
function constrained_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    idx_pow = 3  # position of "^" in operators[2]
    n_bad = count(tree) do node
        if node.degree != 2 || node.op != idx_pow
            false
        else
            exponent = SymbolicRegression.InterfaceDynamicExpressionsModule.DE.get_child(node, 2)
            !(exponent.degree == 0 && exponent.constant &&
              zero(T) <= exponent.val <= one(T))
        end
    end
    n_bad > 0 && return L(10_000 * n_bad)
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    !completed && return L(Inf)
    return sum(i -> abs2(prediction[i] - dataset.y[i]), eachindex(prediction)) /
        dataset.n
end
"""
model = PySRRegressor(
    operators={2: ["+", "*", "^"]},
    loss_function=objective,
)
```

Perform the cheap structural check before numerical evaluation. Count violations and scale a finite penalty. Requiring feature 2 uses the mirror pattern: `any(n -> n.degree == 0 && !n.constant && n.feature == 2, tree) || return L(1e6)`. For templates, inspect a component's tree through `get_tree(ex)` inside `loss_function_expression`.

## Constraints and complexity shaping
- Constraint keys must exactly match configured operator names. With `operators={2: ["+", "*", "^"]}`, use `constraints={"^": (-1, 1)}`; `"pow"` would constrain only an operator actually named `pow`. The tuple sets maximum complexity per argument, so this leaves the base unlimited and restricts the exponent to complexity 1. Whenever `^` is enabled, constrain the exponent unless the domain truly requires unrestricted symbolic exponents; unconstrained exponentiation usually searches very poorly.
- `nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}}` forbids nested trig. Values are maximum counts of each inner operator within each outer operator.
- `complexity_of_operators={"exp": 3}` makes selected operators expensive. `complexity_of_constants=2` discourages free constants. `complexity_of_variables` sets global or per-feature costs.
- Constraints apply to intermediate candidates too. Tight settings can make the target unreachable. Leave slack, for example `maxsize=35` for a desired final size near 30.
- `maxsize` counts every operator, constant, and variable node. A seven-feature linear model is already around complexity 29.
- `warmup_maxsize_by=0.5` ramps the size limit over half the run when search becomes complex too early.
- `parsimony` and `adaptive_parsimony_scaling` are already tuned. Change them only after observing a concrete failure mode.
- For exact known constants such as pi, use a constant-valued input feature; otherwise fit a float and recognize it afterward. For integer-only constants, pass candidate integers as constant features, raise `complexity_of_constants` to discourage fitted constants, or use a typed rounding operator rather than expecting continuous optimization to land exactly on integers.

## Custom values throughout the tree (`TypeSpec`)

Use `TypeSpec` when every feature, constant, intermediate result, prediction, and target must share a non-scalar Julia value type. A custom loss alone changes scoring; a template alone fixes outer structure. Neither changes the algebra flowing through every node.

```python
from pysr import PySRRegressor, TypeSpec

vec2 = TypeSpec(
    name="Vec2",
    fields={"data": "Vector{Float64}"},
    sample="rng -> Vec2(randn(rng, 2))",
    scalar_constants="value -> value.data",
    with_scalar_constants="(value, p) -> Vec2(collect(p))",
)
model = PySRRegressor(
    type_spec=vec2,
    operators={
        1: ["rotate90(a::Vec2) = Vec2([-a.data[2], a.data[1]])"],
        2: ["add(a::Vec2, b::Vec2) = Vec2(a.data + b.data)"],
    },
    elementwise_loss="""
    vector_loss(a::Vec2, b::Vec2)::Float64 = sum(abs2, a.data - b.data)
    """,
)
```

The similarly named APIs serve different layers. `TemplateExpressionSpec(parameters={"scale": 3})` creates an outer-template vector addressable as `scale[group]`. `TypeSpec.scalar_constants` is a Julia hook that extracts the optimizable scalar degrees of freedom from **one custom constant**; `with_scalar_constants` rebuilds that constant after BFGS updates them. In a combined TypeSpec plus template search, these TypeSpec hooks still optimize custom constants, while template-level parameter vectors are currently unavailable.

Every operator must infer exactly the chosen type; add a return annotation when `Base.promote_op` cannot prove it. Put helper definitions and imports in `preamble` or use package-qualified names because each spec is compiled in a fingerprinted private Julia module. The module, operators, loss, and template definitions are replayed on multiprocessing workers and before checkpoint deserialization.

TypeSpec supports ordinary and template searches, prediction, checkpoint reload, and serial, multithreaded, or multiprocessing execution. It currently excludes guesses, weights, units, denoising, feature selection, resampling, multi-output targets, turbo/bumper evaluation, autodiff backends, template-level parameter vectors, other expression specifications, and SymPy/JAX/PyTorch/LaTeX export. A full `loss_function` or `loss_function_expression` also requires `TypeSpec(loss_type="Float64")`; elementwise loss infers its return type.

Keep `checkpoint.pkl`, the exact `TypeSpec` construction source, and package versions. The hall-of-fame CSV cannot reconstruct the private type, and TypeSpec checkpoints are version-locked.

## Dimensional constraints

Physical units, checked during search:

```python
model.fit(X, y, X_units=["Constants.M_sun", "kg", "m"], y_units="kg * m / s^2")
```

- Uses DynamicQuantities.jl notation; `"1"` means explicitly dimensionless.
- Violations are softly penalized via `dimensional_constraint_penalty` (default 1000). Do not crank it to 1e9; a graded penalty is what lets evolution route through slightly-wrong intermediates.
- Fitted constants get wildcard units (printed `[⋅]` or `[?]`), so a lone constant can absorb any units and make an expression valid. Set `dimensionless_constants_only=True` to forbid that.
- Radians count as dimensionless in SI, so units cannot force a variable to appear only inside trig.
- Template expressions and TypeSpec do not support dimensional constraints.

## Parallelism

- Default `parallelism="multithreading"` is right for laptops and single nodes. Thread count is set at Julia startup: set the env var `PYTHON_JULIACALL_THREADS=<n>` *before* importing pysr (`JULIA_NUM_THREADS` is not the right variable under juliacall).
- Keep `populations` at ~2-3x the number of threads/cores so workers always have work (default `populations=31` already covers typical machines).
- If the coordinating thread is saturated on many-core machines, raise `ncycles_per_iteration`; workers then communicate less often.
- `parallelism="multiprocessing"` (with `procs=n`) has much higher startup cost per fit but can run faster steady-state and spans multiple nodes with `cluster_manager="slurm"` (or run SymbolicRegression.jl natively with SlurmClusterManager.jl, see https://ai.damtp.cam.ac.uk/symbolicregression/dev/slurm/). Only worth it for very long runs. Launch the script once, on one node, and let it spawn workers; do not wrap it in `srun`. Custom Julia packages needed on workers go in `worker_imports`.
- Full reproducibility requires `deterministic=True, random_state=<seed>, parallelism="serial"`; parallel seeded runs are not deterministic, and even serial results can differ slightly across CPUs (use `precision=64` to reduce this).

## Saving, resuming, exporting

- Every fit writes `outputs/<run_id>/hall_of_fame.csv` (updated continuously; safe to read mid-run) and `checkpoint.pkl`. Reload with `PySRRegressor.from_file(run_directory=...)`. Pickles are version-locked: load with the same PySR version that wrote them. The CSV plus your construction code is the durable artifact; custom operators need their `extra_sympy_mappings` supplied again on reload.
- `warm_start=True` continues evolution from the previous call's populations on the next `.fit()` in the same process. Search-space parameters (operators, `maxsize`, `expression_spec`, precision, feature count/order) must stay fixed; you can change the loss or weights between warm-started fits to implement staged objectives.
- Exports: `model.sympy(i)`, `model.latex(i)`, `model.latex_table()`, `model.jax(i)` (returns `{'callable', 'parameters'}`, differentiable), `model.pytorch(i)` (trainable module). Custom operators need `extra_jax_mappings`/`extra_torch_mappings` for those backends. A common pattern: pick an equation, export to JAX/PyTorch, and fine-tune its constants by gradient descent on the full dataset.
- Loss printed by Julia can differ from a Python recomputation (32-bit default vs numpy 64-bit); `precision=64` if it matters. Data with values beyond ~1e19 or below ~1e-19 also needs `precision=64` (Float32 overflow).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| First `import pysr` or first `.fit` takes minutes | Normal: Julia download (first install) and JIT compile (each process). Keep the process alive. |
| "Compiling Julia backend" apparently forever | On Apple Silicon: x86 Python under Rosetta; install a native arm64 Python env. In notebooks/embedded shells: stdin monitoring can wedge; pass `input_stream="devnull"`. |
| Hang at startup with a `lock.pid` mentioned | Stale juliapkg lock from a killed process: verify nothing is installing, delete the lock file, `import pysr` once in a fresh process. Shared filesystems: pre-initialize once before launching many jobs. |
| `UnicodeDecodeError` spam in Jupyter | Old PythonCall bug; set `PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION=no` before import, or upgrade. |
| Cannot interrupt search in Jupyter | Known limitation; `q`+Enter works in IPython/terminal, not notebooks. Use `timeout_in_seconds` or run from IPython. |
| Search finds nothing sensible | Loss mismatched to data scale (try log-space loss), operators missing or redundant, `maxsize` too small for the true equation, or constraints exclude it. Check in that order before adding compute. |
| All equations are tiny/trivial | `maxsize` too small, or huge dynamic range with MSE (largest points dominate). |
| `DomainError` from an operator | Custom operator not defined on all reals; add the typed-NaN guard. |
| Fit OK but `.predict`/`.sympy` fails | Missing/wrong `extra_sympy_mappings` (must be sympy functions, not numpy). |
| Equation uses a "forbidden" value like division by ~0 | Any-NaN-anywhere invalidates a candidate, so surviving equations are finite on *your data*; they can still blow up elsewhere in the domain. |
| `ProcessExitedException` wall of text on early stop | Harmless worker teardown noise under multiprocessing. |
| Memory grows across a long run | Mostly fixed in recent Julia; if hit, set `heap_size_hint_in_bytes` (multiprocessing) or upgrade Julia. |
| Results differ run to run | Expected; evolution is stochastic. See determinism recipe under Parallelism. |

## Dropping to Julia

Everything above also exists natively in SymbolicRegression.jl (`SRRegressor` via MLJ), which is preferable when the whole pipeline is Julia or you need deep customization (custom expression types, mutation operators, per-component constraints). From Python, `from pysr import jl` gives the live Julia runtime: `jl.seval(...)` runs arbitrary code, and installed Julia packages can back custom operators and losses. The backend source is readable and small; `src/Options.jl` and `src/CheckConstraints.jl` are the usual extension points, and a dev checkout can be wired in via `pysr/juliapkg.json`.

## PySR 1.x migration

This guide targets PySR 2.x. Old examples may use:

| Old | Preferred PySR 2.x |
|---|---|
| `binary_operators=[...]`, `unary_operators=[...]` | `operators={1: [...], 2: [...], 3: [...]}` |
| `full_objective=` | `loss_function=`; use `loss_function_expression=` for a template expression |
| `equation_file=` | `output_directory=` plus `run_id=`; results live under `<output_directory>/<run_id>/` |
| `batching=True` | usually omit it because `batching="auto"` is the default; retain `True` only to force batching |
| `annealing=False` | retain only when deliberate; annealing is enabled by default |
| `npop`, `ncyclesperiteration`, camelCase names | `population_size`, `ncycles_per_iteration`, snake_case names |
| `pysr(...)`, `best()`, `get_hof()` | `PySRRegressor`, `.fit()`, `get_best()`, `model.equations_` |
| `pysr.install()`, PyCall, `from julia import Main` | no install step; JuliaCall initializes automatically on import |

Migrate old constructor arguments rather than propagating them into new code.
