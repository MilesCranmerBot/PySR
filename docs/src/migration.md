# Migrating from v1 to v2

PySR v2 upgrades the Julia backend from SymbolicRegression.jl v1.11 to v2.
Most `PySRRegressor` constructor calls still work, but several search defaults
changed. The old parametric-expression interface and v1 checkpoints do not
carry over.

Before upgrading, keep a copy of each run directory, including
`hall_of_fame.csv`. Install v2 in a new Python environment if you need to keep a
v1 environment available for old pickles or exact search reproduction.

## Search defaults changed

The following changes affect ordinary `PySRRegressor(...)` searches:

| Setting | v1 default | v2 default |
| --- | --- | --- |
| `batching` | `False` | `"auto"` |
| `batch_size` | `50` | `None` |
| `annealing` | `False` | `True` |
| `crossover_probability` | `0.0259` | `0.2` |
| Mutation selection | Fixed v1 weights | New backend weights, adapted during the search |

With `batching="auto"`, the backend enables batching for large datasets while
still evaluating hall-of-fame candidates on the full dataset. With
`batch_size=None`, it uses the full dataset for at most 1000 rows, then batches
of 128, 256, or 512 as the dataset grows.

The default mutation mix changed, including a new feature mutation that can
replace the input referenced by a variable node. The
`AdaptiveMutationWeightsPlugin` also updates mutation weights from observed
success rates. A `weight_*` argument now supplies an initial weight, rather than
a fixed weight, while that plugin is active.

For a search close to the v1 defaults, set the changed values explicitly and
omit the adaptive-mutation plugin:

```python
from pysr import AdaptiveParsimonyPlugin, PySRRegressor

model = PySRRegressor(
    batching=False,
    batch_size=50,
    annealing=False,
    crossover_probability=0.0259,
    default_plugins=[AdaptiveParsimonyPlugin()],
    weight_add_node=2.47,
    weight_insert_node=0.0112,
    weight_delete_node=0.87,
    weight_do_nothing=0.273,
    weight_mutate_constant=0.0346,
    weight_mutate_operator=0.293,
    weight_mutate_feature=0.0,
    weight_swap_operands=0.198,
    weight_rotate_tree=4.26,
    weight_randomize=0.000502,
    weight_simplify=0.00209,
    weight_optimize=0.0,
    weight_backsolve=0.0,
)
```

This preserves the v1 search configuration, but it does not make a run
bit-for-bit identical. Backend implementation changes still alter random-number
consumption and search trajectories.

Two other backend changes can alter results:

- Simplification now recomputes an expression's cost from scratch. Equivalent
  expressions can therefore appear in a different hall-of-fame order.
- Constant optimization can restart zero-valued constants, allowing trajectories
  that v1 could not reach.

Loss functions are now validated when `fit` is called. An `elementwise_loss` or
`loss_function` with the wrong number of arguments raises a `ValueError` before
the search starts.

## Parametric expressions

`ParametricExpressionSpec` and the `category=` argument to `fit` were removed.
Use `TemplateExpressionSpec.parameters`, append the category to `X`, and index
parameters from the template.

A v1 model such as:

```python
from pysr import ParametricExpressionSpec, PySRRegressor

model = PySRRegressor(
    expression_spec=ParametricExpressionSpec(max_parameters=2),
)
model.fit(X, y, category=category)
```

becomes:

```python
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec

n_categories = len(np.unique(category))
spec = TemplateExpressionSpec(
    combine="f(x1, x2, p1[category], p2[category])",
    expressions=["f"],
    variable_names=["x1", "x2", "category"],
    parameters={"p1": n_categories, "p2": n_categories},
)

# Julia indexing starts at 1. Add one if Python categories start at 0.
X_with_category = np.column_stack([X, category + 1])
model = PySRRegressor(expression_spec=spec)
model.fit(X_with_category, y)
```

The category column must also be present when calling `predict`.

## Template expressions

The deprecated `function_symbols=...` and positional constructor forms of
`TemplateExpressionSpec` were replaced by explicit template fields. State the
combining expression, inner expressions, and variable names directly:

```python
spec = TemplateExpressionSpec(
    combine="sin(f(x1, x2)) + g(x3)^2",
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
)
```

This form also supports optimizable parameter vectors through
`parameters={"p": length}`. See [Template expressions](examples.md#template-expressions)
for complete examples.

## Saved models and checkpoints

v2 uses checkpoint schema 3. A v1 pickle has no compatible schema marker, so
`pickle.load(...)` and `PySRRegressor.from_file(...)` reject it with an
`Unsupported PySR checkpoint schema` error. Warm starts cannot cross the v1 to
v2 boundary. Re-run the search under v2 if you need to continue evolution.

For standard expressions, you can reconstruct the discovered equations from a
copy of `hall_of_fame.csv`. Put the CSV in a run directory without the
incompatible `checkpoint.pkl`, then provide the original operators and feature
metadata:

```python
model = PySRRegressor.from_file(
    run_directory="v1_csv_copy",
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos"],
    n_features_in=3,
    feature_names_in=["x0", "x1", "x2"],
)
```

Custom expression specifications may require a fresh search because their Julia
objects cannot be reconstructed from the CSV alone.

## v1 spellings that still work

These forms emit a deprecation warning but retain their old behavior:

- camelCase keyword arguments such as `fractionReplaced`, `npop`,
  `weightMutateConstant`, and `crossoverProbability`; use snake_case instead.
- `loss=` and `full_objective=`; use `elementwise_loss=` and `loss_function=`.
- `multithreading=True` or `False`; use `parallelism="multithreading"` or
  `parallelism="serial"`. Use `parallelism="multiprocessing", procs=N` for
  process workers.
- `.equations` and `.raw_julia_state_`; use `.equations_` and `.julia_state_`.
- The `pysr()` function, `pysr.install()`, and `pysr.init_julia()`; use
  `PySRRegressor`. Dependencies are installed when PySR is imported.

These constructor arguments are accepted but ignored with a warning:

- `julia_project` and `julia_kwargs`. Configure the backend through `juliapkg`;
  see [Customizing the backend](backend.md).
- `weights`, `variable_names`, and `Xresampled`. Pass them to `fit` instead.

## New APIs in v2

- **N-ary operators:** use
  `operators={1: ["sin"], 2: ["+", "*"], 3: ["clamp"]}`. The legacy
  `binary_operators` and `unary_operators` arguments still work, but cannot be
  combined with `operators`. Constraints for n-ary operators use n-tuples, such
  as `{"clamp": (-1, -1, 1)}`.
- **First-class mutations, crossovers, and plugins:** configure weighted
  instances with `mutations=` and `crossovers=`, and compose search behavior
  with `plugins=`. See the [API reference](api.md#mutations).
- **Custom value types:** `TypeSpec` supports expressions over strings, vectors,
  tensors, structs, and other Julia value types. See
  [Custom value types](examples.md#custom-value-types).
- **Template parameters:** `TemplateExpressionSpec.parameters` supports learned
  scalar or category-indexed parameter vectors.
- **Equation guesses:** seed a search with `guesses=["sin(x1 * 2.1)", ...]`.
  `fraction_replaced_guesses` controls their injection rate.
- **Autodiff backends:** `autodiff_backend="Enzyme"` is available as an
  experimental alternative to `"Zygote"`.
- **Worker controls:** `worker_imports` loads Julia packages on multiprocessing
  workers, and `worker_timeout` controls worker restart timeouts.
- **Additional mutations:** `weight_mutate_feature` controls feature mutation.
  Backsolve is disabled by default and can be configured with
  `BacksolveMutation(...)` through `mutations=`.

## Environment changes

- The `juliacall` requirement changed from `>=0.9.24,<0.9.36` to
  `>=0.9.28,<0.9.29`. Let your Python package manager resolve that narrower
  range when installing v2.
- The minimum versions remain Python 3.9 and Julia 1.10.
