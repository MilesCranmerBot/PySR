# Toy Examples with Code

## Preamble

```python
import numpy as np
from pysr import *
```

## 1. Simple search

Here's a simple example where we
find the expression `2 cos(x3) + x0^2 - 2`.

```python
X = 2 * np.random.randn(100, 5)
y = 2 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2
model = PySRRegressor(binary_operators=["+", "-", "*", "/"])
model.fit(X, y)
print(model)
```

## 2. Custom operator

Here, we define a custom operator and use it to find an expression:

```python
X = 2 * np.random.randn(100, 5)
y = 1 / X[:, 0]
model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1/x},
)
model.fit(X, y)
print(model)
```

## 3. Multiple outputs

Here, we do the same thing, but with multiple expressions at once,
each requiring a different feature.

```python
X = 2 * np.random.randn(100, 5)
y = 1 / X[:, [0, 1, 2]]
model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1/x},
)
model.fit(X, y)
```

## 4. Plotting an expression

For now, let's consider the expressions for output 0.
We can see the LaTeX version of this with:

```python
model.latex()[0]
```

or output 1 with `model.latex()[1]`.

Let's plot the prediction against the truth:

```python
from matplotlib import pyplot as plt
plt.scatter(y[:, 0], model.predict(X)[:, 0])
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()
```

Which gives us:

![Truth vs Prediction](/images/example_plot.png)

We may also plot the output of a particular expression
by passing the index of the expression to `predict` (or
`sympy` or `latex` as well)

## 5. Feature selection

PySR and evolution-based symbolic regression in general performs
very poorly when the number of features is large.
Even, say, 10 features might be too much for a typical equation search.

If you are dealing with high-dimensional data with a particular type of structure,
you might consider using deep learning to break the problem into
smaller "chunks" which can then be solved by PySR, as explained in the paper
[2006.11287](https://arxiv.org/abs/2006.11287).

For tabular datasets, this is a bit trickier. Luckily, PySR has a built-in feature
selection mechanism. Simply declare the parameter `select_k_features=5`, for selecting
the most important 5 features.

Here is an example. Let's say we have 30 input features and 300 data points, but only 2
of those features are actually used:

```python
X = np.random.randn(300, 30)
y = X[:, 3]**2 - X[:, 19]**2 + 1.5
```

Let's create a model with the feature selection argument set up:

```python
model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp"],
    select_k_features=5,
)
```

Now let's fit this:

```python
model.fit(X, y)
```

Before the Julia backend is launched, you can see the string:

```text
Using features ['x3', 'x5', 'x7', 'x19', 'x21']
```

which indicates that the feature selection (powered by a gradient-boosting tree)
has successfully selected the relevant two features.

This fit should find the solution quickly, whereas with the huge number of features,
it would have struggled.

This simple preprocessing step is enough to simplify our tabular dataset,
but again, for more structured datasets, you should try the deep learning
approach mentioned above.

## 6. Denoising

Many datasets, especially in the observational sciences,
contain intrinsic noise. PySR is noise robust itself, as it is simply optimizing a loss function,
but there are still some additional steps you can take to reduce the effect of noise.

One thing you could do, which we won't detail here, is to create a custom log-likelihood
given some assumed noise model. By passing weights to the fit function, and
defining a custom loss function such as `elementwise_loss="myloss(x, y, w) = w * (x - y)^2"`,
you can define any sort of log-likelihood you wish. (However, note that it must be bounded at zero)

However, the simplest thing to do is preprocessing, just like for feature selection. To do this,
set the parameter `denoise=True`. This will fit a Gaussian process (containing a white noise kernel)
to the input dataset, and predict new targets (which are assumed to be denoised) from that Gaussian process.

For example:

```python
X = np.random.randn(100, 5)
noise = np.random.randn(100) * 0.1
y = np.exp(X[:, 0]) + X[:, 1] + X[:, 2] + noise
```

Let's create and fit a model with the denoising argument set up:

```python
model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp"],
    denoise=True,
)
model.fit(X, y)
print(model)
```

If all goes well, you should find that it predicts the correct input equation, without the noise term!

## 7. Julia packages and types

PySR uses [SymbolicRegression.jl](https://github.com/astroautomata/SymbolicRegression.jl)
as its search backend. This is a pure Julia package, and so can interface easily with any other
Julia package.
For some tasks, it may be necessary to load such a package.

For example, let's say we wish to discovery the following relationship:

$$ y = p_{3x + 1} - 5, $$

where $p_i$ is the $i$th prime number, and $x$ is the input feature.

Let's see if we can discover this using
the [Primes.jl](https://github.com/JuliaMath/Primes.jl) package.

First, let's get the Julia backend:

```python
from pysr import jl
```

`jl` stores the Julia runtime.

Now, let's run some Julia code to add the Primes.jl
package to the PySR environment:

```python
jl.seval("""
import Pkg
Pkg.add("Primes")
""")
```

This imports the Julia package manager, and uses it to install
`Primes.jl`. Now let's import `Primes.jl`:

```python
jl.seval("import Primes")
```

Now, we define a custom operator:

```python
jl.seval("""
function p(i::T) where T
    if (0.5 < i < 1000)
        return T(Primes.prime(round(Int, i)))
    else
        return T(NaN)
    end
end
""")
```

We have created a a function `p`, which takes an arbitrary number as input.
`p` first checks whether the input is between 0.5 and 1000.
If out-of-bounds, it returns `NaN`.
If in-bounds, it rounds it to the nearest integer, compures the corresponding prime number, and then
converts it to the same type as input.

Next, let's generate a list of primes for our test dataset.
Since we are using juliacall, we can just call `p` directly to do this:

```python
primes = {i: jl.p(i*1.0) for i in range(1, 999)}
```

Next, let's use this list of primes to create a dataset of $x, y$ pairs:

```python
import numpy as np

X = np.random.randint(0, 100, 100)[:, None]
y = [primes[3*X[i, 0] + 1] - 5 + np.random.randn()*0.001 for i in range(100)]
```

Note that we have also added a tiny bit of noise to the dataset.

Finally, let's create a PySR model, and pass the custom operator. We also need to define the sympy equivalent, which we can leave as a placeholder for now:

```python
from pysr import PySRRegressor
import sympy

class sympy_p(sympy.Function):
    pass

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["p"],
    niterations=100,
    extra_sympy_mappings={"p": sympy_p}
)
```

We are all set to go! Let's see if we can find the true relation:

```python
model.fit(X, y)
```

if all works out, you should be able to see the true relation (note that the constant offset might not be exactly 1, since it is allowed to round to the nearest integer).
You can get the sympy version of the best equation with:

```python
model.sympy()
```

## 8. Complex numbers

PySR can also search for complex-valued expressions. Simply pass
data with a complex datatype (e.g., `np.complex128`),
and PySR will automatically search for complex-valued expressions:

```python
import numpy as np

X = np.random.randn(100, 1) + 1j * np.random.randn(100, 1)
y = (1 + 2j) * np.cos(X[:, 0] * (0.5 - 0.2j))

model = PySRRegressor(
    binary_operators=["+", "-", "*"], unary_operators=["cos"], niterations=100,
)

model.fit(X, y)
```

You can see that all of the learned constants are now complex numbers.
We can get the sympy version of the best equation with:

```python
model.sympy()
```

We can also make predictions normally, by passing complex data:

```python
model.predict(X, -1)
```

to make predictions with the most accurate expression.

## 9. Dimensional constraints

One other feature we can exploit is dimensional analysis.
Say that we know the physical units of each feature and output,
and we want to find an expression that is dimensionally consistent.

We can do this as follows, using `DynamicQuantities.jl` to assign units,
passing a string specifying the units for each variable.
First, let's make some data on Newton's law of gravitation, using
astropy for units:

```python
import numpy as np
from astropy import units as u, constants as const

M = (np.random.rand(100) + 0.1) * const.M_sun
m = 100 * (np.random.rand(100) + 0.1) * u.kg
r = (np.random.rand(100) + 0.1) * const.R_earth
G = const.G

F = G * M * m / r**2
```

We can see the units of `F` with `F.unit`.

Now, let's create our model.
Since this data has such a large dynamic range,
let's also create a custom loss function
that looks at the error in log-space:

```python
elementwise_loss = """function loss_fnc(prediction, target)
    scatter_loss = abs(log((abs(prediction)+1f-20) / (abs(target)+1f-20)))
    sign_loss = 10 * (sign(prediction) - sign(target))^2
    return scatter_loss + sign_loss
end
"""
```

Now let's define our model:

```python
model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square"],
    elementwise_loss=elementwise_loss,
    complexity_of_constants=2,
    maxsize=25,
    niterations=100,
    populations=50,
    # Amount to penalize dimensional violations:
    dimensional_constraint_penalty=10**5,
)
```

and fit it, passing the unit information.
To do this, we need to use the format of [DynamicQuantities.jl](https://symbolicml.org/DynamicQuantities.jl/dev/#Usage).

```python
# Get numerical arrays to fit:
X = pd.DataFrame(dict(
    M=M.to("M_sun").value,
    m=m.to("kg").value,
    r=r.to("R_earth").value,
))
y = F.value

model.fit(
    X,
    y,
    X_units=["Constants.M_sun", "kg", "Constants.R_earth"],
    y_units="kg * m / s^2"
)
```

You can observe that all expressions with a loss under
our penalty are dimensionally consistent!
(The `"[⋅]"` indicates free units in a constant, which can cancel out other units in the expression.)
For example,

```julia
"y[m s⁻² kg] = (M[kg] * 2.6353e-22[⋅])"
```

would indicate that the expression is dimensionally consistent, with
a constant `"2.6353e-22[m s⁻²]"`.

Note that this expression has a large dynamic range so may be difficult to find. Consider searching with a larger `niterations` if needed.

Note that you can also search for exclusively dimensionless constants by settings
`dimensionless_constants_only` to `true`.

## 10. Expression Specifications

Expression specifications let you define a structured equation while retaining
normal prediction and export behavior. Use `TemplateExpressionSpec` when the
outer form is known and one or more inner expressions must be learned.

### Template Expressions

`TemplateExpressionSpec` allows you to define a specific structure for the equation.
For example, let's say we want to learn an equation of the form:

$$ y = \sin(f(x_1, x_2)) + g(x_3) $$

We can do this as follows:

```python
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec

# Create data
X = np.random.randn(1000, 3)
y = np.sin(X[:, 0] + X[:, 1]) + X[:, 2]**2

# Define template: we want sin(f(x1, x2)) + g(x3)
template = TemplateExpressionSpec(
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
    combine="sin(f(x1, x2)) + g(x3)",
)

model = PySRRegressor(
    expression_spec=template,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin"],
    maxsize=10,
)
model.fit(X, y)
```

### Parametric Expressions

When your data has categories with shared equation structure but different parameters,
you can use the `parameters` argument of `TemplateExpressionSpec` to specify learned category-specific parameters.

For example, let's say we want to learn an equation of the form:

$$ y = \alpha \sin(x_1) + \beta $$

where $\alpha$ and $\beta$ are different for each category.

Further, let's say we have 3 categories,
with $\alpha \in \{0.1, 1.5, -0.5\}$ and $\beta \in \{1.0, 2.0, 0.5\}$.

```python
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec

# Create data with 2 features and 3 categories
X = np.random.uniform(-3, 3, (1000, 2))
category = np.random.randint(0, 3, 1000)

# Parameters for each category
offsets = [0.1, 1.5, -0.5]
scales = [1.0, 2.0, 0.5]

# y = scale[category] * sin(x1) + offset[category]
y = np.array([
    scales[c] * np.sin(x1) + offsets[c]
    for x1, c in zip(X[:, 0], category)
])
```

Now, let's define our parametric expression:

```python
template = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x1", "x2", "category"],
    parameters={"p1": 3, "p2": 3},  # One parameter per category
    combine="f(x1, x2, p1[category], p2[category])"
)
```

Next, we pass the category as a _column_ in `X`
corresponding to the index we defined in `variable_names`.

**Note that because Julia is 1-indexed, we need to add 1 to the category index.**

```python
category_p_one = category + 1
X_with_category = np.column_stack([X, category_p_one])
```

Now, we can fit our model:

```python
model = PySRRegressor(
    expression_spec=template,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin"],
    maxsize=10,
)
model.fit(X_with_category, y)

# Predicting on new data
# model.predict(X_test_with_category)
```

See [Expression Specifications](/api/#expression-specifications) for more details.

You can use this approach for more complex cases,
where you have multiple expressions in the template and parameters that vary by category.

### Learning multiple outputs jointly

You can use `TemplateExpressionSpec` to learn several scalar expressions jointly
and compare their combined predictions with a vector target. This is useful when
the outputs share a known outer structure. Each learned expression still operates
on scalar values; the template combines their predictions and computes a scalar
residual.

For example, say we have 3-dimensional vectors where each component
follows a pattern with a shared term. Say the true model is:

$$\begin{align*}
y_1 &= \exp(x_1) + x_2^2 \\
y_2 &= \exp(x_1) + \sin(x_3) \\
y_3 &= \exp(x_1) + x_1 \cdot x_2
\end{align*}$$

Let's set this up:

```python
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec

n = 200
rstate = np.random.RandomState(0)
x1 = rstate.uniform(-2, 2, n)
x2 = rstate.uniform(-2, 2, n)
x3 = rstate.uniform(-2, 2, n)

# True model with shared component exp(x1):
y1 = np.exp(x1) + x2**2
y2 = np.exp(x1) + np.sin(x3)
y3 = np.exp(x1) + x1 * x2

# Add some noise
y1 += 0.05 * rstate.randn(n)
y2 += 0.05 * rstate.randn(n)
y3 += 0.05 * rstate.randn(n)
```

Now, we put everything in `X`; BOTH features and targets:

```python
X = np.column_stack([x1, x2, x3, y1, y2, y3])
```

Now, we can define our template expression:

```python
spec = TemplateExpressionSpec(
    expressions=["f1", "f2", "f3", "shared"],
    variable_names=["x1", "x2", "x3", "y1", "y2", "y3"],
    combine="""
        v = shared(x1, x2, x3)
        y1_predicted = v + f1(x1, x2, x3)
        y2_predicted = v + f2(x1, x2, x3)
        y3_predicted = v + f3(x1, x2, x3)

        residuals = (
            abs2(y1 - y1_predicted) +
            abs2(y2 - y2_predicted) +
            abs2(y3 - y3_predicted)
        )

        residuals
    """
)
```

Now, we can fit our model using this template. Since
we already computed the per-row squared error inside the template,
we can pass a dummy `y` to the `fit` method, and also define
an `elementwise_loss` that simply returns the residuals (which get
summed over the data):

```python
model = PySRRegressor(
    expression_spec=spec,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "sin"],
    maxsize=20,
    niterations=50,
    elementwise_loss="(pred, target) -> pred",
)

dummy_y = np.zeros(n)
model.fit(X, dummy_y)
```

After running, PySR should find both the shared component (`exp(x1)`) as well as individual components (`square(x2)`, `sin(x3)`, and `x1 * x2`).

You can access the individual expressions through the Julia objects:

```python
# Simply get the expression with the highest score:
idx = model.equations_.score.idxmax()

# Extract the Julia object:
julia_expr = model.equations_.loc[idx, 'julia_expression']

# Access individual subexpressions:
for name in ['f1', 'f2', 'f3', 'shared']:
    tree = getattr(julia_expr.trees, name)
    print(f"{name}: {tree}")
```

We can also evaluate individual expressions:

```python
from pysr import jl
from pysr.julia_helpers import jl_array

SR = jl.SymbolicRegression

# Get individual trees
f1_tree = julia_expr.trees.f1
shared_tree = julia_expr.trees.shared

# Evaluate at specific points (x1=1, x2=2, x3=3)
test_inputs = jl_array(np.array([[1.0], [2.0], [3.0]]))
f1_result, _ = SR.eval_tree_array(f1_tree, test_inputs, model.julia_options_)
shared_result, _ = SR.eval_tree_array(shared_tree, test_inputs, model.julia_options_)

print(f"f1 at (1,2,3): {f1_result[0]}")  # Should be ~4.0 for x2^2
print(f"shared at (1,2,3): {shared_result[0]}")  # Should be ~2.718 for exp(1)
```

## 11. Using TensorBoard for Logging

You can use TensorBoard to visualize the search progress, as well as
record hyperparameters and final metrics (like `min_loss` and `pareto_volume` - the latter of which
is a performance measure of the entire Pareto front).

```python
import numpy as np
from pysr import PySRRegressor, TensorBoardLoggerSpec

rstate = np.random.RandomState(42)

# Uniform dist between -3 and 3:
X = rstate.uniform(-3, 3, (1000, 2))
y = np.exp(X[:, 0]) + X[:, 1]

# Create a logger that writes to "logs/run*":
logger_spec = TensorBoardLoggerSpec(
    log_dir="logs/run",
    log_interval=10,  # Log every 10 iterations
)

model = PySRRegressor(
    binary_operators=["+", "*", "-", "/"],
    logger_spec=logger_spec,
)
model.fit(X, y)
```

You can then view the logs with:

```bash
tensorboard --logdir logs/
```

## 12. Using differential operators

As part of the `TemplateExpressionSpec` described above,
you can also use differential operators within the template.
The operator for this is `D` which takes an expression as the first argument,
and the argument _index_ we are differentiating as the second argument.
This lets you compute integrals via evolution.

For example, let's say we wish to find the integral of $\frac{1}{x^2 \sqrt{x^2 - 1}}$
in the range $x > 1$.
We can compute the derivative of a function $f(x)$, and compare that
to numerical samples of $\frac{1}{x^2\sqrt{x^2-1}}$. Then, by extension,
$f(x)$ represents the indefinite integral of it with some constant offset!

```python
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec

x = np.random.uniform(1, 10, (1000,))  # Integrand sampling points
y = 1 / (x**2 * np.sqrt(x**2 - 1))     # Evaluation of the integrand

expression_spec = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x"],
    combine="df = D(f, 1); df(x)",
)

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    expression_spec=expression_spec,
    maxsize=20,
)
model.fit(x[:, np.newaxis], y)
```

If everything works, you should find something that simplifies to $\frac{\sqrt{x^2 - 1}}{x}$.

Here, we write out a full function in Julia.

## 13. Custom value types

After working with custom operators, losses, and expression specifications,
`TypeSpec` lets you change the value type used throughout a search.

The template example above coordinates several scalar-valued expression trees.
With `TypeSpec`, vectors or tensors flow through each expression, constant, and
operator.

Each specification creates a private Julia `Value` type. Its preamble,
operators, and loss are replayed on multiprocessing workers and when loading a
checkpoint. Put every required Julia definition in those source fields or use a
package-qualified name.

TypeSpec supports the default expression shape, prediction, checkpoint reload,
and serial, multithreaded, or multiprocessing search. It does not support
guesses, weights, units, denoising, feature selection, resampling, multi-output
targets, turbo or bumper evaluation, autodiff backends, alternate expression
specifications, or SymPy, JAX, Torch, and LaTeX export. Restoring a TypeSpec
model requires its `checkpoint.pkl`; a hall-of-fame CSV is insufficient.

### Vector-valued expression trees

This example searches for a program over two-dimensional vectors:

$$
y = \operatorname{rotate90}(x_1) + 2x_2 +
\begin{bmatrix}0.5 \\ -1.0\end{bmatrix}.
$$

Each cell of `X` and `y` contains one vector:

```python
import numpy as np
import pandas as pd

from pysr import PySRRegressor, TypeSpec

rng = np.random.default_rng(0)
x1 = [rng.normal(size=2) for _ in range(128)]
x2 = [rng.normal(size=2) for _ in range(128)]
X = pd.DataFrame({"x1": x1, "x2": x2})
y = np.empty(128, dtype=object)
offset = np.array([0.5, -1.0])
y[:] = [np.array([-a[1], a[0]]) + 2 * b + offset for a, b in zip(x1, x2)]
```

PySR wraps each vector in a private Julia `Value` type. Operator and loss source
can use `Value` directly, while the hooks explain how vectors participate in
evolution and constant optimization:

```python
type_spec = TypeSpec(
    fields={"data": "Vector{Float64}"},
    # What is an empty or zero-like value of this type?
    init_value="() -> Value(zeros(2))",
    # How should a new constant be sampled?
    sample_value="(rng, options) -> Value(randn(rng, 2))",
    # How should evolution mutate an existing constant?
    mutate_value=(
        "(rng, value, temperature, options) -> "
        "Value(value.data + temperature * randn(rng, 2))"
    ),
    # How many scalar constants does one vector contain?
    count_scalar_constants=2,
    is_valid="value -> all(isfinite, value.data)",
    can_optimize=True,
    # Flatten a vector into the optimizer's buffer, starting at one-based `idx`:
    pack_scalar_constants="""
    (buffer, idx, value) -> begin
        buffer[idx:idx+1] .= value.data
        idx + 2
    end
    """,
    # Rebuild the vector and return `(first_unused_index, rebuilt_value)`:
    unpack_scalar_constants="""
    (buffer, idx, value) -> (idx + 2, Value(copy(buffer[idx:idx+1])))
    """,
    number_type="Float64",
)

model = PySRRegressor(
    type_spec=type_spec,
    operators={
        1: [
            "rotate90(a::Value) = Value([-a.data[2], a.data[1]])",
            "double(a::Value) = Value(2a.data)",
        ],
        2: ["add_vectors(a::Value, b::Value) = Value(a.data + b.data)"],
    },
    elementwise_loss=(
        "vector_loss(a::Value, b::Value)::Float64 = "
        "sum(abs2, a.data - b.data)"
    ),
    niterations=40,
    populations=4,
    maxsize=10,
    progress=False,
)

model.fit(X, y)
print(model.equations_)
```

The target can be represented as
`add_vectors(add_vectors(rotate90(x1), double(x2)), [0.5, -1.0])`, including a
learned vector-valued constant. PySR searches over both the program structure
and the two components of that constant.

<details>
<summary>String-valued expressions and discrete constants</summary>

Strings demonstrate a value type whose constants are evolved discretely rather
than optimized with BFGS. This search learns to join two transformed strings
with a sampled separator:

```python
import numpy as np
import pandas as pd

from pysr import PySRRegressor, TypeSpec

X = pd.DataFrame(
    {
        "first": ["Py", "symbolic", "hello", "left"],
        "second": ["SR", "regression", "world", "right"],
    }
)
y = np.array(
    [f"{a.lower()}-{b.upper()}" for a, b in X.itertuples(index=False)],
    dtype=object,
)

type_spec = TypeSpec(
    fields={"data": "String"},
    init_value='() -> Value("")',
    sample_value='(rng, options) -> Value(rand(rng, ("", "-", "_")))',
    mutate_value=(
        '(rng, value, temperature, options) -> '
        'Value(rand(rng, ("", "-", "_")))'
    ),
    count_scalar_constants=1,
    is_valid="value -> true",
    can_optimize=False,
)

model = PySRRegressor(
    type_spec=type_spec,
    operators={
        1: [
            "string_lowercase(x::Value) = Value(lowercase(x.data))",
            "string_uppercase(x::Value) = Value(uppercase(x.data))",
        ],
        2: ["string_concat(a::Value, b::Value) = Value(a.data * b.data)"],
    },
    elementwise_loss=(
        "string_loss(a::Value, b::Value)::Float64 = "
        "Float64(Base.editdistance(a.data, b.data))"
    ),
    niterations=40,
)

model.fit(X, y)
print(model.equations_)
```

Here `can_optimize=False` is appropriate because a string has no continuous
scalar representation for BFGS. Evolution still samples and mutates string
constants such as the separator.

</details>

<details>
<summary>Advanced: recovering a neural network with tensor constants</summary>

`TypeSpec` can place scalar, vector, and matrix constants in one Julia value
type. The scalar-constant hooks let the optimizer flatten each constant for
BFGS and then rebuild its original shape.

Here we recover a two-layer neural network

$$ y = W_2\operatorname{relu}(W_1x + b_1) + b_2 $$

from vector-valued data. Safe operators return an invalid value for shape
mismatches, so arbitrary expressions from the search cannot throw dimension
errors:

```python
import numpy as np
import pandas as pd

from pysr import PySRRegressor, TypeSpec

preamble = """
const NNPayload = Union{Float64, Vector{Float64}, Matrix{Float64}}

safe_matmul(a::Matrix{Float64}, b::Vector{Float64}) =
    size(a, 2) == length(b) ? a * b : NaN
safe_matmul(::NNPayload, ::NNPayload) = NaN

safe_add(a::Float64, b::Float64) = a + b
safe_add(a::T, b::T) where {T<:Union{Vector{Float64}, Matrix{Float64}}} =
    size(a) == size(b) ? a + b : NaN
safe_add(::NNPayload, ::NNPayload) = NaN

# Constants sample a random rank, generating only the payload that was chosen:
function random_nn_payload(rng)
    rank = rand(rng, 0:2)
    rank == 0 ? randn(rng) : rank == 1 ? randn(rng, 2) : randn(rng, 2, 2)
end
"""

type_spec = TypeSpec(
    fields={"data": "NNPayload"},
    init_value="() -> Value(0.0)",
    sample_value="(rng, options) -> Value(random_nn_payload(rng))",
    # Mutations usually perturb every scalar in the payload, but occasionally
    # resample a fresh rank:
    mutate_value="""
    (rng, value, temperature, options) -> if rand(rng) < 0.1
        Value(random_nn_payload(rng))
    else
        Value(value.data .+ temperature .* randn(rng, size(value.data)...))
    end
    """,
    # Return the number of Float64 entries used to represent one constant:
    count_scalar_constants="value -> length(value.data)",
    # Write those entries into `buffer` from the one-based `idx`, then return
    # the first unused index:
    pack_scalar_constants="""
    (buffer, idx, value) -> begin
        n = length(value.data)
        buffer[idx:idx+n-1] .= value.data isa Float64 ? value.data : vec(value.data)
        idx + n
    end
    """,
    # Read the entries back and return `(first_unused_index, rebuilt_value)`:
    unpack_scalar_constants="""
    (buffer, idx, value) -> begin
        n = length(value.data)
        data = value.data isa Float64 ? buffer[idx] :
            reshape(copy(buffer[idx:idx+n-1]), size(value.data))
        (idx + n, Value(data))
    end
    """,
    number_type="Float64",
    is_valid="value -> all(isfinite, value.data)",
    can_optimize=True,
    preamble=preamble,
)
```

Generate training data from fixed $2\times2$ weights and two-element biases:

```python
rng = np.random.default_rng(0)
x_values = rng.normal(size=(64, 2))
W1 = np.array([[1.2, -0.7], [0.5, 1.1]])
b1 = np.array([0.3, -0.2])
W2 = np.array([[0.8, -1.0], [1.3, 0.4]])
b2 = np.array([-0.4, 0.2])
y_values = (W2 @ np.maximum(x_values @ W1.T + b1, 0).T).T + b2

X = pd.DataFrame({"x": list(x_values)})
y = pd.Series(list(y_values), dtype=object)
```

Search with matrix multiplication, elementwise ReLU, and addition. BFGS is the
default constant optimizer:

```python
model = PySRRegressor(
    type_spec=type_spec,
    operators={
        1: ["nn_relu(a::Value) = Value(max.(a.data, 0.0))"],
        2: [
            "nn_matmul(a::Value, b::Value) = Value(safe_matmul(a.data, b.data))",
            "nn_add(a::Value, b::Value) = Value(safe_add(a.data, b.data))",
        ],
    },
    elementwise_loss=(
        "nn_mse(a::Value, b::Value)::Float64 = "
        "a.data isa Vector && b.data isa Vector && size(a.data) == size(b.data) "
        "? sum(abs2, a.data .- b.data) / length(a.data) : 1.0e6"
    ),
    niterations=100,
    populations=4,
    maxsize=11,
)

model.fit(X, y)
print(model.equations_)
```

The search recovers a two-layer form such as
`nn_matmul(W2, nn_add(b, nn_relu(nn_matmul(W1, nn_add(x, c)))))`. Both biases
are absorbed into the fitted constants, through $b_1 = W_1c$ and $b_2 = W_2b$;
each displayed `Value` contains the fitted matrix or vector payload.

</details>

## 15. Additional features

For the many other features available in PySR, please
read the [Options section](options.md).
