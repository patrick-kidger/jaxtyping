# Full API

## Annotating array types

Each array is denoted by a type `dtype[shape]`, such as `f32["batch channels"]`.

### Shape

The shape should be a string of space-separated symbols, such as "a b c d". Each symbol can be:
- `int`: fixed-size axis, e.g. `f32["28 28"]`.
- `str`: variable-size axis, e.g. `f32["channels"]`.
- `_`: anonymous axis, e.g. `f32["batch channels _ _"]`.
- `...`: anonymous zero or more axes, e.g. `f32["... c h w"]`
- `*name`: zero or more variable-size axes, e.g. `f32["*batch c h w"]`
- Append `#` to a dimension size to indicate that it can be that size *or* equal to one -- i.e. broadcasting is acceptable.

When calling a function, variable-size axes will be matched up across all arguments and checked for consistency. (See [runtime type checking](#runtime-type-checking) below.)

Some notes:
- To denote a scalar shape use `""`, e.g. `f32[""]`.
- To denote an arbitrary shape (and only check dtype) use `"..."`, e.g. `f32["..."]`.
- You cannot have multiple variadic axes, i.e. you can only use `...` or `*name` at most once in each array.
- An example of broadcasting in one dimension: `add(x: f32["foo#"], y: f32["foo#"]) -> f32["foo#"]`.
- An example of broadcasting multiple dimensions: `add(x: f32["*foo#"], y: f32["*foo#"]) -> f32["*foo#"]`.

### Dtype

The dtype should be any one of (imported from `jaxtyping`):
- Any dtype at all: `Array`
  - Boolean: `b`
  - Any integer, unsigned integer, floating, or complex: `n` (for <ins>n</ins>umber)
    - Any floating or complex: `x` (for ine<ins>x</ins>act)
      - Any floating point: `f`
        - Floating point: `bf16`, `f16`, `f32`, `f64` (`bf16` is bfloat16)
      - Any complex: `c`
        - Complexes: `c64`, `c128`
    - Any integer or unsigned intger: `t` (for in<ins>t</ins>eger)
      - Any unsigned integer: `u`
        - Unsigned integer: `u8`, `u16`, `u32`, `u64`
      - Any signed integer: `i`
        - Signed integer: `i8`, `i16`, `i32`, `i64`

Unless you really want to force a particular precision, then for most applications you should probably allow any floating-point, any integer, etc. That is, use
```python
from jaxtyping import f
f["some_shape"]
```
rather than
```python
from jaxtyping import f32
f32["some_shape"]
```

## PyTrees

### `jaxtyping.PyTree`

Each PyTree is denoted by a type `PyTree[LeafType]`, such as `PyTree[int]` or `PyTree[Union[str, f32["b c"]]]`.

You can leave off the `[...]`, in which case `PyTree` is simply a suggestively-named alternative to `Any`. ([By definition all types are PyTrees.](https://jax.readthedocs.io/en/latest/pytrees.html))

## Runtime type checking

Single-argument type checking will work with any runtime type checker out-of-the-box.

To enable multi-argument consistency checks (i.e. that shapes match up between arrays), then you have two options, as discussed below. (And if either are too much magic for you, you can safely use neither and stick to just single-argument type checking.)

Regardless of your choice, **this approach synergises beautifully with `jax.jit`!** All shape checks will be performed at trace-time only, and will not impact runtime performance.

### `jaxtyping.jaxtyped`

Decorate a function with this to have shapes checked for consistency across multiple arguments.

Example:

```python
# Import both the annotation and the `jaxtyped` decorator from `jaxtyping`
from jaxtyping import f32, jaxtyped

# Use your favourite typechecker: usually one of the two lines below.
from typeguard import typechecked as typechecker
from beartype import beartype as typechecker

# Write your function. @jaxtyped must be applied above @typechecker!
@jaxtyped
@typechecker
def batch_outer_product(x: f32["b c1"], y: f32["b c2"]) -> f32["b c1 c2"]:
    return x[:, :, None] * y[:, None, :]
```

Note that `@jaxtyped` is applied above the type checker.

#### `jaxtyping.jaxtyped` for advanced users

Put precisely, all `isinstance` shape checks are scoped to the thread-local dynamic context
of a `jaxtyped` call. A new dynamic context will allow different dimensions
sizes to be bound to the same name. After this new dynamic context is finished
then the old one is returned to.

For example, this means you could leave off the `@jaxtyped` decorator to enforce that
this function use the same axes sizes as the function it was called from.

Likewise, this means you can use `isinstance` checks inside a function body
and have them contribute to the same collection of consistency checks performed
by a typechecker against its arguments. (Or even forgo a typechecker altogether,
and just do your own manual `isinstance` checks.)

Only `isinstance` checks that pass will contribute to the store of axis name-size pairs; those
that fail will not. As such it is safe to write e.g. `assert not isinstance(x,
f32["foo"])`.

### `jaxtyping.install_import_hook`

It can be a lot of effort to add `@jaxtyped` decorators all over your codebase.
(Not to mention that double-decorators everywhere are a bit ugly.) The easier
option is usually to use the import import hook.

Example:

```python
from jaxtyping import install_import_hook
# Plus either one of the following:
install_import_hook("foo", ("typeguard", "typechecked"))  # decorate @jaxtyped and @typeguard.typechecked
install_import_hook("foo", ("beartype", "beartype"))  # decorate @jaxtyped and @beartype.beartype
install_import_hook("foo", None)  # decorate only @jaxtyped (if you have manually applied typechecking decorators)
```

Any module imported **afterwards**, whose name begins with the specified string, will automatically have both `@jaxtyped` and the specified typechecker applied to all of their functions. (E.g. in the above example `foo`, `foo.bar`, `foo.bar.qux` would all be hook'd).

The import hook may be uninstalled after you've imported all the modules you're interested in:
```python
hook = install_import_hook(...)
...  # perform imports
hook.uninstall()
```

The import hook can be applied to multiple packages via
```python
install_import_hook(["foo", "bar.baz"], ...)
```

**Example: writing an end-user script**

```python
### entry_point.py
from jaxtyping import install_import_hook
install_import_hook("do_stuff", ("typeguard", "typechecked"))
import do_stuff

### do_stuff.py
from jaxtyping import f32

def g(x: f32["..."]):
    ...
```

**Example: writing a library**

```python
### __init__.py
from jaxtyping import install_import_hook
hook = install_import_hook("my_library_name", ("beartype", "beartype"))
from .subpackage import foo  # full name is my_library_name.subpackage so will be hook'd
from .another_subpackage import bar  # full name is my_library_name.another_subpackage so will be hook'd.
hook.uninstall()
del hook, install_import_hook, jaxtyping  # keep interface tidy
```

#### pytest hook

The import hook can be installed at test-time only, as a pytest hook. The syntax is
```
pytest --jaxtyping-packages=foo,bar.baz,beartype.beartype
```
which will apply the import hook to all modules whose names start with either `foo` or `bar.baz`. The typechecker used in this example is `beartype.beartype`.

## Abstract base classes

### `jaxtyping.AbstractDtype`

The base class of all dtypes. This can be used to create your own custom collection of dtypes (analogous to `n`, `x` etc.) For example:
```python
class u8_or_u16(AbstractDtype):
    dtypes = ["uint8", "uint16"]

u8_or_u16["shape"]
```
which is functionally equivalent to
```python
Union[u8["shape"], u16["shape"]]
```

### `jaxtyping.AbstractArray`

The base class of all shape-and-dtype-specified arrays, e.g. it's a base class
for `f32["foo"]`.
