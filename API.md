# Full API

## Annotating array types

Each array is denoted by a type `dtype[array, shape]`, such as `Float[Array, "batch channels"]`.

### Shape

The shape should be a string of space-separated symbols, such as `"a b c d"`. Each symbol can be either an:
- `int`: fixed-size axis, e.g. `"28 28"`.
- `str`: variable-size axis, e.g. `"channels"`.
- A symbolic expression (without spaces!) in terms of other variable-size axes, e.g. `def remove_last(x: Float[Array, "dim"]) -> Float[Array, "dim-1"]`.

When calling a function, variable-size axes and symbolic axes will be matched up across all arguments and checked for consistency. (See [runtime type checking](#runtime-type-checking) below.)

In addition some modifiers can be applied:
- Prepend `*` to a dimension to indicate that it can match multiple axes, e.g. `"*batch c h w"` will match zero or more batch axes.
- Prepend `#` to a dimension to indicate that it can be that size *or* equal to one -- i.e. broadcasting is acceptable, e.g. `def add(x: Float[Array, "#foo"], y: Float[Array, "#foo"]) -> Float[Array, "#foo"]`.
- Prepend `_` to a dimension to disable any runtime checking of that dimension (so that it can be used just as documentation). This can also be used as just `_` on its own: e.g. `"b c _ _"`.

When using multiple modifiers, their order does not matter.

As a special case:
- `...`: anonymous zero or more axes (equivalent to `*_`) e.g. `"... c h w"`

Some notes:
- To denote a scalar shape use `""`, e.g. `Float[Array, ""]`.
- To denote an arbitrary shape (and only check dtype) use `"..."`, e.g. `Float[Array, "..."]`.
- You cannot have more than one use of multiple-axes, i.e. you can only use `...` or `*name` at most once in each array.
- An example of broadcasting multiple dimensions: `def add(x: Float[Array, "*#foo"], y: Float[Array, "*#foo"]) -> Float[Array, "*#foo"]`.
- A symbolic expression cannot be evaluated unless all of the axes sizes it refers to have already been processed. In practice this usually means that they should only be used in annotations for the return type, and only use axes declared in the arguments.

### Dtype

The dtype should be any one of (imported from `jaxtyping`):
- Any dtype at all: `Shaped`
  - Boolean: `Bool`
  - Any integer, unsigned integer, floating, or complex: `Num`
    - Any floating or complex: `Inexact`
      - Any floating point: `Float`
        - Of particular precision: `BFloat16`, `Float16`, `Float32`, `Float64`
      - Any complex: `Complex`
        - Of particular precision: `Complex64`, `Complex128`
    - Any integer or unsigned intger: `Integer`
      - Any unsigned integer: `UInt`
        - Of particular precision: `UInt8`, `UInt16`, `UInt32`, `UInt64`
      - Any signed integer: `Int`
        - Of particular precision: `Int8`, `Int16`, `Int32`, `Int64`

Unless you really want to force a particular precision, then for most applications you should probably allow any floating-point, any integer, etc. That is, use
```python
from jaxtyping import Array, Float
Float[Array, "some_shape"]
```
rather than
```python
from jaxtyping import Array, Float32
Float32[Array, "some_shape"]
```

### Array

The array should typically be a `jaxtyping.Array`, which is an alias for `jax.numpy.ndarray`.

But you can use other types as well. `jaxtyping` has support for JAX, NumPy, TensorFlow, and PyTorch, e.g.:
```python
Float[np.ndarray, "..."]
Float[tf.Tensor, "..."]
Float[torch.Tensor, "..."]
```

## PyTrees

### `jaxtyping.PyTree`

Each PyTree is denoted by a type `PyTree[LeafType]`, such as `PyTree[int]` or `PyTree[Union[str, Float32[Array, "b c"]]]`.

You can leave off the `[...]`, in which case `PyTree` is simply a suggestively-named alternative to `Any`. ([By definition all types are PyTrees.](https://jax.readthedocs.io/en/latest/pytrees.html))

## Runtime type checking

Single-argument type checking will work with any runtime type checker out-of-the-box.

To enable multi-argument consistency checks (i.e. that shapes match up between arrays), then you have two options, as discussed below. (And if either are too much magic for you, you can safely use neither and stick to just single-argument type checking.)

Regardless of your choice, **this approach synergises beautifully with `jax.jit`!** All shape checks will be performed at trace-time only, and will not impact runtime performance.

### Option 1: `jaxtyping.jaxtyped`

Decorate a function with this to have shapes checked for consistency across multiple arguments.

Example:

```python
# Import both the annotation and the `jaxtyped` decorator from `jaxtyping`
from jaxtyping import Array, Float32, jaxtyped

# Use your favourite typechecker: usually one of the two lines below.
from typeguard import typechecked as typechecker
from beartype import beartype as typechecker

# Write your function. @jaxtyped must be applied above @typechecker!
@jaxtyped
@typechecker
def batch_outer_product(x: Float32[Array, "b c1"],
                        y: Float32[Array, "b c2"]
                      ) -> Float32[Array, "b c1 c2"]:
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
by a typechecker against its arguments. (Or even forgo a typechecker that analyses arguments,
and instead just do your own manual `isinstance` checks.)

Only `isinstance` checks that pass will contribute to the store of axis name-size pairs; those
that fail will not. As such it is safe to write e.g. `assert not isinstance(x,
Float32[Array, "foo"])`.

### Option 2: `jaxtyping.install_import_hook`

It can be a lot of effort to add `@jaxtyped` decorators all over your codebase.
(Not to mention that double-decorators everywhere are a bit ugly.)

The easier option is usually to use the import hook.

This can be used via a `with` block; for example:
```python
from jaxtyping import install_import_hook
# Plus any one of the following:

# decorate @jaxtyped and @typeguard.typechecked
with install_import_hook("foo", ("typeguard", "typechecked")):
    import foo          # Any module imported inside this `with` block, whose name begins
    import foo.bar      # with the specified string, will automatically have both `@jaxtyped`
    import foo.bar.qux  # and the specified typechecker applied to all of their functions.

# decorate @jaxtyped and @beartype.beartype
with install_import_hook("foo", ("beartype", "beartype")):
    ...
    
# decorate only @jaxtyped (if you want that for some reason)
with install_import_hook("foo", None):
    ...
```

If you don't like using the `with` block, the hook can be used without that:
```python
hook = install_import_hook(...):
import ...
hook.uninstall()
```

The import hook can be applied to multiple packages via
```python
install_import_hook(["foo", "bar.baz"], ...)
```

The import hook will automatically decorate all functions, and the `__init__` method of dataclasses.

**Example: writing an end-user script**

```python
### entry_point.py
from jaxtyping import install_import_hook
with install_import_hook("do_stuff", ("typeguard", "typechecked")):
    import do_stuff

### do_stuff.py
from jaxtyping import Array, Float32

def g(x: Float32[Array, "..."]):
    ...
```

**Example: writing a library**

```python
### __init__.py
from jaxtyping import install_import_hook
with install_import_hook("my_library_name", ("beartype", "beartype")):
    from .subpackage import foo  # full name is my_library_name.subpackage so will be hook'd
    from .another_subpackage import bar  # full name is my_library_name.another_subpackage so will be hook'd.
```

#### pytest hook

The import hook can be installed at test-time only, as a pytest hook. The syntax is
```
pytest --jaxtyping-packages=foo,bar.baz,beartype.beartype
```
which will apply the import hook to all modules whose names start with either `foo` or `bar.baz`. The typechecker used in this example is `beartype.beartype`.

## Static type checking

jaxtyping should be compatible with static type checkers (the big three are `mypy`, `pyright`, `pytype`) out of the box.

Due to limitations of static type checkers, only the array type (JAX array vs NumPy array vs PyTorch tensor vs TensorFlow tensor) is checked. Shape and dtype are not checked. [See the FAQ](./FAQ.md) for more details.

## Abstract base classes

### `jaxtyping.AbstractDtype`

The base class of all dtypes. This can be used to create your own custom collection of dtypes (analogous to `Float`, `Inexact` etc.) For example:
```python
class UInt8or16(AbstractDtype):
    dtypes = ["uint8", "uint16"]

UInt8or16[Array, "shape"]
```
which is functionally equivalent to
```python
Union[UInt8[Array, "shape"], UInt16[Array, "shape"]]
```

### `jaxtyping.AbstractArray`

The base class of all shape-and-dtype-specified arrays, e.g. it's a base class
for `Float32[Array, "foo"]`.
