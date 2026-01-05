# Array annotations

The shape and dtypes of arrays can be annotated in the form `dtype[array, shape]`, such as  
`jaxtyping.Float[torch.Tensor, "batch channels"]`.

## Dtype

The dtype should be any one of (all imported from `jaxtyping`):

- Any dtype at all: `Shaped`
    - Boolean: `Bool`
    - Any integer, unsigned integer, floating, or complex: `Num`
        - Any floating or complex: `Inexact`
            - Any floating point: `Float`
                - Of particular precision: `BFloat16`, `Float16`, `Float32`, `Float64`
            - Any complex: `Complex`
                - Of particular precision: `Complex64`, `Complex128`
        - Any integer or unsigned integer: `Integer`
            - Any unsigned integer: `UInt`
                - Of particular precision: `UInt2`, `UInt4`, `UInt8`, `UInt16`, `UInt32`, `UInt64`
            - Any signed integer: `Int`
                - Of particular precision: `Int2`, `Int4`, `Int8`, `Int16`, `Int32`, `Int64`
        - Any floating, integer, or unsigned integer: `Real`.

## Shape

**Symbols**

The shape should be a string of space-separated symbols, such as `"a b c d"`. Each symbol can be either an:

- `int`: fixed-size axis, e.g. `"28 28"`.
- `str`: variable-size axis, e.g. `"channels"`.
- A symbolic expression in terms of other variable-size axes, e.g.  
    `def remove_last(x: Float[torch.Tensor, "dim"]) -> Float[torch.Tensor, "dim-1"]`.  
    Symbolic expressions must not use any spaces, otherwise each piece is treated as as a separate axis.

When calling a function, variable-size axes and symbolic axes will be matched up across all arguments and checked for consistency. (See [Runtime type checking](./runtime-type-checking.md).)

**Modifiers**

In addition some modifiers can be applied:

- Prepend `*` to an axis to indicate that it can match multiple axes, e.g. `"*batch"` will match zero or more batch axes.
- Prepend `#` to an axis to indicate that it can be that size *or* equal to one – i.e. broadcasting is acceptable, e.g.  
    `def add(x: Float[torch.Tensor, "#foo"], y: Float[torch.Tensor, "#foo"]) -> Float[torch.Tensor, "#foo"]`.
- Prepend `_` to an axis to disable any runtime checking of that axis (so that it can be used just as documentation). This can also be used as just `_` on its own: e.g. `"b c _ _"`.
- Documentation-only names (i.e. they're ignored by jaxtyping) can be handled by prepending a name followed by `=` e.g. `Float[torch.Tensor, "rows=4 cols=3"]`.
- Prepend `?` to an axis to indicate that its size can vary within a PyTree structure. (See [PyTree annotations](./pytree.md).)

When using multiple modifiers, their order does not matter.

As a special case:

- `...`: anonymous zero or more axes (equivalent to `*_`) e.g. `"... c h w"`

**Notes**

- To denote a scalar shape use `""`, e.g. `Float[torch.Tensor, ""]`.
- To denote an arbitrary shape (and only check dtype) use `"..."`, e.g. `Float[torch.Tensor, "..."]`.
- You cannot have more than one use of multiple-axes, i.e. you can only use `...` or `*name` at most once in each array.
- A symbolic expression cannot be evaluated unless all of the axes sizes it refers to have already been processed. In practice this usually means that they should only be used in annotations for the return type, and only use axes declared in the arguments.
- Symbolic expressions are evaluated in two stages: they are first evaluated as f-strings using the arguments of the function, and second are evaluated using the processed axis sizes. The f-string evaluation means that they can use local variables by enclosing them with curly braces, e.g. `{variable}`, e.g.
    ```python
    def full(size: int, fill: float) -> Float[jax.Array, "{size}"]:
        return jax.numpy.full((size,), fill)

    class SomeClass:
        some_value = 5

        def full(self, fill: float) -> Float[jax.Array, "{self.some_value}+3"]:
            return jax.numpy.full((self.some_value + 3,), fill)
    ```

## Array

A variety of types are supported here:

**Arrays and Tensors:**

The following frameworks are supported:

```python
jax.Array / jax.numpy.ndarray  # these are both aliases of one another
np.ndarray
torch.Tensor
tf.Tensor
mx.array
```
_Despite the now-historical name, 'jax'typing also supports NumPy + PyTorch + TensorFlow + MLX._

**Duck-type arrays:** anything with `.shape` and `.dtype` attributes. The shape should be a `tuple[int, ...]` and the dtype should be a `str`. For example,
```python
class MyDuckArray:
    @property
    def shape(self) -> tuple[int, ...]:
        return (3, 4, 5)

    @property
    def dtype(self) -> str:
        return "my_dtype"

class MyDtype(jaxtyping.AbstractDtype):
    dtypes = ["my_dtype"]

x = MyDuckArray()
assert isinstance(x, MyDtype[MyDuckArray, "3 4 5"])
# checks that `type(x) == MyDuckArray`
# and that `x.shape == (3, 4, 5)`
# and that `x.dtype == "my_dtype"`
```

**Any:** use `typing.Any` to check just the shape/dtype, but not the array type.

**Unions:** these are unpacked. For example, `SomeDtype[A | B, "some shape"]` is equivalent to  
`SomeDtype[A, "some shape"] | SomeDtype[B, "some shape"]`.

**TypeVars:** in this case the runtime array is checked for matching the bounds or constraints of the `typing.TypeVar`.

**TypeAliasTypes:** Python 3.12 introduced the ability to write `type Foo = int | str`, in which case `Foo` is of type `typing.TypeAliasType`. In this case `SomeDtype[Foo, "some shape"]` corresponds to using the definition provided on the right hand side.

**Existing jaxtyping annotations:**
```python
Image = Float[jax.Array, "channels height width"]
BatchImage = Float[Image, "batch"]
```
in which case the additional shape is prepended, and the acceptable dtypes are the intersection of the two dtype specifiers used. (So that e.g. `BatchImage = Shaped[Image, "batch"]` would work just as well. But `Bool[Image, "batch"]` would throw an error, as there are no dtypes that are both bools and floats.) Thus the above is equivalent to
```python
BatchImage = Float[jax.Array, "batch channels height width"]
```

## JAX-specific types

As `jaxtyping` originally got its start as a JAX-specific library, then we provide some JAX-specific types. These are all only available if JAX is installed.

- `jaxtyping.Array`: alias for `jax.Array`
- `jaxtyping.ArrayLike`: alias for `jax.typing.ArrayLike`
- `jaxtyping.Scalar`: alias for `jaxtyping.Shaped[jax.Array, ""]`
- `jaxtyping.ScalarLike`: alias for `jaxtyping.Shaped[jax.typing.ArrayLike, ""]`
- `jaxtyping.Key`, which is the dtype of `jax.random.key`s. For example `jax.random.key(...)` produces a `jaxtyping.Key[jax.Array, ""]`.
- `jaxtyping.PRNGKeyArray`: alias for `jaxtyping.Key[jax.Array, ""] | jaxtyping.UInt32[jax.Array, "2"]` (Left: new-style typed keys; right: old-style keys. See [JEP 9263](https://docs.jax.dev/en/latest/jep/9263-typed-keys.html).)

Recalling that shape-and-dtype specified jaxtyping arrays can be nested, this means that e.g. you can annotate the output of `jax.random.split` with `Shaped[PRNGKeyArray, "2"]`.
