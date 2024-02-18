# Array annotations

The shape and dtypes of arrays can be annotated in the form `dtype[array, shape]`, such as `Float[Array, "batch channels"]`.

## Shape

**Symbols**

The shape should be a string of space-separated symbols, such as `"a b c d"`. Each symbol can be either an:

- `int`: fixed-size axis, e.g. `"28 28"`.
- `str`: variable-size axis, e.g. `"channels"`.
- A symbolic expression in terms of other variable-size axes, e.g.  
    `def remove_last(x: Float[Array, "dim"]) -> Float[Array, "dim-1"]`.  
    Symbolic expressions must not use any spaces, otherwise each piece is treated as as a separate axis.

When calling a function, variable-size axes and symbolic axes will be matched up across all arguments and checked for consistency. (See [Runtime type checking](./runtime-type-checking.md).)

**Modifiers**

In addition some modifiers can be applied:

- Prepend `*` to an axis to indicate that it can match multiple axes, e.g. `"*batch"` will match zero or more batch axes.
- Prepend `#` to an axis to indicate that it can be that size *or* equal to one -- i.e. broadcasting is acceptable, e.g.  
    `def add(x: Float[Array, "#foo"], y: Float[Array, "#foo"]) -> Float[Array, "#foo"]`.
- Prepend `_` to an axis to disable any runtime checking of that axis (so that it can be used just as documentation). This can also be used as just `_` on its own: e.g. `"b c _ _"`.
- Documentation-only names (i.e. they're ignored by jaxtyping) can be handled by prepending a name followed by `=` e.g. `Float[Array, "rows=4 cols=3"]`.
- Prepend `?` to an axis to indicate that its size can vary within a PyTree structure. (See [PyTree annotations](./pytree.md).)

When using multiple modifiers, their order does not matter.

As a special case:

- `...`: anonymous zero or more axes (equivalent to `*_`) e.g. `"... c h w"`

**Notes**

- To denote a scalar shape use `""`, e.g. `Float[Array, ""]`.
- To denote an arbitrary shape (and only check dtype) use `"..."`, e.g. `Float[Array, "..."]`.
- You cannot have more than one use of multiple-axes, i.e. you can only use `...` or `*name` at most once in each array.
- A symbolic expression cannot be evaluated unless all of the axes sizes it refers to have already been processed. In practice this usually means that they should only be used in annotations for the return type, and only use axes declared in the arguments.
- Symbolic expressions are evaluated in two stages: they are first evaluated as f-strings using the arguments of the function, and second are evaluated using the processed axis sizes. The f-string evaluation means that they can use local variables by enclosing them with curly braces, e.g. `{variable}`, e.g.
    ```python
    def full(size: int, fill: float) -> Float[Array, "{size}"]:
        return jax.numpy.full((size,), fill)

    class SomeClass:
        some_value = 5

        def full(self, fill: float) -> Float[Array, "{self.some_value}+3"]:
            return jax.numpy.full((self.some_value + 3,), fill)
    ```

## Dtype

The dtype should be any one of (all imported from `jaxtyping`):

- Any dtype at all: `Shaped`
    - Boolean: `Bool`
    - PRNG key: `Key`
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
        - Any floating, integer, or unsigned integer: `Real`.

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

## Array

The array should usually be a `jaxtyping.Array`, which is an alias for `jax.numpy.ndarray` (which is itself an alias for `jax.Array`).

`jaxtyping.ArrayLike` is also available, which is an alias for `jax.typing.ArrayLike`. This is a union over JAX arrays and the builtin `bool`/`int`/`float`/`complex`.

You can use non-JAX types as well. jaxtyping also supports NumPy, TensorFlow, and PyTorch, e.g.:
```python
Float[np.ndarray, "..."]
Float[tf.Tensor, "..."]
Float[torch.Tensor, "..."]
```

Shape-and-dtype specified jaxtyping arrays can also be used, e.g.
```python
Image = Float[Array, "channels height width"]
BatchImage = Float[Image, "batch"]
```
in which case the additional shape is prepended, and the acceptable dtypes are the intersection of the two dtype specifiers used. (So that e.g. `BatchImage = Shaped[Image, "batch"]` would work just as well. But `Bool[Image, "batch"]` would throw an error, as there are no dtypes that are both bools and floats.) Thus the above is equivalent to
```python
BatchImage = Float[Array, "batch channels height width"]
```

Note that `jaxtyping.{Array, ArrayLike}` are only available if JAX has been installed.

You can disable the automatic `jax` import by setting the environment variable `JAXTYPING_LOAD_JAX="no"`.

## Scalars, PRNG keys

For convenience, jaxtyping also includes `jaxtyping.Scalar`, `jaxtyping.ScalarLike`, and `jaxtyping.PRNGKeyArray`, defined as:
```python
Scalar = Shaped[Array, ""]
ScalarLike = Shaped[ArrayLike, ""]

# Left: new-style typed keys; right: old-style keys. See JEP 9263.
PRNGKeyArray = Union[Key[Array, ""], UInt32[Array, "2"]]
```

Recalling that shape-and-dtype specified jaxtyping arrays can be nested, this means that e.g. you can annotate the output of `jax.random.split` with `Shaped[PRNGKeyArray, "2"]`, or e.g. an integer scalar with `Int[Scalar, ""]`.

Note that `jaxtyping.{Scalar, ScalarLike, PRNGKeyArray}` are only available if JAX has been installed.
