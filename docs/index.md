# Getting started

jaxtyping is a library providing type annotations **and runtime type-checking** for:

1. shape and dtype of [JAX](https://github.com/google/jax) arrays;
2. [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html).

 *(Now also supports PyTorch, NumPy, and TensorFlow!)*

## Installation

```bash
pip install jaxtyping
```

Requires Python 3.8+.

JAX is an optional dependency, required for a few JAX-specific types. If JAX is not installed then these will not be available, but you may still use jaxtyping to provide shape/dtype annotations for PyTorch/NumPy/TensorFlow/etc.

The annotations provided by jaxtyping are compatible with runtime type-checking packages, so it is common to also install one of these. The two most popular are [typeguard](https://github.com/agronholm/typeguard) (which exhaustively checks every argument) and [beartype](https://github.com/beartype/beartype) (which checks random pieces of arguments).

## Example

```python
from jaxtyping import Array, Float, PyTree

# Accepts floating-point 2D arrays with matching dimensions
def matrix_multiply(x: Float[Array, "dim1 dim2"],
                    y: Float[Array, "dim2 dim3"]
                  ) -> Float[Array, "dim1 dim3"]:
    ...

def accepts_pytree_of_ints(x: PyTree[int]):
    ...

def accepts_pytree_of_arrays(x: PyTree[Float[Array, "batch c1 c2"]]):
    ...
```

## Next steps

Have a read of the [Array annotations](./api/array.md) documentation on the left-hand bar!
