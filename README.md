<h1 align="center">jaxtyping</h1>

Type annotations **and runtime type-checking** for:

1. shape and dtype of [JAX](https://github.com/google/jax) arrays; *(Now also supports PyTorch, NumPy, and TensorFlow!)*
2. [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html).


**For example:**
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

## Installation

```bash
pip install jaxtyping
```

Requires Python 3.9+.

JAX is an optional dependency, required for a few JAX-specific types. If JAX is not installed then these will not be available, but you may still use jaxtyping to provide shape/dtype annotations for PyTorch/NumPy/TensorFlow/etc.

The annotations provided by jaxtyping are compatible with runtime type-checking packages, so it is common to also install one of these. The two most popular are [typeguard](https://github.com/agronholm/typeguard) (which exhaustively checks every argument) and [beartype](https://github.com/beartype/beartype) (which checks random pieces of arguments).

## Documentation

Available at [https://docs.kidger.site/jaxtyping](https://docs.kidger.site/jaxtyping).

## Finally

### See also: other libraries in the JAX ecosystem

[Equinox](https://github.com/patrick-kidger/equinox): neural networks.

[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.

[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.

[Lineax](https://github.com/google/lineax): linear solvers and linear least squares.

[Eqxvision](https://github.com/paganpasta/eqxvision): computer vision models.

[sympy2jax](https://github.com/google/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.

[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).

### Disclaimer

This is not an official Google product.
