# Getting started

jaxtyping is a library providing type annotations **and runtime type-checking** for:

1. shape and dtype of [JAX](https://github.com/google/jax), [NumPy](https://github.com/numpy/numpy), [MLX](https://github.com/ml-explore/mlx), [PyTorch](https://github.com/pytorch/pytorch), [Tensorflow](https://github.com/tensorflow/tensorflow) arrays/tensors.
2. [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html).


## Installation

```bash
pip install jaxtyping
```

Requires Python 3.10+.

JAX is an optional dependency, required for a few JAX-specific types. If JAX is not installed then these will not be available, but you may still use jaxtyping to provide shape/dtype annotations for PyTorch/NumPy/TensorFlow/MLX/etc.

The annotations provided by jaxtyping are compatible with runtime type-checking packages, so it is common to also install one of these. The two most popular are [typeguard](https://github.com/agronholm/typeguard) (which exhaustively checks every argument) and [beartype](https://github.com/beartype/beartype) (which checks random pieces of arguments).

## Example

```python
from jaxtyping import Array, Float, PyTree

# Accepts floating-point 2D arrays with matching axes
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

## See also: other libraries in the JAX ecosystem

**Always useful**  
[Equinox](https://github.com/patrick-kidger/equinox): neural networks and everything not already in core JAX!  

**Deep learning**  
[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.  
[Orbax](https://github.com/google/orbax): checkpointing (async/multi-host/multi-device).  
[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).  
[paramax](https://github.com/danielward27/paramax): parameterizations and constraints for PyTrees.  

**Scientific computing**  
[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.  
[Optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, fixed points, and least squares.  
[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  
[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.  
[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.  
[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)  

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other JAX projects.  
