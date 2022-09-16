import threading

import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked


def test_threading():
    @jaxtyped
    @typecheckef
    def add(x: Float[Array, 'a b'], y: Float[Array, 'a b']) -> F[Array, 'a b']:
        return x + y

    def run():
        a = jnp.array([[1., 2.]])
        b = jnp.array([[2., 3.]])
        add(a, b)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
