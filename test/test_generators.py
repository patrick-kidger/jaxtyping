from typing import AsyncIterator, Iterator

import jax.numpy as jnp
import pytest

from jaxtyping import Array, Float, Shaped

from .helpers import ParamError


try:
    import torch
except ImportError:
    torch = None


def test_generators_simple(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def gen(x: Float[Array, "*"]) -> Iterator[Float[Array, "*"]]:
        yield x

    @jaxtyp(typecheck)
    def foo():
        next(gen(jnp.zeros(2)))
        next(gen(jnp.zeros((3, 4))))

    foo()


def test_generators_return_no_annotations(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def gen(x: Float[Array, "*"]):
        yield x

    @jaxtyp(typecheck)
    def foo():
        next(gen(jnp.zeros(2)))
        next(gen(jnp.zeros((3, 4))))

    foo()


@pytest.mark.asyncio
async def test_async_generators_simple(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    async def gen(x: Float[Array, "*"]) -> AsyncIterator[Float[Array, "*"]]:
        yield x

    @jaxtyp(typecheck)
    async def foo():
        async for _ in gen(jnp.zeros(2)):
            pass
        async for _ in gen(jnp.zeros((3, 4))):
            pass

    await foo()


def test_generators_dont_modify_same_annotations(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def g(x: Float[Array, "1"]) -> Iterator[Float[Array, "1"]]:
        yield x

    @jaxtyp(typecheck)
    def m(x: Float[Array, "1"]) -> Float[Array, "1"]:
        return x

    with pytest.raises(ParamError):
        next(g(jnp.zeros(2)))
    with pytest.raises(ParamError):
        m(jnp.zeros(2))


def test_generators_original_issue(jaxtyp, typecheck):
    # Effectively the same as https://github.com/patrick-kidger/jaxtyping/issues/91
    if torch is None:
        pytest.skip("torch is not available")

    @jaxtyp(typecheck)
    def g(x: Shaped[torch.Tensor, "*"]) -> Iterator[Shaped[torch.Tensor, "*"]]:
        yield x

    @jaxtyp(typecheck)
    def f():
        next(g(torch.zeros(1)))
        next(g(torch.zeros(2)))

    f()
