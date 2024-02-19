from typing import AsyncIterator, Iterator

import jax.numpy as jnp
import pytest

from jaxtyping import Array, Float, jaxtyped, Shaped

from .helpers import ParamError


try:
    import torch
except ImportError:
    torch = None


def test_generators_simple(typecheck):
    @jaxtyped(typechecker=typecheck)
    def gen(x: Float[Array, "*"]) -> Iterator[Float[Array, "*"]]:
        yield x

    @jaxtyped(typechecker=typecheck)
    def foo():
        next(gen(jnp.zeros(2)))
        next(gen(jnp.zeros((3, 4))))

    foo()


def test_generators_double_decorator(typecheck):
    @jaxtyped(typechecker=None)
    @typecheck
    def gen(x: Float[Array, "*"]) -> Iterator[Float[Array, "*"]]:
        yield x

    @jaxtyped(typechecker=None)
    def foo():
        next(gen(jnp.zeros(2)))
        next(gen(jnp.zeros((3, 4))))

    foo()


@pytest.mark.asyncio
async def test_async_generators_simple(typecheck):
    @jaxtyped(typechecker=typecheck)
    async def gen(x: Float[Array, "*"]) -> AsyncIterator[Float[Array, "*"]]:
        yield x

    @jaxtyped(typechecker=typecheck)
    async def foo():
        async for _ in gen(jnp.zeros(2)):
            pass
        async for _ in gen(jnp.zeros((3, 4))):
            pass

    await foo()


@pytest.mark.asyncio
async def test_async_generators_double_decorator(typecheck):
    @jaxtyped(typechecker=None)
    @typecheck
    async def gen(x: Float[Array, "*"]) -> AsyncIterator[Float[Array, "*"]]:
        yield x

    @jaxtyped(typechecker=None)
    async def foo():
        async for _ in gen(jnp.zeros(2)):
            pass
        async for _ in gen(jnp.zeros((3, 4))):
            pass

    await foo()


def test_generators_dont_modify_same_annotations(typecheck):
    @jaxtyped(typechecker=None)
    @typecheck
    def g(x: Float[Array, "1"]) -> Iterator[Float[Array, "1"]]:
        yield x

    @jaxtyped(typechecker=typecheck)
    def m(x: Float[Array, "1"]) -> Iterator[Float[Array, "1"]]:
        return x

    with pytest.raises(ParamError):
        next(g(jnp.zeros(2)))
    with pytest.raises(ParamError):
        m(jnp.zeros(2))


def test_generators_original_issue(typecheck):
    # Effectively the same as https://github.com/patrick-kidger/jaxtyping/issues/91
    if torch is None:
        pytest.skip("torch is not available")

    @jaxtyped(typechecker=None)
    @typecheck
    def g(x: Shaped[torch.Tensor, "*"]) -> Iterator[Shaped[torch.Tensor, "*"]]:
        yield x

    @jaxtyped(typechecker=None)
    def f():
        next(g(torch.zeros(1)))
        next(g(torch.zeros(2)))

    f()
