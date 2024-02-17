from typing import Any, AsyncIterator, Callable, Iterator, Union

import jax.numpy as jnp
import pytest

from jaxtyping import Array, Float, jaxtyped, Shaped
from jaxtyping._decorator import _change_annotations_to_any

from .helpers import ParamError


try:
    import torch
except ImportError:
    torch = None


async def _async_generator(x: Float[Array, "*"]) -> AsyncIterator[Float[Array, "*"]]:
    try:
        import asyncio

        await asyncio.sleep(0.01)
    except ImportError:
        pytest.skip("asyncio not available")
    yield x


async def _async_foo(async_generator):
    await next(async_generator(jnp.zeros(2)))
    await next(async_generator(jnp.zeros((3, 4))))


def _simple_generator(x: Float[Array, "*"]) -> Iterator[Float[Array, "*"]]:
    yield x


def _simple_foo(generator):
    next(generator(jnp.zeros(2)))
    next(generator(jnp.zeros((3, 4))))


@pytest.fixture(
    params=[(_simple_generator, _simple_foo), (_async_generator, _async_foo)]
)
def generator_and_foo(request):
    return request.param


def test_generators_simple(typecheck, generator_and_foo):
    generator, foo = generator_and_foo
    generator = jaxtyped(typechecker=typecheck)(generator)
    foo = jaxtyped(typechecker=typecheck)(foo)

    foo(generator)


def test_generators_double_decorator(typecheck, generator_and_foo):
    generator, foo = generator_and_foo
    generator = jaxtyped(typechecker=None)(typecheck(generator))
    foo = jaxtyped(typechecker=None)(foo)

    foo(generator)


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


### Some unit tests for the annotations modifier
def test_generators_annotations_modifier():
    assert _change_annotations_to_any(Float[Array, "1"]) == Any
    assert _change_annotations_to_any(Iterator[Float[Array, "1"]]) == Iterator[Any]

    assert (
        _change_annotations_to_any(
            Union[
                Union[
                    Iterator[Float[Array, "*"]],
                    Iterator[Callable[..., Float[Array, "1"]]],
                ],
                Iterator[Float[Array, "2"]],
                Float[Array, "3"],
            ]
        )
        == Union[Union[Iterator[Any], Iterator[Callable[..., Any]]], Iterator[Any], Any]
    )
