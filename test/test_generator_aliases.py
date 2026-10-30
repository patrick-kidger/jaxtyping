import asyncio
from collections.abc import AsyncIterator, Iterator

import numpy as np
import pytest

from jaxtyping import Float

from .helpers import ParamError, ReturnError


@pytest.mark.parametrize("decorate_consumer_first", [False, True])
def test_generator_shared_array_alias(jaxtyp, typecheck, decorate_consumer_first):
    Features = Float[np.ndarray, "batch 3"]

    def scale(x: Features) -> Features:
        return 2 * x

    if decorate_consumer_first:
        scale = jaxtyp(typecheck)(scale)

    @jaxtyp(typecheck)
    def batches(x: Features) -> Iterator[Features]:
        yield x

    if not decorate_consumer_first:
        scale = jaxtyp(typecheck)(scale)

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(
        scale(next(batches(x))), [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]
    )
    for invalid in (np.zeros((2, 4)), np.zeros((2, 3), dtype=np.int32)):
        with pytest.raises(ParamError):
            scale(invalid)
        with pytest.raises(ParamError):
            next(batches(invalid))


def test_generator_nested_array_alias(jaxtyp, typecheck):
    Features = Float[np.ndarray, "batch 3"]

    @jaxtyp(typecheck)
    def batches() -> Iterator[tuple[Features, Features | None]]:
        yield np.zeros((2, 3)), None
        yield np.zeros((4, 3)), None

    @jaxtyp(typecheck)
    def bad_scale(x: Features) -> Features:
        return x[:, :2]

    for x, _ in batches():
        with pytest.raises(ReturnError):
            bad_scale(x)


def test_async_generator_shared_array_alias(jaxtyp, typecheck):
    Features = Float[np.ndarray, "batch 3"]

    @jaxtyp(typecheck)
    async def batches() -> AsyncIterator[Features]:
        yield np.zeros((2, 3))
        yield np.zeros((4, 3))

    @jaxtyp(typecheck)
    def scale(x: Features) -> Features:
        return 2 * x

    async def consume():
        async for x in batches():
            np.testing.assert_array_equal(scale(x), x)
            with pytest.raises(ParamError):
                scale(np.zeros((2, 4)))

    asyncio.run(consume())
