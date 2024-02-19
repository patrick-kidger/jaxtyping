from typing import Tuple, Union

import pytest

from jaxtyping import (
    Array,
    Float,
    Float32,
    Integer,
    PRNGKeyArray,
    PyTree,
    Shaped,
)


@pytest.mark.parametrize(
    "make_fn",
    [
        lambda: Float[Array, "4"],
        lambda: Float32[Array, ""],
        lambda: Integer[Array, "1 2 3"],
        lambda: Shaped[PRNGKeyArray, "2"],
        lambda: Float[float, "#*shape"],
        lambda: PyTree[int],
        lambda: PyTree[float],
        lambda: PyTree[Float[Array, ""]],
        lambda: PyTree[Float32[Array, "*m b c"]],
        lambda: PyTree[PyTree[Float32[Array, "1 2 b *"]]],
        lambda: PyTree[Union[str, Float32[Array, "1"]]],
        lambda: PyTree[
            Tuple[int, float, Float[Array, ""], PyTree[Union[Float[Array, ""], float]]]
        ],
    ],
)
def test_equals(make_fn):
    assert make_fn() == make_fn()
