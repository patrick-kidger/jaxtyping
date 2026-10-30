from collections.abc import Iterator
from typing import Callable, get_args, TypeGuard, Unpack

import numpy as np
import pytest

from jaxtyping import Float, jaxtyped


def _decorate_generator(fn, engine):
    # Assertions inspect __wrapped__: on Python 3.14, the outer wrapper can lazily
    # reevaluate annotation expressions instead of exposing the modified hint.
    if engine == "beartype":
        from beartype import beartype

        fn = beartype(fn)
    elif engine == "typeguard":
        from typeguard import typechecked

        fn = typechecked(fn)
    return jaxtyped(typechecker=None)(fn)


@pytest.mark.parametrize("engine", ["context", "beartype", "typeguard"])
def test_generator_preserves_typeguard_annotation(engine):
    def is_int(value: object) -> TypeGuard[int]:
        return isinstance(value, int)

    def predicates() -> Iterator[Callable[[object], TypeGuard[int]]]:
        yield is_int

    annotation = predicates.__annotations__["return"]
    predicates = _decorate_generator(predicates, engine)
    assert predicates.__wrapped__.__annotations__["return"] == annotation
    if engine == "context":
        assert predicates.__wrapped__.__annotations__["return"] is annotation
    predicate = next(predicates())
    assert predicate(3) is True
    assert predicate("three") is False


@pytest.mark.parametrize("engine", ["context", "beartype"])
def test_generator_preserves_unpack_annotation(engine):
    def pairs() -> Iterator[tuple[Unpack[tuple[int, str]]]]:
        yield 1, "one"

    annotation = pairs.__annotations__["return"]
    pairs = _decorate_generator(pairs, engine)
    assert pairs.__wrapped__.__annotations__["return"] == annotation
    if engine == "context":
        assert pairs.__wrapped__.__annotations__["return"] is annotation
    assert list(pairs()) == [(1, "one")]


@pytest.mark.parametrize("engine", ["context", "beartype", "typeguard"])
def test_generator_typeguard_array_descendant(engine):
    Features = Float[np.ndarray, "batch 3"]

    def is_features(value: object) -> TypeGuard[Features]:
        return isinstance(value, Features)

    def predicates() -> Iterator[Callable[[object], TypeGuard[Features]]]:
        yield is_features

    predicates = _decorate_generator(predicates, engine)
    callable_hint = get_args(predicates.__wrapped__.__annotations__["return"])[0]
    guard_hint = get_args(callable_hint)[1]
    generator_features = get_args(guard_hint)[0]
    # The generator's annotation must still bypass checks, while the alias used
    # by the yielded predicate retains its real shape and dtype requirements.
    invalid = np.zeros((2, 4))
    assert isinstance(invalid, generator_features)
    predicate = next(predicates())
    assert predicate(np.zeros((2, 3))) is True
    assert predicate(invalid) is False
    assert predicate(np.zeros((2, 3), dtype=np.int32)) is False


@pytest.mark.parametrize("engine", ["context", "beartype"])
def test_generator_unpack_array_descendant(engine):
    Features = Float[np.ndarray, "batch 3"]

    def pairs() -> Iterator[tuple[Unpack[tuple[Features, str]]]]:
        yield np.zeros((2, 4)), "two"
        yield np.zeros((5, 4)), "five"

    pairs = _decorate_generator(pairs, engine)
    tuple_hint = get_args(pairs.__wrapped__.__annotations__["return"])[0]
    unpack_hint = get_args(tuple_hint)[0]
    unpacked_tuple_hint = get_args(unpack_hint)[0]
    generator_features = get_args(unpacked_tuple_hint)[0]
    for array, label in pairs():
        assert array.shape == ({"two": 2, "five": 5}[label], 4)
        assert isinstance(array, generator_features)
        assert not isinstance(array, Features)
    assert isinstance(np.zeros((2, 3)), Features)
    assert not isinstance(np.zeros((2, 3), dtype=np.int32), Features)
