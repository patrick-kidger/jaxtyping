import pickle

import cloudpickle
import numpy as np


try:
    import torch
except ImportError:
    torch = None

from jaxtyping import AbstractArray, Array, Float, Shaped


def test_pickle():
    for p in (pickle, cloudpickle):
        x = p.dumps(Shaped[Array, ""])
        y = p.loads(x)
        assert y.dtype is Shaped
        assert y.dim_str == ""

        x = p.dumps(AbstractArray)
        y = p.loads(x)
        assert y is AbstractArray

        x = p.dumps(Shaped[np.ndarray, "3 4 hi"])
        y = p.loads(x)
        assert y.dtype is Shaped
        assert y.dim_str == "3 4 hi"

        if torch is not None:
            x = p.dumps(Float[torch.Tensor, "batch length"])
            y = p.loads(x)
            assert y.dtype is Float
            assert y.dim_str == "batch length"


# The three regression tests below pin down that the module-level sentinels in
# `_array_types` survive pickle/cloudpickle round-trips with their identity intact. Each
# test does an `isinstance` check after the round-trip. That is the call path tripping
# the bug described in #390.


def test_pickle_any_dtype():
    for p in (pickle, cloudpickle):
        loaded = p.loads(p.dumps(Shaped[np.ndarray, "2 3"]))
        assert isinstance(np.zeros((2, 3)), loaded)


def test_pickle_anonymous_dim():
    for p in (pickle, cloudpickle):
        loaded = p.loads(p.dumps(Float[np.ndarray, "_ _"]))
        assert isinstance(np.zeros((2, 3)), loaded)


def test_pickle_anonymous_variadic_dim():
    for p in (pickle, cloudpickle):
        loaded = p.loads(p.dumps(Float[np.ndarray, "..."]))
        assert isinstance(np.zeros((2, 3)), loaded)
