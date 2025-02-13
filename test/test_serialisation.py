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
