import cloudpickle
import numpy as np


try:
    import torch
except ImportError:
    torch = None

from jaxtyping import AbstractArray, Array, Shaped


def test_pickle():
    x = cloudpickle.dumps(Shaped[Array, ""])
    cloudpickle.loads(x)

    y = cloudpickle.dumps(AbstractArray)
    cloudpickle.loads(y)

    z = cloudpickle.dumps(Shaped[np.ndarray, ""])
    cloudpickle.loads(z)

    if torch is not None:
        w = cloudpickle.dumps(Shaped[torch.Tensor, ""])
        cloudpickle.loads(w)
