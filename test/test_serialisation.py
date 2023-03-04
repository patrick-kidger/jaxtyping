import cloudpickle
import numpy as np
import torch

from jaxtyping import AbstractArray, Array, Shaped


def test_pickle():
    x = cloudpickle.dumps(Shaped[Array, ""])
    y = cloudpickle.dumps(AbstractArray)
    z = cloudpickle.dumps(Shaped[np.ndarray, ""])
    w = cloudpickle.dumps(Shaped[torch.Tensor, ""])
    cloudpickle.loads(x)
    cloudpickle.loads(y)
    cloudpickle.loads(z)
    cloudpickle.loads(w)
