import cloudpickle

from jaxtyping import AbstractArray, Array, Shaped


def test_pickle():
    cloudpickle.dumps(Shaped[Array, ""])
    cloudpickle.dumps(AbstractArray)
