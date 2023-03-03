import cloudpickle

from jaxtyping import AbstractArray, Array, Shaped


def test_pickle():
    x = cloudpickle.dumps(Shaped[Array, ""])
    y = cloudpickle.dumps(AbstractArray)
    cloudpickle.loads(x)
    cloudpickle.loads(y)
