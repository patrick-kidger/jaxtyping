import abc

from jaxtyping import jaxtyped


class M(metaclass=abc.ABCMeta):
    @jaxtyped
    @classmethod
    def f(cls):
        return 3

    @jaxtyped
    @abc.abstractmethod
    def g(self):
        ...


# Check that the @jaxtyped decorator doesn't blat the __get__ of @classmethod
def test_classmethod():
    assert M.f() == 3


# Check that the @jaxtyped decorator doesn't blat the __isabstractmethod__ of
# @abstractmethod
def test_abstractmethod():
    assert M.g.__isabstractmethod__
