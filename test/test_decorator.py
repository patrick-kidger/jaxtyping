import abc

from jaxtyping import jaxtyped


class M(metaclass=abc.ABCMeta):
    @jaxtyped
    @classmethod
    def f1(cls):
        return 3

    @classmethod
    @jaxtyped
    def f2(cls):
        return 4

    @jaxtyped
    @abc.abstractmethod
    def g1(self):
        ...

    @abc.abstractmethod
    @jaxtyped
    def g2(self):
        ...

    @jaxtyped
    def h(self):
        ...


# Check that the @jaxtyped decorator doesn't blat the __get__ of @classmethod
def test_classmethod():
    assert M.f1() == 3
    assert M.f2() == 4


# Check that the @jaxtyped decorator doesn't blat the __isabstractmethod__ of
# @abstractmethod
def test_abstractmethod():
    assert M.g1.__isabstractmethod__
    assert M.g2.__isabstractmethod__


def test_identity():
    assert M.h is M.h
