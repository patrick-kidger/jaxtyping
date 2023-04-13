import abc

from jaxtyping import jaxtyped


class M(metaclass=abc.ABCMeta):
    @jaxtyped
    def f(self):
        ...

    @jaxtyped
    @classmethod
    def g1(cls):
        return 3

    @classmethod
    @jaxtyped
    def g2(cls):
        return 4

    @jaxtyped
    @staticmethod
    def h1():
        return 3

    @staticmethod
    @jaxtyped
    def h2():
        return 4

    @jaxtyped
    @abc.abstractmethod
    def i1(self):
        ...

    @abc.abstractmethod
    @jaxtyped
    def i2(self):
        ...


class N:
    @jaxtyped
    @property
    def j1(self):
        return 3

    @property
    @jaxtyped
    def j2(self):
        return 4


def test_identity():
    assert M.f is M.f


def test_classmethod():
    assert M.g1() == 3
    assert M.g2() == 4


def test_staticmethod():
    assert M.h1() == 3
    assert M.h2() == 4


# Check that the @jaxtyped decorator doesn't blat the __isabstractmethod__ of
# @abstractmethod
def test_abstractmethod():
    assert M.i1.__isabstractmethod__
    assert M.i2.__isabstractmethod__


def test_property():
    assert N().j1 == 3
    assert N().j2 == 4
