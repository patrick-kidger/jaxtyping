import abc

import jax.random as jr
import pytest

from jaxtyping import Array, Float, jaxtyped

from .helpers import ParamError, ReturnError


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


def test_context(getkey):
    a = jr.normal(getkey(), (3, 4))
    b = jr.normal(getkey(), (5,))
    with jaxtyped("context"):
        assert isinstance(a, Float[Array, "foo bar"])
        assert not isinstance(b, Float[Array, "foo"])
    assert isinstance(a, Float[Array, "foo bar"])
    assert isinstance(b, Float[Array, "foo"])


def test_varargs(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(*args):
        pass

    f(1, 2)


def test_varkwargs(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(**kwargs):
        pass

    f(a=1, b=2)


def test_defaults(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: int, y=1):
        pass

    f(1)


class _GlobalFoo:
    pass


def test_global_stringified_annotation(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: "_GlobalFoo") -> "_GlobalFoo":
        return x

    f(_GlobalFoo())

    @jaxtyp(typecheck)
    def g(x: int) -> "_GlobalFoo":
        return x

    @jaxtyp(typecheck)
    def h(x: "_GlobalFoo") -> int:
        return x

    with pytest.raises(ReturnError):
        g(1)

    with pytest.raises(ParamError):
        h(1)


# This test does not use `jaxtyp(typecheck)` because typeguard does some evil stack
# frame introspection to try and grab local variables.
def test_local_stringified_annotation(typecheck):
    class LocalFoo:
        pass

    @jaxtyped(typechecker=typecheck)
    def f(x: "LocalFoo") -> "LocalFoo":
        return x

    f(LocalFoo())

    @jaxtyped
    @typecheck
    def g(x: "LocalFoo") -> "LocalFoo":
        return x

    g(LocalFoo())

    # We don't check that errors are raised if it goes wrong, since we can't usually
    # resolve local type annotations at runtime. Best we can hope for is not to raise
    # a spurious error about not being able to find the type.
