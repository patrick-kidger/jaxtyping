import abc
from typing import no_type_check

import jax.numpy as jnp
import jax.random as jr
import pytest

from jaxtyping import Array, Float, jaxtyped, print_bindings

from .helpers import ParamError, ReturnError


class M(metaclass=abc.ABCMeta):
    @jaxtyped(typechecker=None)
    def f(self):
        ...

    @jaxtyped(typechecker=None)
    @classmethod
    def g1(cls):
        return 3

    @classmethod
    @jaxtyped(typechecker=None)
    def g2(cls):
        return 4

    @jaxtyped(typechecker=None)
    @staticmethod
    def h1():
        return 3

    @staticmethod
    @jaxtyped(typechecker=None)
    def h2():
        return 4

    @jaxtyped(typechecker=None)
    @abc.abstractmethod
    def i1(self):
        ...

    @abc.abstractmethod
    @jaxtyped(typechecker=None)
    def i2(self):
        ...


class N:
    @jaxtyped(typechecker=None)
    @property
    def j1(self):
        return 3

    @property
    @jaxtyped(typechecker=None)
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
    def f(*args) -> None:
        pass

    f(1, 2)


def test_varkwargs(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(**kwargs) -> None:
        pass

    f(a=1, b=2)


def test_defaults(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: int, y=1) -> None:
        pass

    f(1)


def test_default_bindings(getkey, jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: int, y: int = 1) -> Float[Array, "x {y}"]:
        return jr.normal(getkey(), (x, y))

    f(1)
    f(1, 1)
    f(1, 0)
    f(1, 5)


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

    with pytest.warns(match="As of jaxtyping version 0.2.24"):

        @jaxtyped
        @typecheck
        def g(x: "LocalFoo") -> "LocalFoo":
            return x

    g(LocalFoo())

    # We don't check that errors are raised if it goes wrong, since we can't usually
    # resolve local type annotations at runtime. Best we can hope for is not to raise
    # a spurious error about not being able to find the type.


def test_print_bindings(typecheck, capfd):
    @jaxtyped(typechecker=typecheck)
    def f(x: Float[Array, "foo bar"]):
        print_bindings()

    capfd.readouterr()
    f(jnp.zeros((3, 4)))
    text, _ = capfd.readouterr()
    assert text == (
        "The current values for each jaxtyping axis annotation are as follows."
        "\nfoo=3\nbar=4\n"
    )


def test_no_type_check(typecheck):
    @jaxtyped(typechecker=typecheck)
    @no_type_check
    def f(x: Float[Array, "foo bar"]):
        pass

    @no_type_check
    @jaxtyped(typechecker=typecheck)
    def g(x: Float[Array, "foo bar"]):
        pass

    f("not an array")
    g("not an array")
