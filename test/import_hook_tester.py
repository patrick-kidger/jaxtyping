# Copyright (c) 2022 Google LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import dataclasses
from functools import wraps
from typing import no_type_check

import equinox as eqx
import jax.numpy as jnp
import pytest
from helpers import ParamError, ReturnError

import jaxtyping
from jaxtyping import Float32, Int


#
# Test that functions get checked
#


def g(x: Float32[jnp.ndarray, " b"]):
    pass


g(jnp.array([1.0]))
with pytest.raises(ParamError):
    g(jnp.array(1))


#
# Test that Equinox modules get checked
#


# Dataclass `__init__`, no converter
class Mod1(eqx.Module):
    foo: int
    bar: Float32[jnp.ndarray, " a"]


Mod1(1, jnp.array([1.0]))
with pytest.raises(ParamError):
    Mod1(1.0, jnp.array([1.0]))
with pytest.raises(ParamError):
    Mod1(1, jnp.array(1.0))


# Dataclass `__init__`, converter
class Mod2(eqx.Module):
    a: jnp.ndarray = eqx.field(converter=jnp.asarray)


Mod2(1)  # This will fail unless we run typechecking after conversion


# This silently passes -- the untyped `lambda x: x` launders the value through.
# No easy way to tackle this. That's okay.

# class BadMod2(eqx.Module):
#     a: jnp.ndarray = eqx.field(converter=lambda x: x)


# with pytest.raises(ParamError):
#     BadMod2(1)
# with pytest.raises(ParamError):
#     BadMod2("asdf")


# Custom `__init__`, no converter
class Mod3(eqx.Module):
    foo: int
    bar: Float32[jnp.ndarray, " a"]

    def __init__(self, foo: str, bar: Float32[jnp.ndarray, " a"]):
        self.foo = int(foo)
        self.bar = bar


Mod3("1", jnp.array([1.0]))
with pytest.raises(ParamError):
    Mod3(1, jnp.array([1.0]))
with pytest.raises(ParamError):
    Mod3("1", jnp.array(1.0))


# Custom `__init__`, converter
class Mod4(eqx.Module):
    a: Int[jnp.ndarray, ""] = eqx.field(converter=jnp.asarray)

    def __init__(self, a: str):
        self.a = int(a)


Mod4("1")  # This will fail unless we run typechecking after conversion


# Custom `__post_init__`, no converter
class Mod5(eqx.Module):
    foo: int
    bar: Float32[jnp.ndarray, " a"]

    def __post_init__(self):
        pass


Mod5(1, jnp.array([1.0]))
with pytest.raises(ParamError):
    Mod5(1.0, jnp.array([1.0]))
with pytest.raises(ParamError):
    Mod5(1, jnp.array(1.0))


# Dataclass `__init__`, converter
class Mod6(eqx.Module):
    a: jnp.ndarray = eqx.field(converter=jnp.asarray)

    def __post_init__(self):
        pass


Mod6(1)  # This will fail unless we run typechecking after conversion


#
# Test that dataclasses get checked
#


@dataclasses.dataclass
class D:
    foo: int
    bar: Float32[jnp.ndarray, " a"]


D(1, jnp.array([1.0]))
with pytest.raises(ParamError):
    D(1.0, jnp.array([1.0]))
with pytest.raises(ParamError):
    D(1, jnp.array(1.0))


#
# Test that methods get checked
#


class N(eqx.Module):
    a: jnp.ndarray

    def __init__(self, foo: str):
        self.a = jnp.array(1)

    def foo(self, x: jnp.ndarray):
        pass

    def bar(self) -> jnp.ndarray:
        return self.a


n = N("hi")
with pytest.raises(ParamError):
    N(123)
with pytest.raises(ParamError):
    n.foo("not_an_array_either")
bad_n = eqx.tree_at(lambda x: x.a, n, "not_an_array")
with pytest.raises(ReturnError):
    bad_n.bar()


#
# Test that we don't get called in `super()`.
#


called = False


class Base(eqx.Module):
    x: int

    def __init__(self):
        self.x = "not an int"
        global called
        assert not called
        called = True


class Derived(Base):
    def __init__(self):
        assert not called
        super().__init__()
        assert called
        self.x = 2


Derived()


#
# Test that stringified type annotations work


class Foo:
    pass


class Bar(eqx.Module):
    x: type[Foo]
    y: "type[Foo]"
    # Partially-stringified hints not tested; not supported.


Bar(Foo, Foo)

with pytest.raises(ParamError):
    Bar(1, Foo)


#
# Test that assert isinstance works (even if no arg/return annotations)


def isinstance_test(x):
    assert isinstance(x, Float32[jnp.ndarray, " b"])
    _ = x


isinstance_test(jnp.array([1.0]))
with pytest.raises(AssertionError):
    isinstance_test(jnp.array(1))


@no_type_check
def f(_: Float32[jnp.ndarray, "foo bar"]):
    pass


f("not an array")


# The following classes simulate type hints like `wp.array[wp.vec3]` from the `warp
#  library. `CustomType[int]` evaluates at runtime to an instance of `CustomGeneric`
#  via `__class_getitem__`. According to PEP 484 and standard typing rules, raw
# instances of classes (unless wrapped in typing constructs like `Literal` or
# `Annotated`) are not PEP-compliant type hints. Under the import hook, if
# `@no_type_check` is not respected, decorators like `@jaxtyped` would try to apply
# typecheckers (e.g. `beartype` or `typeguard`) to functions annotated with these
# PEP-noncompliant instances, causing typecheckers to crash at decoration time.
class CustomGeneric:
    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return f"CustomGeneric({self.arg})"


class CustomType:
    def __class_getitem__(cls, item):
        return CustomGeneric(item)


@no_type_check
def invalid_hint_func(x: CustomType[int]):
    pass


invalid_hint_func(None)


@no_type_check
class InvalidHintClass:
    def method(self, x: CustomType[int]):
        pass


InvalidHintClass().method(None)


def dummy_decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


@dummy_decorator
@no_type_check
@dummy_decorator
def decorated_func(x: CustomType[int]):
    pass


decorated_func(None)


@no_type_check
def outer_func():
    def inner_func(x: CustomType[int]):
        pass

    inner_func(None)


outer_func()


@no_type_check
class OuterClass:
    class InnerClass:
        def method(self, x: CustomType[int]):
            pass


OuterClass.InnerClass().method(None)


# Record that we've finished our checks successfully

jaxtyping._test_import_hook_counter += 1
