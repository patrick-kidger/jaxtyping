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


class BadMod2(eqx.Module):
    a: jnp.ndarray = eqx.field(converter=lambda x: x)


with pytest.raises(ParamError):
    BadMod2(1)
with pytest.raises(ParamError):
    BadMod2("asdf")


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
    # Note that this is the *only* kind of partially-stringified type annotation that
    # is supported. This is for compatibility with older Equinox versions.
    z: type["Foo"]


Bar(Foo, Foo, Foo)

with pytest.raises(ParamError):
    Bar(1, Foo, Foo)

# Record that we've finished our checks successfully

jaxtyping._test_import_hook_counter += 1
