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

from typing import Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from jaxtyping import Float, jaxtyped, PyTree

from .helpers import make_mlp, ParamError


def test_direct(typecheck):
    @typecheck
    def g(x: PyTree):
        pass

    g(1)
    g({"a": jnp.array(1), "b": [object()]})
    g(object())

    @typecheck
    def h() -> PyTree:
        return object()

    h()


def test_subscript(getkey, typecheck):
    @typecheck
    def g(x: PyTree[int]):
        pass

    g(1)
    g([1, 2, {"a": 3}])
    g(jax.tree_map(lambda _: 1, make_mlp(getkey())))

    with pytest.raises(ParamError):
        g(object())
    with pytest.raises(ParamError):
        g("hi")
    with pytest.raises(ParamError):
        g([1, 2, {"a": 3}, "bye"])


def test_leaf_pytrees(getkey, typecheck):
    @typecheck
    def g(x: PyTree[eqx.nn.MLP]):
        pass

    g(make_mlp(getkey()))
    g([make_mlp(getkey()), make_mlp(getkey()), {"a": make_mlp(getkey())}])

    with pytest.raises(ParamError):
        g([1, 2])
    with pytest.raises(ParamError):
        g([1, 2, make_mlp()])


def test_nested_pytrees(getkey, typecheck):
    # PyTree[...] is logically equivalent to PyTree[PyTree[...]]
    @typecheck
    def g(x: PyTree[PyTree[eqx.nn.MLP]]):
        pass

    g(make_mlp(getkey()))
    g([make_mlp(getkey()), make_mlp(getkey()), {"a": make_mlp(getkey())}])

    with pytest.raises(ParamError):
        g([1, 2])
    with pytest.raises(ParamError):
        g([1, 2, make_mlp()])


def test_pytree_array(typecheck):
    @jaxtyped
    @typecheck
    def g(x: PyTree[Float[jnp.ndarray, "..."]]):
        pass

    g(jnp.array(1.0))
    g([jnp.array(1.0), jnp.array(1.0), {"a": jnp.array(1.0)}])
    with pytest.raises(ParamError):
        g(jnp.array(1))
    with pytest.raises(ParamError):
        g(1.0)


def test_pytree_shaped_array(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: PyTree[Float[jnp.ndarray, "b c"]]):
        pass

    g(jnp.array([[1.0]]))
    g([jr.normal(getkey(), (1, 1)), jr.normal(getkey(), (1, 1))])
    g([jr.normal(getkey(), (3, 4)), jr.normal(getkey(), (3, 4))])
    with pytest.raises(ParamError):
        g(jnp.array(1.0))
    with pytest.raises(ParamError):
        g(jnp.array([[1]]))
    with pytest.raises(ParamError):
        g([jr.normal(getkey(), (3, 4)), jr.normal(getkey(), (1, 1))])
    with pytest.raises(ParamError):
        g([jr.normal(getkey(), (1, 1)), jr.normal(getkey(), (3, 4))])


def test_pytree_union(typecheck):
    @typecheck
    def g(x: PyTree[Union[int, str]]):
        pass

    g([1])
    g(["hi"])
    g([1, "hi"])
    with pytest.raises(ParamError):
        g(object())


def test_pytree_tuple(typecheck):
    @typecheck
    def g(x: PyTree[Tuple[int, int]]):
        pass

    g((1, 1))
    g([(1, 1)])
    g([(1, 1), {"a": (1, 1)}])
    with pytest.raises(ParamError):
        g(object())
    with pytest.raises(ParamError):
        g(1)
    with pytest.raises(ParamError):
        g((1, 2, 3))
    with pytest.raises(ParamError):
        g([1, 1])
    with pytest.raises(ParamError):
        g([(1, 1), "hi"])
