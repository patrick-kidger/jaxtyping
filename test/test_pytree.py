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

from typing import NamedTuple, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import jaxtyping
from jaxtyping import AnnotationError, Array, Float, PyTree

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
    g(jax.tree.map(lambda _: 1, make_mlp(getkey())))

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


def test_pytree_array(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def g(x: PyTree[Float[jnp.ndarray, "..."]]):
        pass

    g(jnp.array(1.0))
    g([jnp.array(1.0), jnp.array(1.0), {"a": jnp.array(1.0)}])
    with pytest.raises(ParamError):
        g(jnp.array(1))
    with pytest.raises(ParamError):
        g(1.0)


def test_pytree_shaped_array(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
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


def test_pytree_namedtuple(typecheck):
    class CustomNamedTuple(NamedTuple):
        x: Float[jnp.ndarray, "a b"]
        y: Float[jnp.ndarray, "b c"]

    class OtherCustomNamedTuple(NamedTuple):
        x: Float[jnp.ndarray, "a b"]
        y: Float[jnp.ndarray, "b c"]

    @typecheck
    def g(x: PyTree[CustomNamedTuple]):
        ...

    g(
        CustomNamedTuple(
            x=jax.random.normal(jax.random.PRNGKey(42), (3, 2)),
            y=jax.random.normal(jax.random.PRNGKey(420), (2, 5)),
        )
    )
    with pytest.raises(ParamError):
        g(object())
    with pytest.raises(ParamError):
        g(
            OtherCustomNamedTuple(
                x=jax.random.normal(jax.random.PRNGKey(42), (3, 2)),
                y=jax.random.normal(jax.random.PRNGKey(420), (2, 5)),
            )
        )


def test_subclass_pytree():
    x = PyTree
    y = PyTree[int]
    assert issubclass(x, PyTree)
    assert issubclass(y, PyTree)
    assert not issubclass(int, PyTree)


def test_structure_match(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: PyTree[int, " T"], y: PyTree[str, " T"]):
        pass

    f(1, "hi")
    f((3, 4, {"a": 5}), ("a", "b", {"a": "c"}))

    with pytest.raises(ParamError):
        f(1, ("hi",))


def test_structure_prefix(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: PyTree[int, " T"], y: PyTree[str, "T ..."]):
        pass

    f(1, "hi")
    f((3, 4, {"a": 5}), ("a", "b", {"a": "c"}))
    f(1, ("hi",))
    f((1, 2), ({"a": "hi"}, {"a": "bye"}))
    f((1, 2), ({"a": "hi"}, {"not-a": "bye"}))

    with pytest.raises(ParamError):
        f((1, 2), ({"a": "hi"}, {"a": "bye"}, {"a": "oh-no"}))

    with pytest.raises(ParamError):
        f((3, 4, 5), {"a": ("hi", "bye")})


def test_structure_suffix(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: PyTree[int, " T"], y: PyTree[str, "... T"]):
        pass

    f(1, "hi")
    f((3, 4, {"a": 5}), ("a", "b", {"a": "c"}))
    f(1, ("hi",))

    with pytest.raises(ParamError):
        f((3, 4), {"a": (1, 2)})

    with pytest.raises(ParamError):
        f((3, 4, 5), {"a": ("hi", "bye")})


def test_structure_compose(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: PyTree[int, " T"], y: PyTree[int, " S"], z: PyTree[str, "S T"]):
        pass

    f(1, 2, "hi")
    f((1, 2), 2, ("a", "b"))

    with pytest.raises(ParamError):
        f((1, 2), 2, (1, 2))

    f((1, 2), {"a": 3}, {"a": ("hi", "bye")})

    with pytest.raises(ParamError):
        f((1, 2), {"a": 3}, ({"a": "hi"}, {"a": "bye"}))

    @jaxtyp(typecheck)
    def g(x: PyTree[int, " T"], y: PyTree[int, " S"], z: PyTree[str, "T S"]):
        pass

    with pytest.raises(ParamError):
        g((1, 2), {"a": 3}, {"a": ("hi", "bye")})

    g((1, 2), {"a": 3}, ({"a": "hi"}, {"a": "bye"}))


@pytest.mark.parametrize("variadic", (False, True))
def test_treepath_dependence_function(variadic, jaxtyp, typecheck, getkey):
    if variadic:
        jtshape = "*?foo"
        shape = (2, 3)
    else:
        jtshape = "?foo"
        shape = (4,)

    @jaxtyp(typecheck)
    def f(
        x: PyTree[Float[Array, jtshape], " T"], y: PyTree[Float[Array, jtshape], " T"]
    ):
        pass

    x1 = jr.normal(getkey(), shape)
    y1 = jr.normal(getkey(), shape)
    x2 = jr.normal(getkey(), (5,))
    y2 = jr.normal(getkey(), (5,))
    f(x1, y1)
    f((x1, x2), (y1, y2))

    with pytest.raises(ParamError):
        f(x1, y2)

    with pytest.raises(ParamError):
        f((x1, x2), (y2, y1))


@pytest.mark.parametrize("variadic", (False, True))
def test_treepath_dependence_dataclass(variadic, typecheck, getkey):
    if variadic:
        jtshape = "*?foo"
        shape = (2, 3)
    else:
        jtshape = "?foo"
        shape = (4,)

    @jaxtyping.jaxtyped(typechecker=typecheck)
    class A(eqx.Module):
        x: PyTree[Float[Array, jtshape], " T"]
        y: PyTree[Float[Array, jtshape], " T"]

    x1 = jr.normal(getkey(), shape)
    y1 = jr.normal(getkey(), shape)
    x2 = jr.normal(getkey(), (5,))
    y2 = jr.normal(getkey(), (5,))
    A(x1, y1)
    A((x1, x2), (y1, y2))

    with pytest.raises(ParamError):
        A(x1, y2)

    with pytest.raises(ParamError):
        A((x1, x2), (y2, y1))


def test_treepath_dependence_missing_structure_annotation(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def f(x: PyTree[Float[Array, "?foo"], " T"], y: PyTree[Float[Array, "?foo"]]):
        pass

    x1 = jr.normal(getkey(), (2,))
    y1 = jr.normal(getkey(), (2,))
    with pytest.raises(AnnotationError, match="except when contained with structured"):
        f(x1, y1)


def test_treepath_dependence_multiple_structure_annotation(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def f(x: PyTree[PyTree[Float[Array, "?foo"], " S"], " T"]):
        pass

    x1 = jr.normal(getkey(), (2,))
    with pytest.raises(AnnotationError, match="ambiguous which PyTree"):
        f(x1)
