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

import jax.numpy as jnp
import jax.random as jr
import pytest

from jaxtyping import Array, f, f32, jaxtyped

from .helpers import ParamError, ReturnError


def test_basic(typecheck):
    @jaxtyped
    @typecheck
    def g(x: Array["..."]):
        pass

    g(jnp.array(1.0))


def test_return(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f["b c"]) -> f["c b"]:
        return jnp.transpose(x)

    g(jr.normal(getkey(), (3, 4)))

    @jaxtyped
    @typecheck
    def h(x: f["b c"]) -> f["b c"]:
        return jnp.transpose(x)

    with pytest.raises(ReturnError):
        h(jr.normal(getkey(), (3, 4)))


def test_two_args(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: Array["b c"], y: Array["c d"]):
        return x @ y

    g(jr.normal(getkey(), (3, 4)), jr.normal(getkey(), (4, 5)))
    with pytest.raises(ParamError):
        g(jr.normal(getkey(), (3, 4)), jr.normal(getkey(), (5, 4)))

    @jaxtyped
    @typecheck
    def h(x: Array["b c"], y: Array["c d"]) -> Array["b d"]:
        return x @ y

    h(jr.normal(getkey(), (3, 4)), jr.normal(getkey(), (4, 5)))
    with pytest.raises(ParamError):
        h(jr.normal(getkey(), (3, 4)), jr.normal(getkey(), (5, 4)))


def test_any_dtype(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: Array["a b"]) -> Array["a b"]:
        return x

    g(jr.normal(getkey(), (3, 4)))
    g(jnp.array([[True, False]]))
    g(jnp.array([[1, 2], [3, 4]], dtype=jnp.int8))
    g(jnp.array([[1, 2], [3, 4]], dtype=jnp.uint16))
    g(jr.normal(getkey(), (3, 4), dtype=jnp.complex128))
    g(jr.normal(getkey(), (3, 4), dtype=jnp.bfloat16))

    with pytest.raises(ParamError):
        g(jr.normal(getkey(), (1,)))


def test_nested_jaxtyped(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f32["b c"], transpose: bool) -> f32["c b"]:
        return h(x, transpose)

    @jaxtyped
    @typecheck
    def h(x: f32["c b"], transpose: bool) -> f32["b c"]:
        if transpose:
            return jnp.transpose(x)
        else:
            return x

    g(jr.normal(getkey(), (2, 3)), True)
    g(jr.normal(getkey(), (3, 3)), True)
    g(jr.normal(getkey(), (3, 3)), False)
    with pytest.raises(ReturnError):
        g(jr.normal(getkey(), (2, 3)), False)


def test_nested_nojaxtyped(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f32["b c"]):
        return h(x)

    @typecheck
    def h(x: f32["c b"]):
        return x

    with pytest.raises(ParamError):
        g(jr.normal(getkey(), (2, 3)))


def test_isinstance(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f32["b c"]) -> f32[" z"]:
        y = jnp.transpose(x)
        assert isinstance(y, f32["c b"])
        assert not isinstance(
            y, f32["b z"]
        )  # z left unbound as b!=c (unless x symmetric, which it isn't)
        out = jr.normal(getkey(), (500,))
        assert isinstance(out, f32["z"])  # z now bound
        return out

    g(jr.normal(getkey(), (2, 3)))


def test_fixed(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f32["4 5 foo"], y: f32[" foo"]) -> f32["4 5"]:
        return x @ y

    a = jr.normal(getkey(), (4, 5, 2))
    b = jr.normal(getkey(), (2,))
    assert g(a, b).shape == (4, 5)

    c = jr.normal(getkey(), (3, 5, 2))
    with pytest.raises(ParamError):
        g(c, b)


def test_anonymous(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f32["foo _"], y: f32[" _"]):
        pass

    a = jr.normal(getkey(), (3, 4))
    b = jr.normal(getkey(), (5,))
    g(a, b)


def test_named_variadic(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f32["*batch foo"], y: f32[" *batch"], z: f32[" foo"]):
        pass

    c = jr.normal(getkey(), (5,))

    a1 = jr.normal(getkey(), (5,))
    b1 = jr.normal(getkey(), ())
    g(a1, b1, c)

    a2 = jr.normal(getkey(), (3, 5))
    b2 = jr.normal(getkey(), (3,))
    g(a2, b2, c)

    with pytest.raises(ParamError):
        g(a1, b2, c)
    with pytest.raises(ParamError):
        g(a2, b1, c)

    @jaxtyped
    @typecheck
    def h(x: f32[" foo *batch"], y: f32[" foo *batch bar"]):
        pass

    a = jr.normal(getkey(), (4,))
    b = jr.normal(getkey(), (4, 3))
    c = jr.normal(getkey(), (3, 4))
    h(a, b)
    with pytest.raises(ParamError):
        h(a, c)
    with pytest.raises(ParamError):
        h(b, c)


def test_anonymous_variadic(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f32["... foo"], y: f32[" foo"]):
        pass

    a1 = jr.normal(getkey(), (5,))
    a2 = jr.normal(getkey(), (3, 5))
    a3 = jr.normal(getkey(), (3, 4, 5))
    b = jr.normal(getkey(), (5,))
    c = jr.normal(getkey(), (1,))
    g(a1, b)
    g(a2, b)
    g(a3, b)
    with pytest.raises(ParamError):
        g(a1, c)
    with pytest.raises(ParamError):
        g(a2, c)
    with pytest.raises(ParamError):
        g(a3, c)


def test_broadcast_fixed(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f32["#4"]):
        pass

    g(jr.normal(getkey(), (4,)))
    g(jr.normal(getkey(), (1,)))

    with pytest.raises(ParamError):
        g(jr.normal(getkey(), (3,)))


def test_broadcast_named(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f32[" #foo"], y: f32[" #foo"]):
        pass

    a = jr.normal(getkey(), (3,))
    b = jr.normal(getkey(), (4,))
    c = jr.normal(getkey(), (1,))

    g(a, a)
    g(b, b)
    g(c, c)
    g(a, c)
    g(b, c)
    g(c, a)
    g(c, b)

    with pytest.raises(ParamError):
        g(a, b)
    with pytest.raises(ParamError):
        g(b, a)


def test_broadcast_variadic_named(typecheck, getkey):
    @jaxtyped
    @typecheck
    def g(x: f32[" *#foo"], y: f32[" *#foo"]):
        pass

    a = jr.normal(getkey(), (3,))
    b = jr.normal(getkey(), (4,))
    c = jr.normal(getkey(), (4, 4))
    d = jr.normal(getkey(), (5, 6))

    j = jr.normal(getkey(), (1,))
    k = jr.normal(getkey(), (1, 4))
    l = jr.normal(getkey(), (5, 1))  # noqa: E741
    m = jr.normal(getkey(), (1, 1))
    n = jr.normal(getkey(), (2, 1))
    o = jr.normal(getkey(), (1, 6))

    g(a, a)
    g(b, b)
    g(c, c)
    g(d, d)
    g(b, c)
    with pytest.raises(ParamError):
        g(a, b)
    with pytest.raises(ParamError):
        g(a, c)
    with pytest.raises(ParamError):
        g(a, b)
    with pytest.raises(ParamError):
        g(d, b)

    g(a, j)
    g(b, j)
    g(c, j)
    g(d, j)
    g(b, k)
    g(c, k)
    with pytest.raises(ParamError):
        g(d, k)
    with pytest.raises(ParamError):
        g(c, l)
    g(d, l)
    g(a, m)
    g(c, m)
    g(d, m)
    g(a, n)
    g(b, n)
    with pytest.raises(ParamError):
        g(c, n)
    with pytest.raises(ParamError):
        g(d, n)
    g(o, d)
    with pytest.raises(ParamError):
        g(o, c)
    with pytest.raises(ParamError):
        g(o, a)


def test_no_commas():
    with pytest.raises(ValueError):
        f32["foo, bar"]


def test_symbolic(typecheck, getkey):
    @jaxtyped
    @typecheck
    def make_slice(x: f32[" dim"]) -> f32[" dim-1"]:
        return x[1:]

    @jaxtyped
    @typecheck
    def cat(x: f32[" dim"]) -> f32[" 2*dim"]:
        return jnp.concatenate([x, x])

    @jaxtyped
    @typecheck
    def bad_make_slice(x: f32[" dim"]) -> f32[" dim-1"]:
        return x

    @jaxtyped
    @typecheck
    def bad_cat(x: f32[" dim"]) -> f32[" 2*dim"]:
        return jnp.concatenate([x, x, x])

    x = jr.normal(getkey(), (5,))
    assert make_slice(x).shape == (4,)
    assert cat(x).shape == (10,)

    y = jr.normal(getkey(), (3, 4))
    with pytest.raises(ParamError):
        make_slice(y)
    with pytest.raises(ParamError):
        cat(y)

    with pytest.raises(ReturnError):
        bad_make_slice(x)
    with pytest.raises(ReturnError):
        bad_cat(x)


def test_incomplete_symbolic(typecheck, getkey):
    @jaxtyped
    @typecheck
    def foo(x: f32[" 2*dim"]):
        pass

    x = jr.normal(getkey(), (4,))
    with pytest.raises(NameError):
        foo(x)
