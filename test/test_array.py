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

import dataclasses as dc
import sys
from typing import Any, get_args, get_origin, TypeVar, Union

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest


try:
    import torch
except ImportError:
    torch = None

from jaxtyping import (
    AbstractArray,
    AbstractDtype,
    AnnotationError,
    Array,
    ArrayLike,
    Bool,
    Complex,
    Float,
    Float32,
    Int,
    jaxtyped,
    Key,
    PRNGKeyArray,
    Scalar,
    Shaped,
)

from .helpers import ParamError, ReturnError


def test_basic(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def g(x: Shaped[Array, "..."]):
        pass

    g(jnp.array(1.0))


def test_dtypes():
    from jaxtyping import (  # noqa: F401
        Array,
        BFloat16,
        Bool,
        Complex,
        Complex64,
        Complex128,
        Float,
        Float8e4m3b11fnuz,
        Float8e4m3fn,
        Float8e4m3fnuz,
        Float8e5m2,
        Float8e5m2fnuz,
        Float16,
        Float32,
        Float64,
        Inexact,
        Int,
        Int2,
        Int4,
        Int8,
        Int16,
        Int32,
        Int64,
        Num,
        Shaped,
        UInt,
        UInt2,
        UInt4,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
    )

    for key, val in locals().items():
        if issubclass(val, AbstractDtype):
            assert key == val.__name__


def test_numpy_struct_dtype():
    from jaxtyping import make_numpy_struct_dtype

    dtype1 = np.dtype([("first", np.uint8), ("second", bool)])
    Dtype1 = make_numpy_struct_dtype(dtype1, "Dtype1")
    arr = np.array([0, False], dtype=dtype1)

    assert isinstance(arr, Dtype1[np.ndarray, "_"])

    dtype2 = np.dtype([("third", np.uint8), ("second", bool)])
    Dtype2 = make_numpy_struct_dtype(dtype2, "Dtype2")
    assert not isinstance(arr, Dtype2[np.ndarray, "_"])

    dtype3 = np.dtype([("second", bool), ("first", np.uint8)])
    Dtype3 = make_numpy_struct_dtype(dtype3, "Dtype3")
    assert not isinstance(arr, Dtype3[np.ndarray, "_"])


def test_return(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Float[Array, "b c"]) -> Float[Array, "c b"]:
        return jnp.transpose(x)

    g(jr.normal(getkey(), (3, 4)))

    @jaxtyp(typecheck)
    def h(x: Float[Array, "b c"]) -> Float[Array, "b c"]:
        return jnp.transpose(x)

    with pytest.raises(ReturnError):
        h(jr.normal(getkey(), (3, 4)))


def test_two_args(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Shaped[Array, "b c"], y: Shaped[Array, "c d"]):
        return x @ y

    g(jr.normal(getkey(), (3, 4)), jr.normal(getkey(), (4, 5)))
    with pytest.raises(ParamError):
        g(jr.normal(getkey(), (3, 4)), jr.normal(getkey(), (5, 4)))

    @jaxtyp(typecheck)
    def h(x: Shaped[Array, "b c"], y: Shaped[Array, "c d"]) -> Shaped[Array, "b d"]:
        return x @ y

    h(jr.normal(getkey(), (3, 4)), jr.normal(getkey(), (4, 5)))
    with pytest.raises(ParamError):
        h(jr.normal(getkey(), (3, 4)), jr.normal(getkey(), (5, 4)))


def test_any_dtype(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Shaped[Array, "a b"]) -> Shaped[Array, "a b"]:
        return x

    g(jr.normal(getkey(), (3, 4)))
    g(jnp.array([[True, False]]))
    g(jnp.array([[1, 2], [3, 4]], dtype=jnp.int2))
    g(jnp.array([[1, 2], [3, 4]], dtype=jnp.int4))
    g(jnp.array([[1, 2], [3, 4]], dtype=jnp.int8))
    g(jnp.array([[1, 2], [3, 4]], dtype=jnp.uint2))
    g(jnp.array([[1, 2], [3, 4]], dtype=jnp.uint4))
    g(jnp.array([[1, 2], [3, 4]], dtype=jnp.uint16))
    g(jr.normal(getkey(), (3, 4), dtype=jnp.complex64))
    g(jr.normal(getkey(), (3, 4), dtype=jnp.bfloat16))

    with pytest.raises(ParamError):
        g(jr.normal(getkey(), (1,)))


def test_nested_jaxtyped(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Float32[Array, "b c"], transpose: bool) -> Float32[Array, "c b"]:
        return h(x, transpose)

    @jaxtyp(typecheck)
    def h(x: Float32[Array, "c b"], transpose: bool) -> Float32[Array, "b c"]:
        if transpose:
            return jnp.transpose(x)
        else:
            return x

    g(jr.normal(getkey(), (2, 3)), True)
    g(jr.normal(getkey(), (3, 3)), True)
    g(jr.normal(getkey(), (3, 3)), False)
    with pytest.raises(ReturnError):
        g(jr.normal(getkey(), (2, 3)), False)


def test_nested_nojaxtyped(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Float32[Array, "b c"]):
        return h(x)

    @typecheck
    def h(x: Float32[Array, "c b"]):
        return x

    with pytest.raises(ParamError):
        g(jr.normal(getkey(), (2, 3)))


def test_isinstance(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Float32[Array, "b c"]) -> Float32[Array, " z"]:
        y = jnp.transpose(x)
        assert isinstance(y, Float32[Array, "c b"])
        assert not isinstance(
            y, Float32[Array, "b z"]
        )  # z left unbound as b!=c (unless x symmetric, which it isn't)
        out = jr.normal(getkey(), (500,))
        assert isinstance(out, Float32[Array, "z"])  # z now bound
        return out

    g(jr.normal(getkey(), (2, 3)))


def test_fixed(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(
        x: Float32[Array, "4 5 foo"], y: Float32[Array, " foo"]
    ) -> Float32[Array, "4 5"]:
        return x @ y

    a = jr.normal(getkey(), (4, 5, 2))
    b = jr.normal(getkey(), (2,))
    assert g(a, b).shape == (4, 5)

    c = jr.normal(getkey(), (3, 5, 2))
    with pytest.raises(ParamError):
        g(c, b)


def test_anonymous(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Float32[Array, "foo _"], y: Float32[Array, " _"]):
        pass

    a = jr.normal(getkey(), (3, 4))
    b = jr.normal(getkey(), (5,))
    g(a, b)


def test_named_variadic(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(
        x: Float32[Array, "*batch foo"],
        y: Float32[Array, " *batch"],
        z: Float32[Array, " foo"],
    ):
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

    @jaxtyp(typecheck)
    def h(x: Float32[Array, " foo *batch"], y: Float32[Array, " foo *batch bar"]):
        pass

    a = jr.normal(getkey(), (4,))
    b = jr.normal(getkey(), (4, 3))
    c = jr.normal(getkey(), (3, 4))
    h(a, b)
    with pytest.raises(ParamError):
        h(a, c)
    with pytest.raises(ParamError):
        h(b, c)


def test_anonymous_variadic(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Float32[Array, "... foo"], y: Float32[Array, " foo"]):
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


def test_broadcast_fixed(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Float32[Array, "#4"]):
        pass

    g(jr.normal(getkey(), (4,)))
    g(jr.normal(getkey(), (1,)))

    with pytest.raises(ParamError):
        g(jr.normal(getkey(), (3,)))


def test_broadcast_named(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Float32[Array, " #foo"], y: Float32[Array, " #foo"]):
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


def test_broadcast_variadic_named(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def g(x: Float32[Array, " *#foo"], y: Float32[Array, " *#foo"]):
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


def test_variadic_mixed_broadcast(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def f(x: Float[Array, " *foo"], y: Float[Array, " #*foo"]):
        pass

    a = jr.normal(getkey(), (3, 4))
    b = jr.normal(getkey(), (5,))
    with pytest.raises(ParamError):
        f(a, b)

    c = jr.normal(getkey(), (7, 3, 2))
    d = jr.normal(getkey(), (1, 2))
    f(c, d)


def test_variadic_mixed_broadcast2(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def f(x: Float[Array, " *#foo"], y: Float[Array, " *foo"]):
        pass

    a = jr.normal(getkey(), (3, 4))
    b = jr.normal(getkey(), (5,))
    with pytest.raises(ParamError):
        f(a, b)

    c = jr.normal(getkey(), (1, 2))
    d = jr.normal(getkey(), (7, 3, 2))
    f(c, d)


def test_variadic_mixed_broadcast3(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def f(
        x: Float[Array, "*B L D"],
        *,
        y: Float[Array, "*#B J d"],
        z: Bool[Array, "*B L J"],
    ) -> Float[Array, "*B L D"]:
        return x

    x = jr.normal(getkey(), (2, 7, 3, 2, 2))
    y = jr.bernoulli(getkey(), shape=(2, 7, 3, 2, 2))
    z = jr.normal(getkey(), (2, 7, 1, 2, 2))
    f(x, y=z, z=y)


def test_no_commas():
    with pytest.raises(ValueError):
        Float32[Array, "foo, bar"]


def test_symbolic(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def make_slice(x: Float32[Array, " dim"]) -> Float32[Array, " dim-1"]:
        return x[1:]

    @jaxtyp(typecheck)
    def cat(x: Float32[Array, " dim"]) -> Float32[Array, " 2*dim"]:
        return jnp.concatenate([x, x])

    @jaxtyp(typecheck)
    def bad_make_slice(x: Float32[Array, " dim"]) -> Float32[Array, " dim-1"]:
        return x

    @jaxtyp(typecheck)
    def bad_cat(x: Float32[Array, " dim"]) -> Float32[Array, " 2*dim"]:
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


def test_incomplete_symbolic(jaxtyp, typecheck, getkey):
    @jaxtyp(typecheck)
    def foo(x: Float32[Array, " 2*dim"]):
        pass

    x = jr.normal(getkey(), (4,))
    with pytest.raises(AnnotationError):
        foo(x)


def test_deferred_symbolic_good(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def foo(dim: int, fill: Float[Array, ""]) -> Float[Array, " {dim}"]:
        return jnp.full((dim,), fill)

    class A:
        size = 5

        @jaxtyp(typecheck)
        def bar(self, fill: Float[Array, ""]) -> Float[Array, " {self.size}"]:
            return jnp.full((self.size,), fill)

    foo(3, jnp.array(0.0))
    A().bar(jnp.array(0.0))


def test_deferred_symbolic_bad(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def foo(dim: int, fill: Float[Array, ""]) -> Float[Array, " {dim-1}"]:
        return jnp.full((dim,), fill)

    class A:
        size = 5

        @jaxtyp(typecheck)
        def bar(self, fill: Float[Array, ""]) -> Float[Array, " {self.size}-1"]:
            return jnp.full((self.size,), fill)

    with pytest.raises(ReturnError):
        foo(3, jnp.array(0.0))

    with pytest.raises(ReturnError):
        A().bar(jnp.array(0.0))


def test_deferred_symbolic_dataclass(typecheck):
    @jaxtyped(typechecker=typecheck)
    @dc.dataclass
    class A:
        value: int
        array: Float[Array, " {value}"]

    A(3, jnp.zeros(3))

    with pytest.raises(ParamError):
        A(3, jnp.zeros(4))


def _to_set(x) -> set[tuple]:
    return {
        (xi.index_variadic, xi.dims, xi.array_type, xi.dtypes, xi.dim_str)
        if issubclass(xi, AbstractArray)
        else xi
        for xi in x
    }


def test_arraylike(typecheck, getkey):
    floatlike1 = Float32[ArrayLike, ""]
    floatlike2 = Float[ArrayLike, ""]
    floatlike3 = Float32[ArrayLike, "4"]

    assert get_origin(floatlike1) is Union
    assert get_origin(floatlike2) is Union
    assert get_origin(floatlike3) is Union
    assert _to_set(get_args(floatlike1)) == _to_set(
        [
            Float32[Array, ""],
            Float32[np.ndarray, ""],
            Float32[np.number, ""],
            float,
        ]
    )
    assert _to_set(get_args(floatlike2)) == _to_set(
        [
            Float[Array, ""],
            Float[np.ndarray, ""],
            Float[np.number, ""],
            float,
        ]
    )
    assert _to_set(get_args(floatlike3)) == _to_set(
        [
            Float32[Array, "4"],
            Float32[np.ndarray, "4"],
        ]
    )

    shaped1 = Shaped[ArrayLike, ""]
    shaped2 = Shaped[ArrayLike, "4"]
    assert get_origin(shaped1) is Union
    assert get_origin(shaped2) is Union
    assert _to_set(get_args(shaped1)) == _to_set(
        [
            Shaped[Array, ""],
            Shaped[np.ndarray, ""],
            Shaped[np.bool_, ""],
            Shaped[np.number, ""],
            bool,
            int,
            float,
            complex,
        ]
    )
    assert _to_set(get_args(shaped2)) == _to_set(
        [
            Shaped[Array, "4"],
            Shaped[np.ndarray, "4"],
        ]
    )


def test_ignored_names():
    x = Float[np.ndarray, "foo=4"]

    assert isinstance(np.zeros(4), x)
    assert not isinstance(np.zeros(5), x)
    assert not isinstance(np.zeros((4, 5)), x)

    y = Float[np.ndarray, "bar qux foo=bar+qux"]

    assert isinstance(np.zeros((2, 3, 5)), y)
    assert not isinstance(np.zeros((2, 3, 6)), y)

    z = Float[np.ndarray, "bar #foo=bar"]

    assert isinstance(np.zeros((3, 3)), z)
    assert isinstance(np.zeros((3, 1)), z)
    assert not isinstance(np.zeros((3, 4)), z)

    # Weird but legal
    w = Float[np.ndarray, "bar foo=#bar"]

    assert isinstance(np.zeros((3, 3)), w)
    assert isinstance(np.zeros((3, 1)), w)
    assert not isinstance(np.zeros((3, 4)), w)


def test_symbolic_functions():
    x = Float[np.ndarray, "foo bar min(foo,bar)"]

    assert isinstance(np.zeros((2, 3, 2)), x)
    assert isinstance(np.zeros((3, 2, 2)), x)
    assert not isinstance(np.zeros((3, 2, 4)), x)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires Python 3.10")
def test_py310_unions():
    x = np.zeros(3)
    y = Shaped[Array | np.ndarray, "_"]
    assert isinstance(x, get_args(y))


def test_key(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: PRNGKeyArray):
        pass

    f(jr.key(0))
    f(jr.PRNGKey(0))

    with pytest.raises(ParamError):
        f(object())
    with pytest.raises(ParamError):
        f(1)
    with pytest.raises(ParamError):
        f(jnp.array(3))
    with pytest.raises(ParamError):
        f(jnp.array(3.0))


def test_key_dtype(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f1(x: Key[Array, ""]):
        pass

    @jaxtyp(typecheck)
    def f2(x: Key[Scalar, ""]):
        pass

    for f in (f1, f2):
        f(jr.key(0))

        with pytest.raises(ParamError):
            f(jr.PRNGKey(0))
        with pytest.raises(ParamError):
            f(object())
        with pytest.raises(ParamError):
            f(1)
        with pytest.raises(ParamError):
            f(jnp.array(3))
        with pytest.raises(ParamError):
            f(jnp.array(3.0))


def test_extension(jaxtyp, typecheck, getkey):
    X = Shaped[Array, "a b"]
    Y = Shaped[X, "c d"]
    Z = Shaped[Array, "c d a b"]
    assert str(Z) == str(Y)

    X = Float[Array, "a"]
    Y = Float[X, "b"]

    @jaxtyp(typecheck)
    def f(a: X, b: Y): ...

    a = jr.normal(getkey(), (3, 4))
    b = jr.normal(getkey(), (4,))
    c = jr.normal(getkey(), (3,))

    f(b, a)
    with pytest.raises(ParamError):
        f(c, a)
    with pytest.raises(ParamError):
        f(a, a)

    @typecheck
    def g(a: Shaped[PRNGKeyArray, "2"]): ...

    with pytest.raises(ParamError):
        g(jr.PRNGKey(0))
    g(jr.split(jr.PRNGKey(0)))
    with pytest.raises(ParamError):
        g(jr.split(jr.PRNGKey(0), 3))


def test_scalar_variadic_dim():
    assert Float[float, "..."] is float
    assert Float[float, "#*shape"] is float

    # This one is a bit weird -- it should really also assert that shape==(), but we
    # don't implement that.
    assert Float[float, "*shape"] is float


def test_scalar_dtype_mismatch():
    with pytest.raises(ValueError):
        Float[bool, "..."]


def test_custom_array(jaxtyp, typecheck):
    class MyArray1:
        @property
        def dtype(self):
            return "foo"

        @property
        def shape(self):
            return (3,)

    class MyArray2:
        @property
        def dtype(self):
            return "bar"

        @property
        def shape(self):
            return (3,)

    class MyArray3:
        @property
        def dtype(self):
            return "foo"

        @property
        def shape(self):
            return (4,)

    class FooDtype(AbstractDtype):
        dtypes = ["foo"]

    @jaxtyp(typecheck)
    def f(x: FooDtype[MyArray1, "3"]):
        pass

    f(MyArray1())
    with pytest.raises(ParamError):
        f(MyArray2())
    with pytest.raises(ParamError):
        f(MyArray3())

    @jaxtyp(typecheck)
    def g(x: FooDtype[MyArray1, "3"], y: FooDtype[MyArray1, "4"]):
        pass

    with pytest.raises(ParamError):
        g(MyArray1(), MyArray1())

    @jaxtyp(typecheck)
    def h(x: FooDtype[MyArray1, "3"], y: FooDtype[MyArray3, "4"]):
        pass

    with pytest.raises(ParamError):
        g(MyArray1(), MyArray3())


@pytest.mark.parametrize(
    "array_type", [Any, TypeVar("T"), TypeVar("T", bound=ArrayLike)]
)
def test_any(array_type, jaxtyp, typecheck):
    class DuckArray1:
        @property
        def shape(self):
            return 3, 4

        @property
        def dtype(self):
            return np.array([], dtype=np.float32).dtype

    class DuckArray2:
        @property
        def shape(self):
            return 3, 4, 5

        @property
        def dtype(self):
            return np.array([], dtype=np.float32).dtype

    class DuckArray3:
        @property
        def shape(self):
            return 3, 4

        @property
        def dtype(self):
            return np.array([], dtype=np.int32).dtype

    @jaxtyp(typecheck)
    def f(x: Float[array_type, "foo bar"]):
        del x

    f(np.arange(12.0).reshape(3, 4))
    f(jnp.arange(12.0).reshape(3, 4))
    if isinstance(array_type, TypeVar) and array_type.__bound__ is ArrayLike:
        with pytest.raises(ParamError):
            f(DuckArray1())
    else:
        f(DuckArray1())

    # Wrong shape
    with pytest.raises(ParamError):
        f(np.arange(12.0).reshape(3, 2, 2))
    with pytest.raises(ParamError):
        f(jnp.arange(12.0).reshape(3, 2, 2))
    with pytest.raises(ParamError):
        f(DuckArray2())

    # Wrong dtype
    with pytest.raises(ParamError):
        f(np.arange(12).reshape(3, 4))
    with pytest.raises(ParamError):
        f(jnp.arange(12).reshape(3, 4))
    with pytest.raises(ParamError):
        f(DuckArray3())

    # Not an array
    with pytest.raises(ParamError):
        f(1)


def test_non_instantiation():
    with pytest.raises(RuntimeError, match="cannot be instantiated"):
        Float[Array, ""]()  # pyright: ignore[reportCallIssue]


def test_numpy_arraylike_work_around_pep484_badness():
    # PEP484 makes the terrible decision to treat `float = int | float` and
    # `complex = int | float| complex`.
    # In practice neither beartype nor jaxtyping respect this at runtime.
    #
    # Unfortunately somewhere between `numpy==2.0.2` and `numpy==2.3.3`, then
    # `np.typing.ArrayLike` uses this (mis)feature by including  `complex` and nothing
    # else.
    # https://github.com/numpy/numpy/commit/1041f940f91660c91770679c60f6e63539581c72
    #
    # We special-case this to make it work.
    assert isinstance(True, Bool[np.typing.ArrayLike, ""])
    assert isinstance(1, Int[np.typing.ArrayLike, ""])
    assert isinstance(1.0, Float[np.typing.ArrayLike, ""])
    assert isinstance(1 + 1j, Complex[np.typing.ArrayLike, ""])

    assert isinstance(True, Bool[np.typing.ArrayLike, "..."])
    assert isinstance(1, Int[np.typing.ArrayLike, "..."])
    assert isinstance(1.0, Float[np.typing.ArrayLike, "..."])
    assert isinstance(1 + 1j, Complex[np.typing.ArrayLike, "..."])

    assert isinstance(True, Bool[np.typing.ArrayLike, "*foo"])
    assert isinstance(1, Int[np.typing.ArrayLike, "*foo"])
    assert isinstance(1.0, Float[np.typing.ArrayLike, "*foo"])
    assert isinstance(1 + 1j, Complex[np.typing.ArrayLike, "*foo"])


def test_uniontype(typeguard_or_skip):
    # https://github.com/patrick-kidger/jaxtyping/issues/349

    @typeguard_or_skip.typechecked
    def f(x: Float[np.ndarray, ""] | None):
        pass

    with pytest.raises(ParamError):
        f(np.array([1, 2]))
