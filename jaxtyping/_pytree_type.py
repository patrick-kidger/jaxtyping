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

import functools as ft
import typing
from typing import Generic, TypeVar

import jax.tree_util as jtu
import typeguard


_T = TypeVar("_T")


class _FakePyTree(Generic[_T]):
    pass


_FakePyTree.__name__ = "PyTree"
_FakePyTree.__qualname__ = "PyTree"
_FakePyTree.__module__ = "builtins"


class _MetaPyTree(type):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("PyTree cannot be instantiated")

    def __instancecheck__(cls, obj):
        if not hasattr(cls, "leaftype"):
            return True  # Just `isinstance(x, PyTree)`

        # We could use `isinstance` here but that would fail for more complicated
        # types, e.g. PyTree[Tuple[int]]. So at least internally we make a particular
        # choice of typechecker.
        #
        # Deliberately not using @jaxtyped so that we share the same `memo` as whatever
        # dynamic context we're currently in.
        @typeguard.typechecked
        def accepts_leaftype(x: cls.leaftype):
            pass

        def is_leaftype(x):
            try:
                accepts_leaftype(x)
            except _TypeCheckError:
                return False
            else:
                return True

        leaves = jtu.tree_leaves(obj, is_leaf=is_leaftype)
        return all(map(is_leaftype, leaves))

    # Can't return a generic (e.g. _FakePyTree[item]) because generic aliases don't do
    # the custom __instancecheck__ that we want.
    # We can't add that __instancecheck__  via subclassing, e.g.
    # type("PyTree", (Generic[_T],), {}), because dynamic subclassing of typeforms
    # isn't allowed.
    # Likewise we can't do types.new_class("PyTree", (Generic[_T],), {}) because that
    # has __module__ "types", e.g. we get types.PyTree[int].
    @ft.lru_cache(maxsize=None)
    def __getitem__(cls, item):
        name = str(_FakePyTree[item])

        class X(PyTree):
            leaftype = item

        X.__name__ = name
        X.__qualname__ = name
        if getattr(typing, "GENERATING_DOCUMENTATION", False):
            X.__module__ = "builtins"
        else:
            X.__module__ = "jaxtyping"
        return X


try:
    # new typeguard
    _TypeCheckError = (TypeError, typeguard.TypeCheckError)
except AttributeError:
    # old typeguard
    _TypeCheckError = TypeError


# Can't do `class PyTree(Generic[_T]): ...` because we need to override the
# instancecheck for PyTree[foo], but subclassing
# `type(Generic[int])`, i.e. `typing._GenericAlias` is disallowed.
PyTree = _MetaPyTree("PyTree", (), {})
if getattr(typing, "GENERATING_DOCUMENTATION", False):
    PyTree.__module__ = "builtins"
else:
    PyTree.__module__ = "jaxtyping"
PyTree.__doc__ = """Represents a PyTree.

Each PyTree is denoted by a type `PyTree[LeafType]`, such as `PyTree[int]` or
`PyTree[Union[str, Float32[Array, "b c"]]]`.

You can leave off the `[...]`, in which case `PyTree` is simply a suggestively-named
alternative to `Any`.
([By definition all types are PyTrees.](https://jax.readthedocs.io/en/latest/pytrees.html))
"""  # noqa: E501
