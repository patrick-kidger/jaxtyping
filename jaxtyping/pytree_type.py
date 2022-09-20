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
from typing import Generic, TYPE_CHECKING, TypeVar
from typing_extensions import Protocol

import jax
import typeguard


_T = TypeVar("_T")


class _FakePyTree(Generic[_T]):
    pass


_FakePyTree.__name__ = "PyTree"
_FakePyTree.__qualname__ = "PyTree"
_FakePyTree.__module__ = "builtins"
# Can't do type("PyTree", (Generic[_T],), {}) because dynamic subclassing of typeforms
# isn't allowed.
# Can't do types.new_class("PyTree", (Generic[_T],), {}) because that has __module__
# "types", e.g. we get types.PyTree[int].


class _MetaPyTree(type):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("PyTree cannot be instantiated")

    def __instancecheck__(cls, obj):
        return True

    @ft.lru_cache(maxsize=None)
    def __getitem__(cls, item):
        name = str(_FakePyTree[item])
        out = _MetaSubscriptPyTree(name, (), {"leaftype": item})
        if getattr(typing, "GENERATING_DOCUMENTATION", False):
            out.__module__ = "builtins"
        else:
            out.__module__ = "jaxtyping"
        return out


class _MetaSubscriptPyTree(type):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("PyTree cannot be instantiated")

    def __instancecheck__(cls, obj):
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
            except TypeError:
                return False
            else:
                return True

        leaves = jax.tree_leaves(obj, is_leaf=is_leaftype)
        return all(map(is_leaftype, leaves))


if TYPE_CHECKING:
    # Work around pytype bug #1288
    # pytype: skip-file
    class PyTree(Protocol[_T]):
        pass

else:
    PyTree = _MetaPyTree("PyTree", (), {})
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        PyTree.__module__ = "builtins"
    else:
        PyTree.__module__ = "jaxtyping"
# Can't do `class PyTree(Generic[_T]): ...` because we need to override the
# instancecheck for PyTree[foo], but subclassing
# `type(Generic[int])`, i.e. `typing._GenericAlias` is disallowed.
