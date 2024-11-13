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
from typing import Any, Generic, TypeVar

import jax.tree_util as jtu

from ._errors import AnnotationError
from ._storage import (
    clear_treeflatten_memo,
    clear_treepath_memo,
    get_shape_memo,
    set_shape_memo,
    set_treeflatten_memo,
    set_treepath_memo,
)


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
        # Handle beartype doing `isinstance(None, hint)` to check if
        # is `instance`able.
        if obj is None:
            return True

        single_memo, variadic_memo, pytree_memo, arg_memo = get_shape_memo()
        single_memo_bak = single_memo.copy()
        variadic_memo_bak = variadic_memo.copy()
        pytree_memo_bak = pytree_memo.copy()
        arg_memo_bak = arg_memo.copy()
        try:
            out = cls._check(obj, pytree_memo)
        except Exception:
            set_shape_memo(
                single_memo_bak, variadic_memo_bak, pytree_memo_bak, arg_memo_bak
            )
            raise
        if out:
            return True
        else:
            set_shape_memo(
                single_memo_bak, variadic_memo_bak, pytree_memo_bak, arg_memo_bak
            )
            return False

    def _check(cls, obj, pytree_memo):
        if cls.leaftype is Any:

            def is_flatten_leaftype(x):
                return False

            def is_check_leaftype(x):
                return True

        else:
            # We could use `isinstance` here but that would fail for more complicated
            # types, e.g. PyTree[tuple[int]]. So at least internally we make a
            # particular choice of typechecker.
            #
            # Deliberately not using @jaxtyped so that we share the same `memo` as
            # whatever dynamic context we're currently in.
            from ._typeguard import typechecked

            @typechecked
            def accepts_leaftype(x: cls.leaftype):
                pass

            def is_leaftype(x):
                try:
                    accepts_leaftype(x)
                except TypeError:
                    return False
                else:
                    return True

            is_flatten_leaftype = is_check_leaftype = is_leaftype

        set_treeflatten_memo()
        try:
            leaves, structure = jtu.tree_flatten(obj, is_leaf=is_flatten_leaftype)
        finally:
            clear_treeflatten_memo()
        if cls.structure is not None:
            if cls.structure.isidentifier():
                try:
                    prev_structure = pytree_memo[cls.structure]
                except KeyError:
                    pytree_memo[cls.structure] = structure
                else:
                    if prev_structure != structure:
                        return False
            else:
                named_pytree = 0
                pieces = cls.structure.split()
                if pieces[0] == "...":
                    pieces = pieces[1:]
                    prefix = False
                    suffix = True
                elif pieces[-1] == "...":
                    pieces = pieces[:-1]
                    prefix = True
                    suffix = False
                else:
                    prefix = False
                    suffix = False
                for identifier in pieces:
                    try:
                        prev_structure = pytree_memo[identifier]
                    except KeyError as e:
                        raise AnnotationError(
                            f"Cannot process composite structure '{cls.structure}' "
                            f"as the structure name {identifier} has not been seen "
                            "before."
                        ) from e
                    # Not using `PyTreeDef.compose` due to JAX bug #18218.
                    prev_pytree = jtu.tree_unflatten(
                        prev_structure, [0] * prev_structure.num_leaves
                    )
                    named_pytree = jtu.tree_map(lambda _: prev_pytree, named_pytree)
                named_structure = jtu.tree_structure(named_pytree)
                if prefix:
                    dummy_pytree = jtu.tree_unflatten(structure, [0] * len(leaves))
                    dummy_named = jtu.tree_unflatten(
                        named_structure, [0] * named_structure.num_leaves
                    )
                    try:
                        jtu.tree_map(lambda _, __: 0, dummy_named, dummy_pytree)
                    except ValueError:
                        return False
                elif suffix:
                    has_structure = lambda x: jtu.tree_structure(x) == named_structure
                    dummy_pytree = jtu.tree_unflatten(structure, [0] * len(leaves))
                    dummy_leaves = jtu.tree_leaves(dummy_pytree, is_leaf=has_structure)
                    if any(not has_structure(x) for x in dummy_leaves):
                        return False
                else:
                    if structure != named_structure:
                        return False

        try:
            for leaf_index, leaf in enumerate(leaves):
                if cls.structure is not None:
                    set_treepath_memo(leaf_index, cls.structure)
                if not is_check_leaftype(leaf):
                    return False
                clear_treepath_memo()
        finally:
            clear_treepath_memo()
        return True

    # Can't return a generic (e.g. _FakePyTree[item]) because generic aliases don't do
    # the custom __instancecheck__ that we want.
    # We can't add that __instancecheck__  via subclassing, e.g.
    # type("PyTree", (Generic[_T],), {}), because dynamic subclassing of typeforms
    # isn't allowed.
    # Likewise we can't do types.new_class("PyTree", (Generic[_T],), {}) because that
    # has __module__ "types", e.g. we get types.PyTree[int].
    @ft.lru_cache(maxsize=None)
    def __getitem__(cls, item):
        if isinstance(item, tuple):
            if len(item) == 2:

                class X(PyTree):
                    leaftype = item[0]
                    structure = item[1].strip()

                if not isinstance(X.structure, str):
                    raise ValueError(
                        "The structure annotation `struct` in "
                        "`jaxtyping.PyTree[leaftype, struct]` must be be a string, "
                        f"e.g. `jaxtyping.PyTree[leaftype, 'T']`. Got '{X.structure}'."
                    )
                pieces = X.structure.split()
                if len(pieces) == 0:
                    raise ValueError(
                        "The string `struct` in `jaxtyping.PyTree[leaftype, struct]` "
                        "cannot be the empty string."
                    )
                for piece_index, piece in enumerate(pieces):
                    if (piece_index == 0) or (piece_index == len(pieces) - 1):
                        if piece == "...":
                            continue
                    if not piece.isidentifier():
                        raise ValueError(
                            "The string `struct` in "
                            "`jaxtyping.PyTree[leaftype, struct]` must be be a "
                            "whitespace-separated sequence of identifiers, e.g. "
                            "`jaxtyping.PyTree[leaftype, 'T']` or "
                            "`jaxtyping.PyTree[leaftype, 'foo bar']`.\n"
                            "(Here, 'identifier' is used in the same sense as in "
                            "regular Python, i.e. a valid variable name.)\n"
                            f"Got piece '{piece}' in overall structure '{X.structure}'."
                        )
                name = str(_FakePyTree[item[0]])[:-1] + ', "' + item[1].strip() + '"]'
            else:
                raise ValueError(
                    "The subscript `foo` in `jaxtyping.PyTree[foo]` must either be a "
                    "leaf type, e.g. `PyTree[int]`, or a 2-tuple of leaf and "
                    "structure, e.g. `PyTree[int, 'T']`. Received a tuple of length "
                    f"{len(item)}."
                )
        else:
            name = str(_FakePyTree[item])

            class X(PyTree):
                leaftype = item
                structure = None

        X.__name__ = name
        X.__qualname__ = name
        if getattr(typing, "GENERATING_DOCUMENTATION", False):
            X.__module__ = "builtins"
        else:
            X.__module__ = "jaxtyping"
        return X


# Can't do `class PyTree(Generic[_T]): ...` because we need to override the
# instancecheck for PyTree[foo], but subclassing
# `type(Generic[int])`, i.e. `typing._GenericAlias` is disallowed.
PyTree = _MetaPyTree("PyTree", (), {})
if getattr(typing, "GENERATING_DOCUMENTATION", False):
    PyTree.__module__ = "builtins"
else:
    PyTree.__module__ = "jaxtyping"
PyTree.__doc__ = """Represents a PyTree.

Annotations of the following sorts are supported:
```python
a: PyTree
b: PyTree[LeafType]
c: PyTree[LeafType, "T"]
d: PyTree[LeafType, "S T"]
e: PyTree[LeafType, "... T"]
f: PyTree[LeafType, "T ..."]
```

These correspond to:

a. A plain `PyTree` can be used an annotation, in which case `PyTree` is simply a
    suggestively-named alternative to `Any`.
    ([By definition all types are PyTrees.](https://jax.readthedocs.io/en/latest/pytrees.html))

b. `PyTree[LeafType]` denotes a PyTree all of whose leaves match `LeafType`. For
    example, `PyTree[int]` or `PyTree[Union[str, Float32[Array, "b c"]]]`.

c. A structure name can also be passed. In this case
    `jax.tree_util.tree_structure(...)` will be called, and bound to the structure name.
    This can be used to mark that multiple PyTrees all have the same structure:
    ```python
    def f(x: PyTree[int, "T"], y: PyTree[int, "T"]):
        ...
    ```
    Structures are bound to names in the same way as array shape annotations, i.e.
    within the thread-local dynamic context of a [`jaxtyping.jaxtyped`][] decorator.

d. A composite structure can be declared. In this case the variable must have a PyTree
    structure each to the composition of multiple previously-bound PyTree structures.
    For example:
    ```python
    def f(x: PyTree[int, "T"], y: PyTree[int, "S"], z: PyTree[int, "S T"]):
        ...

    x = (1, 2)
    y = {"key": 3}
    z = {"key": (4, 5)}  # structure is the composition of the structures of `y` and `z`
    f(x, y, z)
    ```
    When performing runtime type-checking, all the individual pieces must have already
    been bound to structures, otherwise the composite structure check will throw an error.

e. A structure can begin with a `...`, to denote that the lower levels of the PyTree
    must match the declared structure, but the upper levels can be arbitrary. As in the
    previous case, all named pieces must already have been seen and their structures
    bound.

f. A structure can end with a `...`, to denote that the PyTree must be a prefix of the
    declared structure, but the lower levels can be arbitrary. As in the previous two
    cases, all named pieces must already have been seen and their structures bound.
"""  # noqa: E501
