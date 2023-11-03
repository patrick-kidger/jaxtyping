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

import threading
from typing import Any, Optional

from ._raise import jaxtyping_raise


_shape_storage = threading.local()


def _has_shape_memo():
    return hasattr(_shape_storage, "memo_stack") and len(_shape_storage.memo_stack) != 0


def get_shape_memo():
    if _has_shape_memo():
        single_memo, variadic_memo, pytree_memo, arguments = _shape_storage.memo_stack[
            -1
        ]
    else:
        # `isinstance` happening outside any @jaxtyped decorators, e.g. at the
        # global scope. In this case just create a temporary memo, since we're not
        # going to be comparing against any stored values anyway.
        single_memo = {}
        variadic_memo = {}
        pytree_memo = {}
        arguments = {}
    return single_memo, variadic_memo, pytree_memo, arguments


def set_shape_memo(single_memo, variadic_memo, pytree_memo, arg_memo) -> None:
    if _has_shape_memo():
        _shape_storage.memo_stack[-1] = (
            single_memo,
            variadic_memo,
            pytree_memo,
            arg_memo,
        )


def push_shape_memo(arguments: dict[str, Any]):
    try:
        memo_stack = _shape_storage.memo_stack
    except AttributeError:
        # Can't be done when `_stack_storage` is created for reasons I forget.
        memo_stack = _shape_storage.memo_stack = []
    memos = ({}, {}, {}, arguments.copy())
    memo_stack.append(memos)
    return memos


def pop_shape_memo() -> None:
    _shape_storage.memo_stack.pop()


_treepath_storage = threading.local()


def clear_treepath_memo() -> None:
    _treepath_storage.value = None


def set_treepath_memo(index: Optional[int], structure: str) -> None:
    if hasattr(_treepath_storage, "value") and _treepath_storage.value is not None:
        jaxtyping_raise(
            ValueError(
                "Cannot typecheck annotations of the form "
                "`PyTree[PyTree[Shaped[Array, '?foo'], 'T'], 'S']` as it is ambiguous "
                "which PyTree the `?` annotation refers to."
            )
        )
    if index is None:
        _treepath_storage.value = f"~~delete~~({structure}) "
    else:
        # Appears in error messages, so human-readable
        _treepath_storage.value = f"(Leaf {index} in structure {structure}) "


def get_treepath_memo() -> str:
    if not hasattr(_treepath_storage, "value") or _treepath_storage.value is None:
        jaxtyping_raise(
            ValueError(
                "Cannot use `?` annotations, e.g. `Shaped[Array, '?foo']`, except "
                "when contained with structured `PyTree` annotations, e.g. "
                "`PyTree[Shaped[Array, '?foo'], 'T']`."
            )
        )
    return _treepath_storage.value


_treeflatten_storage = threading.local()


def clear_treeflatten_memo() -> None:
    _treeflatten_storage.value = False


def set_treeflatten_memo():
    _treeflatten_storage.value = True


def get_treeflatten_memo():
    try:
        return _treeflatten_storage.value
    except AttributeError:
        return False
