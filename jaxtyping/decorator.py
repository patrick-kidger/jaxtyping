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
import threading


storage = threading.local()


class _Jaxtyped:
    def __init__(self, fn):
        self.fn = fn

    def __get__(self, instance, owner):
        return ft.wraps(self.fn)(_Jaxtyped(self.fn.__get__(instance, owner)))

    def __call__(self, *args, **kwargs):
        try:
            memo_stack = storage.memo_stack
        except AttributeError:
            memo_stack = storage.memo_stack = []
        memo_stack.append(({}, {}, {}))
        try:
            return self.fn(*args, **kwargs)
        finally:
            memo_stack.pop()


def jaxtyped(fn):
    return ft.wraps(fn)(_Jaxtyped(fn))
