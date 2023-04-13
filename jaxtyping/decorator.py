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
import functools as ft
import inspect
import threading
import types
import weakref


storage = threading.local()


_jaxtyped_fns = weakref.WeakSet()


def jaxtyped(fn):
    if type(fn) is types.FunctionType and fn in _jaxtyped_fns:
        return fn
    elif inspect.isclass(fn):  # allow decorators on class definitions
        if dataclasses.is_dataclass(fn):
            init = jaxtyped(fn.__init__)
            fn.__init__ = init
            return fn
        else:
            raise ValueError(
                "jaxtyped may only be added as a class decorator to dataclasses"
            )
    # It'd be lovely if we could handle arbitrary descriptors, and not just the builtin
    # ones. Unfortunately that means returning a class instance with a __get__ method,
    # and that turns out to break loads of other things. See beartype issue #211 and
    # jaxtyping issue #71.
    elif isinstance(fn, classmethod):
        return classmethod(jaxtyped(fn.__func__))
    elif isinstance(fn, staticmethod):
        return staticmethod(jaxtyped(fn.__func__))
    elif isinstance(fn, property):
        if fn.fget is None:
            fget = None
        else:
            fget = jaxtyped(fn.fget)
        if fn.fset is None:
            fset = None
        else:
            fset = jaxtyped(fn.fset)
        if fn.fdel is None:
            fdel = None
        else:
            fdel = jaxtyped(fn.fdel)
        return property(fget=fget, fset=fset, fdel=fdel)
    else:

        @ft.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            try:
                memo_stack = storage.memo_stack
            except AttributeError:
                memo_stack = storage.memo_stack = []
            memo_stack.append(({}, {}, {}))
            try:
                return fn(*args, **kwargs)
            finally:
                memo_stack.pop()

        _jaxtyped_fns.add(wrapped_fn)
        return wrapped_fn


def _jaxtyped_typechecker(typechecker):
    # typechecker is expected to probably be either `typeguard.typechecked`, or
    # `beartype.beartype`, or `None`.

    if typechecker is None:
        typechecker = lambda x: x

    def _wrapper(kls):
        assert inspect.isclass(kls)
        if dataclasses.is_dataclass(kls):
            init = jaxtyped(typechecker(kls.__init__))
            kls.__init__ = init
        return kls

    return _wrapper
