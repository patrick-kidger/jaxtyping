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

import contextlib
import gc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import typeguard


ParamError = []
ReturnError = []
ParamError.append(TypeError)  # old typeguard
ReturnError.append(TypeError)  # old typeguard

try:
    # new typeguard
    ParamError.append(typeguard.TypeCheckError)
    ReturnError.append(typeguard.TypeCheckError)
except AttributeError:
    pass

try:
    import beartype
except ImportError:
    pass
else:
    ParamError.append(beartype.roar.BeartypeCallHintParamViolation)
    ReturnError.append(beartype.roar.BeartypeCallHintReturnViolation)

ParamError = tuple(ParamError)
ReturnError = tuple(ReturnError)


@eqx.filter_jit
def make_mlp(key):
    return eqx.nn.MLP(2, 2, 2, 2, key=key)


@contextlib.contextmanager
def assert_no_garbage(
    allowed_garbage_predicate: Callable[[Any], bool] = lambda _: False,
):
    try:
        gc.disable()
        gc.collect()
        # It's unclear why, but a second GC is necessary to fully collect
        # existing garbage.
        gc.collect()
        gc.garbage.clear()

        yield

        # Do a GC collection, saving collected objects in gc.garbage.
        gc.set_debug(gc.DEBUG_SAVEALL)
        gc.collect()

        disallowed_garbage = [
            obj for obj in gc.garbage if not allowed_garbage_predicate(obj)
        ]
        assert not disallowed_garbage
    finally:
        # Reset the GC back to normal.
        gc.set_debug(0)
        gc.garbage.clear()
        gc.collect()
        gc.enable()
