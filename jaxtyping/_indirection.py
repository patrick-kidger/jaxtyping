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

# Note that `from typing import Annotated; Bool = Annotated`
# does not work with static type checkers. `Annotated` is a typeform rather
# than a type, meaning it cannot be assigned.
from typing import (
    Annotated as BFloat16,  # noqa: F401
    Annotated as Bool,  # noqa: F401
    Annotated as Complex,  # noqa: F401
    Annotated as Complex64,  # noqa: F401
    Annotated as Complex128,  # noqa: F401
    Annotated as Float,  # noqa: F401
    Annotated as Float16,  # noqa: F401
    Annotated as Float32,  # noqa: F401
    Annotated as Float64,  # noqa: F401
    Annotated as Inexact,  # noqa: F401
    Annotated as Int,  # noqa: F401
    Annotated as Int4,  # noqa: F401
    Annotated as Int8,  # noqa: F401
    Annotated as Int16,  # noqa: F401
    Annotated as Int32,  # noqa: F401
    Annotated as Int64,  # noqa: F401
    Annotated as Integer,  # noqa: F401
    Annotated as Key,  # noqa: F401
    Annotated as Num,  # noqa: F401
    Annotated as Real,  # noqa: F401
    Annotated as Shaped,  # noqa: F401
    Annotated as UInt,  # noqa: F401
    Annotated as UInt4,  # noqa: F401
    Annotated as UInt8,  # noqa: F401
    Annotated as UInt16,  # noqa: F401
    Annotated as UInt32,  # noqa: F401
    Annotated as UInt64,  # noqa: F401
    TYPE_CHECKING,
)


if not TYPE_CHECKING:
    assert False

from jax import (
    Array as PRNGKeyArray,  # noqa: F401
    Array as Scalar,  # noqa: F401
)
from jax.typing import ArrayLike as ScalarLike  # noqa: F401
