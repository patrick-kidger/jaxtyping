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

import typing
import typing_extensions


try:
    import jax
except ImportError:
    has_jax = False
else:
    has_jax = True
    del jax


# Type checkers don't know which branch below will be executed.
if typing.TYPE_CHECKING:
    # For imports, we need to explicitly `import X as X` in order for Pyright to see
    # them as public. See discussion at https://github.com/microsoft/pyright/issues/2277
    from jax import Array as Array
elif has_jax:
    if getattr(typing, "GENERATING_DOCUMENTATION", False):

        class Array:
            pass

        Array.__module__ = "builtins"
    else:
        from jax import Array as Array

from .array_types import (
    AbstractArray as AbstractArray,
    AbstractDtype as AbstractDtype,
    BFloat16 as BFloat16,
    Bool as Bool,
    Complex as Complex,
    Complex64 as Complex64,
    Complex128 as Complex128,
    Float as Float,
    Float16 as Float16,
    Float32 as Float32,
    Float64 as Float64,
    get_array_name_format as get_array_name_format,
    Inexact as Inexact,
    Int as Int,
    Int8 as Int8,
    Int16 as Int16,
    Int32 as Int32,
    Int64 as Int64,
    Integer as Integer,
    Num as Num,
    set_array_name_format as set_array_name_format,
    Shaped as Shaped,
    UInt as UInt,
    UInt8 as UInt8,
    UInt16 as UInt16,
    UInt32 as UInt32,
    UInt64 as UInt64,
)
from .decorator import jaxtyped as jaxtyped
from .import_hook import install_import_hook as install_import_hook


if typing.TYPE_CHECKING:
    _T = typing.TypeVar("_T")

    class PyTree(typing_extensions.Protocol[_T]):
        pass

elif has_jax:
    from .pytree_type import PyTree

del has_jax

__version__ = "0.2.10"
