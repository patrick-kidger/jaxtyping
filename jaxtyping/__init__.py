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


if typing.TYPE_CHECKING:
    from jax.numpy import ndarray as Array
elif has_jax:
    if getattr(typing, "GENERATING_DOCUMENTATION", False):

        class Array:
            pass

        Array.__module__ = "builtins"
    else:
        from jax.numpy import ndarray as Array

from .array_types import (
    AbstractArray,
    AbstractDtype,
    BFloat16,
    Bool,
    Complex,
    Complex64,
    Complex128,
    Float,
    Float16,
    Float32,
    Float64,
    get_array_name_format,
    Inexact,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    Integer,
    Num,
    set_array_name_format,
    Shaped,
    UInt,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from .decorator import jaxtyped
from .import_hook import install_import_hook


if typing.TYPE_CHECKING:
    _T = typing.TypeVar("_T")

    class PyTree(typing_extensions.Protocol[_T]):
        pass

elif has_jax:
    from .pytree_type import PyTree

del has_jax

__version__ = "0.2.8"
