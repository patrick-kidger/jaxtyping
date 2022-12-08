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

from .array_types import AbstractArray as AbstractArray
from .array_types import AbstractDtype as AbstractDtype
from .array_types import BFloat16 as BFloat16
from .array_types import Bool as Bool
from .array_types import Complex as Complex
from .array_types import Complex64 as Complex64
from .array_types import Complex128 as Complex128
from .array_types import Float as Float
from .array_types import Float16 as Float16
from .array_types import Float32 as Float32
from .array_types import Float64 as Float64
from .array_types import get_array_name_format as get_array_name_format
from .array_types import Inexact as Inexact
from .array_types import Int as Int
from .array_types import Int8 as Int8
from .array_types import Int16 as Int16
from .array_types import Int32 as Int32
from .array_types import Int64 as Int64
from .array_types import Integer as Integer
from .array_types import Num as Num
from .array_types import set_array_name_format as set_array_name_format
from .array_types import Shaped as Shaped
from .array_types import UInt as UInt
from .array_types import UInt8 as Uint8
from .array_types import UInt16 as Uint16
from .array_types import UInt32 as Uint32
from .array_types import UInt64 as Uint64
from .decorator import jaxtyped as jaxtyped
from .import_hook import install_import_hook as install_import_hook


if typing.TYPE_CHECKING:
    _T = typing.TypeVar("_T")

    class PyTree(typing_extensions.Protocol[_T]):
        pass

elif has_jax:
    from .pytree_type import PyTree

del has_jax

__version__ = "0.2.9"
