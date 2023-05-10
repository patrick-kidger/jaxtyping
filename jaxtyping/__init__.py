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

import importlib.metadata
import typing


try:
    import jax
except ImportError:
    has_jax = False
else:
    has_jax = True
    del jax

# First import some things as normal
from .array_types import (
    AbstractArray as AbstractArray,
    AbstractDtype as AbstractDtype,
    get_array_name_format as get_array_name_format,
    set_array_name_format as set_array_name_format,
)
from .decorator import jaxtyped as jaxtyped
from .import_hook import install_import_hook as install_import_hook


# Now import Array and ArrayLike
if typing.TYPE_CHECKING:
    # For imports, we need to explicitly `import X as X` in order for Pyright to see
    # them as public. See discussion at https://github.com/microsoft/pyright/issues/2277
    from jax import Array as Array
    from jax.typing import ArrayLike as ArrayLike
elif has_jax:
    if getattr(typing, "GENERATING_DOCUMENTATION", False):

        class Array:
            pass

        Array.__module__ = "builtins"

        class ArrayLike:
            pass

        ArrayLike.__module__ = "builtins"
    else:
        from jax import Array as Array

        try:
            from jax.typing import ArrayLike as ArrayLike
        except (ModuleNotFoundError, ImportError):
            pass


# Import our dtypes
if typing.TYPE_CHECKING:
    # Note that `from typing_extensions import Annotated; Bool = Annotated`
    # does not work with static type checkers. `Annotated` is a typeform rather
    # than a type, meaning it cannot be assigned.
    from typing_extensions import (
        Annotated as BFloat16,
        Annotated as Bool,
        Annotated as Complex,
        Annotated as Complex64,
        Annotated as Complex128,
        Annotated as Float,
        Annotated as Float16,
        Annotated as Float32,
        Annotated as Float64,
        Annotated as Inexact,
        Annotated as Int,
        Annotated as Int8,
        Annotated as Int16,
        Annotated as Int32,
        Annotated as Int64,
        Annotated as Integer,
        Annotated as Key,
        Annotated as Num,
        Annotated as Shaped,
        Annotated as UInt,
        Annotated as UInt8,
        Annotated as UInt16,
        Annotated as UInt32,
        Annotated as UInt64,
    )
else:
    # noqas to work around ruff bug
    from .array_types import (
        BFloat16 as BFloat16,  # noqa: F401
        Bool as Bool,  # noqa: F401
        Complex as Complex,  # noqa: F401
        Complex64 as Complex64,  # noqa: F401
        Complex128 as Complex128,  # noqa: F401
        Float as Float,  # noqa: F401
        Float16 as Float16,  # noqa: F401
        Float32 as Float32,  # noqa: F401
        Float64 as Float64,  # noqa: F401
        Inexact as Inexact,  # noqa: F401
        Int as Int,  # noqa: F401
        Int8 as Int8,  # noqa: F401
        Int16 as Int16,  # noqa: F401
        Int32 as Int32,  # noqa: F401
        Int64 as Int64,  # noqa: F401
        Integer as Integer,  # noqa: F401
        Key as Key,  # noqa: F401
        Num as Num,  # noqa: F401
        Shaped as Shaped,  # noqa: F401
        UInt as UInt,  # noqa: F401
        UInt8 as UInt8,  # noqa: F401
        UInt16 as UInt16,  # noqa: F401
        UInt32 as UInt32,  # noqa: F401
        UInt64 as UInt64,  # noqa: F401
    )


# Now import PyTree
if typing.TYPE_CHECKING:
    # Set up to deliberately confuse a static type checker.
    import typing_extensions

    PyTree: typing_extensions.TypeAlias = getattr(typing, "foo" + "bar")
    # What's going on with this madness?
    #
    # At static-type-checking-time, we want `PyTree` to be a type for which both
    # `PyTree` and `PyTree[Foo]` are equivalent to `Any`.
    # (The intention is that `PyTree` be a runtime-only type; there's no real way to
    # do more with static type checkers.)
    #
    # Unfortunately, this isn't possible: `Any` isn't subscriptable. And there's no
    # equivalent way we can fake this using typing annotations. (In some sense the
    # closest thing would be a `Protocol[T]` with no methods, but that's actually the
    # opposite of what we want: that ends up allowing nothing at all.)
    #
    # The good news for us is that static type checkers have an internal escape hatch.
    # If they can't figure out what a type is, then they just give up and allow
    # anything. (I believe this is sometimes called `Unknown`.) Thus, this odd-looking
    # annotation, which static type checkers aren't smart enough to resolve.
elif has_jax:
    from .pytree_type import PyTree as PyTree  # noqa: F401


# Conveniences
if typing.TYPE_CHECKING:
    from jax import Array as Scalar
    from jax.random import PRNGKeyArray as PRNGKeyArray
    from jax.typing import ArrayLike as ScalarLike
elif has_jax:
    from .array_types import PRNGKeyArray, Scalar, ScalarLike  # noqa: F401

del has_jax


__version__ = importlib.metadata.version("jaxtyping")
