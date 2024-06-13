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
import importlib.metadata
import importlib.util
import typing
import warnings
from typing import Union

from ._array_types import (
    AbstractArray as AbstractArray,
    AbstractDtype as AbstractDtype,
    get_array_name_format as get_array_name_format,
    make_numpy_struct_dtype as make_numpy_struct_dtype,
    set_array_name_format as set_array_name_format,
)
from ._config import config as config
from ._decorator import jaxtyped as jaxtyped
from ._errors import (
    AnnotationError as AnnotationError,
    TypeCheckError as TypeCheckError,
)
from ._import_hook import install_import_hook as install_import_hook
from ._ipython_extension import load_ipython_extension as load_ipython_extension
from ._storage import print_bindings as print_bindings


if typing.TYPE_CHECKING:
    import typing_extensions

    from jax import Array as Array
    from jax.tree_util import PyTreeDef as PyTreeDef
    from jax.typing import ArrayLike as ArrayLike, DTypeLike as DTypeLike

    # Introduce an indirection so that we can `import X as X` to make it clear that
    # these are public.
    from ._indirection import (
        BFloat16 as BFloat16,
        Bool as Bool,
        Complex as Complex,
        Complex64 as Complex64,
        Complex128 as Complex128,
        Float as Float,
        Float16 as Float16,
        Float32 as Float32,
        Float64 as Float64,
        Inexact as Inexact,
        Int as Int,
        Int4 as Int4,
        Int8 as Int8,
        Int16 as Int16,
        Int32 as Int32,
        Int64 as Int64,
        Integer as Integer,
        Key as Key,
        Num as Num,
        PRNGKeyArray as PRNGKeyArray,
        Real as Real,
        Scalar as Scalar,
        ScalarLike as ScalarLike,
        Shaped as Shaped,
        UInt as UInt,
        UInt4 as UInt4,
        UInt8 as UInt8,
        UInt16 as UInt16,
        UInt32 as UInt32,
        UInt64 as UInt64,
    )

    # Set up to deliberately confuse a static type checker.
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
else:
    from ._array_types import (
        BFloat16 as BFloat16,
        Bool as Bool,
        Complex as Complex,
        Complex64 as Complex64,
        Complex128 as Complex128,
        Float as Float,
        Float16 as Float16,
        Float32 as Float32,
        Float64 as Float64,
        Inexact as Inexact,
        Int as Int,
        Int4 as Int4,
        Int8 as Int8,
        Int16 as Int16,
        Int32 as Int32,
        Int64 as Int64,
        Integer as Integer,
        Key as Key,
        Num as Num,
        Real as Real,
        Shaped as Shaped,
        UInt as UInt,
        UInt4 as UInt4,
        UInt8 as UInt8,
        UInt16 as UInt16,
        UInt32 as UInt32,
        UInt64 as UInt64,
    )

    # But crucially, does not actually import jax at all. We do that dynamically in
    # __getattr__ if required. See #178.
    if importlib.util.find_spec("jax") is not None:

        @ft.cache
        def __getattr__(item):
            if item == "Array":
                if getattr(typing, "GENERATING_DOCUMENTATION", False):

                    class Array:
                        pass

                    Array.__module__ = "builtins"
                    Array.__qualname__ = "Array"
                    return Array
                else:
                    import jax

                    return jax.Array
            elif item == "ArrayLike":
                if getattr(typing, "GENERATING_DOCUMENTATION", False):

                    class ArrayLike:
                        pass

                    ArrayLike.__module__ = "builtins"
                    ArrayLike.__qualname__ = "ArrayLike"
                    return ArrayLike
                else:
                    import jax.typing

                    return jax.typing.ArrayLike
            elif item == "PRNGKeyArray":
                if getattr(typing, "GENERATING_DOCUMENTATION", False):

                    class PRNGKeyArray:
                        pass

                    PRNGKeyArray.__module__ = "builtins"
                    PRNGKeyArray.__qualname__ = "PRNGKeyArray"
                    return PRNGKeyArray
                else:
                    # New-style `jax.random.key` have scalar shape and dtype `key<foo>`.
                    # Old-style `jax.random.PRNGKey` have shape `(2,)` and dtype
                    # `uint32`.
                    import jax

                    return Union[Key[jax.Array, ""], UInt32[jax.Array, "2"]]
            elif item == "DTypeLike":
                import jax.typing

                return jax.typing.DTypeLike
            elif item == "Scalar":
                import jax

                return Shaped[jax.Array, ""]
            elif item == "ScalarLike":
                import jax.typing

                return Shaped[jax.typing.ArrayLike, ""]
            elif item == "PyTree":
                from ._pytree_type import PyTree

                return PyTree
            elif item == "PyTreeDef":
                if hasattr(typing, "GENERATING_DOCUMENTATION"):
                    # Most parts of the Equinox ecosystem have
                    # `typing.GENERATING_DOCUMENTATION = True` when generating
                    # documentation, to add whatever shims are necessary to get pretty
                    # docs. E.g. to have type annotations appear as just `PyTree`, not
                    # `jaxtyping.PyTree`.
                    #
                    # As jaxtyping actually wants things to appear as e.g.
                    # `jaxtyping.PyTree`, rather than just `PyTree`, then it sets
                    # `typing.GENERATING_DOCUMENTATION = False`, to disable these shims.
                    #
                    # Here we do only a `hasattr` check, as we want to get this version
                    # of `PyTreeDef` in both the jaxtyping and the Equinox(/etc.) docs.

                    class PyTreeDef:
                        """Alias for `jax.tree_util.PyTreeDef`, which is the type of the
                        return from `jax.tree_util.tree_structure(...)`.
                        """

                    if typing.GENERATING_DOCUMENTATION:
                        # Equinox etc. docs get just `PyTreeDef`.
                        # jaxtyping docs get `jaxtyping.PyTreeDef`.
                        PyTreeDef.__qualname__ = "PyTreeDef"
                        PyTreeDef.__module__ = "builtins"
                    return PyTreeDef
                else:
                    import jax.tree_util

                    return jax.tree_util.PyTreeDef
            else:
                raise AttributeError(f"module jaxtyping has no attribute {item!r}")


check_equinox_version = True  # easy-to-replace line with copybara
if check_equinox_version:
    try:
        eqx_version = importlib.metadata.version("equinox")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        major, minor, patch = eqx_version.split(".")
        equinox_version = (int(major), int(minor), int(patch))
        if equinox_version < (0, 11, 0):
            warnings.warn(
                "jaxtyping version >=0.2.23 should be used with Equinox version "
                ">=0.11.1"
            )


__version__ = importlib.metadata.version("jaxtyping")
