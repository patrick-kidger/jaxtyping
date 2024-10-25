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

import enum
import functools as ft
import importlib.util
import re
import sys
import types
import typing
from dataclasses import dataclass
from typing import (
    Any,
    get_args,
    get_origin,
    Literal,
    NoReturn,
    Optional,
    TypeVar,
    Union,
)


# Bit of a hack, but jaxtyping provides nicer error messages than typeguard. This means
# we sometimes want to use it as our runtime type checker everywhere, even in non-array
# use-cases, for which numpy is too heavy a dependency.
# Honestly we should probably consider factoring out part of jaxtyping into a separate
# package. (Specifically (a) the multi-argument checking and (b) the better error
# messages and (c) the import hook that places the checker on the bottom of the
# decorator stack.) And resist the urge to write our own runtime type-checker, I really
# don't want to have to keep that up-to-date with changes in the Python typing spec...
if importlib.util.find_spec("numpy") is not None:
    import numpy as np

from ._errors import AnnotationError
from ._storage import (
    get_shape_memo,
    get_treeflatten_memo,
    get_treepath_memo,
    set_shape_memo,
)


_array_name_format = "dtype_and_shape"


def get_array_name_format():
    return _array_name_format


def set_array_name_format(value):
    global _array_name_format
    _array_name_format = value


_any_dtype = object()

_anonymous_dim = object()
_anonymous_variadic_dim = object()


class _DimType(enum.Enum):
    named = enum.auto()
    fixed = enum.auto()
    symbolic = enum.auto()


@dataclass(frozen=True)
class _NamedDim:
    name: str
    broadcastable: bool
    treepath: Any


@dataclass(frozen=True)
class _NamedVariadicDim:
    name: str
    broadcastable: bool
    treepath: Any


@dataclass(frozen=True)
class _FixedDim:
    size: str
    broadcastable: bool


@dataclass(frozen=True)
class _SymbolicDim:
    elem: Any
    broadcastable: bool


_AbstractDimOrVariadicDim = Union[
    Literal[_anonymous_dim],
    Literal[_anonymous_variadic_dim],
    _NamedDim,
    _NamedVariadicDim,
    _FixedDim,
    _SymbolicDim,
]
_AbstractDim = Union[Literal[_anonymous_dim], _NamedDim, _FixedDim, _SymbolicDim]


def _check_dims(
    cls_dims: list[_AbstractDim],
    obj_shape: tuple[int, ...],
    single_memo: dict[str, int],
    arg_memo: dict[str, Any],
) -> str:
    assert len(cls_dims) == len(obj_shape)
    for cls_dim, obj_size in zip(cls_dims, obj_shape):
        if cls_dim is _anonymous_dim:
            pass
        elif cls_dim.broadcastable and obj_size == 1:
            pass
        elif type(cls_dim) is _FixedDim:
            if cls_dim.size != obj_size:
                return f"the dimension size {obj_size} does not equal {cls_dim.size} as expected by the type hint"  # noqa: E501
        elif type(cls_dim) is _SymbolicDim:
            try:
                # Support f-string syntax.
                # https://stackoverflow.com/a/53671539/22545467
                elem = eval(f"f'{cls_dim.elem}'", arg_memo.copy())
                # Make a copy to avoid `__builtins__` getting added as a key.
                eval_size = eval(elem, single_memo.copy())
            except NameError as e:
                raise AnnotationError(
                    f"Cannot process symbolic axis '{cls_dim.elem}' as "
                    "some axis names have not been processed. "
                    "Have you applied the `jaxtyped` decorator? "
                    "In practice you should usually only use symbolic axes in "
                    "annotations for return types, referring only to axes "
                    "annotated for arguments."
                ) from e
            if eval_size != obj_size:
                return f"the dimension size {obj_size} does not equal the existing value of {cls_dim.elem}={eval_size}"  # noqa: E501
        else:
            assert type(cls_dim) is _NamedDim
            if cls_dim.treepath:
                name = get_treepath_memo() + cls_dim.name
            else:
                name = cls_dim.name
            try:
                cls_size = single_memo[name]
            except KeyError:
                single_memo[name] = obj_size
            else:
                if cls_size != obj_size:
                    return f"the size of dimension {cls_dim.name} is {obj_size} which does not equal the existing value of {cls_size}"  # noqa: E501
    return ""


def _dtype_is_numpy_struct_array(dtype):
    return dtype.type.__name__ == "void" and dtype is not np.dtype(np.void)


class _MetaAbstractArray(type):
    _skip_instancecheck: bool = False

    def make_transparent(cls):
        cls._skip_instancecheck = True

    def __instancecheck__(cls, obj: Any) -> bool:
        return cls.__instancecheck_str__(obj) == ""

    def __instancecheck_str__(cls, obj: Any) -> str:
        if cls._skip_instancecheck:
            return ""
        if cls.array_type is Any:
            if not (hasattr(obj, "shape") and hasattr(obj, "dtype")):
                return "this value does not have both `shape` and `dtype` attributes."
        else:
            if not isinstance(obj, cls.array_type):
                return f"this value is not an instance of the underlying array type {cls.array_type}"  # noqa: E501
        if get_treeflatten_memo():
            return ""

        if hasattr(obj.dtype, "type") and hasattr(obj.dtype.type, "__name__"):
            # JAX, numpy
            dtype = obj.dtype.type.__name__
            # numpy structured array is strictly a subtype of np.void
            if _dtype_is_numpy_struct_array(obj.dtype):
                dtype = str(obj.dtype)
        elif hasattr(obj.dtype, "as_numpy_dtype"):
            # TensorFlow
            dtype = obj.dtype.as_numpy_dtype.__name__
        else:
            # Everyone else, including PyTorch.
            # This offers an escape hatch for anyone looking to use jaxtyping for their
            # own array-like types.
            dtype = obj.dtype
            if not isinstance(dtype, str):
                *_, dtype = repr(obj.dtype).rsplit(".", 1)

        if cls.dtypes is not _any_dtype:
            in_dtypes = False
            for cls_dtype in cls.dtypes:
                if type(cls_dtype) is str:
                    in_dtypes = dtype == cls_dtype
                elif type(cls_dtype) is re.Pattern:
                    in_dtypes = bool(cls_dtype.match(dtype))
                else:
                    assert False
                if in_dtypes:
                    break
            if not in_dtypes:
                if len(cls.dtypes) == 1:
                    return f"this array has dtype {dtype}, not {cls.dtypes[0]} as expected by the type hint"  # noqa: E501
                else:
                    return f"this array has dtype {dtype}, not any of {cls.dtypes} as expected by the type hint"  # noqa: E501

        single_memo, variadic_memo, pytree_memo, arg_memo = get_shape_memo()
        single_memo_bak = single_memo.copy()
        variadic_memo_bak = variadic_memo.copy()
        pytree_memo_bak = pytree_memo.copy()
        arg_memo_bak = arg_memo.copy()
        try:
            check = cls._check_shape(obj, single_memo, variadic_memo, arg_memo)
        except Exception:
            set_shape_memo(
                single_memo_bak, variadic_memo_bak, pytree_memo_bak, arg_memo_bak
            )
            raise
        if check == "":
            return check
        else:
            set_shape_memo(
                single_memo_bak, variadic_memo_bak, pytree_memo_bak, arg_memo_bak
            )
            return check

    def _check_shape(
        cls,
        obj,
        single_memo: dict[str, int],
        variadic_memo: dict[str, tuple[bool, tuple[int, ...]]],
        arg_memo: dict[str, Any],
    ) -> str:
        if cls.index_variadic is None:
            if len(obj.shape) != len(cls.dims):
                return f"this array has {len(obj.shape)} dimensions, not the {len(cls.dims)} expected by the type hint"  # noqa: E501
            return _check_dims(cls.dims, obj.shape, single_memo, arg_memo)
        else:
            if len(obj.shape) < len(cls.dims) - 1:
                return f"this array has {len(obj.shape)} dimensions, which is fewer than {len(cls.dims) - 1} that is the minimum expected by the type hint"  # noqa: E501
            i = cls.index_variadic
            j = -(len(cls.dims) - i - 1)
            if j == 0:
                j = None
            prefix_check = _check_dims(
                cls.dims[:i], obj.shape[:i], single_memo, arg_memo
            )
            if prefix_check != "":
                return prefix_check
            if j is not None:
                suffix_check = _check_dims(
                    cls.dims[j:], obj.shape[j:], single_memo, arg_memo
                )
                if suffix_check != "":
                    return suffix_check
            variadic_dim = cls.dims[i]
            if variadic_dim is _anonymous_variadic_dim:
                return ""
            else:
                assert type(variadic_dim) is _NamedVariadicDim
                if variadic_dim.treepath:
                    name = get_treepath_memo() + variadic_dim.name
                else:
                    name = variadic_dim.name
                broadcastable = variadic_dim.broadcastable
                try:
                    prev_broadcastable, prev_shape = variadic_memo[name]
                except KeyError:
                    variadic_memo[name] = (broadcastable, obj.shape[i:j])
                    return ""
                else:
                    new_shape = obj.shape[i:j]
                    if prev_broadcastable:
                        try:
                            broadcast_shape = np.broadcast_shapes(new_shape, prev_shape)
                        except ValueError:  # not broadcastable e.g. (3, 4) and (5,)
                            return f"the shape of its variadic dimensions '*{variadic_dim.name}' is {new_shape}, which cannot be broadcast with the existing value of {prev_shape}"  # noqa: E501
                        if not broadcastable and broadcast_shape != new_shape:
                            return f"the shape of its variadic dimensions '*{variadic_dim.name}' is {new_shape}, which the existing value of {prev_shape} cannot be broadcast to"  # noqa: E501
                        variadic_memo[name] = (broadcastable, broadcast_shape)
                    else:
                        if broadcastable:
                            try:
                                broadcast_shape = np.broadcast_shapes(
                                    new_shape, prev_shape
                                )
                            except ValueError:  # not broadcastable e.g. (3, 4) and (5,)
                                return f"the shape of its variadic dimensions '*{variadic_dim.name}' is {new_shape}, which cannot be broadcast with the existing value of {prev_shape}"  # noqa: E501
                            if broadcast_shape != prev_shape:
                                return f"the shape of its variadic dimensions '*{variadic_dim.name}' is {new_shape}, which cannot be broadcast to the existing value of {prev_shape}"  # noqa: E501
                        else:
                            if new_shape != prev_shape:
                                return f"the shape of its variadic dimensions '*{variadic_dim.name}' is {new_shape}, which does not equal the existing value of {prev_shape}"  # noqa: E501
                    return ""
            assert False


@ft.lru_cache(maxsize=None)
def _make_metaclass(base_metaclass):
    class MetaAbstractArray(_MetaAbstractArray, base_metaclass):
        # We have to use identity-based eq/hash behaviour. The reason for this is that
        # when deserializing using cloudpickle (very common, it seems), that cloudpickle
        # will actually attempt to put a partially constructed class in a dictionary.
        # So if we start accessing `cls.index_variadic` and the like here, then that
        # explodes.
        # See
        # https://github.com/patrick-kidger/jaxtyping/issues/198
        # https://github.com/patrick-kidger/jaxtyping/issues/261
        #
        # This does mean that if you want to compare two array annotations for equality
        # (e.g. this happens in jaxtyping's tests as part of checking correctness) then
        # a custom equality function must be used -- we can't put it here.
        def __eq__(cls, other):
            return cls is other

        def __hash__(cls):
            return id(cls)

    return MetaAbstractArray


def _check_scalar(dtype, dtypes, dims):
    for dim in dims:
        if dim is not _anonymous_variadic_dim and not isinstance(
            dim, _NamedVariadicDim
        ):
            return False
    return (_any_dtype is dtypes) or any(d.startswith(dtype) for d in dtypes)


class AbstractArray(metaclass=_MetaAbstractArray):
    """This is the base class of all shape-and-dtype-specified arrays, e.g. it's a base
    class for `Float32[Array, "foo"]`.

    This might be useful if you're trying to inspect type annotations yourself, e.g.
    you can check `issubclass(annotation, jaxtyping.AbstractArray)`.
    """

    array_type: Any
    dtypes: list[str]
    dims: tuple[_AbstractDimOrVariadicDim, ...]
    index_variadic: Optional[int]
    dim_str: str


_not_made = object()

_union_types = [Union]
if sys.version_info >= (3, 10):
    _union_types.append(types.UnionType)


@ft.lru_cache(maxsize=None)
def _make_array_cached(array_type, dim_str, dtypes, name):
    if not isinstance(dim_str, str):
        raise ValueError(
            "Shape specification must be a string. Axes should be separated with "
            "spaces."
        )
    dims = []
    index_variadic = None
    for index, elem in enumerate(dim_str.split()):
        if "," in elem and "(" not in elem:
            # Common mistake.
            # Disable in the case that there's brackets to allow for function calls,
            # e.g. `min(foo,bar)`, in symbolic axes.
            raise ValueError("Axes should be separated with spaces, not commas")
        if elem.endswith("#"):
            raise ValueError(
                "As of jaxtyping v0.1.0, broadcastable axes are now denoted "
                "with a # at the start, rather than at the end"
            )

        if "..." in elem:
            if elem != "...":
                raise ValueError(
                    "Anonymous multiple axes '...' must be used on its own; "
                    f"got {elem}"
                )
            broadcastable = False
            variadic = True
            anonymous = True
            treepath = False
            dim_type = _DimType.named
        else:
            broadcastable = False
            variadic = False
            anonymous = False
            treepath = False
            while True:
                if len(elem) == 0:
                    # This branch needed as just `_` is valid
                    break
                first_char = elem[0]
                if first_char == "#":
                    if broadcastable:
                        raise ValueError(
                            "Do not use # twice to denote broadcastability, e.g. "
                            "`##foo` is not allowed"
                        )
                    broadcastable = True
                    elem = elem[1:]
                elif first_char == "*":
                    if variadic:
                        raise ValueError(
                            "Do not use * twice to denote accepting multiple "
                            "axes, e.g. `**foo` is not allowed"
                        )
                    variadic = True
                    elem = elem[1:]
                elif first_char == "_":
                    if anonymous:
                        raise ValueError(
                            "Do not use _ twice to denote anonymity, e.g. `__foo` "
                            "is not allowed"
                        )
                    anonymous = True
                    elem = elem[1:]
                elif first_char == "?":
                    if treepath:
                        raise ValueError(
                            "Do not use ? twice to denote dependence on location "
                            "within a PyTree, e.g. `??foo` is not allowed"
                        )
                    treepath = True
                    elem = elem[1:]
                # Allow e.g. `foo=4` as an alternate syntax for just `4`, so that one
                # can write e.g. `Float[Array, "rows=3 cols=4"]`
                elif elem.count("=") == 1:
                    _, elem = elem.split("=")
                else:
                    break
            if len(elem) == 0 or elem.isidentifier():
                dim_type = _DimType.named
            else:
                try:
                    elem = int(elem)
                except ValueError:
                    dim_type = _DimType.symbolic
                else:
                    dim_type = _DimType.fixed

        if variadic:
            if index_variadic is not None:
                raise ValueError(
                    "Cannot use variadic specifiers (`*name` or `...`) "
                    "more than once."
                )
            index_variadic = index

        if dim_type is _DimType.fixed:
            if variadic:
                raise ValueError(
                    "Cannot have a fixed axis bind to multiple axes, e.g. "
                    "`*4` is not allowed."
                )
            if anonymous:
                raise ValueError(
                    "Cannot have a fixed axis be anonymous, e.g. `_4` is not allowed."
                )
            if treepath:
                raise ValueError(
                    "Cannot have a fixed axis have tree-path dependence, e.g. `?4` is "
                    "not allowed."
                )
            elem = _FixedDim(elem, broadcastable)
        elif dim_type is _DimType.named:
            if anonymous:
                if broadcastable:
                    raise ValueError(
                        "Cannot have an axis be both anonymous and "
                        "broadcastable, e.g. `#_` is not allowed."
                    )
                if variadic:
                    elem = _anonymous_variadic_dim
                else:
                    elem = _anonymous_dim
            else:
                if variadic:
                    elem = _NamedVariadicDim(elem, broadcastable, treepath)
                else:
                    elem = _NamedDim(elem, broadcastable, treepath)
        else:
            assert dim_type is _DimType.symbolic
            if anonymous:
                raise ValueError(
                    "Cannot have a symbolic axis be anonymous, e.g. "
                    "`_foo+bar` is not allowed"
                )
            if variadic:
                raise ValueError(
                    "Cannot have symbolic multiple-axes, e.g. "
                    "`*foo+bar` is not allowed"
                )
            if treepath:
                raise ValueError(
                    "Cannot have a symbolic axis with tree-path dependence, e.g. "
                    "`?foo+bar` is not allowed"
                )
            elem = _SymbolicDim(elem, broadcastable)
        dims.append(elem)
    dims = tuple(dims)

    # Allow Python built-in numeric types.
    # TODO: do something more generic than this? Should we _make all types
    # that have `shape` and `dtype` attributes or something?
    array_origin = get_origin(array_type)
    if array_origin is not None:
        array_type = array_origin
    if array_type is bool:
        if _check_scalar("bool", dtypes, dims):
            return array_type
        else:
            return _not_made
    elif array_type is int:
        if _check_scalar("int", dtypes, dims):
            return array_type
        else:
            return _not_made
    elif array_type is float:
        if _check_scalar("float", dtypes, dims):
            return array_type
        else:
            return _not_made
    elif array_type is complex:
        if _check_scalar("complex", dtypes, dims):
            return array_type
        else:
            return _not_made
    elif array_type is np.bool_:
        if _check_scalar("bool", dtypes, dims):
            return array_type
        else:
            return _not_made
    elif array_type is np.generic or array_type is np.number:
        if _check_scalar("", dtypes, dims):
            return array_type
        else:
            return _not_made
    if array_type is not Any and issubclass(array_type, AbstractArray):
        if dtypes is _any_dtype:
            dtypes = array_type.dtypes
        elif array_type.dtypes is not _any_dtype:
            dtypes = tuple(x for x in dtypes if x in array_type.dtypes)
            if len(dtypes) == 0:
                raise ValueError(
                    "A jaxtyping annotation cannot be extended with no overlapping "
                    "dtypes. For example, `Bool[Float[Array, 'dim1'], 'dim2']` is an "
                    "error. You probably want to make the outer wrapper be `Shaped`."
                )
        if array_type.index_variadic is not None:
            if index_variadic is None:
                index_variadic = array_type.index_variadic + len(dims)
            else:
                raise ValueError(
                    "Cannot use variadic specifiers (`*name` or `...`) "
                    "in both the original array and the extended array"
                )
        dims = dims + array_type.dims
        dim_str = dim_str + " " + array_type.dim_str
        array_type = array_type.array_type
    try:
        type_str = array_type.__name__
    except AttributeError:
        type_str = repr(array_type)
    if _array_name_format == "dtype_and_shape":
        name = f"{name}[{type_str}, '{dim_str}']"
    elif _array_name_format == "array":
        name = type_str
    else:
        raise ValueError(f"array_name_format {_array_name_format} not recognised")

    return (array_type, name, dtypes, dims, index_variadic, dim_str)


def _make_array(*args, **kwargs):
    out = _make_array_cached(*args, **kwargs)

    if type(out) is tuple:
        array_type, name, dtypes, dims, index_variadic, dim_str = out
        metaclass = (
            _make_metaclass(type)
            if array_type is Any
            else _make_metaclass(type(array_type))
        )

        out = metaclass(
            name,
            (AbstractArray,) if array_type is Any else (array_type, AbstractArray),
            dict(
                array_type=array_type,
                dtypes=dtypes,
                dims=dims,
                index_variadic=index_variadic,
                dim_str=dim_str,
            ),
        )
        if getattr(typing, "GENERATING_DOCUMENTATION", False):
            out.__module__ = "builtins"
        else:
            out.__module__ = "jaxtyping"

    return out


class _MetaAbstractDtype(type):
    def __instancecheck__(cls, obj: Any) -> NoReturn:
        raise AnnotationError(
            f"Do not use `isinstance(x, jaxtyping.{cls.__name__})`. If you want to "
            "check just the dtype of an array, then use "
            f'`jaxtyping.{cls.__name__}[jnp.ndarray, "..."]`.'
        )

    def __getitem__(cls, item: tuple[Any, str]):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "As of jaxtyping v0.2.0, type annotations must now include both an "
                "array type and a shape. For example `Float[Array, 'foo bar']`.\n"
                "Ellipsis can be used to accept any shape: `Float[Array, '...']`."
            )
        array_type, dim_str = item
        dim_str = dim_str.strip()
        if isinstance(array_type, TypeVar):
            bound = array_type.__bound__
            if bound is None:
                constraints = array_type.__constraints__
                if constraints == ():
                    array_type = Any
                else:
                    array_type = Union[constraints]
            else:
                array_type = bound
        del item
        if get_origin(array_type) in _union_types:
            out = [
                _make_array(x, dim_str, cls.dtypes, cls.__name__)
                for x in get_args(array_type)
            ]
            out = tuple(x for x in out if x is not _not_made)
            if len(out) == 0:
                raise ValueError("Invalid jaxtyping type annotation.")
            elif len(out) == 1:
                (out,) = out
            else:
                out = Union[out]
        else:
            out = _make_array(array_type, dim_str, cls.dtypes, cls.__name__)
            if out is _not_made:
                raise ValueError("Invalid jaxtyping type annotation.")
        return out


class AbstractDtype(metaclass=_MetaAbstractDtype):
    """This is the base class of all dtypes. This can be used to create your own custom
    collection of dtypes (analogous to `Float`, `Inexact` etc.)

    You must specify the class attribute `dtypes`. This can either be a string, a
    regex (as returned by `re.compile(...)`), or a tuple/list of strings/regexes.

    At runtime, the array or tensor's dtype is converted to a string and compared
    against the string (an exact match is required) or regex. (String matching is
    performed, rather than just e.g. `array.dtype == dtype`, to provide cross-library
    compatibility between JAX/PyTorch/etc.)

    !!! Example

        ```python
        class UInt8or16(AbstractDtype):
            dtypes = ["uint8", "uint16"]

        UInt8or16[Array, "shape"]
        ```
        which is essentially equivalent to
        ```python
        Union[UInt8[Array, "shape"], UInt16[Array, "shape"]]
        ```
    """

    dtypes: Union[Literal[_any_dtype], list[Union[str, re.Pattern]]]

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "AbstractDtype cannot be instantiated. Perhaps you wrote e.g. "
            '`Float32("shape")` when you mean `Float32[jnp.ndarray, "shape"]`?'
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        dtypes: Union[Literal[_any_dtype], str, list[str]] = cls.dtypes
        if isinstance(dtypes, (str, re.Pattern)):
            dtypes = (dtypes,)
        elif dtypes is not _any_dtype:
            dtypes = tuple(dtypes)
        cls.dtypes = dtypes


_prng_key = "prng_key"
_bool = "bool"
_bool_ = "bool_"
_uint4 = "uint4"
_uint8 = "uint8"
_uint16 = "uint16"
_uint32 = "uint32"
_uint64 = "uint64"
_int4 = "int4"
_int8 = "int8"
_int16 = "int16"
_int32 = "int32"
_int64 = "int64"
_bfloat16 = "bfloat16"
_float16 = "float16"
_float32 = "float32"
_float64 = "float64"
_complex64 = "complex64"
_complex128 = "complex128"


def _make_dtype(_dtypes, name):
    class _Cls(AbstractDtype):
        dtypes = _dtypes

    _Cls.__name__ = name
    _Cls.__qualname__ = name
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        _Cls.__module__ = "builtins"
    else:
        _Cls.__module__ = "jaxtyping"
    return _Cls


UInt4 = _make_dtype(_uint4, "UInt4")
UInt8 = _make_dtype(_uint8, "UInt8")
UInt16 = _make_dtype(_uint16, "UInt16")
UInt32 = _make_dtype(_uint32, "UInt32")
UInt64 = _make_dtype(_uint64, "UInt64")
Int4 = _make_dtype(_int4, "Int4")
Int8 = _make_dtype(_int8, "Int8")
Int16 = _make_dtype(_int16, "Int16")
Int32 = _make_dtype(_int32, "Int32")
Int64 = _make_dtype(_int64, "Int64")
BFloat16 = _make_dtype(_bfloat16, "BFloat16")
Float16 = _make_dtype(_float16, "Float16")
Float32 = _make_dtype(_float32, "Float32")
Float64 = _make_dtype(_float64, "Float64")
Complex64 = _make_dtype(_complex64, "Complex64")
Complex128 = _make_dtype(_complex128, "Complex128")

bools = [_bool, _bool_]
uints = [_uint4, _uint8, _uint16, _uint32, _uint64]
ints = [_int4, _int8, _int16, _int32, _int64]
floats = [_bfloat16, _float16, _float32, _float64]
complexes = [_complex64, _complex128]

# We match NumPy's type hierarachy in what types to provide. See the diagram at
# https://numpy.org/doc/stable/reference/arrays.scalars.html#scalars

Bool = _make_dtype(bools, "Bool")
UInt = _make_dtype(uints, "UInt")
Int = _make_dtype(ints, "Int")
Integer = _make_dtype(uints + ints, "Integer")
Float = _make_dtype(floats, "Float")
Complex = _make_dtype(complexes, "Complex")
Inexact = _make_dtype(floats + complexes, "Inexact")
Real = _make_dtype(floats + uints + ints, "Real")
Num = _make_dtype(uints + ints + floats + complexes, "Num")

Shaped = _make_dtype(_any_dtype, "Shaped")

Key = _make_dtype(_prng_key, "Key")


def make_numpy_struct_dtype(dtype: "np.dtype", name: str):
    """Creates a type annotation for [numpy structured array](https://numpy.org/doc/stable/user/basics.rec.html#structured-arrays)
    It performs an exact match on the name, order, and dtype of all its fields.

    !!! Example

        ```python
        label_t = np.dtype([('first', np.uint8), ('second', np.int8)])
        Label = make_numpy_struct_dtype(label_t, 'Label')
        ```

        after that, you can use it just like any other [`jaxtyping.AbstractDtype`][]:

        ```python
        a: Label[np.ndarray, 'a b'] = np.array([[(1, 0), (0, 1)]], dtype=label_t)
        ```

    **Arguments:**

    - `dtype`: The numpy structured dtype to use.
    - `name`: The name to use for the returned Python class.

    **Returns:**

    A type annotation with classname `name` that matches exactly `dtype` when used like
    any other [`jaxtyping.AbstractDtype`][].
    """
    if not (isinstance(dtype, np.dtype) and _dtype_is_numpy_struct_array(dtype)):
        raise ValueError(f"Expecting a numpy structured array dtype, not {dtype}")
    return _make_dtype(str(dtype), name)
