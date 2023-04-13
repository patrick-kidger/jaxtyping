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
import sys
import types
import typing
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NoReturn,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np

from .decorator import storage


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


class _NamedDim:
    def __init__(self, name, broadcastable):
        self.name = name
        self.broadcastable = broadcastable


class _NamedVariadicDim:
    def __init__(self, name, broadcastable):
        self.name = name
        self.broadcastable = broadcastable


class _FixedDim:
    def __init__(self, size, broadcastable):
        self.size = size
        self.broadcastable = broadcastable


class _SymbolicDim:
    def __init__(self, expr, broadcastable):
        self.expr = expr
        self.broadcastable = broadcastable


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
    cls_dims: List[_AbstractDim],
    obj_shape: Tuple[int],
    single_memo: Dict[str, int],
) -> bool:
    assert len(cls_dims) == len(obj_shape)
    for cls_dim, obj_size in zip(cls_dims, obj_shape):
        if cls_dim is _anonymous_dim:
            pass
        elif cls_dim.broadcastable and obj_size == 1:
            pass
        elif type(cls_dim) is _FixedDim:
            if cls_dim.size != obj_size:
                return False
        elif type(cls_dim) is _SymbolicDim:
            try:
                eval_size = eval(cls_dim.expr, single_memo)
            except NameError as e:
                raise NameError(
                    f"Cannot process symbolic dimension '{cls_dim.expr}' as some "
                    "dimension names have not been processed. In practice you should "
                    "usually only use symbolic dimensions in annotations for return "
                    "types, referring only to dimensions annotated for arguments."
                ) from e
            if eval_size != obj_size:
                return False
        else:
            assert type(cls_dim) is _NamedDim
            try:
                cls_size = single_memo[cls_dim.name]
            except KeyError:
                single_memo[cls_dim.name] = obj_size
            else:
                if cls_size != obj_size:
                    return False
    return True


class _MetaAbstractArray(type):
    def __instancecheck__(cls, obj):
        if not isinstance(obj, cls.array_type):
            return False

        if hasattr(obj.dtype, "type") and hasattr(obj.dtype.type, "__name__"):
            # JAX, numpy
            dtype = obj.dtype.type.__name__
        elif hasattr(obj.dtype, "as_numpy_dtype"):
            # TensorFlow
            dtype = obj.dtype.as_numpy_dtype.__name__
        else:
            # PyTorch
            repr_dtype = repr(obj.dtype).split(".")
            if len(repr_dtype) == 2 and repr_dtype[0] == "torch":
                dtype = repr_dtype[1]
            else:
                raise RuntimeError(
                    "Unrecognised array/tensor type to extract dtype from"
                )

        if cls.dtypes is not _any_dtype and dtype not in cls.dtypes:
            return False

        no_temp_memo = hasattr(storage, "memo_stack") and len(storage.memo_stack) != 0

        if no_temp_memo:
            single_memo, variadic_memo, variadic_broadcast_memo = storage.memo_stack[-1]
            # Make a copy so we don't mutate the original memo during the shape check.
            single_memo = single_memo.copy()
            variadic_memo = variadic_memo.copy()
            variadic_broadcast_memo = variadic_broadcast_memo.copy()
        else:
            # `isinstance` happening outside any @jaxtyped decorators, e.g. at the
            # global scope. In this case just create a temporary memo, since we're not
            # going to be comparing against any stored values anyway.
            single_memo = {}
            variadic_memo = {}
            variadic_broadcast_memo = {}

        if cls._check_shape(obj, single_memo, variadic_memo, variadic_broadcast_memo):
            # We update the memo every time we successfully pass a shape check
            if no_temp_memo:
                storage.memo_stack[-1] = (
                    single_memo,
                    variadic_memo,
                    variadic_broadcast_memo,
                )
            return True
        else:
            return False

    def _check_shape(
        cls,
        obj,
        single_memo: Dict[str, int],
        variadic_memo: Dict[str, Tuple[int, ...]],
        variadic_broadcast_memo: Dict[str, List[Tuple[int, ...]]],
    ):
        if cls.index_variadic is None:
            if obj.ndim != len(cls.dims):
                return False
            return _check_dims(cls.dims, obj.shape, single_memo)
        else:
            if obj.ndim < len(cls.dims) - 1:
                return False
            i = cls.index_variadic
            j = -(len(cls.dims) - i - 1)
            if j == 0:
                j = None
            if not _check_dims(cls.dims[:i], obj.shape[:i], single_memo):
                return False
            if j is not None and not _check_dims(
                cls.dims[j:], obj.shape[j:], single_memo
            ):
                return False
            variadic_dim = cls.dims[i]
            if variadic_dim is _anonymous_variadic_dim:
                return True
            else:
                assert type(variadic_dim) is _NamedVariadicDim
                variadic_name = variadic_dim.name
                try:
                    if variadic_dim.broadcastable:
                        variadic_shapes = variadic_broadcast_memo[variadic_name]
                    else:
                        variadic_shape = variadic_memo[variadic_name]
                except KeyError:
                    if variadic_dim.broadcastable:
                        variadic_broadcast_memo[variadic_name] = [obj.shape[i:j]]
                    else:
                        variadic_memo[variadic_name] = obj.shape[i:j]
                    return True
                else:
                    if variadic_dim.broadcastable:
                        new_shape = obj.shape[i:j]
                        for existing_shape in variadic_shapes:
                            try:
                                np.broadcast_shapes(new_shape, existing_shape)
                            except ValueError:
                                return False
                        variadic_shapes.append(new_shape)
                        return True
                    else:
                        return variadic_shape == obj.shape[i:j]
            assert False


@ft.lru_cache(maxsize=None)
def _make_metaclass(base_metaclass):
    class MetaAbstractArray(_MetaAbstractArray, base_metaclass):
        pass

    return MetaAbstractArray


def _check_scalar(dtype, dtypes, dims):
    if len(dims) != 0:
        return dims == (_anonymous_variadic_dim,)
    return (_any_dtype is dtypes) or any(d.startswith(dtype) for d in dtypes)


class AbstractArray(metaclass=_MetaAbstractArray):
    array_type: Any
    dtypes: List[str]
    dims: List[_AbstractDimOrVariadicDim]
    index_variadic: Optional[int]


_not_made = object()


_union_types = [typing.Union]
if sys.version_info >= (3, 10):
    _union_types.append(types.UnionType)


@ft.lru_cache(maxsize=None)
def _make_array(array_type, dim_str, dtypes, name):
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
            # e.g. `min(foo,bar)`, in symbolic dimensions.
            raise ValueError("Dimensions should be separated with spaces, not commas")
        if elem.endswith("#"):
            raise ValueError(
                "As of jaxtyping v0.1.0, broadcastable dimensions are now denoted "
                "with a # at the start, rather than at the end"
            )

        if "..." in elem:
            if elem != "...":
                raise ValueError(
                    "Anonymous multiple dimension '...' must be used on its own; "
                    f"got {elem}"
                )
            broadcastable = False
            variadic = True
            anonymous = True
            dim_type = _DimType.named
        else:
            broadcastable = False
            variadic = False
            anonymous = False
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
                            "dimensions, e.g. `**foo` is not allowed"
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
                    "Cannot use multiple-dimension specifiers (`*name` or `...`) "
                    "more than once"
                )
            index_variadic = index

        if dim_type is _DimType.fixed:
            if variadic:
                raise ValueError(
                    "Cannot have a fixed axis bind to multiple dimensions, e.g. "
                    "`*4` is not allowed"
                )
            if anonymous:
                raise ValueError(
                    "Cannot have a fixed axis be anonymous, e.g. `_4` is not " "allowed"
                )
            elem = _FixedDim(elem, broadcastable)
        elif dim_type is _DimType.named:
            if anonymous:
                if broadcastable:
                    raise ValueError(
                        "Cannot have a dimension be both anonymous and "
                        "broadcastable, e.g. `#_` is not allowed"
                    )
                if variadic:
                    elem = _anonymous_variadic_dim
                else:
                    elem = _anonymous_dim
            else:
                if variadic:
                    elem = _NamedVariadicDim(elem, broadcastable)
                else:
                    elem = _NamedDim(elem, broadcastable)
        else:
            assert dim_type is _DimType.symbolic
            if anonymous:
                raise ValueError(
                    "Cannot have a symbolic dimension be anonymous, e.g. "
                    "`_foo+bar` is not allowed"
                )
            if variadic:
                raise ValueError(
                    "Cannot have symbolic multiple-dimensions, e.g. "
                    "`*foo+bar` is not allowed"
                )
            elem = compile(elem, "<string>", "eval")
            elem = _SymbolicDim(elem, broadcastable)
        dims.append(elem)
    dims = tuple(dims)

    # Allow Python built-in numeric types.
    # TODO: do something more generic than this? Should we _make all types
    # that have `shape` and `dtype` attributes or something?
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
    metaclass = _make_metaclass(type(array_type))
    out = metaclass(
        name,
        (array_type, AbstractArray),
        dict(
            array_type=array_type,
            dtypes=dtypes,
            dims=dims,
            index_variadic=index_variadic,
        ),
    )
    if getattr(typing, "GENERATING_DOCUMENTATION", False):
        out.__module__ = "builtins"
    else:
        out.__module__ = "jaxtyping"
    return out


class _MetaAbstractDtype(type):
    def __instancecheck__(cls, obj: Any) -> NoReturn:
        raise RuntimeError(
            f"Do not use `isinstance(x, jaxtyping.{cls.__name__})`. If you want to "
            "check just the dtype of an array, then use "
            f'`jaxtyping.{cls.__name__}[jnp.ndarray, "..."]`.'
        )

    def __getitem__(cls, item: Tuple[Any, str]):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "As of jaxtyping v0.2.0, type annotations must now include an explicit "
                "array type. For example `jaxtyping.Float32[jnp.ndarray, 'foo bar']`."
            )
        array_type, dim_str = item
        del item
        if typing.get_origin(array_type) in _union_types:
            out = [
                _make_array(x, dim_str, cls.dtypes, cls.__name__)
                for x in typing.get_args(array_type)
            ]
            out = tuple(x for x in out if x is not _not_made)
            out = Union[out]
        else:
            out = _make_array(array_type, dim_str, cls.dtypes, cls.__name__)
            if out is _not_made:
                raise ValueError("Invalid jaxtyping type annotation.")
        return out


class AbstractDtype(metaclass=_MetaAbstractDtype):
    dtypes: Union[Literal[_any_dtype], List[str]]

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "AbstractDtype cannot be instantiated. Perhaps you wrote e.g. "
            '`Float32("shape")` when you mean `Float32[jnp.ndarray, "shape"]`?'
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        dtypes: Union[Literal[_any_dtype], str, List[str]] = cls.dtypes
        if isinstance(dtypes, str):
            dtypes = (dtypes,)
        elif dtypes is not _any_dtype:
            dtypes = tuple(dtypes)
        cls.dtypes = dtypes


if TYPE_CHECKING:
    # Note that `from typing_extensions import Annotated; ... = Annotated`
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
        Annotated as Num,
        Annotated as Shaped,
        Annotated as UInt,
        Annotated as UInt8,
        Annotated as UInt16,
        Annotated as UInt32,
        Annotated as UInt64,
    )
else:
    _bool = "bool"
    _bool_ = "bool_"
    _uint8 = "uint8"
    _uint16 = "uint16"
    _uint32 = "uint32"
    _uint64 = "uint64"
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

    UInt8 = _make_dtype(_uint8, "UInt8")
    UInt16 = _make_dtype(_uint16, "UInt16")
    UInt32 = _make_dtype(_uint32, "UInt32")
    UInt64 = _make_dtype(_uint64, "UInt64")
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
    uints = [_uint8, _uint16, _uint32, _uint64]
    ints = [_int8, _int16, _int32, _int64]
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
    Num = _make_dtype(uints + ints + floats + complexes, "Num")

    Shaped = _make_dtype(_any_dtype, "Shaped")
