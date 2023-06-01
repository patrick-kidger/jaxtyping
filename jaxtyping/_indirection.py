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
    Annotated as Int8,  # noqa: F401
    Annotated as Int16,  # noqa: F401
    Annotated as Int32,  # noqa: F401
    Annotated as Int64,  # noqa: F401
    Annotated as Integer,  # noqa: F401
    Annotated as Key,  # noqa: F401
    Annotated as Num,  # noqa: F401
    Annotated as Shaped,  # noqa: F401
    Annotated as UInt,  # noqa: F401
    Annotated as UInt8,  # noqa: F401
    Annotated as UInt16,  # noqa: F401
    Annotated as UInt32,  # noqa: F401
    Annotated as UInt64,  # noqa: F401
)

from jax import Array as Scalar  # noqa: F401
from jax.typing import ArrayLike as ScalarLike  # noqa: F401
