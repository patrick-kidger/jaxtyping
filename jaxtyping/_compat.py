"""Compatibility layer for JAX versions and features."""

import re
from importlib.metadata import version
from typing import Final, TYPE_CHECKING, TypeAlias

import jax


# The code under `if TYPE_CHECKING:` is evaluated by static type checkers (mypy,
# pyright) but is *not* executed at runtime. We use it to make the names
# `TypedNdArray` and `LiteralArray` always visible to the checker regardless of
# the installed JAX version:
#   * First, we *try* to import the real classes from `jax._src.literals`.
#   * If those imports fail (e.g., the name does not exist in this JAX version),
#     we define small local stub classes with the same names. These stubs give
#     the checker a symbol to reference without affecting runtime behavior.
if TYPE_CHECKING:
    try:
        from jax._src.literals import TypedNdArray  # JAX ≥ v0.7.3
    except Exception:  # pragma: no cover - used only for type checking

        class TypedNdArray:  # fallback stub for older JAX
            ...

    try:
        from jax._src.literals import LiteralArray  # 0.7.2
    except Exception:  # pragma: no cover

        class LiteralArray:  # fallback stub
            ...

    ArrayLike: TypeAlias = jax.typing.ArrayLike | TypedNdArray | LiteralArray


# At runtime, `ArrayLike` is built dynamically:
# * Starting from `jax.typing.ArrayLike`.
# * Extending with `TypedNdArray` if JAX ≥ 0.7.3.
# * Extending with `LiteralArray` if JAX = 0.7.2.
#
# This ensures that the union contains real classes with correct `__module__`
# values for tools like beartype. Older versions fall back to just
# `jax.typing.ArrayLike`.
else:
    # The following regex extracts the leading numeric groups from the JAX
    # version string, pads with zeros if needed, slices to three components, and
    # converts to integers. This approach is robust to dev, post, or local
    # version tags (e.g. '0.7.11+dev', '0.7.3.dev0', etc.)
    JAX_VERSION: Final = tuple(
        int(x) for x in (re.findall(r"\d+", version("jax")) + ["0", "0", "0"])[:3]
    )

    ArrayLike = jax.typing.ArrayLike  # type: ignore[assignment]

    if JAX_VERSION >= (0, 7, 3):
        # JAX 0.7.3+ has `TypedNdArray`.
        from jax._src.literals import TypedNdArray

        ArrayLike = ArrayLike | TypedNdArray
    elif JAX_VERSION >= (0, 7, 2):
        # JAX 0.7.2 has `LiteralArray`.
        from jax._src.literals import LiteralArray

        ArrayLike = ArrayLike | LiteralArray
