import jax.numpy as jnp
import pytest

from jaxtyping import Array, assert_jaxtype, Float, Float32, Int


def test_assert_jaxtype_success():
    x = jnp.empty((3, 4), dtype=jnp.float32)
    result = assert_jaxtype(x, Float32[Array, "3 4"])
    assert result is x


def test_assert_jaxtype_wrong_shape():
    x = jnp.empty((3, 4), dtype=jnp.float32)
    with pytest.raises(AssertionError, match="does not match"):
        assert_jaxtype(x, Float32[Array, "5 6"])


def test_assert_jaxtype_wrong_dtype():
    x = jnp.empty((3, 4), dtype=jnp.float32)
    with pytest.raises(AssertionError, match="does not match"):
        assert_jaxtype(x, Int[Array, "3 4"])


def test_assert_jaxtype_symbolic_shape():
    x = jnp.empty((3, 4), dtype=jnp.float32)
    result = assert_jaxtype(x, Float[Array, "batch dim"])
    assert result is x


def test_assert_jaxtype_any_shape():
    x = jnp.empty((3, 4, 5), dtype=jnp.float32)
    result = assert_jaxtype(x, Float[Array, "..."])
    assert result is x


def test_assert_jaxtype_error_message_contains_shape_and_dtype():
    x = jnp.empty((3, 4), dtype=jnp.float32)
    with pytest.raises(AssertionError, match=r"shape=\(3, 4\)"):
        assert_jaxtype(x, Int[Array, "3 4"])
    with pytest.raises(AssertionError, match=r"dtype=float32"):
        assert_jaxtype(x, Int[Array, "3 4"])
