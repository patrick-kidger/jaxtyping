from typing import Any

import equinox as eqx
import jax.numpy as jnp
import pytest

from jaxtyping import Array, Float, jaxtyped, PyTree, TypeCheckError


def test_arg_localisation(typecheck):
    @jaxtyped(typechecker=typecheck)
    def f(x: str, y: str, z: int):
        pass

    matches = [
        "Type-check error whilst checking the parameters of f",
        "The problem arose whilst typechecking argument 'z'.",
        "Called with arguments: {'x': 'hi', 'y': 'bye', 'z': 'not-an-int'}",
        r"Parameter annotations: \(x: str, y: str, z: int\).",
    ]
    for match in matches:
        with pytest.raises(TypeCheckError, match=match):
            f("hi", "bye", "not-an-int")

    @jaxtyped(typechecker=typecheck)
    def g(x: Float[Array, "a b"], y: Float[Array, "b c"]):
        pass

    x = jnp.zeros((2, 3))
    y = jnp.zeros((4, 3))
    matches = [
        "Type-check error whilst checking the parameters of g",
        "The problem arose whilst typechecking argument 'y'.",
        r"Called with arguments: {'x': f32\[2,3\], 'y': f32\[4,3\]}",
        (
            r"Parameter annotations: \(x: Float\[Array, 'a b'\], y: "
            r"Float\[Array, 'b c'\]\)."
        ),
        "The current values for each jaxtyping axis annotation are as follows.",
        "a=2",
        "b=3",
    ]
    for match in matches:
        with pytest.raises(TypeCheckError, match=match):
            g(x, y=y)


def test_return(typecheck):
    @jaxtyped(typechecker=typecheck)
    def f(x: PyTree[Any, " T"], y: PyTree[Any, " S"]) -> PyTree[Any, "T S"]:
        return "foo"

    x = (1, 2)
    y = {"a": 1}
    matches = [
        "Type-check error whilst checking the return value of f",
        r"Called with arguments: {'x': \(1, 2\), 'y': {'a': 1}}",
        "Return value: 'foo'",
        r"Return annotation: PyTree\[Any, \"T S\"\].",
        (
            "The current values for each jaxtyping PyTree structure annotation are as "
            "follows."
        ),
        r"T=PyTreeDef\(\(\*, \*\)\)",
        r"S=PyTreeDef\({'a': \*}\)",
    ]
    for match in matches:
        with pytest.raises(TypeCheckError, match=match):
            f(x, y=y)


def test_dataclass_attribute(typecheck):
    @jaxtyped(typechecker=typecheck)
    class M(eqx.Module):
        x: Float[Array, " *foo"]
        y: PyTree[Any, " T"]
        z: int

    x = jnp.zeros((2, 3))
    y = (1, (3, 4))
    z = "not-an-int"

    matches = [
        "Type-check error whilst checking the parameters of M",
        "The problem arose whilst typechecking argument 'z'.",
        (
            r"Called with arguments: {'self': M\(\.\.\.\), 'x': f32\[2,3\], "
            r"'y': \(1, \(3, 4\)\), 'z': 'not-an-int'}"
        ),
        (
            r"Parameter annotations: \(self: Any, x: Float\[Array, '\*foo'\], "
            r"y: PyTree\[Any, \"T\"\], z: int\)."
        ),
        "The current values for each jaxtyping axis annotation are as follows.",
        r"foo=\(2, 3\)",
        (
            "The current values for each jaxtyping PyTree structure annotation are as "
            "follows."
        ),
        r"T=PyTreeDef\(\(\*, \(\*, \*\)\)\)",
    ]
    for match in matches:
        with pytest.raises(TypeCheckError, match=match):
            M(x, y, z)
