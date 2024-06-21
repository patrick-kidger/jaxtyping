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
        "Type-check error whilst checking the parameters of .*<locals>.f",
        "The problem arose whilst typechecking parameter 'z'.",
        "Called with parameters: {'x': 'hi', 'y': 'bye', 'z': 'not-an-int'}",
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
        "Type-check error whilst checking the parameters of .*.<locals>.g",
        "The problem arose whilst typechecking parameter 'y'.",
        r"Called with parameters: {'x': f32\[2,3\], 'y': f32\[4,3\]}",
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
        "Type-check error whilst checking the return value of .*.<locals>.f",
        r"Called with parameters: {'x': \(1, 2\), 'y': {'a': 1}}",
        "Actual value: 'foo'",
        r"Expected type: PyTree\[Any, \"T S\"\].",
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
        "Type-check error whilst checking the parameters of .*.<locals>.M",
        "The problem arose whilst typechecking parameter 'z'.",
        (
            r"Called with parameters: {'self': M\(\.\.\.\), 'x': f32\[2,3\], "
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
