# PyTree annotations

:::jaxtyping.PyTree
    selection:
        members:
            false

---

:::jaxtyping.PyTreeDef

---

## Path-dependent shapes

The prefix `?` may be used to indicate that the axis size can depend on which leaf of a PyTree the array is at. For example:
```python
def f(
    x: PyTree[Shaped[Array, "?foo"], "T"],
    y: PyTree[Shaped[Array, "?foo"], "T"],
):
    pass
```
The above demands that `x` and `y` have matching PyTree structures (due to the `T` annotation), and that their leaves must all be one-dimensional arrays, *and that the corresponding pairs of leaves in `x` and `y` must have the same size as each other*.

Thus the following is allowed:
```python
x0 = jnp.arange(3)
x1 = jnp.arange(5)

y0 = jnp.arange(3) + 1
y1 = jnp.arange(5) + 1

f((x0, x1), (y0, y1))  # x0 matches y0, and x1 matches y1. All good!
```

But this is not:
```python
f((x1, x1), (y0, y1))  # x1 does not have a size matching y0!
```

Internally, all that is happening is that `foo` is replaced with `0foo` for the first leaf, `1foo` for the next leaf, etc., so that each leaf gets a unique version of the name.

---

Note that `jaxtyping.{PyTree, PyTreeDef}` are only available if JAX has been installed.
