# PyTree annotations

:::jaxtyping.PyTree
    selection:
        members:
            false

`jaxtyping.PyTreeDef` is an alias for `jax.tree_util.PyTreeDef`, which is the type of the return from `jax.tree_util.tree_structure(...)`.

:::jaxtyping.PyTreeDef

Note that `jaxtyping.{PyTree, PyTreeDef}` are only available if JAX has been installed.
