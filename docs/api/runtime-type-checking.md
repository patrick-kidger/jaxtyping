# Runtime type checking

(See the [FAQ](../faq.md) for details on static type checking.)

Runtime type checking **synergises beautifully with `jax.jit`!** All shape checks will be performed at trace-time only, and will not impact runtime performance.

Runtime type-checking should be performed using a library like [typeguard](https://github.com/agronholm/typeguard) or [beartype](https://github.com/beartype/beartype).

The types provided by `jaxtyping`, e.g. `Float[Array, "batch channels"]`, are all compatible with `isinstance` checks, e.g. `isinstance(x, Float[Array, "batch channels"])`. This means that jaxtyping should be compatible with all runtime type checkers out-of-the-box.

Some additional context is needed to ensure consistency between multiple argments (i.e. that shapes match up between arrays). For this, you can use either `jaxtyping.jaxtyped` to add this capability to a single function, or `jaxtyping.install_import_hook` to add this capability to a whole codebase. If either are too much magic for you, you can safely use neither and have just single-argument type checking.

::: jaxtyping.jaxtyped

---

It can be a lot of effort to add `@jaxtyped` decorators all over your codebase.
(Not to mention that double-decorators everywhere are a bit ugly.)

The easier option is usually to use the import hook.

::: jaxtyping.install_import_hook
