# Runtime type checking

(See the [FAQ](../faq.md) for details on static type checking.)

Runtime type checking **synergises beautifully with `jax.jit`!** All shape checks will be performed only whilst tracing, and will not impact runtime performance.

There are two approaches: either use [`jaxtyping.jaxtyped`][] to typecheck a single function, or [`jaxtyping.install_import_hook`][] to typecheck a whole codebase.

In either case, the actual business of checking types is performed with the help of a runtime type-checking library. The two most popular are [beartype](https://github.com/beartype/beartype) and [typeguard](https://github.com/agronholm/typeguard). (If using typeguard, then specifically the version `2.*` series should be used. Later versions -- `3` and `4` -- have some known issues.)

!!! warning

    Avoid using `from __future__ import annotations`, or stringified type annotations, where possible. These are largely incompatible with runtime type checking. See also [this FAQ entry](../faq.md#dataclass-annotations-arent-being-checked-properly).

---

::: jaxtyping.jaxtyped

---

::: jaxtyping.install_import_hook

---

#### Pytest hook

The import hook can be installed at test-time only, as a pytest hook. From the command line the syntax is:
```
pytest --jaxtyping-packages=foo,bar.baz,beartype.beartype
```
or in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
addopts = "--jaxtyping-packages=foo,bar.baz,beartype.beartype"
```
or in `pytest.ini`:
```ini
[pytest]
addopts = --jaxtyping-packages=foo,bar.baz,beartype.beartype
```
This example will apply the import hook to all modules whose names start with either `foo` or `bar.baz`. The typechecker used in this example is `beartype.beartype`.

#### IPython extension

If you are running in an IPython environment (for example a Jupyter or Colab notebook), then the jaxtyping hook can be automatically ran via a custom magic:
```python
import jaxtyping
%load_ext jaxtyping
%jaxtyping.typechecker beartype.beartype  # or any other runtime type checker
```
Place this at the start of your notebook -- everything that is directly defined in the notebook, after this magic is run, will be hook'd.

#### Other runtime type-checking libraries

Beartype and typeguard happen to be the two most popular runtime type-checking libraries (at least at time of writing), but jaxtyping should be compatible with all runtime type checkers out-of-the-box. The runtime type-checking library just needs to provide a type-checking decorator (analgous to `beartype.beartype` or `typeguard.typechecked`), and perform `isinstance` checks against jaxtyping's types.
