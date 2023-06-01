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

# This source code is adapted from typeguard:
# https://github.com/agronholm/typeguard/blob/0dd7f7510b7c694e66a0d17d1d58d185125bad5d/src/typeguard/importhook.py
#
# Copied and adapted in compliance with the terms of typeguard's MIT license.
# The original license is reproduced here.
#
# ---------
#
# This is the MIT license: http://www.opensource.org/licenses/mit-license.php
#
# Copyright (c) Alex Grönholm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ---------


import ast
import functools as ft
import sys
from collections.abc import Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import SourceFileLoader
from importlib.util import cache_from_source, decode_source
from inspect import isclass
from typing import Optional, Union
from unittest.mock import patch


# The name of this function is magical
def _call_with_frames_removed(f, *args, **kwargs):
    return f(*args, **kwargs)


def _optimized_cache_from_source(typechecker_hash, /, path, debug_override=None):
    # Version 2: change the position of the `@jaxtyped` decorator, so need a
    #     different name to avoid hitting old __pycache__.
    # Version 3: now also annotating classes.
    # Version 4: I'm honestly not sure, but bumping this fixed some kind of odd error.
    #     Maybe I changed something with hte classes part way through version 3?
    # Version 5: Added support for string-based `typechecker` argument.
    # Version 6: optimization tag now depends on `typechecker` argument, so that
    #    changing the typechecker will hit a different cache.
    return cache_from_source(
        path, debug_override, optimization=f"jaxtyping6{typechecker_hash}"
    )


def _dot_lookup(*elements):
    out = ast.Name(id=elements[0], ctx=ast.Load())
    for element in elements[1:]:
        out = ast.Attribute(out, element, ctx=ast.Load())
    return out


def _str_lookup(string):
    module = ast.parse(string)
    (expr,) = module.body
    return expr.value


class _JaxtypingTransformer(ast.NodeVisitor):
    def __init__(self, *, typechecker) -> None:
        self._parents: list[ast.AST] = []
        self._typechecker = typechecker

    def visit_Module(self, node: ast.Module):
        # Insert "import typeguard; import jaxtping" after any "from __future__ ..."
        # imports
        for i, child in enumerate(node.body):
            if isinstance(child, ast.ImportFrom) and child.module == "__future__":
                continue
            elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Str):
                continue  # module docstring
            else:
                node.body.insert(i, ast.Import(names=[ast.alias("jaxtyping", None)]))
                if self._typechecker is not None:
                    typechecker_module, _ = self._typechecker.split(".", 1)
                    node.body.insert(
                        i, ast.Import(names=[ast.alias(typechecker_module, None)])
                    )
                break

        self._parents.append(node)
        self.generic_visit(node)
        self._parents.pop()
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        func = _dot_lookup("jaxtyping", "_decorator", "_jaxtyped_typechecker")
        if self._typechecker is None:
            args = [ast.Constant(None)]
        else:
            args = [_str_lookup(self._typechecker)]
        node.decorator_list.insert(0, ast.Call(func, args, keywords=[]))
        self._parents.append(node)
        self.generic_visit(node)
        self._parents.pop()
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        has_annotated_args = any(arg for arg in node.args.args if arg.annotation)
        has_annotated_return = bool(node.returns)
        if has_annotated_args or has_annotated_return:
            # Place at the end of the decorator list, as otherwise we wrap e.g.
            # `jax.custom_{jvp,vjp}` and lose the ability to `defjvp` etc.
            #
            # Note that the counter-argument here is that we'd like to place this
            # at the start of the decorator list, in case a typechecking annotation
            # has been manually applied, and we'd need to be above that. In this
            # case we're just going to have to need to ask the user to remove their
            # typechecking annotation (and let this decorator do it instead).
            # It's more important we be compatible with normal JAX code.
            #
            #
            # FWIW, typeguard also wants to be at the end of the decorator list, as it
            # works by recompiling the wrapped function.
            node.decorator_list.append(_dot_lookup("jaxtyping", "jaxtyped"))
            if self._typechecker is not None:
                # Place at the end of the decorator list, as decorators
                # frequently remove annotations from functions and we'd like to
                # use those annotations.
                node.decorator_list.append(_str_lookup(self._typechecker))
        self._parents.append(node)
        self.generic_visit(node)
        self._parents.pop()
        return node


class _JaxtypingLoader(SourceFileLoader):
    def __init__(self, *args, typechecker, **kwargs):
        super().__init__(*args, **kwargs)
        self._typechecker = typechecker
        self._typechecker_hash = str(abs(hash(self._typechecker)))

    def source_to_code(self, data, path, *, _optimize=-1):
        source = decode_source(data)
        tree = _call_with_frames_removed(
            compile,
            source,
            path,
            "exec",
            ast.PyCF_ONLY_AST,
            dont_inherit=True,
            optimize=_optimize,
        )
        tree = _JaxtypingTransformer(typechecker=self._typechecker).visit(tree)
        ast.fix_missing_locations(tree)
        return _call_with_frames_removed(
            compile, tree, path, "exec", dont_inherit=True, optimize=_optimize
        )

    def exec_module(self, module):
        # Use a custom optimization marker - the import lock should make this monkey
        # patch safe
        with patch(
            "importlib._bootstrap_external.cache_from_source",
            ft.partial(_optimized_cache_from_source, self._typechecker_hash),
        ):
            return super().exec_module(module)


class _JaxtypingFinder(MetaPathFinder):
    """Wraps another path finder and instruments the module with `@jaxtyped` and
    `@typechecked` if `should_instrument()` returns `True`.

    Should not be used directly, but rather via `install_import_hook`.
    """

    def __init__(self, modules, original_pathfinder, typechecker):
        self.modules = modules
        self._original_pathfinder = original_pathfinder
        self._typechecker = typechecker

    def find_spec(self, fullname, path=None, target=None):
        if self.should_instrument(fullname):
            spec = self._original_pathfinder.find_spec(fullname, path, target)
            if spec is not None and isinstance(spec.loader, SourceFileLoader):
                spec.loader = _JaxtypingLoader(
                    spec.loader.name, spec.loader.path, typechecker=self._typechecker
                )
                return spec

        return None

    def should_instrument(self, module_name: str) -> bool:
        """Determine whether the module with the given name should be instrumented.

        **Arguments:**

        - `module_name`: the full name of the module that is about to be imported
            (e.g. ``xyz.abc``)
        """
        for module in self.modules:
            if module_name == module or module_name.startswith(module + "."):
                return True

        return False


class ImportHookManager:
    def __init__(self, hook: MetaPathFinder):
        self.hook = hook

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninstall()

    def uninstall(self):
        try:
            sys.meta_path.remove(self.hook)
        except ValueError:
            pass  # already removed


# Deliberately no default for `typechecker` so that folks must opt-in to not having
# a typechecker.
def install_import_hook(modules: Union[str, Sequence[str]], typechecker: Optional[str]):
    """Automatically apply `@jaxtyped`, and optionally a type checker, as decorators.

    !!! Tip "Usage"

        ```python
        from jaxtyping import install_import_hook
        # Plus any one of the following:

        # decorate @jaxtyped and @typeguard.typechecked
        with install_import_hook("foo", "typeguard.typechecked"):
            import foo          # Any module imported inside this `with` block, whose
            import foo.bar      # name begins with the specified string, will
            import foo.bar.qux  # automatically have both `@jaxtyped` and the specified
                                # typechecker applied to all of their functions.

        # decorate @jaxtyped and @beartype.beartype
        with install_import_hook("foo", "beartype.beartype"):
            ...

        # decorate only @jaxtyped (if you want that for some reason)
        with install_import_hook("foo", None):
            ...
        ```

        If you don't like using the `with` block, the hook can be used without that:
        ```python
        hook = install_import_hook(...):
        import ...
        hook.uninstall()
        ```

        The import hook can be applied to multiple packages via
        ```python
        install_import_hook(["foo", "bar.baz"], ...)
        ```

    The import hook will automatically decorate all functions, and the `__init__` method
    of dataclasses.

    If the function already has any decorators on it, then both the `@jaxtyped` and the
    typechecker decorators will get added at the bottom of the decorator list, e.g.
    ```python
    @some_other_decorator
    @jaxtyped
    @beartype.beartype
    def foo(...): ...
    ```

    **Arguments:**:

    - `modules`: the names of the modules in which to automatically apply `@jaxtyped`
        and `@typechecked`.
    - `typechecker`: the module and function of the typechecker you want to use, as a
        string. For example `typechecker="typeguard.typechecked"`, or
        `typechecker="beartype.beartype"`. You may pass `typechecker=None` if you do not
        want to automatically decorate with a typechecker as well.

    **Returns:**

    A context manager that uninstalls the hook on exit, or when you call `.uninstall()`.

    ??? Example "Example: end-user script"

        ```python
        ### entry_point.py
        from jaxtyping import install_import_hook
        with install_import_hook("main", "typeguard.typechecked"):
            import main

        ### main.py
        from jaxtyping import Array, Float32

        def f(x: Float32[Array, "batch channels"]):
            ...
        ```

    ??? Example "Example: writing a library"

        ```python
        ### __init__.py
        from jaxtyping import install_import_hook
        with install_import_hook("my_library_name", "beartype.beartype"):
            from .subpackage import foo  # full name is my_library_name.subpackage so
                                         # will be hook'd
            from .another_subpackage import bar  # full name is my_library_name.another_subpackage
                                                 # so will be hook'd.
        ```

    ??? info "Pytest hook"

        The import hook can be installed at test-time only, as a pytest hook. From the
        command line the syntax is:
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
        This example will apply the import hook to all modules whose names start with
        either `foo` or `bar.baz`. The typechecker used in this example is
        `beartype.beartype`.

        (This is the author's preferred approach to performing runtime type-checking
        with jaxtyping!)
    """  # noqa: E501

    if isinstance(modules, str):
        modules = [modules]

    # Support old less-flexible API.
    if isinstance(typechecker, tuple):
        typechecker = ".".join(typechecker)

    for i, finder in enumerate(sys.meta_path):
        if (
            isclass(finder)
            and finder.__name__ == "PathFinder"
            and hasattr(finder, "find_spec")
        ):
            break
    else:
        raise RuntimeError("Cannot find a PathFinder in sys.meta_path")

    hook = _JaxtypingFinder(modules, finder, typechecker)
    sys.meta_path.insert(0, hook)
    return ImportHookManager(hook)
