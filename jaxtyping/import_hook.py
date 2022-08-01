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
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import SourceFileLoader
from importlib.util import cache_from_source, decode_source
from inspect import isclass
from typing import Iterable, List, Optional, Tuple
from unittest.mock import patch


# The name of this function is magical
def _call_with_frames_removed(f, *args, **kwargs):
    return f(*args, **kwargs)


def _optimized_cache_from_source(path, debug_override=None):
    return cache_from_source(path, debug_override, optimization="jaxtyping")


class _JaxtypingTransformer(ast.NodeVisitor):
    def __init__(self, *, typechecker) -> None:
        self._parents: List[ast.AST] = []
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
                    typechecker_module, _ = self._typechecker
                    node.body.insert(
                        i, ast.Import(names=[ast.alias(typechecker_module, None)])
                    )
                break

        self._parents.append(node)
        self.generic_visit(node)
        self._parents.pop()
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        has_annotated_args = any(arg for arg in node.args.args if arg.annotation)
        has_annotated_return = bool(node.returns)
        if has_annotated_args or has_annotated_return:
            # Place at the start of the decorator list, in case a typechecking
            # annotation has been manually applied; we need to be above that.
            node.decorator_list.insert(
                0,
                ast.Attribute(
                    ast.Name(id="jaxtyping", ctx=ast.Load()), "jaxtyped", ast.Load()
                ),
            )
            if self._typechecker is not None:
                # Place at the end of the decorator list, as decorators
                # frequently remove annotations from functions and we'd like to
                # use those annotations.
                typechecker_module, typechecker_function = self._typechecker
                node.decorator_list.append(
                    ast.Attribute(
                        ast.Name(id=typechecker_module, ctx=ast.Load()),
                        typechecker_function,
                        ast.Load(),
                    )
                )

        self._parents.append(node)
        self.generic_visit(node)
        self._parents.pop()
        return node


class _JaxtypingLoader(SourceFileLoader):
    def __init__(self, *args, typechecker, **kwargs):
        super().__init__(*args, **kwargs)
        self._typechecker = typechecker

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
            _optimized_cache_from_source,
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
def install_import_hook(
    modules: Iterable[str], typechecker: Optional[Tuple[str, str]]
) -> ImportHookManager:
    """Automatically apply `@jaxtyped`, and optionally a type checker, to all classes
    and functions.

    It will only be applied to modules loaded **after** this hook has been installed.

    **Arguments:**:

    - `packages`: the names of the modules in which to automatically apply `@jaxtyped`
        and `@typechecked`.
    - `typechecker`: the module and function of the typechecker you want to use, as a
        2-tuple of strings. For example `typechecker=("typeguard", "typechecked")` or
        `typechecker=("beartype", "beartype")`. You may pass `typechecker=None` if you
        do not want to automatically decorate with a typechecker as well; e.g. if you
        have a codebase that already has these decorators.

    **Returns:**

    A context manager that uninstalls the hook on exit, or when you call `.uninstall()`.

    **Example:**

    Typically you should apply this import hook at the entry point for your own scripts:
    ```python
    # entry_point.py
    from jaxtyped import install_import_hook
    install_import_hook("main", ("beartype", "beartype"))
    import main
    ...  # do whatever you're doing

    # main.py
    from jaxtyped import f32

    def f(x: f32["b c"]):
        pass
    ```
    Which as you can see means you never to import `@jaxtyped`, nor do you need to
    import the typechecker directly (e.g. `beartype.beartype` or
    `typeguard.typechecked`).
    """
    if isinstance(modules, str):
        modules = [modules]

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
