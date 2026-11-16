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

import importlib
import sys

import pytest

import jaxtyping
from jaxtyping._import_hook import (
    _JaxtypingLoader,
    _optimized_cache_from_source,
    install_import_hook,
    Typechecker,
)


def test_cache_path_omits_debug_override_removed_in_python_315(monkeypatch):
    calls = []

    def cache_from_source(path, *, optimization=None):
        calls.append((path, optimization))
        return path + ".pyc"

    monkeypatch.setattr("jaxtyping._import_hook.cache_from_source", cache_from_source)
    result = _optimized_cache_from_source("abc", "pkg/mod.py", debug_override=False)
    assert result == "pkg/mod.py.pyc"
    assert calls == [("pkg/mod.py", "jaxtyping10abc")]


def test_cache_path_still_forwards_debug_override_when_accepted(monkeypatch):
    calls = []

    def cache_from_source(path, debug_override=None, *, optimization=None):
        calls.append((path, debug_override, optimization))
        return "cached"

    monkeypatch.setattr("jaxtyping._import_hook.cache_from_source", cache_from_source)
    assert _optimized_cache_from_source("abc", "pkg/mod.py", False) == "cached"
    assert calls == [("pkg/mod.py", False, "jaxtyping10abc")]


def test_source_to_code_accepts_module_fullname():
    loader = _JaxtypingLoader("mymodule", "mymodule.py", typechecker=Typechecker(None))
    code = loader.source_to_code(
        b"def f(x: int) -> int:\n    return x\n",
        "mymodule.py",
        "mymodule",
    )
    namespace = {}
    exec(code, namespace)
    assert namespace["f"](1) == 1


def test_import_hook_imports_a_module(tmp_path, monkeypatch):
    module_name = "jaxtyping_py315_hook_mod"
    (tmp_path / f"{module_name}.py").write_text(
        "def f(x: int) -> int:\n    return x\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    with install_import_hook(module_name, None):
        module = importlib.import_module(module_name)
    assert module.f(1) == 1


def test_import_hook_beartype_checks_on_current_python(tmp_path, monkeypatch):
    pytest.importorskip("beartype")
    module_name = "jaxtyping_py315_beartype_mod"
    (tmp_path / f"{module_name}.py").write_text(
        "def f(x: int) -> int:\n    return x\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    with install_import_hook(module_name, "beartype.beartype"):
        module = importlib.import_module(module_name)
    assert module.f(1) == 1
    with pytest.raises(jaxtyping.TypeCheckError):
        module.f("a")
