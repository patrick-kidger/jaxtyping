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
import importlib.metadata
import pathlib
import shutil
import sys
import tempfile

import pytest

import jaxtyping


_here = pathlib.Path(__file__).resolve().parent


try:
    typeguard_version = importlib.metadata.version("typeguard")
except Exception as e:
    raise ImportError("Could not find typeguard version") from e
else:
    try:
        major, _, _ = typeguard_version.split(".")
        major = int(major)
    except Exception as e:
        raise ImportError(
            f"Unexpected typeguard version {typeguard_version}; not formatted as "
            "`major.minor.patch`"
        ) from e
if major != 2:
    raise ImportError(
        "jaxtyping's tests required typeguard version 2. (Versions 3 and 4 are both "
        "known to have bugs.)"
    )


assert not hasattr(jaxtyping, "_test_import_hook_counter")
jaxtyping._test_import_hook_counter = 0


@pytest.fixture(scope="module")
def importhook_tempdir():
    with tempfile.TemporaryDirectory() as dir:
        sys.path.append(dir)
        dir = pathlib.Path(dir)
        shutil.copyfile(_here / "helpers.py", dir / "helpers.py")
        yield dir


def _test_import_hook(importhook_tempdir, typechecker):
    counter = jaxtyping._test_import_hook_counter
    stem = f"import_hook_tester{counter}"
    shutil.copyfile(_here / "import_hook_tester.py", importhook_tempdir / f"{stem}.py")
    with jaxtyping.install_import_hook(stem, typechecker):
        importlib.import_module(stem)
    assert counter + 1 == jaxtyping._test_import_hook_counter


def test_import_hook_typeguard_old(importhook_tempdir):
    _test_import_hook(importhook_tempdir, ("typeguard", "typechecked"))


def test_import_hook_typeguard(importhook_tempdir):
    _test_import_hook(importhook_tempdir, "typeguard.typechecked")


def test_import_hook_beartype_old(importhook_tempdir):
    try:
        import beartype  # noqa: F401
    except ImportError:
        pytest.skip("Beartype not installed")
    else:
        _test_import_hook(importhook_tempdir, ("beartype", "beartype"))


def test_import_hook_beartype(importhook_tempdir):
    try:
        import beartype  # noqa: F401
    except ImportError:
        pytest.skip("Beartype not installed")
    else:
        _test_import_hook(importhook_tempdir, "beartype.beartype")


def test_import_hook_beartype_full(importhook_tempdir):
    try:
        import beartype  # noqa: F401
    except ImportError:
        pytest.skip("Beartype not installed")
    else:
        bearchecker = "beartype.beartype(conf=beartype.BeartypeConf(strategy=beartype.BeartypeStrategy.On))"  # noqa: E501
        _test_import_hook(importhook_tempdir, bearchecker)


def test_import_hook_broken_checker(importhook_tempdir):
    with pytest.raises(AttributeError):
        _test_import_hook(importhook_tempdir, "jaxtyping.does_not_exist")


def test_import_hook_transitive(importhook_tempdir):
    typechecker = "typeguard.typechecked"
    counter = jaxtyping._test_import_hook_counter
    transitive_name = "jaxtyping_transitive_test"
    transitive_dir = importhook_tempdir / transitive_name
    transitive_dir.mkdir()
    shutil.copyfile(_here / "import_hook_tester.py", transitive_dir / "tester.py")
    with open(transitive_dir / "__init__.py", "w") as f:
        f.write("from . import tester")
        f.flush()

    importlib.invalidate_caches()
    with jaxtyping.install_import_hook(transitive_name, typechecker):
        importlib.import_module(transitive_name)
    assert counter + 1 == jaxtyping._test_import_hook_counter
