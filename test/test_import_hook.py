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

import pytest

from jaxtyping import install_import_hook


def test_import_hook_typeguard():
    hook = install_import_hook(
        "test.import_hook_tester_typeguard", ("typeguard", "typechecked")
    )
    from . import import_hook_tester_typeguard  # noqa: F401

    hook.uninstall()


def test_import_hook_beartype():
    try:
        import beartype  # noqa: F401
    except ImportError:
        pytest.skip("Beartype not installed")
    else:
        hook = install_import_hook(
            "test.import_hook_tester_beartype", ("beartype", "beartype")
        )
        from . import import_hook_tester_beartype  # noqa: F401

        hook.uninstall()


def test_import_hook_transitive():
    hook = install_import_hook(
        "test.import_hook_tester_transitive", ("typeguard", "typechecked")
    )
    from . import import_hook_tester_transitive  # noqa: F401

    hook.uninstall()


def test_import_hook_broken_checker():
    hook = install_import_hook(
        "test.import_hook_tester_broken_checker", ("jaxtyping", "does_not_exist")
    )
    with pytest.raises(AttributeError):
        from . import import_hook_tester_broken_checker  # noqa: F401
    hook.uninstall()
