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

import sys

from ._import_hook import install_import_hook


def pytest_addoption(parser):
    group = parser.getgroup("jaxtyping")
    group.addoption(
        "--jaxtyping-packages",
        action="store",
        help="comma separated name list of packages and modules to instrument for "
        "type checking with jaxtyping. The last element in the list should be the "
        "type checker to use, e.g. "
        "--jaxtyping-packages=foopackage,barpackage,typeguard.typechecked",
    )


def pytest_load_initial_conftests(early_config, parser, args):
    # We run this function before conftest.py files are loaded,
    # so we can instrument imports before any code is run.
    del early_config
    options = parser.parse_known_args(args)
    value = options.jaxtyping_packages
    if not value:
        return

    maxsplit = -1
    # We avoid splitting on commas inside of the typechecker constructor
    # (e.g. `beartype.beartype(option_a=..., option_b=...)`)
    if index := value.find("(") != -1:
        maxsplit = value[:index].count(",") + 1

    packages = [pkg.strip() for pkg in value.split(",", maxsplit=maxsplit)]
    *packages, typechecker = packages

    already_imported_packages = sorted(
        package for package in packages if package in sys.modules
    )
    if already_imported_packages:
        message = (
            "jaxtyping cannot check these packages because they "
            "are already imported: {}"
        )
        raise RuntimeError(message.format(", ".join(already_imported_packages)))

    install_import_hook(packages, typechecker)
