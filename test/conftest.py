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

import random

import jax.random as jr
import pytest
import typeguard


try:
    import beartype
except ImportError:

    def skip(*args, **kwargs):
        pytest.skip("Beartype not installed")

    typecheck_params = [typeguard.typechecked, skip]
else:
    typecheck_params = [typeguard.typechecked, beartype.beartype]


@pytest.fixture(params=typecheck_params)
def typecheck(request):
    return request.param


@pytest.fixture()
def getkey():
    def _getkey():
        # Not sure what the maximum actually is but this will do
        return jr.PRNGKey(random.randint(0, 2**31 - 1))

    return _getkey


@pytest.fixture(scope="module")
def beartype_or_skip():
    yield pytest.importorskip("beartype")


@pytest.fixture(scope="module")
def typeguard_or_skip():
    yield pytest.importorskip("typeguard")
