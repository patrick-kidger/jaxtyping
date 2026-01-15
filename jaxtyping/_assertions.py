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

from typing import assert_type, TYPE_CHECKING

from jax import Array


def _assert_jaxtype(x, jaxtype):
    """Asserts that a value has the desired jaxtyping type and returns it.

    Raises an `AssertionError` if the input does not match.

    **Arguments:**

    - `x`: The value to validate.
    - `jaxtype`: The jaxtyping annotation to check against.

    **Returns:**

    The input value `x`, unchanged.
    """
    __tracebackhide__ = True
    if isinstance(x, Array):
        instance_repr = f"Array(shape={x.shape}, dtype={x.dtype})"
    else:
        instance_repr = x
    assert isinstance(x, jaxtype), (
        f"Instance {instance_repr} does not match JAX type {jaxtype}."
    )
    return x


if TYPE_CHECKING:
    assert_jaxtype = assert_type
else:
    assert_jaxtype = _assert_jaxtype
