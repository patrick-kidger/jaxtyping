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

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import pytest

from jaxtyping import Float32

from .helpers import ParamError


def g(x: Float32[jnp.ndarray, " b"]):
    pass


g(jnp.array([1.0]))
with pytest.raises(ParamError):
    g(jnp.array(1))


class M(eqx.Module):
    foo: int
    bar: Float32[jnp.ndarray, " a"]


M(1, jnp.array([1.0]))
with pytest.raises(ParamError):
    M(1.0, jnp.array([1.0]))
with pytest.raises(ParamError):
    M(1, jnp.array(1.0))


@dataclasses.dataclass
class D:
    foo: int
    bar: Float32[jnp.ndarray, " a"]


D(1, jnp.array([1.0]))
with pytest.raises(ParamError):
    D(1.0, jnp.array([1.0]))
with pytest.raises(ParamError):
    D(1, jnp.array(1.0))
