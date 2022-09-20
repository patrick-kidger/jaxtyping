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

import threading

import jax.numpy as jnp
from typeguard import typechecked

from jaxtyping import Array, Float, jaxtyped


class _ErrorableThread(threading.Thread):
    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exc = e
        finally:
            del self._target, self._args, self._kwargs

    def join(self, timeout=None):
        super().join(timeout)
        if hasattr(self, "exc"):
            raise self.exc


def test_threading_jaxtyped():
    @jaxtyped
    @typechecked
    def add(x: Float[Array, "a b"], y: Float[Array, "a b"]) -> Float[Array, "a b"]:
        return x + y

    def run():
        a = jnp.array([[1.0, 2.0]])
        b = jnp.array([[2.0, 3.0]])
        add(a, b)

    thread = _ErrorableThread(target=run)
    thread.start()
    thread.join()


def test_threading_nojaxtyped():
    def run():
        a = jnp.array([[1.0, 2.0]])
        assert isinstance(a, Float[Array, "..."])

    thread = _ErrorableThread(target=run)
    thread.start()
    thread.join()
