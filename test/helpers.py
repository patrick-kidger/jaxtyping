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

import equinox as eqx
import typeguard


ParamError = []
ReturnError = []
ParamError.append(TypeError)  # old typeguard
ReturnError.append(TypeError)  # old typeguard

try:
    # new typeguard
    ParamError.append(typeguard.TypeCheckError)
    ReturnError.append(typeguard.TypeCheckError)
except AttributeError:
    pass

try:
    import beartype
except ImportError:
    pass
else:
    ParamError.append(beartype.roar.BeartypeCallHintParamViolation)
    ReturnError.append(beartype.roar.BeartypeCallHintReturnViolation)

ParamError = tuple(ParamError)
ReturnError = tuple(ReturnError)


@eqx.filter_jit
def make_mlp(key):
    return eqx.nn.MLP(2, 2, 2, 2, key=key)
