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

from ._import_hook import JaxtypingTransformer, Typechecker


def choose_typechecker_magics():
    # The import is local to avoid degrading import times when the magic is
    # not needed.
    from IPython.core.magic import line_magic, Magics, magics_class

    @magics_class
    class ChooseTypecheckerMagics(Magics):
        @line_magic("jaxtyping.typechecker")
        def typechecker(self, typechecker):
            # remove old JaxtypingTransformer, if present
            self.shell.ast_transformers = list(
                filter(
                    lambda x: not isinstance(x, JaxtypingTransformer),
                    self.shell.ast_transformers,
                )
            )

            # add new one
            self.shell.ast_transformers.append(
                JaxtypingTransformer(typechecker=Typechecker(typechecker))
            )

    return ChooseTypecheckerMagics


def load_ipython_extension(ipython):
    try:
        ChooseTypecheckerMagics = choose_typechecker_magics()
    except Exception as e:
        # Very broad exception-handling, as e.g. IPython will sometimes be
        # present but fail to import for mysterious reasons.
        raise RuntimeError("Failed to define jaxtyping.typechecker magic") from e

    ipython.register_magics(ChooseTypecheckerMagics)
