from ._import_hook import _JaxtypingTransformer


try:
    from IPython.core.magic import line_magic, Magics, magics_class

    @magics_class
    class ChooseTypecheckerMagics(Magics):
        @line_magic("jaxtyping.typechecker")
        def typechecker(self, typechecker):
            # remove old _JaxtypingTransformer, if present
            self.shell.ast_transformers = list(
                filter(
                    lambda x: not isinstance(x, _JaxtypingTransformer),
                    self.shell.ast_transformers,
                )
            )

            # add new one
            self.shell.ast_transformers.append(
                _JaxtypingTransformer(typechecker=typechecker)
            )

except ImportError:
    pass


def load_ipython_extension(ipython):
    try:
        ipython.register_magics(ChooseTypecheckerMagics)
    except NameError:
        raise NameError(
            "ChooseTypecheckerMagics is not defined.\n\n"
            + "You may be trying to use IPython extension without IPython installed."
        )
