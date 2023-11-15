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
import functools as ft
import inspect
import itertools as it
import sys
import types
import weakref
from typing import Any, get_args, get_origin, get_type_hints, overload


try:
    import jax._src.traceback_util as traceback_util
except ImportError:
    pass
else:
    traceback_util.register_exclusion(__file__)


from ._storage import pop_shape_memo, push_shape_memo


_jaxtyped_fns = weakref.WeakSet()


class TypeCheckError(TypeError):
    pass


TypeCheckError.__module__ = "jaxtyping"  # appears in error messages


@overload
def jaxtyped(*, typechecker=None):
    ...


@overload
def jaxtyped(fn, *, typechecker=None):
    ...


def jaxtyped(fn=None, *, typechecker=None):
    """Decorate a function with this to perform runtime type-checking of its arguments
    and return value. Decorate a dataclass to perform type-checking of its attributes.

    !!! Example

        ```python
        # Import both the annotation and the `jaxtyped` decorator from `jaxtyping`
        from jaxtyping import Array, Float, jaxtyped

        # Use your favourite typechecker: usually one of the two lines below.
        from typeguard import typechecked as typechecker
        from beartype import beartype as typechecker

        # Type-check a function
        @jaxtyped(typechecker=typechecker)
        def batch_outer_product(x: Float[Array, "b c1"],
                                y: Float[Array, "b c2"]
                              ) -> Float[Array, "b c1 c2"]:
            return x[:, :, None] * y[:, None, :]

        # Type-check a dataclass
        @jaxtyped(typechecker=typechecker)
        @dataclass
        class MyDataclass:
            x: int
            y: Float[Array "b c"]
        ```

    **Arguments:**

    - `fn`: The function or dataclass to decorate.
    - `typechecker`: The runtime type-checker to use. This should be a function
        decorator that will raise an exception if there is a type error, e.g.
        ```python
        @typechecker
        def f(x: int):
            pass

        f("a string is not an integer")  # this line should raise an exception
        ```
        Common choices are `typechecker=beartype.beartype` or
        `typechecker=typeguard.typechecked`. Can also be set as `typechecker=None` to
        skip automatic runtime type-checking, but still support manual `isinstance`
        checks inside the function body:
        ```python
        @jaxtyped
        def f(x):
            assert isinstance(x, Float[Array, "batch channel"])
        ```

    **Returns:**

    If `fn` is a function (including a `staticmethod`, `classmethod`, or `property`),
    then a wrapped function is returned.

    If `fn` is a dataclass, then `fn` is returned directly, and additionally its
    `__init__` method is wrapped and modified in-place.

    !!! Info "Old syntax"

        jaxtyping previously (before v0.2.24) recommended using this double-decorator
        syntax:
        ```python
        @jaxtyped
        @typechecker
        def f(...): ...
        ```
        This is still supported, but the `jaxtyped(typechecker=typechecker)` syntax
        discussed above will produce easier-to-debug error messages. Under the hood, the
        new syntax more carefully manipulates the typechecker so as to determine where
        a type-check error arises.

    ??? Info "Notes for advanced users"

        **Dynamic contexts:**

        Put precisely, the axis names in e.g. `Float[Array, "batch channels"]` and the
        structure names in e.g. `PyTree[int, "T"]` are all scoped to the thread-local
        dynamic context of a `jaxtyped`-wrapped function. If from within that function
        we then call another `jaxtyped`-wrapped function, then a new context is pushed
        to the stack. The axis sizes and PyTree structures of this inner function will
        then not be compared against the axis sizes and PyTree structures of the outer
        function. After the inner function returns then this inner context is popped
        from the stack, and the previous context is returned to.

        **isinstance:**

        Binding of a value against a name is done with an `isinstance` check, for
        example `isinstance(jnp.zeros((3, 4)), Float[Array, "dim1 dim2"])` will bind
        `dim1=3` and `dim2=4`. In practice these `isinstance` checks are usually done by
        the run-time typechecker `typechecker` that is supplied as an argument.

        This can also be done manually: add `isinstance` checks inside a function body
        and they will contribute to the same collection of consistency checks as are
        performed by the typechecker on the arguments and return values. (Or you can
        forgo such a typechecker altogether -- i.e. `typechecker=None` -- and only do
        your own manual `isinstance` checks.)

        Only `isinstance` checks that pass will contribute to the store of values; those
        that fail will not. As such it is safe to write e.g.
        `assert not isinstance(x, Float32[Array, "foo"])`.

        **Decoupling contexts from function calls:**

        If you would like a new dynamic context *without* calling a new function, then
        `jaxtyped` may be passed the string `"context"` and used as a context manager:
        ```python
        with jaxtyped("context"):
            assert isinstance(x, Float[Array, "batch channel"])
        ```
        which is equivalent to placing this code inside a new function wrapped in
        `jaxtyped(typechecker=None)`. Usage like this is very rare; it's mostly only
        useful when working at the global scope.

        Conversely, if you would like to call a new function *without* creating a new
        dynamic context (and using the same set of axis and structure values), then
        simply do not add a `jaxtyped` decorator to your inner function, whilst
        continuing to perform type-checking in whatever way you prefer.
    """

    if fn is None:
        return ft.partial(jaxtyped, typechecker=typechecker)
    elif type(fn) is types.FunctionType and fn in _jaxtyped_fns:
        return fn
    elif inspect.isclass(fn):
        if dataclasses.is_dataclass(fn) and typechecker is not None:
            # This does not check that the arguments passed to `__init__` match the
            # type annotations. There may be a custom user `__init__`, or a
            # dataclass-generated `__init__` used alongside
            # `equinox.field(converter=...)`

            init = fn.__init__

            @ft.wraps(init)
            def __init__(self, *args, **kwargs):
                init(self, *args, **kwargs)
                # `fn.__init__` is late-binding to the `__init__` function that
                # we're in now. (Or to someone else's monkey-patch.) Either way,
                # this checks that we're in the "top-level" `__init__`, and not one
                # that is being called via `super()`. We don't want to trigger too
                # early, before all fields have been assigned.
                #
                # We're not checking `if self.__class__ is fn` because Equinox
                # replaces the with a defrozen version of itself during `__init__`,
                # so the check wouldn't trigger.
                #
                # We're not doing this check by adding it to the end of the
                # metaclass `__call__`, because Python doesn't allow you
                # monkey-patch metaclasses.
                if self.__class__.__init__ is fn.__init__:
                    _check_dataclass_annotations(self, typechecker)

            fn.__init__ = __init__
        return fn
    # It'd be lovely if we could handle arbitrary descriptors, and not just the builtin
    # ones. Unfortunately that means returning a class instance with a __get__ method,
    # and that turns out to break loads of other things. See beartype issue #211 and
    # jaxtyping issue #71.
    elif isinstance(fn, classmethod):
        return classmethod(jaxtyped(fn.__func__, typechecker=typechecker))
    elif isinstance(fn, staticmethod):
        return staticmethod(jaxtyped(fn.__func__, typechecker=typechecker))
    elif isinstance(fn, property):
        if fn.fget is None:
            fget = None
        else:
            fget = jaxtyped(fn.fget, typechecker=typechecker)
        if fn.fset is None:
            fset = None
        else:
            fset = jaxtyped(fn.fset, typechecker=typechecker)
        if fn.fdel is None:
            fdel = None
        else:
            fdel = jaxtyped(fn.fdel, typechecker=typechecker)
        return property(fget=fget, fset=fset, fdel=fdel)
    elif fn == "context":
        if typechecker is not None:
            raise ValueError(
                "Cannot use `jaxtyped` as a context with a typechecker. That is, "
                "`with jaxtyped('context', typechecker=...):`. is not allowed. In this "
                "case the type checker does not actually do anything, as there is no "
                "function to type-check."
            )
        return _JaxtypingContext()
    else:
        if typechecker is None:
            # Probably being used in the old style as
            # ```
            # @jaxtyped
            # @typechecker
            # def foo(x: int): ...
            # ```
            # in which case make a best-effort attempt to add shape information for any
            # type errors.

            signature = inspect.signature(fn)

            @ft.wraps(fn)
            def wrapped_fn(*args, **kwargs):  # pyright: ignore
                bound = signature.bind(*args, **kwargs)
                memos = push_shape_memo(bound.arguments)
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if sys.version_info >= (3, 11) and _no_jaxtyping_note(e):
                        shape_info = _exc_shape_info(memos)
                        if shape_info != "":
                            msg = (
                                "The preceding error occurred within the scope of a "
                                "`jaxtyping.jaxtyped` function, and may be due to a "
                                "typecheck error. "
                            )
                            e.add_note(_jaxtyping_note_str(_spacer + msg + shape_info))
                    raise
                finally:
                    pop_shape_memo()

        else:
            # New-style
            # ```
            # @jaxtyped(typechecker=typechecker)
            # def foo(x: int): ...
            # ```
            # in which case we can do a better job reporting errors.

            full_signature = inspect.signature(fn)
            try:
                destring_annotations = get_type_hints(fn, include_extras=True)
            except NameError:
                # Best-effort attempt to destringify annotations.
                pass
            else:
                new_params = []
                for p_name, p_value in full_signature.parameters.items():
                    p_annotation = destring_annotations.get(p_name, p_value.annotation)
                    p_value = p_value.replace(annotation=p_annotation)
                    new_params.append(p_value)
                return_annotation = destring_annotations.get(
                    "return", full_signature.return_annotation
                )
                full_signature = full_signature.replace(
                    parameters=new_params, return_annotation=return_annotation
                )

            param_signature = full_signature.replace(
                return_annotation=inspect.Signature.empty
            )
            module = getattr(fn, "__module__", "generated")

            full_fn, output_name = _make_fn_with_signature(
                "check_return", full_signature, module, output=True
            )
            full_fn = typechecker(full_fn)

            param_fn = _make_fn_with_signature(
                "check_params", param_signature, module, output=False
            )
            param_fn = typechecker(param_fn)

            @ft.wraps(fn)
            def wrapped_fn(*args, **kwargs):
                # Raise bind-time errors before we do any shape analysis. (I.e. skip
                # the pointless jaxtyping information for a non-typechecking failure.)
                bound = param_signature.bind(*args, **kwargs)

                memos = push_shape_memo(bound.arguments)
                try:
                    # First type-check just the parameters before the function is
                    # called.
                    try:
                        param_fn(*args, **kwargs)
                    except Exception as e:
                        if hasattr(e, "_jaxtyping_malformed"):
                            raise
                        else:
                            argmsg = _get_problem_arg(
                                param_signature, args, kwargs, module, typechecker
                            )
                            try:
                                name = fn.__name__
                            except AttributeError:
                                name = fn.__class__.__name__
                            param_values = _pformat(bound.arguments, short_self=True)
                            param_hints = _remove_typing(param_signature)
                            msg = (
                                "Type-check error whilst checking the parameters of "
                                f"{name}.{argmsg}\n"
                                f"Called with arguments: {param_values}\n"
                                f"Parameter annotations: {param_hints}.\n"
                                + _exc_shape_info(memos)
                            )
                            raise TypeCheckError(msg) from e

                    # Actually call the function.
                    out = fn(*args, **kwargs)

                    if full_signature.return_annotation is not inspect.Signature.empty:
                        # Now type-check the return value. We need to include the
                        # parameters in the type-checking here in case there are any
                        # type variables shared across the parameters and return.
                        #
                        # Incidentally this does mean that if `fn` mutates its arguments
                        # so that they no longer satisfy their type annotations, this
                        # will throw an error here. But that's like, super weird, so
                        # don't do that. An error in that scenario is probably still
                        # desirable.
                        #
                        # There is a small performance concern here when used in
                        # non-jit'd contexts, like PyTorch, due to the duplicate
                        # checking of the parameters. Unfortunately there doesn't seem
                        # to be a way around that, so c'est la vie.
                        kwargs[output_name] = out
                        try:
                            full_fn(*args, **kwargs)
                        except Exception as e:
                            if hasattr(e, "_jaxtyping_malformed"):
                                raise
                            else:
                                try:
                                    name = fn.__name__
                                except AttributeError:
                                    name = fn.__class__.__name__
                                param_values = _pformat(
                                    bound.arguments, short_self=True
                                )
                                return_value = _pformat(out, short_self=False)
                                param_hints = _remove_typing(param_signature)
                                return_hint = _remove_typing(
                                    full_signature.return_annotation
                                )
                                if return_hint.startswith(
                                    "<class '"
                                ) and return_hint.endswith("'>"):
                                    return_hint = return_hint[8:-2]
                                msg = (
                                    "Type-check error whilst checking the return value "
                                    f"of {name}.\n"
                                    f"Called with arguments: {param_values}\n"
                                    f"Return value: {return_value}\n"
                                    f"Parameter annotations: {param_hints}.\n"
                                    f"Return annotation: {return_hint}.\n"
                                    + _exc_shape_info(memos)
                                )
                                raise TypeCheckError(msg) from e

                    return out
                finally:
                    pop_shape_memo()

        _jaxtyped_fns.add(wrapped_fn)
        return wrapped_fn


class _JaxtypingContext:
    def __enter__(self):
        push_shape_memo({})

    def __exit__(self, exc_type, exc_value, exc_tb):
        pop_shape_memo()


def _check_dataclass_annotations(self, typechecker):
    """Creates and calls a function that checks the attributes of `self`

    `self` should be a dataclass instancae. `typechecker` should be e.g.
    `beartype.beartype` or `typeguard.typechecked`.
    """
    parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    values = {}
    for field in dataclasses.fields(self):
        annotation = field.type
        if isinstance(annotation, str):
            # Don't check stringified annotations. These are basically impossible to
            # resolve correctly, so just skip them.
            continue
        if get_origin(annotation) is type:
            args = get_args(annotation)
            if len(args) == 1 and isinstance(args[0], str):
                # We also special-case this one kind of partially-stringified type
                # annotation, so as to support Equinox <v0.11.1.
                # This was fixed in Equinox in
                # https://github.com/patrick-kidger/equinox/pull/543
                continue
        try:
            value = getattr(self, field.name)  # noqa: F841
        except AttributeError:
            continue  # allow uninitialised fields, which are allowed on dataclasses

        parameters.append(
            inspect.Parameter(
                field.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=field.type,
            )
        )
        values[field.name] = value

    signature = inspect.Signature(parameters)
    module = self.__class__.__module__
    f = _make_fn_with_signature(
        self.__class__.__name__, signature, module, output=False
    )
    f = jaxtyped(f, typechecker=typechecker)
    f(self, **values)


def _make_fn_with_signature(
    name: str, signature: inspect.Signature, module: str, output: bool
):
    """Dynamically creates a function `fn` with name `name` and signature `signature`.

    If `output=True` then `fn` will consume an additional keyword-only argument (in
    addition to the provided signature), and will directly return this argument. In this
    case the returned value from `_make_fn_with_signature` is a 2-tuple of `(fn, name)`,
    where `fn` is the generated function, and `name` is the name of this extra argument.

    If `output=False` then `fn` will just have a single `pass` statement, and the
    returned value from `_make_fn_with_signature` will just be `fn`.

    ---

    Note that this function operates by dynamically creating and eval'ing a string, not
    simply by assigning `__signature__` and `__annotations__`. The latter is enough for
    typeguard (at least v2), but does not work with beartype (at least v16).
    """
    pos = []
    pos_or_key = []
    varpos = []
    key = []
    varkey = []
    for p in signature.parameters.values():
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            pos.append(p)
        elif p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            pos_or_key.append(p)
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            varpos.append(p)
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            key.append(p)
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            varkey.append(p)
        else:
            assert False

    param_names = frozenset(signature.parameters.keys())
    if output:
        output_name = _gensym(param_names, prefix="ret")
        outstr = "return " + output_name
        param_names = param_names | frozenset({output_name})
        key.append(inspect.Parameter(output_name, kind=inspect.Parameter.KEYWORD_ONLY))
    else:
        outstr = "pass"

    scope = {name: None}
    name_to_annotation = {}
    name_to_default = {}
    param_triples = (
        (p.name, p.annotation, p.default) for p in signature.parameters.values()
    )
    if output:
        triples = it.chain(
            param_triples,
            [
                ("return", signature.return_annotation, inspect.Signature.empty),
                (output_name, Any, inspect.Signature.empty),
            ],
        )
    else:
        triples = it.chain(
            param_triples,
            [("return", signature.return_annotation, inspect.Signature.empty)],
        )
    for p_name, p_annotation, p_default in triples:
        annotation_name = _gensym(frozenset(scope.keys()) | param_names, prefix="T")
        name_to_annotation[p_name] = annotation_name
        if p_annotation is inspect.Signature.empty or isinstance(p_annotation, str):
            # If we have a stringified annotation here it's because the get_type_hints
            # lookup above failed. Typically this occurs when using a local variable as
            # the annotation. In this case we really have no idea what the annotation
            # refers to, so just set it to Any.
            # This does mean that we don't handle partially-stringified local
            # annotations, e.g. `type["Foo"]` for some local type `Foo`. Those will
            # probably just error out. Nothing better we can do about that
            # unfortunately.
            scope[annotation_name] = Any
        else:
            scope[annotation_name] = p_annotation
        default_name = _gensym(frozenset(scope.keys()) | param_names, prefix="default")
        name_to_default[p_name] = default_name
        scope[default_name] = p_default

    argstr_pieces = []
    if len(pos) > 0:
        for p in pos:
            argstr_pieces.append(_make_argpiece(p, name_to_annotation, name_to_default))
        argstr_pieces.append("/")
    if len(pos_or_key) > 0:
        for p in pos_or_key:
            argstr_pieces.append(_make_argpiece(p, name_to_annotation, name_to_default))
    if len(varpos) == 1:
        [p] = varpos
        argstr_pieces.append(
            "*" + _make_argpiece(p, name_to_annotation, name_to_default)
        )
    else:
        assert len(varpos) == 0
        if len(key) > 0:
            argstr_pieces.append("*")
    if len(key) > 0:
        for p in key:
            argstr_pieces.append(_make_argpiece(p, name_to_annotation, name_to_default))
    if len(varkey) == 1:
        [p] = varkey
        argstr_pieces.append(
            "**" + _make_argpiece(p, name_to_annotation, name_to_default)
        )
    else:
        assert len(varkey) == 0
    argstr = ", ".join(argstr_pieces)

    if signature.return_annotation is inspect.Signature.empty:
        retstr = ""
    else:
        retstr = f"-> {name_to_annotation['return']}"

    fnstr = f"def {name}({argstr}){retstr}:\n    {outstr}"
    exec(fnstr, scope)
    fn = scope[name]
    fn.__module__ = module
    assert fn is not None
    if output:
        return fn, output_name
    else:
        return fn


def _gensym(names: frozenset[str], prefix: str) -> str:
    assert prefix.isidentifier()
    output_index = 0
    output_name = prefix + str(output_index)
    while output_name in names:
        output_index += 1
        output_name = prefix + str(output_index)
    assert output_name.isidentifier()
    return output_name


def _make_argpiece(p, name_to_annotation, name_to_default):
    if p.default is inspect.Signature.empty:
        return f"{p.name}: {name_to_annotation[p.name]}"
    else:
        return f"{p.name}: {name_to_annotation[p.name]} = {name_to_default[p.name]}"


def _get_problem_arg(
    param_signature: inspect.Signature, args, kwargs, module, typechecker
) -> str:
    """Determines which argument was likely to be the problematic one responsible for
    raising a type-check error.
    """
    # No performance concerns, as this is only used when we're about to raise an error
    # anyway.
    for keep_name in param_signature.parameters.keys():
        new_parameters = []
        for p_name, p in param_signature.parameters.items():
            if p_name == keep_name:
                new_parameters.append(
                    inspect.Parameter(p.name, p.kind, annotation=p.annotation)
                )
            else:
                new_parameters.append(inspect.Parameter(p.name, p.kind))
        new_signature = inspect.Signature(new_parameters)
        fn = _make_fn_with_signature(
            "check_single_arg", new_signature, module, output=False
        )
        fn = typechecker(fn)  # but no `jaxtyped`; keep the same environment.
        try:
            fn(*args, **kwargs)
        except Exception:
            return f"\nThe problem arose whilst typechecking argument '{keep_name}'."
    else:
        # Could not localise the problem to a single argument -- probably due to
        # e.g. a mismatched typevar, which each individual argument is okay with.
        return ""


def _remove_typing(x):
    x = str(x)
    x = x.replace(" jaxtyping.", " ")
    x = x.replace("[jaxtyping.", "[")
    x = x.replace("'jaxtyping.", "'")
    x = x.replace(" typing.", " ")
    x = x.replace("[typing.", "[")
    x = x.replace("'typing.", "'")
    return x


def _pformat(x, short_self: bool):
    # No performance concerns from delayed imports -- this is only used when we're about
    # to raise an error anyway.
    try:
        # TODO(kidger): this is pretty ugly. We have a circular dependency
        # equinox->jaxtyping->equinox. We could consider moving all the pretty-printing
        # code from equinox into jaxtyping maybe? Or into some shared dependency?
        import equinox as eqx

        pformat = eqx.tree_pformat
        if short_self:
            try:
                self = x["self"]
            except KeyError:
                pass
            else:
                is_self = lambda y: y is self
                pformat = ft.partial(pformat, truncate_leaf=is_self)
    except Exception:
        import pprint

        pformat = ft.partial(pprint.pformat, indent=2, compact=True)
    return pformat(x)


def _exc_shape_info(memos) -> str:
    """Gives debug information on the current state of jaxtyping's internal memos.
    Used in type-checking error messages.
    """
    single_memo, variadic_memo, pytree_memo, _ = memos
    single_memo = {
        name: size
        for name, size in single_memo.items()
        if not name.startswith("~~delete~~")
    }
    variadic_memo = {
        name: shape
        for name, (_, shape) in variadic_memo.items()
        if not name.startswith("~~delete~~")
    }
    pieces = []
    if len(single_memo) > 0 or len(variadic_memo) > 0:
        pieces.append(
            "The current values for each jaxtyping axis annotation are as follows."
        )
        for name, size in single_memo.items():
            pieces.append(f"{name}={size}")
        for name, shape in variadic_memo.items():
            pieces.append(f"{name}={shape}")
    if len(pytree_memo) > 0:
        pieces.append(
            "The current values for each jaxtyping PyTree structure annotation are as "
            "follows."
        )
        for name, structure in pytree_memo.items():
            pieces.append(f"{name}={structure}")
    return "\n".join(pieces)


class _jaxtyping_note_str(str):
    """Used with `_no_jaxtyping_note` to flag that a note came from jaxtyping."""


def _no_jaxtyping_note(e: Exception) -> bool:
    """Checks if any of the exception's notes are from jaxtyping."""
    try:
        notes = e.__notes__
    except AttributeError:
        return True
    else:
        for note in notes:
            if isinstance(note, _jaxtyping_note_str):
                return False
        return True


_spacer = "--------------------\n"
