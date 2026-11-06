import dataclasses
import functools as ft

import numpy as np
import pytest

from jaxtyping import Bool, config, jaxtyped, TypeCheckError


@pytest.mark.parametrize("include_message", [None, False, True])
@pytest.mark.parametrize("remove_stack", [False, True])
@pytest.mark.parametrize("check_return", [False, True])
def test_typechecker_message(
    typecheck, monkeypatch, include_message, remove_stack, check_return
):
    monkeypatch.setattr(config, "jaxtyping_remove_typechecker_stack", remove_stack)
    checker_errors = []

    def recording_typechecker(fn):
        checked = typecheck(fn)

        @ft.wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return checked(*args, **kwargs)
            except Exception as e:
                checker_errors.append(e)
                raise

        return wrapped

    def f(x: int) -> int:
        return "bad return" if check_return else x

    options = {}
    if include_message is not None:
        options["include_typechecker_message"] = include_message
    checked = jaxtyped(f, typechecker=recording_typechecker, **options)
    with pytest.raises(TypeCheckError) as exc:
        checked(1 if check_return else "bad parameter")

    message = str(exc.value)
    if include_message:
        # Use the original failure, rather than the single-argument recheck.
        assert message.endswith(f"\nTypechecker details:\n{checker_errors[0]}")
    else:
        assert "Typechecker details:" not in message

    if remove_stack:
        assert exc.value.__cause__ is None
        assert exc.value.__suppress_context__
    elif check_return:
        assert exc.value.__cause__ is checker_errors[0]
    else:
        assert isinstance(exc.value.__cause__, TypeCheckError)
        assert exc.value.__cause__.__cause__ is checker_errors[-1]

    # Default and explicit False produce the same existing message. Enabling the
    # option only appends the checker details.
    with pytest.raises(TypeCheckError) as default_exc:
        jaxtyped(f, typechecker=recording_typechecker)(
            1 if check_return else "bad parameter"
        )
    assert message.split("\nTypechecker details:\n")[0] == str(default_exc.value)


def test_typechecker_message_forwarding(typecheck):
    decorate = jaxtyped(typechecker=typecheck, include_typechecker_message=True)

    @decorate
    @dataclasses.dataclass
    class Data:
        x: int

    class Methods:
        @decorate
        def method(self, x: int):
            pass

        @decorate
        @classmethod
        def class_method(cls, x: int):
            pass

        @decorate
        @staticmethod
        def static_method(x: int):
            pass

        def get_value(self) -> int:
            return "bad return"

        def set_value(self, x: int):
            pass

        def del_value(self) -> int:
            return "bad return"

        value = decorate(property(get_value, set_value, del_value))

    obj = Methods()
    for fn in (Data, obj.method, obj.class_method, obj.static_method):
        fn(1)
        with pytest.raises(TypeCheckError, match="Typechecker details:"):
            fn("bad parameter")

    with pytest.raises(TypeCheckError, match="Typechecker details:"):
        obj.value
    obj.value = 1
    with pytest.raises(TypeCheckError, match="Typechecker details:"):
        obj.value = "bad parameter"
    with pytest.raises(TypeCheckError, match="Typechecker details:"):
        del obj.value


def test_typechecker_message_resolved_shape(beartype_or_skip):
    @jaxtyped(typechecker=beartype_or_skip.beartype, include_typechecker_message=True)
    def f(x: Bool[np.ndarray, "{n}"], n: int):  # noqa: F821
        pass

    f(np.ones(3, dtype=bool), 3)
    with pytest.raises(TypeCheckError, match="dimension size 4.*\\{n\\}=3"):
        f(np.ones(4, dtype=bool), 3)


def test_typechecker_message_body_error(typecheck):
    error = TypeCheckError("error raised by the function body")

    @jaxtyped(typechecker=typecheck, include_typechecker_message=True)
    def f(x: int):
        raise error

    with pytest.raises(TypeCheckError) as exc:
        f(1)
    assert exc.value is error
    assert str(exc.value) == "error raised by the function body"
