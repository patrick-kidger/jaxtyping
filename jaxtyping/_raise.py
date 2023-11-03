from typing import NoReturn


def jaxtyping_raise(e) -> NoReturn:
    """Raises `e`, whilst adding a tag that it should not be intercepted by
    `TypeCheckError`. All `raise` statements from within `__instancecheck__` should use
    this.
    """
    __tracebackhide__ = True
    try:
        raise e
    except Exception as f:
        f._jaxtyping_malformed = True
        raise


def jaxtyping_raise_from(e, e_base) -> NoReturn:
    __tracebackhide__ = True
    try:
        raise e from e_base
    except Exception as f:
        f._jaxtyping_malformed = True
        raise
