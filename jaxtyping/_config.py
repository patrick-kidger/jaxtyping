import os
from typing import Union


def _maybestr2bool(value: Union[bool, str], error: str) -> bool:
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() in ("0", "false"):
            return False
        elif value.lower() in ("1", "true"):
            return True
        else:
            raise ValueError(error)
    else:
        raise ValueError(error)


class _JaxtypingConfig:
    def __init__(self):
        self.update("jaxtyping_disable", os.environ.get("JAXTYPING_DISABLE", "0"))
        self.update(
            "jaxtyping_remove_typechecker_stack",
            os.environ.get("JAXTYPING_REMOVE_TYPECHECKER_STACK", "0"),
        )

    def update(self, item: str, value):
        if item.lower() == "jaxtyping_disable":
            msg = (
                "Unrecognised value for `JAXTYPING_DISABLE`. Valid values are "
                "`JAXTYPING_DISABLE=0` (the default) or `JAXTYPING_DISABLE=1` (to "
                "disable runtime type checking)."
            )
            self.jaxtyping_disable = _maybestr2bool(value, msg)
        elif item.lower() == "jaxtyping_remove_typechecker_stack":
            msg = (
                "Unrecognised value for `JAXTYPING_REMOVE_TYPECHECKER_STACK`. Valid "
                "values are `JAXTYPING_REMOVE_TYPECHECKER_STACK=0` (the default) or "
                "`JAXTYPING_REMOVE_TYPECHECKER_STACK=1` (to remove the stack frames "
                "from the typechecker in `jaxtyped(typechecker=...)`, when it raises a "
                "runtime type-checking error)."
            )
            self.jaxtyping_remove_typechecker_stack = _maybestr2bool(value, msg)
        else:
            raise ValueError(f"Unrecognised config value {item}")


config = _JaxtypingConfig()
