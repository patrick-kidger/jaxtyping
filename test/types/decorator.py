from dataclasses import dataclass

import numpy as np
from beartype import beartype

from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
@dataclass
class User:
    name: str
    age: int
    items: Float[np.ndarray, " N"]
    timestamps: Int[np.ndarray, " N"]


@jaxtyped(typechecker=beartype)
def transform_user(user: User, increment_age: int = 1) -> User:
    user.age += increment_age
    return user


user = User(
    name="John",
    age=20,
    items=np.random.normal(size=10),
    timestamps=np.random.randint(0, 100, size=10),
)

new_user = transform_user(user, increment_age=2)
