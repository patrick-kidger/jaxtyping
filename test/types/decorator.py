from dataclasses import dataclass

import tensorflow as tf
from beartype import beartype

from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
@dataclass
class User:
    name: str
    age: int
    items: Float[tf.Tensor, "N"]  # noqa: F821
    timestamps: Int[tf.Tensor, "N"]  # noqa: F821


@jaxtyped(typechecker=beartype)
def transform_user(user: User, increment_age: int = 1) -> User:
    user.age += increment_age
    return user


user = User(
    name="John",
    age=20,
    items=tf.random.normal([10]),
    timestamps=tf.random.uniform([10], minval=0, maxval=100, dtype=tf.int32),
)

new_user = transform_user(user, increment_age=2)
