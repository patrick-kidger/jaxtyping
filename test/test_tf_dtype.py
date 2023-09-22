# Tensorflow dependency kept in a separate file, so that we can optionally exclude it
# more easily.
import tensorflow as tf

from jaxtyping import UInt


def test_tf_dtype():
    x = tf.constant(1, dtype=tf.uint8)
    y = tf.constant(1, dtype=tf.float32)
    hint = UInt[tf.Tensor, "..."]
    assert isinstance(x, hint)
    assert not isinstance(y, hint)
