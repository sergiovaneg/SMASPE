"""
Python module containing the specialized loss functions required for the SMASPE
paper.
"""

import numpy as np
import keras
from keras import ops

EPS = keras.backend.epsilon()


@keras.saving.register_keras_serializable("SMASPE")
class MAAPE(keras.Loss):
  def call(self, y_true, y_pred):
    return ops.mean(
        ops.arctan(
            ops.abs(y_true - y_pred) /
            ops.clip(ops.abs(y_true), EPS, np.inf)
        ),
        axis=-1
    )


def maspe(y_true, y_pred):
  return ops.mean(
      ops.arctan(
          ops.square(y_true - y_pred) /
          ops.clip(ops.square(y_true), EPS, np.inf)
      ),
      axis=-1
  )


@keras.saving.register_keras_serializable("SMASPE")
class MASPE(keras.Loss):
  def call(self, y_true, y_pred):
    return maspe(y_true, y_pred)


@keras.saving.register_keras_serializable("SMASPE")
class SMASPE(keras.Loss):
  """
  Class implementation of the SMASPE to make it serializable
  """

  def __init__(
      self,
      y_minus: float = 0.,
      y_plus: float = 1.,
      name=None,
      reduction="sum_over_batch_size",
      dtype=None
  ):
    super().__init__(name, reduction, dtype)
    self.y_minus, self.y_plus = y_minus, y_plus

  def call(self, y_true, y_pred):
    return maspe(
        y_true - self.y_minus, y_pred - self.y_minus
    ) + maspe(
        self.y_plus - y_true, self.y_plus - y_pred
    )
