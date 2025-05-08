"""
Python module implementing the layers of a Canonical
Transformer in Keras.
"""

from collections.abc import Sequence

import keras
from keras import Input
from keras import KerasTensor as Tensor
from keras import layers

from . import embed
from . import recurrent

from pydantic import Field
from typing import Annotated, Optional

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]


@keras.saving.register_keras_serializable(package="GenericTransformer")
class GenericTransformer(layers.Layer):
  """
  Keras implementation of a Generic Transformer architecture
  """

  def __init__(
      self,
      encoder: layers.Layer,
      decoder: layers.Layer,
      output_attention: bool = False,
      **kwargs
  ):
    super(GenericTransformer, self).__init__(**kwargs)

    # wether to output the model attention
    self.output_attention = output_attention

    self.encoder = encoder
    self.decoder = decoder

  def call(
      self,
      inputs: tuple[Tensor | tuple[Tensor], ...]
  ) -> Tensor | tuple[Tensor | tuple[Tensor], ...]:
    # Enc Input -> Dec Input
    x_enc, x_enc_marks, x_dec, x_dec_marks = inputs

    # Enc call
    y_enc, attns = self.encoder([x_enc, x_enc_marks])
    # Dec call
    y = self.decoder([x_dec, x_dec_marks, y_enc])

    if isinstance(y, Sequence):
      y = list(y)
    else:
      y = [y]

    if self.output_attention:
      y.append(keras.ops.stack(attns, axis=2))

    if len(y) == 1:
      y = y[0]

    return y

  def get_config(self):
    config = super().get_config()

    config.update({
        "output_attention": self.output_attention,
        "encoder": keras.saving.serialize_keras_object(self.encoder),
        "decoder": keras.saving.serialize_keras_object(self.decoder)
    })

    return config

  @classmethod
  def from_config(cls, config):
    encoder_config = config.pop("encoder")
    encoder = keras.saving.deserialize_keras_object(encoder_config)

    decoder_config = config.pop("decoder")
    decoder = keras.saving.deserialize_keras_object(decoder_config)

    return cls(encoder, decoder, **config)


def get_mark_inputs(params: dict) -> tuple[Tensor, Tensor]:
  if params["embed_type"] is None:
    x_enc_marks = keras.ops.zeros(
        [1, params["I"], 1]
    )
    x_dec_marks = keras.ops.zeros(
        [1, params["O"], 1]
    )
  else:
    if params["embed_type"] == "temporal":
      # Hardcoded '4 (5)' for [Mon, DoM, DoW, Hour, (Min)]
      if params["freq"] == "t":
        x_enc_marks_shape = (params["I"], 5)
        x_dec_marks_shape = (params["O"], 5)
      elif params["freq"] == "h":
        x_enc_marks_shape = (params["I"], 4)
        x_dec_marks_shape = (params["O"], 4)
      else:
        x_enc_marks_shape = (params["I"], 0)
        x_dec_marks_shape = (params["O"], 0)
    else:
      x_enc_marks_shape = (params["I"], 1)
      x_dec_marks_shape = (params["O"], 1)

    x_enc_marks = Input(
        shape=x_enc_marks_shape,
        name="xm_enc_input",
        dtype="uint32"
    )
    x_dec_marks = Input(
        shape=x_dec_marks_shape,
        name="xm_dec_input",
        dtype="uint32"
    )

  return x_enc_marks, x_dec_marks


def create_recurrent_model(
    *, I: int, O: int,
    d_enc: int, d_dec: int, d_out: int,
    d_model: int = 512, d_ff: int = 512,
    h: int = 16,
    N: Optional[int] = None, M: Optional[int] = 1,
    n_blocks: int = 1,
    rnn_type: str = "lstm",
    dropout_rate: float = 0.,
    output_attention: bool = False,
    embed_type: str = "none",
    freq: str = "t",
    normalize=True,
    **kw  # pylint: disable=unused-argument
) -> keras.Model:
  """Create a Sequence2Sequence Encoder-Decoder model using recurrent kernels

  Helper function that takes an argument list an instantiates the necesary
  layers and inputs for an Encoder-Decoder recurrent architecture using the
  Functional interface from the Keras library.

  Args:
    I:
      Encoder Input sequence length.
    O:
      Decoder Input and Output sequence length.
    d_enc:
      Number of Encoder Input channels.
    d_dec:
      Number of Decoder Input channels.
    d_out:
      Number of Output channels.
    d_model:
      Model embedding depth.
    d_ff:
      Dimension of units/neurons per Feed-Forward layer.
    h:
      Recurrent units (heads) per block.
    N:
      Number of Encoder Blocks.
    M:
      Number of Decoder Blocks.
    n_blocks:
      Alternative way to define the same number for Encoder/Decoder block count.
    dropout_rate:
      Self-explanatory, applied between attention and feed-forward layers.
    output_attention:
      Boolean flag indicating whether to output the attention tensor from the
      Encoder stack.
    embed_type:
      String indicating which kind ("temporal", "fixed", "positional",
      or None) of embedding to use. "None" not recommended for
      this architecture.
    freq:
      In the case of temporal encoding, whether the minute stamps are included
      in the marks' Tensor.
    normalize:
      Boolean flag indicating whether to normalize the sequences at every
      encoder/decoder block.
  Returns:
    A Keras Model instance representing the Recurrent model instance.
  """

  restore_custom_objects()

  params = {
      "I": I,
      "O": O,
      "d_enc": d_enc,
      "d_dec": d_dec,
      "d_out": d_out,
      "d_model": d_model,
      "d_ff": d_ff,
      "h": h,
      "N": N or n_blocks,
      "M": M or n_blocks,
      "rnn_type": rnn_type,
      "dropout_rate": dropout_rate,
      "output_attention": output_attention,
      "embed_type": embed_type,
      "freq": freq,
      "normalize": normalize
  }

  x_enc = Input(
      shape=(params["I"], params["d_enc"]),
      name="x_enc_input"
  )
  x_dec = Input(
      shape=(params["O"], params["d_dec"]),
      name="x_dec_input"
  )

  x_enc_marks, x_dec_marks = get_mark_inputs(params)

  y = GenericTransformer(
      recurrent.RecEncoder(params),
      recurrent.RecDecoder(params),
      params["output_attention"]
  )(
      [
          x_enc, x_enc_marks,
          x_dec, x_dec_marks
      ]
  )

  if params["embed_type"] is None:
    return keras.Model(
        inputs=[x_enc, x_dec],
        outputs=y
    )
  else:
    return keras.Model(
        inputs=[
            x_enc, x_enc_marks,
            x_dec, x_dec_marks
        ],
        outputs=y
    )


def restore_custom_objects():
  keras.saving.get_custom_objects().update({
      "GenericTransformer>GenericTransformer": GenericTransformer
  })

  embed.restore_custom_objects()
  recurrent.restore_custom_objects()
