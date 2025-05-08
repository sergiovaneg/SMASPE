"""
Module containing the auxiliary functions to run 1D models given a set of
parameters used for the SKU experiments.

When passing the entire sequence to the estimation functions, they
"""

from jax import lax, random
from jax import numpy as jnp

KEY = random.key(0)


def additive_hw_init(
        x0: jnp.ndarray,
        m: int) -> tuple[jnp.ndarray, float, float]:
  l = x0.size
  x0 = x0[::-1]

  # x0t = jnp.convolve(
  #     x0,
  #     jnp.ones(2 * m) / (2 * m),
  #     mode="same"
  # )
  # x0s = x0 - x0t

  x0_fft = jnp.fft.rfft(x0)
  x0_fft_freqs = jnp.fft.rfftfreq(l)
  c0 = x0_fft[0]
  c1_abs = jnp.interp(1 / m, x0_fft_freqs, jnp.abs(x0_fft))
  c1_ang = jnp.interp(1 / m, x0_fft_freqs, jnp.angle(x0_fft))

  x0s = jnp.real(c0) / l + 2 * c1_abs / l * jnp.cos(
      2 * jnp.pi * jnp.arange(l) / m + c1_ang
  )
  s0 = jnp.asarray([jnp.mean(x0s[i::m]) for i in range(m)]).flatten()

  x0t = x0 - x0s
  lfit = jnp.polyfit(
      jnp.arange(l, dtype=float),
      x0t,
      1
  )

  return s0, lfit[1], -lfit[0]


def additive_hw_run(
        w: jnp.ndarray,
        x0: jnp.ndarray,
        m: int,
        h: int) -> jnp.ndarray:
  alpha, beta, gamma = jnp.clip(w, 0, 1)

  carry, y_obs = (*additive_hw_init(x0[:3 * m], m), x0[3 * m]), x0[3 * m + 1:]
  if y_obs.shape[-1] == h:
    def iterate(carry, y):
      s0, l0, b0, yt = carry

      lt = alpha * (yt - s0[-1]) + (1 - alpha) * (l0 + b0)
      bt = beta * (lt - l0) + (1 - beta) * b0
      st = gamma * (yt - l0 - b0) + (1 - gamma) * s0[-1]

      yhat = lt + bt + s0[-1]

      return (jnp.concatenate([st[None], s0[:-1]]), lt, bt, y), yhat
  else:
    y_obs = jnp.empty(h)

    def iterate(carry, _):
      s0, l0, b0, yt = carry

      lt = alpha * (yt - s0[-1]) + (1 - alpha) * (l0 + b0)
      bt = beta * (lt - l0) + (1 - beta) * b0
      st = gamma * (yt - l0 - b0) + (1 - gamma) * s0[-1]

      yhat = lt + bt + s0[-1]

      return (jnp.concatenate([st[None], s0[:-1]]), lt, bt, yhat), yhat

  y = lax.scan(
      f=iterate,
      init=carry,
      xs=y_obs
  )

  return y[1]


def arima_run(
        w: jnp.ndarray,
        x0: jnp.ndarray,
        order: tuple[int, int, int],
        h: int) -> jnp.ndarray:
  p, d, q = order

  c = w[0]
  a = w[1:1 + p]
  b = w[1 + p:1 + p + q]
  sigma = w[1 + p + q]

  carry = jnp.diff(x0, d, -1)
  carry, y_obs = carry[p - 1::-1], carry[p:]

  idxs = jnp.arange(q, q + h)[:, None] - jnp.arange(q + 1)[None, :]
  eps = jnp.einsum(  # Pre-apply the MA coefficients
      "tb,b->t",
      sigma * random.normal(KEY, h + q)[idxs],
      jnp.pad(b, [1, 0], "constant", constant_values=1.)
  )

  if y_obs.shape[-1] == h:
    def iterate(y0, x):
      yt = c + jnp.dot(a, y0) + x[0]
      return jnp.concatenate([x[1][None], y0[:-1]]), yt
  else:
    y_obs = jnp.empty_like(eps)

    def iterate(y0, x):
      yt = c + jnp.dot(a, y0) + x[0]
      return jnp.concatenate([yt[None], y0[:-1]]), yt

  y = lax.scan(f=iterate, init=carry, xs=(eps, y_obs))[1]

  for n in range(d):
    y = jnp.cumsum(y) + jnp.diff(x0[:d + p], d - n - 1, -1)[-1]

  return y
