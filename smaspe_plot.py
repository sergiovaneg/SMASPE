"""
Script to plot analysis plot for the Symmetric MASPE
"""

import math
import numpy as np
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error
)

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

sns.set_theme(context="paper",
              style="whitegrid",
              font="Roboto",
              font_scale=2)

RNG = np.random.default_rng(0)

THR: float = 1e-7
N: int = 1000
W: int = 100
STEP: int = 10


def aspe(y: np.ndarray, yhat: np.ndarray) -> jnp.ndarray:
    return jnp.arctan(
        jnp.square(
            (y - yhat) / jnp.where(
                jnp.abs(y) < THR,
                THR,
                jnp.abs(y)
            )
        )
    )


def sym_aspe(y: np.ndarray, yhat: np.ndarray) -> jnp.ndarray:
    return aspe(y, yhat) + aspe(1. - y, 1. - yhat)


grad_sym_aspe = jax.vmap(jax.vmap(jax.value_and_grad(sym_aspe, 1)))


def arctan_square(a: np.ndarray) -> jnp.ndarray:
    return jnp.arctan(jnp.square(a))


dx_arctan_square = jax.vmap(
    jax.grad(arctan_square)
)
dx2_arctan_square = jax.vmap(
    jax.grad(jax.grad(arctan_square))
)


def arctan_abs(a: np.ndarray) -> jnp.ndarray:
    return jnp.arctan(jnp.abs(a))


dx_arctan_abs = jax.vmap(
    jax.grad(arctan_abs)
)
dx2_arctan_abs = jax.vmap(
    jax.grad(jax.grad(arctan_abs))
)

b = RNG.binomial(1, 0.15, N)
z = 1 + np.sin(2 * np.pi * np.arange(N) / (N // 50))**2 * \
    (1 + 9 * b)
z_noisy = np.clip(
    z * RNG.normal(loc=1., scale=0.15, size=z.shape),
    1, np.inf
)
z_ma = np.where(b, z, 1.5)

print(f"Noisy MSE: {mean_squared_error(z, z_noisy)}")
print(f"Noisy MAPE: {mean_absolute_percentage_error(z, z_noisy)}")

print(f"MA MSE: {mean_squared_error(z, z_ma)}")
print(f"MA MAPE: {mean_absolute_percentage_error(z, z_ma)}")

w_idx = np.argmax(
    np.sum(
        b[np.arange(N - W)[:, None] + np.arange(W)[None, :]],
        -1
    )
)

fig = plt.figure(figsize=(12, 6))
plt.plot(
    np.arange(w_idx, w_idx + W),
    z_noisy[w_idx:w_idx + W],
    label=r"$\hat{y}^{(1)}$",
    linewidth=2
)
plt.plot(
    np.arange(w_idx, w_idx + W),
    z_ma[w_idx:w_idx + W],
    label=r"$\hat{y}^{(2)}$",
    linewidth=2
)
plt.plot(
    np.arange(w_idx, w_idx + W),
    z[w_idx:w_idx + W],
    label=r"$y$",
    linewidth=3,
    linestyle="--"
)
plt.legend(loc="upper right")
plt.autoscale(axis="x", tight=True)
plt.gca().set_xticklabels([])
fig.tight_layout(pad=2)
fig.savefig(
    "./plots/synthetic_signals.svg",
    transparent=True
)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
axes[0].plot(
    np.arange(w_idx, w_idx + W),
    ((z - z_noisy)**2)[w_idx:w_idx + W],
    label=r"$(y-\hat{y}^{(1)})^2$",
    linewidth=2
)
axes[0].plot(
    np.arange(w_idx, w_idx + W),
    ((z - z_ma)**2)[w_idx:w_idx + W],
    label=r"$(y-\hat{y}^{(2)})^2$",
    linewidth=2
)
axes[1].plot(
    np.arange(w_idx, w_idx + W),
    (np.abs(z - z_noisy) / np.abs(z))[w_idx:w_idx + W],
    label=r"$\left| \bar{\delta}^{(1)} \right|$",
    linewidth=2
)
axes[1].plot(
    np.arange(w_idx, w_idx + W),
    (np.abs(z - z_ma) / np.abs(z))[w_idx:w_idx + W],
    label=r"$\left| \bar{\delta}^{(2)} \right|$",
    linewidth=2
)
axes[0].legend(loc="upper right")
axes[1].legend(loc="upper right")
axes[0].autoscale(axis="both", tight=True)
axes[1].autoscale(axis="both", tight=True)
plt.gca().set_xticklabels([])
fig.tight_layout(pad=2)
fig.savefig(
    "./plots/synthetic_errors.svg",
    transparent=True
)

x = np.logspace(
    math.log10(THR), math.log10(2.), N // 2,
    endpoint=True
)
x = np.concatenate([
    -np.flip(x), x
])


fig = plt.figure(figsize=(12, 6))
plt.plot(
    x,
    arctan_abs(x),
    label=r"$\arctan{\left| \bar{\delta} \right|}$",
    linewidth=2
)
plt.plot(
    x,
    -dx_arctan_abs(x),
    label=r"$-\frac{d}{d \bar{\delta}}" +
    r"\left( \arctan{\left| \bar{\delta} \right|} \right)$",
    linewidth=2
)
plt.plot(
    x,
    dx2_arctan_abs(x),
    label=r"$\frac{d^2}{d \bar{\delta}^2}" +
    r"\left( \arctan{\left| \bar{\delta} \right|} \right)$",
    linewidth=2
)
plt.legend(loc="upper right")
plt.xlabel(r"$\bar{\delta}$")
fig.tight_layout(pad=2)
fig.savefig(
    "./plots/arctangent_abs.svg",
    transparent=True
)


fig = plt.figure(figsize=(12, 6))
plt.plot(
    x,
    arctan_square(x),
    label=r"$\arctan{\left( \bar{\delta}^2 \right)}$",
    linewidth=2
)
plt.plot(
    x,
    -dx_arctan_square(x),
    label=r"$-\frac{d}{d \bar{\delta}}" +
    r"\left( \arctan{\left( \bar{\delta}^2 \right)} \right)$",
    linewidth=2
)
plt.plot(
    x,
    dx2_arctan_square(x),
    label=r"$\frac{d^2}{d \bar{\delta}^2}" +
    r"\left( \arctan{\left( \bar{\delta}^2 \right)} \right)$",
    linewidth=2
)
plt.legend(loc="upper right")
plt.xlabel(r"$\bar{\delta}$")
fig.tight_layout(pad=2)
fig.savefig(
    "./plots/arctangent_square.svg",
    transparent=True
)

X, Y = np.meshgrid(
    np.linspace(-1, 2, N, endpoint=True),  # y_true
    np.linspace(-1, 2, N, endpoint=True)  # y_pred
)
Z, DZ = grad_sym_aspe(X, Y)

fig = plt.figure(figsize=(12, 6))
cf = plt.contour(
    X, Y,
    Z,
    norm="log",
    cmap=sns.color_palette(
        "viridis",
        as_cmap=True
    ).reversed(),
    alpha=0.5
)
q = plt.quiver(
    X[::STEP, ::STEP], Y[::STEP, ::STEP],
    np.zeros_like(DZ)[::STEP, ::STEP], -DZ[::STEP, ::STEP],
    abs(DZ[::STEP, ::STEP]),
    units="width", pivot="tip",
    norm=colors.LogNorm(),
    cmap=sns.color_palette("viridis", as_cmap=True)
)
plt.xlabel(r"$y$")
plt.ylabel(r"$\hat{y}$")
# plt.colorbar(q)
fig.tight_layout(pad=2)
fig.savefig(
    "./plots/symmetric_loss.svg",
    transparent=True
)
