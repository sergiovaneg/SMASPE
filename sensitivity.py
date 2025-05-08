"""
Script to perform the sensitivity analysis over the relaxation coefficient
gamma for the SMASPE.
"""

from collections.abc import Callable

import numpy as np

import jax
from jax import random
from jax.scipy import optimize
from jax import numpy as jnp

import polars as pl

import keras

import matplotlib.pyplot as plt
import seaborn as sns

from loss_functions import SMASPE
from sku_models import arima_run, additive_hw_run

sns.set_theme(
    context="paper",
    style="whitegrid",
    font="Roboto",
    font_scale=2
)

KEY = random.key(0)
EPS = keras.backend.epsilon()

# Keeping only time-series with at least some intermittence
DATASET = pl.read_csv("./sku.csv").select(
    ["Scode", "Pcode", pl.selectors.matches(r"Wk\d+")]
).with_columns(
    pl.concat_arr(pl.col(r"^Wk\d+$")).alias("ts")
).drop(r"^Wk\d+$").sort(["Scode", "Pcode"]).to_dicts()

X = jnp.stack([jnp.asarray(row["ts"]) for row in DATASET], dtype=float)

SPLIT_IDX = 95
X_TRAIN, X_TEST = X[:, :SPLIT_IDX], X[:, SPLIT_IDX:]
BOUNDS = [
    jnp.min(X_TRAIN, axis=-1, keepdims=True),
    jnp.max(X_TRAIN, axis=-1, keepdims=True)
]
DELTA = BOUNDS[1] - BOUNDS[0]

GAMMA = np.linspace(-0.2, 0.2, 81)

P, D, Q = 4, 1, 4
M = 4


def arima_helper(
        z_train: jnp.ndarray,
        z_test: jnp.ndarray,
        loss: keras.Loss) -> tuple[jnp.ndarray, int]:
    opt_result = optimize.minimize(
        lambda w: loss(
            z_train[P + D:],
            arima_run(w, z_train, [P, D, Q], len(z_train) - P - D)
        ),
        jnp.zeros(1 + P + Q + 1),
        method="BFGS",
        tol=EPS
    )

    pred = arima_run(
        opt_result.x,
        jnp.concatenate(
            [
                z_train,
                z_test
            ]
        ),
        [P, D, Q],
        len(z_train) + len(z_test) - (P + D)
    )
    return pred, opt_result.nit


def hw_helper(
        z_train: jnp.ndarray,
        z_test: jnp.ndarray,
        loss: keras.Loss) -> tuple[jnp.ndarray, int]:
    opt_result = optimize.minimize(
        lambda w: loss(
            z_train[3 * M + 1:],
            additive_hw_run(w, z_train, M, len(z_train) - (3 * M + 1))
        ),
        0.5 * jnp.ones(3),
        method="BFGS",
        tol=EPS
    )

    pred = additive_hw_run(
        opt_result.x,
        jnp.concatenate(
            [
                z_train,
                z_test
            ]
        ),
        M,
        len(z_train) + len(z_test) - (3 * M + 1)
    )

    return pred, opt_result.nit


def sensitivity_lambda(
        gamma: float, helper: Callable[
            [jnp.ndarray, jnp.ndarray, keras.Loss],
            tuple[jnp.ndarray, int]
        ]) -> tuple[float, float]:
    ym, yp = BOUNDS[0] - gamma * DELTA, BOUNDS[1] + gamma * DELTA
    predictions, iterations = jax.vmap(
        lambda z_train, z_test, ym, yp: helper(
            z_train,
            z_test,
            SMASPE(ym, yp)
        )
    )(
        X_TRAIN,
        X_TEST,
        ym[:, 0],
        yp[:, 0]
    )

    return keras.losses.MeanAbsoluteError()(
        X_TEST,
        jnp.clip(predictions[:, -X_TEST.shape[-1]:], 0, jnp.inf)
    ), jnp.median(iterations)


sensitivity_vmap = jax.vmap(
    sensitivity_lambda,
    (0, None),
    (0, 0)
)
arima_results = sensitivity_vmap(GAMMA, arima_helper)
hw_results = sensitivity_vmap(GAMMA, hw_helper)

for label, results in zip(["arima", "hw"], [arima_results, hw_results]):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(GAMMA, results[0])
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("MAE")
    ax.set_ylim(2e2, 4e2)
    ax.autoscale(True, "x", True)
    fig.tight_layout(pad=2)
    fig.savefig(f"./plots/sensitivity_mae_{label}.svg")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(GAMMA, results[1])
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("No. Iterations")
    ax.autoscale(True, "x", True)
    fig.tight_layout(pad=2)
    fig.savefig(f"./plots/sensitivity_iters_{label}.svg")

    plt.close("all")
