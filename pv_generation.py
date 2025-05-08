"""
Script to test SMASPE on the PV generation dataset.
"""
import os

from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import keras

import loss_functions
import pv_utils

from keras_transformer.generic_transformer import create_recurrent_model

sns.set_theme(
    context="paper",
    style="whitegrid",
    font="Roboto",
    font_scale=2
)

keras.utils.set_random_seed(0)

ALPHA: float = 0.001
BATCH_SIZE: int = 64
EPOCHS = 2000
I: int = 72
LIN_THR: float = 1E-2
N_TRIALS: int = 5
O: int = 72
ROOT = "data/"
Y: str = "full_solar"

RELEVANT_VARIABLES = [
    "global_r",
    "air_temperature",
    Y
]

MODEL_PARAMS = {
    "I": I,
    "O": O,
    "d_enc": len(RELEVANT_VARIABLES),
    "d_dec": len(RELEVANT_VARIABLES) - 1,
    "d_out": 1,
    "d_model": 6,
    "d_ff": 10,
    "k": 2,
    "h": 4,
    "n_blocks": 1,
    "rnn_type": "gru",
    "manual_dec_input": True,
    "embed_type": "temporal",
    "freq": "h",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "validation_batch_size": BATCH_SIZE
}

DATASET = pv_utils.dataset_formatter(
    raw_data=pv_utils.get_curated_datasets(
        RELEVANT_VARIABLES
    ),
    model_params=MODEL_PARAMS,
    target=Y,
    stride=24,
    normalize=True,
    context_days=15,
    sigma=0.001,
    beta=np.log(10) / 71.
)
for partition, inputs in DATASET.items():
  DATASET[partition].x[2] = inputs.x[2][..., :-1]

model = create_recurrent_model(**MODEL_PARAMS)
model.summary()


def fit_helper(
    loss: keras.Loss,
    n: int
) -> dict[str, list[float]]:
  def clipped_loss(y_true, y_pred):
    return loss(
        y_true,
        keras.ops.clip(y_pred, 0, np.inf)
    )
  model_tmp = create_recurrent_model(**MODEL_PARAMS)
  model_tmp.compile(
      keras.optimizers.Adam(5e-4),
      loss=loss,
      metrics=[clipped_loss]
  )
  if os.path.exists(f"./initial_{n}.weights.h5"):
    model_tmp.load_weights(f"./initial_{n}.weights.h5")
  else:
    model_tmp.save_weights(f"./initial_{n}.weights.h5")

  ref = {}
  ref["loss"] = loss(
      DATASET["train"].y,
      model_tmp.predict(
          DATASET["train"],
          verbose=0
      )
  )
  ref["val_loss"] = clipped_loss(
      DATASET["validation"].y,
      model_tmp.predict(
          DATASET["validation"],
          verbose=0
      )
  )

  h = model_tmp.fit(
      x=DATASET["train"],
      validation_data=DATASET["validation"],
      epochs=EPOCHS,
      shuffle=True,
      callbacks=[
          keras.callbacks.EarlyStopping(
              monitor="val_clipped_loss",
              patience=EPOCHS // 4,
              min_delta=ALPHA * ref["val_loss"],
              mode="min"
          ),
          keras.callbacks.ModelCheckpoint(
              "./checkpoint.weights.h5",
              monitor="val_clipped_loss",
              save_best_only=True,
              save_weights_only=True,
              mode="min"
          )
      ],
      verbose=0
  )

  return {
      "loss": np.concatenate([
          [ref["loss"]], np.asarray(h.history["loss"])
      ], None),
      "val_loss": np.concatenate([
          [ref["val_loss"]], np.asarray(h.history["val_clipped_loss"])
      ], None),
  }


loss_names = [
    "MAE", "MSE", "MAAPE", "MASPE", "SMASPE (tight)", "SMASPE (loose)"
]
loss_functions = [
    keras.losses.MeanAbsoluteError(),
    keras.losses.MeanSquaredError(),
    loss_functions.MAAPE(),
    loss_functions.MASPE(),
    loss_functions.SMASPE(0., 1.),
    loss_functions.SMASPE(-0.05, 1.05)
]


def eval_helper() -> tuple[dict[str, float], np.ndarray]:
  model.load_weights("./best.weights.h5")

  y_pred = np.clip(
      model.predict(
          DATASET["test"],
          verbose=0
      ),
      0.,
      np.inf
  )

  return {
      key: float(metric(DATASET["test"].y, y_pred))
      for key, metric in zip(loss_names, loss_functions)
  }, DATASET["test"].y - y_pred


loss_fig, val_fig = [plt.figure(figsize=(12, 6)) for _ in range(2)]
err_vectors = []
results = []


def results_sorter(i: int, hs: list[np.ndarray]) -> float:
  return float(np.min(hs[i]["val_loss"][1:]))


for loss_name, loss_fn in zip(  # Skip MAE
    loss_names[1:],
    loss_functions[1:]
):
  print(f"Training with {loss_name}...", flush=True)
  history = {
      "loss": np.inf * np.ones(2),
      "val_loss": np.inf * np.ones(2)
  }
  histories = []
  val_losses = []
  for n_trial in range(N_TRIALS):
    print(f"Trial no. {n_trial + 1}/{N_TRIALS}...", flush=True)
    aux = fit_helper(loss_fn, n_trial)
    histories.append(aux)
    val_losses.append(np.min(history["val_loss"][1:]))
    os.rename("./checkpoint.weights.h5", f"./final_{n_trial}.weights.h5")

  median_idx = sorted(
      range(N_TRIALS),
      key=partial(results_sorter, hs=histories)
  )[N_TRIALS // 2]
  os.symlink(f"./final_{median_idx}.weights.h5", "./best.weights.h5")
  history = histories[median_idx]

  metrics, err = eval_helper()
  metrics["median_epochs"] = float(
      np.median([len(aux["loss"]) for aux in histories]) - 1
  )

  os.remove("./best.weights.h5")
  for n_trial in range(N_TRIALS):
    os.remove(f"./final_{n_trial}.weights.h5")

  err_vectors.append(err.flatten())
  results.append(metrics)

  plt.figure(loss_fig)
  plt.plot(
      history["loss"] / history["loss"][0],
      label=loss_name
  )

  plt.figure(val_fig)
  plt.plot(
      history["val_loss"] / history["val_loss"][0],
      label=loss_name
  )

for n_trial in range(N_TRIALS):
  os.remove(f"./initial_{n_trial}.weights.h5")

for f, s in [(loss_fig, "loss"), (val_fig, "val")]:
  plt.figure(f)
  plt.xlabel("Epoch")
  plt.ylabel("Normalized loss")
  f.legend(loc="upper right")
  plt.autoscale(True, "x", tight=True)
  plt.ylim(top=1.)
  f.tight_layout(pad=2)
  f.savefig(
      f"./plots/pv_{s}_history.svg",
      transparent=True
  )

for loss_name, metrics in zip(
    loss_names[1:],
    results
):
  print(f"Trained with {loss_name}: ")
  print(", ".join([f"{k}: {v:.3}" for k, v in metrics.items()]))

error_fig = plt.figure(figsize=(12, 6))
plt.boxplot(err_vectors, tick_labels=loss_names[1:])
plt.xlabel("Training Loss Function")
plt.ylabel(r"Error Distribution ($y - \hat{y}$)")
plt.yscale("symlog", linthresh=LIN_THR)
error_fig.tight_layout(pad=2)
error_fig.savefig(
    "./plots/pv_err_boxplot.svg",
    transparent=True
)

err_bins = np.concatenate(
    [
        -np.logspace(np.log10(LIN_THR), 1, 6)[::-1],
        np.logspace(np.log10(LIN_THR), 1, 6)
    ]
)

hist_fig, hist_axs = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    sharey=True,
    figsize=(12, 12)
)
reduced_idxs = [
    loss_names.index("MAAPE"),
    loss_names.index("SMASPE (loose)")
]
for fig_idx, err_idx in enumerate(reduced_idxs):
  _, _, _, mpbl = hist_axs[fig_idx].hist2d(
      np.tile(
          np.arange(1, O + 1),
          len(err_vectors[err_idx - 1]) // O
      ),
      err_vectors[err_idx - 1],
      [O, err_bins],
      cmap="viridis",
      norm="log"
  )

  hist_axs[fig_idx].set_ylabel(
      r"$y - \hat{y}_\text{" +
      loss_names[err_idx] +
      r"}$"
  )
  hist_axs[fig_idx].set_yscale("symlog", linthresh=LIN_THR, linscale=0.5)
  cbar = plt.colorbar(mpbl, ax=hist_axs[fig_idx])
  cbar.ax.set_ylabel("Frequency")

hist_axs[-1].set_xlabel(r"Prediction Horizon $[h]$")
hist_fig.tight_layout(pad=2)
hist_fig.savefig(
    "./plots/pv_err_histogram.svg",
    transparent=True
)

N = len(err_vectors[0])
abs_daily_err = np.sum(
    np.abs(np.stack(err_vectors[-2:])[  # Check SMASPE only
        :,
        np.arange(0, N, O)[:, None] + np.arange(O)[None, :]
    ]),
    (0, -1)
)
best_idx, worst_idx = [
    np.argmin(abs_daily_err),
    np.argmax(abs_daily_err)
]
best_ref, worst_ref = [
    DATASET["test"].y[best_idx, :, 0],
    DATASET["test"].y[worst_idx, :, 0]
]
ex_fig, ex_axs = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(12, 12)
)
ex_axs[0].plot(
    np.arange(1, O + 1),
    best_ref,
    ":",
    label="Reference (Best Case)"
)
ex_axs[1].plot(
    np.arange(1, O + 1),
    worst_ref,
    ":",
    label="Reference (Worst Case)"
)
for err, loss_name in zip(err_vectors, loss_names[1:]):
  ex_axs[0].plot(
      np.arange(1, O + 1),
      best_ref - err[O * best_idx:O * (best_idx + 1)],
      label=loss_name
  )
  ex_axs[1].plot(
      np.arange(1, O + 1),
      worst_ref - err[O * worst_idx:O * (worst_idx + 1)],
      label=loss_name
  )
ex_axs[0].legend(loc="upper right")
ex_axs[1].legend(loc="upper right")
ex_axs[1].set_xlabel(r"Prediction Horizon $[h]$")
plt.autoscale(True, "x", tight=True)
ex_fig.tight_layout(pad=2)
ex_fig.savefig(
    "./plots/pv_forecasts.svg",
    transparent=True
)
print("\a")
