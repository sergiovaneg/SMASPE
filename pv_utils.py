"""
Module containing helper functions specific for the PV dataset
"""

import math
import numpy as np
import polars as pl
import keras

type NormParams = dict[str, tuple[float, float]]

STARTING_NORM_PARAMS: NormParams = {
    "cloud_amount": (0., 9.),
    "horizontal_visibility": (0., 5.e4),
    "relative_humidity": (0., 100.),
}

SOLAR_FEATURES = [
    "diffuse_r",
    "full_solar",
    "global_r",
    "sunshine"
]

UNNORMALIZED_FEATURES = ["elspot", "energy"]

ROOT: str = "data/"
RNG: np.random.Generator = np.random.default_rng()
REPS: int = 10


class PVDataset(keras.utils.PyDataset):
    """
    Auxiliary class to enable distributed training and (optional) sample weighting
    """

    def __init__(self, data_dict: dict, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = data_dict["x"], data_dict["y"]
        self.batch_size = batch_size
        self.w = data_dict.get("w", None)

    def __len__(self):
        return math.ceil(self.y.shape[0] / self.batch_size)

    def __getitem__(self, index):
        if index == len(self):
            raise IndexError

        low = index * self.batch_size
        high = min(self.y.shape[0], low + self.batch_size)

        if self.w is not None:
            return (
                [x[low:high, ...] for x in self.x],
                self.y[low:high, ...],
                self.w[low:high, ...]
            )
        else:
            return (
                [x[low:high, ...] for x in self.x],
                self.y[low:high, ...]
            )


def transform_wind_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a Dataframe with cartesian wind columns instead of the original polar
    representation.
    """
    return df.with_columns(
        (
            pl.col("wind_speed") * (2 * np.pi *
                                    pl.col("wind_direction") / 360).sin()
        ).alias("wind_sin"),
        (
            pl.col("wind_speed") * (2 * np.pi *
                                    pl.col("wind_direction") / 360).cos()
        ).alias("wind_cos")
    ).select(
        pl.all().exclude([
            "wind_speed", "wind_direction"
        ])
    )


def add_datetime_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a Dataframe with additional datetime cyclical features.
    """
    return df.with_columns(
        (
            pl.col("datetime").dt.hour() * 2 * np.pi / 24
        ).sin().alias("hour_sin"),
        (
            pl.col("datetime").dt.hour() * 2 * np.pi / 24
        ).cos().alias("hour_cos"),
        (
            pl.col("datetime").dt.weekday() * 2 * np.pi / 7
        ).sin().alias("dow_sin"),
        (
            pl.col("datetime").dt.weekday() * 2 * np.pi / 7
        ).cos().alias("dow_cos"),
        (
            pl.col("datetime").dt.ordinal_day() * 2 * np.pi /
            pl.when(pl.col("datetime").dt.is_leap_year()).then(
                366
            ).otherwise(
                365
            )
        ).sin().alias("doy_sin"),
        (
            pl.col("datetime").dt.ordinal_day() * 2 * np.pi /
            pl.when(pl.col("datetime").dt.is_leap_year()).then(
                366
            ).otherwise(
                365
            )
        ).cos().alias("doy_cos")
    )


def get_dataset_n(
    n: int
) -> pl.DataFrame:
    """
    Returns dataset 'n' according to the filename as a Polars Dataframe.
    """
    return pl.read_parquet(
        source=ROOT + f"dataset_{n}.parquet"
    ).with_columns(  # Remove wrong readings
        pl.all().exclude("^.*_temperature$").clip(lower_bound=0)
    )


def get_curated_datasets(
    relevant_variables: list[str] = None,
    log_solar: bool = False
) -> tuple[pl.DataFrame, ...]:
    """
    Returns dataframes for train/test/validation (2, 1, 0) containing only the
    relevant columns or all columns by default.
    """
    if relevant_variables is not None and "datetime" not in relevant_variables:
        relevant_variables.append("datetime")
    return tuple(
        [
            add_datetime_features(
                transform_wind_features(
                    get_dataset_n(n).with_columns(
                        full_solar=pl.col(
                            "full_solar"
                        ).log1p() if log_solar else pl.col("full_solar")
                    )
                )
            ).select(
                relevant_variables or pl.all()
            )
            for n in [2, 1, 0]
        ]
    )


def set_partial_placeholder(
    x_enc: np.ndarray,
    x_dec: np.ndarray,
    shift: int
) -> np.ndarray:
    x_mod = x_dec.copy()
    x_mod[..., shift:] = np.mean(
        np.concatenate(
            [
                x_enc,
                x_dec[..., :shift]
            ],
            -1
        ),
        -1, keepdims=True
    )

    return x_mod


def dataframe_to_datadict(
    df: pl.DataFrame,
    model_params: dict,
    target: str,
    stride: int,
    updatable: bool = False,
    iterative: bool = False,
    update_stride: int = 1,
    reg: float = 1e-3,
    sigma: float = 0.,
    beta: float = 0.
) -> dict:
    data_dict = {}

    # Ensure that the target variable is the last column
    columns = [col for col in df.columns if col not in [target, "datetime"]]
    columns.append(target)
    x = df.select(columns).to_numpy()
    y = df.select(target).to_numpy()

    assert np.count_nonzero(np.isnan(x)) == 0, "NaN in Input"
    assert np.count_nonzero(np.isnan(y)) == 0, "NaN in Output"

    l_input, l_output = model_params["I"], model_params["O"]

    enc_idx = np.arange(
        x.shape[0] - l_input - l_output,
        step=stride,
        dtype="int"
    )[:, None] + np.arange(l_input)[None, :]
    dec_idx = np.arange(
        l_input,
        x.shape[0] - l_output,
        step=stride,
        dtype="int"
    )[:, None] + np.arange(l_output)[None, :]

    x_enc = x[enc_idx, :]
    x_dec = x[dec_idx, :]
    y = y[dec_idx, :]

    # Replace future output information with context mean and noisy forecasts
    std = x.std(axis=0)[None, None, :]

    if "elspot" in columns:  # No noise for Elspot
        std[..., columns.index("elspot")] = 0

    noise_limits = np.median(
        np.abs(np.diff(x, axis=0)), axis=0
    )[None, None, ...]

    if updatable:
        x_dec_segments = []
        for shift in range(0, l_output, update_stride):
            aux = x_dec.copy()

            aux[:, shift:, :] = aux[:, shift:, :] + np.clip(
                RNG.normal(
                    scale=sigma,
                    size=[aux.shape[0], l_output - shift, aux.shape[-1]]
                ) * std * np.exp(
                    beta * np.arange(l_output - shift)
                )[None, :, None],
                a_min=-noise_limits,
                a_max=noise_limits
            )

            # Placeholder without prediction noise
            aux[..., -1] = set_partial_placeholder(
                x_enc[..., -1],
                x_dec[..., -1],
                shift
            )

            # Day-ahead prices only available for the next day
            if "elspot" in columns:
                elspot_idx = columns.index("elspot")
                aux[..., elspot_idx] = set_partial_placeholder(
                    x_enc[..., elspot_idx],
                    x_dec[..., elspot_idx],
                    shift + 24
                )

            x_dec_segments.append(aux)

        x_dec = np.stack(
            x_dec_segments,
            1
        ).reshape([-1, l_output, x_dec.shape[-1]])

        data_dict["w"] = np.stack(
            [
                np.concatenate(
                    [
                        reg * np.ones([y.shape[0], s], dtype=np.float32),
                        np.ones([y.shape[0], l_output - s], dtype=np.float32)
                    ],
                    axis=-1
                ) for s in range(0, l_output, update_stride)
            ],
            axis=1
        ).reshape([-1, l_output])

        x_enc = x_enc.repeat(math.ceil(l_output / update_stride), 0)
        y = y.repeat(math.ceil(l_output / update_stride), 0)
    elif iterative:
        x_dec_segments = []
        for shift in [0, update_stride]:
            aux = x_dec.copy()

            aux += np.clip(
                RNG.normal(
                    scale=sigma,
                    size=[aux.shape[0], l_output, aux.shape[-1]]
                ) * std * np.exp(
                    beta * np.arange(l_output)
                )[None, :, None],
                a_min=-noise_limits,
                a_max=noise_limits
            )

            # Placeholder with prediction noise
            aux[..., -1] = set_partial_placeholder(
                x_enc[..., -1],
                aux[..., -1],
                shift
            )

            # Day-ahead prices only available for the next day
            if "elspot" in columns:
                elspot_idx = columns.index("elspot")
                aux[..., elspot_idx] = set_partial_placeholder(
                    x_enc[..., elspot_idx],
                    x_dec[..., elspot_idx],
                    24
                )

            x_dec_segments.append(aux)

        x_dec = np.stack(
            x_dec_segments,
            1
        ).reshape([-1, l_output, x_dec.shape[-1]])

        data_dict["w"] = np.tile(
            np.stack(
                [
                    reg * np.ones(l_output),
                    np.ones(l_output)
                ],
                axis=0
            ),
            reps=[y.shape[0], 1]
        )

        x_enc = x_enc.repeat(2, 0)
        y = y.repeat(2, 0)
    else:
        # Copy of x_dec to be modified
        aux = x_dec.copy()

        aux += np.clip(
            RNG.normal(
                scale=sigma,
                size=x_dec.shape
            ) * std * np.exp(
                beta * np.arange(l_output)
            )[None, :, None],
            a_min=-noise_limits,
            a_max=noise_limits
        )

        # Placeholder without prediction noise
        aux[..., -1] = set_partial_placeholder(
            x_enc[..., -1],
            x_dec[..., -1],
            0
        )

        # Day-ahead prices only available for the next day
        if "elspot" in columns:
            elspot_idx = columns.index("elspot")
            aux[..., elspot_idx] = set_partial_placeholder(
                x_enc[..., elspot_idx],
                x_dec[..., elspot_idx],
                24
            )

        x_dec = aux

    data_dict["y"] = y

    match model_params["embed_type"]:
        case "temporal":
            df_time = df.with_columns(
                month=pl.col("datetime").dt.month() - 1,
                day=pl.col("datetime").dt.day() - 1,
                weekday=pl.col("datetime").dt.weekday() - 1,
                hour=pl.col("datetime").dt.hour(),
                minute=pl.col("datetime").dt.minute()
            )
            if model_params["freq"] == "h":
                marks = df_time.select([
                    "month", "day", "weekday", "hour"
                ]).to_numpy()
            else:
                marks = df_time.select([
                    "month", "day", "weekday", "hour", "minute"
                ]).to_numpy()

            assert np.count_nonzero(np.isnan(marks)) == 0, "NaN in marks"

            xm_enc = marks[enc_idx, :]
            xm_dec = marks[dec_idx, :]
        case "positional" | "fixed":
            xm_enc = enc_idx.astype(np.int32)
            xm_dec = dec_idx.astype(np.int32)

    if (updatable or iterative) and model_params["embed_type"] is not None:
        xm_enc = xm_enc.repeat(x_enc.shape[0] // xm_enc.shape[0], 0)
        xm_dec = xm_dec.repeat(x_dec.shape[0] // xm_dec.shape[0], 0)

    if model_params.get("manual_dec_input", True):
        if model_params["embed_type"] is None:
            data_dict["x"] = [x_enc, x_dec]
        else:
            data_dict["x"] = [x_enc, xm_enc, x_dec, xm_dec]
    else:
        if model_params["embed_type"] is None:
            data_dict["x"] = [x_enc]
        else:
            data_dict["x"] = [x_enc, xm_enc, xm_dec]

    return data_dict


def context_normalizer(
    x: np.ndarray,
    context_days: int
):
    x_norm = np.zeros_like(x)
    for day_idx in range(0, len(x), 24):
        context_lb = max(0, day_idx - 24 * context_days)
        context = x[context_lb: context_lb + 24 * context_days]
        median_peak = np.median(context.reshape([-1, 24]).max(axis=1))

        x_norm[day_idx:day_idx + 24] = x[day_idx:day_idx + 24] / median_peak

    return x_norm


def dataset_normalizer(
        df: pl.DataFrame,
        norm_params: NormParams,
        context_days: int = 30
) -> tuple[pl.DataFrame, NormParams]:
    input_features = [col for col in df.columns
                      if col != "datetime"]

    for feature in input_features:
        if feature in UNNORMALIZED_FEATURES:
            continue

        if feature in SOLAR_FEATURES:
            df = df.with_columns(
                pl.Series(
                    feature,
                    context_normalizer(
                        df.get_column(feature).to_numpy(),
                        context_days
                    )
                )
            )
        else:
            if feature not in norm_params:
                x = df.get_column(feature).to_numpy()
                bias = float(x.mean())
                scale = float(x.std())
                norm_params[feature] = (bias, scale)
            df = df.with_columns(
                ((pl.col(feature) -
                  norm_params[feature][0]) /
                 norm_params[feature][1]).alias(feature)
            )

    return df, norm_params


def dataset_formatter(
    raw_data: tuple[pl.DataFrame, ...],
    model_params: dict,
    target: str,
    stride: int,
    updatable: bool = False,
    update_stride: int = 1,
    iterative: bool = False,
    normalize: bool = False,
    context_days=30,
    sigma: float = 0.005,  # std(1h) = 0.005
    beta: float = 0.03  # std(72h) = 0.03
) -> dict[str, PVDataset]:
    result = {}
    norm_params = STARTING_NORM_PARAMS.copy()

    for idx, partition in zip(
        range(3),
        ["train", "validation", "test"]
    ):
        df = raw_data[idx]

        if partition == "train":
            batch_size = model_params["batch_size"]
            if normalize:
                df, norm_params = dataset_normalizer(
                    df,
                    norm_params=norm_params,
                    context_days=context_days
                )
        else:
            batch_size = model_params["validation_batch_size"]
            if normalize:
                df, _ = dataset_normalizer(
                    df,
                    norm_params=norm_params,
                    context_days=context_days
                )

        result[partition] = PVDataset(
            dataframe_to_datadict(
                df=df,
                model_params=model_params,
                target=target,
                stride=stride,
                updatable=updatable,
                update_stride=update_stride,
                iterative=iterative,
                reg=1e-3 if partition == "train" else 0.,
                sigma=sigma,
                beta=beta
            ),
            batch_size
        )
    return result
