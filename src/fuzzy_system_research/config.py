"""
Settings for the backend's models and optimization, read from a YAML file
(by default config/modeling_and_optimization.yml, relative to the working
directory, like input/ and output/).

The file is read once by the entry script, which passes the parsed settings
down explicitly; nothing reads it at import time. Validation is strict:
every key is required and unknown keys are rejected, so the file is the one
source of truth and a typo fails the run instead of falling back to a
default.

File layout:

    time_series_models:
      training_window_length: 252    # closes per ARIMA training window
      evaluation_window_length: 20   # backtest days scored
"""

import os
from dataclasses import dataclass
import yaml

DEFAULT_CONFIG_PATH = os.path.join("config", "modeling_and_optimization.yml")
# Smallest training window accepted, in closes. ARIMA(1, 0, 1) with a
# constant estimates four parameters, so far fewer returns than this would
# make the fits meaningless.
MIN_TRAINING_WINDOW_LENGTH = 30


class ConfigError(ValueError):
    """The configuration file is missing, unreadable, or invalid."""


def _require_int(name: str, value: object, minimum: int) -> None:
    # bool is a subclass of int, so `True` must be rejected explicitly.
    if type(value) is not int:
        raise ConfigError(
            f"time_series_models.{name} must be an integer, got {value!r}."
        )
    if value < minimum:
        raise ConfigError(
            f"time_series_models.{name} must be at least {minimum}, got {value}."
        )


@dataclass(frozen=True)
class TimeSeriesConfig:
    """
    Settings for the walk-forward time series models. Validated on
    construction.

    Attributes
    ----------
    training_window_length : int
        Closing prices (NYSE trading sessions) in each training window,
        ending on the session before the forecast date. At least
        MIN_TRAINING_WINDOW_LENGTH.

    evaluation_window_length : int
        Most recent trading days with an actual close that are backtested
        and scored: one forecast each, plus the live forecast. At least 1.
    """

    training_window_length: int
    evaluation_window_length: int

    def __post_init__(self) -> None:
        _require_int(
            "training_window_length",
            self.training_window_length,
            MIN_TRAINING_WINDOW_LENGTH,
        )
        _require_int("evaluation_window_length", self.evaluation_window_length, 1)

    @property
    def training_window_returns(self) -> int:
        """Daily log returns in each training window: one fewer than closes."""
        return self.training_window_length - 1

    @property
    def arima_window_sessions(self) -> int:
        """
        Most recent trading sessions the walk-forward ARIMA needs: one
        training window's closes plus one session per scored day, giving
        evaluation_window_length forecasts with actuals plus the live
        forecast (252 + 20 = 272 with the current settings).
        """
        return self.training_window_length + self.evaluation_window_length


@dataclass(frozen=True)
class ModelingConfig:
    """
    The whole configuration file. Has one section today; later sections
    (e.g. for the optimization) become further fields.
    """

    time_series_models: TimeSeriesConfig


def _check_keys(section: str, mapping: object, expected: set[str]) -> dict:
    if not isinstance(mapping, dict):
        raise ConfigError(f"{section} must be a mapping, got {mapping!r}.")
    # YAML keys need not be strings, so sort and join them as strings.
    missing = sorted(map(str, expected - set(mapping)))
    unknown = sorted(map(str, set(mapping) - expected))
    if missing:
        raise ConfigError(f"{section} is missing required key(s): {', '.join(missing)}.")
    if unknown:
        raise ConfigError(f"{section} has unknown key(s): {', '.join(unknown)}.")
    return mapping


def parse_config(raw: object) -> ModelingConfig:
    """
    Validate a parsed YAML document and build the configuration. Pure
    function.

    Parameters
    ----------
    raw : object
        The document as returned by yaml.safe_load().

    Returns
    -------
    ModelingConfig
        The validated settings.

    Raises
    ------
    ConfigError
        If a section or key is missing or unknown, or a value has the wrong
        type or is out of range.
    """
    top = _check_keys("The configuration file", raw, {"time_series_models"})
    ts = _check_keys(
        "time_series_models",
        top["time_series_models"],
        {"training_window_length", "evaluation_window_length"},
    )
    return ModelingConfig(
        time_series_models=TimeSeriesConfig(
            training_window_length=ts["training_window_length"],
            evaluation_window_length=ts["evaluation_window_length"],
        )
    )


def load_config(path: str = DEFAULT_CONFIG_PATH) -> ModelingConfig:
    """
    Read and validate the configuration file.

    Parameters
    ----------
    path : str, optional
        Path of the YAML file, relative to the working directory. Defaults
        to DEFAULT_CONFIG_PATH.

    Returns
    -------
    ModelingConfig
        The validated settings.

    Raises
    ------
    ConfigError
        If the file is missing, is not valid YAML, or fails parse_config().
        The message names the file.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(
            f"Configuration file not found: {path} (run from the repository root, "
            "or pass --config)."
        ) from None
    except yaml.YAMLError as error:
        raise ConfigError(f"Configuration file {path} is not valid YAML: {error}") from None
    try:
        return parse_config(raw)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from None
