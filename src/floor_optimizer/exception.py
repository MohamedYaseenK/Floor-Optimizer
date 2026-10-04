"""Project-specific exceptions so callers can catch our failures precisely."""


class FloorOptimizerError(Exception):
    """Base class for all errors raised by this project."""


class ConfigError(FloorOptimizerError):
    """Invalid or missing configuration."""


class DataValidationError(FloorOptimizerError):
    """Input data does not match the expected schema or contract."""
