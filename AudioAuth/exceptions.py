"""Custom exceptions for AudioAuth.

Provides specific exception types for different error scenarios in the watermarking framework.
"""


class AudioAuthError(Exception):
    """Base exception for all AudioAuth-related errors."""
    pass


class ValidationError(AudioAuthError):
    """Raised when validation of inputs or parameters fails."""
    pass


class CheckpointError(AudioAuthError):
    """Raised when there are issues with checkpoint loading or saving."""
    pass


class ConfigError(AudioAuthError):
    """Raised when there are configuration-related errors."""
    pass


class ModelError(AudioAuthError):
    """Raised when there are model-related errors."""
    pass


class DatasetError(AudioAuthError):
    """Raised when there are dataset-related errors."""
    pass


class DistributedError(AudioAuthError):
    """Raised when there are distributed training errors."""
    pass


# Effect Scheduler Exceptions

class EffectSchedulerError(AudioAuthError):
    """Base exception for EffectScheduler errors."""
    pass


class InvalidEffectError(EffectSchedulerError):
    """Raised when an invalid effect is encountered."""
    pass


class InvalidMetricError(EffectSchedulerError):
    """Raised when invalid metric values are provided."""
    pass


class ParameterValidationError(EffectSchedulerError):
    """Raised when parameter validation fails."""
    pass