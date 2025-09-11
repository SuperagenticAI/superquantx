"""Custom exceptions for SuperQuantX.

This module defines custom exceptions used throughout the SuperQuantX library
to provide clear error messages and proper error handling for quantum-agentic
AI research workflows.
"""

from typing import Any


class SuperQuantXError(Exception):
    """Base exception for all SuperQuantX errors."""

    def __init__(self, message: str, error_code: str | None = None) -> None:
        """Initialize SuperQuantX base exception.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling

        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class BackendError(SuperQuantXError):
    """Raised when quantum backend operations fail."""

    def __init__(
        self,
        message: str,
        backend_name: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize backend error.

        Args:
            message: Error description
            backend_name: Name of the backend that failed
            original_error: Original exception that caused this error

        """
        if backend_name:
            full_message = f"Backend '{backend_name}': {message}"
        else:
            full_message = message

        super().__init__(full_message, error_code="BACKEND_ERROR")
        self.backend_name = backend_name
        self.original_error = original_error


class BackendNotAvailableError(BackendError):
    """Raised when a required quantum backend is not available."""

    def __init__(self, backend_name: str, missing_dependencies: list[str] | None = None) -> None:
        """Initialize backend not available error.

        Args:
            backend_name: Name of the unavailable backend
            missing_dependencies: List of missing dependency packages

        """
        if missing_dependencies:
            deps = ", ".join(missing_dependencies)
            message = f"Backend not available. Missing dependencies: {deps}. Install with: pip install superquantx[{backend_name}]"
        else:
            message = f"Backend not available. Install with: pip install superquantx[{backend_name}]"

        super().__init__(message, backend_name, error_code="BACKEND_NOT_AVAILABLE")
        self.missing_dependencies = missing_dependencies or []


class AlgorithmError(SuperQuantXError):
    """Raised when quantum algorithm operations fail."""

    def __init__(
        self,
        message: str,
        algorithm_name: str | None = None,
        step: str | None = None,
    ) -> None:
        """Initialize algorithm error.

        Args:
            message: Error description
            algorithm_name: Name of the algorithm that failed
            step: Specific step where the error occurred (e.g., 'fit', 'predict')

        """
        if algorithm_name and step:
            full_message = f"{algorithm_name}.{step}(): {message}"
        elif algorithm_name:
            full_message = f"{algorithm_name}: {message}"
        else:
            full_message = message

        super().__init__(full_message, error_code="ALGORITHM_ERROR")
        self.algorithm_name = algorithm_name
        self.step = step


class NotFittedError(AlgorithmError):
    """Raised when an algorithm is used before being fitted."""

    def __init__(self, algorithm_name: str) -> None:
        """Initialize not fitted error.

        Args:
            algorithm_name: Name of the algorithm that needs fitting

        """
        message = "This algorithm instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
        super().__init__(message, algorithm_name, error_code="NOT_FITTED")


class InvalidParameterError(SuperQuantXError):
    """Raised when invalid parameters are provided to algorithms or backends."""

    def __init__(
        self,
        parameter_name: str,
        parameter_value: Any,
        expected_type: str | None = None,
        valid_values: list[Any] | None = None,
    ) -> None:
        """Initialize invalid parameter error.

        Args:
            parameter_name: Name of the invalid parameter
            parameter_value: The invalid value provided
            expected_type: Expected type description
            valid_values: List of valid values (for enumerated parameters)

        """
        if valid_values:
            valid_str = ", ".join(map(str, valid_values))
            message = f"Invalid value for parameter '{parameter_name}': {parameter_value}. Valid values are: {valid_str}"
        elif expected_type:
            message = f"Invalid type for parameter '{parameter_name}': got {type(parameter_value).__name__}, expected {expected_type}"
        else:
            message = f"Invalid value for parameter '{parameter_name}': {parameter_value}"

        super().__init__(message, error_code="INVALID_PARAMETER")
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.expected_type = expected_type
        self.valid_values = valid_values


class QuantumCircuitError(SuperQuantXError):
    """Raised when quantum circuit operations fail."""

    def __init__(
        self,
        message: str,
        circuit_info: dict[str, Any] | None = None,
    ) -> None:
        """Initialize quantum circuit error.

        Args:
            message: Error description
            circuit_info: Dictionary with circuit information (qubits, gates, etc.)

        """
        super().__init__(message, error_code="CIRCUIT_ERROR")
        self.circuit_info = circuit_info or {}


class ConfigurationError(SuperQuantXError):
    """Raised when configuration issues are detected."""

    def __init__(self, message: str, config_key: str | None = None) -> None:
        """Initialize configuration error.

        Args:
            message: Error description
            config_key: The configuration key that caused the error

        """
        if config_key:
            full_message = f"Configuration error for '{config_key}': {message}"
        else:
            full_message = f"Configuration error: {message}"

        super().__init__(full_message, error_code="CONFIG_ERROR")
        self.config_key = config_key


class ResearchModeError(SuperQuantXError):
    """Raised when research-only features are used incorrectly."""

    def __init__(self, message: str, feature_name: str | None = None) -> None:
        """Initialize research mode error.

        Args:
            message: Error description
            feature_name: Name of the research feature

        """
        warning_prefix = "⚠️  RESEARCH SOFTWARE WARNING: "
        if feature_name:
            full_message = f"{warning_prefix}{feature_name}: {message}"
        else:
            full_message = f"{warning_prefix}{message}"

        super().__init__(full_message, error_code="RESEARCH_MODE")
        self.feature_name = feature_name


# Utility functions for error handling
def validate_backend_available(backend_name: str, backend_instance: Any) -> None:
    """Validate that a backend is available and properly initialized.

    Args:
        backend_name: Name of the backend to validate
        backend_instance: The backend instance to check

    Raises:
        BackendNotAvailableError: If backend is not available
        BackendError: If backend is improperly initialized

    """
    if backend_instance is None:
        raise BackendNotAvailableError(backend_name)

    if hasattr(backend_instance, 'is_available') and not backend_instance.is_available():
        raise BackendNotAvailableError(backend_name)


def validate_fitted(algorithm_instance: Any, algorithm_name: str) -> None:
    """Validate that an algorithm has been fitted.

    Args:
        algorithm_instance: The algorithm instance to check
        algorithm_name: Name of the algorithm

    Raises:
        NotFittedError: If algorithm is not fitted

    """
    if not getattr(algorithm_instance, 'is_fitted', False):
        raise NotFittedError(algorithm_name)


def validate_parameter(
    parameter_name: str,
    parameter_value: Any,
    expected_type: type | None = None,
    valid_values: list[Any] | None = None,
    allow_none: bool = False,
) -> None:
    """Validate a parameter value.

    Args:
        parameter_name: Name of the parameter
        parameter_value: Value to validate
        expected_type: Expected Python type
        valid_values: List of valid values
        allow_none: Whether None is an acceptable value

    Raises:
        InvalidParameterError: If parameter is invalid

    """
    if parameter_value is None and allow_none:
        return

    if parameter_value is None and not allow_none:
        raise InvalidParameterError(parameter_name, parameter_value, "non-None value")

    if expected_type and not isinstance(parameter_value, expected_type):
        raise InvalidParameterError(
            parameter_name,
            parameter_value,
            expected_type.__name__
        )

    if valid_values and parameter_value not in valid_values:
        raise InvalidParameterError(parameter_name, parameter_value, valid_values=valid_values)
