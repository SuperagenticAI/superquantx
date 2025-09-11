"""Logging configuration for SuperQuantX.

This module provides centralized logging configuration for the SuperQuantX library,
including structured logging, performance tracking, and research experiment logging.
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any

from loguru import logger


class SuperQuantXLogger:
    """Centralized logger for SuperQuantX with research-focused features.

    This logger provides:
    - Structured logging for research experiments
    - Performance tracking and benchmarking
    - Backend-specific logging with context
    - Algorithm execution tracing
    - Research reproducibility aids
    """

    def __init__(self) -> None:
        """Initialize SuperQuantX logger."""
        self._configured = False
        self._log_level = "INFO"
        self._log_file: Path | None = None
        self._experiment_id: str | None = None

    def configure(
        self,
        level: str | int = "INFO",
        log_file: str | Path | None = None,
        experiment_id: str | None = None,
        enable_research_mode: bool = True,
        silence_warnings: bool = True,
    ) -> None:
        """Configure SuperQuantX logging.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for log output
            experiment_id: Unique identifier for research experiments
            enable_research_mode: Enable research-specific logging features
            silence_warnings: Whether to silence non-critical warnings

        """
        if self._configured:
            logger.warning("SuperQuantX logger already configured")
            return

        # Remove default loguru handler
        logger.remove()

        # Configure console output
        log_format = self._get_log_format(enable_research_mode, experiment_id)

        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # Configure file output if requested
        if log_file:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                log_file_path,
                format=log_format,
                level=level,
                rotation="10 MB",
                retention="1 week",
                compression="zip",
                backtrace=True,
                diagnose=True,
            )
            self._log_file = log_file_path

        # Set internal state
        self._log_level = level
        self._experiment_id = experiment_id

        # Configure standard library logging
        self._configure_stdlib_logging(level)

        # Handle warnings
        if silence_warnings:
            self._configure_warnings()

        # Log research disclaimer
        if enable_research_mode:
            self._log_research_disclaimer()

        self._configured = True
        logger.info("SuperQuantX logging configured", extra={"component": "logging"})

    def _get_log_format(self, enable_research_mode: bool, experiment_id: str | None) -> str:
        """Get logging format string based on configuration."""
        base_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} | {message}"

        if enable_research_mode and experiment_id:
            return f"[{experiment_id}] {base_format}"
        elif enable_research_mode:
            return f"[RESEARCH] {base_format}"

        return base_format

    def _configure_stdlib_logging(self, level: str | int) -> None:
        """Configure standard library logging to work with loguru."""
        class InterceptHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                # Get corresponding Loguru level if it exists
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno

                # Find caller from where originated the logged message
                frame, depth = sys._getframe(6), 6
                while frame and frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1

                logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

        # Intercept standard library logging
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

        # Set levels for common quantum computing libraries
        logging.getLogger("pennylane").setLevel(logging.WARNING)
        logging.getLogger("qiskit").setLevel(logging.WARNING)
        logging.getLogger("cirq").setLevel(logging.WARNING)
        logging.getLogger("braket").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

    def _configure_warnings(self) -> None:
        """Configure warning handling for research use."""
        # Capture warnings with loguru
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Still show SuperQuantX-specific warnings
        warnings.filterwarnings("default", category=UserWarning, module="superquantx.*")

    def _log_research_disclaimer(self) -> None:
        """Log the research software disclaimer."""
        logger.warning(
            "⚠️  RESEARCH SOFTWARE WARNING: SuperQuantX is experimental research software. "
            "NOT intended for production use. For research and educational purposes only.",
            extra={"component": "disclaimer"}
        )

    def get_experiment_logger(self, experiment_name: str) -> "loguru.Logger":
        """Get a logger for a specific research experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Configured logger instance with experiment context

        """
        return logger.bind(experiment=experiment_name, component="experiment")

    def get_algorithm_logger(self, algorithm_name: str) -> "loguru.Logger":
        """Get a logger for a specific quantum algorithm.

        Args:
            algorithm_name: Name of the algorithm

        Returns:
            Configured logger instance with algorithm context

        """
        return logger.bind(algorithm=algorithm_name, component="algorithm")

    def get_backend_logger(self, backend_name: str) -> "loguru.Logger":
        """Get a logger for a specific quantum backend.

        Args:
            backend_name: Name of the backend

        Returns:
            Configured logger instance with backend context

        """
        return logger.bind(backend=backend_name, component="backend")

    def log_performance(
        self,
        operation: str,
        duration: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log performance metrics for research analysis.

        Args:
            operation: Name of the operation being measured
            duration: Duration in seconds
            metadata: Additional performance metadata

        """
        performance_data = {
            "operation": operation,
            "duration_seconds": duration,
            "component": "performance"
        }

        if metadata:
            performance_data.update(metadata)

        logger.info(f"Performance: {operation} completed in {duration:.3f}s", extra=performance_data)

    def log_experiment_result(
        self,
        experiment_name: str,
        results: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log experiment results for research tracking.

        Args:
            experiment_name: Name of the experiment
            results: Experiment results dictionary
            metadata: Additional experiment metadata

        """
        result_data = {
            "experiment": experiment_name,
            "results": results,
            "component": "results"
        }

        if metadata:
            result_data.update(metadata)

        logger.info(f"Experiment '{experiment_name}' completed", extra=result_data)

    @property
    def is_configured(self) -> bool:
        """Check if logger has been configured."""
        return self._configured

    @property
    def log_file(self) -> Path | None:
        """Get the current log file path."""
        return self._log_file

    @property
    def experiment_id(self) -> str | None:
        """Get the current experiment ID."""
        return self._experiment_id


# Global logger instance
_logger_instance: SuperQuantXLogger | None = None


def get_logger() -> SuperQuantXLogger:
    """Get the global SuperQuantX logger instance.

    Returns:
        Global logger instance

    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SuperQuantXLogger()
    return _logger_instance


def configure_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    experiment_id: str | None = None,
    enable_research_mode: bool = True,
    silence_warnings: bool = True,
) -> None:
    """Configure SuperQuantX logging (convenience function).

    Args:
        level: Logging level
        log_file: Optional log file path
        experiment_id: Experiment identifier
        enable_research_mode: Enable research logging features
        silence_warnings: Silence non-critical warnings

    """
    logger_instance = get_logger()
    logger_instance.configure(
        level=level,
        log_file=log_file,
        experiment_id=experiment_id,
        enable_research_mode=enable_research_mode,
        silence_warnings=silence_warnings,
    )


# Convenience functions for common logging patterns
def log_backend_operation(backend_name: str, operation: str, **kwargs: Any) -> None:
    """Log a backend operation with context."""
    backend_logger = get_logger().get_backend_logger(backend_name)
    backend_logger.info(f"{operation}", extra=kwargs)


def log_algorithm_step(algorithm_name: str, step: str, **kwargs: Any) -> None:
    """Log an algorithm step with context."""
    algorithm_logger = get_logger().get_algorithm_logger(algorithm_name)
    algorithm_logger.info(f"{step}", extra=kwargs)


def log_research_warning(message: str, **kwargs: Any) -> None:
    """Log a research-specific warning."""
    logger.warning(f"🔬 RESEARCH: {message}", extra=kwargs)


def log_quantum_circuit_info(
    backend: str,
    n_qubits: int,
    n_gates: int,
    shots: int,
    **kwargs: Any
) -> None:
    """Log quantum circuit execution information."""
    circuit_info = {
        "backend": backend,
        "qubits": n_qubits,
        "gates": n_gates,
        "shots": shots,
        "component": "circuit"
    }
    circuit_info.update(kwargs)

    logger.info(
        f"Executing {n_gates} gates on {n_qubits} qubits ({shots} shots) via {backend}",
        extra=circuit_info
    )
