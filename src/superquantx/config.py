"""Global configuration for SuperQuantX.

This module provides configuration management for the SuperQuantX package,
including backend settings, default parameters, and runtime options.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)

class Config:
    """Global configuration class for SuperQuantX."""

    def __init__(self) -> None:
        """Initialize configuration with defaults."""
        self._config: dict[str, Any] = {
            # Backend configuration
            "backends": {
                "default": "auto",
                "auto_selection_strategy": "performance",  # "performance", "cost", "speed", "accuracy"
                "fallback_backend": "pennylane",
                "timeout": 300,  # seconds
                "max_retries": 3,
            },

            # Algorithm defaults
            "algorithms": {
                "default_shots": 1024,
                "optimization_level": 1,
                "seed": None,
                "max_iterations": 1000,
                "convergence_tolerance": 1e-6,
            },

            # Visualization settings
            "visualization": {
                "backend": "matplotlib",  # "matplotlib", "plotly", "both"
                "style": "default",
                "save_figures": False,
                "figure_format": "png",
                "dpi": 150,
            },

            # Benchmarking configuration
            "benchmarks": {
                "default_runs": 5,
                "include_classical": True,
                "save_results": True,
                "results_dir": "benchmark_results",
            },

            # Logging configuration
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None,  # Log file path, None for console only
            },

            # Cache settings
            "cache": {
                "enabled": True,
                "directory": ".superquantx_cache",
                "max_size_mb": 1000,
                "ttl_hours": 24,
            },

            # Platform-specific settings
            "platforms": {
                "ibm": {
                    "token": None,
                    "hub": "ibm-q",
                    "group": "open",
                    "project": "main",
                },
                "braket": {
                    "s3_folder": None,
                    "device_arn": None,
                },
                "azure": {
                    "resource_id": None,
                    "location": None,
                },
                "dwave": {
                    "token": None,
                    "solver": None,
                },
                "rigetti": {
                    "api_key": None,
                    "user_id": None,
                },
            },

            # Development settings
            "development": {
                "debug": False,
                "profile": False,
                "warnings": True,
            },
        }

        # Load configuration from files
        self._load_config_files()

        # Override with environment variables
        self._load_environment_variables()

    def _load_config_files(self) -> None:
        """Load configuration from YAML/JSON files."""
        config_paths = [
            Path.home() / ".superquantx" / "config.yaml",
            Path.home() / ".superquantx" / "config.yml",
            Path.home() / ".superquantx" / "config.json",
            Path.cwd() / "superquantx_config.yaml",
            Path.cwd() / "superquantx_config.yml",
            Path.cwd() / "superquantx_config.json",
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        if config_path.suffix.lower() in [".yaml", ".yml"]:
                            file_config = yaml.safe_load(f)
                        else:
                            file_config = json.load(f)

                    if file_config:
                        self._merge_config(file_config)
                        logger.info(f"Loaded configuration from {config_path}")

                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")

    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "SUPERQUANTX_DEFAULT_BACKEND": ("backends", "default"),
            "SUPERQUANTX_DEFAULT_SHOTS": ("algorithms", "default_shots"),
            "SUPERQUANTX_LOG_LEVEL": ("logging", "level"),
            "SUPERQUANTX_CACHE_ENABLED": ("cache", "enabled"),
            "SUPERQUANTX_DEBUG": ("development", "debug"),

            # Platform tokens
            "IBM_QUANTUM_TOKEN": ("platforms", "ibm", "token"),
            "AWS_BRAKET_S3_FOLDER": ("platforms", "braket", "s3_folder"),
            "AZURE_QUANTUM_RESOURCE_ID": ("platforms", "azure", "resource_id"),
            "DWAVE_API_TOKEN": ("platforms", "dwave", "token"),
            "RIGETTI_API_KEY": ("platforms", "rigetti", "api_key"),
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, value)

    def _merge_config(self, new_config: dict[str, Any]) -> None:
        """Merge new configuration with existing configuration."""
        def merge_dicts(base: dict, update: dict) -> dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = merge_dicts(base[key], value)
                else:
                    base[key] = value
            return base

        self._config = merge_dicts(self._config, new_config)

    def _set_nested_value(self, path: tuple, value: Any) -> None:
        """Set a nested configuration value."""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Type conversion for common types
        if isinstance(value, str):
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif "." in value and value.replace(".", "").isdigit():
                value = float(value)

        current[path[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split(".")
        current = self._config

        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split(".")
        current = self._config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def update(self, updates: dict[str, Any]) -> None:
        """Update configuration with a dictionary of values."""
        self._merge_config(updates)

    def save(self, path: str | Path | None = None) -> None:
        """Save current configuration to file."""
        if path is None:
            path = Path.home() / ".superquantx" / "config.yaml"
        else:
            path = Path(path)

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {path}: {e}")

    def reset(self) -> None:
        """Reset configuration to defaults."""
        self.__init__()

    def to_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()

    def print_config(self) -> None:
        """Print current configuration."""
        import pprint
        pprint.pprint(self._config, indent=2)

    def validate(self) -> bool:
        """Validate current configuration."""
        # Add validation logic here
        return True

# Global configuration instance
config = Config()

def configure(
    backend: str | None = None,
    shots: int | None = None,
    debug: bool | None = None,
    **kwargs
) -> None:
    """Configure SuperQuantX settings.

    Args:
        backend: Default backend to use
        shots: Default number of shots for quantum circuits
        debug: Enable debug mode
        **kwargs: Additional configuration parameters

    """
    updates = {}

    if backend is not None:
        updates["backends.default"] = backend

    if shots is not None:
        updates["algorithms.default_shots"] = shots

    if debug is not None:
        updates["development.debug"] = debug

    # Handle additional keyword arguments
    for key, value in kwargs.items():
        updates[key] = value

    # Apply updates
    for key, value in updates.items():
        config.set(key, value)

    # Setup logging if level changed
    if "logging.level" in updates:
        setup_logging()

def setup_logging() -> None:
    """Setup logging configuration."""
    level = getattr(logging, config.get("logging.level", "INFO").upper())
    format_str = config.get("logging.format")
    log_file = config.get("logging.file")

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_str,
        filename=log_file,
        filemode="a" if log_file else None,
        force=True,
    )

def get_platform_config(platform: str) -> dict[str, Any]:
    """Get configuration for a specific platform.

    Args:
        platform: Platform name (e.g., 'ibm', 'braket', 'azure')

    Returns:
        Platform configuration dictionary

    """
    return config.get(f"platforms.{platform}", {})

def create_default_config(path: str = "./superquantx.json", format: str = "json") -> None:
    """Generate default configuration file.
    
    Args:
        path: Path where to save the configuration file
        format: Format of the file ('json' or 'yaml')
    """
    default_config = {
        "default_backend": "simulator",
        "logging": {
            "level": "INFO",
            "file": "superquantx.log",
            "console": True
        },
        "simulation": {
            "max_qubits": 20,
            "default_shots": 1000,
            "seed": None
        },
        "backends": {
            "simulator": {
                "enabled": True,
                "device": "CPU"
            },
            "pennylane": {
                "enabled": True,
                "device": "default.qubit"
            },
            "qiskit": {
                "enabled": False,
                "provider": "local"
            }
        }
    }
    
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "json":
        with open(path_obj, "w") as f:
            json.dump(default_config, f, indent=2)
    elif format.lower() in ["yaml", "yml"]:
        with open(path_obj, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")

def configure_interactive():
    """Interactive configuration wizard.
    
    Returns:
        Configuration object that can be saved
    """
    print("SuperQuantX Interactive Configuration")
    print("====================================")
    
    # For now, return a basic config object
    # In a real implementation, this would ask interactive questions
    class InteractiveConfig:
        def __init__(self):
            self.config_data = {
                "default_backend": "simulator",
                "logging": {"level": "INFO"},
                "simulation": {"max_qubits": 20}
            }
        
        def save(self, path: str):
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(path_obj, "w") as f:
                json.dump(self.config_data, f, indent=2)
    
    return InteractiveConfig()

def load_config(config_path=None) -> None:
    """Load configuration from file(s).
    
    Args:
        config_path: Path to config file or list of paths. If None, uses default locations.
    """
    if config_path is None:
        # Reload from default locations
        config._load_config_files()
        return
    
    if isinstance(config_path, str):
        config_path = [config_path]
    
    for path in config_path:
        path_obj = Path(path)
        if path_obj.exists():
            try:
                with open(path_obj) as f:
                    if path_obj.suffix.lower() in [".yaml", ".yml"]:
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                
                if file_config:
                    config._merge_config(file_config)
                    logger.info(f"Loaded configuration from {path_obj}")
            except Exception as e:
                logger.warning(f"Failed to load config from {path_obj}: {e}")

def configure_logging(level="INFO", file=None, console=True, format=None, max_file_size="10MB", backup_count=3) -> None:
    """Configure logging settings.
    
    Args:
        level: Logging level
        file: Log file path
        console: Whether to log to console
        format: Log format string
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    config.set("logging.level", level)
    if file is not None:
        config.set("logging.file", file)
    config.set("logging.console", console)
    if format is not None:
        config.set("logging.format", format)
    config.set("logging.max_file_size", max_file_size)
    config.set("logging.backup_count", backup_count)
    
    # Re-setup logging with new settings
    setup_logging()

def configure_simulation(max_qubits=None, default_shots=None, memory_limit=None) -> None:
    """Configure simulation settings.
    
    Args:
        max_qubits: Maximum number of qubits to simulate
        default_shots: Default number of shots
        memory_limit: Memory limit for simulations
    """
    if max_qubits is not None:
        config.set("algorithms.max_qubits", max_qubits)
    if default_shots is not None:
        config.set("algorithms.default_shots", default_shots)
    if memory_limit is not None:
        config.set("simulation.memory_limit", memory_limit)

class ConfigContext:
    """Context manager for temporary configuration changes."""
    
    def __init__(self, **kwargs):
        self.changes = kwargs
        self.original_values = {}
    
    def __enter__(self):
        # Save original values
        for key, value in self.changes.items():
            if key == "backend":
                key = "backends.default"
            elif key == "shots":
                key = "algorithms.default_shots"
            
            self.original_values[key] = config.get(key)
            config.set(key, value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        for key, value in self.original_values.items():
            if value is not None:
                config.set(key, value)

def config_context(**kwargs):
    """Create a configuration context manager.
    
    Usage:
        with config_context(backend="qiskit", shots=10000):
            # Code here uses the specified configuration
            pass
    """
    return ConfigContext(**kwargs)

def get_default_backend() -> str:
    """Get the default backend name."""
    return config.get("backends.default", "simulator")

def get_config(key: str = None) -> Any:
    """Get configuration value(s).
    
    Args:
        key: Configuration key (dot notation). If None, returns all config.
    
    Returns:
        Configuration value or full config dict
    """
    if key is None:
        return config.to_dict()
    return config.get(key)

def get_config_search_paths() -> list:
    """Get list of configuration search paths."""
    return [
        str(Path.home() / ".superquantx" / "config.yaml"),
        str(Path.home() / ".superquantx" / "config.yml"),
        str(Path.home() / ".superquantx" / "config.json"),
        str(Path.cwd() / "superquantx_config.yaml"),
        str(Path.cwd() / "superquantx_config.yml"),
        str(Path.cwd() / "superquantx_config.json"),
    ]

def get_active_config_path() -> str:
    """Get path of currently active configuration file."""
    # For simplicity, return the first existing path
    for path in get_config_search_paths():
        if Path(path).exists():
            return path
    return "Built-in defaults"

class ValidationResult:
    """Result of configuration validation."""
    
    def __init__(self, is_valid: bool, errors: list = None):
        self.is_valid = is_valid
        self.errors = errors or []

def validate_config() -> ValidationResult:
    """Validate current configuration.
    
    Returns:
        ValidationResult with validation status and errors
    """
    errors = []
    
    # Basic validation - check required keys exist
    required_keys = ["backends", "algorithms", "logging"]
    for key in required_keys:
        if config.get(key) is None:
            errors.append(f"Missing required section: {key}")
    
    # Validate backend configuration
    backends = config.get("backends", {})
    if not isinstance(backends, dict):
        errors.append("backends section must be a dictionary")
    
    # Validate logging level
    log_level = config.get("logging.level", "INFO")
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        errors.append(f"Invalid logging level: {log_level}. Must be one of {valid_levels}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)

class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass

def validate_config_schema(config_path: str) -> None:
    """Validate configuration file against schema.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        ConfigValidationError: If validation fails
    """
    path_obj = Path(config_path)
    if not path_obj.exists():
        raise ConfigValidationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(path_obj) as f:
            if path_obj.suffix.lower() in [".yaml", ".yml"]:
                file_config = yaml.safe_load(f)
            else:
                file_config = json.load(f)
    except Exception as e:
        raise ConfigValidationError(f"Invalid configuration file format: {e}")
    
    if not isinstance(file_config, dict):
        raise ConfigValidationError("Configuration must be a dictionary")

def configure_performance(parallel_backends=None, thread_count=None, memory_limit=None, gpu_enabled=None, optimization_level=None) -> None:
    """Configure performance settings.
    
    Args:
        parallel_backends: Number of backends to run in parallel
        thread_count: Number of threads per backend
        memory_limit: Memory limit for simulations
        gpu_enabled: Enable GPU acceleration
        optimization_level: Optimization level (0-3)
    """
    if parallel_backends is not None:
        config.set("performance.parallel_backends", parallel_backends)
    if thread_count is not None:
        config.set("performance.thread_count", thread_count)
    if memory_limit is not None:
        config.set("performance.memory_limit", memory_limit)
    if gpu_enabled is not None:
        config.set("performance.gpu_enabled", gpu_enabled)
    if optimization_level is not None:
        config.set("simulation.optimization_level", optimization_level)

def configure_experiments(experiment_dir="./experiments", auto_save_results=True, save_circuits=True, save_metadata=True, version_control=True) -> None:
    """Configure experiment tracking.
    
    Args:
        experiment_dir: Directory to save experiments
        auto_save_results: Automatically save experiment results
        save_circuits: Save quantum circuits
        save_metadata: Save experiment metadata
        version_control: Enable version control for experiments
    """
    config.set("experiments.directory", experiment_dir)
    config.set("experiments.auto_save_results", auto_save_results)
    config.set("experiments.save_circuits", save_circuits)
    config.set("experiments.save_metadata", save_metadata)
    config.set("experiments.version_control", version_control)

class ExperimentContext:
    """Context manager for experiment tracking."""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.metrics = {}
        self.artifacts = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # In a real implementation, this would save the experiment data
        logger.info(f"Experiment '{self.name}' completed")
        logger.info(f"Parameters: {self.parameters}")
        logger.info(f"Metrics: {self.metrics}")
        logger.info(f"Artifacts: {list(self.artifacts.keys())}")
    
    def log_parameter(self, key: str, value: Any) -> None:
        """Log an experiment parameter."""
        self.parameters[key] = value
    
    def log_metric(self, key: str, value: Any) -> None:
        """Log an experiment metric."""
        self.metrics[key] = value
    
    def log_artifact(self, filename: str, data: Any) -> None:
        """Log an experiment artifact."""
        self.artifacts[filename] = data

def experiment(name: str):
    """Create an experiment context.
    
    Args:
        name: Name of the experiment
        
    Returns:
        ExperimentContext for tracking the experiment
    """
    return ExperimentContext(name)

# Profile management
_profiles = {}

def create_profile(name: str, config_dict: dict) -> None:
    """Create a configuration profile.
    
    Args:
        name: Profile name
        config_dict: Configuration dictionary
    """
    _profiles[name] = config_dict.copy()

def activate_profile(name: str) -> None:
    """Activate a configuration profile.
    
    Args:
        name: Profile name
    """
    if name not in _profiles:
        raise ValueError(f"Profile '{name}' not found")
    
    config._merge_config(_profiles[name])
    logger.info(f"Activated profile: {name}")

class ProfileContext:
    """Context manager for temporary profile activation."""
    
    def __init__(self, name: str):
        self.name = name
        self.original_config = None
    
    def __enter__(self):
        if self.name not in _profiles:
            raise ValueError(f"Profile '{self.name}' not found")
        
        # Save current config
        self.original_config = config.to_dict()
        
        # Apply profile
        config._merge_config(_profiles[self.name])
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original config
        if self.original_config:
            config._config = self.original_config

def profile(name: str):
    """Create a profile context manager.
    
    Args:
        name: Profile name
        
    Returns:
        ProfileContext for temporary profile activation
    """
    return ProfileContext(name)

# Initialize logging on import
setup_logging()
