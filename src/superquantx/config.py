"""Global configuration for SuperQuantX.

This module provides configuration management for the SuperQuantX package,
including backend settings, default parameters, and runtime options.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


logger = logging.getLogger(__name__)

class Config:
    """Global configuration class for SuperQuantX."""

    def __init__(self) -> None:
        """Initialize configuration with defaults."""
        self._config: Dict[str, Any] = {
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

    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Merge new configuration with existing configuration."""
        def merge_dicts(base: Dict, update: Dict) -> Dict:
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

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with a dictionary of values."""
        self._merge_config(updates)

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
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

    def to_dict(self) -> Dict[str, Any]:
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
    backend: Optional[str] = None,
    shots: Optional[int] = None,
    debug: Optional[bool] = None,
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

def get_platform_config(platform: str) -> Dict[str, Any]:
    """Get configuration for a specific platform.
    
    Args:
        platform: Platform name (e.g., 'ibm', 'braket', 'azure')
        
    Returns:
        Platform configuration dictionary

    """
    return config.get(f"platforms.{platform}", {})

# Initialize logging on import
setup_logging()
