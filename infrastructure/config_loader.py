"""
Configuration Loader Module
Loads and manages YAML configuration with environment variable overrides
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    """Loads and caches YAML configuration file"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if cls._config is not None:
            return cls._config
        
        if config_path is None:
            # Find config.yaml in project root or config directory
            possible_paths = [
                Path(__file__).parent / "config.yaml",
                Path.cwd() / "config" / "config.yaml",
                Path.cwd() / "config.yaml"
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
            
            if config_path is None:
                raise FileNotFoundError(
                    "config.yaml not found. Please create it in project root or config/ directory"
                )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cls._config = yaml.safe_load(f)
        
        # Apply environment variable overrides
        cls._apply_env_overrides()
        
        return cls._config
    
    @classmethod
    def _apply_env_overrides(cls):
        """Override config values with environment variables if they exist"""
        if cls._config is None:
            return
        
        # Example: AUDIO2MBTI_LOGGING_LEVEL overrides logging.level
        for key, value in os.environ.items():
            if key.startswith("AUDIO2MBTI_"):
                # Parse hierarchical key: AUDIO2MBTI_LOGGING_LEVEL -> logging.level
                parts = key[11:].lower().split("_")
                current = cls._config
                
                try:
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Convert value to appropriate type
                    if value.lower() in ["true", "false"]:
                        current[parts[-1]] = value.lower() == "true"
                    elif value.isdigit():
                        current[parts[-1]] = int(value)
                    else:
                        current[parts[-1]] = value
                except (KeyError, TypeError, AttributeError):
                    pass  # Skip invalid paths
    
    @classmethod
    def get(cls, key_path: str = None, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'logging.level')"""
        if cls._config is None:
            cls.load()
        
        if key_path is None:
            return cls._config
        
        keys = key_path.split(".")
        value = cls._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @classmethod
    def reload(cls):
        """Force reload configuration"""
        cls._config = None
        return cls.load()


def get_logger(name: str) -> logging.Logger:
    """Get or create logger with configured settings"""
    config = ConfigLoader.get("logging", {})
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Only configure if not already configured
        # Set level
        level = getattr(logging, config.get("level", "INFO"))
        logger.setLevel(level)
        
        # Console handler
        if config.get("console_output", True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(
                config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handler
        log_file = config.get("file")
        if log_file:
            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)
            
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=config.get("max_file_size", 10485760),
                backupCount=config.get("backup_count", 5)
            )
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    return logger
