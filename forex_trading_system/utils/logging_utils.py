

import logging
import logging.config
import yaml
from pathlib import Path
import os
import sys


def find_project_root() -> Path:
    """
    Find the project root directory by looking for the config directory.
    Returns absolute path to project root.
    """
    # Start with the current working directory
    current_dir = Path.cwd()

    # Also check the parent directory if we're in a notebook subdirectory
    parent_dir = current_dir.parent

    # Possible project root indicators
    indicators = ['config', 'setup.py', '.git']

    # Check current directory first
    for indicator in indicators:
        if (current_dir / indicator).exists():
            return current_dir

    # Check parent directory
    for indicator in indicators:
        if (parent_dir / indicator).exists():
            return parent_dir

    # If no project root found, use current directory and warn
    logging.warning(
        f"Project root not found. Using current directory: {current_dir}"
    )
    return current_dir


def setup_logging(
    default_level: int = logging.INFO,
    env_key: str = 'LOG_CFG'
) -> None:
    """
    Setup logging configuration with better path handling.

    Args:
        default_level: Default logging level if config file is not found
        env_key: Environment variable that can override config path
    """
    # Find project root
    project_root = find_project_root()

    # Check for config path in environment variable
    config_path = os.getenv(env_key, None)
    if config_path is None:
        config_path = project_root / 'config' / 'logging_config.yaml'
    else:
        config_path = Path(config_path)

    # Create logs directory if it doesn't exist
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)

    if config_path.exists():
        with open(config_path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())

                # Update log file path to be relative to project root
                log_file = config['handlers']['file']['filename']
                if not os.path.isabs(log_file):
                    config['handlers']['file']['filename'] = str(
                        project_root / log_file
                    )

                logging.config.dictConfig(config)
                print(f"Logging configuration loaded from {config_path}")

            except Exception as e:
                print(f"Error loading logging configuration: {str(e)}")
                print("Using default logging configuration")
                setup_default_logging(default_level, logs_dir)
    else:
        print(f"Logging config file not found at {config_path}")
        print("Using default logging configuration")
        setup_default_logging(default_level, logs_dir)


def setup_default_logging(level: int, logs_dir: Path) -> None:
    """
    Setup a default logging configuration if the config file is not found.

    Args:
        level: Logging level to use
        logs_dir: Directory to store log files
    """
    # Create a default formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup file handler
    file_handler = logging.FileHandler(
        logs_dir / 'trading_system.log',
        mode='a',
        encoding='utf8'
    )
    file_handler.setFormatter(formatter)

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Name of the logger (typically __name__ of the module)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
