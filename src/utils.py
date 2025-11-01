import logging
import os
import sys
from collections import namedtuple

# Define the structure for episode statistics
EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards"])

# Valid actions for Breakout
VALID_ACTIONS = [0, 1, 2, 3]

def setup_logging(log_level="INFO", log_dir="logs"):
    """
    Configures the logging module to log to both a file and the console.
    
    Args:
        log_level (str): The logging level (e.g., "DEBUG", "INFO").
        log_dir (str): The directory to save log files.
    """
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, "training.log")
    
    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
        
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Configure file handler
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Logging setup complete. Logging to console and "
                f"{log_filename}")
