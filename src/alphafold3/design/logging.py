"""Structured logging utilities for AlphaFold 3 design modules."""

import os
import sys
import json
import logging
import datetime
from typing import Dict, Any, Optional, Union
import yaml


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON for easier parsing."""
    
    def format(self, record):
        log_obj = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        # Add any extra attributes
        if hasattr(record, "design_metrics") and record.design_metrics:
            log_obj["metrics"] = record.design_metrics
            
        return json.dumps(log_obj)


class TSVFormatter(logging.Formatter):
    """Tab-separated value formatter for better log file readability."""
    
    def format(self, record):
        timestamp = datetime.datetime.fromtimestamp(record.created).isoformat()
        basic_fields = [timestamp, record.levelname, record.name, record.getMessage()]
        
        # Format exception info if present
        if record.exc_info:
            basic_fields.append(self.formatException(record.exc_info))
        else:
            basic_fields.append("")
            
        # Add metrics if present
        if hasattr(record, "design_metrics") and record.design_metrics:
            metrics_str = json.dumps(record.design_metrics)
            basic_fields.append(metrics_str)
        else:
            basic_fields.append("")
            
        return "\t".join(basic_fields)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    This is a thin wrapper around logging.getLogger that ensures
    all loggers use consistent formatting and behavior.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)


def log_with_metrics(logger: logging.Logger, level: int, msg: str, metrics: Dict[str, Any]):
    """Log a message with associated design metrics.
    
    Args:
        logger: The logger instance
        level: Log level (e.g., logging.INFO)
        msg: Log message
        metrics: Dictionary of metrics to include
    """
    if not logger.isEnabledFor(level):
        return
        
    # Create a log record with metrics
    record = logger.makeRecord(
        logger.name, level, None, 0, msg, None, None, None
    )
    record.design_metrics = metrics
    logger.handle(record)


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    config_file: Optional[str] = None,
) -> None:
    """Configure global logging settings.
    
    Args:
        log_level: Minimum log level to output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format to use (json, tsv, or pretty)
        log_file: Optional path to write logs to (in addition to console)
        config_file: Optional YAML/JSON file with detailed logging configuration
    """
    # Reset existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Use external config if provided
    if config_file and os.path.exists(config_file):
        try:
            if config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logging.config.dictConfig(config)
                return
            elif config_file.endswith(('.yaml', '.yml')):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logging.config.dictConfig(config)
                return
        except Exception as e:
            sys.stderr.write(f"Error loading logging config from {config_file}: {e}\n")
            sys.stderr.write("Falling back to default configuration\n")
    
    # Set level
    log_level_num = getattr(logging, log_level.upper(), logging.INFO)
    root.setLevel(log_level_num)
    
    # Create and configure console handler
    console_handler = logging.StreamHandler()
    
    if log_format.lower() == 'json':
        formatter = JSONFormatter()
    elif log_format.lower() == 'tsv':
        formatter = TSVFormatter()
    else:  # pretty/human readable
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except Exception as e:
            sys.stderr.write(f"Error creating log file {log_file}: {e}\n")
    
    # Quiet some overly verbose loggers
    logging.getLogger('absl').setLevel(logging.WARNING)
    logging.getLogger('jax').setLevel(logging.WARNING)
    # quiet alo matplot lib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Log configuration complete
    logger = get_logger("alphafold3.design.logging")
    logger.info(f"Logging configured: level={log_level}, format={log_format}") 