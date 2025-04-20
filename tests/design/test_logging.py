import os
import json
import tempfile
import pytest
import logging

# Import the modules we want to test
from alphafold3.design import logging as af_logging


def test_get_logger():
    """Test that get_logger returns a logger with the expected name."""
    logger = af_logging.get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_json_formatter():
    """Test that JSONFormatter formats log records as expected."""
    formatter = af_logging.JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    
    # Add design metrics
    record.design_metrics = {"loss": 1.0, "plddt": 0.8}
    
    json_str = formatter.format(record)
    log_obj = json.loads(json_str)
    
    assert log_obj["level"] == "INFO"
    assert log_obj["module"] == "test_logger"
    assert log_obj["message"] == "Test message"
    assert "timestamp" in log_obj
    assert "metrics" in log_obj
    assert log_obj["metrics"]["loss"] == 1.0
    assert log_obj["metrics"]["plddt"] == 0.8


def test_tsv_formatter():
    """Test that TSVFormatter formats log records as expected."""
    formatter = af_logging.TSVFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    
    # Add design metrics
    record.design_metrics = {"loss": 1.0, "plddt": 0.8}
    
    tsv_str = formatter.format(record)
    fields = tsv_str.split("\t")
    
    assert len(fields) == 5
    assert fields[1] == "INFO"
    assert fields[2] == "test_logger"
    assert fields[3] == "Test message"
    
    # Check metrics JSON
    metrics = json.loads(fields[4])
    assert metrics["loss"] == 1.0
    assert metrics["plddt"] == 0.8


def test_log_with_metrics(caplog):
    """Test log_with_metrics function using pytest's caplog fixture."""
    logger = af_logging.get_logger("test_metrics")
    
    # Configure caplog to capture log records
    caplog.set_level(logging.INFO)
    
    # Log a message with metrics
    metrics = {"loss": 1.0, "plddt": 0.8}
    af_logging.log_with_metrics(logger, logging.INFO, "Test metrics", metrics)
    
    # Check that the log record contains the message
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.message == "Test metrics"
    assert record.levelname == "INFO"
    assert record.name == "test_metrics"
    
    # Check that the metrics are attached to the record
    assert hasattr(record, "design_metrics")
    assert record.design_metrics["loss"] == 1.0
    assert record.design_metrics["plddt"] == 0.8


def test_configure_logging():
    """Test that configure_logging doesn't crash."""
    # Use temporary file for log output
    with tempfile.NamedTemporaryFile(suffix=".log") as tmp:
        # Configure logging to write to the temp file
        af_logging.configure_logging(
            log_level="INFO",
            log_format="pretty",
            log_file=tmp.name,
        )
        
        # Get a logger and log something
        logger = af_logging.get_logger("test_configure")
        logger.info("Test configure logging")
        
        # Check that something was written to the file
        tmp.flush()
        with open(tmp.name, "r") as f:
            content = f.read()
        assert "Test configure logging" in content


def test_configure_logging_with_json_format():
    """Test configure_logging with JSON format."""
    with tempfile.NamedTemporaryFile(suffix=".log") as tmp:
        af_logging.configure_logging(
            log_level="INFO",
            log_format="json",
            log_file=tmp.name,
        )
        
        logger = af_logging.get_logger("test_json")
        logger.info("Test JSON logging")
        
        # Log with metrics
        metrics = {"loss": 1.0, "plddt": 0.8}
        af_logging.log_with_metrics(logger, logging.INFO, "Test metrics", metrics)
        
        # Check file content
        tmp.flush()
        with open(tmp.name, "r") as f:
            lines = f.readlines()
        
        # Should have at least 2 lines
        assert len(lines) >= 2
        
        # Parse the lines as JSON
        first_record = json.loads(lines[0])
        assert first_record["level"] == "INFO"
        assert first_record["module"] == "test_json"
        assert first_record["message"] == "Test JSON logging"
        
        second_record = json.loads(lines[1])
        assert second_record["message"] == "Test metrics"
        assert "metrics" in second_record
        assert second_record["metrics"]["loss"] == 1.0
        assert second_record["metrics"]["plddt"] == 0.8 