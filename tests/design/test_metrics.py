import os
import json
import tempfile
import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Import the modules we want to test
from alphafold3.design import metrics


def test_convert_jax_metrics():
    """Test conversion of JAX values to Python types."""
    # Create test metrics with various types
    test_metrics = {
        "float_jax": jnp.float32(1.5),
        "int_jax": jnp.int32(42),
        "array_scalar_jax": jnp.array(2.5),
        "array_vector_jax": jnp.array([1.0, 2.0, 3.0]),
        "numpy_scalar": np.float32(3.5),
        "numpy_array": np.array([4.0, 5.0, 6.0]),
        "python_float": 6.5,
        "python_int": 7,
        "python_str": "test",
        "python_bool": True,
    }
    
    # Convert the metrics
    result = metrics.convert_jax_metrics(test_metrics)
    
    # Check the results
    assert isinstance(result["float_jax"], float)
    assert result["float_jax"] == 1.5
    assert isinstance(result["int_jax"], float)  # JAX ints convert to float
    assert result["int_jax"] == 42.0
    assert isinstance(result["array_scalar_jax"], float)
    assert result["array_scalar_jax"] == 2.5
    assert isinstance(result["numpy_scalar"], float)
    assert result["numpy_scalar"] == 3.5
    assert isinstance(result["python_float"], float)
    assert result["python_float"] == 6.5
    assert isinstance(result["python_int"], int)
    assert result["python_int"] == 7
    assert isinstance(result["python_str"], str)
    assert result["python_str"] == "test"
    assert isinstance(result["python_bool"], bool)
    assert result["python_bool"] is True
    
    # Check that multi-dimensional arrays are skipped
    assert "array_vector_jax" not in result
    assert "numpy_array" not in result


def test_metrics_collector_basic():
    """Test basic functionality of MetricsCollector."""
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a metrics collector
        collector = metrics.create_metrics_collector(
            output_dir=tmpdir,
            run_name="test_run",
            write_interval=1  # Write on every record
        )
        
        # Record some metrics
        test_metrics = {"loss": 1.0, "plddt": 0.8}
        collector.record(step=0, metrics=test_metrics)
        
        # Record more metrics
        test_metrics2 = {"loss": 0.8, "plddt": 0.9}
        collector.record(step=1, metrics=test_metrics2)
        
        # Check the metrics history
        assert len(collector.metrics_history) == 2
        assert collector.metrics_history[0]["loss"] == 1.0
        assert collector.metrics_history[0]["plddt"] == 0.8
        assert collector.metrics_history[0]["step"] == 0
        assert "timestamp" in collector.metrics_history[0]
        
        assert collector.metrics_history[1]["loss"] == 0.8
        assert collector.metrics_history[1]["plddt"] == 0.9
        assert collector.metrics_history[1]["step"] == 1
        
        # Check that metrics file was created
        metrics_dir = os.path.join(tmpdir, "metrics")
        assert os.path.isdir(metrics_dir)
        
        # Should be one .json file in the metrics directory
        files = os.listdir(metrics_dir)
        assert len(files) == 1
        assert files[0].endswith(".json")
        
        # Check file contents
        with open(os.path.join(metrics_dir, files[0]), "r") as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Parse the lines
        record1 = json.loads(lines[0])
        record2 = json.loads(lines[1])
        
        assert record1["loss"] == 1.0
        assert record1["plddt"] == 0.8
        assert record1["step"] == 0
        
        assert record2["loss"] == 0.8
        assert record2["plddt"] == 0.9
        assert record2["step"] == 1
        
        # Close the collector
        collector.close()


def test_metrics_collector_csv():
    """Test MetricsCollector with CSV output format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a metrics collector with CSV format
        collector = metrics.create_metrics_collector(
            output_dir=tmpdir,
            run_name="test_csv",
            format="csv",
            write_interval=1
        )
        
        # Record some metrics
        test_metrics = {"loss": 1.0, "plddt": 0.8, "step_name": "init"}
        collector.record(step=0, metrics=test_metrics)
        
        # Record more metrics
        test_metrics2 = {"loss": 0.8, "plddt": 0.9, "step_name": "final"}
        collector.record(step=1, metrics=test_metrics2)
        
        # Check that metrics file was created
        metrics_dir = os.path.join(tmpdir, "metrics")
        files = os.listdir(metrics_dir)
        assert len(files) == 1
        assert files[0].endswith(".csv")
        
        # Check file contents (should be CSV format)
        import csv
        with open(os.path.join(metrics_dir, files[0]), "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2
        
        # Check the values
        assert float(rows[0]["loss"]) == 1.0
        assert float(rows[0]["plddt"]) == 0.8
        assert rows[0]["step_name"] == "init"
        assert int(rows[0]["step"]) == 0
        
        assert float(rows[1]["loss"]) == 0.8
        assert float(rows[1]["plddt"]) == 0.9
        assert rows[1]["step_name"] == "final"
        assert int(rows[1]["step"]) == 1
        
        # Close the collector
        collector.close()


def test_get_best_metrics():
    """Test the get_best_metrics method of MetricsCollector."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = metrics.create_metrics_collector(
            output_dir=tmpdir,
            run_name="test_best",
            write_interval=10  # Don't write for this test
        )
        
        # Record some metrics with varying loss values
        collector.record(step=0, metrics={"loss": 3.0, "plddt": 0.5})
        collector.record(step=1, metrics={"loss": 2.0, "plddt": 0.6})
        collector.record(step=2, metrics={"loss": 1.0, "plddt": 0.7})
        collector.record(step=3, metrics={"loss": 1.5, "plddt": 0.8})
        
        # Get best metrics for loss (min)
        best = collector.get_best_metrics(key="loss", mode="min")
        assert best["loss"] == 1.0
        assert best["plddt"] == 0.7
        assert best["step"] == 2
        
        # Get best metrics for plddt (max)
        best = collector.get_best_metrics(key="plddt", mode="max")
        assert best["loss"] == 1.5
        assert best["plddt"] == 0.8
        assert best["step"] == 3
        
        # Close the collector
        collector.close()


def test_get_last_metrics():
    """Test the get_last_metrics method of MetricsCollector."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = metrics.create_metrics_collector(
            output_dir=tmpdir,
            run_name="test_last",
            write_interval=10  # Don't write for this test
        )
        
        # Record some metrics
        collector.record(step=0, metrics={"loss": 3.0, "plddt": 0.5})
        collector.record(step=1, metrics={"loss": 2.0, "plddt": 0.6})
        
        # Get last metrics
        last = collector.get_last_metrics()
        assert last["loss"] == 2.0
        assert last["plddt"] == 0.6
        assert last["step"] == 1
        
        # Record another metric
        collector.record(step=2, metrics={"loss": 1.0, "plddt": 0.7})
        
        # Get updated last metrics
        last = collector.get_last_metrics()
        assert last["loss"] == 1.0
        assert last["plddt"] == 0.7
        assert last["step"] == 2
        
        # Close the collector
        collector.close() 