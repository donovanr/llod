"""
Tests for the calc.py command-line tool
"""

import os
import subprocess
import sys
import tempfile
import pandas as pd
from pathlib import Path


# Find the root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CALC_SCRIPT = ROOT_DIR / "calc.py"


def test_calc_script_exists():
    """Verify that the calc.py script exists"""
    assert CALC_SCRIPT.exists(), f"Script not found: {CALC_SCRIPT}"


def test_calc_script_is_executable():
    """Verify that the calc.py script is executable"""
    assert os.access(CALC_SCRIPT, os.X_OK), f"Script is not executable: {CALC_SCRIPT}"


def test_example_data_exists():
    """Verify that example data files exist"""
    for filename in ["benzoylecgonine.csv", "estrone.csv"]:
        data_file = DATA_DIR / filename
        assert data_file.exists(), f"Example data file not found: {data_file}"


def run_calc_command(args):
    """Run the calc.py script with the given arguments and return the result"""
    cmd = [sys.executable, str(CALC_SCRIPT)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def test_calc_basic_usage():
    """Test basic usage of calc.py with default settings"""
    # Run the command with example data
    data_file = DATA_DIR / "benzoylecgonine.csv"
    result = run_calc_command([str(data_file)])

    # Check if the command was successful
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"

    # Check if the output contains expected strings
    assert "Results:" in result.stdout
    assert "LLOD" in result.stdout  # Changed from "LLOD:" to "LLOD"
    assert "LLOQ" in result.stdout  # Changed from "LLOQ:" to "LLOQ"
    assert "intercept" in result.stdout
    assert "slope" in result.stdout
    assert "r_squared" in result.stdout


def test_calc_with_different_weight_types():
    """Test calc.py with different weight types"""
    data_file = DATA_DIR / "benzoylecgonine.csv"

    for weight_type in ["none", "1/x", "1/x^2"]:
        result = run_calc_command([str(data_file), "--weight_type", weight_type])

        # Check if the command was successful
        assert result.returncode == 0, f"Command failed with weight_type={weight_type}: {result.stderr}"

        # Check if the output contains expected strings
        assert "Results:" in result.stdout
        assert "LLOD" in result.stdout  # Changed from "LLOD:" to "LLOD"
        assert "LLOQ" in result.stdout  # Changed from "LLOQ:" to "LLOQ"


def test_calc_with_sig_figs():
    """Test calc.py with different significant figures settings"""
    data_file = DATA_DIR / "benzoylecgonine.csv"

    for sig_figs in [1, 3, 6]:
        result = run_calc_command([str(data_file), "--sig_figs", str(sig_figs)])

        # Check if the command was successful
        assert result.returncode == 0, f"Command failed with sig_figs={sig_figs}: {result.stderr}"

        # Check if the output contains expected strings
        assert "Results:" in result.stdout


def test_calc_output_file():
    """Test calc.py with output file option"""
    data_file = DATA_DIR / "benzoylecgonine.csv"

    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
        output_path = temp_file.name

        # Run the command with output file
        result = run_calc_command([str(data_file), "--output", output_path])

        # Check if the command was successful
        assert result.returncode == 0, f"Command failed with output option: {result.stderr}"

        # Check if the output file was created and contains data
        assert os.path.exists(output_path), f"Output file was not created: {output_path}"

        # Check if the file has the expected content
        df = pd.read_csv(output_path)
        assert "Parameter" in df.columns
        assert "Value" in df.columns
        assert len(df) > 0

        # Check if specific parameters are in the output
        parameters = df["Parameter"].tolist()
        assert "LLOD" in parameters
        assert "LLOQ" in parameters
        assert "intercept" in parameters
        assert "slope" in parameters
        assert "r_squared" in parameters


def test_error_handling_missing_file():
    """Test that calc.py properly handles a missing input file"""
    # Use a non-existent file path
    data_file = "nonexistent_file.csv"
    result = run_calc_command([data_file])

    # Check if the command failed with expected error
    assert result.returncode != 0, "Command should have failed with missing file"
    assert "not found" in result.stdout or "not found" in result.stderr


def test_error_handling_invalid_csv():
    """Test that calc.py properly handles a CSV file with invalid format"""
    # Create a temporary CSV file with invalid format (missing required columns)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w") as temp_file:
        temp_file.write("a,b\n1,2\n3,4\n")
        temp_file.flush()

        # Run the command with invalid CSV
        result = run_calc_command([temp_file.name])

        # Check if the command failed with expected error
        assert result.returncode != 0, "Command should have failed with invalid CSV"
        assert "must contain 'x' and 'y' columns" in result.stdout or "must contain 'x' and 'y' columns" in result.stderr
