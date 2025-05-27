"""
Tests for the core llodlloq functions
"""

import numpy as np
import pytest
from src.llodlloq import weighted_least_squares, format_with_sig_figs


def test_format_with_sig_figs():
    """Test the format_with_sig_figs function"""
    # Test with different significant figures
    assert format_with_sig_figs(123.456789, 3) == "123"
    assert format_with_sig_figs(123.456789, 4) == "123.5"
    assert format_with_sig_figs(0.00123456789, 3) == "0.00123"
    assert format_with_sig_figs(0.00123456789, 1) == "0.001"
    assert format_with_sig_figs(1.23e-5, 2) == "1.2e-05"
    assert format_with_sig_figs(1.23e5, 2) == "1.2e+05"


def test_weighted_least_squares_with_no_weighting():
    """Test weighted_least_squares with no weighting"""
    # Create test data
    x = np.array([1, 2, 5, 10, 20, 50])
    y = np.array([2, 3.5, 8, 15, 30, 75])

    # Calculate with no weighting
    results = weighted_least_squares(x, y, weight_type="none")

    # Check if all expected keys are present
    expected_keys = ["intercept", "slope", "LLOD", "LLOQ", "r_squared"]
    for key in expected_keys:
        assert key in results

    # Intercept should be positive
    assert results["intercept"] > 0

    # R-squared should be between 0 and 1
    assert 0 <= results["r_squared"] <= 1

    # LLOD should be less than LLOQ
    assert results["LLOD"] < results["LLOQ"]


def test_weighted_least_squares_with_1_over_x():
    """Test weighted_least_squares with 1/x weighting"""
    x = np.array([1, 2, 5, 10, 20, 50])
    y = np.array([2, 3.5, 8, 15, 30, 75])

    results = weighted_least_squares(x, y, weight_type="1/x")

    # Check if all expected keys are present
    expected_keys = ["intercept", "slope", "LLOD", "LLOQ", "r_squared"]
    for key in expected_keys:
        assert key in results

    # Intercept should be positive
    assert results["intercept"] > 0

    # R-squared should be between 0 and 1
    assert 0 <= results["r_squared"] <= 1

    # LLOD should be less than LLOQ
    assert results["LLOD"] < results["LLOQ"]


def test_weighted_least_squares_with_1_over_x_squared():
    """Test weighted_least_squares with 1/x^2 weighting"""
    x = np.array([1, 2, 5, 10, 20, 50])
    y = np.array([2, 3.5, 8, 15, 30, 75])

    results = weighted_least_squares(x, y, weight_type="1/x^2")

    # Check if all expected keys are present
    expected_keys = ["intercept", "slope", "LLOD", "LLOQ", "r_squared"]
    for key in expected_keys:
        assert key in results

    # Intercept should be positive
    assert results["intercept"] > 0

    # R-squared should be between 0 and 1
    assert 0 <= results["r_squared"] <= 1

    # LLOD should be less than LLOQ
    assert results["LLOD"] < results["LLOQ"]


def test_weighted_least_squares_invalid_weight_type():
    """Test weighted_least_squares with invalid weight type"""
    x = np.array([1, 2, 5, 10, 20, 50])
    y = np.array([2, 3.5, 8, 15, 30, 75])

    # Should raise ValueError for invalid weight type
    with pytest.raises(ValueError):
        weighted_least_squares(x, y, weight_type="invalid")


def test_weighted_least_squares_perfect_linear():
    """Test weighted_least_squares with perfect linear data"""
    # Create perfect power law data y = 2*x^1
    x = np.array([1, 2, 5, 10, 20, 50])
    y = 2 * x**1  # Perfect power relationship

    # Calculate with no weighting
    results = weighted_least_squares(x, y, weight_type="none")

    # For perfect data, r_squared should be very close to 1
    assert results["r_squared"] > 0.999

    # Slope should be close to 1
    assert 0.99 < results["slope"] < 1.01

    # Intercept should be close to 2
    assert 1.99 < results["intercept"] < 2.01
