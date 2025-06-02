"""
Test to verify R-squared calculation matches scikit-learn's implementation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.llodlloq import weighted_least_squares


def test_rsquared_calculation_matches_sklearn():
    """Test that our R-squared calculation matches scikit-learn's implementation."""
    # Test data
    data = pd.DataFrame({"x": [2, 5, 10, 50, 100, 500], "y": [2.9, 5.1, 8.1, 28.1, 52.5, 124.2]})

    # Get results from our weighted_least_squares function
    results = weighted_least_squares(data.x.values, data.y.values, weight_type="none")
    r_squared_llodlloq = results["r_squared"]

    # Calculate R-squared using scikit-learn directly
    X = data.x.values.reshape(-1, 1)
    log_X = np.log(X)
    log_y = np.log(data.y.values)

    model = LinearRegression()
    model.fit(log_X, log_y)
    log_y_pred = model.predict(log_X)
    r_squared_sklearn = r2_score(log_y, log_y_pred)

    # Print results for debugging
    print("\nR-squared calculation comparison:")
    print(f"llodlloq method: {r_squared_llodlloq:.10f}")
    print(f"sklearn method: {r_squared_sklearn:.10f}")
    print(f"Difference: {abs(r_squared_llodlloq - r_squared_sklearn):.10e}")

    # Verify the values match within a small tolerance
    tolerance = 1e-10
    assert abs(r_squared_llodlloq - r_squared_sklearn) < tolerance, f"R-squared values don't match: llodlloq={r_squared_llodlloq}, sklearn={r_squared_sklearn}"


def test_rsquared_implementation_directly():
    """Test the R-squared calculation implementation directly."""
    # Test data
    data = pd.DataFrame({"x": [2, 5, 10, 50, 100, 500], "y": [2.9, 5.1, 8.1, 28.1, 52.5, 124.2]})

    # Reshape x for sklearn
    X = data.x.values.reshape(-1, 1)

    # Create log transformed x and y
    log_X = np.log(X)
    log_y = np.log(data.y.values)

    # Create and fit the unweighted linear regression model
    model = LinearRegression()
    model.fit(log_X, log_y)

    # Calculate predictions
    log_y_pred = model.predict(log_X)

    # Method 1: Calculate R-squared using llodlloq.py method
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    ss_res = np.sum((log_y - log_y_pred) ** 2)
    r_squared_llodlloq = 1 - (ss_res / ss_tot)

    # Method 2: Calculate R-squared using scikit-learn's built-in function
    r_squared_sklearn = r2_score(log_y, log_y_pred)

    # Print results for debugging
    print("\nDirect R-squared calculation comparison:")
    print(f"Manual calculation: {r_squared_llodlloq:.10f}")
    print(f"sklearn function: {r_squared_sklearn:.10f}")
    print(f"Difference: {abs(r_squared_llodlloq - r_squared_sklearn):.10e}")

    # Verify the values match within a small tolerance
    tolerance = 1e-10
    assert abs(r_squared_llodlloq - r_squared_sklearn) < tolerance, f"R-squared values don't match: manual={r_squared_llodlloq}, sklearn={r_squared_sklearn}"

    # Print model details for reference
    print("\nModel details:")
    print(f"Slope: {model.coef_[0]:.6f}")
    print(f"Intercept (log space): {model.intercept_:.6f}")
    print(f"Intercept (original space): {np.exp(model.intercept_):.6f}")


def test_different_datasets():
    """Test R-squared calculation on different datasets."""
    # Define a few different test datasets
    datasets = [
        # Original datasets
        {"x": [2, 5, 10, 50, 100, 500], "y": [2.9, 5.1, 8.1, 28.1, 52.5, 124.2]},
        {"x": [0.0034, 0.034, 0.34, 3.4], "y": [15.3, 112, 1513, 10647]},
        # Perfect power law relationship y = 2*x^0.5
        {"x": [1, 4, 9, 16, 25, 36], "y": [2, 4, 6, 8, 10, 12]},
        # Dataset with more noise
        {"x": [1, 2, 5, 10, 20, 50, 100], "y": [1.1, 2.3, 4.7, 10.5, 18.9, 49.2, 102.7]},
    ]

    for i, dataset in enumerate(datasets):
        # Convert to DataFrame
        data = pd.DataFrame(dataset)

        # Get results from our weighted_least_squares function
        results = weighted_least_squares(data.x.values, data.y.values, weight_type="none")
        r_squared_llodlloq = results["r_squared"]

        # Calculate R-squared using scikit-learn directly
        X = data.x.values.reshape(-1, 1)
        log_X = np.log(X)
        log_y = np.log(data.y.values)

        model = LinearRegression()
        model.fit(log_X, log_y)
        log_y_pred = model.predict(log_X)
        r_squared_sklearn = r2_score(log_y, log_y_pred)

        print(f"\nDataset {i+1} R-squared comparison:")
        print(f"llodlloq method: {r_squared_llodlloq:.10f}")
        print(f"sklearn method: {r_squared_sklearn:.10f}")

        # Verify the values match within a small tolerance
        tolerance = 1e-10
        assert abs(r_squared_llodlloq - r_squared_sklearn) < tolerance, f"Dataset {i+1}: R-squared values don't match: " f"llodlloq={r_squared_llodlloq}, sklearn={r_squared_sklearn}"
