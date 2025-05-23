# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import argparse
import sys


def weighted_least_squares(x, y, weight_type="1/x^2"):

    X = x.reshape(-1, 1)

    # create log transformed x and y
    log_X = np.log(X)
    log_y = np.log(y)

    # Calculate weights based on the specified type
    if weight_type == "none":
        weights = None
    elif weight_type == "1/x":
        weights = 1 / x
    elif weight_type == "1/x^2":
        weights = 1 / (x**2)
    else:
        raise ValueError("weight_type must be either '1/x' or '1/x^2'")

    # Create and fit the weighted linear regression model
    model = LinearRegression()
    model.fit(log_X, log_y, sample_weight=weights)

    # Extract key results
    intercept = np.exp(model.intercept_).item()
    slope = model.coef_[0].item()
    LLOD = (3 / intercept) ** (1 / slope)
    LLOQ = (10 / intercept) ** (1 / slope)

    output = {
        "intercept": intercept,
        "slope": slope,
        "LLOD": LLOD,
        "LLOQ": LLOQ,
    }

    return output


def format_with_sig_figs(value, sig_figs):
    """Format a number to specified significant figures"""
    return f"{value:.{sig_figs}g}"


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Calculate LLOD and LLOQ from CSV data"
    )
    parser.add_argument(
        "csv_file", type=str, help="Path to CSV file containing x and y columns"
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="1/x^2",
        choices=["none", "1/x", "1/x^2"],
        help="Weight type for regression (default: 1/x^2)",
    )
    parser.add_argument(
        "--sig_figs",
        type=int,
        default=3,
        help="Number of significant figures to display in results (default: 3)",
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Read the CSV file
        data = pd.read_csv(args.csv_file)

        # Check if x and y columns exist
        if "x" not in data.columns or "y" not in data.columns:
            print("Error: CSV file must contain 'x' and 'y' columns")
            sys.exit(1)

        # Run the analysis
        out = weighted_least_squares(
            data.x.values, data.y.values, weight_type=args.weight_type
        )

        # Format the output with the specified number of significant figures
        formatted_out = {
            key: format_with_sig_figs(value, args.sig_figs)
            for key, value in out.items()
        }

        print("Results:")
        for key, value in formatted_out.items():
            print(f"{key}: {value}")

    except FileNotFoundError:
        print(f"Error: File '{args.csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
