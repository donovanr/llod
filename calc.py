#!/usr/bin/env python3
"""
Command-line script for calculating LLOD and LLOQ from concentration-response data.
"""

import argparse
import sys
import pandas as pd

# Import functions from src.llodlloq module
try:
    from src.llodlloq import weighted_least_squares, format_with_sig_figs
except ImportError:
    # Fall back for different import scenarios
    try:
        from llodlloq import weighted_least_squares, format_with_sig_figs
    except ImportError:
        from llodlloq.src.llodlloq import weighted_least_squares, format_with_sig_figs


def main():
    """Main function for command-line execution."""
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
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output file path for saving results as CSV",
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

        # Check for non-positive values
        if (data["x"] <= 0).any():
            print("Error: 'x' values must be positive for log transformation")
            sys.exit(1)

        if (data["y"] <= 0).any():
            print("Error: 'y' values must be positive for log transformation")
            sys.exit(1)

        # Check for sufficient data points
        if len(data) < 3:
            print("Error: At least 3 data points are required for regression")
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

        # Display results
        print("\nResults:")
        max_key_length = max(len(key) for key in formatted_out.keys())
        for key, value in formatted_out.items():
            print(f"{key.ljust(max_key_length)}: {value}")

        # Save results to CSV if output path is provided
        if args.output:
            results_df = pd.DataFrame(
                {
                    "Parameter": list(formatted_out.keys()),
                    "Value": list(formatted_out.values()),
                }
            )
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")

    except FileNotFoundError:
        print(f"Error: File '{args.csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
