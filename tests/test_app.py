"""
Tests for the Streamlit app
"""

from unittest.mock import patch

import pandas as pd

# Import app functions
from app import process_uploaded_data, calculate_all_results, calculate_visualization_data


def test_process_uploaded_data():
    """Test process_uploaded_data function"""
    # Create a test DataFrame
    data = pd.DataFrame({"x": [2, 5, 10, 50, 100, 500], "y": [2.9, 5.1, 8.1, 28.1, 52.5, 124.2]})

    # Mock st.error function to prevent it from raising errors during tests
    with patch("streamlit.error"):
        # Process data
        processed = process_uploaded_data(data)

        # Check that the function returns the data unchanged
        assert processed is not None
        pd.testing.assert_frame_equal(processed, data)

        # Test with missing columns
        bad_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert process_uploaded_data(bad_data) is None

        # Test with non-positive values
        bad_data_2 = pd.DataFrame({"x": [0, 2, 3], "y": [1, 2, 3]})
        assert process_uploaded_data(bad_data_2) is None


def test_calculate_all_results():
    """Test calculate_all_results function"""
    # Create a test DataFrame
    data = pd.DataFrame({"x": [2, 5, 10, 50, 100, 500], "y": [2.9, 5.1, 8.1, 28.1, 52.5, 124.2]})

    # Mock st.error and st.spinner
    with patch("streamlit.error"), patch("streamlit.spinner"):
        # Calculate results
        results, formatted_results = calculate_all_results(data, 3)

        # Check that results are returned
        assert results is not None
        assert formatted_results is not None

        # Check that all weight types are in the results
        for weight_type in ["1/x^2", "1/x", "none"]:
            assert weight_type in results
            assert weight_type in formatted_results

            # Check that each result has the expected keys
            for key in ["intercept", "slope", "LLOD", "LLOQ", "r_squared"]:
                assert key in results[weight_type]
                assert key in formatted_results[weight_type]


def test_calculate_visualization_data():
    """Test calculate_visualization_data function"""
    # Create a test DataFrame
    data = pd.DataFrame({"x": [2, 5, 10, 50, 100, 500], "y": [2.9, 5.1, 8.1, 28.1, 52.5, 124.2]})

    # Mock st.error
    with patch("streamlit.error"), patch("streamlit.spinner"):
        # First calculate results
        results, _ = calculate_all_results(data, 3)

        # Calculate visualization data
        vis_data = calculate_visualization_data(data, results)

        # Check that visualization data is returned
        assert vis_data is not None

        # Check that all expected keys are in the visualization data
        expected_keys = ["point_data", "fit_data", "threshold_data", "x_domain", "y_domain"]
        for key in expected_keys:
            assert key in vis_data

        # Check that the point data has the expected columns
        point_df = vis_data["point_data"]
        assert "x" in point_df.columns
        assert "y" in point_df.columns
        assert "Series" in point_df.columns

        # Check that the fit data has the expected columns
        fit_df = vis_data["fit_data"]
        assert "x" in fit_df.columns
        assert "y" in fit_df.columns
        assert "Series" in fit_df.columns
        assert "Weight_Type" in fit_df.columns
