import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io

# Import functions from llodlloq module
from src.llodlloq import weighted_least_squares, format_with_sig_figs

# Set page configuration
st.set_page_config(
    page_title="LLOD/LLOQ Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("LLOD/LLOQ Calculator")
st.write(
    """
Upload a CSV file with 'x' and 'y' columns to calculate the Limit of Detection (LLOD)
and Limit of Quantification (LLOQ) using weighted least squares regression.
"""
)

# Initialize session state for recalculation
if "data" not in st.session_state:
    st.session_state.data = None
if "results_calculated" not in st.session_state:
    st.session_state.results_calculated = False

# Sidebar with controls
st.sidebar.header("Settings")

weight_type = st.sidebar.selectbox(
    "Weight Type",
    options=["1/x^2", "1/x", "none"],
    index=0,
    help="Type of weighting to apply in the regression",
)

sig_figs = st.sidebar.slider(
    "Significant Figures",
    min_value=1,
    max_value=6,
    value=3,
    help="Number of significant figures to display in results",
)

# Add recalculate button to sidebar
recalculate = st.sidebar.button(
    "Recalculate", key="recalculate", help="Click to recalculate with current settings"
)

# File upload section
st.subheader("Data Input")

# Two columns for upload options
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload a CSV file with 'x' and 'y' columns", type=["csv"], key="file_uploader"
    )

with col2:
    use_sample = st.checkbox("Use sample data instead", value=False)


def process_data(data):
    """Process the data and calculate LLOD/LLOQ"""
    # Check if x and y columns exist
    if "x" not in data.columns or "y" not in data.columns:
        st.error("Error: CSV file must contain 'x' and 'y' columns")
        return None

    # Check for non-positive values in x
    if (data["x"] <= 0).any():
        st.error("Error: 'x' values must be positive for log transformation")
        return None

    # Check for non-positive values in y
    if (data["y"] <= 0).any():
        st.error("Error: 'y' values must be positive for log transformation")
        return None

    try:
        # Run the analysis
        results = weighted_least_squares(
            data.x.values, data.y.values, weight_type=weight_type
        )

        # Format the output with the specified number of significant figures
        formatted_results = {
            key: format_with_sig_figs(value, sig_figs) for key, value in results.items()
        }

        return data, results, formatted_results

    except Exception as e:
        st.error(f"Error during calculation: {str(e)}")
        return None


def create_visualization_data(data, results):
    """Create data for visualization with distinct series labels"""
    # Create x range for prediction line
    x_range = np.logspace(np.log10(min(data.x)), np.log10(max(data.x)), 100)
    y_pred = results["intercept"] * x_range ** results["slope"]

    # Create dataframe for visualization with unique Series value
    line_data = pd.DataFrame(
        {"x": x_range, "y": y_pred, "Series": "Fitted Curve"}  # Unique series name
    )

    # Create dataframe for original data points with unique Series value
    point_data = pd.DataFrame(
        {"x": data.x, "y": data.y, "Series": "Data Points"}  # Unique series name
    )

    # Combine for full visualization dataset
    viz_data = pd.concat([point_data, line_data])

    # Create LLOD and LLOQ reference lines
    llod = float(results["LLOD"])
    lloq = float(results["LLOQ"])

    # Create reference values for LLOD and LLOQ lines with unique Series values
    llod_data = pd.DataFrame(
        {
            "x": [llod] * 2,
            "y": [min(data.y) * 0.9, max(data.y) * 1.1],
            "Series": "LLOD",  # Unique series name
        }
    )

    lloq_data = pd.DataFrame(
        {
            "x": [lloq] * 2,
            "y": [min(data.y) * 0.9, max(data.y) * 1.1],
            "Series": "LLOQ",  # Unique series name
        }
    )

    # Combine all data
    threshold_data = pd.concat([llod_data, lloq_data])

    return viz_data, threshold_data


def display_results(data, results, formatted_results):
    """Display the results and plots"""
    # Results section
    st.header("Results")

    # Create a dataframe for clean display
    results_df = pd.DataFrame(
        {
            "Parameter": list(formatted_results.keys()),
            "Value": list(formatted_results.values()),
        }
    )

    # Show results in table format
    col1, col2 = st.columns([1, 2])
    with col1:
        st.table(results_df)

    with col2:
        st.write(
            """
        ### Parameters Explained
        - **Intercept**: The y-value when x=1 in the power model (y = intercept * x^slope)
        - **Slope**: The power to which x is raised in the model
        - **LLOD**: Limit of Detection, calculated as (3/intercept)^(1/slope)
        - **LLOQ**: Limit of Quantification, calculated as (10/intercept)^(1/slope)
        """
        )

    # Create visualization data with distinct series
    viz_data, threshold_data = create_visualization_data(data, results)

    # LLOD and LLOQ Visualization
    st.subheader("Concentration-Response with LLOD and LLOQ")

    # Create base visualization for data points and fit line
    points_and_lines = alt.Chart(viz_data).encode(
        x=alt.X("x:Q", scale=alt.Scale(type="log"), title="Concentration"),
        y=alt.Y("y:Q", scale=alt.Scale(type="log"), title="Response"),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(scheme="dark2"),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )

    # Points for data
    points = points_and_lines.mark_circle(size=60).transform_filter(
        alt.datum.Series == "Data Points"
    )

    # Line for the fit
    lines = points_and_lines.mark_line(strokeWidth=2).transform_filter(
        alt.datum.Series == "Fitted Curve"
    )

    # Threshold lines
    threshold_chart = (
        alt.Chart(threshold_data)
        .encode(
            x="x:Q",
            y=alt.Y(
                "y:Q", scale=alt.Scale(domain=[min(data.y) * 0.9, max(data.y) * 1.1])
            ),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(scheme="dark2"),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Series:N", title="Threshold"),
                alt.Tooltip("x:Q", title="Value", format=".3g"),
            ],
        )
        .mark_rule(strokeDash=[4, 4], strokeWidth=2)
    )

    # Combine all charts
    full_chart = (
        alt.layer(points, lines, threshold_chart)
        .resolve_scale(color="shared")  # Use shared color scale for consistent colors
        .properties(width=700, height=400)
    )

    # Display the chart
    st.altair_chart(full_chart, use_container_width=True)

    # Add LLOD/LLOQ values annotation under the chart
    st.markdown(
        f"""
    **LLOD = {formatted_results['LLOD']}** | **LLOQ = {formatted_results['LLOQ']}**
    """
    )

    # Download results as CSV
    results_csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results as CSV",
        data=results_csv,
        file_name="llodlloq_results.csv",
        mime="text/csv",
    )

    # Add methodology explanation
    st.subheader("Methodology")
    st.write(
        f"""
    This calculator uses weighted least squares regression in log-log space to model the relationship
    between concentration (x) and response (y). The model follows the power law form:

    $y = a x^b$

    Where:
    - a is the intercept ({formatted_results['intercept']})
    - b is the slope ({formatted_results['slope']})

    The LLOD is calculated as the concentration that would produce a response 3 times the background:
    $LLOD = (3/intercept)^{{1/slope}}$

    The LLOQ is calculated as the concentration that would produce a response 10 times the background:
    $LLOQ = (10/intercept)^{{1/slope}}$

    The regression was performed using {weight_type} weighting in log-log space.
    """
    )


# Main processing logic
if use_sample:
    sample_data = """x,y
2,2.9
5,5.1
10,8.1
50,28.1
100,52.5
500,124.2"""
    st.session_state.data = pd.read_csv(io.StringIO(sample_data))
    st.success("Using sample data")

elif uploaded_file is not None:
    try:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success(f"File '{uploaded_file.name}' uploaded successfully")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.session_state.data = None

# Display uploaded data
if st.session_state.data is not None:
    st.subheader("Data Preview")
    st.dataframe(st.session_state.data)

    # Process and display results
    # Either recalculate button was pressed or we're calculating for the first time
    if recalculate or not st.session_state.results_calculated:
        result = process_data(st.session_state.data)
        if result:
            data, raw_results, formatted_results = result
            st.session_state.results_calculated = True
            display_results(data, raw_results, formatted_results)
    else:
        # Reuse previous calculation results
        result = process_data(st.session_state.data)
        if result:
            data, raw_results, formatted_results = result
            display_results(data, raw_results, formatted_results)
else:
    st.info("Please upload a CSV file or use the sample data to begin analysis")

# Add footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center">
    <p>Developed for calculation of Limit of Detection (LLOD) and Limit of Quantification (LLOQ) from concentration-response data.</p>
    <p>Source code available on <a href="https://github.com/donovanr/llod" target="_blank">GitHub</a>.</p>
</div>
""",
    unsafe_allow_html=True,
)
