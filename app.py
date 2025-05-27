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

# Initialize session state for application state management
if "data" not in st.session_state:
    st.session_state.data = None
if "data_points_chart" not in st.session_state:
    st.session_state.data_points_chart = None
if "x_domain" not in st.session_state:
    st.session_state.x_domain = None
if "y_domain" not in st.session_state:
    st.session_state.y_domain = None
if "last_weight_type" not in st.session_state:
    st.session_state.last_weight_type = None
if "calculated_results" not in st.session_state:
    st.session_state.calculated_results = {}

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


# Cache data processing function to avoid recalculation
@st.cache_data
def process_uploaded_data(data):
    """
    Process the uploaded data once and cache the result.
    This doesn't depend on the weighting type, so it can be cached.
    """
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

    return data


def calculate_results(data, weight_type):
    """Calculate results based on data and weight type"""
    try:
        # Run the analysis
        results = weighted_least_squares(
            data.x.values, data.y.values, weight_type=weight_type
        )

        # Format the output with the specified number of significant figures
        formatted_results = {
            key: format_with_sig_figs(value, sig_figs) for key, value in results.items()
        }

        return results, formatted_results
    except Exception as e:
        st.error(f"Error during calculation: {str(e)}")
        return None, None


# Cache the creation of the base data visualization
@st.cache_data
def create_base_visualization(data):
    """
    Create and cache the base data visualization (just the data points).
    This doesn't depend on weighting type, so it can be cached.
    """
    try:
        # Determine safe x and y domains for log scaling
        x_min = max(1e-10, min(data.x))
        x_max = max(data.x)
        y_min = max(1e-10, min(data.y))
        y_max = max(data.y)

        # Add padding in log space
        x_padding = 0.3  # 30% padding
        y_padding = 0.3

        log_x_min = np.log10(x_min) - x_padding
        log_x_max = np.log10(x_max) + x_padding
        log_y_min = np.log10(y_min) - y_padding
        log_y_max = np.log10(y_max) + y_padding

        # Convert back to linear space with padding
        x_min_padded = 10 ** log_x_min
        x_max_padded = 10 ** log_x_max
        y_min_padded = 10 ** log_y_min
        y_max_padded = 10 ** log_y_max

        # Save domains for reuse
        x_domain = [x_min_padded, x_max_padded]
        y_domain = [y_min_padded, y_max_padded]

        # Create dataframe for data points
        point_data = pd.DataFrame({
            "x": data.x,
            "y": data.y,
            "Series": "Data Points"
        })

        # Create base chart with only data points
        base_chart = alt.Chart(point_data).encode(
            x=alt.X("x:Q",
                   scale=alt.Scale(type="log", domain=x_domain),
                   title="Concentration"),
            y=alt.Y("y:Q",
                   scale=alt.Scale(type="log", domain=y_domain),
                   title="Response"),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(scheme="dark2"),
                legend=alt.Legend(title=None, orient="top"),
            ),
        ).mark_circle(size=60)

        return base_chart, x_domain, y_domain

    except Exception as e:
        st.error(f"Error creating base visualization: {str(e)}")
        return None, None, None


def create_model_visualization(results, x_domain, y_domain):
    """
    Create visualization elements for the model fit and thresholds.
    This depends on the weighting type and changes when it changes.
    """
    try:
        # Create x range for prediction line using the same domain as the base chart
        x_range = np.logspace(np.log10(x_domain[0]), np.log10(x_domain[1]), 100)

        # Calculate predicted y values
        y_pred = results["intercept"] * x_range ** results["slope"]

        # Create dataframe for fit line
        line_data = pd.DataFrame({
            "x": x_range,
            "y": y_pred,
            "Series": "Fitted Curve"
        })

        # Create fit line chart
        fit_line = alt.Chart(line_data).encode(
            x=alt.X("x:Q", scale=alt.Scale(type="log", domain=x_domain)),
            y=alt.Y("y:Q", scale=alt.Scale(type="log", domain=y_domain)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(scheme="dark2"),
                legend=alt.Legend(title=None, orient="top"),
            ),
        ).mark_line(strokeWidth=2)

        # Create LLOD and LLOQ reference lines
        llod = float(results["LLOD"])
        lloq = float(results["LLOQ"])

        # Create vertical lines for LLOD and LLOQ
        llod_data = pd.DataFrame({
            "x": [llod] * 2,
            "y": y_domain,  # Use the full y domain
            "Series": "LLOD"
        })

        lloq_data = pd.DataFrame({
            "x": [lloq] * 2,
            "y": y_domain,  # Use the full y domain
            "Series": "LLOQ"
        })

        # Combine threshold data
        threshold_data = pd.concat([llod_data, lloq_data])

        # Create threshold chart
        threshold_chart = alt.Chart(threshold_data).encode(
            x=alt.X("x:Q", scale=alt.Scale(type="log", domain=x_domain)),
            y=alt.Y("y:Q", scale=alt.Scale(type="log", domain=y_domain)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(scheme="dark2"),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Series:N", title="Threshold"),
                alt.Tooltip("x:Q", title="Value", format=".3g"),
            ],
        ).mark_rule(strokeDash=[4, 4], strokeWidth=2)

        return fit_line, threshold_chart

    except Exception as e:
        st.error(f"Error creating model visualization: {str(e)}")
        return None, None


def display_results(data, raw_results, formatted_results, weight_type):
    """Display the results and combined visualization"""
    # Results section
    st.header("Results")

    # Create a dataframe for clean display
    results_df = pd.DataFrame({
        "Parameter": list(formatted_results.keys()),
        "Value": list(formatted_results.values()),
    })

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
        - **R²**: Coefficient of determination for the unweighted power fit
        """
        )

    # Visualization section
    st.subheader("Concentration-Response with LLOD and LLOQ")

    try:
        # Check if we already have a base chart cached in session state
        if st.session_state.data_points_chart is None:
            # Create and cache the base chart
            base_chart, x_domain, y_domain = create_base_visualization(data)
            if base_chart is not None:
                st.session_state.data_points_chart = base_chart
                st.session_state.x_domain = x_domain
                st.session_state.y_domain = y_domain

        # Get the model-specific visualization components
        if st.session_state.data_points_chart is not None:
            fit_line, threshold_chart = create_model_visualization(
                raw_results,
                st.session_state.x_domain,
                st.session_state.y_domain
            )

            if fit_line is not None and threshold_chart is not None:
                # Combine all charts
                full_chart = (
                    alt.layer(st.session_state.data_points_chart, fit_line, threshold_chart)
                    .resolve_scale(color="shared")
                    .properties(width=700, height=400)
                )

                # Display the chart
                st.altair_chart(full_chart, use_container_width=True)

                # Add LLOD/LLOQ values annotation under the chart
                st.markdown(
                    f"""
                **LLOD = {formatted_results['LLOD']}** | **LLOQ = {formatted_results['LLOQ']}** | **R² = {formatted_results['r_squared']}**
                """
                )
            else:
                st.warning("Could not create model visualization elements.")
        else:
            st.warning("Could not create base visualization. Please check your data.")

    except Exception as e:
        st.error(f"Error rendering visualization: {str(e)}")
        st.info("The calculation was successful, but the visualization could not be rendered. You can still see the numeric results above.")

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
    data = pd.read_csv(io.StringIO(sample_data))
    st.session_state.data = data
    st.success("Using sample data")

elif uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.success(f"File '{uploaded_file.name}' uploaded successfully")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.session_state.data = None

# Display uploaded data
if st.session_state.data is not None:
    # Process the data once and cache the result
    processed_data = process_uploaded_data(st.session_state.data)

    if processed_data is not None:
        st.subheader("Data Preview")
        st.dataframe(processed_data)

        # Check if we need to recalculate based on weight type change
        weight_changed = st.session_state.last_weight_type != weight_type

        # Calculate or retrieve results
        if recalculate or weight_changed or weight_type not in st.session_state.calculated_results:
            # Need to calculate new results
            raw_results, formatted_results = calculate_results(processed_data, weight_type)

            if raw_results is not None:
                # Store results for this weight type
                st.session_state.calculated_results[weight_type] = (raw_results, formatted_results)
                st.session_state.last_weight_type = weight_type

                # Display results and visualization
                display_results(processed_data, raw_results, formatted_results, weight_type)
        else:
            # Reuse previously calculated results for this weight type
            if weight_type in st.session_state.calculated_results:
                raw_results, formatted_results = st.session_state.calculated_results[weight_type]
                display_results(processed_data, raw_results, formatted_results, weight_type)
            else:
                # Should not happen, but just in case
                raw_results, formatted_results = calculate_results(processed_data, weight_type)
                if raw_results is not None:
                    st.session_state.calculated_results[weight_type] = (raw_results, formatted_results)
                    st.session_state.last_weight_type = weight_type
                    display_results(processed_data, raw_results, formatted_results, weight_type)
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
