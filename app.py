import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io

# Import functions from llodlloq module
from src.llodlloq import weighted_least_squares, format_with_sig_figs

# Set page configuration - change to centered layout
st.set_page_config(
    page_title="LLOD/LLOQ Calculator",
    page_icon="📊",
    layout="centered",  # Change from 'wide' to 'centered'
)

# Add custom CSS to control the max width of the content and improve spacing
st.markdown("""
<style>
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    section[data-testid="stSidebar"] {
        width: 0px !important;
    }
    h1, h2, h3 {
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    .element-container {
        margin-bottom: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "need_recalculation" not in st.session_state:
    st.session_state.need_recalculation = True
if "last_sig_figs" not in st.session_state:
    st.session_state.last_sig_figs = 3
if "results" not in st.session_state:
    st.session_state.results = None
if "formatted_results" not in st.session_state:
    st.session_state.formatted_results = None
if "visualization_data" not in st.session_state:
    st.session_state.visualization_data = None

# App title and description
st.title("LLOD/LLOQ Calculator")
st.write(
    """
Upload a CSV file with 'x' and 'y' columns to calculate the Limit of Detection (LLOD)
and Limit of Quantification (LLOQ) using weighted least squares regression.
"""
)

# 1. DATA INPUT SECTION
st.header("Data Input")

# Example data for download
example_data = """x,y
2,2.9
5,5.1
10,8.1
50,28.1
100,52.5
500,124.2"""

# File upload row
col1, col2 = st.columns([2, 1])

with col1:
    st.write("Upload your CSV file with 'x' and 'y' columns:")
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="file_uploader",
        help="CSV file must contain 'x' and 'y' columns with positive values"
    )

with col2:
    st.write("Or try with example data:")
    # Create a download button for example data
    example_data_bytes = example_data.encode('utf-8')
    st.download_button(
        label="Download Example Data",
        data=example_data_bytes,
        file_name="example_data.csv",
        mime="text/csv",
        help="Download example data to try with the calculator"
    )

    # Recalculate button
    recalculate = st.button(
        "Recalculate",
        key="recalculate",
        help="Click to recalculate with current settings"
    )
    if recalculate:
        st.session_state.need_recalculation = True

# Handle file upload
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Check if this is a new file (different from what's in session state)
        if st.session_state.file_name != uploaded_file.name:
            st.session_state.data = data
            st.session_state.file_name = uploaded_file.name
            st.session_state.need_recalculation = True
            st.session_state.visualization_data = None  # Reset visualization data
            st.success(f"File '{uploaded_file.name}' uploaded successfully")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.session_state.data = None
        st.session_state.file_name = None
        st.session_state.visualization_data = None


# Cache data processing function to avoid recalculation
@st.cache_data(show_spinner=False)
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


def calculate_all_results(data, sig_figs):
    """Calculate results for all weighting types"""
    with st.spinner("Calculating results for all weighting types..."):
        try:
            results = {}
            formatted_results = {}

            # Calculate results for each weighting type
            for weight_type in ["1/x^2", "1/x", "none"]:
                # Run the analysis
                weight_results = weighted_least_squares(
                    data.x.values, data.y.values, weight_type=weight_type
                )

                # Format the output with the specified number of significant figures
                weight_formatted = {
                    key: format_with_sig_figs(value, sig_figs) for key, value in weight_results.items()
                }

                results[weight_type] = weight_results
                formatted_results[weight_type] = weight_formatted

            return results, formatted_results
        except Exception as e:
            st.error(f"Error during calculation: {str(e)}")
            return None, None


def calculate_visualization_data(data, results):
    """
    Create visualization data for all weighting types in a single dataframe
    """
    try:
        # First determine the x and y domains that include all data and thresholds
        all_x_values = list(data.x)
        all_y_values = list(data.y)

        # Calculate maximum ranges across all weighting types
        all_llod_values = []
        all_lloq_values = []

        # For each weighting type
        for weight_type, result in results.items():
            intercept = result["intercept"]
            slope = result["slope"]
            llod = float(result["LLOD"])
            lloq = float(result["LLOQ"])

            # Add LLOD and LLOQ values to lists
            all_llod_values.append(llod)
            all_lloq_values.append(lloq)

            # Calculate y values at LLOD and LLOQ
            y_at_llod = intercept * (llod ** slope)
            y_at_lloq = intercept * (lloq ** slope)
            all_y_values.extend([y_at_llod, y_at_lloq])

        # Add all threshold values to x values
        all_x_values.extend(all_llod_values + all_lloq_values)

        # Calculate safe domain with padding
        x_min = max(1e-10, min(all_x_values))
        x_max = max(all_x_values)
        y_min = max(1e-10, min(all_y_values))
        y_max = max(all_y_values)

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

        # Create x range for prediction lines (100 points from min to max)
        x_range = np.logspace(np.log10(x_min_padded), np.log10(x_max_padded), 100)

        # Create domain limits
        x_domain = [x_min_padded, x_max_padded]
        y_domain = [y_min_padded, y_max_padded]

        # Now create visualization data for all weighting types

        # Data points (same for all weighting types)
        point_data = pd.DataFrame({
            "x": data.x,
            "y": data.y,
            "Series": "Data Points",
            "Weight_Type": "All"  # Same points shown for all weight types
        })

        # Create fit lines for each weighting type
        fit_data_frames = []
        threshold_data_frames = []

        for weight_type, result in results.items():
            intercept = result["intercept"]
            slope = result["slope"]
            llod = float(result["LLOD"])
            lloq = float(result["LLOQ"])

            # Calculate y values for the fit line
            y_pred = intercept * x_range ** slope

            # Create fit line dataframe
            fit_df = pd.DataFrame({
                "x": x_range,
                "y": y_pred,
                "Series": "Fitted Curve",
                "Weight_Type": weight_type
            })

            # Create LLOD and LLOQ threshold dataframes
            llod_df = pd.DataFrame({
                "x": [llod] * 2,
                "y": [y_min_padded, y_max_padded],
                "Series": "LLOD",
                "Weight_Type": weight_type
            })

            lloq_df = pd.DataFrame({
                "x": [lloq] * 2,
                "y": [y_min_padded, y_max_padded],
                "Series": "LLOQ",
                "Weight_Type": weight_type
            })

            fit_data_frames.append(fit_df)
            threshold_data_frames.append(pd.concat([llod_df, lloq_df]))

        # Combine all dataframes
        fit_data = pd.concat(fit_data_frames)
        threshold_data = pd.concat(threshold_data_frames)

        # Return all visualization data
        return {
            "point_data": point_data,
            "fit_data": fit_data,
            "threshold_data": threshold_data,
            "x_domain": x_domain,
            "y_domain": y_domain
        }

    except Exception as e:
        st.error(f"Error creating visualization data: {str(e)}")
        return None


def create_visualization(vis_data, selected_weight):
    """
    Create visualization for the selected weight type
    """
    try:
        # Extract data from the visualization data dictionary
        point_data = vis_data["point_data"]
        fit_data = vis_data["fit_data"]
        threshold_data = vis_data["threshold_data"]
        x_domain = vis_data["x_domain"]
        y_domain = vis_data["y_domain"]

        # Base chart for data points (always visible)
        points_chart = alt.Chart(point_data).encode(
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

        # Filter the fit data for the selected weight type
        selected_fit_data = fit_data[fit_data['Weight_Type'] == selected_weight]
        selected_threshold_data = threshold_data[threshold_data['Weight_Type'] == selected_weight]

        # Chart for fit line
        fit_chart = alt.Chart(selected_fit_data).encode(
            x=alt.X("x:Q", scale=alt.Scale(type="log", domain=x_domain)),
            y=alt.Y("y:Q", scale=alt.Scale(type="log", domain=y_domain)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(scheme="dark2"),
                legend=alt.Legend(title=None, orient="top"),
            ),
        ).mark_line(strokeWidth=2)

        # Chart for threshold lines
        threshold_chart = alt.Chart(selected_threshold_data).encode(
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

        # Combine all charts
        combined_chart = alt.layer(
            points_chart,
            fit_chart,
            threshold_chart
        ).resolve_scale(
            color='shared'
        ).properties(
            width=600,  # Adjusted chart width to fit the narrower layout
            height=400
        )

        return combined_chart

    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None


# Display uploaded data and run calculations
if st.session_state.data is not None:
    # Process the data once and cache the result
    processed_data = process_uploaded_data(st.session_state.data)

    if processed_data is not None:
        # 2. DATA PREVIEW SECTION
        st.header("Data Preview")
        st.dataframe(processed_data, height=200)

        # Calculate results if needed
        if st.session_state.need_recalculation or st.session_state.visualization_data is None:
            # Calculate results for all weighting types
            results, formatted_results = calculate_all_results(processed_data, st.session_state.last_sig_figs)

            if results:
                # Store results in session state
                st.session_state.results = results
                st.session_state.formatted_results = formatted_results

                # Calculate visualization data
                vis_data = calculate_visualization_data(processed_data, results)
                if vis_data:
                    st.session_state.visualization_data = vis_data
                    st.session_state.need_recalculation = False

        # 3. MAIN PLOT SECTION
        if st.session_state.visualization_data and st.session_state.formatted_results:
            # Controls for weight type and sig figs
            st.header("Concentration-Response with LLOD and LLOQ")

            # 4. WEIGHT TYPE AND SIG FIGS CONTROLS (right under the plot)
            col1, col2 = st.columns([1, 1])

            with col1:
                selected_weight = st.selectbox(
                    "Select Weight Type",
                    options=["1/x^2", "1/x", "none"],
                    index=0
                )

            with col2:
                sig_figs = st.slider(
                    "Significant Figures",
                    min_value=1,
                    max_value=6,
                    value=st.session_state.last_sig_figs,
                    help="Number of significant figures to display in results"
                )

            # Check if sig_figs has changed
            if sig_figs != st.session_state.last_sig_figs:
                # Recalculate with new sig_figs
                results, formatted_results = calculate_all_results(processed_data, sig_figs)
                if results:
                    st.session_state.results = results
                    st.session_state.formatted_results = formatted_results
                    st.session_state.last_sig_figs = sig_figs
            else:
                # Use the existing results
                formatted_results = st.session_state.formatted_results

            # Create chart for selected weight type
            chart = create_visualization(st.session_state.visualization_data, selected_weight)

            if chart:
                # Display the chart
                st.altair_chart(chart, use_container_width=True)

                # Get the results for the selected weight type
                weight_results = formatted_results[selected_weight]

                # Add LLOD/LLOQ values annotation under the chart
                st.markdown(
                    f"""
                **LLOD = {weight_results['LLOD']}** | **LLOQ = {weight_results['LLOQ']}** | **R² = {weight_results['r_squared']}**
                """
                )

                # 5. RESULTS SECTION
                st.header("Results")

                # Create a dataframe for display
                results_df = pd.DataFrame({
                    "Parameter": list(weight_results.keys()),
                    "Value": list(weight_results.values()),
                })

                # Show results table
                st.table(results_df)

                # Create download button for results
                results_csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download Results as CSV",
                    data=results_csv,
                    file_name=f"llodlloq_results_{selected_weight.replace('/', '_')}.csv",
                    mime="text/csv",
                )

                # 6. PARAMETERS EXPLAINED SECTION
                st.header("Parameters Explained")
                st.write(
                    """
                - **Intercept**: The y-value when x=1 in the power model (y = intercept * x^slope)
                - **Slope**: The power to which x is raised in the model
                - **LLOD**: Limit of Detection, calculated as (3/intercept)^(1/slope)
                - **LLOQ**: Limit of Quantification, calculated as (10/intercept)^(1/slope)
                - **R²**: Coefficient of determination for the unweighted power fit
                """
                )

                # 7. METHODOLOGY SECTION
                st.header("Methodology")
                st.write(
                    f"""
                This calculator uses weighted least squares regression in log-log space to model the relationship
                between concentration (x) and response (y). The model follows the power law form:

                $y = a x^b$

                Where:
                - a is the intercept ({weight_results['intercept']})
                - b is the slope ({weight_results['slope']})

                The LLOD is calculated as the concentration that would produce a response 3 times the background:
                $LLOD = (3/intercept)^{{1/slope}}$

                The LLOQ is calculated as the concentration that would produce a response 10 times the background:
                $LLOQ = (10/intercept)^{{1/slope}}$

                The regression was performed using {selected_weight} weighting in log-log space.
                """
                )
else:
    st.info("Please upload a CSV file to begin analysis. You can use the example data provided.")

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
