# LLOD/LLOQ Calculator

A tool to calculate Limit of Detection (LLOD) and Limit of Quantification (LLOQ) from concentration-response data using weighted least squares regression.

## Streamlit App

The easiest way to use this tool is through the web app:

[Open LLOD/LLOQ Calculator App](https://llodlloq.streamlit.app)

The app allows you to:
- Upload your own CSV data or use sample data
- Choose weighting options (none, 1/x, or 1/x²)
- Adjust significant figures in results
- Visualize concentration-response curves with LLOD and LLOQ indicators
- Download results as CSV

## Command Line Usage

For batch processing, you can also use the command line tool.

### Installation

This script uses `uv` for dependency management (if you need to install uv, please follow [these instructions](https://docs.astral.sh/uv/getting-started/installation/)).

Install the LLOD/LLOQ calculator and its dependencies, first clone this repo:

```bash
git clone https://github.com/donovanr/llod.git
cd llod
```

### Usage

You can run the command-line script directly with:

```bash
python calc.py path/to/data.csv
```

Or with uv:

```bash
uv run calc.py path/to/data.csv
```

### Arguments

- `csv_file`: Path to a CSV file containing concentration-response data. The file must contain columns named 'x' (concentration) and 'y' (response).

### Optional Arguments

- `--weight_type`: Type of weighting to apply in the regression:
  - `none`: No weighting
  - `1/x`: Weight by 1/x (inverse of concentration)
  - `1/x^2`: Weight by 1/x² (default)
- `--sig_figs`: Number of significant figures to display in the results (default: 3)
- `--output`: Path to save results as a CSV file (optional)

### Examples

Basic usage:
```bash
uv run python calc.py data/estrone.csv
```

With weighting and significant figures options:
```bash
uv run python calc.py data/estrone.csv --weight_type 1/x --sig_figs 4
```

Save results to a CSV file:
```bash
uv run python calc.py data/estrone.csv --output results.csv
```

## Input Data Format

The CSV file should contain two columns: 'x' and 'y'. For example:

```
x,y
2,2.9
5,5.1
10,8.1
50,28.1
100,52.5
500,124.2
```

## Example Output

```
Results:
intercept: 1.864
slope   : 0.635
LLOD    : 2.115
LLOQ    : 14.09
```

## Running the Streamlit App Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Methodology

This calculator uses weighted least squares regression in log-log space to model the relationship between concentration (x) and response (y). The model follows the power law form:

y = a·x^b

Where:
- a is the intercept
- b is the slope

The LLOD is calculated as the concentration that would produce a response 3 times the background:
LLOD = (3/intercept)^(1/slope)

The LLOQ is calculated as the concentration that would produce a response 10 times the background:
LLOQ = (10/intercept)^(1/slope)
```
