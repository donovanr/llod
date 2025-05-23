# LLOD/LLOQ Calculator

A simple Python script to calculate Limit of Detection (LLOD) and Limit of Quantification (LLOQ) from concentration-response data using weighted least squares regression.

## Installation

This script uses `uv` for dependency management (if you need to install uv, please follow [these instructions](https://docs.astral.sh/uv/getting-started/installation/)).

Install the llod/lloq script and its dependencies, first clone this repo:

```bash
git clone https://github.com/donovanr/llod.git
cd llod
```

and then run directly with:

```bash
uv run llodlloq.py data/estrone.csv
```

## Usage

```bash
uv run python llodlloq.py path/to/data.csv [--weight_type {none,1/x,1/x^2}] [--sig_figs SIG_FIGS]
```

## Arguments

- `csv_file`: Path to a CSV file containing concentration-response data. The file must contain columns named ‘x’ (concentration) and ‘y’ (response).

## Optional Arguments


- `--weight_type`: Type of weighting to apply in the regression:
  - `none`: No weighting
  - `1/x`: Weight by 1/x (inverse of concentration)
  - `1/x^2`: Weight by 1/x² (default)
- `--sig_figs`: Number of significant figures to display in the results (default: 3)

## Input Data Format

The CSV file should contain two columns: ‘x’ and ‘y’. For example:

```
x,y
2,2.9
5,5.1
10,8.1
50,28.1
100,52.5
500,124.2
```

## Output

```bash
$ uv run python llodlloq.py data/estrone.csv --sig_figs 4

Results:
intercept: 1.864
slope: 0.635
LLOD: 2.115
LLOQ: 14.09
```
