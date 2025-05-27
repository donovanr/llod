import numpy as np
from sklearn.linear_model import LinearRegression


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

    # Calculate R-squared for unweighted power fit (standard R^2)
    # First, fit an unweighted model in log-log space
    unweighted_model = LinearRegression()
    unweighted_model.fit(log_X, log_y)

    # Calculate predictions and R^2 in log space
    log_y_pred = unweighted_model.predict(log_X)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    ss_res = np.sum((log_y - log_y_pred) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    output = {
        "intercept": intercept,
        "slope": slope,
        "LLOD": LLOD,
        "LLOQ": LLOQ,
        "r_squared": r_squared,  # Add R-squared value
    }

    return output


def format_with_sig_figs(value, sig_figs):
    """Format a number to specified significant figures"""
    return f"{value:.{sig_figs}g}"
