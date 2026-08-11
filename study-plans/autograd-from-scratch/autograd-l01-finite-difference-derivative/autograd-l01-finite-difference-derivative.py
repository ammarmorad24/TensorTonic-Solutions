import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    
    # Create an array of powers: [0, 1, 2, ..., n-1]
    powers = np.arange(len(coefficients))
    
    # Calculate f(x) by multiplying coefficients by x^powers and summing
    f_x = np.sum(coefficients * (x ** powers))
    
    # Calculate f(x + h)
    f_xh = np.sum(coefficients * ((x + h) ** powers))
    
    # Calculate the forward-difference slope
    slope = (f_xh - f_x) / h
    
    return f_x, f_xh, slope

