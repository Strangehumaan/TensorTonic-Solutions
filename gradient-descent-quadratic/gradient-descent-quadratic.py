def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    
    for i in range(steps):
        dx = 2 * a * x0 + b
        x0-= lr * dx
    return x0