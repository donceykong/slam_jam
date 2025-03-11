import numpy as np
import matplotlib.pyplot as plt

def gauss_newton(f, J, x0, y, tol=1e-6, max_iter=1):
    """
    Gauss-Newton optimization method.
    
    Parameters:
    f  : function - Residual function, should return a vector of residuals.
    J  : function - Jacobian of the residual function.
    x0 : np.array - Initial guess.
    y  : np.array - Observed values.
    tol : float - Convergence tolerance.
    max_iter : int - Maximum number of iterations.
    
    Returns:
    x : np.array - Optimized parameters.
    """
    x = x0.copy()
    for i in range(max_iter):
        r = f(x) - y  # Compute residuals
        Jx = J(x)      # Compute Jacobian
        
        # Solve the normal equations J.T * J * delta_x = -J.T * r
        delta_x, _, _, _ = np.linalg.lstsq(Jx, -r, rcond=None)
        x += delta_x  # Update x
        
        # Check convergence
        if np.linalg.norm(delta_x) < tol:
            print(f"Converged in {i+1} iterations.")
            return x
    
    print("Max iterations reached.")
    return x

# Example usage: General polynomial fitting with Gauss-Newton
def example():
    # Generate synthetic data from a polynomial with noise
    np.random.seed(42)
    x_data = np.linspace(-5, 5, 50)
    true_params = np.array([2.0, -3.0, 1.0, 0.5])  # True coefficients for y = 2x^3 - 3x^2 + 1x + 0.5
    y_true = sum(true_params[i] * x_data**i for i in range(len(true_params)))
    y_noisy = y_true + np.random.normal(scale=2.0, size=y_true.shape)  # Add Gaussian noise
    
    degree = len(true_params) - 1  # Adjust polynomial degree automatically
    
    # Residual function: Difference between model and observed data
    def f(params):
        return sum(params[i] * x_data**i for i in range(len(params)))
    
    # Jacobian function
    def J(params):
        J_mat = np.zeros((len(x_data), len(params)))
        for i in range(len(params)):
            J_mat[:, i] = x_data**i  # Derivative w.r.t each coefficient
        return J_mat
    
    # Initial guess for parameters
    x0 = np.ones(degree + 1)
    result = gauss_newton(f, J, x0, y_noisy)
    print("Optimized parameters:", result)
    
    # Plot results
    plt.scatter(x_data, y_noisy, label="Noisy Data", color='red')
    plt.plot(x_data, y_true, label="True Curve", linestyle="dashed")
    plt.plot(x_data, f(result), label="Fitted Curve", color='blue')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("General Polynomial Fitting with Gauss-Newton Method")
    plt.show()
    
if __name__ == "__main__":
    example()
