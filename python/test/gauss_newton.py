import numpy as np
import matplotlib.pyplot as plt

def gauss_newton(f, J, x0, y, tol=1e-6, max_iter=10):
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

# Example usage: Curve fitting with Gauss-Newton
def example():
    # Generate synthetic data from y = a * x^2 + b * x + c with noise
    np.random.seed(42)
    x_data = np.linspace(-5, 5, 50)
    true_params = np.array([2.0, -3.0, 1.0])  # True a, b, c
    y_true = true_params[0] * x_data**2 + true_params[1] * x_data + true_params[2]
    y_noisy = y_true + np.random.normal(scale=5.0, size=y_true.shape)  # Add Gaussian noise
    
    # Residual function: Difference between model and observed data
    def f(params):
        a, b, c = params
        return a * x_data**2 + b * x_data + c
    
    # Jacobian function
    def J(params):
        a, b, c = params
        J_mat = np.zeros((len(x_data), 3))
        J_mat[:, 0] = x_data**2  # Derivative w.r.t a
        J_mat[:, 1] = x_data     # Derivative w.r.t b
        J_mat[:, 2] = 1          # Derivative w.r.t c
        return J_mat
    
    # Initial guess for parameters
    x0 = np.array([1.0, 1.0, 1.0])
    result = gauss_newton(f, J, x0, y_noisy)
    print("Optimized parameters:", result)
    
    # Plot results
    plt.scatter(x_data, y_noisy, label="Noisy Data", color='red')
    plt.plot(x_data, y_true, label="True Curve", linestyle="dashed")
    plt.plot(x_data, f(result), label="Fitted Curve", color='blue')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Curve Fitting with Gauss-Newton Method")
    plt.show()
    
if __name__ == "__main__":
    example()