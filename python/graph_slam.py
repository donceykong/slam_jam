# import numpy as np

# # Information matrix (Omega)
# omega = np.array([
#     [3, -1,  0, -1],
#     [-1, 3, -1, -1],
#     [0, -1,  1,  0],
#     [-1, -1,  0,  2],
# ])

# # Information vector (Xi)
# xi = np.array([-13, 4, -2, 11])

# # Solve for x* using a robust method
# try:
#     result = np.linalg.solve(omega, xi)
#     print(f"result:\n{result}")
# except np.linalg.LinAlgError:
#     result = "Matrix is singular and cannot be solved."

# result



'''
Example using Levenburg-Marquadt iterative optimization
'''

import numpy as np

# Simulated landmark positions (Initial estimates)
x = np.array([0.0, 0.0, 0.0, 0.0])  # 4 landmark positions (to be optimized)

# Observations (measurement constraints)
measurements = np.array([-13, 4, -2, 11])  # Simulated observations

# Information matrix (Hessian approximation)
omega = np.array([
    [3, -1,  0, -1],
    [-1, 3, -1, -1],
    [0, -1,  1,  0],
    [-1, -1,  0,  2],
])

# LM parameters
lambda_damping = 0.1    # Initial damping factor
lambda_factor = 5       # Factor to increase/decrease lambda
max_iterations = 200
tolerance = 1e-9

def error_function(x):
    """ Computes the residual error vector e = Omega * x - measurements """
    return omega @ x - measurements

def jacobian():
    """ Returns the Jacobian matrix J (which is just Omega in this case) """
    return omega  # In landmark-based problems, Jacobian is often constant

# Levenberg-Marquardt Iterations
for iteration in range(max_iterations):
    e = error_function(x)  # Compute error
    J = jacobian()  # Compute Jacobian
    H = J.T @ J  # Approximate Hessian
    g = J.T @ e  # Gradient

    # Check for convergence
    if np.linalg.norm(g) < tolerance:
        print("Converged!")
        break

    # Solve LM equation: (H + lambda * I) dx = -g
    I = np.eye(len(x))  # Identity matrix
    H_damped = H + lambda_damping * I
    dx = np.linalg.solve(H_damped, -g)

    # Update step
    new_x = x + dx
    new_error = np.linalg.norm(error_function(new_x))
    old_error = np.linalg.norm(error_function(x))

    # Adaptive damping update
    if new_error < old_error:  # Accept update, reduce damping
        x = new_x
        lambda_damping /= lambda_factor
    else:  # Reject update, increase damping
        lambda_damping *= lambda_factor

    print(f"Iteration {iteration+1}: Error = {old_error:.6f}, Lambda = {lambda_damping:.6f}")

print("Optimized Landmark Positions:", x)

