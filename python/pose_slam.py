import numpy as np

# Simulated 2D pose-graph SLAM problem (poses connected by odometry)
num_poses = 5  # Number of robot poses

# Initial pose estimates (x, y, theta)
poses = np.zeros((num_poses, 3))  # [x, y, theta] for each pose

# Odometry constraints (relative transformations)
odometry_measurements = np.array([
    [1.0, 0.0, 0.1],  # Move 1m forward, slight rotation
    [1.0, 0.0, 0.0],  # Move 1m forward, no rotation
    [1.0, 0.0, -0.1],  # Move 1m forward, slight rotation back
    [1.0, 0.0, 0.2],  # Move 1m forward, larger rotation
])

# Information matrix (inverse of covariance) for each constraint
information_matrices = np.array([
    np.diag([10.0, 10.0, 5.0]),  # High confidence in translation, lower in rotation
    np.diag([10.0, 10.0, 5.0]),
    np.diag([10.0, 10.0, 5.0]),
    np.diag([10.0, 10.0, 5.0]),
])

# LM parameters
lambda_damping = 1.0
lambda_factor = 10
max_iterations = 20
tolerance = 1e-6

def pose_error(xi, xj, zij):
    """ Computes pose error given two poses and a relative measurement. """
    dx = xj[0] - xi[0]
    dy = xj[1] - xi[1]
    dtheta = xj[2] - xi[2]
    
    # Expected transformation from xi to xj
    cos_theta, sin_theta = np.cos(xi[2]), np.sin(xi[2])
    predicted = np.array([
        cos_theta * zij[0] - sin_theta * zij[1] + xi[0],  # x
        sin_theta * zij[0] + cos_theta * zij[1] + xi[1],  # y
        xi[2] + zij[2]  # theta
    ])
    
    error = np.array([dx, dy, dtheta]) - predicted + np.array([xi[0], xi[1], xi[2]])
    return error

def compute_jacobian(xi, xj):
    """ Computes the Jacobian of the pose error function w.r.t xi and xj. """
    cos_theta, sin_theta = np.cos(xi[2]), np.sin(xi[2])
    J_i = np.array([
        [-1,  0, sin_theta * (xj[0] - xi[0]) + cos_theta * (xj[1] - xi[1])],
        [ 0, -1, cos_theta * (xj[0] - xi[0]) - sin_theta * (xj[1] - xi[1])],
        [ 0,  0, -1]
    ])
    J_j = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    return J_i, J_j

# Levenberg-Marquardt Iterations
for iteration in range(max_iterations):
    H = np.zeros((num_poses * 3, num_poses * 3))  # Hessian approx.
    b = np.zeros(num_poses * 3)  # Gradient vector
    total_error = 0.0

    for i in range(num_poses - 1):
        xi = poses[i]
        xj = poses[i + 1]
        zij = odometry_measurements[i]
        omega = information_matrices[i]

        e = pose_error(xi, xj, zij)
        J_i, J_j = compute_jacobian(xi, xj)

        # Block Hessian updates
        H_i = J_i.T @ omega @ J_i
        H_j = J_j.T @ omega @ J_j
        H_ij = J_i.T @ omega @ J_j

        idx_i = i * 3
        idx_j = (i + 1) * 3

        H[idx_i:idx_i+3, idx_i:idx_i+3] += H_i
        H[idx_j:idx_j+3, idx_j:idx_j+3] += H_j
        H[idx_i:idx_i+3, idx_j:idx_j+3] += H_ij
        H[idx_j:idx_j+3, idx_i:idx_i+3] += H_ij.T

        # Gradient update
        b[idx_i:idx_i+3] += J_i.T @ omega @ e
        b[idx_j:idx_j+3] += J_j.T @ omega @ e

        total_error += np.linalg.norm(e)

    # Convergence check
    if np.linalg.norm(b) < tolerance:
        print("Converged!")
        break

    # Solve LM system: (H + λI) Δx = -b
    I = np.eye(num_poses * 3)
    H_damped = H + lambda_damping * I
    dx = np.linalg.solve(H_damped, -b)

    # Update poses
    poses += dx.reshape(num_poses, 3)

    # Adaptive damping adjustment
    if total_error < np.linalg.norm(b):  # Accept step
        lambda_damping /= lambda_factor
    else:  # Reject step
        lambda_damping *= lambda_factor

    print(f"Iteration {iteration+1}: Error = {total_error:.6f}, Lambda = {lambda_damping:.6f}")

print("Optimized Poses:\n", poses)
