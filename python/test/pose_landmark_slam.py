import numpy as np

# Simulated 2D Pose-Graph SLAM with Landmarks
num_poses = 5   # Number of robot poses
num_landmarks = 3  # Number of landmarks

# Initialize poses (x, y, theta)
poses = np.zeros((num_poses, 3))  # Robot poses (to be optimized)

# Initialize landmarks (x, y)
landmarks = np.array([
    [2.0, 1.0],  # Landmark 1
    [3.5, -1.0],  # Landmark 2
    [5.0, 0.5],  # Landmark 3
])  # Landmarks (to be optimized)

# Odometry constraints (relative pose transformations)
odometry_measurements = np.array([
    [1.0, 0.0, 0.1],  # Move 1m forward, slight rotation
    [1.0, 0.0, 0.0],  # Move 1m forward, no rotation
    [1.0, 0.0, -0.1],  # Move 1m forward, slight rotation back
    [1.0, 0.0, 0.2],  # Move 1m forward, larger rotation
])

# Landmark observations (each tuple is: [pose_idx, landmark_idx, z_x, z_y])
landmark_observations = np.array([
    [0, 0, 2.0, 1.0],  # Pose 0 sees Landmark 0
    [1, 0, 1.0, 1.0],  # Pose 1 sees Landmark 0
    [2, 1, 2.5, -1.0],  # Pose 2 sees Landmark 1
    [3, 1, 2.0, -1.0],  # Pose 3 sees Landmark 1
    [3, 2, 2.5, 0.5],  # Pose 3 sees Landmark 2
    [4, 2, 1.5, 0.5],  # Pose 4 sees Landmark 2
])

# Information matrices (confidence in measurements)
pose_information = np.diag([10.0, 10.0, 5.0])  # Pose constraints
landmark_information = np.diag([20.0, 20.0])  # Stronger landmark constraints

# LM Parameters
lambda_damping = 1.0
lambda_factor = 10
max_iterations = 20
tolerance = 1e-6

def pose_error(xi, xj, zij):
    """Computes pose-pose error based on odometry."""
    dx = xj[0] - xi[0]
    dy = xj[1] - xi[1]
    dtheta = xj[2] - xi[2]

    cos_theta, sin_theta = np.cos(xi[2]), np.sin(xi[2])
    predicted = np.array([
        cos_theta * zij[0] - sin_theta * zij[1] + xi[0],
        sin_theta * zij[0] + cos_theta * zij[1] + xi[1],
        xi[2] + zij[2]
    ])

    return np.array([dx, dy, dtheta]) - predicted + np.array([xi[0], xi[1], xi[2]])

def landmark_error(xi, lj, zij):
    """Computes pose-landmark error based on landmark observation."""
    cos_theta, sin_theta = np.cos(xi[2]), np.sin(xi[2])
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    predicted = R @ zij + xi[:2]  # Transform landmark measurement to world frame
    error = lj - predicted
    return error

def compute_jacobian_pose(xi, xj):
    """Computes Jacobian for pose-pose constraints."""
    cos_theta, sin_theta = np.cos(xi[2]), np.sin(xi[2])
    J_i = np.array([
        [-1,  0, sin_theta * (xj[0] - xi[0]) + cos_theta * (xj[1] - xi[1])],
        [ 0, -1, cos_theta * (xj[0] - xi[0]) - sin_theta * (xj[1] - xi[1])],
        [ 0,  0, -1]
    ])
    J_j = np.eye(3)
    return J_i, J_j

def compute_jacobian_landmark(xi):
    """Computes Jacobian for pose-landmark constraints."""
    cos_theta, sin_theta = np.cos(xi[2]), np.sin(xi[2])
    J_pose = np.array([
        [-cos_theta, -sin_theta, sin_theta * xi[0] - cos_theta * xi[1]],
        [sin_theta, -cos_theta, -cos_theta * xi[0] - sin_theta * xi[1]]
    ])
    J_landmark = np.eye(2)
    return J_pose, J_landmark

# Levenberg-Marquardt Iterations
for iteration in range(max_iterations):
    H = np.zeros((num_poses * 3 + num_landmarks * 2, num_poses * 3 + num_landmarks * 2))
    b = np.zeros(num_poses * 3 + num_landmarks * 2)
    total_error = 0.0

    # Pose Constraints
    for i in range(num_poses - 1):
        xi = poses[i]
        xj = poses[i + 1]
        zij = odometry_measurements[i]

        e = pose_error(xi, xj, zij)
        J_i, J_j = compute_jacobian_pose(xi, xj)

        idx_i, idx_j = i * 3, (i + 1) * 3

        H[idx_i:idx_i+3, idx_i:idx_i+3] += J_i.T @ pose_information @ J_i
        H[idx_j:idx_j+3, idx_j:idx_j+3] += J_j.T @ pose_information @ J_j
        H[idx_i:idx_i+3, idx_j:idx_j+3] += J_i.T @ pose_information @ J_j
        H[idx_j:idx_j+3, idx_i:idx_i+3] += J_j.T @ pose_information @ J_i.T

        b[idx_i:idx_i+3] += J_i.T @ pose_information @ e
        b[idx_j:idx_j+3] += J_j.T @ pose_information @ e

        total_error += np.linalg.norm(e)

    # Landmark Constraints
    for obs in landmark_observations:
        # pose_idx, landmark_idx, z_x, z_y = map(int, obs[:2])  # Convert indices to integers
        pose_idx, landmark_idx, z_x, z_y = obs
        print(pose_idx)
        
        xi = poses[pose_idx]
        lj = landmarks[landmark_idx]
        zij = np.array([z_x, z_y])

        e = landmark_error(xi, lj, zij)
        J_pose, J_landmark = compute_jacobian_landmark(xi)

        idx_pose, idx_landmark = pose_idx * 3, num_poses * 3 + landmark_idx * 2

        H[idx_pose:idx_pose+3, idx_pose+3] += J_pose.T @ landmark_information @ J_pose
        b[idx_pose:idx_pose+3] += J_pose.T @ landmark_information @ e

        total_error += np.linalg.norm(e)

    print(f"Iteration {iteration+1}: Error = {total_error:.6f}")

print("Optimized Poses:\n", poses)
print("Optimized Landmarks:\n", landmarks)
