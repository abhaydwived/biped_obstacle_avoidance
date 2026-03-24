import numpy as np

# Link lengths and offsets
l1 = 0.130
l2 = 0.130
l3 = 0.05
d1 = 0.115
d2 = 0.0635
base_offset_y = 0.0  # not needed anymore if using PyBullet world frame

def inverse_kinematics(x, y, z, yaw=0, pitch=0, roll=0, pitch_desired=0, leg="left", elbow_up=False, foot_outward="yes"):
    # ✅ Use PyBullet convention: X (lateral), Y (forward), Z (up)

    # Apply lateral hip offset (in X axis)
    if leg == "left":
        x -= d2
    elif leg == "right":
        x += d2
    else:
        raise ValueError(f"Unknown leg identifier: {leg}")

    # Account for vertical hip offset (in Z axis)
    z += d1

    # Hip yaw angle
    theta1 = np.arctan2(x, y)

    # Projected length in sagittal plane (Y-Z)
    xt = np.sqrt(x**2 + y**2)
    yt = z

    # Ankle position
    theta = pitch_desired - pitch
    x_ankle = xt - l3 * np.cos(theta)
    y_ankle = yt - l3 * np.sin(theta)

    dist = np.hypot(x_ankle, y_ankle)
    if dist > l1 + l2 or dist < abs(l1 - l2):
        print(f"[IK ERROR] {leg} foot target {(x, y, z)} is unreachable | dist={dist:.3f}, max={l1 + l2:.3f}")
        return 0.0, 0.0, 0.0, 0.0

    # Law of cosines for knee
    c3 = (x_ankle**2 + y_ankle**2 - l1**2 - l2**2) / (2 * l1 * l2)
    c3 = np.clip(c3, -1.0, 1.0)
    theta3 = np.arccos(c3)
    if elbow_up:
        theta3 = -theta3

    # Hip pitch
    gamma = np.arctan2(y_ankle, x_ankle)
    beta = np.arccos((l1**2 + x_ankle**2 + y_ankle**2 - l2**2) / (2 * l1 * dist))
    theta2 = gamma - beta if not elbow_up else gamma + beta

    # Ankle pitch
    theta4 = theta - (theta2 + theta3)

    # Normalize angles
    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
    theta3 = (theta3 + np.pi) % (2 * np.pi) - np.pi
    theta4 = (theta4 + np.pi) % (2 * np.pi) - np.pi

    return theta1, theta2, theta3, theta4
