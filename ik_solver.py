import numpy as np

# Link lengths and offsets (adjust if needed)
l1 = 0.130
l2 = 0.130
l3 = 0.05
d1 = 0.115
d2 = 0.0635
base_offset_y = 0.0



def inverse_kinematics(x, y, z, yaw=0, pitch=0, roll=0, pitch_desired=0, leg="left", elbow_up=False, foot_outward="yes"):
    xtz = (x)*(np.cos(yaw)*np.cos(pitch)) + (y-base_offset_y)*(np.sin(yaw)*np.cos(pitch)) + (z)*(np.sin(pitch))
    ytz = (x)*((np.cos(yaw)*np.sin(pitch)*np.sin(roll)) - (np.sin(yaw)*np.cos(roll))) + (y-base_offset_y)*((np.sin(yaw)*np.sin(pitch)*np.sin(roll))+(np.cos(yaw)*np.cos(roll))) + (z)*(np.cos(pitch)*np.sin(roll))

    if (leg=="left"):
        ztz = (x)*((np.cos(yaw)*np.sin(pitch)*np.cos(roll)) + (np.sin(yaw)*np.sin(roll))) + (y-base_offset_y)*((np.sin(yaw)*np.sin(pitch)*np.cos(roll)) - (np.cos(yaw)*np.sin(roll))) + (z)*(np.cos(pitch)*np.cos(roll)) - d2
    elif (leg=="right"):
        ztz = (x)*((np.cos(yaw)*np.sin(pitch)*np.cos(roll)) + (np.sin(yaw)*np.sin(roll))) + (y-base_offset_y)*((np.sin(yaw)*np.sin(pitch)*np.cos(roll)) - (np.cos(yaw)*np.sin(roll))) + (z)*(np.cos(pitch)*np.cos(roll)) + d2
    else:
        raise ValueError(f"Unknown leg identifier: {leg}")

    theta1 = np.arctan2(ztz, xtz)  # Hip yaw
    xt = np.sqrt(xtz**2 + ztz**2)
    yt = ytz + d1

    theta = pitch_desired - pitch
    x_ankle = xt - l3 * np.cos(theta)
    y_ankle = yt - l3 * np.sin(theta)

    dist = np.hypot(x_ankle, y_ankle)
    if dist > l1 + l2 or dist < abs(l1 - l2):
        print(f"[IK ERROR] {leg} foot target {x, y, z} is unreachable | dist={dist:.3f}, max={l1 + l2:.3f}")


    c3 = (x_ankle**2 + y_ankle**2 - l1**2 - l2**2) / (2 * l1 * l2)
    c3 = np.clip(c3, -1, 1)
    theta3_pos = np.arccos(c3)
    theta3_neg = -np.arccos(c3)

    gamma = np.arctan2(y_ankle, x_ankle)
    beta = np.arccos((l1**2 + x_ankle**2 + y_ankle**2 - l2**2) / (2 * l1 * np.sqrt(x_ankle**2 + y_ankle**2)))

    theta2_a = gamma - beta
    theta2_b = gamma + beta

    if elbow_up:
        theta2 = theta2_b
        theta3 = theta3_neg
    else:
        theta2 = theta2_a
        theta3 = theta3_pos

    theta4 = theta - (theta2 + theta3)

    # Normalize angles
    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
    theta3 = (theta3 + np.pi) % (2 * np.pi) - np.pi
    theta4 = (theta4 + np.pi) % (2 * np.pi) - np.pi

    return theta1, theta2, theta3, theta4
