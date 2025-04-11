import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ex2_utils import generate_responses_1
from ex1_utils import gausssmooth

def get_kernel_weights(x, y, kernel_size):
    # Calculate normalized squared distances from center
    dist_sq = (x**2 + y**2) / (kernel_size**2)
    weights = np.zeros_like(dist_sq)
    # Only keep weights within kernel radius (dist <= 1)
    mask = dist_sq <= 1.0
    weights[mask] = 1 - dist_sq[mask]  # Epanechnikov profile
    return weights

def seek_mode(response_map, start_pos, kernel_size=15, max_iter=100, epsilon=1e-2):
    # Initialize position and clip to image bounds
    pos = np.array(start_pos, dtype=np.float32)
    pos[0] = np.clip(pos[0], 0, response_map.shape[1] - 1)
    pos[1] = np.clip(pos[1], 0, response_map.shape[0] - 1)

    trajectory = [pos.copy()]  # Store search path
    half_size = int(np.ceil(kernel_size))
    num_iterations = 0  

    for _ in range(max_iter):
        num_iterations += 1  

        # Define search window bounds (clipped to image edges)
        x_min = max(0, int(pos[0]) - half_size)
        x_max = min(response_map.shape[1], int(pos[0]) + half_size + 1)
        y_min = max(0, int(pos[1]) - half_size)
        y_max = min(response_map.shape[0], int(pos[1]) + half_size + 1)

        # Create coordinate grids relative to center
        x_coords = np.arange(x_min, x_max) - pos[0]
        y_coords = np.arange(y_min, y_max) - pos[1]
        x, y = np.meshgrid(x_coords, y_coords)

        # Compute kernel weights and get response values
        weights = get_kernel_weights(x, y, kernel_size)
        window_response = response_map[y_min:y_max, x_min:x_max]

        # Compute weighted response
        weighted_response = window_response * weights
        total_weight = np.sum(weighted_response)

        # Stop if no significant weights
        if total_weight <= 1e-10:
            break

        # Calculate mean shift vector
        shift_x = np.sum(x * weighted_response) / total_weight
        shift_y = np.sum(y * weighted_response) / total_weight

        # Update position and clip to image bounds
        new_pos = pos + np.array([shift_x, shift_y])
        new_pos[0] = np.clip(new_pos[0], 0, response_map.shape[1] - 1)
        new_pos[1] = np.clip(new_pos[1], 0, response_map.shape[0] - 1)

        # Check convergence
        if np.linalg.norm(new_pos - pos) < epsilon:
            break

        pos = new_pos
        trajectory.append(pos.copy())

    return pos, trajectory, num_iterations

# Custom response map generators for testing mean-shift behavior

def generate_responses_2():
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[30, 30] = 1.0    # Primary peak
    responses[70, 70] = 0.8    # Secondary peak
    return gausssmooth(responses, 7)  # Apply Gaussian smoothing

def generate_responses_3():
    responses = np.zeros((100, 100), dtype=np.float32)
    
    # Create diagonal ridge with sinusoidal intensity variation
    for i in range(100):
        intensity = 0.5 + 0.5 * np.sin(i * 0.1)  # Varying intensity
        pos = int(0.7 * i)  # Diagonal position
        if 0 <= pos < 100:
            responses[i, pos] = intensity
    
    # Add perpendicular structures
    responses[30:70, 60] = 0.6  # Vertical line
    responses[80, 20:80] = 0.4   # Horizontal segment
    
    # Add noise and smooth
    responses += np.random.normal(0, 0.03, (100, 100))
    return gausssmooth(responses, sigma=5)