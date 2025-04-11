import numpy as np
from ex2_utils import Tracker
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram, backproject_histogram

class MeanShiftTracker(Tracker):

    def __init__(self, params):
        self.parameters = params  # Store parameters

    def initialize(self, image, region):
        
        if len(region) == 8:  # Convert polygon region to bounding box
            x_ = np.array(region[::2])  # Extract x coordinates
            y_ = np.array(region[1::2])  # Extract y coordinates
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        # Convert region values to integers
        region_int = [int(x) for x in region]
        x0 = max(region_int[0], 0)
        y0 = max(region_int[1], 0)
        x1 = min(region_int[0] + (int(region[2]) | 1), image.shape[1] - 1)
        y1 = min(region_int[1] + (int(region[3]) | 1), image.shape[0] - 1)

        # Initialize tracker position
        self.position = (int(region[0] + region[2] / 2), int(region[1] + region[3] / 2))
        self.size = (int(region[2]) | 1, int(region[3]) | 1)  # Ensure size is odd

        # Extract patch from the image
        t = image[int(y0):int(y1), int(x0):int(x1)]

        # Create Epanechnikov kernel
        self.weights = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.kernel_size)

        # Normalize kernel
        self.nc = np.sum(self.weights)

        # Compute histogram and normalize it properly
        q = extract_histogram(t, self.parameters.n_bins, self.weights[0:t.shape[0], 0:t.shape[1]])

        # **Corrected normalization**
        q_sum = np.sum(q) + 1e-6  # Avoid division by zero
        self.q = q / q_sum  # Normalize correctly


    def track(self, image):
        # Perform tracking using the Mean-Shift algorithm
        
        x_k, y_k = self.position
        min_distance = 1
        position_array = [[x_k, y_k]]

        # Create grid for the Epanechnikov kernel weights
        l = np.arange(-int(self.size[0] / 2), int(self.size[0] / 2) + 1)
        k = np.arange(-int(self.size[1] / 2), int(self.size[1] / 2) + 1)
        xi, yi = np.meshgrid(l, k)

        idx = 0
        while min_distance >= 0.2 and idx < 50:  # Iteration until convergence or max iterations
            idx += 1

            # Get the current patch and its histogram
            nf = get_patch(image, (x_k, y_k), self.size)[0]
            p = extract_histogram(nf, self.parameters.n_bins, self.weights[0:nf.shape[0], 0:nf.shape[1]])

            p_sum = np.sum(p) + 1e-6  # Avoid division by zero
            p = p / p_sum  # Proper normalization

            # Compute the weight map
            v = np.sqrt(np.divide(self.q, p + self.parameters.eps))
            wi = backproject_histogram(nf, v, self.parameters.n_bins)
            w = np.sum(wi)

            if w < 1e-6:  # Prevent division by zero
                break
            x_shift = np.sum(wi * xi) / w
            y_shift = np.sum(wi * yi) / w


            # Update position
            x_k += x_shift
            y_k += y_shift

            # Calculate the distance moved
            min_distance = np.sqrt(x_shift**2 + y_shift**2)

            position_array.append([x_k, y_k])

        # Update the position and model
        self.position = (x_k, y_k)
        self.q = self.q * (1 - self.parameters.alpha) + self.parameters.alpha * p

        # Calculate the bounding box
        x = min(max(round(int(self.position[0] - self.size[0] / 2)), 0), image.shape[1] - self.size[0])
        y = min(max(round(int(self.position[1] - self.size[1] / 2)), 0), image.shape[0] - self.size[1])

        return [x, y, self.size[0], self.size[1]]  # Return the updated bounding box

# Class to store and manage tracking parameters
class MSParams():
    def __init__(self, kernel_size=15, n_bins=16, alpha=0.01, eps= 1e-7):
        # Initialize parameters for the MeanShift tracker
        self.kernel_size = kernel_size  # Size of the Epanechnik kernel
        self.n_bins = n_bins  # Number of histogram bins
        self.alpha = alpha  # Learning rate for updating the model
        self.eps = eps  # Small epsilon value to avoid division by zero
        