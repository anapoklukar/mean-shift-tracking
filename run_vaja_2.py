import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ex2_utils import generate_responses_1
from ex1_utils import gausssmooth
from tracking_methods import seek_mode, generate_responses_2, generate_responses_3

# Generate response map
response_map = generate_responses_2()
h_img, w_img = response_map.shape

# Define evenly spread start points in [0,100] x [0,100]
grid_x = np.linspace(0, 100, 5)  # 5 points along x-axis
grid_y = np.linspace(0, 100, 5)  # 5 points along y-axis
starting_points = [(int(x), int(y)) for x in grid_x for y in grid_y]  # Create grid

# Define kernel sizes to test
kernel_sizes = [3, 9, 21, 41]
tolerance = 1e-5  # Fixed termination criteria
max_iter = 1000  # Max iterations before stopping

# Create subplots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten the 2D array for easy indexing

# Store results
average_iterations = {}
non_converged_counts = {}

for i, kernel_size in enumerate(kernel_sizes):
    ax = axes[i]
    ax.imshow(response_map, cmap='hot', origin='lower', extent=[0, w_img, 0, h_img])
    ax.set_title(f"Kernel Size: {kernel_size}")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    
    total_iterations = 0
    non_converged = 0
    
    for start in starting_points:
        mode, path, iterations = seek_mode(response_map, start, kernel_size, max_iter, tolerance)
        path = np.array(path)
        
        ax.plot(path[:, 0], path[:, 1], color="blue", marker=".", linewidth=2, markersize=6)  # Convergence path
        ax.scatter(start[0], start[1], color="cyan", marker="x", s=80)  # Start point
        ax.scatter(mode[0], mode[1], color="blue", marker="o", s=80)  # End mode
        
        total_iterations += iterations
        if iterations >= max_iter:
            non_converged += 1  # Track non-convergence cases
    
    # Compute average iterations
    average_iterations[kernel_size] = total_iterations / len(starting_points)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("mean_shift_convergence_kernels_2.pdf", dpi=300)
plt.show()

# Print summary statistics
print("Kernel Size | Avg. Iterations")
print("-----------------------------")
for k in kernel_sizes:
    print(f"{k:^11} | {average_iterations[k]:^15.2f}")
    
# Generate response map
response_map = generate_responses_3()
h_img, w_img = response_map.shape

# Define evenly spread start points in [0,100] x [0,100]
grid_x = np.linspace(0, 100, 5)  # 5 points along x-axis
grid_y = np.linspace(0, 100, 5)  # 5 points along y-axis
starting_points = [(int(x), int(y)) for x in grid_x for y in grid_y]  # Create grid

# Define tolerance values to test
tolerances = [0.001, 0.01, 0.1, 1]
kernel_size = 15  # Fixed kernel size
max_iter = 1000  # Max iterations before stopping

# Create subplots (2x3 grid to fit 5 tolerances)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten the 2D array for easy indexing

# Store results
average_iterations = {}
non_converged_counts = {}

for i, tolerance in enumerate(tolerances):
    ax = axes[i]
    ax.imshow(response_map, cmap='hot', origin='lower', extent=[0, w_img, 0, h_img])
    ax.set_title(f"Tolerance: {tolerance}")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    
    total_iterations = 0
    non_converged = 0
    
    for start in starting_points:
        mode, path, iterations = seek_mode(response_map, start, kernel_size, max_iter, tolerance)
        path = np.array(path)
        
        ax.plot(path[:, 0], path[:, 1], color="blue", marker=".", linewidth=2, markersize=6)  # Convergence path
        ax.scatter(start[0], start[1], color="cyan", marker="x", s=80)  # Start point
        ax.scatter(mode[0], mode[1], color="blue", marker="o", s=80)  # End mode
        
        total_iterations += iterations
    
    # Compute average iterations
    average_iterations[tolerance] = total_iterations / len(starting_points)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("mean_shift_convergence_tolerances_3.pdf", dpi=300)
plt.show()

# Print summary statistics
print("Tolerance | Avg. Iterations")
print("---------------------------")
for t in tolerances:
    print(f"{t:^9} | {average_iterations[t]:^15.2f}")

import time
import os
import cv2
from sequence_utils import VOTSequence
from ncc_tracker_example import NCCTracker, NCCParams
from ms_tracker import MeanShiftTracker, MSParams

# set the path to the directory where you have the sequences
dataset_path = 'vot2014'  # TODO: set to the dataset path on your disk

# initialize a dictionary to save the number of failures and FPS for each sequence
results_dict = {}

# visualization and setup parameters
win_name = 'Tracking window'
reinitialize = True
show_gt = False
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN

# Loop through all sequences in the dataset directory
for sequence_name in os.listdir(dataset_path):
    sequence_path = os.path.join(dataset_path, sequence_name)

    # Check if it's a directory
    if os.path.isdir(sequence_path):
        print(f"Processing sequence: {sequence_name}")
        
        # create sequence object
        sequence = VOTSequence(dataset_path, sequence_name)
        init_frame = 0
        n_failures = 0

        # create parameters and tracker objects
        parameters = MSParams()
        tracker = MeanShiftTracker(parameters)

        time_all = 0

        # initialize visualization window
        sequence.initialize_window(win_name)
        
        # tracking loop - goes over all frames in the video sequence
        frame_idx = 0
        while frame_idx < sequence.length():
            img = cv2.imread(sequence.frame(frame_idx))
            
            # initialize or track
            if frame_idx == init_frame:
                # initialize tracker (at the beginning of the sequence or after tracking failure)
                t_ = time.time()
                tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
                time_all += time.time() - t_
                predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
            else:
                # track on current frame - predict bounding box
                t_ = time.time()
                predicted_bbox = tracker.track(img)
                time_all += time.time() - t_

            # calculate overlap (needed to determine failure of a tracker)
            gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
            o = sequence.overlap(predicted_bbox, gt_bb)

            # draw ground-truth and predicted bounding boxes, frame numbers and show image
            if show_gt:
                sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
            sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
            sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
            sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
            sequence.show_image(img, video_delay)

            if o > 0 or not reinitialize:
                # increase frame counter by 1
                frame_idx += 1
            else:
                # increase frame counter by 5 and set re-initialization to the next frame
                frame_idx += 5
                init_frame = frame_idx
                n_failures += 1

        # Calculate FPS for the current sequence
        fps = sequence.length() / time_all

        # Save the number of failures and FPS for the current sequence
        results_dict[sequence_name] = {'failures': n_failures, 'fps': fps}

        print(f"Tracking finished for {sequence_name}. Failures: {n_failures}, FPS: {fps:.1f}")

# Save results to a text file
with open('tracking_results.txt', 'w') as f:
    for sequence_name, results in results_dict.items():
        f.write(f"{sequence_name}: {results['failures']} failures, {results['fps']:.1f} FPS\n")

print("All sequences processed. Results saved in 'tracking_results.txt'.")

# Initialize variables to accumulate total failures and FPS
total_failures = 0
total_fps = 0.0
count = 0

# Open the results file and read it
with open('tracking_results.txt', 'r') as f:
    for line in f:
        # Extract sequence name, failures, and FPS from each line
        parts = line.split(':')
        if len(parts) > 1:
            sequence_info = parts[1].split(',')
            failures = int(sequence_info[0].split()[0])  # Get the number of failures
            fps = float(sequence_info[1].split()[0])  # Get the FPS
            
            # Accumulate the failures and FPS
            total_failures += failures
            total_fps += fps
            count += 1

# Calculate the average FPS
average_fps = total_fps / count if count > 0 else 0

# Print the results
print(f"Total Failures: {total_failures}")
print(f"Average FPS: {average_fps:.2f}")

import matplotlib.pyplot as plt
import numpy as np

# Data from tracking_results.txt
sequences = [
    "surfing", "skating", "car", "fish2", "torus", "gymnastics", "hand2", "sunshade", 
    "bolt", "sphere", "basketball", "tunnel", "david", "ball", "fish1", "hand1", 
    "fernando", "diving", "jogging", "polarbear", "bicycle", "drunk", "motocross", 
    "trellis", "woman"
]
failures = [1, 5, 0, 4, 1, 0, 2, 0, 4, 0, 2, 3, 2, 0, 6, 0, 1, 1, 2, 0, 2, 0, 3, 1, 2]
fps = [
    854.0, 307.6, 586.8, 580.0, 501.4, 394.2, 444.9, 455.8, 534.0, 235.2, 309.2, 
    601.1, 281.6, 581.4, 687.2, 568.7, 207.0, 378.4, 541.1, 643.7, 572.7, 505.6, 
    141.7, 424.1, 425.0
]

# Sort sequences by FPS for better visualization
sorted_indices = np.argsort(fps)[::-1]
sequences = [sequences[i] for i in sorted_indices]
failures = [failures[i] for i in sorted_indices]
fps = [fps[i] for i in sorted_indices]

# Create figure and axis
fig, ax = plt.subplots(figsize=(16, 8))

# Set positions and width for the bars
x = np.arange(len(sequences))
width = 0.35

# Plot FPS bars
fps_bars = ax.bar(x - width/2, fps, width, label='FPS', color='royalblue', alpha=0.7)

# Plot Failures bars on the same axis but with different scale
ax2 = ax.twinx()
failures_bars = ax2.bar(x + width/2, failures, width, label='Failures', color='crimson', alpha=0.7)

# Add labels, title, and ticks
ax.set_ylabel('FPS', color='royalblue')
ax2.set_ylabel('Number of Failures', color='crimson')
ax.set_xticks(x)
ax.set_xticklabels(sequences, rotation=90)
ax.tick_params(axis='y', labelcolor='royalblue')
ax2.tick_params(axis='y', labelcolor='crimson')

# Add value labels on top of each bar
for bar in fps_bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=8)

for bar in failures_bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height}',
            ha='center', va='bottom', fontsize=8, color='crimson')

plt.tight_layout()
plt.savefig("tracking_results.pdf", dpi=1000)
plt.show()

import time
import os
import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sequence_utils import VOTSequence
from ms_tracker import MeanShiftTracker, MSParams

# set the path to the directory where you have the sequences
dataset_path = 'vot2014'  # TODO: set to the dataset path on your disk

# Initialize PDF for saving failure screenshots
pdf_path = 'tracking_failures.pdf'
pdf_pages = PdfPages(pdf_path)

# visualization and setup parameters
win_name = 'Tracking window'
reinitialize = True
show_gt = True  # Show ground truth for better visualization
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN

# Loop through all sequences in the dataset directory
for sequence_name in os.listdir(dataset_path):
    sequence_path = os.path.join(dataset_path, sequence_name)

    # Check if it's a directory
    if os.path.isdir(sequence_path):
        print(f"Processing sequence: {sequence_name}")
        
        # create sequence object
        sequence = VOTSequence(dataset_path, sequence_name)
        init_frame = 0

        # create parameters and tracker objects
        parameters = MSParams()
        tracker = MeanShiftTracker(parameters)

        # initialize visualization window
        sequence.initialize_window(win_name)
        
        # tracking loop
        frame_idx = 0
        while frame_idx < sequence.length():
            img = cv2.imread(sequence.frame(frame_idx))
            
            if frame_idx == init_frame:
                tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
                predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
            else:
                predicted_bbox = tracker.track(img)

            gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
            o = sequence.overlap(predicted_bbox, gt_bb)

            # Draw annotations
            if show_gt:
                sequence.draw_region(img, gt_bb, (0, 255, 0), 1)  # Green for GT
            sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)  # Red for prediction
            sequence.draw_text(img, f'Frame: {frame_idx + 1}/{sequence.length()}', (25, 25))
            
            # Check for failure
            if o <= 0 and reinitialize:
                # Save failure screenshot to PDF
                fig = plt.figure(figsize=(10, 6))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f'Failure in {sequence_name} - Frame {frame_idx}')
                plt.axis('off')
                pdf_pages.savefig(fig)
                plt.close()
                
                # Jump frames after failure
                frame_idx += 5
                init_frame = frame_idx
            else:
                frame_idx += 1

            sequence.show_image(img, video_delay)

print(f"All sequences processed. Saving PDF with failure screenshots to {pdf_path}")
pdf_pages.close()

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sequence_utils import VOTSequence
from ms_tracker import MeanShiftTracker, MSParams

# Configuration
dataset_path = 'vot2014'
win_name = 'Tracking Window'
video_delay = 1  # Faster visualization for parameter testing

# Parameter combinations to test
kernel_sizes = [3, 15, 41]
#n_bins_list = [8, 16, 32]
#alphas = [0.005, 0.01, 0.02]
#epsilons = [1e-8, 1e-6, 1e-4]  # Added epsilon variations

results = []

def run_tracking_sequence(params, sequence_name):
    sequence = VOTSequence(dataset_path, sequence_name)
    tracker = MeanShiftTracker(params)
    
    init_frame = 0
    n_failures = 0
    time_all = 0
    
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        
        if frame_idx == init_frame:
            t_start = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, 'rectangle'))
            time_all += time.time() - t_start
            predicted_bbox = sequence.get_annotation(frame_idx, 'rectangle')
        else:
            t_start = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_start

        o = sequence.overlap(predicted_bbox, 
                           sequence.get_annotation(frame_idx, 'rectangle'))
        
        if o > 0:
            frame_idx += 1
        else:
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    fps = sequence.length() / max(0.001, time_all)  # Avoid division by zero
    return fps, n_failures

# Get all sequence names
sequence_names = [name for name in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, name))]

# Run experiments for all parameter combinations
for kernel_size in kernel_sizes:
    for n_bins in n_bins_list:
        for alpha in alphas:
            for eps in epsilons:
                total_failures = 0
                total_fps = 0
                sequence_count = 0
                
                params = MSParams(kernel_size=kernel_size, 
                                n_bins=n_bins, 
                                alpha=alpha,
                                eps=eps)
                
                for seq_name in sequence_names:
                    try:
                        fps, failures = run_tracking_sequence(params, seq_name)
                        total_failures += failures
                        total_fps += fps
                        sequence_count += 1
                    except Exception as e:
                        print(f'  Error processing {seq_name}: {str(e)}')
                        continue
                
                avg_fps = total_fps / sequence_count if sequence_count > 0 else 0
                results.append({
                    'kernel': kernel_size,
                    'bins': n_bins,
                    'alpha': alpha,
                    'epsilon': eps,
                    'avg_fps': avg_fps,
                    'total_failures': total_failures,
                    'sequences_tested': sequence_count
                })

# Print summary
print('\n=== Top 5 Parameter Combinations ===')
results_sorted = sorted(results, key=lambda x: (x['total_failures'], -x['avg_fps']))
for i, param in enumerate(results_sorted[:5]):
    print(f"{i+1}. Kernel: {param['kernel']}, Bins: {param['bins']}, "
          f"Alpha: {param['alpha']:.3f}, Eps: {param['epsilon']:.1e} - "
          f"{param['total_failures']} total failures, "
          f"{param['avg_fps']:.1f} avg FPS")

# Save results to CSV for further analysis
import csv
with open('tracking_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['kernel', 'bins', 'alpha', 'epsilon', 'avg_fps', 'total_failures']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results_sorted:
        writer.writerow(result)
        
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sequence_utils import VOTSequence
from ms_tracker import MeanShiftTracker, MSParams

# Configuration
dataset_path = 'vot2014'
win_name = 'Tracking Window'

# Visualization parameters (from second version)
reinitialize = True
show_gt = False
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN

# Only testing kernel sizes now (other parameters fixed)
kernel_sizes = [3, 15, 41]
fixed_bins = 16      # Fixed number of bins
fixed_alpha = 0.01   # Fixed learning rate
fixed_eps = 1e-7     # Fixed epsilon value

results = []

def run_tracking_sequence(params, sequence_name):
    sequence = VOTSequence(dataset_path, sequence_name)
    tracker = MeanShiftTracker(params)
    
    init_frame = 0
    n_failures = 0
    time_all = 0
    
    # Initialize visualization window (from second version)
    sequence.initialize_window(win_name)
    
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        
        if frame_idx == init_frame:
            t_start = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, 'rectangle'))
            time_all += time.time() - t_start
            predicted_bbox = sequence.get_annotation(frame_idx, 'rectangle')
        else:
            t_start = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_start

        o = sequence.overlap(predicted_bbox, 
                           sequence.get_annotation(frame_idx, 'rectangle'))
        
        # Visualization (from second version)
        if show_gt:
            sequence.draw_region(img, sequence.get_annotation(frame_idx, 'rectangle'), (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        sequence.show_image(img, video_delay)
        
        if o > 0:
            frame_idx += 1
        else:
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    fps = sequence.length() / max(0.001, time_all)  # Avoid division by zero
    return fps, n_failures

# Get all sequence names
sequence_names = [name for name in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, name))]

# Run experiments for kernel sizes only
for kernel_size in kernel_sizes:
    total_failures = 0
    total_fps = 0
    sequence_count = 0
    
    params = MSParams(kernel_size=kernel_size, 
                    n_bins=fixed_bins, 
                    alpha=fixed_alpha,
                    eps=fixed_eps)
    
    print(f'\nTesting Kernel: {kernel_size} (Fixed: Bins={fixed_bins}, Alpha={fixed_alpha:.3f}, Eps={fixed_eps:.1e})')
    
    for seq_name in sequence_names:
        try:
            fps, failures = run_tracking_sequence(params, seq_name)
            total_failures += failures
            total_fps += fps
            sequence_count += 1
            print(f'  {seq_name}: {fps:.1f} FPS, {failures} failures')
        except Exception as e:
            print(f'  Error processing {seq_name}: {str(e)}')
            continue
    
    avg_fps = total_fps / sequence_count if sequence_count > 0 else 0
    results.append({
        'kernel': kernel_size,
        'avg_fps': avg_fps,
        'total_failures': total_failures,
        'sequences_tested': sequence_count
    })

# Sort results by kernel size
results_sorted = sorted(results, key=lambda x: x['kernel'])

# Print summary
print('\n=== Results by Kernel Size ===')
print(f'Fixed parameters: Bins={fixed_bins}, Alpha={fixed_alpha:.3f}, Eps={fixed_eps:.1e}\n')
for result in results_sorted:
    print(f"Kernel: {result['kernel']} - "
          f"{result['total_failures']} total failures, "
          f"{result['avg_fps']:.1f} avg FPS")
    
    
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sequence_utils import VOTSequence
from ms_tracker import MeanShiftTracker, MSParams

# Configuration
dataset_path = 'vot2014'
win_name = 'Tracking Window'

# Visualization parameters
reinitialize = True
show_gt = False
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN

# Testing different numbers of histogram bins
n_bins_list = [8, 16, 32]  # Three different bin counts to test
fixed_kernel = 15           # Fixed kernel size
fixed_alpha = 0.01          # Fixed learning rate
fixed_eps = 1e-7            # Fixed epsilon value

results = []

def run_tracking_sequence(params, sequence_name):
    sequence = VOTSequence(dataset_path, sequence_name)
    tracker = MeanShiftTracker(params)
    
    init_frame = 0
    n_failures = 0
    time_all = 0
    
    sequence.initialize_window(win_name)
    
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        
        if frame_idx == init_frame:
            t_start = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, 'rectangle'))
            time_all += time.time() - t_start
            predicted_bbox = sequence.get_annotation(frame_idx, 'rectangle')
        else:
            t_start = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_start

        o = sequence.overlap(predicted_bbox, 
                           sequence.get_annotation(frame_idx, 'rectangle'))
        
        if show_gt:
            sequence.draw_region(img, sequence.get_annotation(frame_idx, 'rectangle'), (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        sequence.show_image(img, video_delay)
        
        if o > 0:
            frame_idx += 1
        else:
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    fps = sequence.length() / max(0.001, time_all)
    return fps, n_failures

# Get all sequence names
sequence_names = [name for name in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, name))]

# Run experiments for different bin counts
for n_bins in n_bins_list:
    total_failures = 0
    total_fps = 0
    sequence_count = 0
    
    params = MSParams(kernel_size=fixed_kernel,
                    n_bins=n_bins,
                    alpha=fixed_alpha,
                    eps=fixed_eps)
    
    print(f'\nTesting Bins: {n_bins} (Fixed: Kernel={fixed_kernel}, Alpha={fixed_alpha:.3f}, Eps={fixed_eps:.1e})')
    
    for seq_name in sequence_names:
        try:
            fps, failures = run_tracking_sequence(params, seq_name)
            total_failures += failures
            total_fps += fps
            sequence_count += 1
            print(f'  {seq_name}: {fps:.1f} FPS, {failures} failures')
        except Exception as e:
            print(f'  Error processing {seq_name}: {str(e)}')
            continue
    
    avg_fps = total_fps / sequence_count if sequence_count > 0 else 0
    results.append({
        'bins': n_bins,
        'avg_fps': avg_fps,
        'total_failures': total_failures,
        'sequences_tested': sequence_count
    })

# Sort results by number of bins
results_sorted = sorted(results, key=lambda x: x['bins'])

# Print summary
print('\n=== Results by Number of Bins ===')
print(f'Fixed parameters: Kernel={fixed_kernel}, Alpha={fixed_alpha:.3f}, Eps={fixed_eps:.1e}\n')
for result in results_sorted:
    print(f"Bins: {result['bins']} - "
          f"{result['total_failures']} total failures, "
          f"{result['avg_fps']:.1f} avg FPS")
    
    
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sequence_utils import VOTSequence
from ms_tracker import MeanShiftTracker, MSParams

# Configuration
dataset_path = 'vot2014'
win_name = 'Tracking Window'

# Visualization parameters
reinitialize = True
show_gt = False
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN

# Testing different learning rates (alpha)
alpha_list = [0.001, 0.01, 0.1]  # Three different alpha values to test
fixed_kernel = 15                   # Fixed kernel size
fixed_bins = 16                     # Fixed number of bins
fixed_eps = 1e-7                    # Fixed epsilon value

results = []

def run_tracking_sequence(params, sequence_name):
    sequence = VOTSequence(dataset_path, sequence_name)
    tracker = MeanShiftTracker(params)
    
    init_frame = 0
    n_failures = 0
    time_all = 0
    
    sequence.initialize_window(win_name)
    
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        
        if frame_idx == init_frame:
            t_start = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, 'rectangle'))
            time_all += time.time() - t_start
            predicted_bbox = sequence.get_annotation(frame_idx, 'rectangle')
        else:
            t_start = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_start

        o = sequence.overlap(predicted_bbox, 
                           sequence.get_annotation(frame_idx, 'rectangle'))
        
        if show_gt:
            sequence.draw_region(img, sequence.get_annotation(frame_idx, 'rectangle'), (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        sequence.show_image(img, video_delay)
        
        if o > 0:
            frame_idx += 1
        else:
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    fps = sequence.length() / max(0.001, time_all)
    return fps, n_failures

# Get all sequence names
sequence_names = [name for name in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, name))]

# Run experiments for different alpha values
for alpha in alpha_list:
    total_failures = 0
    total_fps = 0
    sequence_count = 0
    
    params = MSParams(kernel_size=fixed_kernel,
                    n_bins=fixed_bins,
                    alpha=alpha,
                    eps=fixed_eps)
    
    print(f'\nTesting Alpha: {alpha:.3f} (Fixed: Kernel={fixed_kernel}, Bins={fixed_bins}, Eps={fixed_eps:.1e})')
    
    for seq_name in sequence_names:
        try:
            fps, failures = run_tracking_sequence(params, seq_name)
            total_failures += failures
            total_fps += fps
            sequence_count += 1
            print(f'  {seq_name}: {fps:.1f} FPS, {failures} failures')
        except Exception as e:
            print(f'  Error processing {seq_name}: {str(e)}')
            continue
    
    avg_fps = total_fps / sequence_count if sequence_count > 0 else 0
    results.append({
        'alpha': alpha,
        'avg_fps': avg_fps,
        'total_failures': total_failures,
        'sequences_tested': sequence_count
    })

# Sort results by alpha value
results_sorted = sorted(results, key=lambda x: x['alpha'])

# Print summary
print('\n=== Results by Learning Rate (Alpha) ===')
print(f'Fixed parameters: Kernel={fixed_kernel}, Bins={fixed_bins}, Eps={fixed_eps:.1e}\n')
for result in results_sorted:
    print(f"Alpha: {result['alpha']:.3f} - "
          f"{result['total_failures']} total failures, "
          f"{result['avg_fps']:.1f} avg FPS")
    
    
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sequence_utils import VOTSequence
from ms_tracker import MeanShiftTracker, MSParams

# Configuration
dataset_path = 'vot2014'
win_name = 'Tracking Window'

# Visualization parameters
reinitialize = True
show_gt = False
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN

# Testing different convergence thresholds (epsilon)
eps_list = [1e-8, 1e-4, 1]  # Three different epsilon values to test
fixed_kernel = 15               # Fixed kernel size
fixed_bins = 16                 # Fixed number of bins
fixed_alpha = 0.01              # Fixed learning rate

results = []

def run_tracking_sequence(params, sequence_name):
    sequence = VOTSequence(dataset_path, sequence_name)
    tracker = MeanShiftTracker(params)
    
    init_frame = 0
    n_failures = 0
    time_all = 0
    
    sequence.initialize_window(win_name)
    
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        
        if frame_idx == init_frame:
            t_start = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, 'rectangle'))
            time_all += time.time() - t_start
            predicted_bbox = sequence.get_annotation(frame_idx, 'rectangle')
        else:
            t_start = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_start

        o = sequence.overlap(predicted_bbox, 
                           sequence.get_annotation(frame_idx, 'rectangle'))
        
        if show_gt:
            sequence.draw_region(img, sequence.get_annotation(frame_idx, 'rectangle'), (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        sequence.show_image(img, video_delay)
        
        if o > 0:
            frame_idx += 1
        else:
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    fps = sequence.length() / max(0.001, time_all)
    return fps, n_failures

# Get all sequence names
sequence_names = [name for name in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, name))]

# Run experiments for different epsilon values
for eps in eps_list:
    total_failures = 0
    total_fps = 0
    sequence_count = 0
    
    params = MSParams(kernel_size=fixed_kernel,
                    n_bins=fixed_bins,
                    alpha=fixed_alpha,
                    eps=eps)
    
    print(f'\nTesting Epsilon: {eps:.1e} (Fixed: Kernel={fixed_kernel}, Bins={fixed_bins}, Alpha={fixed_alpha:.3f})')
    
    for seq_name in sequence_names:
        try:
            fps, failures = run_tracking_sequence(params, seq_name)
            total_failures += failures
            total_fps += fps
            sequence_count += 1
            print(f'  {seq_name}: {fps:.1f} FPS, {failures} failures')
        except Exception as e:
            print(f'  Error processing {seq_name}: {str(e)}')
            continue
    
    avg_fps = total_fps / sequence_count if sequence_count > 0 else 0
    results.append({
        'epsilon': eps,
        'avg_fps': avg_fps,
        'total_failures': total_failures,
        'sequences_tested': sequence_count
    })

# Sort results by epsilon value
results_sorted = sorted(results, key=lambda x: x['epsilon'])

# Print summary
print('\n=== Results by Convergence Threshold (Epsilon) ===')
print(f'Fixed parameters: Kernel={fixed_kernel}, Bins={fixed_bins}, Alpha={fixed_alpha:.3f}\n')
for result in results_sorted:
    print(f"Epsilon: {result['epsilon']:.1e} - "
          f"{result['total_failures']} total failures, "
          f"{result['avg_fps']:.1f} avg FPS")