"""
Simple test to visualize selected points
"""

import cv2
import numpy as np
import os
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Test parameters
img_path = "STM_img_simu/TPM_image/001.png"
scan_zoom_in_scale = "20n"
scan_position = (0e-9, 0e-9)

print("Starting visualization test...")

# Load image
print(f"Loading image from: {img_path}")
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Error: Could not load image from {img_path}")
    exit(1)

image = cv2.resize(image, (304, 304), interpolation=cv2.INTER_AREA)
print(f"Image loaded and resized to: {image.shape}")


# Convert scan scale
def convert_si_prefix(scale_str):
    """Convert SI prefix string to meters"""
    if isinstance(scale_str, (int, float)):
        return float(scale_str)

    scale_str = scale_str.strip().lower()

    # Handle different formats
    if scale_str.endswith("m"):
        scale_str = scale_str[:-1]  # Remove 'm'

    # Extract number and unit
    import re

    match = re.match(r"^(\d*\.?\d*)\s*([a-zA-Z]*)$", scale_str)
    if not match:
        return float(scale_str)

    number_str, unit = match.groups()
    number = float(number_str) if number_str else 1.0

    # SI prefix conversions
    prefixes = {
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "": 1.0,
        "k": 1e3,
        "M": 1e6,
        "G": 1e9,
    }

    multiplier = prefixes.get(unit, 1.0)
    return number * multiplier


zoom_out_scale = convert_si_prefix(scan_zoom_in_scale)
zoom_out_scale_nano = zoom_out_scale * 10**9

print(
    f"Scan scale: {scan_zoom_in_scale} = {zoom_out_scale} m = {zoom_out_scale_nano} nm"
)

# Create some mock selected molecules for visualization
print("Creating mock selected molecules...")
selected_molecules = [
    (
        1,
        -5e-9,
        -3e-9,
        0.9,
        0,
        -4e-9,
        -2e-9,
    ),  # class, x, y, confidence, area, keypoint_x, keypoint_y
    (2, 2e-9, 4e-9, 0.8, 0, 3e-9, 5e-9),
    (1, -2e-9, 6e-9, 0.85, 0, -1e-9, 7e-9),
]

print(f"Mock molecules created: {len(selected_molecules)}")

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Show image
ax.imshow(
    image,
    cmap="gray",
    extent=[
        -zoom_out_scale_nano / 2,
        zoom_out_scale_nano / 2,
        -zoom_out_scale_nano / 2,
        zoom_out_scale_nano / 2,
    ],
)

# Plot selected molecules
colors = ["red", "blue", "green", "orange", "purple"]
for i, molecule in enumerate(selected_molecules):
    mol_class, x, y, conf, area, kx, ky = molecule

    # Convert to nanometers for display
    x_nm = x * 1e9
    y_nm = y * 1e9
    kx_nm = kx * 1e9
    ky_nm = ky * 1e9

    color = colors[i % len(colors)]

    # Plot molecule center
    ax.plot(
        x_nm,
        y_nm,
        "o",
        color=color,
        markersize=12,
        label=f"Molecule {i+1} (Class {mol_class})",
    )

    # Plot keypoint
    ax.plot(kx_nm, ky_nm, "s", color=color, markersize=8, alpha=0.7)

    # Connect molecule center to keypoint
    ax.plot([x_nm, kx_nm], [y_nm, ky_nm], "--", color=color, alpha=0.5)

    # Add text annotation
    ax.annotate(
        f"M{i+1}",
        (x_nm, y_nm),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        color=color,
        weight="bold",
    )

ax.set_xlabel("X Position (nm)")
ax.set_ylabel("Y Position (nm)")
ax.set_title("Selected Molecules Visualization Test")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Create output directory
os.makedirs("test_mark", exist_ok=True)
save_path = "test_mark/simple_visualization_test.png"

plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Visualization saved to: {save_path}")

# Show plot
plt.show()

print("Simple visualization test completed!")
