"""
Real molecular detection test with visualization
"""

import cv2
import numpy as np
import os
import json
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Import the standalone functions
try:
    from test_simulation_standalone import (
        convert_si_prefix,
        molecular_seeker_standalone,
        auto_select_molecules_standalone,
        visualize_molecules_on_image,
    )

    print("✅ Successfully imported standalone functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

print("=== REAL MOLECULAR DETECTION TEST ===")

# Configuration
img_simu_path = "AI_TPM/STM_img_simu/TPM_image/001.png"
scan_zoom_in_scale = "20n"
scan_position = (0e-9, 0e-9)
max_molecules_per_scan = 3

# Convert scan scale
zoom_out_scale = convert_si_prefix(scan_zoom_in_scale)
zoom_out_scale_nano = zoom_out_scale * 10**9

print(
    f"Scan scale: {scan_zoom_in_scale} = {zoom_out_scale} m = {zoom_out_scale_nano} nm"
)

# Load and prepare image
image = cv2.imread(img_simu_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"❌ Error: Could not load image from {img_simu_path}")
    exit(1)

image = cv2.resize(image, (304, 304), interpolation=cv2.INTER_AREA)
print(f"✅ Image loaded and resized to: {image.shape}")

# Run molecular detection
print("\n🔍 Running molecular detection...")
shape_key_points_result = molecular_seeker_standalone(
    image,
    scan_position=scan_position,
    scan_edge=zoom_out_scale_nano,
    keypoint_model_path="AI_TPM/keypoint/best_0417.pt",
    save_dir="test_output",
    molecular_filter_threshold=1e-9,
)

if shape_key_points_result is not None and len(shape_key_points_result) > 0:
    print(f"✅ Detected {len(shape_key_points_result)} molecules")

    # Test intelligent molecule selection
    selected_molecules = auto_select_molecules_standalone(
        shape_key_points_result,
        max_molecules_per_scan,
        "intelligent",
        scan_position,
    )

    print(f"✅ Selected {len(selected_molecules)} molecules for processing")

    # Print detailed results
    print("\n📋 DETECTED MOLECULES:")
    for i, molecule in enumerate(shape_key_points_result):
        class_id, x, y, conf, area, kx, ky = molecule
        print(
            f"  Molecule {i+1}: Class={class_id}, Pos=({x*1e9:.2f}, {y*1e9:.2f}) nm, "
            f"Conf={conf:.3f}, Keypoint=({kx*1e9:.2f}, {ky*1e9:.2f}) nm"
        )

    print("\n🎯 SELECTED MOLECULES:")
    for i, molecule in enumerate(selected_molecules):
        class_id, x, y, conf, area, kx, ky = molecule
        print(
            f"  Selected {i+1}: Class={class_id}, Pos=({x*1e9:.2f}, {y*1e9:.2f}) nm, "
            f"Conf={conf:.3f}, Keypoint=({kx*1e9:.2f}, {ky*1e9:.2f}) nm"
        )

    # Generate spectroscopy points using keypoints directly
    print("\n⚡ Using keypoints as spectroscopy points...")
    spectroscopy_map = {}
    for i, mol in enumerate(selected_molecules):
        mol_class, x, y, conf, area, kx, ky = mol

        # Use keypoint as the primary spectroscopy point (optimal for spectroscopy)
        points = []
        # Point 1: At keypoint (this is the optimal spectroscopy location)
        points.append((kx, ky))

        # For high-confidence molecules, add molecule center as secondary point
        if conf > 0.85:
            points.append((x, y))

        spectroscopy_map[i] = points

    total_spectroscopy_points = sum(len(points) for points in spectroscopy_map.values())
    print(
        f"✅ Using keypoints as spectroscopy points: {total_spectroscopy_points} points for {len(selected_molecules)} molecules"
    )

    # Create comprehensive visualization
    print("\n🎨 Creating visualization...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

    # Colors for different classes
    class_colors = {
        1: "red",
        2: "blue",
        3: "green",
        4: "orange",
        5: "purple",
        6: "cyan",
        7: "magenta",
    }

    # Panel 1: Original image
    ax1.imshow(
        image,
        cmap="gray",
        extent=[
            -zoom_out_scale_nano / 2,
            zoom_out_scale_nano / 2,
            -zoom_out_scale_nano / 2,
            zoom_out_scale_nano / 2,
        ],
    )
    ax1.set_title("Original STM Image", fontsize=14, fontweight="bold")
    ax1.set_xlabel("X Position (nm)")
    ax1.set_ylabel("Y Position (nm)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: All detected molecules
    ax2.imshow(
        image,
        cmap="gray",
        extent=[
            -zoom_out_scale_nano / 2,
            zoom_out_scale_nano / 2,
            -zoom_out_scale_nano / 2,
            zoom_out_scale_nano / 2,
        ],
    )

    for i, mol in enumerate(shape_key_points_result):
        mol_class, x, y, conf, area, kx, ky = mol
        color = class_colors.get(mol_class, "yellow")
        alpha = (
            0.3 if mol not in selected_molecules else 1.0
        )  # Highlight selected molecules

        # Molecule center
        ax2.plot(x * 1e9, y * 1e9, "o", color=color, markersize=8, alpha=alpha)
        # Keypoint
        ax2.plot(kx * 1e9, ky * 1e9, "s", color=color, markersize=6, alpha=alpha)
        # Connection
        ax2.plot(
            [x * 1e9, kx * 1e9],
            [y * 1e9, ky * 1e9],
            "--",
            color=color,
            alpha=alpha * 0.5,
        )

        # Add number for selected molecules
        if mol in selected_molecules:
            selected_idx = selected_molecules.index(mol) + 1
            ax2.annotate(
                f"{selected_idx}",
                (x * 1e9, y * 1e9),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="circle", facecolor=color, alpha=0.8),
            )

    ax2.set_title(
        f"All Detected Molecules ({len(shape_key_points_result)} total, {len(selected_molecules)} selected)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("X Position (nm)")
    ax2.set_ylabel("Y Position (nm)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Selected molecules only
    ax3.imshow(
        image,
        cmap="gray",
        extent=[
            -zoom_out_scale_nano / 2,
            zoom_out_scale_nano / 2,
            -zoom_out_scale_nano / 2,
            zoom_out_scale_nano / 2,
        ],
    )

    for i, mol in enumerate(selected_molecules):
        mol_class, x, y, conf, area, kx, ky = mol
        color = class_colors.get(mol_class, "yellow")

        # Molecule center (larger marker)
        ax3.plot(
            x * 1e9,
            y * 1e9,
            "o",
            color=color,
            markersize=12,
            label=(
                f"Class {mol_class}"
                if i == 0 or mol_class not in [m[0] for m in selected_molecules[:i]]
                else ""
            ),
        )
        # Keypoint
        ax3.plot(kx * 1e9, ky * 1e9, "s", color=color, markersize=8)
        # Connection
        ax3.plot([x * 1e9, kx * 1e9], [y * 1e9, ky * 1e9], "--", color=color, alpha=0.6)

        # Number the molecules
        ax3.annotate(
            f"{i+1}",
            (x * 1e9, y * 1e9),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="circle", facecolor=color),
        )

    ax3.set_title(f"Selected Molecules for Analysis", fontsize=14, fontweight="bold")
    ax3.set_xlabel("X Position (nm)")
    ax3.set_ylabel("Y Position (nm)")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Spectroscopy points
    ax4.imshow(
        image,
        cmap="gray",
        extent=[
            -zoom_out_scale_nano / 2,
            zoom_out_scale_nano / 2,
            -zoom_out_scale_nano / 2,
            zoom_out_scale_nano / 2,
        ],
    )

    for i, mol in enumerate(selected_molecules):
        mol_class, x, y, conf, area, kx, ky = mol
        color = class_colors.get(mol_class, "yellow")

        # Molecule center (medium marker)
        ax4.plot(x * 1e9, y * 1e9, "o", color=color, markersize=8, alpha=0.6)
        # Keypoint
        ax4.plot(kx * 1e9, ky * 1e9, "s", color=color, markersize=6, alpha=0.6)

        # Spectroscopy points (highlighting that they're at keypoints)
        if i in spectroscopy_map:
            for j, point in enumerate(spectroscopy_map[i]):
                # Make spectroscopy points larger and more prominent since they're at keypoints
                ax4.plot(
                    point[0] * 1e9,
                    point[1] * 1e9,
                    "*",
                    color=color,
                    markersize=16,
                    markeredgecolor="white",
                    markeredgewidth=1,
                )
                # Number the spectroscopy points
                ax4.annotate(
                    f"S{j+1}",
                    (point[0] * 1e9, point[1] * 1e9),
                    xytext=(8, -18),
                    textcoords="offset points",
                    fontsize=10,
                    color="white",
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
                )

    # Add statistics
    stats_text = f"""Real Detection Results (Keypoints as Spectroscopy):
• Scan Area: {zoom_out_scale_nano:.1f} nm × {zoom_out_scale_nano:.1f} nm
• Total Molecules: {len(shape_key_points_result)}
• Selected: {len(selected_molecules)}
• Keypoint-based Spectroscopy Points: {total_spectroscopy_points}
• Classes: {len(set(mol[0] for mol in selected_molecules))}

Legend:
○ Molecule Centers  □ Keypoints (Primary)
★ Spectroscopy Points (at Keypoints)  ⋯ Connections"""

    ax4.text(
        0.02,
        0.98,
        stats_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax4.set_title(
        f"Keypoint-based Spectroscopy Points ({total_spectroscopy_points} total)",
        fontsize=14,
        fontweight="bold",
    )
    ax4.set_xlabel("X Position (nm)")
    ax4.set_ylabel("Y Position (nm)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save results
    os.makedirs("test_mark", exist_ok=True)

    # Save visualization
    viz_path = "test_mark/real_molecular_detection_result.png"
    plt.savefig(viz_path, dpi=200, bbox_inches="tight")
    print(f"✅ Real detection visualization saved to: {viz_path}")

    # Save data
    data = {
        "scan_parameters": {
            "scale": scan_zoom_in_scale,
            "scale_meters": zoom_out_scale,
            "scale_nm": zoom_out_scale_nano,
            "position": scan_position,
            "image_size": image.shape,
            "max_molecules_per_scan": max_molecules_per_scan,
        },
        "all_detected_molecules": [
            {
                "id": i + 1,
                "class": mol[0],
                "position_nm": [mol[1] * 1e9, mol[2] * 1e9],
                "confidence": mol[3],
                "area": mol[4],
                "keypoint_nm": [mol[5] * 1e9, mol[6] * 1e9],
                "selected": mol in selected_molecules,
            }
            for i, mol in enumerate(shape_key_points_result)
        ],
        "selected_molecules": [
            {
                "id": i + 1,
                "original_id": shape_key_points_result.index(mol) + 1,
                "class": mol[0],
                "position_nm": [mol[1] * 1e9, mol[2] * 1e9],
                "confidence": mol[3],
                "area": mol[4],
                "keypoint_nm": [mol[5] * 1e9, mol[6] * 1e9],
            }
            for i, mol in enumerate(selected_molecules)
        ],
        "spectroscopy_points": {
            str(mol_idx + 1): [[p[0] * 1e9, p[1] * 1e9] for p in points]
            for mol_idx, points in spectroscopy_map.items()
        },
        "summary": {
            "total_detected": len(shape_key_points_result),
            "total_selected": len(selected_molecules),
            "total_spectroscopy_points": total_spectroscopy_points,
            "classes_detected": list(set(mol[0] for mol in shape_key_points_result)),
            "classes_selected": list(set(mol[0] for mol in selected_molecules)),
        },
    }

    json_path = "test_mark/real_molecular_detection_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Real detection data saved to: {json_path}")

    plt.clf()
    plt.close()

    print(f"\n🎉 REAL MOLECULAR DETECTION COMPLETED!")
    print("=" * 70)
    print("Generated files:")
    print(f"  📊 Real Detection Visualization: {viz_path}")
    print(f"  📋 Real Detection Data: {json_path}")
    print(f"\nResults Summary:")
    print(f"  • Total molecules detected: {len(shape_key_points_result)}")
    print(f"  • Molecules selected for analysis: {len(selected_molecules)}")
    print(f"  • Total spectroscopy points: {total_spectroscopy_points}")
    print(f"  • Classes detected: {set(mol[0] for mol in shape_key_points_result)}")
    print("=" * 70)

else:
    print("❌ No molecules detected - check image path and model availability")
