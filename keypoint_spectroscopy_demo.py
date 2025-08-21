"""
Focused test: Using Keypoints as Spectroscopy Points
This demonstrates the optimal approach where keypoints are used directly as spectroscopy measurement locations.
"""

import cv2
import numpy as np
import os
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_keypoint_spectroscopy_demo():
    print("🎯 KEYPOINT-BASED SPECTROSCOPY DEMONSTRATION")
    print("=" * 60)

    # Load test image
    img_path = "STM_img_simu/TPM_image/001.png"
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Could not load {img_path}")
        return

    image = cv2.resize(image, (304, 304), interpolation=cv2.INTER_AREA)
    scan_area_nm = 20.0  # 20 nm scan area

    # Example detected molecules with keypoints
    # Format: (class, mol_x, mol_y, confidence, area, keypoint_x, keypoint_y)
    detected_molecules = [
        (
            1,
            -4.2e-9,
            -3.1e-9,
            0.91,
            145,
            -4.0e-9,
            -2.9e-9,
        ),  # Keypoint closer to optimal spot
        (2, 3.5e-9, 4.2e-9, 0.88, 132, 3.7e-9, 4.0e-9),  # Keypoint at specific feature
        (1, -1.8e-9, 6.1e-9, 0.87, 156, -1.6e-9, 6.3e-9),  # Keypoint at molecular edge
        (3, 6.8e-9, -1.9e-9, 0.93, 171, 7.1e-9, -1.7e-9),  # Keypoint at reactive site
    ]

    print(f"📍 Detected {len(detected_molecules)} molecules with keypoints")

    # Use keypoints directly as spectroscopy points
    spectroscopy_points = []
    for i, mol in enumerate(detected_molecules):
        class_id, mx, my, conf, area, kx, ky = mol

        # The keypoint IS the optimal spectroscopy point
        spectroscopy_points.append(
            {
                "molecule_id": i + 1,
                "class": class_id,
                "molecule_pos": (mx, my),
                "spectroscopy_pos": (kx, ky),  # Keypoint position
                "confidence": conf,
                "distance_nm": np.sqrt((mx - kx) ** 2 + (my - ky) ** 2) * 1e9,
            }
        )

        print(
            f"  Molecule {i+1} (Class {class_id}): "
            f"Mol=({mx*1e9:.1f},{my*1e9:.1f})nm → "
            f"Spectroscopy=({kx*1e9:.1f},{ky*1e9:.1f})nm "
            f"[Δ={np.sqrt((mx-kx)**2+(my-ky)**2)*1e9:.1f}nm]"
        )

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    extent = [-scan_area_nm / 2, scan_area_nm / 2, -scan_area_nm / 2, scan_area_nm / 2]
    colors = ["red", "blue", "green", "orange", "purple"]

    # Panel 1: Original STM Image
    ax1.imshow(image, cmap="gray", extent=extent)
    ax1.set_title("Original STM Image", fontweight="bold", fontsize=12)
    ax1.set_xlabel("X Position (nm)")
    ax1.set_ylabel("Y Position (nm)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Detected Molecules with Keypoints
    ax2.imshow(image, cmap="gray", extent=extent)
    for i, mol in enumerate(detected_molecules):
        class_id, mx, my, conf, area, kx, ky = mol
        color = colors[class_id % len(colors)]

        # Molecule center
        ax2.plot(
            mx * 1e9,
            my * 1e9,
            "o",
            color=color,
            markersize=10,
            label=(
                f"Class {class_id}"
                if i == 0 or class_id not in [m[0] for m in detected_molecules[:i]]
                else ""
            ),
        )

        # Keypoint (highlighted as important)
        ax2.plot(
            kx * 1e9,
            ky * 1e9,
            "s",
            color=color,
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=2,
        )

        # Connection line
        ax2.plot(
            [mx * 1e9, kx * 1e9],
            [my * 1e9, ky * 1e9],
            "--",
            color=color,
            alpha=0.7,
            linewidth=2,
        )

        # Labels
        ax2.annotate(
            f"M{i+1}",
            (mx * 1e9, my * 1e9),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            color=color,
        )
        ax2.annotate(
            f"K{i+1}",
            (kx * 1e9, ky * 1e9),
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            color=color,
        )

    ax2.set_title("Molecules → Keypoints", fontweight="bold", fontsize=12)
    ax2.set_xlabel("X Position (nm)")
    ax2.set_ylabel("Y Position (nm)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Keypoints = Spectroscopy Points
    ax3.imshow(image, cmap="gray", extent=extent)
    for i, point in enumerate(spectroscopy_points):
        mol_id = point["molecule_id"]
        class_id = point["class"]
        mx, my = point["molecule_pos"]
        sx, sy = point["spectroscopy_pos"]  # Same as keypoint
        color = colors[class_id % len(colors)]

        # Show molecule (smaller, transparent)
        ax3.plot(mx * 1e9, my * 1e9, "o", color=color, markersize=8, alpha=0.4)

        # Spectroscopy point (large, prominent star at keypoint location)
        ax3.plot(
            sx * 1e9,
            sy * 1e9,
            "*",
            color=color,
            markersize=18,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

        # Connection (to show relationship)
        ax3.plot(
            [mx * 1e9, sx * 1e9], [my * 1e9, sy * 1e9], ":", color=color, alpha=0.5
        )

        # Label spectroscopy points
        ax3.annotate(
            f"S{mol_id}",
            (sx * 1e9, sy * 1e9),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
        )

    ax3.set_title("Keypoints as Spectroscopy Points", fontweight="bold", fontsize=12)
    ax3.set_xlabel("X Position (nm)")
    ax3.set_ylabel("Y Position (nm)")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Analysis Summary
    ax4.imshow(image, cmap="gray", extent=extent)

    # Plot everything with clear distinction
    for i, point in enumerate(spectroscopy_points):
        class_id = point["class"]
        mx, my = point["molecule_pos"]
        sx, sy = point["spectroscopy_pos"]
        color = colors[class_id % len(colors)]

        # Molecule center (small)
        ax4.plot(mx * 1e9, my * 1e9, "o", color=color, markersize=6, alpha=0.5)

        # Spectroscopy point (large star)
        ax4.plot(
            sx * 1e9,
            sy * 1e9,
            "*",
            color=color,
            markersize=16,
            markeredgecolor="yellow",
            markeredgewidth=2,
        )

        # Show precision circle around spectroscopy point
        circle = plt.Circle(
            (sx * 1e9, sy * 1e9),
            0.5,
            fill=False,
            color=color,
            linestyle="--",
            alpha=0.6,
        )
        ax4.add_patch(circle)

    # Add comprehensive statistics
    avg_distance = np.mean([p["distance_nm"] for p in spectroscopy_points])
    avg_confidence = np.mean([p["confidence"] for p in spectroscopy_points])

    stats_text = f"""Keypoint Spectroscopy Analysis:

✓ Spectroscopy Strategy: Use Keypoints Directly
✓ Total Molecules: {len(detected_molecules)}
✓ Spectroscopy Points: {len(spectroscopy_points)} (1:1 ratio)
✓ Avg Keypoint Offset: {avg_distance:.1f} nm
✓ Avg Confidence: {avg_confidence:.2f}

Advantages:
• Keypoints identify optimal measurement sites
• AI-determined reactive/important regions
• Sub-nanometer precision positioning
• Eliminates random point placement

Precision:
○ Molecule Centers (detected)
★ Spectroscopy Points (at keypoints)
⊙ ±0.5nm precision circles"""

    ax4.text(
        0.02,
        0.98,
        stats_text,
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )
    ax4.set_title("Strategic Spectroscopy Positioning", fontweight="bold", fontsize=12)
    ax4.set_xlabel("X Position (nm)")
    ax4.set_ylabel("Y Position (nm)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save results
    os.makedirs("test_mark", exist_ok=True)

    # Save visualization
    viz_path = "test_mark/keypoint_spectroscopy_strategy.png"
    plt.savefig(viz_path, dpi=200, bbox_inches="tight")
    print(f"✅ Keypoint spectroscopy visualization saved: {viz_path}")

    # Save detailed data
    data = {
        "strategy": "keypoints_as_spectroscopy_points",
        "scan_parameters": {
            "area_nm": scan_area_nm,
            "image_size": image.shape,
            "total_molecules": len(detected_molecules),
        },
        "spectroscopy_points": [
            {
                "molecule_id": p["molecule_id"],
                "class": p["class"],
                "molecule_position_nm": [
                    p["molecule_pos"][0] * 1e9,
                    p["molecule_pos"][1] * 1e9,
                ],
                "spectroscopy_position_nm": [
                    p["spectroscopy_pos"][0] * 1e9,
                    p["spectroscopy_pos"][1] * 1e9,
                ],
                "keypoint_offset_nm": p["distance_nm"],
                "confidence": p["confidence"],
            }
            for p in spectroscopy_points
        ],
        "analysis": {
            "average_keypoint_offset_nm": avg_distance,
            "average_confidence": avg_confidence,
            "precision_radius_nm": 0.5,
            "total_spectroscopy_points": len(spectroscopy_points),
        },
    }

    json_path = "test_mark/keypoint_spectroscopy_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Keypoint spectroscopy data saved: {json_path}")

    plt.clf()
    plt.close()

    print("\n🎉 KEYPOINT SPECTROSCOPY DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("Key Findings:")
    print(f"  • {len(spectroscopy_points)} optimal spectroscopy points identified")
    print(f"  • Average keypoint precision: {avg_distance:.1f} nm from molecule center")
    print(f"  • Average detection confidence: {avg_confidence:.1f}%")
    print("  • Strategy: Use AI-determined keypoints as measurement sites")
    print("  • Benefit: Eliminates guesswork in spectroscopy positioning")
    print("=" * 60)


if __name__ == "__main__":
    create_keypoint_spectroscopy_demo()
