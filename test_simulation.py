"""
Updated test script using standalone functions instead of AI_Nanonis_Spectroscopy object
"""

import cv2
import numpy as np
import os
import json
from test_simulation_standalone import (
    convert_si_prefix,
    molecular_seeker_standalone,
    auto_select_molecules_standalone,
    visualize_molecules_on_image,
    visualize_spectroscopy_points,
)

# Configuration - equivalent to original nanonis object settings
img_simu_path = "AI_TPM/STM_img_simu/TPM_image/052.png"
scan_zoom_in_scale = "20n"  # Equivalent to nanonis.scan_zoom_in_list[0]
scan_position = (0e-9, 0e-9)  # Equivalent to nanonis.nanocoodinate
max_molecules_per_scan = 3

# Convert scan scale (equivalent to nanonis.convert())
zoom_out_scale = convert_si_prefix(scan_zoom_in_scale)
zoom_out_scale_nano = zoom_out_scale * 10**9

print(
    f"Scan scale: {scan_zoom_in_scale} = {zoom_out_scale} m = {zoom_out_scale_nano} nm"
)

# Load and prepare image (equivalent to nanonis.image_for)
image = cv2.imread(img_simu_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Error: Could not load image from {img_simu_path}")
    exit(1)

image = cv2.resize(image, (304, 304), interpolation=cv2.INTER_AREA)
print(f"Image loaded and resized to: {image.shape}")

# Run molecular detection (equivalent to nanonis.molecular_seeker())
shape_key_points_result = molecular_seeker_standalone(
    image,
    scan_position=scan_position,
    scan_edge=zoom_out_scale_nano,
    keypoint_model_path="AI_TPM/keypoint/best_0820.pt",
    save_dir="AI_TPM/test_output",
    molecular_filter_threshold=1e-9,
)

if shape_key_points_result is not None:
    print(
        f"shape_key_points_result_shape: {len(shape_key_points_result)} molecules detected"
    )

    # Test intelligent molecule selection
    selected_molecules = auto_select_molecules_standalone(
        shape_key_points_result,
        max_molecules_per_scan,
        "intelligent",  # Can be: "intelligent", "closest", "quality"
        scan_position,
    )

    print(f"Selected {len(selected_molecules)} molecules for processing")

    # Print results with original indices and class-specific keypoint information
    for i, molecule in enumerate(selected_molecules):
        # Find the original index of this molecule in the complete detection results
        original_index = -1
        for j, original_mol in enumerate(shape_key_points_result):
            # Compare coordinates to find matching molecule (with small tolerance for floating point)
            if (
                abs(molecule[1] - original_mol[1]) < 1e-12
                and abs(molecule[2] - original_mol[2]) < 1e-12
            ):
                original_index = j
                break

        mol_class = int(molecule[0])
        print(
            f"Original Molecule Index: {original_index+1}: "
            f"class={mol_class}, "
            f"pos=({molecule[1]*1e9:.2f}, {molecule[2]*1e9:.2f}) nm"
        )

        # Print keypoints based on molecule class
        if mol_class == 1:
            # Class 1: Print all 4 meaningful keypoints
            keypoint_labels = ["KP1", "KP2", "KP3", "KP4"]
            num_keypoints = 4
            print("    Class 1 molecule - All 4 keypoints are meaningful:")
        else:
            # Class 0 and 2: Only print the first keypoint (only kp1 is meaningful)
            keypoint_labels = ["KP1"]
            num_keypoints = 1
            print(
                f"    Class {mol_class} molecule - Only first keypoint is meaningful:"
            )

        for kp_idx in range(num_keypoints):
            x_idx = 5 + (kp_idx * 2)
            y_idx = 6 + (kp_idx * 2)

            if x_idx < len(molecule) and y_idx < len(molecule):
                kp_x, kp_y = molecule[x_idx], molecule[y_idx]
                print(
                    f"    {keypoint_labels[kp_idx]}: ({kp_x*1e9:.2f}, {kp_y*1e9:.2f}) nm"
                )
            else:
                print(f"    {keypoint_labels[kp_idx]}: Not available")

        # Show remaining keypoints as "meaningless" for classes 0 and 2
        if mol_class != 1 and len(molecule) > 7:
            print("    Meaningless keypoints (KP2-KP4): present but not reliable")

        print()  # Empty line for readability

    # Visualize the results
    print("\nGenerating visualization...")
    visualize_molecules_on_image(
        image=image,
        all_molecules=shape_key_points_result,
        selected_molecules=selected_molecules,
        scan_position=scan_position,
        scan_edge=zoom_out_scale_nano,
        save_path="AI_TPM/test_mark/molecule_detection_result.png",
        show_plot=True,
    )

    # Use natural keypoints from molecular detection for spectroscopy
    print("\n" + "=" * 60)
    print("SPECTROSCOPY POINTS FROM MOLECULAR DETECTION")
    print("=" * 60)

    # Extract keypoints directly from detected molecules (no additional generation needed)
    print("\nUsing keypoints naturally returned by molecular_seeker_standalone()...")
    spectroscopy_map = {}

    for mol_idx, molecule in enumerate(selected_molecules):
        # Extract keypoints from molecule detection results based on class
        # Format: [class, x, y, w, h, kp1x, kp1y, kp2x, kp2y, kp3x, kp3y, kp4x, kp4y]
        mol_center = (molecule[1], molecule[2])  # Molecule center position
        mol_class = int(molecule[0])

        # Extract meaningful keypoints based on molecule class
        keypoints = []

        if mol_class == 1:
            # Class 1: All 4 keypoints are meaningful
            for kp_idx in range(4):
                x_idx = 5 + (kp_idx * 2)
                y_idx = 6 + (kp_idx * 2)

                if x_idx < len(molecule) and y_idx < len(molecule):
                    keypoints.append((molecule[x_idx], molecule[y_idx]))
            print(
                f"Molecule {mol_idx+1} (Class 1): Found {len(keypoints)} meaningful keypoints"
            )
        else:
            # Class 0 and 2: Only the first keypoint is meaningful
            if len(molecule) >= 7:  # Ensure kp1x, kp1y exist
                keypoints.append((molecule[5], molecule[6]))  # Only kp1x, kp1y
            print(
                f"Molecule {mol_idx+1} (Class {mol_class}): Found {len(keypoints)} meaningful keypoint"
            )

        # Use meaningful keypoints as spectroscopy points (limit to max 2 for this demo)
        spectroscopy_points = keypoints[:2] if len(keypoints) >= 2 else keypoints

        spectroscopy_map[mol_idx] = spectroscopy_points

    if spectroscopy_map:
        print(f"\nSpectroscopy points extracted from {len(spectroscopy_map)} molecules")

        # Print detailed spectroscopy points
        total_points = 0
        for mol_idx, points in spectroscopy_map.items():
            mol_info = selected_molecules[mol_idx]
            print(
                f"  Molecule {mol_idx+1} (class {mol_info[0]}): {len(points)} spectroscopy points"
            )

            for point_idx, point in enumerate(points):
                print(
                    f"    Keypoint {point_idx+1}: ({point[0]*1e9:.2f}, {point[1]*1e9:.2f}) nm"
                )
                total_points += 1

        print(f"\nTotal spectroscopy points available: {total_points}")
        print("These points are ready for actual spectroscopy measurements.")

        # Create spectroscopy visualization (without simulation results)
        print("\nGenerating spectroscopy visualization...")
        visualize_spectroscopy_points(
            image=image,
            selected_molecules=selected_molecules,
            spectroscopy_map=spectroscopy_map,
            spectroscopy_results=None,  # No simulation results
            scan_position=scan_position,
            scan_edge=zoom_out_scale,  # In meters
            save_path="AI_TPM/test_mark/spectroscopy_points_result.png",
            show_plot=True,
        )

        # Save spectroscopy points data to JSON for reference
        import json
        import time

        spectroscopy_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scan_info": {
                "position": scan_position,
                "edge_size": zoom_out_scale,
            },
            "molecules": [],
            "summary": {
                "total_molecules": len(selected_molecules),
                "total_spectroscopy_points": total_points,
                "points_per_molecule": [
                    len(points) for points in spectroscopy_map.values()
                ],
            },
        }

        for mol_idx, points in spectroscopy_map.items():
            mol_info = selected_molecules[mol_idx]
            mol_class = int(mol_info[0])

            # Extract meaningful keypoints for this molecule based on class
            meaningful_keypoints = []
            if mol_class == 1:
                # Class 1: All 4 keypoints are meaningful
                for kp_idx in range(4):
                    x_idx = 5 + (kp_idx * 2)
                    y_idx = 6 + (kp_idx * 2)

                    if x_idx < len(mol_info) and y_idx < len(mol_info):
                        meaningful_keypoints.append([mol_info[x_idx], mol_info[y_idx]])
            else:
                # Class 0 and 2: Only first keypoint is meaningful
                if len(mol_info) >= 7:
                    meaningful_keypoints.append([mol_info[5], mol_info[6]])

            # Also extract all raw keypoints for reference
            all_raw_keypoints = []
            for kp_idx in range(4):
                x_idx = 5 + (kp_idx * 2)
                y_idx = 6 + (kp_idx * 2)

                if x_idx < len(mol_info) and y_idx < len(mol_info):
                    all_raw_keypoints.append([mol_info[x_idx], mol_info[y_idx]])

            mol_data = {
                "molecule_index": mol_idx,
                "molecule_class": mol_class,
                "molecule_position": [mol_info[1], mol_info[2]],
                "meaningful_keypoints": meaningful_keypoints,  # Only class-appropriate keypoints
                "all_raw_keypoints": all_raw_keypoints,  # All 4 keypoints for reference
                "keypoint_reliability": {
                    "class_1_has_4_meaningful": mol_class == 1,
                    "class_0_2_has_1_meaningful": mol_class in [0, 2],
                    "meaningful_count": len(meaningful_keypoints),
                },
                "spectroscopy_points": points,  # Selected keypoints for spectroscopy
                "ready_for_measurement": True,
            }
            spectroscopy_data["molecules"].append(mol_data)

        # Save to JSON file
        json_path = "AI_TPM/test_mark/spectroscopy_points.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(spectroscopy_data, f, indent=2, default=str)

        print(f"Spectroscopy points data saved to: {json_path}")

    else:
        print("No spectroscopy points available")

else:
    print("No molecules detected - skipping spectroscopy point extraction")

print("\n" + "=" * 60)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("Files generated:")
print("  - AI_TPM/test_mark/molecule_detection_result.png (molecule detection)")
if "spectroscopy_map" in locals() and spectroscopy_map:
    print(
        "  - AI_TPM/test_mark/spectroscopy_points_result.png (spectroscopy visualization)"
    )
    print("  - AI_TPM/test_mark/spectroscopy_points.json (spectroscopy points data)")
print("\nThis test demonstrates the complete workflow:")
print("  1. Molecule detection from STM images")
print("  2. Intelligent molecule selection")
print("  3. Natural keypoint extraction for spectroscopy")
print("  4. Spectroscopy points visualization")
print("  5. Data export for real measurements")
print("=" * 60)
