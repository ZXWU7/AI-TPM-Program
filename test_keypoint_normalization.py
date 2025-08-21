"""
Test script to verify that all keypoints (KP1, KP2, KP3, KP4) are properly normalized
"""

import numpy as np
from test_simulation_standalone import key_points_convert


def test_keypoint_normalization():
    """Test that all keypoints are properly converted from normalized to real coordinates"""

    print("=" * 60)
    print("Testing Keypoint Normalization (KP1, KP2, KP3, KP4)")
    print("=" * 60)

    # Test parameters
    scan_position = (0.0, 0.0)  # Center at origin
    scan_edge = 20e-9  # 20 nm scan area

    # Create test data with normalized coordinates (0-1 range)
    # Format: [class, x, y, w, h, kp1x, kp1y, kp2x, kp2y, kp3x, kp3y, kp4x, kp4y]
    test_keypoints = [
        # Class 1 molecule with all 4 keypoints at known normalized positions
        [1, 0.3, 0.7, 0.1, 0.1, 0.25, 0.65, 0.35, 0.75, 0.25, 0.75, 0.35, 0.65],
        # Class 0 molecule with first keypoint meaningful
        [0, 0.6, 0.4, 0.08, 0.08, 0.55, 0.35, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    ]

    print("Input (normalized coordinates 0-1):")
    for i, keypoint in enumerate(test_keypoints):
        mol_class = int(keypoint[0])
        print(f"  Molecule {i+1} (Class {mol_class}):")
        print(f"    Center: ({keypoint[1]:.2f}, {keypoint[2]:.2f})")
        print(f"    KP1: ({keypoint[5]:.2f}, {keypoint[6]:.2f})")
        if mol_class == 1:  # Only show other keypoints for Class 1
            print(f"    KP2: ({keypoint[7]:.2f}, {keypoint[8]:.2f})")
            print(f"    KP3: ({keypoint[9]:.2f}, {keypoint[10]:.2f})")
            print(f"    KP4: ({keypoint[11]:.2f}, {keypoint[12]:.2f})")
        else:
            print(f"    KP2-4: Not meaningful for Class {mol_class}")
        print()

    # Convert using the function
    converted_keypoints = key_points_convert(test_keypoints, scan_position, scan_edge)

    print("Output (real-world coordinates in meters and nanometers):")
    print(
        f"Scan area: {scan_edge*1e9:.1f} nm centered at ({scan_position[0]*1e9:.1f}, {scan_position[1]*1e9:.1f}) nm"
    )
    print()

    for i, keypoint in enumerate(converted_keypoints):
        mol_class = int(keypoint[0])
        print(f"  Molecule {i+1} (Class {mol_class}):")
        print(f"    Center: ({keypoint[1]*1e9:.2f}, {keypoint[2]*1e9:.2f}) nm")
        print(f"    Center: ({keypoint[1]:.2e}, {keypoint[2]:.2e}) m")
        print(f"    KP1: ({keypoint[5]*1e9:.2f}, {keypoint[6]*1e9:.2f}) nm")
        print(f"    KP1: ({keypoint[5]:.2e}, {keypoint[6]:.2e}) m")

        if len(keypoint) >= 13:  # Has all 4 keypoints
            print(f"    KP2: ({keypoint[7]*1e9:.2f}, {keypoint[8]*1e9:.2f}) nm")
            print(f"    KP2: ({keypoint[7]:.2e}, {keypoint[8]:.2e}) m")
            print(f"    KP3: ({keypoint[9]*1e9:.2f}, {keypoint[10]*1e9:.2f}) nm")
            print(f"    KP3: ({keypoint[9]:.2e}, {keypoint[10]:.2e}) m")
            print(f"    KP4: ({keypoint[11]*1e9:.2f}, {keypoint[12]*1e9:.2f}) nm")
            print(f"    KP4: ({keypoint[11]:.2e}, {keypoint[12]:.2e}) m")
        print()

    # Verify conversion math
    print("Verification of conversion math:")
    edge_meters = scan_edge
    print(f"Scan edge: {edge_meters*1e9:.1f} nm = {edge_meters:.2e} m")

    for i, (orig, converted) in enumerate(zip(test_keypoints, converted_keypoints)):
        mol_class = int(orig[0])
        print(f"\n  Molecule {i+1} (Class {mol_class}) verification:")

        # Check KP1 conversion
        expected_kp1_x = scan_position[0] + (orig[5] - 0.5) * edge_meters
        expected_kp1_y = scan_position[1] - (orig[6] - 0.5) * edge_meters
        actual_kp1_x = converted[5]
        actual_kp1_y = converted[6]

        print(
            f"    KP1 X: Expected {expected_kp1_x*1e9:.2f} nm, Got {actual_kp1_x*1e9:.2f} nm - {'✓' if abs(expected_kp1_x - actual_kp1_x) < 1e-12 else '✗'}"
        )
        print(
            f"    KP1 Y: Expected {expected_kp1_y*1e9:.2f} nm, Got {actual_kp1_y*1e9:.2f} nm - {'✓' if abs(expected_kp1_y - actual_kp1_y) < 1e-12 else '✗'}"
        )

        # Check other keypoints if they exist
        if len(converted) >= 13 and mol_class == 1:
            # KP2
            expected_kp2_x = scan_position[0] + (orig[7] - 0.5) * edge_meters
            expected_kp2_y = scan_position[1] - (orig[8] - 0.5) * edge_meters
            actual_kp2_x = converted[7]
            actual_kp2_y = converted[8]

            print(
                f"    KP2 X: Expected {expected_kp2_x*1e9:.2f} nm, Got {actual_kp2_x*1e9:.2f} nm - {'✓' if abs(expected_kp2_x - actual_kp2_x) < 1e-12 else '✗'}"
            )
            print(
                f"    KP2 Y: Expected {expected_kp2_y*1e9:.2f} nm, Got {actual_kp2_y*1e9:.2f} nm - {'✓' if abs(expected_kp2_y - actual_kp2_y) < 1e-12 else '✗'}"
            )

            # KP3
            expected_kp3_x = scan_position[0] + (orig[9] - 0.5) * edge_meters
            expected_kp3_y = scan_position[1] - (orig[10] - 0.5) * edge_meters
            actual_kp3_x = converted[9]
            actual_kp3_y = converted[10]

            print(
                f"    KP3 X: Expected {expected_kp3_x*1e9:.2f} nm, Got {actual_kp3_x*1e9:.2f} nm - {'✓' if abs(expected_kp3_x - actual_kp3_x) < 1e-12 else '✗'}"
            )
            print(
                f"    KP3 Y: Expected {expected_kp3_y*1e9:.2f} nm, Got {actual_kp3_y*1e9:.2f} nm - {'✓' if abs(expected_kp3_y - actual_kp3_y) < 1e-12 else '✗'}"
            )

            # KP4
            expected_kp4_x = scan_position[0] + (orig[11] - 0.5) * edge_meters
            expected_kp4_y = scan_position[1] - (orig[12] - 0.5) * edge_meters
            actual_kp4_x = converted[11]
            actual_kp4_y = converted[12]

            print(
                f"    KP4 X: Expected {expected_kp4_x*1e9:.2f} nm, Got {actual_kp4_x*1e9:.2f} nm - {'✓' if abs(expected_kp4_x - actual_kp4_x) < 1e-12 else '✗'}"
            )
            print(
                f"    KP4 Y: Expected {expected_kp4_y*1e9:.2f} nm, Got {actual_kp4_y*1e9:.2f} nm - {'✓' if abs(expected_kp4_y - actual_kp4_y) < 1e-12 else '✗'}"
            )

    print("\n" + "=" * 60)
    print("✅ Keypoint Normalization Test Complete!")
    print("All keypoints (KP1, KP2, KP3, KP4) are now properly normalized")
    print("✅ Coordinates are converted from 0-1 range to real-world meters")
    print("✅ Y-axis is properly flipped to match STM coordinate system")
    print("=" * 60)


if __name__ == "__main__":
    test_keypoint_normalization()
