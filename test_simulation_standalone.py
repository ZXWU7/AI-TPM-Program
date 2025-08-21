"""
Standalone test script for molecule detection without full AI_Nanonis_Spectroscopy object
This script can test molecular seeking functionality using just image processing.
"""

import cv2
import numpy as np
import os
import re
import matplotlib

matplotlib.use("TkAgg")  # Set interactive backend before importing pyplot
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# Import required detection functions
try:
    from keypoint.detect import key_detect
    from utils import filter_close_bboxes
except ImportError as e:
    print(f"Warning: Could not import detection functions: {e}")
    print("Please ensure keypoint and utils modules are available")


def generate_spectroscopy_points_standalone(
    selected_molecules,
    scan_position=(0e-9, 0e-9),
    scan_edge=20e-9,
    points_per_molecule=2,
    use_keypoints=True,
    keypoints_model_path="AI_TPM/keypoints_model/best_0417.pt",
    image=None,
):
    """
    Generate spectroscopy points for selected molecules using AI model or keypoints.

    Args:
        selected_molecules: List of selected molecule tuples
        scan_position: Scan center position in meters
        scan_edge: Scan edge size in meters
        points_per_molecule: Maximum number of spectroscopy points per molecule
        use_keypoints: Whether to use AI keypoints model or molecule keypoints
        keypoints_model_path: Path to the AI keypoints model
        image: STM image for AI detection

    Returns:
        dict: Mapping of molecule_index -> list of spectroscopy points (x, y) in meters
    """
    try:
        spectroscopy_map = {}

        if use_keypoints and image is not None:
            print("Using AI keypoints model for spectroscopy point generation...")

            # Use AI model to detect interest points
            try:
                key_points_result = key_detect(
                    image, keypoints_model_path, "AI_TPM/test_output/keypoints"
                )

                if key_points_result:
                    # Convert detected keypoints to absolute coordinates
                    all_interest_points = []

                    for detection in key_points_result:
                        # Convert detection coordinates relative to scan
                        for i in range(
                            5, len(detection), 2
                        ):  # Keypoints start at index 5
                            if i + 1 < len(detection):
                                # Convert from pixel to real coordinates
                                rel_x = (
                                    detection[i] - 152
                                ) / 304  # Normalize to -0.5 to 0.5
                                rel_y = (detection[i + 1] - 152) / 304

                                # Convert to absolute position
                                abs_x = scan_position[0] + rel_x * scan_edge
                                abs_y = scan_position[1] + rel_y * scan_edge

                                all_interest_points.append((abs_x, abs_y))

                    print(f"AI detected {len(all_interest_points)} interest points")

                    # Map interest points to molecules
                    for mol_idx, molecule in enumerate(selected_molecules):
                        mol_center = (molecule[1], molecule[2])  # Position in meters
                        molecule_points = []
                        search_radius = 3e-9  # 3 nm search radius

                        # Find nearby interest points
                        for point in all_interest_points:
                            distance = np.sqrt(
                                (point[0] - mol_center[0]) ** 2
                                + (point[1] - mol_center[1]) ** 2
                            )
                            if distance <= search_radius:
                                molecule_points.append(point)

                        # Select best points
                        if molecule_points:
                            molecule_points.sort(
                                key=lambda p: np.sqrt(
                                    (p[0] - mol_center[0]) ** 2
                                    + (p[1] - mol_center[1]) ** 2
                                )
                            )
                            max_points = min(points_per_molecule, len(molecule_points))
                            spectroscopy_map[mol_idx] = molecule_points[:max_points]
                        else:
                            # Fallback to molecule center
                            spectroscopy_map[mol_idx] = [mol_center]

                else:
                    print("AI model found no keypoints, using molecule keypoints...")
                    use_keypoints = False  # Fall back to molecule keypoints

            except Exception as e:
                print(f"AI keypoints detection failed: {e}")
                use_keypoints = False  # Fall back to molecule keypoints

        if not use_keypoints:
            print("Using molecule detection keypoints for spectroscopy points...")

            # Use keypoints from molecule detection results
            for mol_idx, molecule in enumerate(selected_molecules):
                points = []

                # Extract keypoints from molecule detection (indices 5, 6, 7, 8, ...)
                if len(molecule) > 5:
                    keypoints_available = (len(molecule) - 5) // 2
                    num_points = min(points_per_molecule, keypoints_available)

                    for i in range(num_points):
                        kp_x_idx = 5 + (i * 2)
                        kp_y_idx = 6 + (i * 2)

                        if kp_y_idx < len(molecule):
                            point_x = molecule[kp_x_idx]  # Already in meters
                            point_y = molecule[kp_y_idx]  # Already in meters
                            points.append((point_x, point_y))

                # Fallback to molecule center if no keypoints
                if not points:
                    mol_center = (molecule[1], molecule[2])
                    points.append(mol_center)

                spectroscopy_map[mol_idx] = points

        # Print summary
        total_points = sum(len(points) for points in spectroscopy_map.values())
        print(
            f"Generated {total_points} spectroscopy points for {len(spectroscopy_map)} molecules"
        )

        return spectroscopy_map

    except Exception as e:
        print(f"Error in generate_spectroscopy_points_standalone: {e}")
        return {}


def perform_spectroscopy_simulation(
    spectroscopy_map,
    selected_molecules,
    simulation_success_rate=0.8,
    bias_range=(-2.0, 2.0),
    measurement_points=512,
):
    """
    Simulate spectroscopy measurements for testing purposes.

    Args:
        spectroscopy_map: Dict mapping molecule_index -> spectroscopy points
        selected_molecules: List of selected molecules
        simulation_success_rate: Probability of successful measurement
        bias_range: Voltage range for spectroscopy
        measurement_points: Number of measurement points

    Returns:
        dict: Mapping of molecule_index -> list of measurement results
    """
    try:
        results = {}

        for mol_idx, points in spectroscopy_map.items():
            molecule_results = []

            for point_idx, point in enumerate(points):
                # Simulate measurement
                success = np.random.random() < simulation_success_rate

                if success:
                    # Generate simulated spectroscopy data
                    bias_values = np.linspace(
                        bias_range[0], bias_range[1], measurement_points
                    )
                    current_values = (
                        np.random.random(measurement_points) * 1e-9
                    )  # Random current data

                    result = {
                        "success": True,
                        "position": point,
                        "bias_range": bias_range,
                        "measurement_points": measurement_points,
                        "data": {
                            "bias": bias_values.tolist(),
                            "current": current_values.tolist(),
                        },
                        "channels": ["Bias(V)", "Current(A)"],
                        "measurement_type": "bias_spectroscopy_simulation",
                    }
                else:
                    result = {
                        "success": False,
                        "position": point,
                        "error": "Simulated measurement failure",
                        "measurement_type": "bias_spectroscopy_simulation",
                    }

                molecule_results.append(result)

                # Print result
                mol_pos = selected_molecules[mol_idx][1:3]  # Get molecule position
                status = "SUCCESS" if success else "FAILED"
                print(
                    f"  Point {point_idx+1} at ({point[0]*1e9:.1f}, {point[1]*1e9:.1f}) nm: {status}"
                )

            results[mol_idx] = molecule_results

        return results

    except Exception as e:
        print(f"Error in perform_spectroscopy_simulation: {e}")
        return {}


def visualize_spectroscopy_points(
    image,
    selected_molecules,
    spectroscopy_map,
    spectroscopy_results=None,
    scan_position=(0, 0),
    scan_edge=20e-9,
    save_path="AI_TPM/test_mark/spectroscopy_points.png",
    show_plot=True,
):
    """
    Visualize spectroscopy points on the STM image.

    Args:
        image: STM image
        selected_molecules: List of selected molecules
        spectroscopy_map: Dict mapping molecule_index -> spectroscopy points
        spectroscopy_results: Dict mapping molecule_index -> measurement results
        scan_position: Scan center position in meters
        scan_edge: Scan edge size in meters
        save_path: Path to save the visualization
        show_plot: Whether to show the plot
    """
    try:
        # Create color image for visualization
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image.copy()

        image_height, image_width = vis_image.shape[:2]

        def real_to_pixel_coords(real_x: float, real_y: float):
            """Convert real-world coordinates to pixel coordinates"""
            # Normalize to 0-1 based on scan area
            # CORRECTED: Use negative sign for Y to properly reverse the coordinate conversion
            norm_x = (real_x - scan_position[0]) / scan_edge + 0.5
            norm_y = (
                -(real_y - scan_position[1]) / scan_edge + 0.5
            )  # FIXED: Added negative sign

            # Convert to pixel coordinates
            pixel_x = int(norm_x * image_width)
            pixel_y = int(
                norm_y * image_height
            )  # SIMPLIFIED: No need for (1.0 - norm_y) anymore

            # Ensure within bounds
            pixel_x = max(0, min(image_width - 1, pixel_x))
            pixel_y = max(0, min(image_height - 1, pixel_y))

            return pixel_x, pixel_y

        # Draw molecules
        for mol_idx, molecule in enumerate(selected_molecules):
            mol_x, mol_y = molecule[1], molecule[2]  # Position in meters
            mol_px, mol_py = real_to_pixel_coords(mol_x, mol_y)

            # Draw molecule center with smaller markers
            cv2.circle(
                vis_image, (mol_px, mol_py), 3, (255, 100, 100), 1  # Even smaller
            )  # Blue circle, smaller

            # Calculate text position with better overlap avoidance
            mol_label = f"M{mol_idx+1}"
            text_width = len(mol_label) * 6  # Estimate text width
            text_height = 10  # Estimate text height

            # Try different positions around the molecule
            positions = [
                (mol_px + 5, mol_py - 5),  # top-right
                (mol_px - text_width - 5, mol_py - 5),  # top-left
                (mol_px + 5, mol_py + text_height + 5),  # bottom-right
                (mol_px - text_width - 5, mol_py + text_height + 5),  # bottom-left
            ]

            # Choose the first position that fits
            text_x, text_y = positions[0]  # default
            for pos_x, pos_y in positions:
                if (
                    5 <= pos_x <= image_width - text_width - 5
                    and text_height + 5 <= pos_y <= image_height - 5
                ):
                    text_x, text_y = pos_x, pos_y
                    break

            # Remove molecule text labels - only keep legend
            # cv2.putText(
            #     vis_image,
            #     mol_label,
            #     (text_x, text_y),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.3,  # Smaller font
            #     (255, 255, 255),
            #     1,
            # )

        # Draw spectroscopy points
        for mol_idx, points in spectroscopy_map.items():
            for point_idx, point in enumerate(points):
                spec_px, spec_py = real_to_pixel_coords(point[0], point[1])

                # Determine color based on results with smaller markers
                if spectroscopy_results and mol_idx in spectroscopy_results:
                    results = spectroscopy_results[mol_idx]
                    if point_idx < len(results):
                        success = results[point_idx].get("success", False)
                        color = (
                            (0, 255, 0) if success else (0, 165, 255)
                        )  # Green/Orange
                        marker_size = 4 if success else 3
                    else:
                        color = (128, 128, 128)  # Gray
                        marker_size = 3
                else:
                    color = (0, 255, 255)  # Yellow for planned points
                    marker_size = 3

                # Draw spectroscopy point with smaller marker
                cv2.circle(
                    vis_image, (spec_px, spec_py), marker_size - 1, color, -1
                )  # Even smaller

                # Calculate text position with better overlap avoidance
                spec_label = f"S{point_idx+1}"
                text_width = len(spec_label) * 5  # Estimate text width
                text_height = 10  # Estimate text height

                # Try different positions around the point
                positions = [
                    (spec_px + 4, spec_py - 4),  # top-right
                    (spec_px - text_width - 4, spec_py - 4),  # top-left
                    (spec_px + 4, spec_py + text_height + 4),  # bottom-right
                    (
                        spec_px - text_width - 4,
                        spec_py + text_height + 4,
                    ),  # bottom-left
                ]

                # Choose the first position that fits
                text_x, text_y = positions[0]  # default
                for pos_x, pos_y in positions:
                    if (
                        5 <= pos_x <= image_width - text_width - 5
                        and text_height + 5 <= pos_y <= image_height - 5
                    ):
                        text_x, text_y = pos_x, pos_y
                        break

                # Remove spectroscopy point text labels - only keep legend
                # cv2.putText(
                #     vis_image,
                #     spec_label,
                #     (text_x, text_y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.25,  # Very small font
                #     color,
                #     1,
                # )

        # Add scan center marker with black color and thinner cross
        center_px, center_py = real_to_pixel_coords(scan_position[0], scan_position[1])
        cv2.drawMarker(
            vis_image, (center_px, center_py), (0, 0, 0), cv2.MARKER_CROSS, 8, 1
        )

        # Save image with color correction
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert BGR to RGB to fix color inversion when saving
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR))
        print(f"Spectroscopy visualization saved to: {save_path}")

        # Show plot if requested with enhanced formatting
        if show_plot:
            plt.figure(figsize=(8, 6))  # Further reduced for better legend proportion
            # Convert BGR to RGB to fix color inversion
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            plt.imshow(vis_image_rgb)
            plt.title(
                "Spectroscopy Points Visualization\n"
                f"Yellow: Planned | Green: Success | Orange: Failed | Gray: Unknown",
                fontsize=10,  # Smaller font for smaller figure
                pad=10,
            )

            # Add enhanced legend
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="lightcoral",
                    markersize=12,
                    label="Molecules",
                    markeredgecolor="black",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="yellow",
                    markersize=10,
                    label="Planned Spectroscopy Points",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="lime",
                    markersize=10,
                    label="Successful Measurements",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="orange",
                    markersize=10,
                    label="Failed Measurements",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="+",
                    color="black",
                    markersize=12,
                    label="Scan Center",
                ),
            ]
            plt.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=9,  # Smaller font for smaller figure
                markerscale=1.0,
                frameon=True,
                fancybox=True,
                shadow=True,
            )

            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return vis_image

    except Exception as e:
        print(f"Error in visualize_spectroscopy_points: {e}")
        return None


def convert_si_prefix(input_data: str) -> float:
    """
    Converts a number followed by an SI prefix into number * 10^{prefix exponent}

    Args:
        input_data : str (e.g., "20n", "1u", "5m")

    Returns:
        Float value in base SI units
    """
    si_prefix = {
        "y": 1e-24,  # yocto
        "z": 1e-21,  # zepto
        "a": 1e-18,  # atto
        "f": 1e-15,  # femto
        "p": 1e-12,  # pico
        "n": 1e-9,  # nano
        "u": 1e-6,  # micro
        "m": 1e-3,  # milli
        "": 1,  # no prefix
        "k": 1e3,  # kilo
        "M": 1e6,  # mega
        "G": 1e9,  # giga
        "T": 1e12,  # tera
        "P": 1e15,  # peta
        "E": 1e18,  # exa
        "Z": 1e21,  # zetta
        "Y": 1e24,  # yotta
    }

    regex = re.compile(r"^(-)?([0-9.]+)\s*([A-Za-z]*)$")
    match = regex.match(str(input_data))

    if match is None:
        raise ValueError(
            f"Malformed number: {input_data} is not a correctly formatted number"
        )

    groups = match.groups()
    sign = -1 if groups[0] == "-" else 1

    try:
        return sign * float(groups[1]) * si_prefix[groups[2]]
    except KeyError:
        raise ValueError(f"Unknown SI prefix: {groups[2]}")


def key_points_convert(
    key_points_result: List,
    scan_position: Tuple[float, float],
    scan_edge: float,
) -> List:
    """
    Convert normalized keypoint coordinates to real world coordinates

    Args:
        key_points_result: List of detected keypoints [[class, x, y, w, h, key_x, key_y], ...]
        scan_position: Center position of scan in meters (x, y)
        scan_edge: Edge length of scan area in nanometers

    Returns:
        List of converted keypoints with real coordinates
    """
    if not key_points_result or len(key_points_result) == 0:
        return []

    # Convert scan_edge from nm to meters if it's a number
    if isinstance(scan_edge, (int, float)):
        edge_meters = scan_edge * 1e-9
    else:
        # Handle string format like "20n"
        edge_meters = convert_si_prefix(str(scan_edge))

    converted_points = []

    for keypoint in key_points_result:
        if len(keypoint) < 7:
            continue  # Skip malformed keypoints

        mol_class, norm_x, norm_y, norm_w, norm_h, key_x, key_y = keypoint[:7]

        # Convert normalized coordinates (0-1) to real world coordinates
        # Assuming image coordinates: (0,0) is top-left, (1,1) is bottom-right
        # Real world coordinates: scan_position is center

        real_x = scan_position[0] + (norm_x - 0.5) * edge_meters
        real_y = scan_position[1] - (norm_y - 0.5) * edge_meters
        real_w = norm_w * edge_meters
        real_h = norm_h * edge_meters
        real_key_x = scan_position[0] + (key_x - 0.5) * edge_meters
        real_key_y = scan_position[1] - (key_y - 0.5) * edge_meters

        converted_keypoint = [
            mol_class,
            real_x,
            real_y,
            real_w,
            real_h,
            real_key_x,
            real_key_y,
        ]

        # Convert additional keypoints (KP2, KP3, KP4) from normalized to real coordinates
        if len(keypoint) > 7:
            additional_keypoints = keypoint[7:]
            # Process keypoints in pairs (x, y)
            for i in range(0, len(additional_keypoints), 2):
                if i + 1 < len(additional_keypoints):
                    # Convert normalized keypoint coordinates to real world coordinates
                    norm_kp_x = additional_keypoints[i]
                    norm_kp_y = additional_keypoints[i + 1]

                    real_kp_x = scan_position[0] + (norm_kp_x - 0.5) * edge_meters
                    real_kp_y = scan_position[1] - (norm_kp_y - 0.5) * edge_meters

                    converted_keypoint.extend([real_kp_x, real_kp_y])
                else:
                    # If there's an odd number, just append the last value (shouldn't happen normally)
                    converted_keypoint.append(additional_keypoints[i])

        converted_points.append(converted_keypoint)

    return converted_points


def molecular_seeker_standalone(
    image,
    scan_position,
    scan_edge,
    keypoint_model_path,
    save_dir,
    molecular_filter_threshold,
):
    """
    Standalone molecular seeker function that doesn't require AI_Nanonis_Spectroscopy object

    Args:
        image: Input STM image (grayscale)
        scan_position: Center position of scan in meters (x, y)
        scan_edge: Edge length of scan area in nanometers or SI string format
        keypoint_model_path: Path to keypoint detection model
        save_dir: Directory to save detection results
        molecular_filter_threshold: Threshold for filtering close bounding boxes

    Returns:
        List of detected molecules with real world coordinates or None if no detection
    """
    print(f"Seeking molecules in image shape: {image.shape}")
    print(f"Scan position: {scan_position}, Scan edge: {scan_edge}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created save directory: {save_dir}")

    try:
        # Run keypoint detection
        print("Running keypoint detection...")
        key_points_result = key_detect(image, keypoint_model_path, save_dir)
        print(f"Detected {len(key_points_result)} raw keypoints")

        if len(key_points_result) == 0:
            print("No keypoints detected")
            return None

        # Convert normalized coordinates to real world coordinates
        print("Converting coordinates to real world...")
        molecular_list = key_points_convert(
            key_points_result, scan_position=scan_position, scan_edge=scan_edge
        )
        print(f"Converted {len(molecular_list)} molecular coordinates")

        # Filter close bounding boxes
        print("Filtering close molecules...")
        try:
            molecular_list = filter_close_bboxes(
                molecular_list, molecular_filter_threshold
            )
            print(f"After filtering: {len(molecular_list)} molecules remain")
        except NameError:
            print("Warning: filter_close_bboxes not available, skipping filtering")
        except Exception as e:
            print(f"Warning: Error in filtering: {e}, skipping filtering")

        return molecular_list

    except Exception as e:
        print(f"Error in molecular_seeker_standalone: {e}")
        import traceback

        traceback.print_exc()
        return None


def auto_select_molecules_standalone(
    shape_key_points_result: List,
    max_molecules_per_scan: int = 3,
    selection_strategy: str = "intelligent",
    scan_center: Tuple[float, float] = (0.0, 0.0),
) -> List:
    """
    Standalone intelligent molecule selection function

    Args:
        shape_key_points_result: List of detected molecules
        max_molecules_per_scan: Maximum number of molecules to select
        selection_strategy: Strategy for selection ("intelligent", "closest", "quality", "distance_spread", "random")
        scan_center: Center of scan area for distance calculations

    Returns:
        List of selected molecules
    """
    print(f"Auto-selecting molecules using '{selection_strategy}' strategy")

    if not shape_key_points_result:
        return []

    if selection_strategy == "intelligent":
        # Use a combination of quality and spatial distribution
        selected_molecules = []

        # First, sort by a composite score
        def composite_score(molecule_data):
            # molecule_data format: [class, x, y, w, h, key_x, key_y, ...]
            mol_class = molecule_data[0]
            mol_x = molecule_data[1]  # Already in meters
            mol_y = molecule_data[2]

            # Distance from scan center (prefer closer molecules)
            distance = np.sqrt(
                (mol_x - scan_center[0]) ** 2 + (mol_y - scan_center[1]) ** 2
            )
            distance_score = 1.0 / (distance * 1e9 + 1.0)  # Normalize to nm scale

            # Class confidence (assuming lower class numbers indicate higher confidence)
            class_score = 1.0 / (np.abs(mol_class - 1.0) + 0.1)

            # Size quality (prefer molecules with reasonable size)
            mol_w = molecule_data[3] * 1e9  # Convert to nm
            mol_h = molecule_data[4] * 1e9
            # Prefer molecules around 0.5-2nm size
            size_score = 1.0 / (abs(mol_w - 1.0) + abs(mol_h - 1.0) + 0.1)

            return distance_score * 0.5 + class_score * 0.3 + size_score * 0.2

        # Sort by composite score
        scored_molecules = [
            (mol, composite_score(mol)) for mol in shape_key_points_result
        ]
        scored_molecules.sort(key=lambda x: x[1], reverse=True)

        # Select molecules ensuring spatial distribution
        for mol_data, score in scored_molecules:
            if len(selected_molecules) >= max_molecules_per_scan:
                break

            mol_x = mol_data[1]
            mol_y = mol_data[2]

            # Check minimum distance to already selected molecules
            too_close = False
            min_separation = 3e-9  # 3nm minimum separation

            for selected_mol in selected_molecules:
                sel_x = selected_mol[1]
                sel_y = selected_mol[2]
                dist = np.sqrt((mol_x - sel_x) ** 2 + (mol_y - sel_y) ** 2)
                if dist < min_separation:
                    too_close = True
                    break

            if not too_close:
                selected_molecules.append(mol_data)
                print(
                    f"  Selected molecule {len(selected_molecules)}: class={mol_data[0]}, score={score:.3f}"
                )

    elif selection_strategy == "closest":
        # Select molecules closest to scan center
        def distance_from_center(molecule_data):
            mol_x = molecule_data[1]
            mol_y = molecule_data[2]
            return np.sqrt(
                (mol_x - scan_center[0]) ** 2 + (mol_y - scan_center[1]) ** 2
            )

        sorted_molecules = sorted(shape_key_points_result, key=distance_from_center)
        selected_molecules = sorted_molecules[:max_molecules_per_scan]

    elif selection_strategy == "quality":
        # Select molecules based on class confidence and size quality
        def quality_score(molecule_data):
            mol_class = molecule_data[0]
            mol_w = molecule_data[3] * 1e9  # Convert to nm
            mol_h = molecule_data[4] * 1e9

            class_score = 1.0 / (mol_class + 1.0)
            size_score = 1.0 / (abs(mol_w - 1.0) + abs(mol_h - 1.0) + 0.1)
            return class_score * 0.7 + size_score * 0.3

        sorted_molecules = sorted(
            shape_key_points_result, key=quality_score, reverse=True
        )
        selected_molecules = sorted_molecules[:max_molecules_per_scan]

    return selected_molecules


def visualize_molecules_on_image(
    image: np.ndarray,
    all_molecules: List,
    selected_molecules: List,
    scan_position: Tuple[float, float],
    scan_edge: float,
    save_path: str,
    show_plot: bool,
) -> np.ndarray:
    """
    Visualize detected and selected molecules on the original image

    Args:
        image: Original STM image (grayscale)
        all_molecules: List of all detected molecules with real-world coordinates
        selected_molecules: List of selected molecules (subset of all_molecules)
        scan_position: Center position of scan in meters (x, y)
        scan_edge: Edge length of scan area in nanometers
        save_path: Path to save the annotated image (optional)
        show_plot: Whether to display the plot

    Returns:
        Annotated image array
    """

    # Convert to color image for better visualization
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = image.copy()

    # Convert scan_edge to meters if it's in nanometers
    if scan_edge > 1e-6:  # Assume it's in nanometers if > 1 micrometer
        edge_meters = scan_edge * 1e-9
    else:
        edge_meters = scan_edge

    image_height, image_width = vis_image.shape[:2]

    def real_to_pixel_coords(real_x: float, real_y: float) -> Tuple[int, int]:
        """Convert real-world coordinates to pixel coordinates"""
        # Normalize to 0-1 based on scan area
        # CORRECTED: Use negative sign for Y to properly reverse the coordinate conversion
        norm_x = (real_x - scan_position[0]) / edge_meters + 0.5
        norm_y = (
            -(real_y - scan_position[1]) / edge_meters + 0.5
        )  # FIXED: Added negative sign

        # Convert to pixel coordinates
        pixel_x = int(norm_x * image_width)
        # SIMPLIFIED: No need for (1.0 - norm_y) since we already handle the flip above
        pixel_y = int(norm_y * image_height)

        # Ensure within bounds
        pixel_x = max(0, min(image_width - 1, pixel_x))
        pixel_y = max(0, min(image_height - 1, pixel_y))

        return pixel_x, pixel_y

    # Define colors for 4 keypoints (RGB format for OpenCV)
    keypoint_colors = [
        (0, 255, 255),  # Yellow for keypoint 1
        (255, 0, 255),  # Magenta for keypoint 2
        (0, 255, 0),  # Green for keypoint 3
        (255, 255, 0),  # Cyan for keypoint 4
    ]

    # Draw all detected molecules in blue
    for i, molecule in enumerate(all_molecules):
        mol_x, mol_y = molecule[1], molecule[2]  # Real coordinates in meters

        # Convert to pixel coordinates
        mol_px, mol_py = real_to_pixel_coords(mol_x, mol_y)

        # Draw molecule bounding box (blue for all molecules)
        mol_w = molecule[3] * 1e9  # Convert width to nm
        mol_h = molecule[4] * 1e9  # Convert height to nm

        # Estimate bounding box in pixels
        box_w_px = max(5, int((mol_w / (edge_meters * 1e9)) * image_width))
        box_h_px = max(5, int((mol_h / (edge_meters * 1e9)) * image_height))

        # Draw bounding rectangle with thinner line
        cv2.rectangle(
            vis_image,
            (mol_px - box_w_px // 2, mol_py - box_h_px // 2),
            (mol_px + box_w_px // 2, mol_py + box_h_px // 2),
            (255, 100, 100),
            1,
        )  # Light blue, thinner

        # Draw molecule center with smaller marker
        cv2.circle(vis_image, (mol_px, mol_py), 3, (255, 150, 150), -1)  # Blue circle

        # Draw keypoints based on molecule class
        # Format: [class, x, y, w, h, kp1x, kp1y, kp2x, kp2y, kp3x, kp3y, kp4x, kp4y]
        mol_class = int(molecule[0])

        if mol_class == 1:
            # Class 1: Draw all 4 meaningful keypoints
            num_keypoints = 4
        else:
            # Class 0 and 2: Only draw the first keypoint (only kp1 is meaningful)
            num_keypoints = 1

        for kp_idx in range(num_keypoints):
            x_idx = 5 + (kp_idx * 2)
            y_idx = 6 + (kp_idx * 2)

            if x_idx < len(molecule) and y_idx < len(molecule):
                key_x, key_y = molecule[x_idx], molecule[y_idx]
                key_px, key_py = real_to_pixel_coords(key_x, key_y)

                # Draw keypoint with smaller marker and unique color
                cv2.circle(vis_image, (key_px, key_py), 2, keypoint_colors[kp_idx], -1)

                # Add keypoint number with smaller font to avoid overlap
                if num_keypoints == 1:
                    # For classes 0 and 2, just show "K1"
                    kp_label = "K1"
                else:
                    # For class 1, show K1, K2, K3, K4
                    kp_label = f"K{kp_idx+1}"

                # Calculate text position to avoid overlap with better logic
                text_width = len(kp_label) * 5  # Estimate text width
                text_height = 10  # Estimate text height

                # Try different positions: right, left, top, bottom
                positions = [
                    (key_px + 5, key_py - 5),  # top-right
                    (key_px - text_width - 5, key_py - 5),  # top-left
                    (key_px + 5, key_py + text_height + 5),  # bottom-right
                    (key_px - text_width - 5, key_py + text_height + 5),  # bottom-left
                ]

                # Choose the first position that fits in image bounds
                text_x, text_y = positions[0]  # default
                for pos_x, pos_y in positions:
                    if (
                        5 <= pos_x <= image_width - text_width - 5
                        and text_height + 5 <= pos_y <= image_height - 5
                    ):
                        text_x, text_y = pos_x, pos_y
                        break

                # Remove keypoint text labels - only keep legend
                # cv2.putText(
                #     vis_image,
                #     kp_label,
                #     (text_x, text_y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.3,  # Even smaller font
                #     keypoint_colors[kp_idx],
                #     1,
                # )

        # Add molecule ID with better overlap avoidance
        mol_label = f"M{i+1}"
        text_width = len(mol_label) * 7  # Estimate text width
        text_height = 12  # Estimate text height

        # Try different positions around the molecule
        positions = [
            (mol_px + 5, mol_py - 5),  # top-right
            (mol_px - text_width - 5, mol_py - 5),  # top-left
            (mol_px + 5, mol_py + text_height + 5),  # bottom-right
            (mol_px - text_width - 5, mol_py + text_height + 5),  # bottom-left
        ]

        # Choose the first position that fits
        text_x, text_y = positions[0]  # default
        for pos_x, pos_y in positions:
            if (
                5 <= pos_x <= image_width - text_width - 5
                and text_height + 5 <= pos_y <= image_height - 5
            ):
                text_x, text_y = pos_x, pos_y
                break

        # Remove molecule text labels - only keep legend
        # cv2.putText(
        #     vis_image,
        #     mol_label,
        #     (text_x, text_y),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.35,  # Slightly smaller font
        #     (255, 255, 255),
        #     1,
        # )

    # Draw selected molecules in red (overwrite blue)
    selected_indices = []
    for selected_mol in selected_molecules:
        # Find the index of this molecule in all_molecules
        for i, mol in enumerate(all_molecules):
            if (
                abs(mol[1] - selected_mol[1]) < 1e-12
                and abs(mol[2] - selected_mol[2]) < 1e-12
            ):
                selected_indices.append(i)
                break

    for idx, selected_mol in enumerate(selected_molecules):
        mol_x, mol_y = selected_mol[1], selected_mol[2]

        # Convert to pixel coordinates
        mol_px, mol_py = real_to_pixel_coords(mol_x, mol_y)

        # Draw selected molecule with thicker red outline
        mol_w = selected_mol[3] * 1e9
        mol_h = selected_mol[4] * 1e9

        box_w_px = max(8, int((mol_w / (edge_meters * 1e9)) * image_width))
        box_h_px = max(8, int((mol_h / (edge_meters * 1e9)) * image_height))

        # Draw thin red bounding rectangle
        cv2.rectangle(
            vis_image,
            (mol_px - box_w_px // 2, mol_py - box_h_px // 2),
            (mol_px + box_w_px // 2, mol_py + box_h_px // 2),
            (0, 0, 255),
            2,
        )  # Red, thinner

        # Draw selected molecule center (smaller red circle)
        cv2.circle(vis_image, (mol_px, mol_py), 3, (0, 0, 255), -1)  # Red circle

        # Draw keypoints for selected molecules based on class
        mol_class = int(selected_mol[0])

        if mol_class == 1:
            # Class 1: Draw all 4 meaningful keypoints with enhanced visibility
            num_keypoints = 4
        else:
            # Class 0 and 2: Only draw the first keypoint (only kp1 is meaningful)
            num_keypoints = 1

        for kp_idx in range(num_keypoints):
            x_idx = 5 + (kp_idx * 2)
            y_idx = 6 + (kp_idx * 2)

            if x_idx < len(selected_mol) and y_idx < len(selected_mol):
                key_x, key_y = selected_mol[x_idx], selected_mol[y_idx]
                key_px, key_py = real_to_pixel_coords(key_x, key_y)

                # Draw smaller keypoint with unique color for selected molecules
                cv2.circle(vis_image, (key_px, key_py), 3, keypoint_colors[kp_idx], -1)

                # Add thin black border for better visibility
                cv2.circle(vis_image, (key_px, key_py), 3, (0, 0, 0), 1)

                # Add keypoint number with smaller font to avoid overlap
                if num_keypoints == 1:
                    # For classes 0 and 2, just show "K1"
                    kp_label = "K1"
                else:
                    # For class 1, show K1, K2, K3, K4
                    kp_label = f"K{kp_idx+1}"

                # Calculate text position to avoid overlap with improved logic
                text_width = len(kp_label) * 6  # Estimate text width
                text_height = 12  # Estimate text height

                # Try different positions around the keypoint
                positions = [
                    (key_px + 6, key_py - 6),  # top-right
                    (key_px - text_width - 6, key_py - 6),  # top-left
                    (key_px + 6, key_py + text_height + 6),  # bottom-right
                    (key_px - text_width - 6, key_py + text_height + 6),  # bottom-left
                ]

                # Choose the first position that fits in image bounds
                text_x, text_y = positions[0]  # default
                for pos_x, pos_y in positions:
                    if (
                        5 <= pos_x <= image_width - text_width - 5
                        and text_height + 5 <= pos_y <= image_height - 5
                    ):
                        text_x, text_y = pos_x, pos_y
                        break

                # Remove keypoint text labels - only keep legend
                # cv2.putText(
                #     vis_image,
                #     kp_label,
                #     (text_x, text_y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.35,  # Smaller font
                #     (255, 255, 255),
                #     1,
                # )

        # Add selection number with better overlap avoidance
        sel_label = f"S{idx+1}"
        text_width = len(sel_label) * 7  # Estimate text width
        text_height = 12  # Estimate text height

        # Try different positions around the molecule center
        positions = [
            (mol_px + 8, mol_py + 15),  # bottom-right
            (mol_px - text_width - 8, mol_py + 15),  # bottom-left
            (mol_px + 8, mol_py - 8),  # top-right
            (mol_px - text_width - 8, mol_py - 8),  # top-left
        ]

        # Choose the first position that fits
        text_x, text_y = positions[0]  # default
        for pos_x, pos_y in positions:
            if (
                5 <= pos_x <= image_width - text_width - 5
                and text_height + 5 <= pos_y <= image_height - 5
            ):
                text_x, text_y = pos_x, pos_y
                break

        # Remove selection text labels - only keep legend
        # cv2.putText(
        #     vis_image,
        #     sel_label,
        #     (text_x, text_y),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.4,  # Smaller font
        #     (0, 0, 255),
        #     1,
        # )

    # Add scan center marker with black color and thinner cross
    center_px, center_py = real_to_pixel_coords(scan_position[0], scan_position[1])
    cv2.drawMarker(vis_image, (center_px, center_py), (0, 0, 0), cv2.MARKER_CROSS, 8, 1)

    # Create matplotlib figure with smaller image to make legend more prominent
    if show_plot:
        try:
            plt.figure(
                figsize=(8, 7)
            )  # Further reduced from (12, 10) for better proportions
            plt.subplot(1, 1, 1)

            # Convert BGR to RGB to fix color inversion
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            plt.imshow(vis_image_rgb)

            plt.title(
                f"Molecular Detection Results (Class-based Keypoints)\n"
                f"Total: {len(all_molecules)} detected, {len(selected_molecules)} selected\n"
                f"Class 1: 4 keypoints, Class 0&2: 1 keypoint | Scan: {scan_edge:.1f}nm",
                fontsize=12,  # Smaller font for reduced figure size
            )

            # Add legend with keypoint colors and class information
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="lightcoral",
                    markersize=9,
                    label="All Molecules",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="red",
                    markersize=9,
                    label="Selected Molecules",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="yellow",
                    markersize=7,
                    label="Keypoint 1 (All Classes)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="magenta",
                    markersize=7,
                    label="Keypoint 2 (Class 1 only)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="lime",
                    markersize=7,
                    label="Keypoint 3 (Class 1 only)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="cyan",
                    markersize=7,
                    label="Keypoint 4 (Class 1 only)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="+",
                    color="black",
                    markersize=8,
                    label="Scan Center",
                ),
            ]
            plt.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=10,  # Smaller font for reduced figure size
                markerscale=1.0,  # Appropriate marker size
                frameon=True,
                fancybox=True,
                shadow=True,
            )

            plt.axis("off")
            plt.tight_layout()

            if save_path:
                # Convert to RGB before saving to fix color inversion
                plt.savefig(
                    save_path, dpi=200, bbox_inches="tight"
                )  # Reduced DPI for smaller file
                print(f"Visualization saved to: {save_path}")

            try:
                print("Interactive plot displayed")
            except Exception as e:
                print(f"Could not display interactive plot: {e}")
                print("Plot saved to file instead")

        except Exception as e:
            print(f"Error creating matplotlib visualization: {e}")
            print("Falling back to OpenCV image save only")
            if save_path:
                cv2.imwrite(save_path.replace(".png", "_opencv.png"), vis_image)
                print(
                    f"OpenCV image saved to: {save_path.replace('.png', '_opencv.png')}"
                )

    # Also save the OpenCV image if requested (fix color inversion)
    if save_path and not show_plot:
        # Convert BGR to RGB before saving to fix color inversion
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR))
        print(f"Annotated image saved to: {save_path}")

    # Alternative: Show using OpenCV window (always works)
    if show_plot:
        try:
            # Scale up the image for better visibility
            display_scale = 2.0  # Scale factor for enlarging the display
            display_height = int(vis_image.shape[0] * display_scale)
            display_width = int(vis_image.shape[1] * display_scale)
            enlarged_image = cv2.resize(
                vis_image,
                (display_width, display_height),
                interpolation=cv2.INTER_CUBIC,
            )

            cv2.imshow("Molecular Detection Results", enlarged_image)
            print("Press any key to close the OpenCV window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("OpenCV window closed")
        except Exception as e:
            print(f"Could not display OpenCV window: {e}")

    return vis_image


def test_molecule_detection():
    """
    Main test function for standalone molecule detection
    """
    print("=" * 60)
    print("Standalone Molecule Detection Test")
    print("=" * 60)

    # Configuration
    img_simu_path = "AI_TPM/STM_img_simu/TPM_image/001.png"
    scan_zoom_in_scale = "20n"  # 20 nanometers
    scan_position = (0.0, 0.0)  # Center position in meters
    max_molecules_per_scan = 3
    keypoint_model_path = "AI_TPM/keypoint/best_0417.pt"

    # Convert scan scale to numerical values
    zoom_out_scale = convert_si_prefix(scan_zoom_in_scale)  # Convert to meters
    zoom_out_scale_nano = zoom_out_scale * 1e9  # Convert to nanometers

    print(
        f"Scan scale: {scan_zoom_in_scale} = {zoom_out_scale} m = {zoom_out_scale_nano} nm"
    )
    print(f"Image path: {img_simu_path}")

    # Load and prepare image
    if not os.path.exists(img_simu_path):
        print(f"Error: Image file not found: {img_simu_path}")
        print("Please ensure the image file exists or update the path")
        return

    print("Loading and preprocessing image...")
    image = cv2.imread(img_simu_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {img_simu_path}")
        return

    # Resize image to expected size
    image = cv2.resize(image, (304, 304), interpolation=cv2.INTER_AREA)
    print(f"Image loaded and resized to: {image.shape}")

    # Run molecular seeker
    print("\n" + "=" * 40)
    print("Running Molecular Detection")
    print("=" * 40)

    shape_key_points_result = molecular_seeker_standalone(
        image,
        scan_position=scan_position,
        scan_edge=zoom_out_scale_nano,
        keypoint_model_path=keypoint_model_path,
        save_dir="AI_TPM/test_detection_output",
        molecular_filter_threshold=0.5,  # Example threshold
    )

    if shape_key_points_result is None or len(shape_key_points_result) == 0:
        print("No molecules detected. Test completed.")
        return

    print(f"\nDetection Results:")
    print(f"Total molecules detected: {len(shape_key_points_result)}")

    # Print detailed results with class-specific keypoint information
    for i, molecule in enumerate(shape_key_points_result):
        mol_class = int(molecule[0])
        print(
            f"  Molecule {i+1}: class={mol_class}, "
            f"pos=({molecule[1]*1e9:.2f}, {molecule[2]*1e9:.2f}) nm, "
            f"size=({molecule[3]*1e9:.2f}, {molecule[4]*1e9:.2f}) nm"
        )

        # Show keypoint information based on class
        if mol_class == 1:
            print(f"    Class 1: 4 meaningful keypoints available")
        else:
            print(f"    Class {mol_class}: Only 1 meaningful keypoint (KP1)")

    print(f"\nClass distribution:")
    class_counts = {}
    for molecule in shape_key_points_result:
        mol_class = int(molecule[0])
        class_counts[mol_class] = class_counts.get(mol_class, 0) + 1

    for class_id, count in sorted(class_counts.items()):
        keypoint_info = "4 keypoints" if class_id == 1 else "1 keypoint"
        print(f"  Class {class_id}: {count} molecules ({keypoint_info} each)")

    # Test intelligent molecule selection
    print("\n" + "=" * 40)
    print("Testing Intelligent Molecule Selection")
    print("=" * 40)

    # Test different selection strategies
    strategies = ["intelligent", "closest", "quality", "random"]

    for strategy in strategies:
        print(f"\n--- Testing '{strategy}' strategy ---")
        selected_molecules = auto_select_molecules_standalone(
            shape_key_points_result, max_molecules_per_scan, strategy, scan_position
        )

        print(f"Selected {len(selected_molecules)} molecules:")
        for i, mol in enumerate(selected_molecules):
            print(
                f"  {i+1}. class={mol[0]}, pos=({mol[1]*1e9:.2f}, {mol[2]*1e9:.2f}) nm"
            )

    # Generate visualization for the 'intelligent' strategy
    print("\n" + "=" * 40)
    print("Generating Visualization")
    print("=" * 40)

    # Use intelligent strategy for the final visualization
    final_selected = auto_select_molecules_standalone(
        shape_key_points_result, max_molecules_per_scan, "intelligent", scan_position
    )

    print(f"Creating visualization with {len(final_selected)} selected molecules...")
    visualize_molecules_on_image(
        image=image,
        all_molecules=shape_key_points_result,
        selected_molecules=final_selected,
        scan_position=scan_position,
        scan_edge=zoom_out_scale_nano,
        save_path="standalone_detection_result.png",
        show_plot=True,
    )

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Test the standalone molecular detection
    test_molecule_detection()
