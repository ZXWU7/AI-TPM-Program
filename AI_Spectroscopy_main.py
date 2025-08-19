import multiprocessing
import os
import time
import cv2
import numpy as np
from skimage import exposure

from Auto_spectroscopy_class import AI_Nanonis_Spectroscopy
from EvaluationCNN.detect import predict_image_quality
from keypoint.detect import key_detect
from mol_segment.detect import segmented_image
from molecule_spectr_registry import Molecule, Registry
from utils import *

if __name__ == "__main__":

    # Configuration parameters
    code_simulation_mode = True

    if code_simulation_mode:
        img_simu_path = "AI_TPM/STM_img_simu/TPM_image/001.png"

    # Spectroscopy mode settings
    manipulation_enabled = True  # Set to False to disable manipulation
    human_molecule_selection = False  # Set to False for manual selection
    max_molecules_per_scan = 10  # Maximum molecules to process per scan
    max_spectroscopy_points = 2  # Maximum spectroscopy points per molecule
    quality_threshold = 0.7  # Scan quality threshold

    # Initialize Nanonis controller for spectroscopy
    nanonis = AI_Nanonis_Spectroscopy()

    # Initialize tip and create log folders
    nanonis.tip_init(mode="new")

    # Activate monitoring threads
    nanonis.monitor_thread_activate()

    # Set initial tip parameters
    voltage = "-4.0"  # V for spectroscopy
    current = "0.1n"  # A

    # Get initial scan frame
    ScanFrame = nanonis.ScanFrameGet()
    center_x = ScanFrame["center_x"]
    center_y = ScanFrame["center_y"]

    nanonis.nanocoodinate = (center_x, center_y)

    # Convert tip parameters using nanonis convert method
    tip_bias = nanonis.convert(voltage)
    tip_current = nanonis.convert(current)
    zoom_out_scale = nanonis.convert(nanonis.scan_zoom_in_list[0])

    zoom_out_scale_nano = zoom_out_scale * 1e9  # Convert to nanometers for display

    print(
        f"Scan parameters: {nanonis.scan_zoom_in_list[0]} = {zoom_out_scale} m = {zoom_out_scale_nano} nm"
    )

    # Create necessary directories
    if not os.path.exists(nanonis.mol_tip_induce_path):
        os.makedirs(nanonis.mol_tip_induce_path)

    spectroscopy_save_path = nanonis.mol_tip_induce_path + "/spectroscopy/"
    if not os.path.exists(spectroscopy_save_path):
        os.makedirs(spectroscopy_save_path)

    print("=" * 60)
    print("AI SPECTROSCOPY WORKFLOW STARTED")
    print("=" * 60)
    print(f"Manipulation enabled: {manipulation_enabled}")
    print(f"Human molecule selection: {human_molecule_selection}")
    print(f"Quality threshold: {quality_threshold}")
    print(f"Max molecules per scan: {max_molecules_per_scan}")
    print("=" * 60)

    # Record session start time
    session_start_time = time.time()

    # Main spectroscopy loop
    scan_count = 0

    while tip_in_boundary(
        nanonis.inter_closest, nanonis.plane_size, nanonis.real_scan_factor
    ):
        nanonis.AdjustTip_flag = nanonis.AdjustTipToPiezoCenter()
        nanonis.move_to_next_point()

        print(f"\n--- SCAN {scan_count} ---")

        # Step 1: Perform batch scan
        print("Performing batch scan...")
        nanonis.batch_scan_producer(
            nanonis.nanocoodinate,
            nanonis.scan_zoom_in_list[0],
            nanonis.scan_square_Buffer_pix,
            0,
        )
        scan_count += 1

        # Handle simulation mode
        if code_simulation_mode:
            nanonis.image_for = cv2.imread(img_simu_path, cv2.IMREAD_GRAYSCALE)
            nanonis.image_for = cv2.resize(
                nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA
            )

        # Step 2: Assess scan quality using CNN
        print("Evaluating scan quality...")
        scan_quality = nanonis.image_recognition()

        # If simulation mode, force good quality
        if code_simulation_mode:
            scan_quality = 1
            nanonis.skip_flag = 0

        if scan_quality < quality_threshold:
            print("Poor scan quality, skipping to next position...")
            nanonis.move_to_next_point()
            continue

        elif scan_quality >= quality_threshold:
            print("Good scan quality detected!")

            # Step 3: Molecule detection using AI
            print("Detecting molecules and keypoints...")

            # Use molecule seeker to detect molecules
            shape_key_points_result = nanonis.molecular_seeker(
                nanonis.image_for,
                scan_posion=nanonis.nanocoodinate,
                scan_edge=zoom_out_scale_nano,
            )

            if shape_key_points_result is None:
                print("No molecules detected in the image.")
                nanonis.move_to_next_point()
                continue

            detected_molecules_count = len(shape_key_points_result)
            print(f"Detected {detected_molecules_count} molecules")

            # Step 4: Select molecules for processing
            if human_molecule_selection:
                # Human selection: use interactive selection
                selected_molecules = nanonis.human_select_molecules_for_manipulation(
                    shape_key_points_result, nanonis.image_for
                )
                print(
                    f"Human selected {len(selected_molecules)} molecules for processing"
                )
            else:
                # Enhanced automatic molecule selection with intelligent strategy
                print(
                    f"\n=== Automatic Molecule Selection (max: {max_molecules_per_scan}) ==="
                )

                # Use intelligent selection strategy by default, can be changed to:
                # "intelligent", "closest", "quality", "distance_spread", "random"
                selection_strategy = "intelligent"

                selected_molecules = nanonis.auto_select_molecules_for_processing(
                    shape_key_points_result, max_molecules_per_scan, selection_strategy
                )

                print(
                    f"Automatically selected {len(selected_molecules)} molecules using '{selection_strategy}' strategy"
                )

                if len(selected_molecules) == 0:
                    print(
                        "No molecules selected automatically. Skipping this scan area."
                    )
                    continue

            # Step 5: Process each selected molecule
            for mol_count, key_points in enumerate(selected_molecules):
                print(
                    f"\n  Processing molecule {mol_count + 1}/{len(selected_molecules)}"
                )

                # Convert key_points to molecule position for registration
                # molecule coordinates are now in meters (consistent with test simulation)
                mol_position_meters = (
                    key_points[1],
                    key_points[2],
                )  # Already in meters

                # Register the molecule in the registry
                molecular_index = nanonis.molecule_registry.register_molecule(
                    position=mol_position_meters,
                    key_points=key_points[5:],  # Extract keypoint coordinates
                    site_states=np.array([int(key_points[0])]),  # Class as site state
                    orientation=0,
                    molecule_type=f"class_{int(key_points[0])}",
                )

                # Get the registered molecule object
                molecule = nanonis.molecule_registry.molecules[molecular_index]

                print(
                    f"  Registered molecule {molecular_index} at position {mol_position_meters}"
                )

                # Get zoom and timing parameters
                zoom_in_scale = nanonis.convert(nanonis.scan_zoom_in_list[-1])
                zoom_in_scale_nano = zoom_in_scale * 10**9

                image_save_time = time.strftime(
                    "%Y%m%d_%H%M%S", time.localtime(time.time())
                )

                mol_old_position = molecule.position

                if code_simulation_mode:
                    nanonis.image_for = cv2.imread(img_simu_path, cv2.IMREAD_GRAYSCALE)
                    nanonis.image_for = cv2.resize(
                        nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA
                    )

                # Prepare key points for processing (already registered above)
                key_points_result = [key_points]

                # Create visualization image
                image_for_rgb = cv2.cvtColor(nanonis.image_for, cv2.COLOR_GRAY2BGR)

                # Step 6: Manipulation (if enabled)
                manipulation_success = True
                if manipulation_enabled:
                    print(f"  Performing manipulation on molecule {mol_count + 1}")

                    # Extract manipulation position from key points
                    manipulation_position = (
                        key_points_result[0][5],  # Already in meters
                        key_points_result[0][6],  # Already in meters
                    )

                    # Perform tip manipulation
                    print("  Starting tip manipulation...")
                    try:
                        # Apply bias pulse for manipulation
                        nanonis.perform_tip_manipulation(
                            tip_position=manipulation_position,
                            tip_bias=tip_bias,
                            tip_current=tip_current,
                            tip_induce_mode="pulse",
                        )

                        # Mark molecule as manipulated
                        nanonis.molecule_registry.update_molecule(
                            molecular_index,
                            is_manipulated=True,
                            operated_time=time.time(),
                        )
                        molecule = nanonis.molecule_registry.molecules[molecular_index]
                        print(f"  Manipulation completed on molecule {mol_count + 1}")

                    except Exception as e:
                        print(f"  Manipulation failed: {e}")
                        manipulation_success = False

                # Step 7: Spectroscopy measurements
                # Determine if spectroscopy should be performed
                should_do_spectroscopy = False

                if manipulation_enabled:
                    # If manipulation is enabled, do spectroscopy on all detected molecules
                    should_do_spectroscopy = manipulation_success
                else:
                    # If manipulation is disabled, do spectroscopy on all interested molecules
                    should_do_spectroscopy = True

                if should_do_spectroscopy:
                    print(f"  Performing spectroscopy on molecule {mol_count + 1}")

                    # Determine spectroscopy points based on keypoints
                    spectroscopy_points = []

                    # Use keypoints for spectroscopy measurements
                    if len(key_points_result) > 0 and len(key_points_result[0]) >= 7:
                        # Extract keypoints from the detection result
                        # key_points format: [class, center_x, center_y, width, height, keypoint1_x, keypoint1_y, ...]
                        keypoints_start_index = 5  # keypoints start from index 5

                        # Calculate how many keypoint pairs are available
                        available_keypoints = (
                            len(key_points_result[0]) - keypoints_start_index
                        ) // 2
                        num_points_per_molecule = min(
                            max_spectroscopy_points, available_keypoints
                        )

                        for i in range(num_points_per_molecule):
                            keypoint_x_idx = keypoints_start_index + (i * 2)
                            keypoint_y_idx = keypoints_start_index + (i * 2) + 1

                            if keypoint_y_idx < len(key_points_result[0]):
                                point_x = key_points_result[0][
                                    keypoint_x_idx
                                ]  # Already in meters
                                point_y = key_points_result[0][
                                    keypoint_y_idx
                                ]  # Already in meters
                                spectroscopy_points.append((point_x, point_y))

                    # Fallback to molecule center if no keypoints available
                    if len(spectroscopy_points) == 0:
                        center_point = (molecule.position[0], molecule.position[1])
                        spectroscopy_points.append(center_point)
                        print(
                            "    No keypoints available, using molecule center as fallback"
                        )

                    print(f"  Spectroscopy points: {len(spectroscopy_points)}")

                    # Perform spectroscopy at each point
                    spectroscopy_results_molecule = []
                    successful_measurements = 0

                    for point_idx, spec_point in enumerate(spectroscopy_points):
                        print(
                            f"    Point {point_idx + 1}/{len(spectroscopy_points)}: {spec_point}"
                        )

                        # Perform spectroscopy measurement
                        try:
                            result = nanonis.perform_spectroscopy_measurement(
                                spectroscopy_point=spec_point,
                                molecular_index=molecular_index,
                                point_index=point_idx,
                                image_save_time=image_save_time,
                                image_with_molecule=image_for_rgb,
                            )

                            spectroscopy_results_molecule.append(result)
                            if result.get("success", False):
                                successful_measurements += 1

                        except Exception as e:
                            print(
                                f"    Spectroscopy failed at point {point_idx + 1}: {e}"
                            )
                            spectroscopy_results_molecule.append(
                                {
                                    "success": False,
                                    "error": str(e),
                                    "position": spec_point,
                                }
                            )

                    # Update molecule with spectroscopy results
                    molecule.mark_spectroscopy_completed(
                        points=spectroscopy_points,
                        results=spectroscopy_results_molecule,
                        spectroscopy_time=time.time(),
                    )

                    print(
                        f"  Spectroscopy completed: {successful_measurements}/{len(spectroscopy_points)} successful"
                    )

                # Create detailed result recording with visualizations
                result_data = {
                    "scan_info": {
                        "scan_count": scan_count,
                        "molecule_count": mol_count + 1,
                        "timestamp": image_save_time,
                        "scan_position": nanonis.nanocoodinate,
                        "zoom_scale": zoom_out_scale_nano,
                    },
                    "molecule_info": {
                        "molecular_index": molecular_index,
                        "molecule_type": f"class_{int(key_points[0])}",
                        "position_meters": mol_position_meters,
                        "detected_keypoints": key_points[5:],
                        "is_manipulated": (
                            manipulation_success if manipulation_enabled else False
                        ),
                    },
                    "manipulation_info": {
                        "enabled": manipulation_enabled,
                        "success": (
                            manipulation_success if manipulation_enabled else False
                        ),
                        "position": (
                            manipulation_position if manipulation_enabled else None
                        ),
                        "bias_voltage": tip_bias if manipulation_enabled else None,
                        "current": tip_current if manipulation_enabled else None,
                    },
                    "spectroscopy_info": {
                        "enabled": should_do_spectroscopy,
                        "points": spectroscopy_points if should_do_spectroscopy else [],
                        "total_measurements": (
                            len(spectroscopy_points) if should_do_spectroscopy else 0
                        ),
                        "successful_measurements": (
                            successful_measurements if should_do_spectroscopy else 0
                        ),
                        "results": (
                            spectroscopy_results_molecule
                            if should_do_spectroscopy
                            else []
                        ),
                    },
                }

                # Create annotated visualization image using class method
                annotated_image = nanonis.create_annotated_image(
                    image_for_rgb=image_for_rgb,
                    mol_position_nm=mol_position_meters,  # Already in meters
                    manipulation_position=(
                        manipulation_position if manipulation_enabled else None
                    ),
                    manipulation_success=(
                        manipulation_success if manipulation_enabled else False
                    ),
                    spectroscopy_points=(
                        spectroscopy_points if should_do_spectroscopy else None
                    ),
                    spectroscopy_results=(
                        spectroscopy_results_molecule
                        if should_do_spectroscopy
                        else None
                    ),
                    mol_count=mol_count,
                    scan_center=nanonis.nanocoodinate,
                    scan_scale=zoom_out_scale_nano,
                )

                # Save comprehensive results using class method
                file_paths = nanonis.save_molecule_results(
                    result_data=result_data,
                    annotated_image=annotated_image,
                    mol_count=mol_count,
                    image_save_time=image_save_time,
                    spectroscopy_results_molecule=(
                        spectroscopy_results_molecule
                        if should_do_spectroscopy
                        else None
                    ),
                    spectroscopy_points=(
                        spectroscopy_points if should_do_spectroscopy else None
                    ),
                )

                print(f"  Results saved")
                if file_paths["spectroscopy_folder"]:
                    print(
                        f"    - Spectroscopy data: {file_paths['spectroscopy_folder']}"
                    )

                print(f"  Molecule {mol_count + 1} processing complete")

            print(f"Scan {scan_count} processing complete")

            # Save checkpoint after each scan using existing method
            nanonis.save_checkpoint()

        # Print session summary periodically
        if scan_count % 5 == 0 and scan_count > 0:  # Every 5 scans
            summary = nanonis.spectroscopy_workflow_summary()
            print(f"\n--- SESSION SUMMARY (After {scan_count} scans) ---")
            print(
                f"Total molecules detected: {summary['molecule_statistics']['total_detected']}"
            )
            print(
                f"Total manipulations performed: {summary['molecule_statistics']['manipulated']}"
            )
            print(
                f"Total spectroscopy measurements: {summary['spectroscopy_statistics']['total_measurements']}"
            )
            print(
                f"Spectroscopy success rate: {summary['spectroscopy_statistics']['success_rate']:.1f}%"
            )
            print("-" * 50)

    # Final summary and cleanup using existing methods
    print("\n" + "=" * 60)
    print("AI SPECTROSCOPY WORKFLOW COMPLETED")
    print("=" * 60)

    # Get comprehensive summary using existing method
    final_summary = nanonis.spectroscopy_workflow_summary()

    print(f"Total scans performed: {scan_count}")
    print(
        f"Total molecules detected: {final_summary['molecule_statistics']['total_detected']}"
    )
    print(
        f"Total manipulations performed: {final_summary['molecule_statistics']['manipulated']}"
    )
    print(
        f"Total spectroscopy measurements: {final_summary['spectroscopy_statistics']['total_measurements']}"
    )
    print(
        f"Overall spectroscopy success rate: {final_summary['spectroscopy_statistics']['success_rate']:.1f}%"
    )

    # Export comprehensive results using existing method
    results_file = nanonis.export_spectroscopy_results()

    # Save final checkpoint
    nanonis.save_checkpoint()

    print(f"Results exported to: {results_file}")
    print("=" * 60)
