import multiprocessing
import os
import time
import cv2
import numpy as np
from skimage import exposure
from SAC_H_nanonis import Env, ReplayBuffer, SACAgent
from Auto_spectroscopy_class import AI_Nanonis_Spectroscopy
from EvaluationCNN.detect import predict_image_quality
from keypoint.detect import key_detect
from mol_segment.detect import segmented_image
from molecule_spectr_registry import Molecule, Registry
from utils import *


if __name__ == "__main__":

    # Spectroscopy mode settings
    manipulation_enabled = True  # Set to False to disable manipulation
    human_molecule_selection = False  # Set to False for manual selection
    should_do_spectroscopy = True  # Set to False to skip spectroscopy
    max_molecules_per_scan = 5  # Maximum molecules to process per scan
    max_spectroscopy_points = 5  # Maximum spectroscopy points per molecule
    quality_threshold = 0.7  # Scan quality threshold

    # Initialize Nanonis controller for spectroscopy
    nanonis = AI_Nanonis_Spectroscopy()
    nanonis.code_simulation_mode = False
    nanonis.simulation_image_path = "AI_TPM/STM_img_simu/TPM_image/052.png"  # Set simulation image path in nanonis object

    env = Env(polar_space=False)
    agent = SACAgent(env)


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
    width = ScanFrame["width"]
    height = ScanFrame["height"]

    nanonis.nanocoodinate = (center_x, center_y)

    # Convert tip parameters using nanonis convert method
    tip_bias = nanonis.convert(voltage)
    tip_current = nanonis.convert(current)
    zoom_out_scale = nanonis.convert(nanonis.scan_zoom_in_list[0])

    zoom_out_scale_nano = zoom_out_scale * 1e9  # Convert to nanometers for display

    print(
        f"Scan parameters: {nanonis.scan_zoom_in_list[0]} = {zoom_out_scale} m = {zoom_out_scale_nano} nm"
    )

    # Create necessary directories - separate manipulation and spectroscopy folders
    if not os.path.exists(nanonis.mol_tip_induce_path):
        os.makedirs(nanonis.mol_tip_induce_path)

    # Create main directories for the two-phase workflow
    manipulation_main_path = nanonis.mol_tip_induce_path + "/manipulation/"
    if not os.path.exists(manipulation_main_path):
        os.makedirs(manipulation_main_path)

    print("=" * 60)
    print("AI Two-Phase Spectroscopy Workflow")
    print("=" * 60)
    print("Phase 1: Manipulation (class 0 molecules)")
    print("Phase 2: Spectroscopy (class 1 molecules)")
    print(f"Manipulation: {manipulation_enabled}, Human selection: {human_molecule_selection}")
    print(f"Quality threshold: {quality_threshold}, Max per scan: {max_molecules_per_scan}")
    print("=" * 60)

    # Initialize grid-based scanning
    print("Initializing grid-based scanning system...")
    
    # Set large scan area center (50nm x 50nm area around current position)
    # You can modify these coordinates to specify the large scan area
    large_scan_center = nanonis.nanocoodinate  # Use current position as center
    large_scan_size = width

    nanonis.initialize_grid_positions(large_scan_center=large_scan_center, large_scan_size=large_scan_size)

    # Display grid information
    grid_status = nanonis.get_grid_status()
    print(f"Grid initialized with {grid_status['total_positions']} scan positions")
    print(f"Large scan area: {large_scan_size*1e9:.0f}nm x {large_scan_size*1e9:.0f}nm centered at {large_scan_center}")
    print("=" * 60)

    # Record session start time
    session_start_time = time.time()

    # Main spectroscopy loop - grid-based scanning
    scan_count = 0

    while True:
        # Move to next grid position
        has_next_position = nanonis.move_to_next_point()
        
        if not has_next_position:
            print("\n" + "=" * 60)
            print("GRID SCANNING COMPLETED - ALL POSITIONS VISITED!")
            print("=" * 60)
            break
            
        # Get current progress
        grid_status = nanonis.get_grid_status()
        
        nanonis.AdjustTip_flag = nanonis.AdjustTipToPiezoCenter()

        print(f"\n--- SCAN {scan_count} ---")
        print(f"Grid Progress: {grid_status['progress_percent']:.1f}% "
              f"({grid_status['current_position']}/{grid_status['total_positions']})")
        print(f"Current position: ({nanonis.nanocoodinate[0]*1e9:.2f}, {nanonis.nanocoodinate[1]*1e9:.2f}) nm")

        # Step 1: Perform batch scan (handles simulation mode internally)
        print("Performing batch scan...")
        nanonis.batch_scan_producer(
            nanonis.nanocoodinate,
            nanonis.small_scan_frame_size,  # Use 10nm scan size from grid system
            nanonis.scan_square_Buffer_pix,
            0,
        )
        scan_count += 1

        # Step 2: Assess scan quality using CNN
        print("Evaluating scan quality...")
        scan_quality = nanonis.image_recognition()

        # If simulation mode, force good quality
        if nanonis.code_simulation_mode:
            scan_quality = 1
            nanonis.skip_flag = 0

        if scan_quality < quality_threshold:
            print("Poor scan quality, skipping to next position...")
            # Reset skip flag for next position
            nanonis.skip_flag = 0
            continue

        elif scan_quality >= quality_threshold:
            print("Good scan quality detected!")

            # Step 3: Molecule detection using AI
            print("Detecting molecules and keypoints...")

            # Use molecule seeker to detect molecules with correct scan edge
            small_scan_size_nm = nanonis.convert(nanonis.small_scan_frame_size) * 1e9  # Convert to nm
            
            shape_key_points_result = nanonis.molecular_seeker(
                nanonis.image_for,
                scan_position=nanonis.nanocoodinate,
                scan_edge=small_scan_size_nm,  # Use 10nm scan size
            )
            print(shape_key_points_result)

            if shape_key_points_result is None:
                print("No molecules detected in the image.")
                continue

            detected_molecules_count = len(shape_key_points_result)
            print(f"Detected {detected_molecules_count} molecules")
            
            # Show class distribution of detected molecules
            class_counts = {}
            for mol in shape_key_points_result:
                mol_class = int(mol[0])
                class_counts[mol_class] = class_counts.get(mol_class, 0) + 1
            print(f"Class distribution: {class_counts}")

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
                    f"\n=== MANIPULATION PHASE: Automatic Molecule Selection (max: {max_molecules_per_scan}) ==="
                )

                # Use intelligent selection strategy by default, can be changed to:
                # "intelligent", "closest", "quality", "distance_spread", "random"
                selection_strategy = "intelligent"
                
                # Enhanced selection with test_simulation insights:
                selected_molecules = nanonis.auto_select_molecules_for_processing(
                    shape_key_points_result, 
                    max_molecules_per_scan, 
                    mode='manipulation', 
                    selection_strategy=selection_strategy
                )

                print(
                    f"Automatically selected {len(selected_molecules)} molecules using '{selection_strategy}' strategy"
                )

                if len(selected_molecules) == 0:
                    print(
                        "No molecules selected automatically. Skipping this scan area."
                    )
                    continue

            # Step 5: MANIPULATION PHASE - Process each selected molecule
            print("\n" + "="*50)
            print("Phase 1: Manipulation")
            print("="*50)
            
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
                # Convert flat keypoint list to (x,y) pairs
                keypoint_coords = key_points[5:]  # Extract keypoint coordinates
                # Ensure even number of coordinates for pairing
                if len(keypoint_coords) % 2 != 0:
                    keypoint_coords = keypoint_coords[:-1]  # Remove last odd coordinate
                keypoint_pairs = [(keypoint_coords[i], keypoint_coords[i+1]) 
                                for i in range(0, len(keypoint_coords), 2)]
                
                molecular_index = nanonis.molecule_registry.register_molecule(
                    position=mol_position_meters,
                    key_points=keypoint_pairs,  # Pass as (x,y) pairs
                    site_states=np.array([int(key_points[0])]),  # Class as site state
                    orientation=0,
                    molecule_type=f"class_{int(key_points[0])}",
                )

                # Get the registered molecule object
                molecule = nanonis.molecule_registry.molecules[molecular_index]

                print(
                    f"  Registered molecule {molecular_index} at position {mol_position_meters}"
                )

                # Get zoom and timing parameters - use small scan frame size from grid system
                small_scan_scale = nanonis.convert(nanonis.small_scan_frame_size)  # 10nm in meters
                small_scan_scale_nano = small_scan_scale * 1e9  # Convert to nanometers

                image_save_time = time.strftime(
                    "%Y%m%d_%H%M%S", time.localtime(time.time())
                )

                mol_old_position = molecule.position

                # if code_simulation_mode:
                #     nanonis.image_for = cv2.imread(img_simu_path, cv2.IMREAD_GRAYSCALE)
                #     nanonis.image_for = cv2.resize(
                #         nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA
                #     )

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

                time.sleep(3)  # Allow time for manipulation to complete

                # Step 7: Save manipulation results - always create manipulation folder
                manipulation_folder = (
                    nanonis.mol_tip_induce_path
                    + f"/manipulation/scan_{scan_count}_mol_{mol_count+1}_{image_save_time}/"
                )
                if not os.path.exists(manipulation_folder):
                    os.makedirs(manipulation_folder, exist_ok=True)
                print(f"    Created manipulation folder: {manipulation_folder}")

                # Create detailed result recording for manipulation only
                result_data = {
                    "scan_info": {
                        "scan_count": scan_count,
                        "molecule_count": mol_count + 1,
                        "timestamp": image_save_time,
                        "scan_position": nanonis.nanocoodinate,
                        "zoom_scale": small_scan_scale_nano,  # Use 10nm scan scale
                        "manipulation_folder": manipulation_folder,
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
                        "enabled": False,  # No spectroscopy in this phase
                        "points": [],
                        "total_measurements": 0,
                        "successful_measurements": 0,
                        "results": [],
                    },
                }

                # Create annotated visualization image using class method (manipulation only)
                annotated_image = nanonis.create_annotated_image(
                    image_for_rgb=image_for_rgb,
                    mol_position_nm=mol_position_meters,  # Already in meters
                    manipulation_position=(
                        manipulation_position if manipulation_enabled else None
                    ),
                    manipulation_success=(
                        manipulation_success if manipulation_enabled else False
                    ),
                    spectroscopy_points=None,  # No spectroscopy in this phase
                    spectroscopy_results=None,  # No spectroscopy in this phase
                    mol_count=mol_count,
                    scan_center=nanonis.nanocoodinate,
                    scan_scale=small_scan_scale_nano,  # Use 10nm scan scale
                    is_manipulated=(manipulation_success if manipulation_enabled else False),
                )

                # Save manipulation results only using class method
                file_paths = nanonis.save_molecule_results(
                    result_data=result_data,
                    annotated_image=annotated_image,
                    mol_count=mol_count,
                    image_save_time=image_save_time,
                    spectroscopy_results_molecule=None,  # No spectroscopy in this phase
                    spectroscopy_points=None,  # No spectroscopy in this phase
                )

                print(f"  Manipulation results saved")
                print(f"    - Manipulation folder: {file_paths['molecule_folder']}")

                print(f"  Molecule {mol_count + 1} manipulation complete")

            print(f"Scan {scan_count} manipulation phase complete")

            # Step 8: Spectroscopy phase - Rescan the same area to detect manipulated molecules
            if should_do_spectroscopy:
                print("\n" + "="*50)
                print("Phase 2: Spectroscopy")
                print("="*50)
                print("Rescanning the same area to detect manipulated molecules...")
                
                # Wait for tip stabilization after manipulation
                time.sleep(5)
                
                # Perform another scan of the same area (handles simulation mode internally)
                nanonis.batch_scan_producer(
                    nanonis.nanocoodinate,
                    nanonis.small_scan_frame_size,  # Use 10nm scan size
                    nanonis.scan_square_Buffer_pix,
                    0,
                )
                
                # Detect molecules in the rescanned image (including manipulated ones)
                spectroscopy_shape_key_points_result = nanonis.spectr_point_seeker(
                    nanonis.image_for,
                    scan_position=nanonis.nanocoodinate,
                    scan_edge=small_scan_size_nm,  # Use 10nm scan size
                )
                
                if spectroscopy_shape_key_points_result is not None:
                    print(f"Detected {len(spectroscopy_shape_key_points_result)} molecules for spectroscopy")
                    
                    # Show class distribution of detected molecules
                    class_counts = {}
                    for mol in spectroscopy_shape_key_points_result:
                        mol_class = int(mol[0])
                        class_counts[mol_class] = class_counts.get(mol_class, 0) + 1
                    print(f"Spectroscopy class distribution: {class_counts}")
                    
                    # Step 8.1: Select molecules for spectroscopy using intelligent strategy
                    print(f"Selecting class 1 molecules for spectroscopy (max: {max_molecules_per_scan})")
                    
                    # Use intelligent selection strategy for spectroscopy
                    # Enhanced with test_simulation insights: prefer class 1 but allow flexibility
                    spectroscopy_selection_strategy = "intelligent"
                    
                    selected_spectroscopy_molecules = nanonis.auto_select_molecules_for_processing(
                        spectroscopy_shape_key_points_result, 
                        max_molecules_per_scan, 
                        mode='spectroscopy',
                        selection_strategy=spectroscopy_selection_strategy
                    )
                    
                    print(
                        f"Selected {len(selected_spectroscopy_molecules)}/{len(spectroscopy_shape_key_points_result)} molecules for spectroscopy using '{spectroscopy_selection_strategy}' strategy"
                    )
                    
                    if len(selected_spectroscopy_molecules) == 0:
                        print("No molecules selected for spectroscopy. Skipping spectroscopy phase.")
                    else:
                        # Process each selected molecule for spectroscopy
                        for spec_mol_count, spec_key_points in enumerate(selected_spectroscopy_molecules):
                            print(f"\n  Performing spectroscopy on molecule {spec_mol_count + 1}/{len(selected_spectroscopy_molecules)}")
                            
                            # Create spectroscopy folder
                            spectroscopy_folder = (
                                nanonis.mol_tip_induce_path
                                + f"/spectroscopy/scan_{scan_count}_mol_{spec_mol_count+1}_{image_save_time}/"
                            )
                            if not os.path.exists(spectroscopy_folder):
                                os.makedirs(spectroscopy_folder, exist_ok=True)
                            print(f"    Created spectroscopy folder: {spectroscopy_folder}")
                            
                            # Extract molecule position
                            spec_mol_position_meters = (spec_key_points[1], spec_key_points[2])
                            
                            # Register molecule for spectroscopy
                            # Convert flat keypoint list to (x,y) pairs
                            spec_keypoint_coords = spec_key_points[5:]  # Extract keypoint coordinates
                            # Ensure even number of coordinates for pairing
                            if len(spec_keypoint_coords) % 2 != 0:
                                spec_keypoint_coords = spec_keypoint_coords[:-1]  # Remove last odd coordinate
                            spec_keypoint_pairs = [(spec_keypoint_coords[i], spec_keypoint_coords[i+1]) 
                                                  for i in range(0, len(spec_keypoint_coords), 2)]
                            
                            spec_molecular_index = nanonis.molecule_registry.register_molecule(
                                position=spec_mol_position_meters,
                                key_points=spec_keypoint_pairs,  # Pass as (x,y) pairs
                                site_states=np.array([int(spec_key_points[0])]),
                                orientation=0,
                                molecule_type=f"class_{int(spec_key_points[0])}_spectroscopy",
                            )
                            
                            # Determine spectroscopy points based on keypoints
                            spectroscopy_points = []
                            
                            # Use keypoints for spectroscopy measurements
                            if len(spec_key_points) >= 9:  # Need at least 9 elements to access second keypoint (indices 7,8)
                                # Only use the second keypoint for spectroscopy (indices 7,8)
                                point_x = spec_key_points[7]  # Second keypoint X coordinate
                                point_y = spec_key_points[8]  # Second keypoint Y coordinate
                                spectroscopy_points.append((point_x, point_y))
                                print("    Using second keypoint for spectroscopy measurement")
                            
                            # Fallback to molecule center if no keypoints available
                            if len(spectroscopy_points) == 0:
                                center_point = (spec_mol_position_meters[0], spec_mol_position_meters[1])
                                spectroscopy_points.append(center_point)

                            
                            print(f"    Spectroscopy points: {len(spectroscopy_points)}")
                            
                            # Perform spectroscopy at each point
                            spectroscopy_results_molecule = []
                            successful_measurements = 0
                            
                            for point_idx, spec_point in enumerate(spectroscopy_points):
                                print(f"      Point {point_idx + 1}/{len(spectroscopy_points)}: {spec_point}")
                                
                                # Perform spectroscopy measurement
                                try:
                                    result = nanonis.perform_spectroscopy_measurement(
                                        spectroscopy_point=spec_point,
                                        molecular_index=spec_molecular_index,
                                        point_index=point_idx,
                                        image_save_time=image_save_time,
                                        image_with_molecule=cv2.cvtColor(nanonis.image_for, cv2.COLOR_GRAY2BGR),
                                        molecule_folder=spectroscopy_folder,
                                        scan_center=nanonis.nanocoodinate,
                                    )
                                    
                                    spectroscopy_results_molecule.append(result)
                                    if result.get("success", False):
                                        successful_measurements += 1
                                        
                                except Exception as e:
                                    print(f"      Spectroscopy failed at point {point_idx + 1}: {e}")
                                    spectroscopy_results_molecule.append({
                                        "success": False,
                                        "error": str(e),
                                        "position": spec_point,
                                    })
                            
                            # Update molecule with spectroscopy results
                            spec_molecule = nanonis.molecule_registry.molecules[spec_molecular_index]
                            spec_molecule.mark_spectroscopy_completed(
                                points=spectroscopy_points,
                                results=spectroscopy_results_molecule,
                                spectroscopy_time=time.time(),
                            )
                            
                            print(f"    Spectroscopy completed: {successful_measurements}/{len(spectroscopy_points)} successful")
                            
                            # Create spectroscopy result data
                            spectroscopy_result_data = {
                                "scan_info": {
                                    "scan_count": scan_count,
                                    "molecule_count": spec_mol_count + 1,
                                    "timestamp": image_save_time,
                                    "scan_position": nanonis.nanocoodinate,
                                    "zoom_scale": small_scan_scale_nano,
                                    "spectroscopy_folder": spectroscopy_folder,
                                },
                                "molecule_info": {
                                    "molecular_index": spec_molecular_index,
                                    "molecule_type": f"class_{int(spec_key_points[0])}_spectroscopy",
                                    "position_meters": spec_mol_position_meters,
                                    "detected_keypoints": spec_key_points[5:],
                                    "is_manipulated": True,  # Assume detected in post-manipulation scan
                                },
                                "manipulation_info": {
                                    "enabled": False,  # No manipulation in spectroscopy phase
                                    "success": False,
                                    "position": None,
                                    "bias_voltage": None,
                                    "current": None,
                                },
                                "spectroscopy_info": {
                                    "enabled": True,
                                    "points": spectroscopy_points,
                                    "total_measurements": len(spectroscopy_points),
                                    "successful_measurements": successful_measurements,
                                    "results": spectroscopy_results_molecule,
                                },
                            }
                            
                            # Create annotated image for spectroscopy
                            spectroscopy_annotated_image = nanonis.create_annotated_image(
                                image_for_rgb=cv2.cvtColor(nanonis.image_for, cv2.COLOR_GRAY2BGR),
                                mol_position_nm=spec_mol_position_meters,
                                manipulation_position=None,  # No manipulation in this phase
                                manipulation_success=False,
                                spectroscopy_points=spectroscopy_points,
                                spectroscopy_results=spectroscopy_results_molecule,
                                mol_count=spec_mol_count,
                                scan_center=nanonis.nanocoodinate,
                                scan_scale=small_scan_scale_nano,
                                is_manipulated=True,
                            )
                            
                            # Save spectroscopy results
                            spectroscopy_file_paths = nanonis.save_molecule_results(
                                result_data=spectroscopy_result_data,
                                annotated_image=spectroscopy_annotated_image,
                                mol_count=spec_mol_count,
                                image_save_time=image_save_time,
                                spectroscopy_results_molecule=spectroscopy_results_molecule,
                                spectroscopy_points=spectroscopy_points,
                            )
                            
                            print(f"    Spectroscopy results saved")
                            print(f"      - Spectroscopy folder: {spectroscopy_file_paths['molecule_folder']}")
                            if spectroscopy_file_paths["spectroscopy_files"]:
                                print(f"      - Spectroscopy files: {len(spectroscopy_file_paths['spectroscopy_files'])} measurements")
                            
                            print(f"    Molecule {spec_mol_count + 1} spectroscopy complete")
                
                else:
                    print("No molecules detected in post-manipulation scan for spectroscopy")
                
                print("="*50)
                print("Spectroscopy phase completed")
                print("="*50)

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
