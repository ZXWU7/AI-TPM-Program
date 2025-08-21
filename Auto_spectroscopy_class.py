# Wuzhengxiao 2025/7/27

import json
import os
import pickle
import random
import re
import shutil
import socket
import sys
import threading
import time
from collections import deque
from multiprocessing import Process, Queue

import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import NanonisController
from DQN.agent import *
from EvaluationCNN.detect import predict_image_quality
from keypoint.detect import key_detect
from mol_segment.detect import segmented_image
from molecule_spectr_registry import Molecule, Registry
from SAC_H_nanonis import Env, ReplayBuffer, SACAgent
from square_fitting import mol_Br_site_detection
from utils import *


class AI_Nanonis_Spectroscopy(NanonisController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.strftime(
            "%Y-%m-%d %H-%M-%S", time.localtime(time.time())
        )  # record the start time of the scan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.circle_list = (
            []
        )  # the list to save the circle which importantly to the tip path
        self.circle_list_save = []  # the list to save the circle as npy file
        self.nanocoodinate_list = (
            []
        )  # the list to save the coodinate which send to nanonis directly
        self.visual_circle_buffer_list = []

        # Line scan and quality monitoring
        self.line_scan_change_times = 0  # initialize the line scan change times to 0
        self.AdjustTip_flag = (
            0  # 0: the tip is not adjusted, 1: the tip is adjusted just now
        )
        self.small_scan_frame_size = "10n"
        self.Scan_edge = self.small_scan_frame_size  # set the scan square edge length
        # self.scan_zoom_V1 = (
        #     "20n"  # the zoom of the scan, 20nm. the scale for positioning the molecule
        # )
        # self.scan_zoom_V2 = (
        #     "10n"  # the zoom of the scan, 10nm. the scale for tip induce
        # )
        # self.scan_zoom_V3 = "5n"  # the zoom of the scan, 5nm. the scale for tip induce
        self.scan_zoom_in_list = [self.small_scan_frame_size]  # the list of the zoom in scan adge
        # FIXED: Use consistent scan size across all functions (using first element only)
        self.molecular_registration_list = (
            []
        )  # the list to save the molecular registration data
        self.molecular_filter_threshold = 1e-9  # if the distance between the molecular is less than the threshold, the molecular will be filterd
        self.zoom_in_tip_speed = "4n"  # the speed of the zoom in tip

        self.scan_square_Buffer_pix = 304  # scan pix
        self.half_scan_pix = self.scan_square_Buffer_pix // 2
        self.plane_edge = "1.3u"  # plane size repersent the area of the Scanable surface  2um*2um    2000pix  ==>>  2um
        self.Z_fixed_scan_time = (
            10  # if the scan is after the Z fixed, how many seconds will the scan cost
        )
        self.without_Z_fixed_scan_time = 10  # if the scan is without the Z fixed, how many seconds will the scan cost
        self.linescan_max_min_threshold = "800p"  # the max and min threshold , if the line scan data is out of the threshold
        self.scan_max_min_threshold = "1n"  # the max and min threshold , if the line scan data is out of the threshold
        
        # Grid-based scanning parameters
        self.large_scan_frame_size = "50n"  # Large scan frame size (50nm)   # Small scan frame size (10nm)
        self.bias_spectroscopy_start = -2.0  # V
        self.bias_spectroscopy_end = 2.0     # V
        self.spectr_points= 201  # Number of points in spectroscopy
        self.grid_overlap = 0.1               # 0.1 (10%) overlap between adjacent scans
        self.grid_initialized = False        # Flag to track if grid is initialized
        self.large_scan_center = None        # Center position of the large scan frame
        self.grid_positions = []             # List of grid positions for scanning
        self.current_grid_index = 0          # Current position in the grid
        self.grid_completed = False          # Flag to track if grid scanning is complete
        self.len_threshold_list = 10
        self.threshold_list = deque(
            [], maxlen=self.len_threshold_list
        )  # the list to save the threshold of the line scan data, if threshold_list is full with 1, the scan will be skiped
        self.skip_list = deque(
            [], maxlen=10
        )  # the list to save the skip flag, if the skip_list is full with 1, gave the tip a aggressive tip shaper
        self.skip_flag = 0  # 0: the scan is not skiped, 1: the scan is skiped
        self.aggressive_tip_flag = (
            0  # 0: the tip is not aggressive, 1: the tip is aggressive
        )
        self.real_scan_factor = (
            0.7  # the real scan area is 70% of the scan square edge length
        )
        self.tip_move_mode = (
            0  # 0: continuous_random_move_mode, 1: para_or_vert_move_mode
        )
        self.line_scan_activate = (
            1  # 0: line scan is not activate, 1: line scan is activate
        )
        self.equalization_alpha = 0.3  # the alpha of the equalization
        self.scan_qulity_threshold = 0.7  # the threshold of the scan qulity, higher means more strict for the scan qulity
        self.scan_qulity = 1  # 1: the scan qulity is good, 0: the scan qulity is bad

        self.code_simulation_mode = True
        self.simulation_image_path = None
        self.nanonis_mode = "real"  # the nanonis mode, 'demo' or 'real'
  
        if self.nanonis_mode == "demo":
            self.signal_channel_list = [
                0,
                14,
            ]  # the mode channel list, 0 is Current, 14 is Z_m
        elif self.nanonis_mode == "real":
            self.signal_channel_list = [
                0,
                8,
                14,
            ]  # the mode channel list, 0 is Current, 8 is Bias, 14 is Z_m

        # Model paths for AI systems

        self.quality_model_path = (
            "AI_TPM/EvaluationCNN/CNN_V1.pth"  # the path of the quality model weights
        )

        # self.segment_model_path = "AI_TPM/mol_segment/unet_model-zzw-Ni_V2.pth"  # the path of the segment model weights  (no need in spectroscopy mode)

        self.keypoint_model_path = (
            "AI_TPM/keypoint/best_0417.pt"  # molecule-center keypoint detection model path
        )
        self.spectr_model_path = (
            "AI_TPM/keypoint/best_0820.pt"  # spectroscopy point detection model path
        )

        self.manipulation_check_ai_path = "AI_TPM/Manipulation_Check_AI/"  # the path of the trained AI agent for manipulation classification

        # self.keypoints_model_path = "AI_TPM/keypoints_model/best_0417.pt"  # the path of the trained AI agent for spectroscopy point prediction

        # self.DQN_init_model_path = (
        #     "./DQN/pre-train_DQN.pth"  # the path of the pre-train DQN model
        # )

        # self.DQN_save_main_path = (
        #     "./DQN/train_results/" + self.start_time
        # )  # the path of the DQN model saving

        self.main_data_save_path = "AI_TPM/log"

        self.log_path = self.main_data_save_path + "/" + self.start_time

        self.Scan_edge_SI = self.convert(
            self.Scan_edge
        )  # the scan edge in SI unit   Scan_edge_SI = 30 * 1e-9

        self.plane_size = int(
            self.convert(self.plane_edge) * 1e9
        )  # the plane size in pix  plane_size = 2000
        self.scan_square_edge = int(
            self.convert(self.Scan_edge) * 1e9
        )  # the scan square edge in pix

        self.tip_path_img = (
            np.ones((self.plane_size, self.plane_size, 3), np.uint8) * 255
        )  # the tip path image

        self.inter_closest = (
            round(self.plane_size / 2),
            round(self.plane_size / 2),
        )  # initialize the inter_closest
        # R_init = round(scan_square_edge*(math.sqrt(2))*1.5)
        self.R_init = self.scan_square_edge - 1  # initialize the Radius of tip step
        self.R_max = self.R_init * 3
        self.R_step = int(0.5 * self.R_init)
        self.R = self.R_init

        # initialize the other parameters that appear in the function
        self.Scan_data = {}  # the dictionary to save the scan data
        self.image_for = None  # 2D nparray the image of the scan data, have been nomalized and linear background
        self.image_back = None
        self.equalization_for = (
            None  # the equalization image of the image_for and image_back
        )
        self.equalization_back = None
        self.image_for_tensor = (
            None  # the tensor of the image, 4 dimension, [1, 1, 256, 256]
        )
        self.image_back_tensor = None
        self.image_save_time = None  # when the image is saved in log
        self.npy_data_save_path = (
            self.log_path + "/" + "npy"
        )  # self.log_path = './log/' + self.start_time
        self.image_data_save_path = self.log_path + "/" + "image"
        self.segmented_image_path = None  # the path of the segmented image saving
        self.nemo_nanocoodinate = (
            None  # the nanocoodinate of the nemo point, the format is SI unit
        )
        self.coverage = None  # the moleculer coverage of the image
        self.line_start_time = None
        self.episode_start_time = None  # the start time of the episode
        # self.molecular_registration_list = []  # the list to save the molecular registration data

        # initialize the queue, the Queue is used to communicate between different threads
        self.lineScanQueue = Queue(5)  # lineScan_data_producer → lineScan_data_consumer
        self.lineScanQueue_back = Queue(
            5
        )  # lineScan_data_consumer → lineScan_data_producer
        self.ScandataQueue = Queue(5)  # batch_scan_producer → batch_scan_consumer

        self.tippathvisualQueue = Queue(5)  # main program  → tip_path_visualization

        self.PyQtimformationQueue = Queue(5)  # PyQt_GUI → main program
        self.PyQtPulseQueue = Queue(5)  # PyQt_GUI → HandPulsethreading
        self.PyQtTipShaperQueue = Queue(5)  # PyQt_GUI → TipShaperthreading

        self.scanqulityqueue = Queue()  # TipShaperthreading → main program

        # Initialize molecule registry for tracking detected and manipulated molecules
        self.molecule_registry = Registry()

        self.mol_tip_induce_path = self.log_path + "/mol_tip_induce"

        # Initialize SAC-related paths if needed  ##################################### modify to add SAC agent
        self.SAC_buffer_path = self.log_path + "/SAC_buffer"
        self.SAC_aug_buffer_path = self.log_path + "/SAC_aug_buffer"
        self.SAC_origin_action_path = self.log_path + "/SAC_actions"

        # Initialize manipulation parameters
        self.tip_manipulate_speed = "2n"  # Speed for tip manipulation movements
        self.SAC_XYaction_edge = 50  # Action space edge for XY manipulation
        self.SAC_XYaction_100 = 100  # Action space scaling factor
        self.zoom_in_100 = 100  # Zoom scaling factor

        # # Initialize communication queues for SAC agent if needed
        # self.auto2SAC_queue = Queue(10)  # Main program → SAC agent
        # self.SAC2auto_queue = Queue(10)  # SAC agent → Main program
        # self.q_save_exp = Queue(100)  # Experience saving queue

    def is_serializable(self, value):
        """Attempt to serialize the value, return True if serializable, False otherwise."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

        
    def save_checkpoint(self):
        """ Save the current state of serializable instance attributes to a file. """
        serializable_attrs = {k: v for k, v in self.__dict__.items() if self.is_serializable(v)}
        filename = os.path.join(self.log_path,'checkpoint.json')
        with open(filename, 'w') as file:
            json.dump(serializable_attrs, file)

    def create_annotated_image(
        self,
        image_for_rgb,
        mol_position_nm,
        manipulation_position=None,
        manipulation_success=False,
        spectroscopy_points=None,
        spectroscopy_results=None,
        mol_count=0,
        scan_center=None,
        scan_scale=None,
        is_manipulated=False,
    ):
        """
        Create an annotated STM image with molecule center, manipulation points, and spectroscopy points.

        Args:
            image_for_rgb: RGB image to annotate
            mol_position_nm: molecule position in meters
            manipulation_position: manipulation position in meters (optional)
            manipulation_success: whether manipulation was successful
            spectroscopy_points: list of spectroscopy points in meters (optional)
            spectroscopy_results: list of spectroscopy results (optional)
            mol_count: molecule index for labeling
            scan_center: scan center coordinates in meters
            scan_scale: scan scale in nanometers
            is_manipulated: whether this molecule has been manipulated

        Returns:
            numpy.ndarray: annotated image
        """
        # Convert to color image for better visualization
        if len(image_for_rgb.shape) == 2:
            vis_image = cv2.cvtColor(image_for_rgb, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image_for_rgb.copy()

        # Convert scan_scale to meters if it's in nanometers
        if scan_scale > 1e-6:  # Assume it's in nanometers if > 1 micrometer
            edge_meters = scan_scale * 1e-9
        else:
            edge_meters = scan_scale

        image_height, image_width = vis_image.shape[:2]

        def real_to_pixel_coords(real_x: float, real_y: float):
            """Convert real-world coordinates to pixel coordinates"""
            # Normalize to 0-1 based on scan area
            norm_x = (real_x - scan_center[0]) / edge_meters + 0.5
            norm_y = (real_y - scan_center[1]) / edge_meters + 0.5

            # Convert to pixel coordinates
            pixel_x = int(norm_x * image_width)
            # NOTE: y_pixel 0 is at the top, which is inverse to norm_y
            # So we need to flip the y coordinate: y_pixel = height - norm_y * height
            pixel_y = int((1.0 - norm_y) * image_height)

            # Ensure within bounds
            pixel_x = max(0, min(image_width - 1, pixel_x))
            pixel_y = max(0, min(image_height - 1, pixel_y))

            return pixel_x, pixel_y

        # Draw molecule center
        mol_x, mol_y = (
            mol_position_nm[0],
            mol_position_nm[1],
        )  # Real coordinates in meters
        
        # Convert molecule position to pixel coordinates
        mol_px, mol_py = real_to_pixel_coords(mol_x, mol_y)

        # Determine molecule box size and colors based on manipulation status
        mol_w = 2.0  # Default molecule width in nm
        mol_h = 2.0  # Default molecule height in nm
        
        # Calculate bounding box in pixels
        box_w_px = max(10, int((mol_w / (edge_meters * 1e9)) * image_width))
        box_h_px = max(10, int((mol_h / (edge_meters * 1e9)) * image_height))

        # Use red rectangle for all molecules as requested
        box_color = (0, 0, 255)  # Red (BGR format)
        box_thickness = 2
        center_color = (0, 0, 255)  # Red center
        center_radius = 3
        label_color = (0, 0, 255)  # Red label
        
        # Determine label prefix based on manipulation status
        if is_manipulated or manipulation_success:
            label_prefix = "M-Manip"
        else:
            label_prefix = "M"

        # Draw molecule bounding rectangle
        cv2.rectangle(
            vis_image,
            (mol_px - box_w_px // 2, mol_py - box_h_px // 2),
            (mol_px + box_w_px // 2, mol_py + box_h_px // 2),
            box_color,
            box_thickness,
        )

        # Draw molecule center
        cv2.circle(vis_image, (mol_px, mol_py), center_radius, center_color, -1)

        # Add molecule ID with status
        cv2.putText(
            vis_image,
            f"{label_prefix}{mol_count+1}",
            (mol_px + 8, mol_py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            label_color,
            1,
        )

        # Draw manipulation point if manipulation was performed
        if manipulation_position is not None:
            manip_x, manip_y = manipulation_position[0], manipulation_position[1]
            manip_px, manip_py = real_to_pixel_coords(manip_x, manip_y)

            if manipulation_success:
                manip_color = (0, 255, 0)  # Green for successful manipulation
                marker_size = 6
                marker_type = cv2.MARKER_TRIANGLE_UP
                label = "Manipulation success"
            else:
                manip_color = (0, 165, 255)  # Orange for failed manipulation
                marker_size = 5
                marker_type = cv2.MARKER_TRIANGLE_DOWN
                label = "Manipulation failed"

            # Draw manipulation marker
            cv2.drawMarker(vis_image, (manip_px, manip_py), manip_color, marker_type, marker_size, 2)
            
            # Add manipulation status text
            cv2.putText(
                vis_image,
                label,
                (manip_px + 8, manip_py + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                manip_color,
                1,
            )

        # Draw spectroscopy points as blue points
        if spectroscopy_points:
            print(f"Drawing {len(spectroscopy_points)} spectroscopy points as blue points")
            
            for spec_idx, spec_point in enumerate(spectroscopy_points):
                spec_x, spec_y = spec_point[0], spec_point[1]
                spec_px, spec_py = real_to_pixel_coords(spec_x, spec_y)

                # Use blue color for all spectroscopy points as requested
                spec_color = (255, 0, 0)  # Blue (BGR format)
                
                # Determine marker based on success status
                if spectroscopy_results and spec_idx < len(spectroscopy_results):
                    spec_result = spectroscopy_results[spec_idx]
                    if spec_result.get("success", False):
                        marker_size = 6
                        status_symbol = "✓"
                    else:
                        marker_size = 4
                        status_symbol = "✗"
                else:
                    marker_size = 5
                    status_symbol = ""

                # Draw blue circle for spectroscopy point
                cv2.circle(vis_image, (spec_px, spec_py), marker_size, spec_color, -1)
                
                # Draw outer circle for better visibility
                cv2.circle(vis_image, (spec_px, spec_py), marker_size + 2, spec_color, 1)
                
                # Add spectroscopy point label
                label_text = f"S{spec_idx+1}{status_symbol}"
                cv2.putText(
                    vis_image,
                    label_text,
                    (spec_px + 8, spec_py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    spec_color,
                    1,
                )

                # Draw thin line connecting molecule center to spectroscopy point
                cv2.line(vis_image, (mol_px, mol_py), (spec_px, spec_py), spec_color, 1, cv2.LINE_AA)

                print(f"  Spectroscopy point {spec_idx+1}: ({spec_x*1e9:.2f}, {spec_y*1e9:.2f}) nm -> pixel ({spec_px}, {spec_py})")

        # Add scan center marker
        if scan_center is not None:
            center_px, center_py = real_to_pixel_coords(scan_center[0], scan_center[1])
            cv2.drawMarker(
                vis_image, (center_px, center_py), (255, 255, 255), cv2.MARKER_CROSS, 12, 2
            )
            cv2.putText(
                vis_image,
                "Scan Center",
                (center_px + 15, center_py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
            )

        # Add legend in top-left corner
        legend_y = 20
        legend_spacing = 18
        
        cv2.putText(vis_image, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += legend_spacing
        
        # Molecule legend - always red box
        cv2.putText(vis_image, "Red Box: Molecule", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        legend_y += legend_spacing
        
        # Spectroscopy legend - always blue points
        if spectroscopy_points:
            cv2.putText(vis_image, "Blue Point: Spectroscopy", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        return vis_image

    def save_molecule_results(
        self,
        result_data,
        annotated_image,
        mol_count,
        image_save_time,
        spectroscopy_results_molecule=None,
        spectroscopy_points=None,
    ):
        """
        Save comprehensive results for a processed molecule including annotated image and data files.
        All files for each molecule are saved in a dedicated directory.

        Args:
            result_data: dictionary containing all result information
            annotated_image: annotated STM image
            mol_count: molecule index
            image_save_time: timestamp for file naming
            spectroscopy_results_molecule: list of spectroscopy results (optional)
            spectroscopy_points: list of spectroscopy points (optional)

        Returns:
            dict: file paths where results were saved
        """
        # Create dedicated directory for this molecule (or use existing one)
        scan_info = result_data['scan_info']
        if scan_info.get('molecule_folder') or scan_info.get('manipulation_folder') or scan_info.get('spectroscopy_folder'):
            # Use existing molecule folder if provided (check multiple possible keys)
            molecule_folder = (scan_info.get('molecule_folder') or 
                             scan_info.get('manipulation_folder') or 
                             scan_info.get('spectroscopy_folder'))
            print(f"    Using existing molecule folder: {molecule_folder}")
        else:
            # Create new dedicated directory for this molecule
            molecule_folder = (
                self.mol_tip_induce_path
                + f"/scan_{result_data['scan_info']['scan_count']}_mol_{mol_count+1}_{image_save_time}/"
            )
            if not os.path.exists(molecule_folder):
                os.makedirs(molecule_folder)
            print(f"    Created new molecule folder: {molecule_folder}")

        # Save annotated image in molecule folder
        mol_image_path = molecule_folder + f"annotated_scan_mol_{mol_count+1}.png"
        cv2.imwrite(mol_image_path, annotated_image)

        # Save detailed JSON result for this molecule
        json_result_path = molecule_folder + f"molecule_{mol_count+1}_results.json"
        with open(json_result_path, "w") as f:
            json.dump(result_data, f, indent=2, default=str)

        file_paths = {
            "molecule_folder": molecule_folder,
            "annotated_image": mol_image_path,
            "result_data": json_result_path,
            "spectroscopy_files": [],
        }

        # Save individual spectroscopy data files if spectroscopy was performed
        if spectroscopy_results_molecule and spectroscopy_points:
            print(f"    Saving {len(spectroscopy_results_molecule)} spectroscopy results to {molecule_folder}")

            for spec_idx, spec_result in enumerate(spectroscopy_results_molecule):
                # Save individual spectroscopy JSON data
                spec_json_path = molecule_folder + f"spectroscopy_point_{spec_idx+1}.json"
                spec_data = {
                    "measurement_info": {
                        "scan_count": result_data["scan_info"]["scan_count"],
                        "molecule_index": mol_count + 1,
                        "point_index": spec_idx + 1,
                        "timestamp": image_save_time,
                        "position": spectroscopy_points[spec_idx],
                        "success": spec_result.get("success", False),
                    },
                    "spectroscopy_data": spec_result,
                    "molecule_context": result_data["molecule_info"],
                }
                with open(spec_json_path, "w") as f:
                    json.dump(spec_data, f, indent=2, default=str)

                # Files should already be in molecule folder, so just verify paths
                if spec_result.get("success", False):
                    # Verify the files exist in the molecule folder
                    dat_file = spec_result.get("filename")
                    vis_file = spec_result.get("vis_filename")
                    
                    if not dat_file or not os.path.exists(dat_file):
                        print(f"      Warning: Spectroscopy data file not found: {dat_file}")
                    if not vis_file or not os.path.exists(vis_file):
                        print(f"      Warning: Visualization file not found: {vis_file}")
                else:
                    dat_file = None
                    vis_file = None

                file_paths["spectroscopy_files"].append({
                    "point_index": spec_idx + 1,
                    "json_file": spec_json_path,
                    "dat_file": dat_file,
                    "vis_file": vis_file,
                })

            print(f"    All spectroscopy files organized in: {molecule_folder}")

        return file_paths

    def perform_spectroscopy_measurement(
        self,
        spectroscopy_point,
        molecular_index,
        point_index,
        image_save_time,
        image_with_molecule,
        molecule_folder=None,
        scan_center=None,
    ):
        """
        Perform bias spectroscopy measurement at a specific point with comprehensive error handling
        
        Args:
            spectroscopy_point: (x, y) coordinates for measurement in meters
            molecular_index: Index of the molecule
            point_index: Index of the spectroscopy point
            image_save_time: Timestamp for file naming
            image_with_molecule: STM image with molecule annotations
            molecule_folder: Directory path to save spectroscopy files (optional)
            scan_center: Center coordinates of the scan area in meters (optional)

        Returns:
            dict: Measurement results and metadata
        """
        print(f"Starting spectroscopy measurement at point {point_index+1}")
        print(f"Position: ({spectroscopy_point[0]*1e9:.2f}, {spectroscopy_point[1]*1e9:.2f}) nm")
        print(f"Molecule index: {molecular_index}")

        measurement_start_time = time.time()

        try:
            # Validate input coordinates
            if not isinstance(spectroscopy_point, (tuple, list)) or len(spectroscopy_point) != 2:
                raise ValueError("spectroscopy_point must be a tuple/list of (x, y) coordinates")
            
            # Check system status before measurement
            initial_status = self.BiasSpectrStatusGet()
            if initial_status.get('Status', 1) != 0:
                print(f"Warning: Initial spectroscopy status: {initial_status}")
            
            # Move tip to spectroscopy point with enhanced error checking
            print(f"Moving tip to spectroscopy position...")
            try:
                # FolMeSpeedSet expects (Speed in m/s, Custom_speed flag 0/1)
                # Convert zoom_in_tip_speed to proper speed value and use custom speed
                tip_speed = self.convert(self.zoom_in_tip_speed)  # Convert "4n" to meters/second
                self.FolMeSpeedSet(tip_speed, 1)  # Use custom speed mode
                self.TipXYSet(spectroscopy_point[0], spectroscopy_point[1])
                self.FolMeSpeedSet(tip_speed, 0)  # Reset to default speed

                # Allow tip to stabilize with status monitoring
                print("Stabilizing tip position...")
                time.sleep(1.5)  # Increased stabilization time
                
                # Verify tip position if possible
                try:
                    current_pos = self.TipXYGet()
                    pos_error = np.sqrt((current_pos[0] - spectroscopy_point[0])**2 + 
                                      (current_pos[1] - spectroscopy_point[1])**2)
                    if pos_error > 1e-9:  # 1 nm tolerance
                        print(f"Warning: Tip positioning error: {pos_error*1e9:.2f} nm")
                except:
                    print("Could not verify tip position")
                    
            except Exception as e:
                raise Exception(f"Tip positioning failed: {e}")

            # Configure spectroscopy parameters with validation
            spectroscopy_bias_start = self.bias_spectroscopy_start
            spectroscopy_bias_end = self.bias_spectroscopy_end
            spectroscopy_points = self.spectr_points
            spectroscopy_channels = [0, 22]  # Current (A), Lock-in signal

            print(f"Configuring spectroscopy:")
            print(f"  Bias range: {spectroscopy_bias_start}V to {spectroscopy_bias_end}V")
            print(f"  Points: {spectroscopy_points}")
            print(f"  Channels: {spectroscopy_channels}")
            
            # Configure spectroscopy parameters with error checking
            try:
                channel_result = self.BiasSpectrChsSet(spectroscopy_channels)
                limits_result = self.BiasSpectrLimitsSet(spectroscopy_bias_start, spectroscopy_bias_end)
                print("Spectroscopy configuration successful")
            except Exception as e:
                raise Exception(f"Spectroscopy configuration failed: {e}")

            # Perform the spectroscopy measurement
            save_base_name = f"bias_spec_{image_save_time}_mol_{molecular_index}_pt_{point_index}"
            print(f"Starting measurement: {save_base_name}")
            
            try:
                spectro_result = self.BiasSpectrStart(get_data=1, save_base_name=save_base_name)
                measurement_time = time.time() - measurement_start_time
                print(f"Measurement completed in {measurement_time:.2f} seconds")
                
            except Exception as e:
                raise Exception(f"BiasSpectrStart failed: {e}")
            
            # Validate and extract measurement data
            if not isinstance(spectro_result, dict):
                raise Exception(f"Invalid result type: {type(spectro_result)}")
                
            if "data" not in spectro_result:
                raise Exception("No data field in spectroscopy result")
                
            spectroscopy_data = spectro_result["data"]
            channels_names = spectro_result.get("channels_names", [f"Channel_{i}" for i in spectroscopy_channels])
            
            # Validate data quality
            if not isinstance(spectroscopy_data, np.ndarray):
                raise Exception(f"Invalid data type: {type(spectroscopy_data)}")
                
            if spectroscopy_data.size == 0:
                raise Exception("Empty spectroscopy data array")
                
            print(f"Data validation successful:")
            print(f"  Shape: {spectroscopy_data.shape}")
            print(f"  Channels: {channels_names}")
            print(f"  Data range: [{np.min(spectroscopy_data):.2e}, {np.max(spectroscopy_data):.2e}]")
            
            # Check final measurement status
            final_status = self.BiasSpectrStatusGet()
            print(f"Final status: {final_status}")
            if final_status.get('Status', 1) != 0:
                print(f"Warning: Non-zero final status: {final_status}")
                # Don't raise error for non-zero status as it might be normal

            # Create enhanced visualization image with tip position
            vis_image = image_with_molecule.copy()

            # Get coordinate conversion parameters with proper defaults
            if scan_center is None:
                scan_center = getattr(self, 'nanocoodinate', (0.0, 0.0))
            
            # Use scan frame size for coordinate conversion
            try:
                zoom_scale = self.convert(self.small_scan_frame_size)  # Use grid scan size in meters
            except:
                zoom_scale = self.convert(self.scan_zoom_in_list[0])  # Fallback to zoom list
                
            edge_meters = zoom_scale
            image_height, image_width = vis_image.shape[:2]

            def real_to_pixel_coords(real_x: float, real_y: float):
                """Convert real-world coordinates to pixel coordinates with bounds checking"""
                try:
                    # Normalize to 0-1 based on scan area
                    norm_x = (real_x - scan_center[0]) / edge_meters + 0.5
                    norm_y = (real_y - scan_center[1]) / edge_meters + 0.5

                    # Convert to pixel coordinates
                    pixel_x = int(norm_x * image_width)
                    pixel_y = int((1.0 - norm_y) * image_height)  # Flip y-axis

                    # Ensure within bounds
                    pixel_x = max(0, min(image_width - 1, pixel_x))
                    pixel_y = max(0, min(image_height - 1, pixel_y))

                    return pixel_x, pixel_y
                except:
                    # Return center if conversion fails
                    return image_width // 2, image_height // 2

            # Convert spectroscopy point coordinates to pixel coordinates
            tip_pixel_x, tip_pixel_y = real_to_pixel_coords(
                spectroscopy_point[0], spectroscopy_point[1]
            )

            # Draw enhanced tip position visualization
            cv2.circle(vis_image, (tip_pixel_x, tip_pixel_y), 6, (0, 0, 255), -1)  # Red filled circle
            cv2.circle(vis_image, (tip_pixel_x, tip_pixel_y), 8, (255, 255, 255), 2)  # White border
            
            # Add measurement label with background
            label_text = f"Spec{point_index+1}"
            label_pos = (tip_pixel_x + 10, tip_pixel_y - 10)
            
            # Add text background for better visibility
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(vis_image, 
                         (label_pos[0]-2, label_pos[1]-text_height-2),
                         (label_pos[0]+text_width+2, label_pos[1]+baseline+2),
                         (0, 0, 0), -1)  # Black background
            
            cv2.putText(vis_image, label_text, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # Cyan text

            # Prepare save paths with enhanced organization
            if molecule_folder and os.path.exists(molecule_folder):
                spectroscopy_save_path = molecule_folder
                print(f"Saving to molecule folder: {os.path.basename(molecule_folder)}")
            else:
                spectroscopy_save_path = self.mol_tip_induce_path + "/spectroscopy/"
                if not os.path.exists(spectroscopy_save_path):
                    os.makedirs(spectroscopy_save_path)
                print(f"Saving to default spectroscopy folder")

            # Save enhanced visualization
            vis_filename = os.path.join(spectroscopy_save_path, 
                                      f"spectroscopy_vis_mol{molecular_index}_pt{point_index}.png")
            cv2.imwrite(vis_filename, vis_image)

            # Create comprehensive data file with metadata
            data_filename = os.path.join(spectroscopy_save_path, 
                                       f"bias_spec_mol{molecular_index}_pt{point_index}.dat")


            # Save data with proper formatting
            try:
                if spectroscopy_data.ndim == 1:
                    # Reshape 1D data to column vector
                    save_data = spectroscopy_data.reshape(-1, 1)
                elif spectroscopy_data.ndim == 2:
                    # Transpose if channels are rows (make columns for easier reading)
                    if spectroscopy_data.shape[0] < spectroscopy_data.shape[1]:
                        save_data = spectroscopy_data.T
                    else:
                        save_data = spectroscopy_data
                else:
                    # Flatten higher-dimensional data
                    save_data = spectroscopy_data.reshape(spectroscopy_data.shape[0], -1)
                
                # Create header with metadata
                header = "# Bias Spectroscopy Data\n"
                header += f"# Molecule Index: {molecular_index}\n"
                header += f"# Point Index: {point_index}\n"
                header += f"# Position (m): {spectroscopy_point[0]:.6e}, {spectroscopy_point[1]:.6e}\n"
                header += f"# Bias Range (V): {spectroscopy_bias_start} to {spectroscopy_bias_end}\n"
                header += f"# Points: {spectroscopy_points}\n"
                header += f"# Channels: {', '.join(channels_names)}\n"
                header += f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                header += f"# Data Shape: {save_data.shape}\n"
                
                # Write data to file with header
                np.savetxt(data_filename, save_data, delimiter='\t', header=header, 
                          fmt='%.6e', comments='')
                
                print(f"Data saved: {save_data.shape} to {os.path.basename(data_filename)}")
                
            except Exception as e:
                print(f"Warning: Data save failed: {e}")
                # Save as pickle as fallback
                fallback_filename = data_filename.replace('.dat', '.pkl')
                with open(fallback_filename, 'wb') as f:
                    pickle.dump({
                        'data': spectroscopy_data,
                        'channels': channels_names,
                    }, f)
                print(f"Data saved as pickle fallback: {os.path.basename(fallback_filename)}")

            # Calculate basic statistics for validation
            if spectroscopy_data.size > 0:
                data_stats = {
                    'mean': np.mean(spectroscopy_data, axis=-1).tolist(),
                    'std': np.std(spectroscopy_data, axis=-1).tolist(),
                    'min': np.min(spectroscopy_data, axis=-1).tolist(),
                    'max': np.max(spectroscopy_data, axis=-1).tolist()
                }
            else:
                data_stats = {}

            print(f"Measurement completed successfully!")
            print(f"Files saved:")
            print(f"  Data: {os.path.basename(data_filename)}")
            print(f"  Visualization: {os.path.basename(vis_filename)}")

            return {
                "success": True,
                "filename": data_filename,
                "vis_filename": vis_filename,
                "data": spectroscopy_data,
                "position": spectroscopy_point,
                "channels": channels_names,
                "bias_range": (spectroscopy_bias_start, spectroscopy_bias_end),
                "measurement_type": "bias_spectroscopy",
                "measurement_duration": measurement_time,
                "data_statistics": data_stats,
                "point_index": point_index,
                "molecular_index": molecular_index,
                "save_path": spectroscopy_save_path
            }

        except Exception as e:
            error_msg = str(e)
            print(f"Spectroscopy measurement failed: {error_msg}")
            
            # Try to get current status for debugging
            try:
                current_status = self.BiasSpectrStatusGet()
                print(f"Current spectroscopy status: {current_status}")
            except:
                print("Could not retrieve spectroscopy status")
            
            # Create error log with detailed information
            error_details = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "position": spectroscopy_point,
                "molecular_index": molecular_index,
                "point_index": point_index,
                "error": error_msg,
                "measurement_duration": time.time() - measurement_start_time
            }
            
            return {
                "success": False,
                "error": error_msg,
                "error_details": error_details,
                "position": spectroscopy_point,
                "measurement_type": "bias_spectroscopy",
                "point_index": point_index,
                "molecular_index": molecular_index
            }

    def load_checkpoint(self):
        """Loads class attributes from a json file and updates the instance."""
        checkpoint_json = get_latest_checkpoint(self.main_data_save_path)
        with open(checkpoint_json, "r") as file:
            data = json.load(file)
            self.__dict__.update(data)
        checkpoint_tip_img = get_latest_checkpoint(
            self.main_data_save_path, checkpoint_name="tip_path.jpg"
        )
        self.tip_path_img = cv2.imread(checkpoint_tip_img, cv2.IMREAD_COLOR)

    # def a function to initialize the nanonis controller, set the tip to the center of the scan frame
    # mode = 'new' : the tip is initialized to the center and create a new log folder, mode = 'latest' : load the latest checkpoint

    def tip_init(self, mode="new"):
        self.mode = mode
        if mode == "new":
            self.ScanPause()
            self.ZCtrlOnSet()
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)

        elif mode == "latest":
            self.ScanPause()
            self.ZCtrlOnSet()  # TODO
            self.load_checkpoint()

    def SafeTipthreading(self, SafeTipthrehold="5n", safe_withdraw_step=100):
        print("The tipsafe monitoring activated")
        # if the type of SafeTipthrehold is string, convert it to float
        if type(SafeTipthrehold) == str:
            SafeTipthrehold = self.convert(SafeTipthrehold)

        current_list = []
        while True:
            time.sleep(0.5)
            try:
                current = self.CurrentGet()
            except:
                current = 0
            current_list.append(current)
            # print(current_list)
            if len(current_list) >= 8:
                current_list.pop(0)
                # if all the current absoulte value is bigger than SafeTipthrehold, stop the scan and withdraw the tip than move Motor Z- 50? steps
                if all(abs(current) > SafeTipthrehold for current in current_list):
                    # raise ValueError('The tunneling current is too large, the tip protection is activated, and the scan stops.')
                    print(
                        "The tunneling current is too large, the tip protection is activated."
                    )
                    self.ScanStop()
                    self.Withdraw()
                    self.MotorMoveSet("Z-", safe_withdraw_step)
                    current_list = []
                    print("The tip is withdrawed")
                    break

    def tip_path_visualization(self):
        """
        Visualize tip path - with error handling for headless environments
        """
        try:
            print("Starting tip path visualization...")
            
            square_color_hex = "#BABABA"  # good image color
            square_bad_color_hex = "#FE5E5E"  # bad image color
            line_color_hex = "#8AAEFA"  # tip path line color
            border_color_hex = "#64FF00"  # the border color of the whole plane
            scan_border_color_hex = (
                "#FFC8CB"  # usually the area of the 70% of the whole plane
            )

            sample_bad_color_hex = (
                "#FF6E6E"  # FF6E6E       #000000      # the color of the bad LineScan data
            )

            color_max = "#385723"  # 385723  #C55A11
            color_min = "#C5E0B4"  # C5E0B4  #2E75B6

            border_color = Hex_to_BGR(border_color_hex)
            scan_border_color = Hex_to_BGR(scan_border_color_hex)
            square_good_color = Hex_to_BGR(square_color_hex)
            square_bad_color = Hex_to_BGR(square_bad_color_hex)
            sample_bad_color = Hex_to_BGR(sample_bad_color_hex)
            line_color = Hex_to_BGR(line_color_hex)

            cv2.namedWindow("Tip Path", cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("Tip Path", 800, 800)

            # creat a 400*400 pix white image by numpy array
            cv2.rectangle(
                self.tip_path_img,
                (0, 0),
                (self.plane_size, self.plane_size),
                border_color,
                10,
            )  # draw the border of the plane, which is the same color as the nanonis scan border on the scan controloer

            scan_border_left_top = round(self.plane_size / 2 * (1 - self.real_scan_factor))
            scan_border_right_bottom = round(
                self.plane_size / 2 * (1 + self.real_scan_factor)
            )
            cv2.rectangle(
                self.tip_path_img,
                (scan_border_left_top, scan_border_left_top),
                (scan_border_right_bottom, scan_border_right_bottom),
                scan_border_color,
                10,
            )

            while True:
                try:
                    if not self.tippathvisualQueue.empty():
                        circle = self.tippathvisualQueue.get_nowait()

                        if (
                            len(circle) <= 2
                        ):  # data is a point which is the point of ready to scan
                            cv2.circle(
                                self.tip_path_img,
                                (round(circle[0]), round(circle[1])),
                                round(self.scan_square_edge / 5),
                                (255, 0, 0),
                                -1,
                            )
                        elif (
                            len(circle) >= 4
                        ):  # data is a circle which is the circle of already scanned
                            left_top, right_bottom = center_to_square(
                                (circle[0], circle[1]), self.scan_square_edge
                            )
                            t = circle[4]  # the coverge fiactor of the image
                            t = 0.5
                            cover_color = interpolate_colors(color_min, color_max, t)

                            if (
                                self.skip_flag == 1
                            ):  # if the scan is skiped, the color of the square is set on
                                cover_color = sample_bad_color
                            if circle[3] == 1:
                                square_color = square_good_color  # good image color
                            elif circle[3] == 0:
                                square_color = square_bad_color  # bad image color

                            if len(self.circle_list) == 1:
                                cv2.rectangle(
                                    self.tip_path_img, left_top, right_bottom, cover_color, -1
                                )  # the box will be colored
                                cv2.rectangle(
                                    self.tip_path_img, left_top, right_bottom, square_color, 3
                                )
                                self.visual_circle_buffer_list.append(
                                    circle
                                )  # add the first circle to the self.visual_circle_buffer_list

                            elif len(self.circle_list) > 1:
                                (Xn_1, Yn_1) = (
                                    self.visual_circle_buffer_list[-1][0],
                                    self.visual_circle_buffer_list[-1][1],
                                )
                                cv2.rectangle(
                                    self.tip_path_img, left_top, right_bottom, cover_color, -1
                                )  # the box will be colored
                                cv2.rectangle(
                                    self.tip_path_img, left_top, right_bottom, square_color, 3
                                )  # the edge of the box
                                cv2.line(
                                    self.tip_path_img,
                                    (round(Xn_1), round(Yn_1)),
                                    (round(circle[0]), round(circle[1])),
                                    line_color,
                                    4,
                                )  # ues the last circle center to draw the line to show the tip path
                                self.visual_circle_buffer_list.append(
                                    circle
                                )  # add the new circle to the self.visual_circle_buffer_list
                                # delete the first circle in the self.visual_circle_buffer_list
                                if len(self.visual_circle_buffer_list) > 2:
                                    self.visual_circle_buffer_list.pop(0)

                            else:
                                raise ValueError("the length of the circle list is not right")

                            cv2.imwrite(self.log_path + "/tip_path" + ".jpg", self.tip_path_img)
                            
                        elif circle == "end":
                            # save the tip path image
                            cv2.imwrite(self.log_path + "/tip_path" + ".jpg", self.tip_path_img)
                            print(
                                "The tip path image is saved as "
                                + self.log_path
                                + "/tip_path"
                                + ".jpg"
                            )
                            break

                        else:
                            raise ValueError("the circle data is not a point or a circle")

                        cv2.imshow("Tip Path", self.tip_path_img)

                    cv2.waitKey(100)
                    
                except Exception as e:
                    print(f"Error in tip path visualization loop: {e}")
                    # Continue the loop even if there's an error
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            print(f"Error starting tip path visualization: {e}")
            print("Tip path visualization disabled - continuing without visualization")
            # Just consume queue items to prevent blocking
            while True:
                try:
                    if not self.tippathvisualQueue.empty():
                        self.tippathvisualQueue.get_nowait()
                    time.sleep(0.1)
                except:
                    pass

    # activate all the threads of monitor
    def monitor_thread_activate(self):
        Safe_Tip_thread = threading.Thread(
            target=self.SafeTipthreading, args=("5n", 100), daemon=True
        )  # the SafeTipthrehold is 5n, the safe_withdraw_step is 100
        tip_visualization_thread = threading.Thread(
            target=self.tip_path_visualization, daemon=True
        )  # the tip path visualization thread
        batch_scan_consumer_thread = threading.Thread(
            target=self.batch_scan_consumer, daemon=True
        )  # the batch scan consumer thread

        Safe_Tip_thread.start()
        tip_visualization_thread.start()  # Re-enable tip visualization thread
        batch_scan_consumer_thread.start()

    def create_dummy_scan_data(self, Scan_pix=304):
        """Create realistic dummy scan data for simulation mode"""
        # Create more realistic height data with some structure
        np.random.seed(42)  # For reproducible results
        base_height = 1e-10  # 1 Angstrom baseline
        noise_level = 1e-11  # 0.1 Angstrom noise

        # Create a base pattern with some topographic features
        x = np.linspace(-1, 1, Scan_pix)
        y = np.linspace(-1, 1, Scan_pix)
        X, Y = np.meshgrid(x, y)

        # Add some atomic-scale features
        data = base_height + noise_level * np.random.random((Scan_pix, Scan_pix))
        data += 5e-12 * np.sin(X * 10) * np.cos(Y * 10)  # Regular atomic lattice
        data += 2e-11 * np.exp(
            -((X - 0.2) ** 2 + (Y + 0.1) ** 2) / 0.1
        )  # Defect or molecule

        return {
            "data": data.astype(np.float64),
            "row": Scan_pix,
            "col": Scan_pix,
            "scan_direction": 0,
            "channel_name": "Z (m)",
        }

    def batch_scan_producer(self, Scan_posion = (0.0 ,0.0), Scan_edge = "30n", Scan_pix = 304 ,  angle = 0):
        # Check if in simulation mode and handle accordingly
        if hasattr(self, 'code_simulation_mode') and self.code_simulation_mode:
            print('Simulation mode: loading test image')
            # Load simulation image if available
            if hasattr(self, 'simulation_image_path'):
                try:
                    sim_image = cv2.imread(self.simulation_image_path, cv2.IMREAD_GRAYSCALE)
                    if sim_image is not None:
                        Scan_data_for = {
                            'data': sim_image,  # Normalize to 0-1
                            'row': Scan_pix,
                            'col': Scan_pix,
                            'scan_direction': 0,
                            'channel_name': 'Z (m)',
                        }
                        Scan_data_back = Scan_data_for.copy()  # Use same image for both directions
                        print(f'Loaded: {self.simulation_image_path}')
                    else:
                        print('Failed to load image, using default')
                        Scan_data_for = {'data': np.ones((Scan_pix, Scan_pix), np.float32) * 0.5,
                            'row': Scan_pix, 'col': Scan_pix, 'scan_direction': 0, 'channel_name': 'Z (m)'}
                        Scan_data_back = Scan_data_for.copy()
                except Exception as e:
                    print(f'Error loading image: {e}')
                    Scan_data_for = {'data': np.ones((Scan_pix, Scan_pix), np.float32) * 0.5,
                        'row': Scan_pix, 'col': Scan_pix, 'scan_direction': 0, 'channel_name': 'Z (m)'}
                    Scan_data_back = Scan_data_for.copy()
            else:
                print('Using default test data')
                Scan_data_for = {'data': np.ones((Scan_pix, Scan_pix), np.float32) * 0.5,
                    'row': Scan_pix, 'col': Scan_pix, 'scan_direction': 0, 'channel_name': 'Z (m)'}
                Scan_data_back = Scan_data_for.copy()
        
        elif self.skip_flag == 1:
            print('creating skip data...')
            Scan_data_for = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                'row': Scan_pix,
                'col': Scan_pix,
                'scan_direction': 0,
                'channel_name': 'Z (m)',
                }
            Scan_data_back = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                'row': Scan_pix,
                'col': Scan_pix,
                'scan_direction': 0,
                'channel_name': 'Z (m)',
                }
        else:
            print('Scaning the image...')
            # ScanBuffer = self.ScanBufferGet()
            self.ScanBufferSet(Scan_pix, Scan_pix, self.signal_channel_list) # 14 is the index of the Z_m channel in real scan mode , in demo mode, the index of the Z_m channel is 30
            self.ScanPropsSet(Continuous_scan = 2 , Bouncy_scan = 2,  Autosave = 1, Series_name = ' ', Comment = 'inter_closest')# close continue_scan & bouncy_scan, but save all data
            self.ScanFrameSet(Scan_posion[0], Scan_posion[1], Scan_edge, Scan_edge, angle=angle)
            
            self.ScanStart()
            time.sleep(0.5)

            # while self.ScanStatusGet() == 1: # detect the scan status until the scan is complete.
            #     #   !!!   Note：Do not use self.WaitEndOfScan() here   !!!!, it will block the program!!!!!!
            #     time.sleep(0.5)
            
            self.WaitEndOfScan() # wait for the scan to be complete

            try:   # some times the scan data is not successful because of the TCP/IP communication problem
                Scan_data_for = self.ScanFrameData(self.signal_channel_list[-1], data_dir=1)
            except: #if is not successful, set fake data
                Scan_data_for = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                                'row': Scan_pix,
                                'col': Scan_pix,
                                'scan_direction': 0,
                                'channel_name': 'Z (m)',
                                }
            time.sleep(1)
            try:
                Scan_data_back = self.ScanFrameData(self.signal_channel_list[-1], data_dir=0)
            except:
                Scan_data_back = {'data': np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                                'row': Scan_pix,
                                'col': Scan_pix,
                                'scan_direction': 0,
                                'channel_name': 'Z (m)',
                                }
            # if the first element and the last element of the Scan_data_for and Scan_data_back is NaN, the scan is not successful
            if np.isnan(Scan_data_for['data'][0][0]) or np.isnan(Scan_data_for['data'][-1][-1]) or np.isnan(Scan_data_back['data'][0][0]) or np.isnan(Scan_data_back['data'][-1][-1]):
                Scan_data_for['data'][np.isnan(Scan_data_for['data'])] = 0
                Scan_data_back['data'][np.isnan(Scan_data_back['data'])] = 0

        self.image_for = linear_normalize_whole(Scan_data_for['data'])                     # image_for and image_back are 2D nparray
        self.image_back = linear_normalize_whole(Scan_data_back['data'])

        self.image_for = images_equalization(self.image_for, alpha=self.equalization_alpha)
        self.image_back = images_equalization(self.image_back, alpha=self.equalization_alpha)

        self.image_for_tensor = torch.tensor(self.image_for, dtype=torch.float32, device=self.device).unsqueeze(0)  # for instance tensor.shape are [1, 1, 256, 256]  which is for DQN
        self.image_back_tensor = torch.tensor(self.image_back, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.Scan_data = {'Scan_data_for':Scan_data_for, 'Scan_data_back':Scan_data_back}
        self.Scan_image = {'image_for':self.image_for, 'image_back':self.image_back}

        self.ScandataQueue.put(self.Scan_data) # put the batch scan data into the queue, blocking if Queue is full
        print('Scaning complete! \n ready to save...')
        return self.Scan_image

    def batch_scan_consumer(self):

        self.npy_data_save_path = (
            self.log_path + "/" + "npy"
        )  # self.log_path = './log/' + self.start_time
        self.image_data_save_path = self.log_path + "/" + "image"
        self.equalization_save_path = self.log_path + "/" + "equalize"

        if not os.path.exists(
            self.equalization_save_path
        ):  # check the save path exist or not
            os.makedirs(self.equalization_save_path)

        if not os.path.exists(self.npy_data_save_path):
            os.makedirs(self.npy_data_save_path)

        if not os.path.exists(self.image_data_save_path):
            os.makedirs(self.image_data_save_path)

        while True:
            if not self.ScandataQueue.empty():
                Scan_data = self.ScandataQueue.get()
                Scan_data_for = Scan_data["Scan_data_for"]["data"]
                Scan_data_back = Scan_data["Scan_data_back"]["data"]
                # preprocess the scan data, and save the scan data and image
                image_for = linear_normalize_whole(Scan_data_for)
                image_back = linear_normalize_whole(Scan_data_back)

                equalization_for = images_equalization(
                    image_for, alpha=self.equalization_alpha
                )
                equalization_back = images_equalization(
                    image_back, alpha=self.equalization_alpha
                )  # equalize the image

                self.image_save_time = time.strftime(
                    "%Y-%m-%d %H-%M-%S", time.localtime(time.time())
                )

                npy_data_save_path_for = (
                    self.npy_data_save_path
                    + "/"
                    + "Scan_data_for_"
                    + self.image_save_time
                    + ".npy"
                )
                npy_data_save_path_back = (
                    self.npy_data_save_path
                    + "/"
                    + "Scan_data_back_"
                    + self.image_save_time
                    + ".npy"
                )
                image_data_save_path_for = (
                    self.image_data_save_path
                    + "/"
                    + "Scan_data_for"
                    + self.image_save_time
                    + ".png"
                )
                image_data_save_path_back = (
                    self.image_data_save_path
                    + "/"
                    + "Scan_data_back"
                    + self.image_save_time
                    + ".png"
                )
                equalization_save_path_for = (
                    self.equalization_save_path
                    + "/"
                    + "Scan_data_for"
                    + self.image_save_time
                    + ".png"
                )
                equalization_save_path_back = (
                    self.equalization_save_path
                    + "/"
                    + "Scan_data_back"
                    + self.image_save_time
                    + ".png"
                )

                cv2.imwrite(equalization_save_path_for, equalization_for)
                cv2.imwrite(
                    equalization_save_path_back, equalization_back
                )  # save the equalization image
                np.save(npy_data_save_path_for, Scan_data_for, allow_pickle=True)
                np.save(npy_data_save_path_back, Scan_data_back, allow_pickle=True)
                cv2.imwrite(image_data_save_path_for, image_for)  # save the image
                cv2.imwrite(image_data_save_path_back, image_back)  # save the image
                cv2.namedWindow("image_for", cv2.WINDOW_NORMAL)
                cv2.namedWindow("image_back", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("image_for", 400, 400)
                cv2.resizeWindow("image_back", 400, 400)
                cv2.imshow("image_for", image_for)
                cv2.imshow("image_back", image_back)
            cv2.waitKey(100)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

    # def a function that content the line scan producer and consumer
    def line_scan_thread_activate(self):
        if self.line_scan_activate == 1:

            if self.AdjustTip_flag == 1:
                scan_continue_time = self.Z_fixed_scan_time
            else:
                # scan_continue_time = self.without_Z_fixed_scan_time
                # scan_continue_time = -(1/200)*self.R**2 + 1.1*self.R
                scan_continue_time = (-(1 / 200) * self.R**2 + 1.1 * self.R) / 3

            # In demo mode, reduce scan time to minimize issues
            if self.nanonis_mode == "demo":
                scan_continue_time = min(
                    scan_continue_time, 5.0
                )  # Max 5 seconds in demo mode
                print(
                    f"Demo mode: Line scan time reduced to {scan_continue_time:.2f} s"
                )
            else:
                print(f"Line scan will continue for {scan_continue_time:.2f} s")
            t1 = threading.Thread(
                target=self.lineScan_data_producer,
                args=(0, scan_continue_time),
                daemon=True,
            )  # lunch the line scan data producer
            t2 = threading.Thread(
                target=self.lineScan_data_consumer, daemon=True
            )  # lunch the line scan data consumer
            t1.start()
            t2.start()
            print("line scanning...")
            t1.join()
            t2.join()

    def safe_lineScanGet(self, signal_index, fit_line=True, max_retries=50):
        """
        Safe wrapper for lineScanGet that handles IndexError when scan data is empty.

        Args:
            signal_index: Index of the signal channel
            fit_line: Whether to fit the line (currently unused)
            max_retries: Maximum number of retries before giving up

        Returns:
            dict: Contains 'line_Scan_data_for' and 'line_Scan_data_back', or None if failed
        """
        retry_count = 0

        # In demo mode, be more patient with retries
        if self.nanonis_mode == "demo":
            initial_wait = 0.5  # Wait longer initially in demo mode
            retry_sleep = 0.2
        else:
            initial_wait = 0.1
            retry_sleep = 0.1

        # Initial wait to let scan data populate
        time.sleep(initial_wait)

        while retry_count < max_retries:
            try:
                # Call the original lineScanGet method from parent class
                result = super().lineScanGet(signal_index, fit_line)

                # Validate the result has valid data
                if (
                    result
                    and "line_Scan_data_for" in result
                    and "line_Scan_data_back" in result
                ):
                    if retry_count > 0:
                        print(
                            f"Success: Line scan data retrieved after {retry_count + 1} attempts"
                        )
                    return result
                else:
                    retry_count += 1
                    if retry_count <= 3:
                        print(
                            f"Warning: Invalid line scan data structure (attempt {retry_count}/{max_retries})"
                        )

            except IndexError as e:
                retry_count += 1
                if retry_count <= 3:  # Only print first 3 warnings to avoid spam
                    print(
                        f"Warning: Scan data not ready (attempt {retry_count}/{max_retries}): {e}"
                    )

            except Exception as e:
                retry_count += 1
                if retry_count <= 3:
                    print(
                        f"Warning: Unexpected error in lineScanGet (attempt {retry_count}/{max_retries}): {e}"
                    )

            if retry_count < max_retries:
                time.sleep(retry_sleep)
            else:
                break

        # If we exhausted all retries, try to create dummy data for demo mode
        if self.nanonis_mode == "demo":
            print("Warning: Creating dummy line scan data for demo mode")
            dummy_data = np.random.random(304) * 1e-12  # Random height data
            return {"line_Scan_data_for": dummy_data, "line_Scan_data_back": dummy_data}

        print(f"Error: Failed to get line scan data after {max_retries} attempts")
        return None

    def lineScan_data_producer(self, angle=0, scan_continue_time=90):

        end_time = time.time() + scan_continue_time
        consecutive_failures = 0
        max_consecutive_failures = 5  # Limit consecutive failures

        self.ScanPropsSet(
            Continuous_scan=1,
            Bouncy_scan=1,
            Autosave=3,
            Series_name=" ",
            Comment="LineScan",
        )
        self.lineScanmode(3, angle=angle)
        Z_m_index = self.signal_channel_list[-1]
        self.ScanStart()
        self.WaitEndOfScan()

        print(
            f"Line scan producer started, will run for {scan_continue_time:.1f} seconds"
        )

        while True:
            lineScandata = self.safe_lineScanGet(Z_m_index)

            # If safe_lineScanGet returns None, skip this iteration
            if lineScandata is None:
                consecutive_failures += 1
                print(
                    f"Warning: Line scan data failed ({consecutive_failures}/{max_consecutive_failures})"
                )

                if consecutive_failures >= max_consecutive_failures:
                    print(
                        "Error: Too many consecutive line scan failures, ending line scan"
                    )
                    break

                time.sleep(0.5)  # Wait longer after failure
                continue
            else:
                consecutive_failures = 0  # Reset counter on success

            if self.lineScanQueue.full():
                self.lineScanQueue.get()
            if (
                time.time() > end_time or self.skip_flag
            ):  # if the time is over or the scan is skiped, break the loop
                self.lineScanQueue.put("end")

                if (
                    self.aggressive_tip_flag
                ):  # aggressive_tip_flag = 1 means the scan is skiped 5 times continuously, so that might be the super terable tip!
                    time.sleep(1)
                    self.TipShaper(TipLift="-6n")
                    time.sleep(1)
                    self.BiasPulse(-6, width=0.05)
                    time.sleep(1)
                    self.aggressive_tip_flag = 0  # reset the aggressive_tip_flag

                break

            self.lineScanQueue.put(lineScandata)

    def lineScan_data_consumer(self):
        self.line_start_time = time.strftime(
            "%Y-%m-%d %H-%M-%S", time.localtime(time.time())
        )
        lineScan_save_path = self.log_path + "/" + "lineScan" + "/"
        if not os.path.exists(
            lineScan_save_path
            + self.line_start_time
            + "_line_scan_"
            + str(self.line_scan_change_times).zfill(5)
        ):

            os.makedirs(
                lineScan_save_path
                + self.line_start_time
                + "_line_scan_"
                + str(self.line_scan_change_times).zfill(5)
            )

        while True:
            lineScandata_1 = self.lineScanQueue.get()
            # print('lineScan_data_consumer_works')

            # time_per_line = nanonis.ScanSpeedGet()['Forward time per line']
            # time.sleep(time_per_line) # wait for the line scan data to be collected

            time.sleep(0.153)  # wait for the line scan data to be collected
            if (
                lineScandata_1 == "end"
            ):  # if the line scan data producer is over, stop the line scan data consumer than draw the line scan data

                print("lineScan complete! ")

                break  # end the lineScan_data_consumer
            if (
                len(self.nanocoodinate_list) > 1
            ):  # if the nanocoodinate list have more than one nanocoodinate, save the line scan data
                lineScandata_1_for = fit_line(
                    lineScandata_1["line_Scan_data_for"]
                )  # t-1 line scan data for
                lineScandata_1_back = fit_line(
                    lineScandata_1["line_Scan_data_back"]
                )  # t-1 line scan data back
                # np.save(lineScan_save_path + self.line_start_time + '_line_scan_' + str(self.line_scan_change_times).zfill(5) +'/lineScandata_'+str(number_of_line_scan).zfill(5) +'.npy',
                #         [self.AdjustTip_flag ,self.nanocoodinate_list[-1], self.nanocoodinate_list[-2] ,lineScandata_1_for, lineScandata_1_back])
                # print('lineScandata_{}_for.npy'.format(number_of_line_scan) + ' is saved')

                # if the line scan max - min is bigger than 500, set the skip flag to 1
                if linescan_max_min_check(lineScandata_1_for) > self.convert(
                    self.linescan_max_min_threshold
                ):
                    self.threshold_list.append(1)
                else:
                    self.threshold_list.append(0)
                # if the threshold_list is full with 1, the scan will be skiped
                if len(self.threshold_list) == self.len_threshold_list and all(
                    threshold == 1 for threshold in self.threshold_list
                ):
                    self.skip_flag = 1  # switch the skip flag to 1, because the line scan data is out of the threshold several times continuously
                    self.threshold_list.clear()  # clear the threshold list, recount the threshold

                self.skip_list.append(
                    self.skip_flag
                )  # add the skip flag to the skip list

                # if len(self.skip_list) ==10 and all(skip == 1 for skip in self.skip_list):

                if len(self.skip_list) >= 5 and all(
                    skip == 1 for skip in self.skip_list
                ):
                    self.aggressive_tip_flag = 1
                    self.skip_list.clear()  # clear the skip list, recount the skip flag
                    print("The line scan is skiped")

    # def a function to move the tip to the next point, and append all data to the circle_list and nanocoodinate_list
    def move_to_next_point(self):
        """
        Move to the next grid position for systematic scanning with 0.1 overlap.

        The function implements a 2D grid-based scanning approach where:
        - Large scan frame: 50nm x 50nm (acquisition area)
        - Small scan frame: 10nm x 10nm (manipulation area)
        - Grid overlap: 0.9 (90% overlap between adjacent scans)
        - Grid step: 1nm (10nm * (1 - 0.9))
        """
        
        # Initialize grid on first call
        if not self.grid_initialized:
            print("First call to move_to_next_point - initializing grid...")
            # Use current position or default to origin for large scan center
            current_position = getattr(self, 'nanocoodinate', (0.0, 0.0))
            large_scan_size=self.convert(self.large_scan_frame_size)
            self.initialize_grid_positions(large_scan_center=current_position, large_scan_size=large_scan_size)

        # Check if grid scanning is complete
        print(f"DEBUG: grid_completed={self.grid_completed}, current_grid_index={self.current_grid_index}, total_positions={len(self.grid_positions)}")
        if self.grid_completed or self.current_grid_index >= len(self.grid_positions):
            print("Grid scanning completed. All positions have been visited.")
            self.grid_completed = True
            return False  # Indicate no more positions available
        
        # Get next grid position
        next_position = self.grid_positions[self.current_grid_index]
        
        # Update coordinate lists for compatibility with existing code
        self.nanocoodinate = next_position
        self.nanocoodinate_list.append(self.nanocoodinate)
        
        # Convert to pixel coordinates for visualization (if needed)
        # This maintains compatibility with existing visualization code
        if hasattr(self, 'plane_size'):
            # Convert meters to pixel coordinates for visualization
            pixel_x = int(self.plane_size/2 + (next_position[0] / self.convert(self.plane_edge)) * self.plane_size)
            pixel_y = int(self.plane_size/2 + (next_position[1] / self.convert(self.plane_edge)) * self.plane_size)
            self.inter_closest = (pixel_x, pixel_y)
        else:
            # Fallback to direct coordinates
            self.inter_closest = next_position
        
        # Update visualization queue (non-blocking to prevent hanging)
        try:
            self.tippathvisualQueue.put(self.inter_closest, block=False)
        except:
            # If queue is full, remove old item and add new one
            try:
                self.tippathvisualQueue.get_nowait()
                self.tippathvisualQueue.put(self.inter_closest, block=False)
            except:
                pass  # Skip visualization update if queue issues persist
        
        # Print progress information
        progress_percent = (self.current_grid_index + 1) / len(self.grid_positions) * 100
        print(f"Grid position {self.current_grid_index + 1}/{len(self.grid_positions)} ({progress_percent:.1f}%)")
        print(f"Moving to: ({next_position[0]*1e9:.2f}, {next_position[1]*1e9:.2f}) nm")
        print(f"Scan center coordinates: {self.nanocoodinate}")
        
        # Set scan frame to the new position
        print("Setting scan frame...")
        self.ScanFrameSet(
            self.nanocoodinate[0],
            self.nanocoodinate[1],  # Remove the offset, center the scan properly
            self.small_scan_frame_size,  # Use the 10nm scan size
            self.small_scan_frame_size,
            angle=0,
        )
        
        # Save progress
        np.save(self.log_path + "/nanocoodinate_list.npy", self.nanocoodinate_list)
        
        # Save grid progress
        grid_progress = {
            'current_index': self.current_grid_index,
            'total_positions': len(self.grid_positions),
            'completed_positions': self.nanocoodinate_list,
            'progress_percent': progress_percent
        }
        np.save(self.log_path + "/grid_progress.npy", grid_progress)
        
        # Increment grid index for next call
        self.current_grid_index += 1
        
        return True  # Indicate successful move to next position

    def initialize_grid_positions(self, large_scan_center=None, large_scan_size=None):
        """
        Initialize 2D grid positions for systematic scanning with 0.9 overlap
        
        Args:
            large_scan_center: Center position of the large scan frame (x, y) in meters
            large_scan_size: Size of the large scan frame (width, height) in meters
                              If None, uses current tip position or (0, 0)
        """
        print("Initializing 2D grid for systematic scanning...")
        
        # Set the large scan frame center
        if large_scan_center is not None:
            self.large_scan_center = large_scan_center
        elif hasattr(self, 'nanocoodinate') and self.nanocoodinate:
            self.large_scan_center = self.nanocoodinate
        else:
            self.large_scan_center = (0.0, 0.0)  # Default to origin
            
        # Convert scan sizes to meters
        if large_scan_size:
            # large_scan_size is already converted to meters in the main file
            large_scan_size = large_scan_size  # Use the passed value directly
        else:
            large_scan_size = self.convert(self.large_scan_frame_size)  # 50nm in meters
            
        small_scan_size = self.convert(self.small_scan_frame_size)  # 10nm in meters
        
        # Calculate optimal number of scans based on scan size ratio and overlap
        # For 10% overlap, effective coverage per scan is 90% of scan size
        effective_scan_coverage = small_scan_size * (1.0 - self.grid_overlap)  # 10nm * 0.9 = 9nm
        
        # Use conservative margin to avoid boundary issues
        margin_factor = 0.8  # More conservative margin
        coverage_distance = large_scan_size * margin_factor
        
        # Calculate number of scans needed: coverage_distance / effective_coverage
        # Add 1 because we need n+1 scans to cover n gaps
        scans_needed = coverage_distance / effective_scan_coverage
        grid_points_per_axis = max(1, int(np.ceil(scans_needed)))
        
        # Calculate actual step size based on the number of grid points
        if grid_points_per_axis > 1:
            grid_step = coverage_distance / (grid_points_per_axis - 1)
        else:
            grid_step = 0  # Single scan at center
            
        print(f"Large scan area: {large_scan_size*1e9:.1f}nm x {large_scan_size*1e9:.1f}nm")
        print(f"Small scan size: {small_scan_size*1e9:.1f}nm x {small_scan_size*1e9:.1f}nm")
        print(f"Coverage distance: {coverage_distance*1e9:.1f}nm (margin factor: {margin_factor})")
        print(f"Scans needed: {scans_needed:.2f}")
        print(f"Grid points per axis: {grid_points_per_axis}")
        print(f"Grid step size: {grid_step*1e9:.2f}nm")
        print(f"Total scans: {grid_points_per_axis}x{grid_points_per_axis} = {grid_points_per_axis**2} points")
        
        # Generate grid positions
        self.grid_positions = []
        center_x, center_y = self.large_scan_center
        
        # Calculate grid extent
        half_extent = (grid_points_per_axis - 1) * grid_step / 2
        
        for i in range(grid_points_per_axis):
            for j in range(grid_points_per_axis):
                # Calculate position relative to center
                x_offset = (i - (grid_points_per_axis - 1) / 2) * grid_step
                y_offset = (j - (grid_points_per_axis - 1) / 2) * grid_step
                
                grid_x = center_x + x_offset
                grid_y = center_y + y_offset
                
                self.grid_positions.append((grid_x, grid_y))
        
        # Reset grid state
        self.current_grid_index = 0
        self.grid_completed = False
        self.grid_initialized = True
        
        # Save initial grid configuration
        grid_info = {
            'large_scan_center': self.large_scan_center,
            'large_scan_size': large_scan_size,
            'small_scan_size': small_scan_size,
            'grid_step': grid_step,
            'grid_overlap': self.grid_overlap,
            'grid_positions': self.grid_positions,
            'total_positions': len(self.grid_positions)
        }
        
        np.save(self.log_path + "/grid_configuration.npy", grid_info)
        
        print(f"Grid initialized with {len(self.grid_positions)} scan positions")
        print(f"Grid center: ({center_x*1e9:.2f}, {center_y*1e9:.2f}) nm")
        print(f"Grid extent: ±{half_extent*1e9:.2f} nm from center")
        
        # Debug: Show first few and last few positions
        if len(self.grid_positions) > 0:
            print(f"First position: ({self.grid_positions[0][0]*1e9:.2f}, {self.grid_positions[0][1]*1e9:.2f}) nm")
            if len(self.grid_positions) > 1:
                print(f"Second position: ({self.grid_positions[1][0]*1e9:.2f}, {self.grid_positions[1][1]*1e9:.2f}) nm")
            if len(self.grid_positions) > 4:
                print(f"Fifth position: ({self.grid_positions[4][0]*1e9:.2f}, {self.grid_positions[4][1]*1e9:.2f}) nm")
            print(f"Last position: ({self.grid_positions[-1][0]*1e9:.2f}, {self.grid_positions[-1][1]*1e9:.2f}) nm")

    def set_large_scan_area(self, center_x, center_y):
        """
        Set the large scan area center position and reinitialize the grid.
        
        Args:
            center_x: X coordinate of the large scan center in meters
            center_y: Y coordinate of the large scan center in meters
        """
        print(f"Setting large scan area center to: ({center_x*1e9:.2f}, {center_y*1e9:.2f}) nm")
        self.large_scan_center = (center_x, center_y)
        self.initialize_grid_positions(large_scan_center=self.large_scan_center)
        
    def get_grid_status(self):
        """
        Get current grid scanning status.
        
        Returns:
            dict: Grid status information including progress and remaining positions
        """
        if not self.grid_initialized:
            return {
                'initialized': False,
                'total_positions': 0,
                'current_position': 0,
                'remaining_positions': 0,
                'progress_percent': 0.0,
                'completed': False
            }
        
        remaining = len(self.grid_positions) - self.current_grid_index
        progress = (self.current_grid_index / len(self.grid_positions)) * 100 if len(self.grid_positions) > 0 else 100
        
        return {
            'initialized': True,
            'total_positions': len(self.grid_positions),
            'current_position': self.current_grid_index,
            'remaining_positions': remaining,
            'progress_percent': progress,
            'completed': self.grid_completed,
            'large_scan_center': self.large_scan_center,
            'current_coordinates': self.nanocoodinate if hasattr(self, 'nanocoodinate') else None
        }

    def reset_grid_scanning(self):
        """
        Reset grid scanning to start from the beginning.
        """
        print("Resetting grid scanning to start position...")
        self.current_grid_index = 0
        self.grid_completed = False
        self.nanocoodinate_list = []
        if hasattr(self, 'circle_list'):
            self.circle_list = []
        print("Grid scanning reset complete.")

    # def a function to predict the scan qulity
    def image_recognition(self):
        print("Starting image recognition...")
        
        # judge the gap between the max and min of the image
        # if the gap is bigger than the threshold, the scan is skiped
        print("Analyzing image data...")
        data_linear = linear_whole(self.Scan_data["Scan_data_for"]["data"])
        print(
            "The gap between the max and min of the image is "
            + str(linescan_max_min_check(data_linear))
        )
        if linescan_max_min_check(linear_whole(data_linear)) >= self.convert(
            self.scan_max_min_threshold
        ):
            self.skip_flag = 1
            print("The scan is skipped")

        # use CNN to predict the image quality
        print("Predicting image quality with CNN...")
        probability = predict_image_quality(self.image_for, self.quality_model_path)
        print("The probability of the good image is " + str(round(probability, 2)))
        
        print("Determining scan quality...")
        if (
            probability > self.scan_qulity_threshold and self.skip_flag == 0
        ):  # 0.5 is the self.scan_qulity_threshold of the probability
            scan_qulity = 1  # good image
        else:
            scan_qulity = 0  # bad image

        print(f"Scan quality determined: {scan_qulity} (1=good, 0=bad)")

        # calculate the R depend on the scan_qulity
        print("Calculating R value...")
        if len(self.circle_list) == 0:  # if the circle_list is empty, initialize the R
            self.R = self.R_init
        else:
            self.R = increase_radius(
                scan_qulity,
                self.circle_list[-1][2],
                self.R_init,
                self.R_max,
                self.R_step,
            )  # increase the R

        print("Updating circle list...")
        self.circle_list.append(
            [self.inter_closest[0], self.inter_closest[1], self.R, scan_qulity]
        )
        # save the circle_list as a npy file
        self.circle_list_save.append(
            [
                self.inter_closest[0],
                self.inter_closest[1],
                self.R,
                scan_qulity,
                self.coverage,
            ]
        )
        
        print("Saving circle list...")
        np.save(self.log_path + "/circle_list.npy", self.circle_list_save)
        
        # Update visualization queue (non-blocking to prevent hanging)
        try:
            self.tippathvisualQueue.put(
                [
                    self.inter_closest[0],
                    self.inter_closest[1],
                    self.R,
                    scan_qulity,
                    self.coverage,
                ],
                block=False
            )
        except:
            # If queue is full, remove old item and add new one
            try:
                self.tippathvisualQueue.get_nowait()
                self.tippathvisualQueue.put(
                    [
                        self.inter_closest[0],
                        self.inter_closest[1],
                        self.R,
                        scan_qulity,
                        self.coverage,
                    ],
                    block=False
                )
            except:
                pass  # Skip visualization update if queue issues persist
        # save the scan image via the scan_qulity

        # create the good_scan and bad_scan folder in self.image_data_save_path if not exist
        if not os.path.exists(self.image_data_save_path + "/good_scan"):
            os.makedirs(self.image_data_save_path + "/good_scan")
        if not os.path.exists(self.image_data_save_path + "/bad_scan"):
            os.makedirs(self.image_data_save_path + "/bad_scan")
        self.image_save_time = time.strftime(
            "%Y-%m-%d %H-%M-%S", time.localtime(time.time())
        )
        if scan_qulity == 1:
            cv2.imwrite(
                self.image_data_save_path
                + "/good_scan/"
                + "Scan_data_for"
                + self.image_save_time
                + ".png",
                self.image_for,
            )  # save the image in the good_scan folder
            cv2.imwrite(
                self.image_data_save_path
                + "/good_scan/"
                + "Scan_data_back"
                + self.image_save_time
                + ".png",
                self.image_back,
            )  # save the image in the good_scan folder

        else:
            cv2.imwrite(
                self.image_data_save_path
                + "/bad_scan/"
                + "Scan_data_for"
                + self.image_save_time
                + ".png",
                self.image_for,
            )  # save the image in the bad_scan folder
            cv2.imwrite(
                self.image_data_save_path
                + "/bad_scan/"
                + "Scan_data_back"
                + self.image_save_time
                + ".png",
                self.image_back,
            )  # save the image in the bad_scan folder

        print(f"Image recognition completed. Returning scan quality: {scan_qulity}")
        # return scan_qulity
        return scan_qulity

    def key_points_convert(self, key_points_result, scan_position=None, scan_edge=None):
        """
        Convert normalized keypoint coordinates to real world coordinates

        Args:
            key_points_result: List of detected keypoints [[class, x, y, w, h, key_x, key_y], ...]
            scan_posion: Center position of scan in meters (x, y)
            scan_edge: Edge length of scan area in nanometers or SI string format

        Returns:
            List of converted keypoints with real coordinates in meters
        """
        if not key_points_result or len(key_points_result) == 0:
            return []

        # Use provided scan position or default to inter_closest
        if scan_position is not None:
            # Ensure scan_position is in meters
            if abs(scan_position[0]) > 1e-6:  # If value > 1 micrometer, assume it's in nm
                scan_position = (scan_position[0] * 1e-9, scan_position[1] * 1e-9)
            else:
                scan_position = scan_position  # Already in meters
        else:
            # Convert inter_closest from nm to meters if needed
            if abs(self.inter_closest[0]) > 1e-6:
                scan_position = (
                    self.inter_closest[0] * 1e-9,
                    self.inter_closest[1] * 1e-9,
                )
            else:
                scan_position = self.inter_closest

       
        edge_meters = self.convert(self.scan_zoom_in_list[0])

        converted_points = []

        for keypoint in key_points_result:
            if len(keypoint) < 7:
                continue  # Skip malformed keypoints

            mol_class, norm_x, norm_y, norm_w, norm_h, key_x, key_y = keypoint[:7]

            # Convert normalized coordinates (0-1) to real world coordinates in meters
            # Assuming image coordinates: (0,0) is top-left, (1,1) is bottom-right
            # Real world coordinates: scan_position is center

            real_x = scan_position[0] + (norm_x - 0.5) * edge_meters
            real_y = scan_position[1] - (norm_y - 0.5) * edge_meters  # Flip Y axis
            real_w = norm_w * edge_meters
            real_h = norm_h * edge_meters
            real_key_x = scan_position[0] + (key_x - 0.5) * edge_meters
            real_key_y = scan_position[1] - (key_y - 0.5) * edge_meters  # Flip Y axis

            converted_keypoint = [
                mol_class,
                real_x,
                real_y,
                real_w,
                real_h,
                real_key_x,
                real_key_y,
            ]

            # Add any additional keypoint data
            if len(keypoint) > 7:
                # Convert additional keypoints from normalized to meters
                for i in range(7, len(keypoint), 2):
                    if i + 1 < len(keypoint):
                        extra_key_x = (
                            scan_position[0] + (keypoint[i] - 0.5) * edge_meters
                        )
                        extra_key_y = (
                            scan_position[1] - (keypoint[i + 1] - 0.5) * edge_meters
                        )
                        converted_keypoint.extend([extra_key_x, extra_key_y])

            converted_points.append(converted_keypoint)

        return converted_points

    def _map_class_to_type(
        self, molecule_class
    ):  ################################## adjust to our spectial molecules ####################################
        """
        Map detection class number to molecule type name
        Customize this mapping based on your trained model's classes
        """
        class_mapping = {
            0: "AAA",
            1: "BBB",
            2: "CCC",
            3: "DDD",
            # Add more mappings as needed based on your trained model
        }
        return class_mapping.get(molecule_class, f"class_{molecule_class}")

    def molecular_tri_seeker(self, image, n=4, scan_posion=(0.0, 0.0), scan_edge=30):
        """Seek molecules in triangular lattice pattern"""
        key_point_save_dir = self.image_data_save_path + "/key_points"
        if not os.path.exists(key_point_save_dir):
            os.makedirs(key_point_save_dir)

        key_points_result = key_detect(
            image, self.keypoint_model_path, key_point_save_dir
        )

        if len(key_points_result) == 0:
            return None

        # Filter and select up to n molecules
        molecular_list = self.key_points_convert(
            key_points_result, scan_posion=scan_posion, scan_edge=scan_edge
        )
        molecular_list = filter_close_bboxes(
            molecular_list, self.molecular_filter_threshold
        )

        # Return up to n molecules
        selected_molecules = molecular_list[: min(n, len(molecular_list))]

        return selected_molecules

    def auto_select_molecules_for_processing(
        self,
        shape_key_points_result,
        max_molecules_per_scan,
        mode,
        selection_strategy="intelligent",
    ):
        """
        Intelligently select molecules for processing using enhanced molecular_tracker logic

        Args:
            shape_key_points_result: List of detected molecules from molecular_seeker
            max_molecules_per_scan: Maximum number of molecules to select
            selection_strategy: Strategy for selection ("intelligent", "closest", "quality")

        Returns:
            list: Selected molecules for processing
        """
        print(f"Selecting molecules: {selection_strategy} strategy")

        if not shape_key_points_result:
            return []

        # Get current scan center for reference
        scan_center = getattr(self, "nanocoodinate", (0.0, 0.0))

        # Enhanced class filtering with test_simulation insights
        # Instead of rigid filtering, use intelligent scoring to prefer target classes
        if mode == 'manipulation':
            # Prefer class 0 but don't exclude others completely
            print(f"Manipulation mode: preferring class 0 molecules from {len(shape_key_points_result)} detected")
        elif mode == 'spectroscopy':
            # Prefer class 1 but don't exclude others completely  
            print(f"Spectroscopy mode: preferring class 1 molecules from {len(shape_key_points_result)} detected")
        else:
            print(f"General mode: considering all {len(shape_key_points_result)} molecules")

        # Use all molecules for scoring (no pre-filtering) - test_simulation approach
        candidate_molecules = shape_key_points_result

        if selection_strategy == "intelligent":
            # Use a combination of quality and spatial distribution
            selected_molecules = []

            # First, sort by a composite score with test_simulation insights
            def composite_score(molecule_data, mode=mode):
                # molecule_data format: [class, x, y, w, h, key_x, key_y, ...]
                # Coordinates are already in meters from key_points_convert
                mol_class = molecule_data[0]
                mol_x = molecule_data[1]  # Already in meters
                mol_y = molecule_data[2]  # Already in meters

                # Distance from scan center (prefer closer molecules)
                distance = np.sqrt(
                    (mol_x - scan_center[0]) ** 2 + (mol_y - scan_center[1]) ** 2
                )
                distance_score = 1.0 / (distance * 1e9 + 1.0)  # Normalize to nm scale

                # Enhanced class score using test_simulation approach
                if mode == 'manipulation':
                    # Prefer class 0, but don't exclude others (like test_simulation prefers class 1)
                    class_score = 1.0 / (np.abs(mol_class - 0.0) + 0.1)
                elif mode == 'spectroscopy':
                    # Prefer class 1 (same as test_simulation logic)
                    class_score = 1.0 / (np.abs(mol_class - 1.0) + 0.1) + 1.0 / (np.abs(mol_class - 2.0) + 0.3)
                else:
                    # Default: prefer class 1 (test_simulation default)
                    class_score = 1.0 

                return distance_score * 0.5 + class_score * 0.3

            # Sort by composite score using all candidate molecules
            scored_molecules = [
                (mol, composite_score(mol)) for mol in candidate_molecules
            ]
            scored_molecules.sort(key=lambda x: x[1], reverse=True)

            # Select molecules ensuring spatial distribution
            for mol_data, score in scored_molecules:
                if len(selected_molecules) >= max_molecules_per_scan:
                    break

                mol_x = mol_data[1]  # Already in meters
                mol_y = mol_data[2]  # Already in meters

                # Check minimum distance to already selected molecules
                too_close = False
                min_separation = 3e-9  # 3nm minimum separation

                for selected_mol in selected_molecules:
                    sel_x = selected_mol[1]  # Already in meters
                    sel_y = selected_mol[2]  # Already in meters
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
            # Select molecules closest to scan center from candidate molecules
            def distance_from_center(molecule_data):
                mol_x = molecule_data[1]  # Already in meters
                mol_y = molecule_data[2]  # Already in meters
                return np.sqrt(
                    (mol_x - scan_center[0]) ** 2 + (mol_y - scan_center[1]) ** 2
                )

            sorted_molecules = sorted(candidate_molecules, key=distance_from_center)
            selected_molecules = sorted_molecules[:max_molecules_per_scan]
            
            # Debug output
            for i, mol_data in enumerate(selected_molecules):
                print(f"  Selected molecule {i+1}: class={mol_data[0]}, distance={distance_from_center(mol_data)*1e9:.2f}nm")

        elif selection_strategy == "quality":
            # Select molecules based on enhanced quality scoring from candidate molecules
            def quality_score(molecule_data):
                mol_class = molecule_data[0]
                mol_w = molecule_data[3] * 1e9  # Convert to nm
                mol_h = molecule_data[4] * 1e9

                # Enhanced class scoring using test_simulation approach
                if mode == 'manipulation':
                    class_score = 1.0 / (np.abs(mol_class - 0.0) + 0.1)
                elif mode == 'spectroscopy':
                    class_score = 1.0 / (np.abs(mol_class - 1.0) + 0.1)
                else:
                    class_score = 1.0 / (np.abs(mol_class - 1.0) + 0.1)
                    
                size_score = 1.0 / (abs(mol_w - 1.0) + abs(mol_h - 1.0) + 0.1)
                return class_score * 0.7 + size_score * 0.3

            sorted_molecules = sorted(
                candidate_molecules, key=quality_score, reverse=True
            )
            selected_molecules = sorted_molecules[:max_molecules_per_scan]
            
            # Debug output
            for i, mol_data in enumerate(selected_molecules):
                print(f"  Selected molecule {i+1}: class={mol_data[0]}, quality_score={quality_score(mol_data):.3f}")

        elif selection_strategy == "distance_spread":
            # Ensure spatial distribution from candidate molecules
            selected_molecules = []
            remaining_molecules = candidate_molecules.copy()

            # Start with the molecule closest to center
            if remaining_molecules:
                center_mol = min(
                    remaining_molecules,
                    key=lambda x: np.sqrt(
                        (x[1] - scan_center[0]) ** 2 + (x[2] - scan_center[1]) ** 2
                    ),
                )
                selected_molecules.append(center_mol)
                remaining_molecules.remove(center_mol)
                print(f"  Selected center molecule: class={center_mol[0]}")

            # Add molecules that are far from already selected ones
            while (
                len(selected_molecules) < max_molecules_per_scan and remaining_molecules
            ):
                best_mol = None
                best_min_dist = 0

                for mol in remaining_molecules:
                    mol_x = mol[1]  # Already in meters
                    mol_y = mol[2]  # Already in meters

                    min_dist_to_selected = float("inf")
                    for sel_mol in selected_molecules:
                        sel_x = sel_mol[1]  # Already in meters
                        sel_y = sel_mol[2]  # Already in meters
                        dist = np.sqrt((mol_x - sel_x) ** 2 + (mol_y - sel_y) ** 2)
                        min_dist_to_selected = min(min_dist_to_selected, dist)

                    if min_dist_to_selected > best_min_dist:
                        best_min_dist = min_dist_to_selected
                        best_mol = mol

                if best_mol:
                    selected_molecules.append(best_mol)
                    remaining_molecules.remove(best_mol)
                else:
                    break

        print(
            f"Auto-selected {len(selected_molecules)} molecules using '{selection_strategy}' strategy"
        )
        
        # Enhanced validation with test_simulation insights - warn but don't fail for mixed classes
        if selected_molecules:
            final_classes = [int(mol[0]) for mol in selected_molecules]
            unique_classes = list(set(final_classes))
            
            if mode == 'manipulation':
                class_0_count = sum(1 for cls in final_classes if cls == 0)
                other_count = len(final_classes) - class_0_count
                if class_0_count > 0:
                    print(f"Selected {class_0_count} class 0 molecules for manipulation")
                if other_count > 0:
                    other_classes = [cls for cls in unique_classes if cls != 0]
                    print(f"Warning: Also selected {other_count} molecules from classes {other_classes}")
                
            elif mode == 'spectroscopy':
                class_1_count = sum(1 for cls in final_classes if cls == 1)
                other_count = len(final_classes) - class_1_count
                if class_1_count > 0:
                    print(f"Selected {class_1_count} class 1 molecules for spectroscopy")
                if other_count > 0:
                    other_classes = [cls for cls in unique_classes if cls != 1]
                    print(f"Warning: Also selected {other_count} molecules from classes {other_classes}")
            
            print(f"Final selection: {len(selected_molecules)} molecules, classes: {final_classes}")
        else:
            print(f"No molecules selected for {mode}")
            
        return selected_molecules

    def molecular_seeker(self, image, scan_position=(0.0, 0.0), scan_edge=30):
        """Seek all molecules in the image"""
        key_point_save_dir = self.image_data_save_path + "/key_points"
        if not os.path.exists(key_point_save_dir):
            os.makedirs(key_point_save_dir)

        key_points_result = key_detect(
            image, self.keypoint_model_path, key_point_save_dir
        )  # key_points_result is [[class, x, y, w, h, key_x, key_y], ......]
        # normalized coodination (0-1) in the key_points_result assume left-bottom is (0,1) and right-top is (1,0)

        if len(key_points_result) == 0:
            return None

        molecular_list = self.key_points_convert(
            key_points_result, scan_position=scan_position, scan_edge=scan_edge
        )
        molecular_list = filter_close_bboxes(
            molecular_list, self.molecular_filter_threshold
        )

        return molecular_list
    
    def spectr_point_seeker(self,image, scan_position=(0.0, 0.0), scan_edge=30):
        """Seek all spectroscopy points in the image"""
        key_point_save_dir = self.image_data_save_path + "/key_points"
        if not os.path.exists(key_point_save_dir):
            os.makedirs(key_point_save_dir)

        key_points_result = key_detect(
            image, self.spectr_model_path, key_point_save_dir
        )  # key_points_result is [[class, x, y, w, h, key_x, key_y], ......]
        # normalized coodination (0-1) in the key_points_result assume left-bottom is (0,0) and right-top is (1,1)

        if len(key_points_result) == 0:
            return None

        molecular_list = self.key_points_convert(
            key_points_result, scan_position=scan_position, scan_edge=scan_edge
        )
        molecular_list = filter_close_bboxes(
            molecular_list, self.molecular_filter_threshold
        )

        return molecular_list


    def interest_points_seeker(
        self,
        image,
        registered_molecules,
        scan_position=(0.0, 0.0),
        scan_edge=30,
        use_all_molecules=False,
        points_per_molecule=3,
    ):
        """
        AI Agent: Find spectroscopy interest points for molecules

        Args:
            image: The STM image containing molecules
            registered_molecules: List of tuples (molecule, index, type, is_manipulated)
            scan_position: Center position of the scan
            scan_edge: Edge length of the scan area
            use_all_molecules: If True, use all molecules; if False, use only manipulated molecules

        Returns:
            dict: Mapping of molecular_index -> list of spectroscopy points in nanometers
        """
        # Filter molecules based on the use_all_molecules flag
        if use_all_molecules:
            target_molecules = registered_molecules
            print(
                f"AI Agent: Finding spectroscopy interest points for ALL {len(target_molecules)} molecules"
            )
        else:
            target_molecules = [
                (mol, idx, mol_type, is_manip)
                for mol, idx, mol_type, is_manip in registered_molecules
                if is_manip
            ]
            print(
                f"AI Agent: Finding spectroscopy interest points for {len(target_molecules)} MANIPULATED molecules"
            )

        if not target_molecules:
            print("No target molecules found - no spectroscopy points needed")
            return {}

        try:
            # Use the keypoints model to detect all interest points in the image
            key_point_save_dir = self.image_data_save_path + "/interest_points"
            if not os.path.exists(key_point_save_dir):
                os.makedirs(key_point_save_dir)

            # Import the key_detect function
            from keypoint.detect import key_detect

            # Use the correct model path for interest points detection
            interest_points_model_path = self.keypoints_model_path

            # Run AI detection on the full image to get all interest points
            key_points_result = key_detect(
                image, interest_points_model_path, key_point_save_dir
            )  # key_points_result is [[class, x, y, w, h, key_x, key_y], ......]

            if not key_points_result:
                print("AI model found no interest points, using fallback positions")
                return None

            # Convert all detected keypoints to absolute nanometer coordinates
            all_interest_points = []

            molecular_points_list = self.key_points_convert(
                key_points_result, scan_position=scan_position, scan_edge=scan_edge
            )
            molecular_points_list = filter_close_bboxes(
                molecular_points_list, self.molecular_filter_threshold
            )

            for molecule_detected in molecular_points_list:
                for i in range(5, len(molecule_detected), 2):
                    x_nm = molecule_detected[i] * 1e-9
                    y_nm = molecule_detected[i + 1] * 1e-9
                    all_interest_points.append((x_nm, y_nm))

            print(f"AI detected {len(all_interest_points)} total interest points")

            # Map interest points to target molecules
            molecule_spectroscopy_map = {}

            for (
                molecule,
                molecular_index,
                molecule_type,
                is_manipulated,
            ) in target_molecules:
                mol_center = molecule.position

                # Find interest points near this molecule
                molecule_points = []
                search_radius = 3e-9  # 3 nanometers search radius

                for interest_point in all_interest_points:
                    distance = np.sqrt(
                        (interest_point[0] - mol_center[0]) ** 2
                        + (interest_point[1] - mol_center[1]) ** 2
                    )

                    if distance <= search_radius:
                        molecule_points.append(interest_point)

                # Select the best interest points for this molecule
                if molecule_points:
                    # Sort by distance to molecule center and take closest ones
                    molecule_points.sort(
                        key=lambda p: np.sqrt(
                            (p[0] - mol_center[0]) ** 2 + (p[1] - mol_center[1]) ** 2
                        )
                    )
                    # Take top 2-3 closest points
                    max_points = min(points_per_molecule, len(molecule_points))
                    selected_points = molecule_points[:max_points]
                else:
                    # Fallback: add molecule center and a nearby point
                    selected_points = None
                    print(f"No nearby AI points for molecule {molecular_index}")

                molecule_spectroscopy_map[molecular_index] = selected_points
                print(
                    f"Mapped {len(selected_points)} spectroscopy points to molecule {molecular_index}"
                )

            print(
                f"Created spectroscopy map for {len(molecule_spectroscopy_map)} molecules"
            )
            return molecule_spectroscopy_map

        except Exception as e:
            print(f"Error in AI interest point detection: {e}")
            print("Using fallback interest points")
            return None

    def perform_tip_manipulation(
        self, tip_position, tip_bias, tip_current, tip_induce_mode="pulse"
    ):
        """
        Perform tip manipulation operation at specified position

        Args:
            tip_position: (x, y) coordinates (m)
            tip_bias: Target bias voltage
            tip_current: Target current
            tip_induce_mode: 'CC', 'CH', or 'pulse'

        Returns:
            dict: Manipulation result with success status
        """
        print("Starting tip manipulation...")

        try:
            # Move tip to manipulation position
            print(f"Moving tip to manipulation position: {tip_position}")
            self.FolMeSpeedSet(self.convert(self.zoom_in_tip_speed), 1)
            self.TipXYSet(tip_position[0], tip_position[1])
            self.FolMeSpeedSet(self.convert(self.zoom_in_tip_speed), 0)
            time.sleep(1)

            # Manipulation parameters
            lift_time = 2  # s
            max_hold_time = 20  # s
            stop_current_threshold = tip_current / 2

            # Get initial tip conditions
            tip_bias_init = self.BiasGet()
            tip_current_init = self.SetpointGet()

            abs_tip_bias_init = abs(tip_bias_init)
            abs_tip_current_init = abs(tip_current_init)

            # Set absolute values initially
            self.BiasSet(abs_tip_bias_init)
            self.SetpointSet(abs_tip_current_init)

            manipulation_success = False

            if tip_induce_mode == "CC":
                manipulation_success = self._perform_cc_manipulation(
                    tip_bias,
                    tip_current,
                    abs_tip_bias_init,
                    abs_tip_current_init,
                    lift_time,
                    max_hold_time,
                    stop_current_threshold,
                )

            elif tip_induce_mode == "CH":
                manipulation_success = self._perform_ch_manipulation(
                    tip_bias, tip_current, abs_tip_bias_init, abs_tip_current_init
                )

            elif tip_induce_mode == "pulse":
                manipulation_success = self._perform_pulse_manipulation(
                    tip_bias,
                    tip_current,
                    abs_tip_bias_init,
                    abs_tip_current_init,
                    lift_time,
                )

            # Always restore initial conditions
            self.BiasSet(tip_bias_init)
            self.SetpointSet(tip_current_init)
            self.ZCtrlOnSet()

            time.sleep(1)
            print("Tip manipulation completed.")

            return {
                "success": manipulation_success,
                "position": tip_position,
                "mode": tip_induce_mode,
                "bias": tip_bias,
                "current": tip_current,
            }

        except Exception as e:
            print(f"Tip manipulation failed: {e}")
            # Restore initial conditions on error
            try:
                self.BiasSet(tip_bias_init)
                self.SetpointSet(tip_current_init)
                self.ZCtrlOnSet()
            except:
                pass

            return {
                "success": False,
                "error": str(e),
                "position": tip_position,
                "mode": tip_induce_mode,
            }

    def _perform_cc_manipulation(
        self,
        tip_bias,
        tip_current,
        abs_tip_bias_init,
        abs_tip_current_init,
        lift_time,
        max_hold_time,
        stop_current_threshold,
    ):
        """Perform CC (Constant Current) manipulation"""
        from collections import deque

        steps = 50
        time_interval = lift_time / steps

        # Calculate increments
        bias_step = (tip_bias - abs_tip_bias_init) / steps
        current_step = (tip_current - abs_tip_current_init) / steps

        # Gradually set bias and current
        for i in range(steps):
            self.BiasSet(abs_tip_bias_init + bias_step * (i + 1))
            self.SetpointSet(abs_tip_current_init + current_step * (i + 1))
            time.sleep(time_interval)

        self.ZCtrlOff()

        # Monitor current signal
        start_time = time.time()
        signal_history = deque(maxlen=10)

        while True:
            try:
                signal_current = self.SignalValsGet(0)["0"] * 1e9
                signal_current = round(signal_current, 4)
                signal_history.append(signal_current)

                # Check timeout
                if time.time() - start_time > max_hold_time:
                    print(f"Manipulation timeout after {max_hold_time} seconds")
                    return True  # Consider timeout as success

                # Check if current consistently below threshold
                if len(signal_history) == signal_history.maxlen and all(
                    value < stop_current_threshold for value in signal_history
                ):
                    print(
                        f"Manipulation completed after {time.time() - start_time:.2f} seconds"
                    )
                    return True

                time.sleep(0.1)  # Brief delay between measurements

            except Exception as e:
                print(f"Error monitoring current: {e}")
                return False

        return True

    def _perform_ch_manipulation(
        self, tip_bias, tip_current, abs_tip_bias_init, abs_tip_current_init
    ):
        """Perform CH (Constant Height) manipulation"""
        steps = 50
        current_time_interval = 1 / steps  # 1s total
        voltage_time_interval = 4 / steps  # 4s total

        # Calculate increments
        current_step = (tip_current - abs_tip_current_init) / steps
        bias_step = (tip_bias - abs_tip_bias_init) / steps

        # Gradually set current first
        for i in range(steps):
            self.SetpointSet(abs_tip_current_init + current_step * (i + 1))
            time.sleep(current_time_interval)

        time.sleep(1)  # Wait for stabilization

        self.ZCtrlOff()

        # Gradually set bias
        for i in range(steps):
            self.BiasSet(abs_tip_bias_init + bias_step * (i + 1))
            time.sleep(voltage_time_interval)

        time.sleep(8)  # Wait for manipulation

        return True

    def _perform_pulse_manipulation(
        self, tip_bias, tip_current, abs_tip_bias_init, abs_tip_current_init, lift_time
    ):
        """Perform pulse manipulation"""
        steps = 50
        time_interval = lift_time / steps

        # Calculate current increment
        current_step = (tip_current - abs_tip_current_init) / steps

        # Gradually set current
        for i in range(steps):
            self.SetpointSet(abs_tip_current_init + current_step * (i + 1))
            time.sleep(time_interval)

        self.ZCtrlOff()

        # Apply bias pulse
        self.BiasPulse(tip_bias, 0.1)
        time.sleep(1)

        return True

    def human_select_molecules_for_manipulation(self, molecules_list, image):
        """
        Display image with labeled molecules and let human select which ones to manipulate

        Args:
            molecules_list: List of detected molecules with global nanometer coordinates
            image: STM image to display

        Returns:
            list: Selected molecules for manipulation based on human input
        """
        if not molecules_list:
            print("No molecules available for selection")
            return []

        print(f"\n=== Human Molecule Selection Mode ===")
        print(f"Detected {len(molecules_list)} molecules")

        # Create visualization image with molecule labels
        vis_image = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if len(image.shape) == 2
            else image.copy()
        )

        # Get scan parameters for coordinate conversion
        zoom_scale = self.convert(
            self.scan_zoom_in_list[0]
        )  # Use first element consistently
        pixels_per_nm = self.scan_square_Buffer_pix / (
            zoom_scale * 1e9
        )  # pixels per nanometer
        center_pixel = (
            self.scan_square_Buffer_pix // 2,
            self.scan_square_Buffer_pix // 2,
        )  # Center of 304x304 image

        # Draw molecules with labels on the image
        for idx, molecule in enumerate(molecules_list):
            # molecule coordinates are already in global nanometer coordinates
            # Convert global nanometer coordinates to image pixel coordinates
            center_x = int(
                center_pixel[0]
                + (molecule[1] - self.nanocoodinate[0]) * 1e9 * pixels_per_nm
            )
            center_y = int(
                center_pixel[1]
                - (molecule[2] - self.nanocoodinate[1]) * 1e9 * pixels_per_nm
            )

            # Draw molecule circle and label
            cv2.circle(vis_image, (center_x, center_y), 10, (0, 255, 0), 2)
            cv2.putText(
                vis_image,
                f"{idx}",
                (center_x + 12, center_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Save and display the labeled image
        labeled_image_path = self.image_data_save_path + "/human_selection/"
        if not os.path.exists(labeled_image_path):
            os.makedirs(labeled_image_path)

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        image_filename = f"{labeled_image_path}/molecules_labeled_{timestamp}.png"
        cv2.imwrite(image_filename, vis_image)

        # Display image for human review
        cv2.namedWindow("Molecule Selection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Molecule Selection", 800, 800)
        cv2.imshow("Molecule Selection", vis_image)
        cv2.waitKey(1000)  # Display for 1 second initially

        print(f"Image with labeled molecules saved to: {image_filename}")
        print(f"Molecules are labeled from 0 to {len(molecules_list)-1}")
        print(
            "Please review the displayed image and select molecules for manipulation."
        )

        # Get human input for molecule selection
        selected_indices = []
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            try:
                user_input = input(
                    f"\nEnter molecule indices to manipulate (comma-separated, e.g., '0,2,5'): "
                ).strip()

                if not user_input:
                    print("No input provided. Skipping manipulation.")
                    break

                # Parse user input
                indices = [int(idx.strip()) for idx in user_input.split(",")]

                # Validate indices
                valid_indices = []
                for idx in indices:
                    if 0 <= idx < len(molecules_list):
                        valid_indices.append(idx)
                    else:
                        print(
                            f"Warning: Index {idx} is out of range (0-{len(molecules_list)-1})"
                        )

                if valid_indices:
                    selected_indices = valid_indices
                    break
                else:
                    print("No valid indices provided.")
                    attempt += 1

            except ValueError:
                print("Invalid input format. Please enter comma-separated numbers.")
                attempt += 1
            except KeyboardInterrupt:
                print("\nSelection cancelled by user.")
                break

        # Close the display window
        cv2.destroyWindow("Molecule Selection")

        # Return selected molecules based on indices
        if selected_indices:
            selected_molecules = [molecules_list[idx] for idx in selected_indices]
            print(
                f"Selected {len(selected_molecules)} molecules for manipulation: {selected_indices}"
            )
            return selected_molecules
        else:
            print("No molecules selected.")
            return []

    def spectroscopy_workflow_summary(self):
        """
        Generate a comprehensive summary of the spectroscopy workflow session

        Returns:
            dict: Summary statistics and results
        """
        if not hasattr(self, "molecule_registry") or not self.molecule_registry:
            return {"error": "No molecule registry available"}

        total_molecules = len(self.molecule_registry.molecules)
        manipulated_molecules = sum(
            1 for mol in self.molecule_registry.molecules if mol.is_manipulated
        )
        spectroscopy_completed = sum(
            1 for mol in self.molecule_registry.molecules if mol.spectroscopy_completed
        )

        total_spectroscopy_points = sum(
            mol.spectroscopy_points_count for mol in self.molecule_registry.molecules
        )
        successful_measurements = sum(
            mol.successful_measurements for mol in self.molecule_registry.molecules
        )

        success_rate = (
            (successful_measurements / total_spectroscopy_points * 100)
            if total_spectroscopy_points > 0
            else 0
        )

        summary = {
            "session_info": {
                "start_time": self.start_time,
                "session_duration": (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    if hasattr(self, "start_time")
                    else "Unknown"
                ),
            },
            "molecule_statistics": {
                "total_detected": total_molecules,
                "manipulated": manipulated_molecules,
                "spectroscopy_completed": spectroscopy_completed,
                "manipulation_rate": (
                    (manipulated_molecules / total_molecules * 100)
                    if total_molecules > 0
                    else 0
                ),
                "spectroscopy_completion_rate": (
                    (spectroscopy_completed / total_molecules * 100)
                    if total_molecules > 0
                    else 0
                ),
            },
            "spectroscopy_statistics": {
                "total_measurements": total_spectroscopy_points,
                "successful_measurements": successful_measurements,
                "success_rate": success_rate,
                "average_points_per_molecule": (
                    (total_spectroscopy_points / spectroscopy_completed)
                    if spectroscopy_completed > 0
                    else 0
                ),
            },
            "molecule_details": [
                mol.get_spectroscopy_summary()
                for mol in self.molecule_registry.molecules
            ],
        }

        return summary

    def export_spectroscopy_results(self, export_path=None):
        """
        Export all spectroscopy results to a structured format

        Args:
            export_path: Path to save the export file (optional)

        Returns:
            str: Path to the exported file
        """
        if export_path is None:
            export_path = os.path.join(self.log_path, "spectroscopy_export.json")

        # Generate comprehensive export data
        export_data = {
            "session_summary": self.spectroscopy_workflow_summary(),
            "detailed_results": [],
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }

        # Add detailed results for each molecule
        if hasattr(self, "molecule_registry") and self.molecule_registry:
            for idx, molecule in enumerate(self.molecule_registry.molecules):
                molecule_data = {
                    "molecule_index": idx,
                    "position": molecule.position,
                    "molecule_type": molecule.molecule_type,
                    "registration_time": molecule.registration_time,
                    "is_manipulated": molecule.is_manipulated,
                    "operated_time": molecule.operated_time,
                    "spectroscopy_completed": molecule.spectroscopy_completed,
                    "spectroscopy_time": molecule.spectroscopy_time,
                    "spectroscopy_points": molecule.spectroscopy_points,
                    "spectroscopy_results": molecule.spectroscopy_results,
                    "success_rate": molecule.get_spectroscopy_success_rate(),
                }
                export_data["detailed_results"].append(molecule_data)

        # Save to file
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Spectroscopy results exported to: {export_path}")
        return export_path

    def reset_spectroscopy_session(self):
        """
        Reset the spectroscopy session for a new run
        """
        if hasattr(self, "molecule_registry"):
            self.molecule_registry = Registry()

        # Reset session tracking variables
        self.start_time = time.strftime(
            "%Y-%m-%d %H-%M-%S", time.localtime(time.time())
        )
        self.skip_flag = 0
        self.scan_qulity = 1

        # Create new log directory for the new session
        self.log_path = self.main_data_save_path + "/" + self.start_time
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        print(f"Spectroscopy session reset. New session: {self.start_time}")

    def validate_spectroscopy_setup(self):
        """
        Validate that all necessary components are available for spectroscopy workflow

        Returns:
            dict: Validation results with status and recommendations
        """
        validation = {
            "status": "PASS",
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }

        # Check essential components
        if not hasattr(self, "molecule_registry"):
            validation["errors"].append("Molecule registry not initialized")
            validation["status"] = "FAIL"

        if not hasattr(self, "tip_manipulate_speed"):
            validation["warnings"].append("Tip manipulation speed not set")
            validation["recommendations"].append("Set tip_manipulate_speed parameter")

        # Check model paths
        if not os.path.exists(self.quality_model_path):
            validation["warnings"].append(
                f"Quality model not found: {self.quality_model_path}"
            )
            validation["recommendations"].append(
                "Check quality model path and file existence"
            )

        if not os.path.exists(self.keypoint_model_path):
            validation["warnings"].append(
                f"Keypoint model not found: {self.keypoint_model_path}"
            )
            validation["recommendations"].append(
                "Check keypoint model path and file existence"
            )

        # Check directories
        if not os.path.exists(self.log_path):
            validation["warnings"].append("Log directory not found")
            validation["recommendations"].append(
                "Run tip_init() to create log directories"
            )

        # Check nanonis mode
        if self.nanonis_mode not in ["demo", "real"]:
            validation["errors"].append(f"Invalid nanonis mode: {self.nanonis_mode}")
            validation["status"] = "FAIL"

        # Determine overall status
        if validation["errors"]:
            validation["status"] = "FAIL"
        elif validation["warnings"]:
            validation["status"] = "WARNING"

        return validation

    def perform_batch_manipulation(
        self, selected_molecules, tip_bias, tip_current, tip_induce_mode="CC"
    ):
        """
        Perform manipulation on selected molecules

        Args:
            selected_molecules: List of molecules to manipulate
            tip_bias: Bias voltage for manipulation
            tip_current: Current for manipulation
            tip_induce_mode: Manipulation mode ('CC', 'CH', 'pulse')

        Returns:
            dict: Results of manipulation for each molecule
        """

        manipulation_results = {}

        print(f"Starting batch manipulation of {len(selected_molecules)} molecules")

        for idx, molecule in enumerate(selected_molecules):
            # molecule coordinates are already in nanometer units from molecular_seeker
            # Extract nm coordinates (molecule format: [class, x_nm, y_nm, w_nm, h_nm, ...])
            mol_position_nano = (
                molecule[1] * 1e-9,
                molecule[2] * 1e-9,
            )  # Already in nm units

            print(
                f"Manipulating molecule {idx + 1}/{len(selected_molecules)} at {mol_position_nano} nm"
            )

            # Perform manipulation
            result = self.perform_tip_manipulation(
                mol_position_nano, tip_bias, tip_current, tip_induce_mode
            )

            manipulation_results[idx] = {
                "molecule_data": molecule,
                "position": mol_position_nano,
                "manipulation_result": result,
                "timestamp": time.time(),
            }

            # Brief pause between manipulations
            time.sleep(0.5)

        return manipulation_results

    def save_manipulation_results(self, manipulation_results, image_save_time):
        """Save manipulation results to file"""
        manipulation_save_path = self.mol_tip_induce_path + "/manipulation/"
        if not os.path.exists(manipulation_save_path):
            os.makedirs(manipulation_save_path)

        results_filename = f"manipulation_results_{image_save_time}.json"
        results_filepath = os.path.join(manipulation_save_path, results_filename)

        # Make results JSON serializable
        serializable_results = {}
        for key, value in manipulation_results.items():
            serializable_results[str(key)] = {
                "molecule_data": [float(x) for x in value["molecule_data"]],
                "position": [float(x) for x in value["position"]],
                "manipulation_result": value["manipulation_result"],
                "timestamp": value["timestamp"],
            }

        with open(results_filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Manipulation results saved to {results_filepath}")
        return results_filepath
