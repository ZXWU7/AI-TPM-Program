# Wuzhengxiao 2025/7/27

import json
import os
import pickle
import random
import re
import shutil
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
        self.nanocoodinate_list = []
        # the list to save the coodinate which send to nanonis directly
        self.visual_circle_buffer_list = []

        # Line scan and quality monitoring
        self.line_scan_change_times = 0  # initialize the line scan change times to 0
        self.AdjustTip_flag = (
            0  # 0: the tip is not adjusted, 1: the tip is adjusted just now
        )

        self.Scan_edge = "20n"  # set the initial scan square edge length
        # self.scan_zoom_V1 = (
        #     "20n"  # the zoom of the scan, 20nm. the scale for positioning the molecule
        # )
        # self.scan_zoom_V2 = (
        #     "10n"  # the zoom of the scan, 10nm. the scale for tip induce
        # )
        # self.scan_zoom_V3 = "5n"  # the zoom of the scan, 5nm. the scale for tip induce
        self.zoom_out_scale = "100n"
        self.scan_zoom_in_list = ["20n"]  # the list of the zoom in scan adge
        # FIXED: Use consistent scan size across all functions (using first element only)
        self.molecular_registration_list = (
            []
        )  # the list to save the molecular registration data
        self.molecular_filter_threshold = 5  # if the distance between the molecular is less than the threshold, the molecular will be filterd
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

        self.nanonis_mode = "demo"  # the nanonis mode, 'demo' or 'real'

        if self.nanonis_mode == "demo":
            self.signal_channel_list = [
                0,
                30,
            ]  # the mode channel list, 0 is Current, 30 is Z_m
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
            "AI_TPM/keypoint/best.pt"  # molecule-center keypoint detection model path
        )

        self.manipulation_check_ai_path = "AI_TPM/Manipulation_Check_AI/"  # the path of the trained AI agent for manipulation classification

        self.keypoints_model_path = "AI_TPM/keypoints_model/best_0417.pt"  # the path of the trained AI agent for spectroscopy point prediction

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
        """
        Save the current state of serializable instance attributes to a file.
         Handles numpy arrays, torch tensors, deques, and common non-serializable types.
        """

        def make_serializable(val):
            # Convert numpy arrays to lists
            if isinstance(val, np.ndarray):
                return val.tolist()
            # Convert torch tensors to lists
            if "torch" in sys.modules and isinstance(val, torch.Tensor):
                return val.cpu().detach().tolist()
            # Convert deque to list
            if isinstance(val, deque):
                return list(val)
            # Recursively handle dicts
            if isinstance(val, dict):
                return {k: make_serializable(v) for k, v in val.items()}
            # Recursively handle lists/tuples
            if isinstance(val, (list, tuple)):
                return [make_serializable(v) for v in val]
            # Otherwise, return as is (if serializable)
            return val

        serializable_attrs = {}
        failed_attrs = []
        for k, v in self.__dict__.items():
            try:
                serializable_attrs[k] = make_serializable(v)
                # Test if it can be JSON serialized
                json.dumps(serializable_attrs[k])
            except Exception:
                failed_attrs.append(k)

        filename = os.path.join(self.log_path, "checkpoint.json")
        # Backup previous checkpoint if exists
        if os.path.exists(filename):
            backup_name = filename.replace(".json", "_backup.json")
            shutil.copy(filename, backup_name)

        with open(filename, "w") as file:
            json.dump(serializable_attrs, file, indent=2)

        if failed_attrs:
            print(
                f"Warning: The following attributes could not be serialized and were skipped: {failed_attrs}"
            )

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
        # For keypoint, use the same position as molecule center if not provided separately
        key_x, key_y = mol_x, mol_y

        # Convert to pixel coordinates
        mol_px, mol_py = real_to_pixel_coords(mol_x, mol_y)
        key_px, key_py = real_to_pixel_coords(key_x, key_y)

        # Draw molecule bounding box (blue for all molecules)
        mol_w = 1.0  # Convert width to nm (assume 1nm default)
        mol_h = 1.0  # Convert height to nm (assume 1nm default)

        # Estimate bounding box in pixels
        box_w_px = max(5, int((mol_w / (edge_meters * 1e9)) * image_width))
        box_h_px = max(5, int((mol_h / (edge_meters * 1e9)) * image_height))

        # Draw bounding rectangle
        cv2.rectangle(
            vis_image,
            (mol_px - box_w_px // 2, mol_py - box_h_px // 2),
            (mol_px + box_w_px // 2, mol_py + box_h_px // 2),
            (255, 100, 100),
            1,
        )  # Light blue

        # Draw molecule center
        cv2.circle(vis_image, (mol_px, mol_py), 3, (255, 150, 150), -1)  # Blue circle

        # Draw keypoint
        cv2.circle(vis_image, (key_px, key_py), 2, (0, 255, 255), -1)  # Yellow circle

        # Add molecule ID
        cv2.putText(
            vis_image,
            f"M{mol_count+1}",
            (mol_px + 5, mol_py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )

        # Draw selected molecule with thicker red outline (this is the main molecule)
        # Draw thick red bounding rectangle
        box_w_px = max(8, int((mol_w / (edge_meters * 1e9)) * image_width))
        box_h_px = max(8, int((mol_h / (edge_meters * 1e9)) * image_height))

        cv2.rectangle(
            vis_image,
            (mol_px - box_w_px // 2, mol_py - box_h_px // 2),
            (mol_px + box_w_px // 2, mol_py + box_h_px // 2),
            (0, 0, 255),
            2,
        )  # Red, thick

        # Draw selected molecule center (larger red circle)
        cv2.circle(vis_image, (mol_px, mol_py), 4, (0, 0, 255), -1)  # Red circle

        # Draw keypoint (larger red circle)
        cv2.circle(
            vis_image, (key_px, key_py), 3, (0, 100, 255), -1
        )  # Red-orange circle

        # Add selection number
        cv2.putText(
            vis_image,
            f"S{mol_count+1}",
            (mol_px + 8, mol_py + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

        # Draw manipulation point if manipulation was performed
        if manipulation_position is not None:
            manip_x, manip_y = manipulation_position[0], manipulation_position[1]
            manip_px, manip_py = real_to_pixel_coords(manip_x, manip_y)

            if manipulation_success:
                color = (0, 255, 0)  # Green for successful manipulation
                marker_size = 5
                label = "Manip Success"
            else:
                color = (0, 165, 255)  # Orange for failed manipulation
                marker_size = 4
                label = "Manip Failed"

            cv2.circle(vis_image, (manip_px, manip_py), marker_size, color, -1)
            cv2.putText(
                vis_image,
                label,
                (manip_px + 8, manip_py + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
            )

        # Draw spectroscopy points if spectroscopy was performed
        if spectroscopy_points:
            for spec_idx, spec_point in enumerate(spectroscopy_points):
                spec_x, spec_y = spec_point[0], spec_point[1]
                spec_px, spec_py = real_to_pixel_coords(spec_x, spec_y)

                # Different colors for successful vs failed measurements
                if spectroscopy_results and spec_idx < len(spectroscopy_results):
                    spec_result = spectroscopy_results[spec_idx]
                    if spec_result.get("success", False):
                        color = (0, 255, 0)  # Green for success
                        marker_size = 4
                    else:
                        color = (0, 165, 255)  # Orange for failure
                        marker_size = 3
                else:
                    color = (128, 128, 128)  # Gray for unknown status
                    marker_size = 3

                cv2.circle(vis_image, (spec_px, spec_py), marker_size, color, -1)
                cv2.putText(
                    vis_image,
                    f"Spec{spec_idx+1}",
                    (spec_px + 6, spec_py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )

        # Add scan center marker
        center_px, center_py = real_to_pixel_coords(scan_center[0], scan_center[1])
        cv2.drawMarker(
            vis_image, (center_px, center_py), (255, 255, 255), cv2.MARKER_CROSS, 10, 2
        )

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
        # Save annotated image
        mol_image_path = (
            self.mol_tip_induce_path
            + f"/annotated_scan_{image_save_time}_mol_{mol_count+1}.png"
        )
        cv2.imwrite(mol_image_path, annotated_image)

        # Save detailed JSON result for this molecule
        json_result_path = (
            self.mol_tip_induce_path
            + f"/result_data_{image_save_time}_mol_{mol_count+1}.json"
        )
        with open(json_result_path, "w") as f:
            json.dump(result_data, f, indent=2, default=str)

        file_paths = {
            "annotated_image": mol_image_path,
            "result_data": json_result_path,
            "spectroscopy_folder": None,
        }

        # Save individual spectroscopy data files if spectroscopy was performed
        if spectroscopy_results_molecule and spectroscopy_points:
            spec_folder = (
                self.mol_tip_induce_path
                + f"/spectroscopy/scan_{result_data['scan_info']['scan_count']}_mol_{mol_count+1}/"
            )
            if not os.path.exists(spec_folder):
                os.makedirs(spec_folder)

            file_paths["spectroscopy_folder"] = spec_folder

            for spec_idx, spec_result in enumerate(spectroscopy_results_molecule):
                spec_file_path = spec_folder + f"spectroscopy_point_{spec_idx+1}.json"
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
                with open(spec_file_path, "w") as f:
                    json.dump(spec_data, f, indent=2, default=str)

        return file_paths

    def perform_spectroscopy_measurement(
        self,
        spectroscopy_point,
        molecular_index,
        point_index,
        image_save_time,
        image_with_molecule,
    ):
        """
        Perform bias spectroscopy measurement at a specific point with tip visualization

        Args:
            spectroscopy_point: (x, y) coordinates for measurement in nanometers
            molecular_index: Index of the molecule
            point_index: Index of the spectroscopy point
            image_save_time: Timestamp for file naming
            image_with_molecule: STM image with molecule annotations

        Returns:
            dict: Measurement results and metadata
        """
        print(
            f"Performing bias spectroscopy at point {spectroscopy_point} on molecule {molecular_index}"
        )

        try:
            # Move tip to spectroscopy point
            print(f"Moving tip to spectroscopy position: {spectroscopy_point}")
            self.FolMeSpeedSet(self.zoom_in_tip_speed, 1)
            self.TipXYSet(spectroscopy_point[0], spectroscopy_point[1])
            self.FolMeSpeedSet(self.zoom_in_tip_speed, 0)
            time.sleep(1)

            # Set bias spectroscopy parameters
            spectroscopy_bias_start = -2.0  # V
            spectroscopy_bias_end = 2.0  # V
            spectroscopy_points = 512

            # Configure spectroscopy channels for bias spectroscopy
            spectroscopy_channels = [0, 1, 2]  # Current, dI/dV, d2I/dV2
            self.BiasSpectrChsSet(spectroscopy_channels)

            # Set bias spectroscopy limits
            self.BiasSpectrLimitsSet(spectroscopy_bias_start, spectroscopy_bias_end)

            # Start bias spectroscopy measurement
            save_base_name = (
                f"bias_spec_{image_save_time}_mol_{molecular_index}_pt_{point_index}"
            )
            self.BiasSpectrStart(save_base_name)

            # Wait for completion
            print("Waiting for bias spectroscopy to complete...")
            while self.BiasSpectrStatusGet() == 1:
                time.sleep(0.1)

            # Get spectroscopy data
            spectroscopy_data = []
            channels_names = ["Bias(V)", "Current(A)", "dI/dV", "d2I/dV2"]

            # Create visualization image with tip position
            vis_image = image_with_molecule.copy()

            # Convert spectroscopy point to pixel coordinates for visualization
            # Assuming image is 304x304 and represents the scan area
            image_center_x = 152  # Center of 304x304 image
            image_center_y = 152

            # Get scan area size from current zoom level
            zoom_scale = self.convert(
                self.scan_zoom_in_list[0]
            )  # Use first element consistently
            pixels_per_nm = 304 / (zoom_scale * 1e9)  # pixels per nanometer

            # Convert nanometer offset to pixel offset
            tip_pixel_x = int(
                image_center_x + spectroscopy_point[0] * 1e9 * pixels_per_nm
            )
            tip_pixel_y = int(
                image_center_y + spectroscopy_point[1] * 1e9 * pixels_per_nm
            )

            # Draw tip position on image
            cv2.circle(
                vis_image, (tip_pixel_x, tip_pixel_y), 5, (0, 0, 255), -1
            )  # Red circle for tip
            cv2.putText(
                vis_image,
                f"Spec{point_index}",
                (tip_pixel_x + 8, tip_pixel_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )

            # Save data and visualization
            spectroscopy_save_path = self.mol_tip_induce_path + "/spectroscopy/"
            if not os.path.exists(spectroscopy_save_path):
                os.makedirs(spectroscopy_save_path)

            # Save visualization image
            vis_filename = (
                spectroscopy_save_path
                + f"bias_spec_vis_{image_save_time}_mol_{molecular_index}_pt_{point_index}.png"
            )
            cv2.imwrite(vis_filename, vis_image)

            # Save spectroscopy data file
            spectroscopy_filename = (
                spectroscopy_save_path
                + f"bias_spec_{image_save_time}_mol_{molecular_index}_pt_{point_index}.dat"
            )

            # Create header with measurement parameters
            header_info = f"Bias Spectroscopy Measurement\n"
            header_info += f"Position: {spectroscopy_point} nm\n"
            header_info += f"Molecule: {molecular_index}, Point: {point_index}\n"
            header_info += (
                f"Bias Range: {spectroscopy_bias_start}V to {spectroscopy_bias_end}V\n"
            )
            header_info += f"Points: {spectroscopy_points}\n"
            header_info += f"Channels: {', '.join(channels_names)}\n"
            header_info += f"Timestamp: {image_save_time}\n"

            # Save data (even if empty for record keeping)
            np.savetxt(
                spectroscopy_filename, spectroscopy_data, header=header_info, fmt="%.6e"
            )

            print(f"Bias spectroscopy completed and saved to {spectroscopy_filename}")
            print(f"Visualization saved to {vis_filename}")

            return {
                "success": True,
                "filename": spectroscopy_filename,
                "vis_filename": vis_filename,
                "data": spectroscopy_data,
                "position": spectroscopy_point,
                "channels": channels_names,
                "bias_range": (spectroscopy_bias_start, spectroscopy_bias_end),
                "measurement_type": "bias_spectroscopy",
            }

        except Exception as e:
            print(f"Bias spectroscopy measurement failed at {spectroscopy_point}: {e}")
            return {
                "success": False,
                "error": str(e),
                "position": spectroscopy_point,
                "measurement_type": "bias_spectroscopy",
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

            if not self.tippathvisualQueue.empty():
                circle = self.tippathvisualQueue.get()

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

                        # cv2.rectangle(self.tip_path_img, left_top, right_bottom, square_color, -1)
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
                        # cv2.rectangle(self.tip_path_img, left_top, right_bottom, square_color, -1)                        # draw the square

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
                    # save circle_list as a npy file
                    # np.save(self.log_path + '/circle_list.npy', self.circle_list)
                    # save coodinate_list as a npy file
                    # np.save(self.log_path + '/nanocoodinate_list.npy', self.nanocoodinate_list)
                elif circle == "end":
                    # save the tip path image
                    cv2.imwrite(self.log_path + "/tip_path" + ".jpg", self.tip_path_img)
                    # np.save(self.log_path + '/circle_list.npy', self.circle_list)
                    # np.save(self.log_path + '/nanocoodinate_list.npy', self.nanocoodinate_list)
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
        # tip_visualization_thread.start()
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

    def batch_scan_producer(
        self, Scan_posion=(0.0, 0.0), Scan_edge="20n", Scan_pix=304, angle=0
    ):
        if self.skip_flag == 1:
            print("creating skip data...")
            Scan_data_for = {
                "data": np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                "row": Scan_pix,
                "col": Scan_pix,
                "scan_direction": 0,
                "channel_name": "Z (m)",
            }
            Scan_data_back = {
                "data": np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                "row": Scan_pix,
                "col": Scan_pix,
                "scan_direction": 0,
                "channel_name": "Z (m)",
            }
        else:
            print("Scaning the image...")
            # ScanBuffer = self.ScanBufferGet()
            self.ScanBufferSet(
                Scan_pix, Scan_pix, self.signal_channel_list
            )  # 14 is the index of the Z_m channel in real scan mode , in demo mode, the index of the Z_m channel is 30
            self.ScanPropsSet(
                Continuous_scan=2,
                Bouncy_scan=2,
                Autosave=1,
                Series_name=" ",
                Comment="inter_closest",
            )  # close continue_scan & bouncy_scan, but save all data
            self.ScanFrameSet(
                Scan_posion[0], Scan_posion[1], Scan_edge, Scan_edge, angle=angle
            )

            self.ScanStart()
            time.sleep(0.5)

            # while self.ScanStatusGet() == 1: # detect the scan status until the scan is complete.
            #     #   !!!   Note：Do not use self.WaitEndOfScan() here   !!!!, it will block the program!!!!!!
            #     time.sleep(0.5)

            self.WaitEndOfScan()  # wait for the scan to be complete
            print(self.signal_channel_list[-1])
            try:  # some times the scan data is not successful because of the TCP/IP communication problem
                Scan_data_for = self.ScanFrameData(
                    self.signal_channel_list[-1], data_dir=1
                )
                if Scan_data_for["data"].shape == (0, 0):
                    Scan_data_for["data"] = cv2.imread(
                        "AI_TPM/STM_img_simu/TPM_image/001.png", cv2.IMREAD_GRAYSCALE
                    )  # read the simu image
                    Scan_data_for["data"] = cv2.resize(
                        Scan_data_for["data"],
                        (304, 304),
                        interpolation=cv2.INTER_AREA,
                    )
            except:  # if is not successful, set fake data
                Scan_data_for = {
                    "data": np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                    "row": Scan_pix,
                    "col": Scan_pix,
                    "scan_direction": 0,
                    "channel_name": "Z (m)",
                }
            time.sleep(1)
            try:
                Scan_data_back = self.ScanFrameData(
                    self.signal_channel_list[-1], data_dir=0
                )  # failed because the TCP problem
                if Scan_data_back["data"].shape == (0, 0):
                    Scan_data_back["data"] = cv2.imread(
                        "AI_TPM/STM_img_simu/TPM_image/001.png", cv2.IMREAD_GRAYSCALE
                    )  # read the simu image
                    Scan_data_back["data"] = cv2.resize(
                        Scan_data_back["data"],
                        (304, 304),
                        interpolation=cv2.INTER_AREA,
                    )

            except:
                Scan_data_back = {
                    "data": np.ones((Scan_pix, Scan_pix), np.uint8) * 0.1,
                    "row": Scan_pix,
                    "col": Scan_pix,
                    "scan_direction": 0,
                    "channel_name": "Z (m)",
                }

            # if the first element and the last element of the Scan_data_for and Scan_data_back is NaN, the scan is not successful
            if (
                np.isnan(Scan_data_for["data"][0][0])
                or np.isnan(Scan_data_for["data"][-1][-1])
                or np.isnan(Scan_data_back["data"][0][0])
                or np.isnan(Scan_data_back["data"][-1][-1])
            ):
                Scan_data_for["data"][np.isnan(Scan_data_for["data"])] = 0
                Scan_data_back["data"][np.isnan(Scan_data_back["data"])] = 0

        self.image_for = linear_normalize_whole(
            Scan_data_for["data"]
        )  # image_for and image_back are 2D nparray
        self.image_back = linear_normalize_whole(Scan_data_back["data"])

        self.image_for = images_equalization(
            self.image_for, alpha=self.equalization_alpha
        )
        self.image_back = images_equalization(
            self.image_back, alpha=self.equalization_alpha
        )

        self.image_for_tensor = torch.tensor(
            self.image_for, dtype=torch.float32, device=self.device
        ).unsqueeze(
            0
        )  # for instance tensor.shape are [1, 1, 256, 256]  which is for DQN
        self.image_back_tensor = torch.tensor(
            self.image_back, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.Scan_data = {
            "Scan_data_for": Scan_data_for,
            "Scan_data_back": Scan_data_back,
        }
        self.Scan_image = {"image_for": self.image_for, "image_back": self.image_back}
        self.ScandataQueue.put(
            self.Scan_data
        )  # put the batch scan data into the queue, blocking if Queue is full
        print("Scaning complete! \n ready to save...")
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
        if len(self.nanocoodinate_list) == 0:
            self.nanocoodinate = pix_to_nanocoordinate(
                self.inter_closest, plane_size=self.plane_size
            )  # init the first point of the nanocoodinate_list
            self.nanocoodinate_list.append(self.nanocoodinate)
        else:
            if self.tip_move_mode == 0:
                self.inter_closest = Next_inter(
                    self.circle_list, plane_size=self.plane_size
                )  # calculate the next inter_closest
            self.nanocoodinate = pix_to_nanocoordinate(
                self.inter_closest, plane_size=self.plane_size
            )
            self.nanocoodinate_list.append(self.nanocoodinate)

        self.tippathvisualQueue.put(
            self.inter_closest
        )  # put the inter_closest to the tippathvisualQueue
        print("The scan center is " + str(self.inter_closest))

        print("move to the area...")
        self.ScanFrameSet(
            self.nanocoodinate[0],
            self.nanocoodinate[1] + 0.5 * self.Scan_edge_SI,
            self.Scan_edge,
            self.Scan_edge,
            angle=0,
        )
        np.save(self.log_path + "/nanocoodinate_list.npy", self.nanocoodinate_list)

    def move_to_next_point_V2(self):
        """
        Move to the next point in the scan path based on the current tip position.
        This method updates the inter_closest position and appends it to the nanocoodinate_list.
        """
        if len(self.nanocoodinate_list) == 0:
            self.nanocoodinate = pix_to_nanocoordinate(
                self.inter_closest, plane_size=self.plane_size
            )  # Initialize the first point of the nanocoodinate_list
            self.nanocoodinate_list.append(self.nanocoodinate)
        else:
            if self.tip_move_mode == 0:
                self.inter_closest = Next_inter(
                    self.circle_list, plane_size=self.plane_size
                )
            self.nanocoodinate = pix_to_nanocoordinate(
                self.inter_closest, plane_size=self.plane_size
            )
            self.nanocoodinate_list.append(self.nanocoodinate)

    def generate_systematic_grid_positions(
        self, center_position=(0, 0), scan_area_size=None, grid_spacing=None, ratio=0.9
    ):
        """
        Generate a 2D systematic cubic grid of scan positions covering the specified area.

        Args:
            center_position: (x, y) center position in meters, default is (0, 0)
            scan_area_size: size of the scan area in meters, default uses zoom_out_scale
            grid_spacing: spacing between grid points in meters, default uses Scan_edge

        Returns:
            list: List of (x, y) coordinates in meters representing scan positions
        """
        # Use default values if not provided
        if scan_area_size is None:
            scan_area_size = self.convert(
                self.zoom_out_scale
            )  # Convert "100n" to meters

        if grid_spacing is None:
            grid_spacing = ratio * self.convert(
                self.scan_zoom_in_list[0]
            )  # Convert "20n" to meters

        # Calculate the number of grid points in each direction
        # Ensure we have odd numbers to include the center point
        num_points_x = int(scan_area_size / grid_spacing)
        num_points_y = int(scan_area_size / grid_spacing)

        # Make sure we have odd numbers for symmetric grid around center
        if num_points_x % 2 == 0:
            num_points_x += 1
        if num_points_y % 2 == 0:
            num_points_y += 1

        # Calculate half-ranges for symmetric grid
        half_range_x = (num_points_x - 1) // 2
        half_range_y = (num_points_y - 1) // 2

        # Generate grid positions
        grid_positions = []

        for i in range(-half_range_x, half_range_x + 1):
            for j in range(-half_range_y, half_range_y + 1):
                x_pos = center_position[0] + i * grid_spacing
                y_pos = center_position[1] + j * grid_spacing
                grid_positions.append((x_pos, y_pos))

        print(
            f"  - Scan area: {scan_area_size*1e9:.1f} nm x {scan_area_size*1e9:.1f} nm"
        )
        print(f"  - Grid size: {num_points_x} x {num_points_y}")

        return grid_positions

    def generate_optimized_scan_path(self, grid_positions):
        """
        Generate an optimized scan path through the grid positions to minimize travel distance.
        Uses a simple zigzag pattern for efficient scanning.

        Args:
            grid_positions: list of (x, y) coordinates in meters

        Returns:
            list: Optimized sequence of (x, y) coordinates
        """
        if not grid_positions:
            return []

        # Sort positions by y-coordinate first, then by x-coordinate
        # This creates rows of points
        sorted_positions = sorted(grid_positions, key=lambda pos: (pos[1], pos[0]))

        # Group by y-coordinate (rows)
        rows = {}
        for pos in sorted_positions:
            y = pos[1]
            if y not in rows:
                rows[y] = []
            rows[y].append(pos)

        # Create zigzag pattern: alternate direction for each row
        optimized_path = []
        reverse_row = False

        for y in sorted(rows.keys()):
            row_positions = sorted(rows[y], key=lambda pos: pos[0])
            if reverse_row:
                row_positions.reverse()
            optimized_path.extend(row_positions)
            reverse_row = not reverse_row

        print(f"Optimized scan path with {len(optimized_path)} positions")
        print(f"  - Pattern: Zigzag rows for minimal travel distance")

        return optimized_path

    def move_to_grid_position(self, position_index=0, grid_positions=None):
        """
        Move to a specific position in the systematic grid.

        Args:
            position_index: index of the position in the grid list
            grid_positions: list of grid positions, if None uses stored grid

        Returns:
            tuple: (x, y) position in meters where the tip moved
        """
        if grid_positions is None:
            if not hasattr(self, "systematic_grid_positions"):
                raise ValueError("No grid positions available. Generate grid first.")
            grid_positions = self.systematic_grid_positions

        if position_index >= len(grid_positions):
            raise ValueError(
                f"Position index {position_index} out of range (max: {len(grid_positions)-1})"
            )

        target_position = grid_positions[position_index]

        # Update coordinate tracking
        self.nanocoodinate = target_position
        self.nanocoodinate_list.append(self.nanocoodinate)

        print(
            f"Moving to grid position {position_index+1}/{len(grid_positions)}: "
            f"({target_position[0]*1e9:.1f}, {target_position[1]*1e9:.1f}) nm"
        )

        # Set scan frame at the target position
        self.ScanFrameSet(
            target_position[0],
            target_position[1] + 0.5 * self.Scan_edge_SI,
            self.Scan_edge,
            self.Scan_edge,
            angle=0,
        )

        # Save updated coordinate list
        np.save(self.log_path + "/nanocoodinate_list.npy", self.nanocoodinate_list)

        return target_position

    def initialize_systematic_scanning(
        self, center_position=(0, 0), scan_area_size=None, grid_spacing=None
    ):
        """
        Initialize systematic 2D grid scanning with optimized path.

        Args:
            center_position: (x, y) center position in meters
            scan_area_size: size of the scan area in meters (default: zoom_out_scale)
            grid_spacing: spacing between grid points in meters (default: Scan_edge)

        Returns:
            dict: Information about the generated grid and path
        """
        # Generate grid positions
        grid_positions = self.generate_systematic_grid_positions(
            center_position, scan_area_size, grid_spacing
        )

        # Optimize scan path
        optimized_path = self.generate_optimized_scan_path(grid_positions)

        # Store for later use
        self.systematic_grid_positions = optimized_path
        self.grid_scan_index = 0

        # Create visualization data
        grid_info = {
            "total_positions": len(optimized_path),
            "center_position": center_position,
            "scan_area_size": scan_area_size or self.convert(self.zoom_out_scale),
            "grid_spacing": grid_spacing or self.convert(self.Scan_edge),
            "positions": optimized_path,
        }

        print("Systematic scanning initialized:")
        print(f"  - Total scan positions: {grid_info['total_positions']}")
        print(
            f"  - Scan area: {grid_info['scan_area_size']*1e9:.1f} nm x {grid_info['scan_area_size']*1e9:.1f} nm"
        )
        print(f"  - Grid spacing: {grid_info['grid_spacing']*1e9:.1f} nm")

        return grid_info

    def move_to_next_grid_position(self):
        """
        Move to the next position in the systematic grid scan.

        Returns:
            tuple: (x, y) position in meters, or None if scan complete
        """
        if not hasattr(self, "systematic_grid_positions"):
            raise ValueError(
                "Systematic scanning not initialized. Call initialize_systematic_scanning() first."
            )

        if self.grid_scan_index >= len(self.systematic_grid_positions):
            print("Systematic grid scan completed!")
            return None

        position = self.move_to_grid_position(
            self.grid_scan_index, self.systematic_grid_positions
        )
        self.grid_scan_index += 1

        return position

    def visualize_systematic_grid(self, grid_positions=None, save_path=None):
        """
        Create a visualization of the systematic grid scan pattern.

        Args:
            grid_positions: list of grid positions, if None uses stored grid
            save_path: path to save the visualization, if None uses default log path

        Returns:
            numpy.ndarray: visualization image
        """
        if grid_positions is None:
            if not hasattr(self, "systematic_grid_positions"):
                raise ValueError("No grid positions available. Generate grid first.")
            grid_positions = self.systematic_grid_positions

        # Create visualization image
        vis_img = np.ones((800, 800, 3), dtype=np.uint8) * 255  # White background

        # Calculate conversion factors
        if grid_positions:
            # Find the bounds of the grid
            x_coords = [pos[0] for pos in grid_positions]
            y_coords = [pos[1] for pos in grid_positions]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Add padding
            padding = max(abs(x_max - x_min), abs(y_max - y_min)) * 0.1
            x_min -= padding
            x_max += padding
            y_min -= padding
            y_max += padding

            # Convert to pixel coordinates
            def to_pixel(x, y):
                px = int(((x - x_min) / (x_max - x_min)) * 700 + 50)
                py = int(((y - y_min) / (y_max - y_min)) * 700 + 50)
                py = 750 - py  # Flip Y coordinate
                return px, py

            # Draw grid positions and connections
            for i, pos in enumerate(grid_positions):
                px, py = to_pixel(pos[0], pos[1])

                # Draw scan area rectangle
                scan_size_px = int(
                    (self.convert(self.Scan_edge) / (x_max - x_min)) * 700
                )
                scan_size_px = max(5, scan_size_px)  # Minimum visible size

                cv2.rectangle(
                    vis_img,
                    (px - scan_size_px // 2, py - scan_size_px // 2),
                    (px + scan_size_px // 2, py + scan_size_px // 2),
                    (200, 200, 255),
                    1,
                )  # Light red rectangles

                # Draw position marker
                cv2.circle(vis_img, (px, py), 3, (0, 0, 255), -1)  # Red dots

                # Draw connection to next position
                if i < len(grid_positions) - 1:
                    next_pos = grid_positions[i + 1]
                    next_px, next_py = to_pixel(next_pos[0], next_pos[1])
                    cv2.line(
                        vis_img, (px, py), (next_px, next_py), (0, 255, 0), 1
                    )  # Green lines

                # Add position number for first few positions
                if i < 20:
                    cv2.putText(
                        vis_img,
                        str(i + 1),
                        (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 0, 0),
                        1,
                    )

            # Draw start and end markers
            if grid_positions:
                start_px, start_py = to_pixel(
                    grid_positions[0][0], grid_positions[0][1]
                )
                end_px, end_py = to_pixel(grid_positions[-1][0], grid_positions[-1][1])

                cv2.circle(
                    vis_img, (start_px, start_py), 8, (0, 255, 0), 2
                )  # Green start
                cv2.putText(
                    vis_img,
                    "START",
                    (start_px + 10, start_py),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                cv2.circle(vis_img, (end_px, end_py), 8, (255, 0, 0), 2)  # Blue end
                cv2.putText(
                    vis_img,
                    "END",
                    (end_px + 10, end_py),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        # Add title and info
        cv2.putText(
            vis_img,
            "Systematic Grid Scan Pattern",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        if grid_positions:
            info_text = f"Total positions: {len(grid_positions)}"
            cv2.putText(
                vis_img,
                info_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            area_size = self.convert(self.zoom_out_scale) * 1e9
            spacing = self.convert(self.Scan_edge) * 1e9
            info_text2 = (
                f"Area: {area_size:.0f}nm x {area_size:.0f}nm, Spacing: {spacing:.0f}nm"
            )
            cv2.putText(
                vis_img,
                info_text2,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

        # Save visualization
        if save_path is None:
            save_path = self.log_path + "/systematic_grid_visualization.png"

        cv2.imwrite(save_path, vis_img)
        print(f"Grid visualization saved to: {save_path}")

        return vis_img

    def save_grid_configuration(self, grid_info=None, filename=None):
        """
        Save the systematic grid configuration to a JSON file for reproducibility.

        Args:
            grid_info: grid information dictionary
            filename: custom filename for saving
        """
        if grid_info is None and hasattr(self, "systematic_grid_positions"):
            grid_info = {
                "total_positions": len(self.systematic_grid_positions),
                "center_position": (0, 0),  # Default center
                "scan_area_size": self.convert(self.zoom_out_scale),
                "grid_spacing": self.convert(self.Scan_edge),
                "positions": self.systematic_grid_positions,
            }

        if grid_info is None:
            raise ValueError("No grid information available to save.")

        if filename is None:
            filename = self.log_path + "/systematic_grid_config.json"

        # Convert positions to lists for JSON serialization
        save_data = grid_info.copy()
        save_data["positions"] = [
            (float(pos[0]), float(pos[1])) for pos in grid_info["positions"]
        ]
        save_data["center_position"] = (
            float(grid_info["center_position"][0]),
            float(grid_info["center_position"][1]),
        )
        save_data["scan_area_size"] = float(grid_info["scan_area_size"])
        save_data["grid_spacing"] = float(grid_info["grid_spacing"])

        # Add metadata
        save_data["generated_time"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )
        save_data["zoom_out_scale"] = self.zoom_out_scale
        save_data["scan_edge"] = self.Scan_edge

        with open(filename, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"Grid configuration saved to: {filename}")
        return filename

    def example_systematic_scanning_usage(self):
        """
        Example method demonstrating how to use the systematic scanning functionality.
        This shows the typical workflow for setting up and using systematic grid scanning.
        """
        print("=== Systematic Scanning Example ===")

        # Example 1: Basic grid with default parameters
        print("\n1. Creating basic systematic grid...")
        grid_info = self.initialize_systematic_scanning()

        # Visualize the grid
        self.visualize_systematic_grid()

        # Save configuration
        self.save_grid_configuration(grid_info)

        # Example 2: Custom grid with specific parameters
        print("\n2. Creating custom systematic grid...")
        center = (5e-9, -3e-9)  # 5nm right, 3nm down from origin
        area_size = 80e-9  # 80 nm total area
        spacing = 15e-9  # 15 nm between scan points

        custom_grid_info = self.initialize_systematic_scanning(
            center_position=center, scan_area_size=area_size, grid_spacing=spacing
        )

        # Visualize custom grid
        custom_vis_path = self.log_path + "/custom_grid_visualization.png"
        self.visualize_systematic_grid(save_path=custom_vis_path)

        # Example of how to scan through positions
        print("\n3. Example scanning loop...")
        print("   (This would be integrated into your main scanning loop)")
        print("   for scan_iteration in range(total_scans_needed):")
        print("       position = self.move_to_next_grid_position()")
        print("       if position is None:")
        print("           break  # Grid scan completed")
        print("       # Perform actual STM scan at this position")
        print("       # self.batch_scan_producer(position, self.Scan_edge)")

        return grid_info, custom_grid_info

    # def a function to predict the scan qulity
    def image_recognition(self):
        # judge the gap between the max and min of the image
        # if the gap is bigger than the threshold, the scan is skiped
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
        probability = predict_image_quality(self.image_for, self.quality_model_path)
        print("The probability of the good image is " + str(round(probability, 2)))
        if (
            probability > self.scan_qulity_threshold and self.skip_flag == 0
        ):  # 0.5 is the self.scan_qulity_threshold of the probability
            scan_qulity = 1  # good image
        else:
            scan_qulity = 0  # bad image

        # calculate the R depend on the scan_qulity
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
        np.save(self.log_path + "/circle_list.npy", self.circle_list_save)
        self.tippathvisualQueue.put(
            [
                self.inter_closest[0],
                self.inter_closest[1],
                self.R,
                scan_qulity,
                self.coverage,
            ]
        )
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

        # return scan_qulity
        return scan_qulity

    def key_points_convert(self, key_points_result, scan_posion=None, scan_edge=None):
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
        if scan_posion is not None:
            # Ensure scan_posion is in meters
            if abs(scan_posion[0]) > 1e-6:  # If value > 1 micrometer, assume it's in nm
                scan_position = (scan_posion[0] * 1e-9, scan_posion[1] * 1e-9)
            else:
                scan_position = scan_posion  # Already in meters
        else:
            # Convert inter_closest from nm to meters if needed
            if abs(self.inter_closest[0]) > 1e-6:
                scan_position = (
                    self.inter_closest[0] * 1e-9,
                    self.inter_closest[1] * 1e-9,
                )
            else:
                scan_position = self.inter_closest

        # Handle scan edge - convert to meters
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
        selection_strategy="intelligent",
    ):
        """
        Intelligently select molecules for processing using enhanced molecular_tracker logic

        Args:
            shape_key_points_result: List of detected molecules from molecular_seeker
            max_molecules_per_scan: Maximum number of molecules to select
            selection_strategy: Strategy for selection ("intelligent", "closest", "quality", "distance_spread", "random")

        Returns:
            list: Selected molecules for processing
        """
        print(f"Auto-selecting molecules using '{selection_strategy}' strategy")

        if not shape_key_points_result:
            return []

        # Get current scan center for reference
        scan_center = getattr(self, "nanocoodinate", (0.0, 0.0))

        if selection_strategy == "intelligent":
            # Use a combination of quality and spatial distribution
            selected_molecules = []

            # First, sort by a composite score
            def composite_score(molecule_data):
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

                # Class confidence (assuming lower class numbers indicate higher confidence)
                class_score = 1.0 / (mol_class + 1.0)

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
            # Select molecules closest to scan center
            def distance_from_center(molecule_data):
                mol_x = molecule_data[1]  # Already in meters
                mol_y = molecule_data[2]  # Already in meters
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

        elif selection_strategy == "distance_spread":
            # Ensure spatial distribution
            selected_molecules = []
            remaining_molecules = shape_key_points_result.copy()

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

        elif selection_strategy == "random":
            # Random selection
            import random

            selected_molecules = random.sample(
                shape_key_points_result,
                min(max_molecules_per_scan, len(shape_key_points_result)),
            )

        else:  # Default: simple truncation
            selected_molecules = shape_key_points_result[:max_molecules_per_scan]

        print(
            f"Auto-selected {len(selected_molecules)} molecules using '{selection_strategy}' strategy"
        )
        return selected_molecules

    def molecular_seeker(self, image, scan_posion=(0.0, 0.0), scan_edge=30):
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
            key_points_result, scan_posion=scan_posion, scan_edge=scan_edge
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
            self.FolMeSpeedSet(self.zoom_in_tip_speed, 1)
            self.TipXYSet(tip_position[0], tip_position[1])
            self.FolMeSpeedSet(self.zoom_in_tip_speed, 0)
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
