import multiprocessing

# import matplotlib.pyplot as plt
from skimage import exposure, io

from Auto_scan_class import *

# from core import NanonisController
from DQN.agent import *

# from EvaluationCNN.detect import predict_image_quality
# from keypoint.detect import key_detect
# from mol_segment.detect import segmented_image
# from molecule_registry import Molecule, Registry
# from SAC_H_nanonis import ReplayBuffer
# from square_fitting import mol_Br_site_detection
from utils import *

# matplotlib.use('TkAgg')

if __name__ == "__main__":

    code_simulation_mode = False

    if code_simulation_mode:
        ############################################################################################################
        img_simu_30_path = "./STM_img_simu/TPM_image/001.png"
        img_simu_7_path = "./STM_img_simu/TPM_image/001.png"
    ############################################################################################################
    # who is the tip agent
    tip_agent_mode = "SAC"  # 'SAC' or 'human' or 'fixed'

    tip_induce_mode = "pulse"  # 'CC' or 'CH'  CC means Constant I   'CH' means Constant height  'pulse' means pulse mode

    SAC_init_mode = "latest"  # 'new' or 'latest' new: create a new model, latest: load the latest checkpoint model

    lattice_shape = "triangle"

    action_visualize = False

    tri_size = 4

    # nanonis.SAC_init(mode = 'new') # deflaut mode is 'latest' mode = 'new' : create a new model, mode = 'latest' : load the latest checkpoint
    nanonis = Mustard_AI_Nanonis()

    env = Env(polar_space=False)
    agent = SACAgent(env)

    nanonis.tip_init(
        mode="new"
    )  # deflaut mode is 'new' mode = 'new' : the tip is initialized to the center and create a new log folder,
    # mode = 'latest' : load the latest checkpoint

    # nanonis.DQN_init(mode = 'new') # deflaut mode is 'latest' mode = 'new' : create a new model, mode = 'latest' : load the latest checkpoint

    nanonis.monitor_thread_activate()  # activate the monitor thread

    # create a agent thearad
    if tip_agent_mode == "SAC":
        SAC_agent = threading.Thread(
            target=nanonis.SAC_agent_threading,
            args=(
                SAC_init_mode,
                nanonis.auto2SAC_queue,
                nanonis.SAC2auto_queue,
                nanonis.q_save_exp,
            ),
            daemon=True,
        )
        SAC_agent.start()

    if action_visualize:

        action_visualize_Queue = multiprocessing.Queue(50)
        plot_process = multiprocessing.Process(
            target=agent.plot_actions, args=(action_visualize_Queue, env)
        )
        plot_process.start()

    voltage = "0.5"  # V
    current = "0.1n"  # A
    tip_bias = nanonis.convert(voltage)
    tip_current = nanonis.convert(current)

    zoom_out_scale = nanonis.convert(nanonis.scan_zoom_in_list[0])
    zoom_out_scale_nano = zoom_out_scale * 10**9

    temp_buffer = []
    mol_pos_nano_list = []

    state_buffer_list = []
    action_buffer_list = []
    next_state_buffer_list = []
    reward_buffer_list = []
    done_buffer_list = []
    info_buffer_list = []

    buffer_list = []

    # if nanonis.mol_tip_induce_path is not exist, create it
    if not os.path.exists(nanonis.mol_tip_induce_path):
        os.makedirs(nanonis.mol_tip_induce_path)
    # SAC_buffer
    if not os.path.exists(nanonis.SAC_buffer_path + "/buffer"):
        os.makedirs(nanonis.SAC_buffer_path + "/buffer")
    if not os.path.exists(nanonis.SAC_buffer_path + "/aug_buffer"):
        os.makedirs(nanonis.SAC_buffer_path + "/aug_buffer")
    if not os.path.exists(nanonis.SAC_aug_buffer_path + "/buffer"):
        os.makedirs(nanonis.SAC_aug_buffer_path + "/buffer")

    while tip_in_boundary(
        nanonis.inter_closest, nanonis.plane_size, nanonis.real_scan_factor
    ):

        nanonis.move_to_next_point()  # move the scan area to the next point

        # nanonis.AdjustTip_flag = nanonis.AdjustTipToPiezoCenter()                       # check & adjust the tip to the center of the piezo

        # nanonis.line_scan_thread_activate()  # activate the line scan, producer-consumer architecture, pre-check the tip and sample

        nanonis.batch_scan_producer(
            nanonis.nanocoodinate,
            nanonis.scan_zoom_in_list[0],
            nanonis.scan_square_Buffer_pix,
            0,
        )  # Scan the area
        if code_simulation_mode:
            #########################################################################
            nanonis.image_for = cv2.imread(
                img_simu_30_path, cv2.IMREAD_GRAYSCALE
            )  # read the simu image
            nanonis.image_for = cv2.resize(
                nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA
            )
        #########################################################################

        scan_qulity = nanonis.image_recognition()  # assement the scan qulity

        # nanonis.image_segmention(nanonis.image_for)                                   # segment the image ready to tip shaper
        #########################################################################
        scan_qulity == 1
        nanonis.skip_flag = 0
        #########################################################################
        # scan_qulity = 0

        if scan_qulity == 0:
            pass

        elif scan_qulity == 1:
            # TODO: 1.molecular detection
            #       2.regestration of all molecular position, find the candidate
            #       3.move the tip to the candidate,
            empty_count = 0
            if lattice_shape == "triangle":
                shape_key_points_result, angle = nanonis.molecular_tri_seeker(
                    nanonis.image_for,
                    n=tri_size,
                    scan_posion=nanonis.nanocoodinate,
                    scan_edge=zoom_out_scale_nano,
                )
            if lattice_shape == "all":
                shape_key_points_result, angle = nanonis.molecular_seeker(
                    nanonis.image_for,
                    scan_posion=nanonis.nanocoodinate,
                    scan_edge=zoom_out_scale_nano,
                )

            if shape_key_points_result == None:  #  No molecule
                print("No molecule detected in the image.")
                continue

            for mol_count, key_points in enumerate(shape_key_points_result):
                # How to select the next molecule
                molecule, molecular_index = nanonis.molecular_tracker(
                    tracker_position=nanonis.nanocoodinate,
                    tracker_scale=nanonis.scan_zoom_in_list[-1],
                    model="top",
                    move_tip=False,
                )  # find the molecule in the image

                mol_pos_nano_list.append(molecule.position)
                if empty_count >= 3:
                    print("No molecule detected 3 times, scan a new area.")
                    break
                # if there are moleculars that need to be manipulated
                if molecule:  # molecule is not None
                    state = None  # init the state

                    # reaction_path_flag = 1

                    # genarate the reaction_path_flag 0 or 1 randomly
                    # if random.random() > 0.5:
                    #     reaction_path_flag = 1

                    init_scan_flag = 0
                    # while np.array_equal(molecule.site_states, np.array([1])): # 分子没反应完并且没坏掉
                    zoom_in_scale = nanonis.convert(nanonis.scan_zoom_in_list[-1])
                    zoom_in_scale_nano = (
                        zoom_in_scale * 10**9
                    )  #  zoom_in_scale_nano = 7

                    # if init_scan_flag != 0:
                    #     # move the tip to the molecule position.
                    #     nanonis.FolMeSpeedSet(nanonis.zoom_in_tip_speed, 1)
                    #     tracker_scale = nanonis.convert(nanonis.scan_zoom_in_list[-1])
                    #     nanonis.TipXYSet(nanonis.nanocoodinate[0], nanonis.nanocoodinate[1]+0.5*tracker_scale)
                    #     nanonis.FolMeSpeedSet(nanonis.zoom_in_tip_speed, 0)
                    #     time.sleep(0.5)

                    #     nanonis.batch_scan_producer(nanonis.nanocoodinate, zoom_in_scale, nanonis.scan_square_Buffer_pix, 0)
                    # init_scan_flag +=1

                    image_save_time = time.strftime(
                        "%Y%m%d_%H%M%S", time.localtime(time.time())
                    )  # save the image with time
                    mol_old_position = molecule.position

                    if code_simulation_mode:
                        #########################################################################
                        nanonis.image_for = cv2.imread(
                            img_simu_7_path, cv2.IMREAD_GRAYSCALE
                        )  # read the simu image
                        nanonis.image_for = cv2.resize(
                            nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA
                        )
                    #########################################################################

                    molecule_nanocood = (
                        molecule.position[0] * 10**9,
                        molecule.position[1] * 10**9,
                    )
                    gamma_corrected = exposure.adjust_gamma(
                        nanonis.image_for, gamma=1.8
                    )

                    # only one result in the key_points_result
                    # key_points_result = nanonis.single_molecular_seeker(nanonis.image_for, scan_posion = nanonis.nanocoodinate, scan_edge = zoom_in_scale_nano, one_molecular_pos =(molecule.position[0],molecule.position[1]))
                    key_points_result = [key_points]
                    molecule = nanonis.molecule_registry.molecules[molecular_index]

                    no_mol_binary_arr = None
                    if key_points_result == None:  #  No molecule
                        print("No molecule detected in the image.")
                        cv2.imwrite(
                            nanonis.mol_tip_induce_path
                            + "/image_for"
                            + image_save_time
                            + ".png",
                            nanonis.image_for,
                        )
                        # no_mol_binary_arr = np.array([3])   # 3 means bad molecule
                        nanonis.molecule_registry.update_molecule(
                            molecular_index, site_states=no_mol_binary_arr, status=2
                        )  # mol was broken
                        empty_count += 1  # did not find the molecule if the empty_count >= 3, scan a new area
                        binary_arr = no_mol_binary_arr

                        # break
                    else:
                        empty_count = 0

                        # H_site_state = np.array([int(key_points_result[0][0]+1)])

                        # mols_center

                        # binary_arr = H_site_state        # 1 is H, 2 no H, 3 is bad

                    # print('the state of the H sites:', binary_arr) # 0000 1111 2222

                    # save the new position, new angle, to the registry
                    nanonis.molecule_registry.update_molecule(
                        molecular_index,
                        position=molecule.position,
                        site_states=binary_arr,
                        orientation=angle,
                    )
                    molecule = nanonis.molecule_registry.molecules[molecular_index]
                    # mol_pos_nano_list.append(molecule.position)

                    # # TODO: calculate 4 Br atoms position.
                    # Br_pos_nano = cal_Br_pos(molecule.position, nanonis.mol_scale, angle)
                    # nanonis.molecule_registry.update_molecule(molecular_index, Br_postion = Br_pos_nano)
                    # molecule = nanonis.molecule_registry.molecules[molecular_index]

                    # draw the XYedge frame in scan_for
                    image_for_rgb = cv2.cvtColor(nanonis.image_for, cv2.COLOR_GRAY2BGR)
                    square_side = (
                        nanonis.SAC_XYaction_edge
                        / zoom_in_scale
                        * nanonis.scan_square_Buffer_pix
                    )
                    center = (
                        int(key_points_result[0][5] * nanonis.scan_square_Buffer_pix),
                        int(key_points_result[0][6] * nanonis.scan_square_Buffer_pix),
                    )
                    rot_rect = (center, (square_side, square_side), angle)
                    box = cv2.boxPoints(rot_rect)
                    box = np.int0(box)
                    cv2.drawContours(image_for_rgb, [box], 0, (0, 255, 0), 1)

                    # corner1: x最小（最靠近y轴）, corner2: y最大（最远离x轴）
                    # corner3: x最大（最远离y轴）, corner4: y最小（最靠近x轴）
                    # corner1_idx = np.argmin(box[:, 0])
                    # corner2_idx = np.argmax(box[:, 1])
                    # corner3_idx = np.argmax(box[:, 0])
                    # corner4_idx = np.argmin(box[:, 1])

                    # cv2.putText(image_for_rgb, "1", (box[corner1_idx, 0], box[corner1_idx, 1]),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # cv2.putText(image_for_rgb, "2", (box[corner2_idx, 0], box[corner2_idx, 1]),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # cv2.putText(image_for_rgb, "3", (box[corner3_idx, 0], box[corner3_idx, 1]),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # cv2.putText(image_for_rgb, "4", (box[corner4_idx, 0], box[corner4_idx, 1]),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # show the image
                    # cv2.imshow('image', image_for_rgb)
                    # cv2.waitKey(0)

                    #########################################################################
                    # Fixed mode
                    #########################################################################
                    if tip_agent_mode == "fixed":
                        print("Please click the tip position on the image.")
                        print(
                            "left click: move the tip to the click position, \nright click: quit the tip induce.\nmiddle click: change the Bias and Setpoint.\nspace: change the Br sites state."
                        )
                        (
                            result,
                            image_for_rgb,
                            (voltage_click, current_click),
                            br_sites_array,
                        ) = mouse_click_tip(
                            image_for_rgb, sigle_state=True
                        )  # result is the pos of the tip， mouse_info = [x, y, click_flag]
                        if (
                            br_sites_array is not None
                        ):  # force the Br sites to be the new state
                            nanonis.molecule_registry.update_molecule(
                                molecular_index, site_states=br_sites_array
                            )
                            molecule = nanonis.molecule_registry.molecules[
                                molecular_index
                            ]
                            binary_arr = br_sites_array
                            print("the state of the Br sites:", br_sites_array)

                    #########################################################################
                    # Human mode
                    #########################################################################
                    if tip_agent_mode == "human":
                        print("Please click the tip position on the image.")
                        print(
                            "left click: move the tip to the click position, \nright click: quit the tip induce.\nmiddle click: change the Bias and Setpoint.\nspace: change the Br sites state."
                        )
                        (
                            result,
                            image_for_rgb,
                            (voltage_click, current_click),
                            br_sites_array,
                        ) = mouse_click_tip(
                            image_for_rgb, mode="sigle"
                        )  # result is the pos of the tip， mouse_info = [x, y, click_flag]
                        if (
                            br_sites_array is not None
                        ):  # force the Br sites to be the new state
                            nanonis.molecule_registry.update_molecule(
                                molecular_index, site_states=br_sites_array
                            )
                            molecule = nanonis.molecule_registry.molecules[
                                molecular_index
                            ]
                            binary_arr = br_sites_array
                            print("the state of the Br sites:", br_sites_array)
                    #########################################################################
                    # SAC mode
                    #########################################################################
                    elif tip_agent_mode == "SAC":
                        # result_human, image_for_rgb, (voltage_click, current_click), br_sites_array= mouse_click_tip(image_for_rgb, mode ='sigle')
                        result_human = None
                        voltage_click = None
                        current_click = None
                        br_sites_array = None

                        if (
                            br_sites_array is not None
                        ):  # force the Br sites to be the new state
                            nanonis.molecule_registry.update_molecule(
                                molecular_index, site_states=br_sites_array
                            )
                            molecule = nanonis.molecule_registry.molecules[
                                molecular_index
                            ]
                            binary_arr = br_sites_array
                            print("the state of the Br sites:", br_sites_array)

                        state4SAC = np.array(
                            [binary_arr[1]]
                        )  # put the first element as state to SAC

                        nanonis.auto2SAC_queue.put(
                            state4SAC
                        )  # send the molecule state to the SAC agent
                        print("SAC agent is thinking...")
                        action_from_SAC = (
                            nanonis.SAC2auto_queue.get()
                        )  # get the action from the SAC agent
                        # save the action to the npy
                        save_action_to_npy(
                            action_from_SAC,
                            file_path=nanonis.SAC_origin_action_path + "/actions.npy",
                        )

                        print(
                            f"Tip position:  X:{action_from_SAC[0]}, Y:{action_from_SAC[1]}"
                        )
                        print(
                            f"Tip Bias: {action_from_SAC[2]}V, Tip Current: {action_from_SAC[3]}nA"
                        )

                        # action_from_SAC = np.array([new_x, new_y, V, I])

                        if action_from_SAC is not None:
                            origin_action = action_from_SAC
                            action_from_SAC = action_from_SAC.tolist()

                            origin_action_X = action_from_SAC[0]
                            origin_action_Y = action_from_SAC[1]
                            origin_action_V = action_from_SAC[2]
                            origin_action_I = action_from_SAC[3]

                            dx_rot = action_from_SAC[0]
                            dy_rot = action_from_SAC[1]

                            # grid the X Y position
                            (dx_rot, dy_rot) = find_nearest_grid_point(
                                nanonis.env.H_xy_grid_points, (dx_rot, dy_rot)
                            )

                            # if click the middle button, change the Bias and Setpoint
                            if voltage_click or current_click:
                                tip_bias = nanonis.convert(str(voltage_click))
                                tip_current = nanonis.convert(str(current_click))
                            else:
                                # (action_from_SAC[2],action_from_SAC[3]) =  find_nearest_grid_point(nanonis.env.vi_grid_points, (action_from_SAC[2],action_from_SAC[3]))

                                tip_bias = action_from_SAC[2]
                                tip_current = action_from_SAC[3] * 10**-9

                            # result for nanonis coordinate convertion
                            result = reverse_point_in_rotated_rect_polar(
                                dx_rot,
                                dy_rot,
                                (
                                    nanonis.scan_square_Buffer_pix,
                                    nanonis.scan_square_Buffer_pix,
                                ),
                                center,
                                angle,
                                nanonis.SAC_XYaction_100,
                                nanonis.zoom_in_100,
                                nanonis.scan_square_Buffer_pix,
                            )

                        if result_human is not None:
                            result = result_human

                        if no_mol_binary_arr is not None:
                            result = None
                            no_mol_binary_arr = None  # init the no_mol_binary_arr

                        if (
                            2 in binary_arr or 3 in binary_arr
                        ):  # after the molecule is fully reacted or bad, find the next molecule
                            result = None
                            # temp_buffer = [] # init the temp_buffer
                            # break
                    #########################################################################

                    log_path = (
                        nanonis.mol_tip_induce_path
                        + "/image_for"
                        + image_save_time
                        + ".log"
                    )
                    # r_frac, theta_deg, dx_rot, dy_rot = point_in_rotated_rect_polar(result, (nanonis.scan_square_Buffer_pix,nanonis.scan_square_Buffer_pix), center, angle, nanonis.SAC_XYaction_100)
                    if voltage_click or current_click:
                        tip_bias = nanonis.convert(str(voltage_click))
                        tip_current = nanonis.convert(str(current_click))

                    # click the left mouse button
                    if result:  # molcule do not have to tip manipulation anymore
                        r_frac, theta_deg, dx_rot, dy_rot = point_in_rotated_rect_polar(
                            result,
                            (
                                nanonis.scan_square_Buffer_pix,
                                nanonis.scan_square_Buffer_pix,
                            ),
                            center,
                            angle,
                            nanonis.SAC_XYaction_100,
                        )
                        dx_rot = (
                            dx_rot
                            * nanonis.zoom_in_100
                            / nanonis.scan_square_Buffer_pix
                        )
                        dy_rot = (
                            dy_rot
                            * nanonis.zoom_in_100
                            / nanonis.scan_square_Buffer_pix
                        )

                        # exchange the dx_rot and dy_rot
                        # dx_rot, dy_rot = dy_rot, dx_rot

                        nanonis.molecule_registry.update_molecule(
                            molecular_index, operated=1, operated_time=time.time()
                        )
                        molecule = nanonis.molecule_registry.molecules[molecular_index]

                        mouse_click_pos_nano = matrix_to_cartesian(
                            result[0],
                            result[1],
                            center=nanonis.nanocoodinate,
                            side_length=zoom_in_scale,
                        )
                        # mouse_click_pos_nano = matrix_to_cartesian(result[1], result[0],center = mol_old_position ,side_length=zoom_in_scale)

                        # move the tip to the click position
                        print("Move the tip to the click position.")
                        nanonis.FolMeSpeedSet(nanonis.tip_manipulate_speed, 1)
                        nanonis.TipXYSet(
                            mouse_click_pos_nano[0], mouse_click_pos_nano[1]
                        )
                        # time.sleep(0.5)
                        nanonis.FolMeSpeedSet(nanonis.tip_manipulate_speed, 0)
                        time.sleep(1)

                        ##############################  Tip manipulation  ########################################
                        print("Start the tip manipulation...")
                        # print(f'Tip position:  X:{action_from_SAC[0]}, Y:{action_from_SAC[1]}')
                        # print(f'Tip Bias: {action_from_SAC[2]}V, Tip Current: {action_from_SAC[3]}nA')
                        # do the tip manipulation
                        lift_time = 1  # s
                        max_hold_time = 20  # s
                        pulse_time = 0.1  # s
                        # tip_current would highly impact the stop current threshold
                        # stop_current_threshold = 0.3  # nA
                        stop_current_threshold = action_from_SAC[3] / 2

                        tip_bias_init = nanonis.BiasGet()
                        tip_current_init = nanonis.SetpointGet()

                        abs_tip_bias_init = abs(tip_bias_init)
                        abs_tip_current_init = abs(tip_current_init)

                        # nanonis.BiasSet(abs_tip_bias_init)
                        # nanonis.SetpointSet(abs_tip_current_init)

                        if tip_induce_mode == "CC":
                            nanonis.BiasSet(abs_tip_bias_init)
                            nanonis.SetpointSet(abs_tip_current_init)
                            # set the tip bias and current should be changed in 4s
                            steps = 50  # 分成50步
                            time_interval = lift_time / steps  # 每步的时间间隔

                            # 计算每步的增量
                            bias_step = (tip_bias - abs_tip_bias_init) / steps
                            current_step = (tip_current - abs_tip_current_init) / steps

                            # nanonis.ZCtrlOff()

                            for i in range(steps):
                                # 逐步设置Bias和Setpoint
                                nanonis.BiasSet(abs_tip_bias_init + bias_step * (i + 1))
                                nanonis.SetpointSet(
                                    abs_tip_current_init + current_step * (i + 1)
                                )
                                time.sleep(time_interval)

                            nanonis.ZCtrlOff()

                            # time.sleep(12)   # wait for the tip induce

                            start_time = time.time()
                            signal_history = deque(maxlen=10)
                            while True:
                                # 保留两位小数
                                signal_current = nanonis.SignalValsGet(0)["0"] * 1e9
                                signal_current = round(signal_current, 4)
                                signal_history.append(
                                    signal_current
                                )  # 将当前值添加到deque中
                                # print(signal_current)

                                # 检查是否已经超过 15 秒
                                if time.time() - start_time > max_hold_time:
                                    print(
                                        f"The loop has been running for {max_hold_time} seconds, exiting the tip induce."
                                    )
                                    break
                                # if signal_current < stop_current_threshold:
                                #     print(f"The loop has been running for {time.time() - start_time} seconds, exiting the tip induce.")
                                #     break
                                # 如果deque中的所有值都小于stop_current_threshold，则跳出循环
                                if len(signal_history) == signal_history.maxlen and all(
                                    value < stop_current_threshold
                                    for value in signal_history
                                ):
                                    print(
                                        f"The loop has been running for {time.time() - start_time} seconds, exiting the tip induce."
                                    )
                                    break

                            # initialize the tip bias and current
                            # nanonis.ZCtrlOff()
                            nanonis.BiasSet(tip_bias_init)
                            nanonis.SetpointSet(tip_current_init)
                            nanonis.ZCtrlOnSet()

                        elif tip_induce_mode == "CH":
                            nanonis.BiasSet(abs_tip_bias_init)
                            nanonis.SetpointSet(abs_tip_current_init)
                            steps = 50  # 分成50步
                            current_time_interval = 1 / steps  # 1s 每步的时间间隔
                            voltage_time_interval = 4 / steps  # 4s 每步的时间间隔
                            # 计算每步的增量
                            current_step = (tip_current - abs_tip_current_init) / steps
                            bias_step = (tip_bias - abs_tip_bias_init) / steps
                            for i in range(steps):
                                # 逐步设置Setpoint
                                nanonis.SetpointSet(
                                    abs_tip_current_init + current_step * (i + 1)
                                )
                                time.sleep(current_time_interval)
                            # off the Z control after the tip current is set
                            time.sleep(1)  # wait for the tip stabilize

                            nanonis.ZCtrlOff()
                            for i in range(steps):
                                # 逐步设置Bias
                                nanonis.BiasSet(abs_tip_bias_init + bias_step * (i + 1))
                                time.sleep(voltage_time_interval)

                            time.sleep(8)  # wait for the tip induce

                            # initialize the tip bias and current
                            nanonis.BiasSet(tip_bias_init)
                            nanonis.SetpointSet(tip_current_init)
                            nanonis.ZCtrlOnSet()

                        elif tip_induce_mode == "pulse":

                            # set the tip bias and current should be changed in 4s
                            steps = 20  # 分成50步
                            time_interval = lift_time / steps  # 每步的时间间隔

                            # 计算每步的增量
                            current_step = (tip_current - abs_tip_current_init) / steps

                            # nanonis.ZCtrlOff()

                            for i in range(steps):
                                # 逐步设置Setpoint
                                nanonis.SetpointSet(
                                    abs_tip_current_init + current_step * (i + 1)
                                )
                                time.sleep(time_interval)

                            # nanonis.ZCtrlOff()

                            time.sleep(0.3)  # wait for the tip induce
                            nanonis.BiasPulse(tip_bias, pulse_time)
                            time.sleep(0.3)  # wait for the tip induce
                            # nanonis.BiasSet(tip_bias_init)
                            nanonis.SetpointSet(tip_current_init)
                            # nanonis.ZCtrlOnSet()

                        time.sleep(1)
                        print("Tip manipulation is done.")
                        ##############################  Tip manipulation  ########################################
                        #
                        #
                        #
                        ##############################  save the log and experience  ########################################
                        # save the nanonis.image_for and image_for_rgb in nanonis.mol_tip_induce_path

                        cv2.imwrite(
                            nanonis.mol_tip_induce_path
                            + "/image_for"
                            + image_save_time
                            + ".png",
                            nanonis.image_for,
                        )
                        cv2.imwrite(
                            nanonis.mol_tip_induce_path
                            + "/image_for_rgb"
                            + image_save_time
                            + ""
                            + "_"
                            + voltage
                            + "V_"
                            + current
                            + "A_"
                            + "tip_cood"
                            + str((dx_rot, dy_rot))
                            + ".png",
                            image_for_rgb,
                        )

                        with open(log_path, "w", encoding="utf-8") as f:  # save the log
                            f.write(f"time: {image_save_time}\n")
                            f.write(f"molecule_position: {molecule.position}\n")
                            f.write(f"angle: {angle}\n")
                            f.write(f"binary_arr: {binary_arr}\n")
                            f.write(f"dx_rot: {dx_rot}, dy_rot: {dy_rot}\n")
                            f.write(
                                f"tip_bias: {tip_bias}, tip_current: {tip_current}\n"
                            )

                        state_buffer_list.append(np.array(binary_arr))
                        action_buffer_list.append(
                            np.array(
                                [
                                    origin_action_X,
                                    origin_action_Y,
                                    tip_bias,
                                    tip_current * 10**9,
                                ]
                            )
                        )

                        if tip_agent_mode == "SAC":
                            # send next_state, reward, done, info to the SAC agent
                            # nanonis.auto2SAC_queue.put([next_state, reward, done, info])
                            if (
                                mol_count != len(shape_key_points_result) - 1
                            ):  # if not the last molecule, wait for the next molecule
                                nanonis.q_save_exp.put(
                                    "wait"
                                )  # tell the SAC that the exp is not valid,don't block the SAC agent

                            # if tip_agent_mode == 'fixed':

                            # if len(temp_buffer)==0:    # temp_buffer is empty
                            #     state = np.array(binary_arr)
                            #     # state = np.append(state, reaction_path_flag)
                            #     action = np.array([dx_rot, dy_rot, tip_bias, tip_current*10**9])
                            #     # action = origin_action
                            #     temp_buffer.append(state)
                            #     temp_buffer.append(action)
                            #     if tip_agent_mode == 'SAC':
                            #         # send next_state, reward, done, info to the SAC agent
                            #         # nanonis.auto2SAC_queue.put([next_state, reward, done, info])
                            #         nanonis.q_save_exp.put('wait')  # tell the SAC that the exp is not valid,don't block the SAC agent
                            # else:
                            #     next_state = np.array(binary_arr)
                            #     # next_state = np.append(next_state, reaction_path_flag)

                            #     # state_legal, trans_type, change_path_flag = nanonis.buffer4log.legalize_state(temp_buffer[0])
                            #     state_legal = temp_buffer[0]
                            #     state_legal = np.array([state_legal[0]])
                            #     action_for_buffer = np.array([origin_action_X, origin_action_Y, origin_action_V, origin_action_I])

                            #     # action_for_buffer = nanonis.buffer4log.transform_action(action_for_buffer, trans_type)      # float32
                            #     # action_corrected = nanonis.buffer4log.transform_action(action, trans_type)                  # int
                            #     action_corrected = action

                            #     # next_state_transform = nanonis.buffer4log.transform_state(next_state, trans_type)
                            #     next_state_transform = next_state

                            #     # reward,info = nanonis.env.reward_culculate(temp_buffer[0],next_state,0)
                            #     state_legal = np.array([state_legal[0]])

                            #     reward,info = nanonis.env.H_reward_culculate(state_legal,next_state_transform,0)
                            #     done = 0
                            #     if 2 in binary_arr or 3 in binary_arr or info["reaction"] in ["bad", "wrong","skip"]:
                            #         done = 1
                            #     temp_buffer.append(reward)
                            #     temp_buffer.append(next_state)

                            #     temp_buffer.append(done)
                            #     temp_buffer.append(info)
                            #     #  state, action, reward, next_state, done, info
                            #     nanonis.buffer4log.add(temp_buffer[0],temp_buffer[1],temp_buffer[2],temp_buffer[3],temp_buffer[4],temp_buffer[5])

                            #     legal_exp = [state_legal,action_for_buffer,temp_buffer[2],next_state_transform,temp_buffer[4],temp_buffer[5]]
                            #     if tip_agent_mode == 'SAC':
                            #         # send next_state, reward, done, info to the SAC agent
                            #         # nanonis.auto2SAC_queue.put([next_state, reward, done, info])
                            #         nanonis.q_save_exp.put(legal_exp)
                            #         if action_visualize:

                            #             if action_visualize_Queue.full():  # 如果队列满了，则等待一段时间
                            #                 action_visualize_Queue.get()

                            #             action_visualize_Queue.put((temp_buffer[0],temp_buffer[1],temp_buffer[2],temp_buffer[3],temp_buffer[4],temp_buffer[5]))

                            #     #  state_legal, action_corrected, reward, next_state_transform, done, info
                            #     # nanonis.buffer4aug.aug_exp(state_legal,action_for_buffer,temp_buffer[2],next_state_transform,temp_buffer[4],temp_buffer[5])
                            #     nanonis.buffer4log.save(nanonis.SAC_buffer_path)
                            #     # nanonis.buffer4log.plot_train(agent.hyperparameters, save_path = nanonis.SAC_aug_origin_action_vi_path)

                            #     # nanonis.buffer4aug.save(nanonis.SAC_aug_buffer_path)

                            #     temp_buffer = [] # init the temp_buffer
                            #     state = np.array(binary_arr)
                            #     # state = np.append(state, reaction_path_flag)
                            #     action = np.array([dx_rot, dy_rot, tip_bias, tip_current*10**9])  # x y v i
                            #     # action = origin_action
                            #     temp_buffer.append(state)
                            #     temp_buffer.append(action)

                            nanonis.save_checkpoint()
                            # break

                            # if 2 in binary_arr or 3 in binary_arr:  # after the molecule is fully reacted or bad, find the next molecule
                            #     temp_buffer = [] # init the temp_buffer
                            #     break

                    # click the right mouse button
                    else:
                        print("Quit the tip induce, the molecule is not operated.")
                        state_buffer_list.append(np.array(binary_arr))
                        action_buffer_list.append(np.array([0, 0, 0, 0]))

                        # if len(temp_buffer)==0: # the mol is no need to be manipulated in the first step
                        #     binary_arr = np.array([3])
                        #     nanonis.molecule_registry.update_molecule(molecular_index, site_states = binary_arr, status=2)
                        #     molecule = nanonis.molecule_registry.molecules[molecular_index]
                        if tip_agent_mode == "SAC":
                            # send next_state, reward, done, info to the SAC agent
                            # nanonis.auto2SAC_queue.put([next_state, reward, done, info])
                            nanonis.q_save_exp.put(
                                "wait"
                            )  # tell the SAC that the exp is not valid,don't block the SAC agent
                        # break

                        # if br_sites_array is not None: # force the Br sites to be the new state that the user input
                        #     binary_arr = np.array([br_sites_array[0]])
                        # # else:                          # force the Br sites to be the bad state
                        # #     binary_arr = np.array([3])
                        # nanonis.molecule_registry.update_molecule(molecular_index, site_states = binary_arr, status=2)
                        # molecule = nanonis.molecule_registry.molecules[molecular_index]

                        # cv2.imwrite(nanonis.mol_tip_induce_path + '/image_for'+ image_save_time +'.png', nanonis.image_for)
                        # with open(log_path, 'w', encoding='utf-8') as f:                                # save the log
                        #     f.write(f"time: {image_save_time}\n")
                        #     f.write(f"molecule_position: {molecule.position}\n")
                        #     f.write(f"angle: {angle}\n")
                        #     f.write(f"binary_arr: {binary_arr}\n")
                        #     f.write(f"dx_rot: , dy_rot: \n")
                        #     f.write(f"tip_bias: , tip_current: \n")

                        # next_state = np.array(binary_arr)
                        # # next_state = np.append(next_state, reaction_path_flag)

                        # state_legal = temp_buffer[0]

                        # action_for_buffer = np.array([origin_action_X, origin_action_Y, origin_action_V, origin_action_I])

                        # action_for_buffer = action_for_buffer # float32

                        # action_corrected = action            # int
                        # next_state_transform = next_state

                        # reward,info = nanonis.env.H_reward_culculate(state_legal,next_state_transform,0)
                        # done = 0
                        # if 2 in binary_arr or 3 in binary_arr or info["reaction"] in ["bad", "wrong","skip"]:
                        #     done = 1
                        # temp_buffer.append(reward)
                        # temp_buffer.append(next_state)
                        # temp_buffer.append(done)
                        # temp_buffer.append(info)
                        # #  state, action, reward, next_state, done, info
                        # nanonis.buffer4log.add(temp_buffer[0],temp_buffer[1],temp_buffer[2],temp_buffer[3],temp_buffer[4],temp_buffer[5])

                        # legal_exp = [state_legal,action_for_buffer,temp_buffer[2],next_state_transform,temp_buffer[4],temp_buffer[5]]
                        # if tip_agent_mode == 'SAC':
                        #     # send next_state, reward, done, info to the SAC agent
                        #     # nanonis.auto2SAC_queue.put([next_state, reward, done, info])
                        #     nanonis.q_save_exp.put(legal_exp)
                        #     if action_visualize:

                        #         if action_visualize_Queue.full():  # 如果队列满了，则等待一段时间
                        #             action_visualize_Queue.get()

                        #         action_visualize_Queue.put((temp_buffer[0],temp_buffer[1],temp_buffer[2],temp_buffer[3],temp_buffer[4],temp_buffer[5]))

                        # #  state_legal, action_corrected, reward, next_state_transform, done, info
                        # # nanonis.buffer4aug.aug_exp(state_legal,action_for_buffer,temp_buffer[2],next_state_transform,temp_buffer[4],temp_buffer[5])
                        # nanonis.buffer4log.save(nanonis.SAC_buffer_path)
                        # # nanonis.buffer4log.plot_train(agent.hyperparameters, save_path = nanonis.SAC_aug_origin_action_vi_path)

                        # # nanonis.buffer4aug.save(nanonis.SAC_aug_buffer_path)

                        # temp_buffer = []

                        nanonis.save_checkpoint()

                        # break # jump out of the loop, scan the next molecule
                    # break
            ############################################################################################################
            # else:  # there is no molecule in this area

            # check the result
            # move the tip to the molecule position.
            nanonis.FolMeSpeedSet(nanonis.zoom_in_tip_speed, 1)
            tracker_scale = nanonis.convert(nanonis.scan_zoom_in_list[-1])
            nanonis.TipXYSet(
                nanonis.nanocoodinate[0], nanonis.nanocoodinate[1] + 0.5 * tracker_scale
            )
            nanonis.FolMeSpeedSet(nanonis.zoom_in_tip_speed, 0)
            time.sleep(0.5)

            nanonis.batch_scan_producer(
                nanonis.nanocoodinate, zoom_in_scale, nanonis.scan_square_Buffer_pix, 0
            )

            if code_simulation_mode:
                #########################################################################
                nanonis.image_for = cv2.imread(
                    img_simu_30_path, cv2.IMREAD_GRAYSCALE
                )  # read the simu image
                nanonis.image_for = cv2.resize(
                    nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA
                )
            #########################################################################

            key_points_result = nanonis.batch_molecular_seeker(
                nanonis.image_for,
                scan_posion=nanonis.nanocoodinate,
                scan_edge=zoom_in_scale_nano,
                one_molecular_pos_list=mol_pos_nano_list,
            )

            # if there is no mol, the tip must be bad so much
            if key_points_result is None:
                # init all list for exp
                state_buffer_list = []
                action_buffer_list = []
                reward_buffer_list = []
                next_state_buffer_list = []
                done_buffer_list = []
                info_buffer_list = []
                buffer_list = []
                nanonis.save_checkpoint()
                # continue

            # draw the XYedge frame in scan_for
            image_for_rgb = cv2.cvtColor(nanonis.image_for, cv2.COLOR_GRAY2BGR)
            if key_points_result:
                for box_index, key_points in enumerate(key_points_result):
                    if key_points:
                        # image_for_rgb =  cv2.cvtColor(nanonis.image_for, cv2.COLOR_GRAY2BGR)
                        square_side = (
                            nanonis.SAC_XYaction_edge
                            / zoom_in_scale
                            * nanonis.scan_square_Buffer_pix
                        )
                        center = (
                            int(
                                key_points_result[box_index][5]
                                * nanonis.scan_square_Buffer_pix
                            ),
                            int(
                                key_points_result[box_index][6]
                                * nanonis.scan_square_Buffer_pix
                            ),
                        )
                        rot_rect = (center, (square_side, square_side), angle)
                        box = cv2.boxPoints(rot_rect)
                        box = np.int0(box)
                        cv2.drawContours(image_for_rgb, [box], 0, (0, 255, 0), 1)

                        # 计算 box 的中心点
                        box_center_x = int((box[0][0] + box[2][0]) / 2)
                        box_center_y = int((box[0][1] + box[2][1]) / 2)

                        # 在 box 的中心绘制 box_index
                        cv2.putText(
                            image_for_rgb,
                            str(box_index),
                            (box_center_x, box_center_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    # if key_points is None:

                    if key_points_result[box_index]:  # find the molecule
                        H_site_state = np.array(
                            [int(key_points_result[box_index][0] + 1)]
                        )
                    else:  # not find the molecule
                        H_site_state = np.array([3])  # after state

                    next_state_buffer_list.append(H_site_state)

            formatted_output = " ".join(
                str(arr[0]) for arr in next_state_buffer_list
            )  ## format the output
            print("all mol state is:", formatted_output)

            result, image_for_rgb, (voltage_click, current_click), br_sites_array = (
                mouse_click_tip(
                    image_for_rgb, mode="batch", state_num=len(next_state_buffer_list)
                )
            )
            if br_sites_array is not None:  # force the Br sites to be the new state
                next_state_buffer_list = br_sites_array

            cv2.imwrite(
                nanonis.mol_tip_induce_path + "/image_for" + image_save_time + ".png",
                image_for_rgb,
            )
            # for i, mol_pos_nano in enumerate(mol_pos_nano_list): #check the mol one by one

            # # if key_points_result[i]:  # find the molecule
            # #     H_site_state = np.array([int(key_points_result[i][0]+1)])
            # # else: # not find the molecule
            # #     H_site_state = np.array([2])   # after state

            # # next_state_buffer_list.append(H_site_state)
            # # culculate the reward
            # reward,info = nanonis.env.H_reward_culculate(state_buffer_list[i],next_state_buffer_list[i],0)
            # # append the reward to the buffer
            # reward_buffer_list.append(reward)
            # # append the info to the buffer
            # info_buffer_list.append(info)
            # # append the done to the buffer
            # if info["reaction"] in ["bad", "success"]:
            #     done_buffer_list.append(1)
            # else:
            #     done_buffer_list.append(0)

            # use all the list to create experience
            # for i in range(len(mol_pos_nano_list)):
            # buffer = [state_buffer_list[i],action_buffer_list[i],reward_buffer_list[i],next_state_buffer_list[i],done_buffer_list[i],info_buffer_list[i]]
            # nanonis.buffer4log.add(buffer[0],buffer[1],buffer[2],buffer[3],buffer[4],buffer[5])
            # buffer_list.append(buffer)
            # nanonis.buffer4log.save(nanonis.SAC_buffer_path)
            # nanonis.q_save_exp.put(buffer_list)

            # init all list for exp
            # state_buffer_list = []
            # action_buffer_list = []
            # reward_buffer_list = []
            # next_state_buffer_list = []
            # done_buffer_list = []
            # info_buffer_list = []
            # buffer_list = []
            mol_pos_nano_list = []

            nanonis.molecule_registry.clear_the_registry()

            # break   # Jump out of the loop, scan the next point

        scan_qulity == 1
        nanonis.skip_flag = 0
        nanonis.save_checkpoint()  # save the checkpoint
