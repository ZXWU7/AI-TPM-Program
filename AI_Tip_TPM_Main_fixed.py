from skimage import exposure, io
# from PIL import Image
from Auto_scan_class import *
from DQN.agent import *
from utils import *

if __name__ == '__main__':
    
    code_simulation_mode = False  # True: code simulation mode, False: real mode

    if code_simulation_mode:
    ############################################################################################################
        img_simu_30_path = './STM_img_simu/TPM_image/001.png'
        img_simu_7_path = './STM_img_simu/TPM_image/001.png'
    ############################################################################################################
    # who is the tip agent
    tip_agent_mode = 'fixed'    # 'SAC' or 'human' or 'fixed'

    tip_induce_mode = 'pulse'      # 'CC' or 'CH'  CC means Constant I   'CH' means Constant height  'pulse' means pulse mode

    lattice_shape = 'all'   # 'triangle' or 'all' or 'costom'  

    action_visualize = False

    tri_size = 5
    
    # nanonis.SAC_init(mode = 'new') # deflaut mode is 'latest' mode = 'new' : create a new model, mode = 'latest' : load the latest checkpoint
    nanonis = Mustard_AI_Nanonis()
    
    env = Env(polar_space=False)
    agent = SACAgent(env)

    nanonis.tip_init(mode = 'new') # deflaut mode is 'new' mode = 'new' : the tip is initialized to the center and create a new log folder, 
                                                            # mode = 'latest' : load the latest checkpoint
    
    # nanonis.DQN_init(mode = 'new') # deflaut mode is 'latest' mode = 'new' : create a new model, mode = 'latest' : load the latest checkpoint    

    nanonis.monitor_thread_activate()                                                # activate the monitor thread

    voltage = '-4.05'  # V
    current = '0.06n' # A

    ScanFrame = nanonis.ScanFrameGet()
    drift_center_x = ScanFrame['center_x']
    drift_center_y = ScanFrame['center_y']

    tip_bias = nanonis.convert(voltage)
    tip_current = nanonis.convert(current)
    x = 0.0
    y = 0.0
    action = np.array([x, y, tip_bias, tip_current])  # x y v i

    zoom_out_scale = nanonis.convert(nanonis.scan_zoom_in_list[0])
    zoom_out_scale_nano = zoom_out_scale*10**9
    



    mol_pos_nano_list = []

    # if nanonis.mol_tip_induce_path is not exist, create it
    if not os.path.exists(nanonis.mol_tip_induce_path):
        os.makedirs(nanonis.mol_tip_induce_path)
    # SAC_buffer
    if not os.path.exists(nanonis.SAC_buffer_path + '/buffer'):
        os.makedirs(nanonis.SAC_buffer_path + '/buffer')
    if not os.path.exists(nanonis.SAC_buffer_path + '/aug_buffer'):
        os.makedirs(nanonis.SAC_buffer_path + '/aug_buffer')
    if not os.path.exists(nanonis.SAC_aug_buffer_path + '/buffer'):
        os.makedirs(nanonis.SAC_aug_buffer_path + '/buffer')

    main_state_time = time.time()
    main_time = 14400 #14400s 

    while tip_in_boundary(nanonis.inter_closest, nanonis.plane_size, nanonis.real_scan_factor):
        
        nanonis.move_to_next_point(drift=[drift_center_x,drift_center_y])                                                    # move the scan area to the next point

        # nanonis.AdjustTip_flag = nanonis.AdjustTipToPiezoCenter()                       # check & adjust the tip to the center of the piezo

        nanonis.line_scan_thread_activate()                                             # activate the line scan, producer-consumer architecture, pre-check the tip and sample

        nanocoodinate = (nanonis.nanocoodinate[0] + drift_center_x, nanonis.nanocoodinate[1]+ drift_center_y)

        nanonis.batch_scan_producer(nanocoodinate, nanonis.scan_zoom_in_list[0], nanonis.scan_square_Buffer_pix, 0)    # Scan the area
        if code_simulation_mode:
        #########################################################################
            nanonis.image_for = cv2.imread(img_simu_30_path, cv2.IMREAD_GRAYSCALE)           # read the simu image
            nanonis.image_for = cv2.resize(nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA)
        #########################################################################

        scan_qulity = nanonis.image_recognition()                                       # assement the scan qulity 

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
            # if lattice_shape == 'triangle':
            gamma_corrected = exposure.adjust_gamma(nanonis.image_for, gamma=1.8)
            shape_key_points_result, angle = nanonis.molecular_tri_seeker(gamma_corrected,n=tri_size, scan_posion = nanocoodinate, scan_edge = zoom_out_scale_nano,shanp = lattice_shape )
            # if lattice_shape == 'all':
            #     shape_key_points_result = nanonis.molecular_seeker(nanonis.image_for, scan_posion = nanonis.nanocoodinate, scan_edge = zoom_out_scale_nano)
            #     angle = 0
            # if lattice_shape == 'costom':
            #     shape_key_points_result = nanonis.molecular_seeker(nanonis.image_for, scan_posion = nanonis.nanocoodinate, scan_edge = zoom_out_scale_nano)

            if shape_key_points_result == None:    #  No molecule
                print('No molecule detected in the image.')
                continue
            for mol_count,key_points in enumerate(shape_key_points_result):
                # How to select the next molecule
                molecule, molecular_index = nanonis.molecular_tracker(tracker_position = nanocoodinate, tracker_scale = nanonis.scan_zoom_in_list[-1], model= 'closest', move_tip= False) # find the molecule in the image
                
                
                # if there are moleculars that need to be manipulated
                if molecule:    # molecule is not None
                    mol_pos_nano_list.append(molecule.position)
                    state = None  # init the state 
                    zoom_in_scale = nanonis.convert(nanonis.scan_zoom_in_list[-1])
                    zoom_in_scale_nano = zoom_in_scale*10**9   #  zoom_in_scale_nano = 7

                    image_save_time = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))    # save the image with time
                    mol_old_position = molecule.position
                    
                    if code_simulation_mode:
                    #########################################################################
                        nanonis.image_for = cv2.imread(img_simu_7_path, cv2.IMREAD_GRAYSCALE)           # read the simu image
                        nanonis.image_for = cv2.resize(nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA)
                    #########################################################################

                    molecule_nanocood = (molecule.position[0]*10**9,molecule.position[1]*10**9)
                    # gamma_corrected = exposure.adjust_gamma(nanonis.image_for, gamma=1.8)

                    # only one result in the key_points_result
                    # key_points_result = nanonis.single_molecular_seeker(nanonis.image_for, scan_posion = nanonis.nanocoodinate, scan_edge = zoom_in_scale_nano, one_molecular_pos =(molecule.position[0],molecule.position[1]))
                    key_points_result = [key_points]
                    molecule = nanonis.molecule_registry.molecules[molecular_index]

                    no_mol_binary_arr = None
                    if key_points_result == None:    #  No molecule
                        print('No molecule detected in the image.')
                        cv2.imwrite(nanonis.mol_tip_induce_path + '/image_for'+ image_save_time +'.png', nanonis.image_for)
                    else:
                        empty_count = 0    

                    # print('the state of the H sites:', binary_arr) # 0000 1111 2222

                    # save the new position, new angle, to the registry
                    binary_arr = np.array([int(key_points_result[0][0]+1)])
                    nanonis.molecule_registry.update_molecule(molecular_index, position = molecule.position, site_states = binary_arr, orientation = angle)
                    molecule = nanonis.molecule_registry.molecules[molecular_index]
                    # mol_pos_nano_list.append(molecule.position)

                    # draw the XYedge frame in scan_for
                    image_for_rgb =  cv2.cvtColor(nanonis.image_for, cv2.COLOR_GRAY2BGR)
                    square_side = nanonis.SAC_XYaction_edge/zoom_in_scale * nanonis.scan_square_Buffer_pix
                    center = (int(key_points_result[0][5]*nanonis.scan_square_Buffer_pix), int(key_points_result[0][6]*nanonis.scan_square_Buffer_pix))
                    rot_rect = (center, (square_side, square_side), angle)
                    box = cv2.boxPoints(rot_rect)
                    box = np.int0(box)
                    cv2.drawContours(image_for_rgb, [box], 0, (0, 255, 0), 1)



                    #########################################################################
                    # Fixed mode
                    #########################################################################
                    if tip_agent_mode == 'fixed':
                        # result_human, image_for_rgb, (voltage_click, current_click), br_sites_array= mouse_click_tip(image_for_rgb, mode ='sigle')
                        result_human = None
                        voltage_click = None    
                        current_click = None
                        br_sites_array = None

                        if br_sites_array is not None:  # force the Br sites to be the new state
                            nanonis.molecule_registry.update_molecule(molecular_index,site_states = br_sites_array)
                            molecule = nanonis.molecule_registry.molecules[molecular_index]
                            binary_arr = br_sites_array
                            print('the state of the Br sites:', br_sites_array)

                        # state4SAC = np.array([binary_arr[1]]) # put the first element as state to SAC

                        
                        # nanonis.auto2SAC_queue.put(state4SAC) # send the molecule state to the SAC agent
                        # print("SAC agent is thinking...")
                        # action = nanonis.SAC2auto_queue.get() # get the action from the SAC agent
                        # save the action to the npy
                        # save_action_to_npy(action, file_path=nanonis.SAC_origin_action_path + "/actions.npy")



                        print(f'Tip position:  X:{action[0]}, Y:{action[1]}')
                        print(f'Tip Bias: {action[2]}V, Tip Current: {action[3]}nA')

                        
                        # action = np.array([new_x, new_y, V, I])
                        
                        if action is not None:
                            origin_action = action
                            # action = action.tolist()
                            
                            origin_action_X = action[0]
                            origin_action_Y = action[1]
                            origin_action_V = action[2]
                            origin_action_I = action[3]
                            
                            dx_rot = action[0]
                            dy_rot = action[1]
                            
                            # grid the X Y position
                            (dx_rot,dy_rot) =  find_nearest_grid_point(nanonis.env.H_xy_grid_points, (dx_rot,dy_rot))

                            # if click the middle button, change the Bias and Setpoint
                            if voltage_click or current_click:
                                tip_bias = nanonis.convert(str(voltage_click))
                                tip_current = nanonis.convert(str(current_click))
                            else:
                                # (action[2],action[3]) =  find_nearest_grid_point(nanonis.env.vi_grid_points, (action[2],action[3]))

                                tip_bias = action[2]
                                tip_current = action[3]
                            
                            # result for nanonis coordinate convertion
                            result = reverse_point_in_rotated_rect_polar(dx_rot, dy_rot, (nanonis.scan_square_Buffer_pix,nanonis.scan_square_Buffer_pix), center, angle, nanonis.SAC_XYaction_100, nanonis.zoom_in_100, nanonis.scan_square_Buffer_pix)
                            
                        if result_human is not None:
                            result = result_human

                        if  no_mol_binary_arr is not None:
                            result = None
                            no_mol_binary_arr = None # init the no_mol_binary_arr

                        
                        if 2 in binary_arr or 3 in binary_arr:  # after the molecule is fully reacted or bad, find the next molecule
                            result = None
                            # temp_buffer = [] # init the temp_buffer
                            # break

                    #########################################################################

                    log_path = nanonis.mol_tip_induce_path + '/image_for' + image_save_time + '.log'
                    # r_frac, theta_deg, dx_rot, dy_rot = point_in_rotated_rect_polar(result, (nanonis.scan_square_Buffer_pix,nanonis.scan_square_Buffer_pix), center, angle, nanonis.SAC_XYaction_100)
                    if  voltage_click or current_click:
                        tip_bias = nanonis.convert(str(voltage_click))
                        tip_current = nanonis.convert(str(current_click))

                
                    
                    # click the left mouse button
                    if result: # molcule do not have to tip manipulation anymore
                        r_frac, theta_deg, dx_rot, dy_rot = point_in_rotated_rect_polar(result, (nanonis.scan_square_Buffer_pix,nanonis.scan_square_Buffer_pix), center, angle, nanonis.SAC_XYaction_100)
                        dx_rot = dx_rot * nanonis.zoom_in_100/ nanonis.scan_square_Buffer_pix
                        dy_rot = dy_rot * nanonis.zoom_in_100/ nanonis.scan_square_Buffer_pix

                        # exchange the dx_rot and dy_rot
                        # dx_rot, dy_rot = dy_rot, dx_rot

                        nanonis.molecule_registry.update_molecule(molecular_index, operated = 1, operated_time = time.time())
                        molecule = nanonis.molecule_registry.molecules[molecular_index]

                        mouse_click_pos_nano = matrix_to_cartesian(result[0], result[1],center = nanocoodinate ,side_length=zoom_in_scale)
                        # mouse_click_pos_nano = matrix_to_cartesian(result[1], result[0],center = mol_old_position ,side_length=zoom_in_scale)

                        # move the tip to the click position
                        print('Move the tip to the click position.')
                        nanonis.FolMeSpeedSet(nanonis.tip_manipulate_speed, 1)
                        nanonis.TipXYSet(mouse_click_pos_nano[0], mouse_click_pos_nano[1])
                        # time.sleep(0.5)
                        nanonis.FolMeSpeedSet(nanonis.tip_manipulate_speed, 0)
                        time.sleep(1)






##############################  Tip manipulation  ########################################
                        print('Start the tip manipulation...')
                        lift_time = 0.5  # s
                        max_hold_time = 20  # s
                        pulse_time = 0.1  # s

                        tip_bias_init = nanonis.BiasGet()
                        tip_current_init = nanonis.SetpointGet()

                        abs_tip_bias_init = abs(tip_bias_init)
                        abs_tip_current_init = abs(tip_current_init)

                        if tip_induce_mode == 'pulse':

                            # set the tip bias and current should be changed in 4s
                            steps = 20  # 分成50步
                            time_interval = lift_time / steps  # 每步的时间间隔

                            # 计算每步的增量
                            current_step = (tip_current - abs_tip_current_init) / steps
                            

                            # nanonis.ZCtrlOff()

                            for i in range(steps):
                                # 逐步设置Setpoint
                                nanonis.SetpointSet(abs_tip_current_init + current_step * (i + 1))
                                time.sleep(time_interval)
                            
                            # nanonis.ZCtrlOff()

                            # 初始化参数
                            pre_pulse_duration = 0.2  # 前0.5秒提取信号
                            post_pulse_duration = 0.2  # 后0.5秒提取信号
                            max_iterations = 1  # 最大循环次数
                            threshold = '20p'  # 插值阈值
                            threshold = nanonis.convert(threshold) * 1e9  # 将阈值转换为纳米单位
                            pulse_time = 0.1  # nanonis.pulse的持续时间

                            # 开始循环
                            for iteration in range(max_iterations):
                                # 提取前0.5秒的信号
                                pre_pulse_signals = []
                                start_time = time.time()
                                while time.time() - start_time < pre_pulse_duration:
                                    signal_Z = nanonis.SignalValsGet(30)['0'] * 1e9  # 获取信号值
                                    pre_pulse_signals.append(signal_Z)
                                    time.sleep(0.01)  # 防止过于频繁的采样

                                # 计算前0.5秒信号的平均值
                                pre_pulse_avg = sum(pre_pulse_signals) / len(pre_pulse_signals)
                                time.sleep(0.1)  # 等待0.5秒
                                # 施加脉冲
                                nanonis.BiasPulse(tip_bias, pulse_time)
                                time.sleep(0.1)  # 等待脉冲施加完成
                                # 提取后0.5秒的信号
                                post_pulse_signals = []
                                start_time = time.time()
                                while time.time() - start_time < post_pulse_duration:
                                    signal_Z = nanonis.SignalValsGet(30)['0'] * 1e9  # 获取信号值
                                    post_pulse_signals.append(signal_Z)
                                    time.sleep(0.01)  # 防止过于频繁的采样

                                # 计算后0.5秒信号的平均值
                                post_pulse_avg = sum(post_pulse_signals) / len(post_pulse_signals)

                                # 计算插值
                                signal_difference = abs(post_pulse_avg - pre_pulse_avg)

                                print(f"Iteration {iteration + 1}: Pre-pulse Avg = {pre_pulse_avg}, Post-pulse Avg = {post_pulse_avg}, Difference = {signal_difference}")

                                # 判断是否满足中止条件
                                if signal_difference > threshold:
                                    print("Signal difference exceeds threshold. Exiting loop.")
                                    break

                            # 如果达到最大循环次数仍未满足条件，强制跳出
                            if iteration == max_iterations - 1:
                                print("Maximum iterations reached. Forcing exit.")
                            # nanonis.BiasSet(tip_bias_init)
                            nanonis.SetpointSet(tip_current_init)
                            # nanonis.ZCtrlOnSet()

                        time.sleep(1)
                        print('Tip manipulation is done.')
##############################  Tip manipulation  ########################################
# 
# 
# 
##############################  save the log and experience  ########################################
                        # save the nanonis.image_for and image_for_rgb in nanonis.mol_tip_induce_path
                        
                        cv2.imwrite(nanonis.mol_tip_induce_path + '/image_for'+ image_save_time +'.png', nanonis.image_for)
                        cv2.imwrite(nanonis.mol_tip_induce_path + '/image_for_rgb'+ image_save_time +''+ '_'+ voltage+'V_' + current +'A_' + 'tip_cood'+ str((dx_rot, dy_rot)) +'.png', image_for_rgb)

                        
                        with open(log_path, 'w', encoding='utf-8') as f:                                # save the log
                            f.write(f"time: {image_save_time}\n")
                            f.write(f"molecule_position: {molecule.position}\n")
                            f.write(f"angle: {angle}\n")
                            f.write(f"binary_arr: {binary_arr}\n")
                            f.write(f"dx_rot: {dx_rot}, dy_rot: {dy_rot}\n")
                            f.write(f"tip_bias: {tip_bias}, tip_current: {tip_current}\n")

                            nanonis.save_checkpoint()  

############################################################################################################
            while True:
                nanonis.FolMeSpeedSet(nanonis.zoom_in_tip_speed, 1)
                tracker_scale = nanonis.convert(nanonis.scan_zoom_in_list[-1])
                nanonis.TipXYSet(nanocoodinate[0], nanocoodinate[1]+0.5*tracker_scale)
                nanonis.FolMeSpeedSet(nanonis.zoom_in_tip_speed, 0)
                time.sleep(0.5)

                nanonis.batch_scan_producer(nanocoodinate, zoom_in_scale, nanonis.scan_square_Buffer_pix, 0)

                if code_simulation_mode:
                #########################################################################
                    nanonis.image_for = cv2.imread(img_simu_30_path, cv2.IMREAD_GRAYSCALE)           # read the simu image
                    nanonis.image_for = cv2.resize(nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA)
                #########################################################################

                key_points_result = nanonis.batch_molecular_seeker(nanonis.image_for, scan_posion = nanocoodinate, scan_edge = zoom_in_scale_nano, one_molecular_pos_list = mol_pos_nano_list)
                
                # draw the XYedge frame in scan_for
                image_for_rgb =  cv2.cvtColor(nanonis.image_for, cv2.COLOR_GRAY2BGR)
                if key_points_result:
                    for box_index,key_points in enumerate(key_points_result):
                        if key_points :
                            # image_for_rgb =  cv2.cvtColor(nanonis.image_for, cv2.COLOR_GRAY2BGR)
                            square_side = nanonis.SAC_XYaction_edge/zoom_in_scale * nanonis.scan_square_Buffer_pix
                            center = (int(key_points_result[box_index][5]*nanonis.scan_square_Buffer_pix), int(key_points_result[box_index][6]*nanonis.scan_square_Buffer_pix))
                            rot_rect = (center, (square_side, square_side), angle)
                            box = cv2.boxPoints(rot_rect)
                            box = np.intp(box)
                            cv2.drawContours(image_for_rgb, [box], 0, (0, 255, 0), 1)

                            # 计算 box 的中心点
                            box_center_x = int((box[0][0] + box[2][0]) / 2)
                            box_center_y = int((box[0][1] + box[2][1]) / 2)

                            # 在 box 的中心绘制 box_index
                            cv2.putText(image_for_rgb, str(box_index), (box_center_x, box_center_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                # result, image_for_rgb, (voltage_click, current_click), br_sites_array= mouse_click_tip(image_for_rgb, mode = 'batch', state_num = len(next_state_buffer_list))

                # init all list for exp
                state_buffer_list = []
                action_buffer_list = []
                reward_buffer_list = []
                next_state_buffer_list = []
                done_buffer_list = []
                info_buffer_list = []
                buffer_list = []
                mol_pos_nano_list = []

                nanonis.molecule_registry.clear_the_registry()

                # pulse the Program
                # rescan = input("Program paused. \nPress Enter to continue... \nInput anything to rescan the image.")
                rescan = None
                if not rescan:
                    break
            rescan = None
            if time.time() - main_state_time > main_time:
                break


        
            # break   # Jump out of the loop, scan the next point    
                
        scan_qulity == 1                    
        nanonis.skip_flag = 0
        nanonis.save_checkpoint()                                                       # save the checkpoint