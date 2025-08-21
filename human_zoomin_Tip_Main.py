from Auto_scan_class import *
from core import NanonisController
from DQN.agent import *
from EvaluationCNN.detect import predict_image_quality
from keypoint.detect import key_detect
from mol_segment.detect import segmented_image
from molecule_registry import Molecule, Registry
from SAC_Br_nanonis import ReplayBuffer
from square_fitting import mol_Br_site_detection
from utils import *

if __name__ == '__main__':


    # img_simu_30_path = './STM_img_simu/30/059 [Z_fwd] image30.png'
    # img_simu_7_path = './STM_img_simu/7/117 [Z_fwd] image10.png'


    nanonis = Mustard_AI_Nanonis()
    # nanonis.tip_init(mode = 'new') # deflaut mode is 'new' mode = 'new' : the tip is initialized to the center and create a new log folder, mode = 'latest' : load the latest checkpoint
    
    # nanonis.DQN_init(mode = 'new') # deflaut mode is 'latest' mode = 'new' : create a new model, mode = 'latest' : load the latest checkpoint    

    nanonis.monitor_thread_activate()                                                # activate the monitor thread
    
    voltage = '1.5'
    current = '0.15n'
    tip_bias = nanonis.convert(voltage)
    tip_current = nanonis.convert(current)

    zoom_out_scale = nanonis.convert(nanonis.scan_zoom_in_list[0])
    zoom_out_scale_nano = zoom_out_scale*10**9
    


    temp_buffer = []

    # if nanonis.mol_tip_induce_path is not exist, create it
    if not os.path.exists(nanonis.mol_tip_induce_path):
        os.makedirs(nanonis.mol_tip_induce_path)
    # SAC_buffer
    if not os.path.exists(nanonis.SAC_buffer_path):
        os.makedirs(nanonis.SAC_buffer_path)

    # while tip_in_boundary(nanonis.inter_closest, nanonis.plane_size, nanonis.real_scan_factor):
        
    #     nanonis.move_to_next_point()                                                    # move the scan area to the next point

    #     nanonis.AdjustTip_flag = nanonis.AdjustTipToPiezoCenter()                       # check & adjust the tip to the center of the piezo

    #     nanonis.line_scan_thread_activate()                                             # activate the line scan, producer-consumer architecture, pre-check the tip and sample

    #     nanonis.batch_scan_producer(nanonis.nanocoodinate, nanonis.Scan_edge, nanonis.scan_square_Buffer_pix, 0)    # Scan the area
        
    #     # nanonis.image_for = cv2.imread(img_simu_30_path, cv2.IMREAD_GRAYSCALE)           # read the simu image
    #     # resize the image 304*304
    #     # nanonis.image_for = cv2.resize(nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA)


    #     scan_qulity = nanonis.image_recognition()                                       # assement the scan qulity 

    #     # nanonis.image_segmention(nanonis.image_for)                                   # segment the image ready to tip shaper        
    #     scan_qulity == 1
    #     # scan_qulity = 0
    #     # nanonis.create_trajectory(scan_qulity)                                          # create the trajectory for tip shaper DQN

    #     if scan_qulity == 0:
    #         pass
    #         # nanonis.DQN_upgrate()                                                       # optimize the model and update the target network
        
    #     elif scan_qulity == 1:
    #         # TODO: 1.molecular detection 
    #         #       2.regestration of all molecular position, find the candidate
    #         #       3.move the tip to the candidate,
    Frame = nanonis.ScanFrameGet()

    molecule.position = [Frame['center_x'], Frame['center_y']]

    # nanonis.molecular_seeker(nanonis.image_for,scan_posion = nanonis.nanocoodinate, scan_edge = zoom_out_scale_nano)
    # while True:
        # molecule, molecular_index = nanonis.molecular_tracker(tracker_position = nanonis.nanocoodinate, tracker_scale = nanonis.scan_zoom_in_list[-1])
        # if there are moleculars that need to be manipulated
        # if molecule:    # molecule is not None
    state = None  # init the state 
    
    reaction_path_flag = 0
    while not np.array_equal(molecule.site_states, np.array([1, 1, 1, 1])) and 2 not in molecule.site_states: # 分子没反应完并且没坏掉
        zoom_in_scale = nanonis.convert(nanonis.scan_zoom_in_list[-1])
        zoom_in_scale_nano = zoom_in_scale*10**9
        image_save_time = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))    # save the image with time
        nanonis.batch_scan_producer(molecule.position, zoom_in_scale, nanonis.scan_square_Buffer_pix, 0)
        # nanonis.image_for = cv2.imread(img_simu_7_path, cv2.IMREAD_GRAYSCALE)           # read the simu image
        # nanonis.image_for = cv2.resize(nanonis.image_for, (304, 304), interpolation=cv2.INTER_AREA)
        molecule_nanocood = (molecule.position[0]*10**9,molecule.position[1]*10**9)
        # segment the image
        # mask,_,_ = segmented_image(nanonis.image_for, nanonis.segmented_image_path, model_path = nanonis.segment_model_path)
        # key_points_result = nanonis.molecular_seeker(nanonis.image_for, scan_posion = molecule_nanocood, scan_edge = zoom_in_scale_nano)
        
        # if key_points_result == None:    #  No molecule
        #     print('No molecule detected in the image.')
        #     cv2.imwrite(nanonis.mol_tip_induce_path + '/image_for'+ image_save_time +'.png', nanonis.image_for)
        #     break

        # angle, Br_site_state_list, patches = mol_Br_site_detection(nanonis.image_for, mask, key_points_result, zoom_in_scale, nanonis.mol_scale, nanonis.Br_scale, nanonis.square_save_path, nanonis.site_model_path)
        angle = 0
        Br_site_state_list = [0,0,0,0]
        binary_arr = [1 if state > nanonis.Br_site_threshold else 0 for state in Br_site_state_list]
        
    
        print('the state of the Br sites:', binary_arr)
        # save the new position, new angle, to the registry
        nanonis.molecule_registry.update_molecule(molecular_index, position = molecule.position, site_states = binary_arr, orientation = angle)
        # TODO: calculate 4 Br atoms position.
        Br_pos_nano = cal_Br_pos(molecule.position, nanonis.mol_scale, angle)
        nanonis.molecule_registry.update_molecule(molecular_index, Br_postion = Br_pos_nano)
        # draw the XYedge frame in scan_for
        image_for_rgb =  cv2.cvtColor(nanonis.image_for, cv2.COLOR_GRAY2BGR)
        # square_side = nanonis.SAC_XYaction_edge/zoom_in_scale * nanonis.scan_square_Buffer_pix
        # center = (int(key_points_result[0][5]*nanonis.scan_square_Buffer_pix), int(key_points_result[0][6]*nanonis.scan_square_Buffer_pix))
        # rot_rect = (center, (square_side, square_side), angle)
        # box = cv2.boxPoints(rot_rect)
        # box = np.int0(box)
        # cv2.drawContours(image_for_rgb, [box], 0, (0, 255, 0), 2)
        
        # show the image
        # cv2.imshow('image', image_for_rgb)
        # cv2.waitKey(0)

        print("Please click the tip position on the image.")
        result, image_for_rgb, (voltage_click, current_click), br_sites_array= mouse_click_tip(image_for_rgb)   #result is the pos of the tip， mouse_info = [x, y, click_flag]
        
        if br_sites_array is not None:  # force the Br sites to be the new state
            nanonis.molecule_registry.update_molecule(molecular_index,site_states = br_sites_array)

        log_path = nanonis.mol_tip_induce_path + '/image_for' + image_save_time + '.log'
        # r_frac, theta_deg, dx_rot, dy_rot = point_in_rotated_rect_polar(result, (nanonis.scan_square_Buffer_pix,nanonis.scan_square_Buffer_pix), center, angle, nanonis.SAC_XYaction_100)
        if  voltage_click or current_click:
            tip_bias = nanonis.convert(str(voltage_click))
            tip_current = nanonis.convert(str(current_click))

        if result: # molcule do not have to tip manipulation anymore
            center = (int(0.5*nanonis.scan_square_Buffer_pix), int(0.5*molecule.position[1]*nanonis.scan_square_Buffer_pix))
            r_frac, theta_deg, dx_rot, dy_rot = point_in_rotated_rect_polar(result, (nanonis.scan_square_Buffer_pix,nanonis.scan_square_Buffer_pix), center, angle, nanonis.SAC_XYaction_100)
            nanonis.molecule_registry.update_molecule(molecular_index, operated = 1, operated_time = time.time())
            mouse_click_pos_nano = matrix_to_cartesian(result[0], result[1],center = molecule.position ,side_length=zoom_in_scale)
            # move the tip to the click position
            nanonis.FolMeSpeedSet(nanonis.tip_manipulate_speed, 1)
            nanonis.TipXYSet(mouse_click_pos_nano[0], mouse_click_pos_nano[1])
            nanonis.FolMeSpeedSet(nanonis.tip_manipulate_speed, 0)
            time.sleep(1)
            # do the tip manipulation
            tip_bias_init = nanonis.BiasGet()
            tip_current_init = nanonis.SetpointGet()
            # set the tip bias and current should be changed in 4s

            steps = 20  # 分成20步
            time_interval = 4 / steps  # 每步的时间间隔

            # 计算每步的增量
            bias_step = (tip_bias - tip_bias_init) / steps
            current_step = (tip_current - tip_current_init) / steps

            for i in range(steps):
                # 逐步设置Bias和Setpoint
                nanonis.BiasSet(tip_bias_init + bias_step * (i + 1))
                nanonis.SetpointSet(tip_current_init + current_step * (i + 1))
                time.sleep(time_interval)
            
            time.sleep(6)   # wait for the tip induce

            # initialize the tip bias and current
            nanonis.BiasSet(tip_bias_init)
            nanonis.SetpointSet(tip_current_init)

            time.sleep(1)

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


            if len(temp_buffer)==0:    # temp_buffer is empty
                state = np.array(binary_arr)
                state = np.append(state, reaction_path_flag)
                action = np.array([dx_rot, dy_rot, tip_bias, tip_current*10**9])
                temp_buffer.append(state)
                temp_buffer.append(action)
            else:
                next_state = np.array(binary_arr) 
                state = np.append(state, reaction_path_flag)
                state_legal, trans_type = nanonis.buffer4log.legalize_state(state)
                action_corrected = nanonis.buffer4log.transform_action(action, trans_type)
                next_state_transform = nanonis.buffer4log.transform_state(next_state, trans_type)
                
                reward,info = nanonis.env.reward_culculate(temp_buffer[0],next_state,0)
                if 2 in state or all(state[:4]) or info["reaction"] in ["bad", "wrong"]:
                    done = 1                                
                temp_buffer.append(reward)
                temp_buffer.append(next_state)
                
                temp_buffer.append(done)
                temp_buffer.append(info)
                nanonis.buffer4log.add(temp_buffer[0],temp_buffer[1],temp_buffer[2],temp_buffer[3],temp_buffer[4],temp_buffer[6])
                nanonis.buffer4aug.aug_exp(temp_buffer[0],temp_buffer[1],temp_buffer[2],temp_buffer[3],temp_buffer[4],temp_buffer[6])
                nanonis.buffer4log.save(nanonis.SAC_buffer_path)
                nanonis.buffer4aug.save(nanonis.SAC_buffer_path)

                temp_buffer = [] # init the temp_buffer
                state = np.array(binary_arr)
                state = np.append(state, reaction_path_flag)
                action = np.array([dx_rot, dy_rot, tip_bias, tip_current*10**9])  # x y v i
                temp_buffer.append(state)
                temp_buffer.append(action)







        else:
            print('Quit the tip induce, the molecule is not operated.')
            # if buffer is empty, break the loop
            if len(temp_buffer)==0:
                break
            if br_sites_array is not None: # force the Br sites to be the new state
                site_states = br_sites_array
            else:
                site_states = np.array([2,2,2,2]) 
            nanonis.molecule_registry.update_molecule(molecular_index, site_states = np.array([2,2,2,2]), status=2)
            cv2.imwrite(nanonis.mol_tip_induce_path + '/image_for'+ image_save_time +'.png', nanonis.image_for)
            with open(log_path, 'w', encoding='utf-8') as f:                                # save the log
                f.write(f"time: {image_save_time}\n")
                f.write(f"molecule_position: {molecule.position}\n")
                f.write(f"angle: {angle}\n")
                f.write(f"binary_arr: {binary_arr}\n")
                f.write(f"dx_rot: , dy_rot: \n")
                f.write(f"tip_bias: , tip_current: \n")

                next_state = np.array(binary_arr)                         
                state_legal, trans_type = nanonis.buffer4log.legalize_state(state)
                action_corrected = nanonis.buffer4log.transform_action(action, trans_type)
                next_state_transform = nanonis.buffer4log.transform_state(next_state, trans_type)
                
                reward,info = nanonis.env.reward_culculate(temp_buffer[0],next_state,0)
                if 2 in state or all(state[:4]) or info["reaction"] in ["bad", "wrong"]:
                    done = 1                                
                temp_buffer.append(reward)
                temp_buffer.append(next_state)
                temp_buffer.append(done)
                temp_buffer.append(info)
                nanonis.buffer4log.add(temp_buffer[0],temp_buffer[1],temp_buffer[2],temp_buffer[3],temp_buffer[4],temp_buffer[6])
                nanonis.buffer4aug.aug_exp(temp_buffer[0],temp_buffer[1],temp_buffer[2],temp_buffer[3],temp_buffer[4],temp_buffer[6])
                nanonis.buffer4log.save(nanonis.SAC_buffer_path)
                nanonis.buffer4aug.save(nanonis.SAC_buffer_path)                                
                temp_buffer = []

            break # jump out of the loop, scan the next molecule 
                

        nanonis.skip_flag = 0
        # nanonis.save_checkpoint()                                                       # save the checkpoint