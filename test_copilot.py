# import math

# import cv2
# import numpy as np

# # # 初始化全局变量
# # boxes = []  # 所有框的列表，包含原始和伪造的
# # history_stack = []  # 操作历史记录
# # dragging = False
# # current_operation = []  # 当前操作中的修改
# # angle = 35.154730569873465  # 旋转角度
# # image_size = 304
# # side_length = 20

# # 解析key_results


# def interactive_keypoint_selector(key_results, image_size=304, angle=35.154730569873465, side_length=20):
#     """交互式关键点选择函数
#     参数：
#         key_results: 关键点识别结果列表
#         image_size: 图像尺寸（默认304）
#         angle: 旋转角度（默认35.15°）
#     返回：
#         选中的key_result列表（包含原始和伪造的）
#     """
#     # 初始化状态变量
#     boxes = []
#     history_stack = []
#     dragging = False
#     current_operation = []
#     selected_results = None

#     # 计算旋转矩形
#     def calculate_rotated_square(center_x, center_y, angle_deg, side_length=side_length):
#         angle_rad = math.radians(angle_deg)
#         cos_theta = math.cos(angle_rad)
#         sin_theta = math.sin(angle_rad)
#         half = side_length // 2

#         points = [(-half, -half), (-half, half),
#                  (half, half), (half, -half)]

#         rotated_points = []
#         for (x, y) in points:
#             rx = x * cos_theta - y * sin_theta
#             ry = x * sin_theta + y * cos_theta
#             rotated_points.append((int(center_x + rx), int(center_y + ry)))
#         return rotated_points

#     # 初始化boxes
#     for res in key_results:
#         kpt_x = res[5] * image_size
#         kpt_y = res[6] * image_size
#         poly = calculate_rotated_square(kpt_x, kpt_y, angle)
#         boxes.append({
#             'poly': poly,
#             'key_result': res,
#             'selected': False,
#             'is_fake': False
#         })

#     # 鼠标回调函数
#     def mouse_callback(event, x, y, flags, param):
#         nonlocal dragging, current_operation, boxes, history_stack

#         if event == cv2.EVENT_LBUTTONDOWN:
#             dragging = True
#             current_operation = []
#             in_box = False

#             # 检测点击现有框
#             for i, box in enumerate(boxes):
#                 contour = np.array(box['poly'], dtype=np.int32).reshape((-1, 1, 2))
#                 if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
#                     prev_state = box['selected']
#                     box['selected'] = not prev_state
#                     current_operation.append((i, prev_state))
#                     in_box = True

#             # 创建伪造框
#             if not in_box:
#                 x_percent = x / image_size
#                 y_percent = y / image_size
#                 fake_key = (0.0, 0.5, 0.5, 0.1, 0.1, x_percent, y_percent)
#                 poly = calculate_rotated_square(x, y, angle)
#                 boxes.append({
#                     'poly': poly,
#                     'key_result': fake_key,
#                     'selected': True,
#                     'is_fake': True
#                 })
#                 current_operation.append((len(boxes)-1, False))

#             if current_operation:
#                 history_stack.append(current_operation.copy())
#                 current_operation.clear()

#         elif event == cv2.EVENT_MOUSEMOVE and dragging:
#             # 拖拽选择
#             for i, box in enumerate(boxes):
#                 contour = np.array(box['poly'], dtype=np.int32).reshape((-1, 1, 2))
#                 if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
#                     if not box['selected']:
#                         prev_state = box['selected']
#                         box['selected'] = True
#                         current_operation.append((i, prev_state))

#         elif event == cv2.EVENT_LBUTTONUP and dragging:
#             dragging = False
#             if current_operation:
#                 history_stack.append(current_operation.copy())
#                 current_operation.clear()

#         elif event == cv2.EVENT_RBUTTONDOWN:
#             # 撤销操作
#             if history_stack:
#                 last_op = history_stack.pop()
#                 for idx, prev_state in last_op:
#                     boxes[idx]['selected'] = prev_state

#     # 创建窗口
#     cv2.namedWindow('Keypoint Selector')
#     cv2.setMouseCallback('Keypoint Selector', mouse_callback)

#     # 创建空白图像（假设单通道输入）
#     image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

#     while True:
#         display_img = image.copy()

#         # 绘制所有框
#         for box in boxes:
#             color = (0, 255, 0) if not box['selected'] else (0, 0, 255)
#             thickness = 1 if not box['selected'] else 2
#             pts = np.array(box['poly'], dtype=np.int32).reshape((-1, 1, 2))
#             cv2.polylines(display_img, [pts], True, color, thickness)

#         cv2.imshow('Keypoint Selector', display_img)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord(' '):  # 空格键确认选择
#             selected_results = [box['key_result'] for box in boxes if box['selected']]
#             break
#         elif key == ord('r'):  # 重置选择
#             for box in boxes:
#                 box['selected'] = False
#             history_stack.clear()
#         elif key == 27:  # ESC退出
#             selected_results = []
#             break

#     cv2.destroyAllWindows()
#     return selected_results

# # 使用示例
# if __name__ == "__main__":
#     key_results = [
#     (0.0, 0.842144787311554, 0.90380859375, 0.16787107288837433, 0.16386719048023224, 0.84375, 0.90625),
#     (0.0, 0.5298095345497131, 0.7046264410018921, 0.162109375, 0.16015625, 0.5289062261581421, 0.704296886920929),
#     (0.0, 0.544543445110321, 0.519421398639679, 0.1689453125, 0.16025392711162567, 0.5453124642372131, 0.518359363079071),
#     (0.0, 0.7579101324081421, 0.09057006984949112, 0.17763669788837433, 0.16425780951976776, 0.7593749761581421, 0.08173827826976776),
#     (0.0, 0.39691162109375, 0.4164489805698395, 0.15507811307907104, 0.15449218451976776, 0.39726561307907104, 0.416015625),
#     (0.0, 0.692626953125, 0.6279662847518921, 0.15996094048023224, 0.1591796875, 0.693359375, 0.6253905892372131),
#     (0.0, 0.5660644173622131, 0.33607175946235657, 0.171875, 0.17070311307907104, 0.5628905892372131, 0.33613279461860657),
#     (0.0, 0.3628784120082855, 0.7807068228721619, 0.16855467855930328, 0.1602538824081421, 0.36445310711860657, 0.780468761920929),
#     (0.0, 0.903637707233429, 0.18942870199680328, 0.18027345836162567, 0.17128904163837433, 0.9039062857627869, 0.19140625),
#     (0.0, 0.858508288860321, 0.5450195670127869, 0.15458980202674866, 0.1532226800918579, 0.8617187142372131, 0.5445312261581421),
#     (0.0, 0.11798705905675888, 0.20683594048023224, 0.1689453125, 0.16816405951976776, 0.11582031100988388, 0.20429687201976776),
#     (0.0, 0.08746337890625, 0.3848632574081421, 0.17324219644069672, 0.17929686605930328, 0.07773437350988388, 0.3804687559604645),
#     (0.0, 0.2753051817417145, 0.13356932997703552, 0.19101563096046448, 0.18613280355930328, 0.27558591961860657, 0.13095702230930328),
#     (0.0, 0.21525879204273224, 0.6694579720497131, 0.16074217855930328, 0.16152341663837433, 0.21328125894069672, 0.669921875),
#     (0.0, 0.06778564304113388, 0.7387939095497131, 0.13876952230930328, 0.16835935413837433, 0.052001953125, 0.7386718392372131),
#     (0.0, 0.42082521319389343, 0.239654541015625, 0.1689453423023224, 0.16718749701976776, 0.41796875, 0.2431640625),
#     (0.0, 0.44257810711860657, 0.07417602092027664, 0.19589842855930328, 0.14511717855930328, 0.44140625, 0.05380859598517418),
#     (0.0, 0.7147216796875, 0.4421630799770355, 0.16640622913837433, 0.17099608480930328, 0.712890625, 0.439453125),
#     (0.0, 0.6844726204872131, 0.8011535406112671, 0.17441408336162567, 0.16982419788837433, 0.684374988079071, 0.800000011920929),
#     (0.0, 0.23344725370407104, 0.489013671875, 0.17558594048023224, 0.17949217557907104, 0.232421875, 0.4886718690395355),
#     (0.0, 0.19512939453125, 0.8559021353721619, 0.166015625, 0.16406254470348358, 0.19833983480930328, 0.85546875),
#     (0.0, 0.5843262076377869, 0.16587524116039276, 0.18916015326976776, 0.1953125, 0.5835937261581421, 0.16357421875),
#     (0.0, 0.37822267413139343, 0.599255383014679, 0.16933594644069672, 0.16396482288837433, 0.37910154461860657, 0.5960937142372131),
#     (0.0, 0.080902099609375, 0.5613159537315369, 0.16640625894069672, 0.1830078363418579, 0.07197265326976776, 0.559765636920929),
#     (0.0, 0.728955090045929, 0.26123046875, 0.16884763538837433, 0.16347655653953552, 0.727734386920929, 0.2583984434604645),
#     (0.0, 0.523449718952179, 0.881359875202179, 0.17216792702674866, 0.15703125298023224, 0.522265613079071, 0.87890625),
#     (0.0, 0.87548828125, 0.3656005859375, 0.15937504172325134, 0.16054688394069672, 0.875781238079071, 0.36328125),
#     (0.0, 0.06206054612994194, 0.9166014790534973, 0.12617187201976776, 0.16787107288837433, 0.02675781212747097, 0.9296875),
#     (0.0, 0.852001965045929, 0.7257323861122131, 0.16142578423023224, 0.15439455211162567, 0.8531250357627869, 0.72265625),
#     (0.0, 0.12814940512180328, 0.06146240234375, 0.18447263538837433, 0.11835936456918716, 0.12275391072034836, 0.02543945237994194),
#     (0.0, 0.9145264029502869, 0.04013672098517418, 0.1597656011581421, 0.07524413615465164, 0.91796875, 0.0),
#     (0.0, 0.68035888671875, 0.950927734375, 0.16621093451976776, 0.09438474476337433, 0.6800780892372131, 0.981249988079071),
#     (0.0, 0.9603636860847473, 0.822338879108429, 0.08457033336162567, 0.17587892711162567, 1.0, 0.8203125),
#     (0.0, 0.965624988079071, 0.6426758170127869, 0.07109370827674866, 0.16962890326976776, 1.0, 0.6390624642372131),
#     (2.0, 0.25902098417282104, 0.3128418028354645, 0.1958007663488388, 0.18476562201976776, 0.2574218809604645, 0.3089843690395355),
#     (0.0, 0.9739990234375, 0.4640869200229645, 0.05473628640174866, 0.17216797173023224, 1.0, 0.4624999761581421),
#     (0.0, 0.18186034262180328, 0.9734862446784973, 0.1845703125, 0.055712949484586716, 0.18037109076976776, 1.0)
#     ]
#     image_size = 304     # 可选参数
#     angle = 35.1547      # 可选参数
#     # 调用函数
#     selected = interactive_keypoint_selector(
#         key_results=key_results,
#         image_size=image_size,
#         angle=angle,
#         side_length=20
#     )

#     # 输出结果处理
#     print(f"最终选择了 {len(selected)} 个关键点：")
#     for res in selected:
#         print(f"- 位置：({res[5]:.4f}, {res[6]:.4f})")


