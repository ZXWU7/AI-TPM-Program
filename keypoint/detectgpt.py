import torch
import cv2

# 加载模型
model = torch.load('weights/best.pt')

# 定义类别名称列表
class_names = ['m1', 'm2']

# 定义检测函数
def detect(img):
    # 预处理图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)

    # 模型推理
    model.eval()
    with torch.no_grad():
        output = model(img)

    # 后处理输出
    boxes = output[0]['boxes'].cpu().numpy()
    scores = output[0]['scores'].cpu().numpy()
    labels = output[0]['labels'].cpu().numpy()
    keypoints = output[0]['keypoints'].cpu().numpy()
    keep = scores > 0.5  # 设置置信度阈值

    boxes = boxes[keep]
    labels = labels[keep]
    keypoints = keypoints[keep]

    # 将坐标转换为原始图像的坐标
    height, width, _ = img.shape[2:]
    boxes = boxes * [width, height, width, height]
    keypoints = keypoints * [width, height, 1, height, 1, width, height, width]

    return labels, boxes, keypoints

# 加载测试图片
img = cv2.imread('data\images\020Cu1 20×20.jpg')

# 进行检测
labels, boxes, keypoints = detect(img)

# 可视化输出图片
for label, box, keypoint in zip(labels, boxes, keypoints):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for i in range(0, len(keypoint), 2):
        x, y = int(keypoint[i]), int(keypoint[i+1])
        cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
    cv2.putText(img, class_names[label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow('detections', img)
cv2.waitKey(0)

# 将结果保存到txt文档中
with open('results.txt', 'w') as f:
    for label, box, keypoint in zip(labels, boxes, keypoints):
        x1, y1, x2, y2 = box.astype(int)
        f.write(f"{class_names[label]} {x1} {y1} {x2} {y2} {keypoint[0]} {keypoint[1]} {keypoint[2]}\
")

