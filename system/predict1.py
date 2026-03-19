import os
import cv2
import torch
import numpy as np
from flcore.trainmodel.models import get_model
from config import cfg

# ===================== 固定配置（一次配置，终身使用）=====================
CLASSES = ["wheat", "corn", "rice"]
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
IMG_SIZE = (320, 320)
MODEL_SAVE_DIR = cfg.save_dir  # 模型保存目录（从配置文件读取）
# 固定待预测图像目录：放在system下的predict_images文件夹，只需替换里面的图像
PREDICT_IMG_DIR = "predict_images"
# 固定待预测图像名称：默认读取文件夹里的predict.jpg（你只需替换这张图）
PREDICT_IMG_NAME = "predict.jpg"
PREDICT_IMG_PATH = os.path.join(PREDICT_IMG_DIR, PREDICT_IMG_NAME)
# ======================================================================

def create_predict_dir():
    """自动创建待预测图像目录（如果不存在）"""
    if not os.path.exists(PREDICT_IMG_DIR):
        os.makedirs(PREDICT_IMG_DIR)
        print(f"已创建待预测图像目录：{PREDICT_IMG_DIR}")
        print(f"请将需要预测的图像放入该目录，并重命名为：{PREDICT_IMG_NAME}")

def find_latest_model(prefix):
    """自动查找最新训练轮次的模型文件"""
    model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.startswith(prefix) and f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError(f"在 {MODEL_SAVE_DIR} 中未找到 {prefix} 模型文件")
    # 按轮次排序，取最新（最大轮次）
    model_files.sort(key=lambda x: int(x.split("round")[-1].split(".")[0]), reverse=True)
    latest_model = os.path.join(MODEL_SAVE_DIR, model_files[0])
    print(f"找到最新{prefix}模型：{latest_model}")
    return latest_model

def preprocess(img_path):
    """图像预处理"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"未在 {img_path} 找到图像，请检查文件是否存在/格式是否正确")
    img_ori = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(cfg.device)
    return img_ori, img

def predict():
    """核心预测逻辑：固定路径+自动加载模型+无需参数"""
    # 1. 检查并创建待预测目录
    create_predict_dir()
    # 2. 自动查找最新模型
    resnet_path = find_latest_model("resnet18")
    yolo_path = find_latest_model("yolov8n")
    # 3. 加载模型
    resnet18 = get_model("ResNet18").to(cfg.device)
    yolov8n = get_model("YOLOv8n").to(cfg.device)
    resnet18.load_state_dict(torch.load(resnet_path, map_location=cfg.device))
    yolov8n.load_state_dict(torch.load(yolo_path, map_location=cfg.device))
    resnet18.eval()
    yolov8n.eval()
    # 4. 预处理图像（固定路径）
    img_ori, img = preprocess(PREDICT_IMG_PATH)
    h, w = img_ori.shape[:2]
    # 5. YOLOv8n检测目标框
    with torch.no_grad():
        yolo_pred = yolov8n(img)
    boxes = yolo_pred[:, :4].cpu().numpy()  # x1,y1,x2,y2
    # 6. ResNet18精准分类
    with torch.no_grad():
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # 防止裁剪越界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            # 裁剪目标区域并分类
            crop = img_ori[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, IMG_SIZE)
            crop = crop / 255.0
            crop = torch.FloatTensor(crop).permute(2, 0, 1).unsqueeze(0).to(cfg.device)
            pred = resnet18(crop)
            cls_id = torch.argmax(pred, dim=1).item()
            cls_name = CLASSES[cls_id]
            conf = torch.softmax(pred, dim=1)[0][cls_id].item()
            # 绘制框和标签（带置信度）
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(img_ori, (x1, y1), (x2, y2), COLORS[cls_id], 2)
            cv2.putText(img_ori, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[cls_id], 2)
    # 7. 保存结果（固定名称，不覆盖）
    result_path = f"predict_result_{os.path.splitext(PREDICT_IMG_NAME)[0]}.jpg"
    cv2.imwrite(result_path, img_ori)
    print(f"\n预测完成！结果已保存为：{result_path}")
    # 8. 显示结果（关闭可注释）
    cv2.imshow("Crop Recognition Result", img_ori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 无需传入任何参数，直接运行即可
    try:
        predict()
    except Exception as e:
        print(f"预测出错：{str(e)}")