from driving_models import *
import os
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import re

def load_images(folder_path):
    images = []
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)
        if image is not None:
            images.append(image)
    return np.array(images)

def predict_with_model(model, images):
    # 将图像数据转换为模型输入的格式
    input_images = preprocess_images(images)

    # 使用模型进行预测
    predictions = model.predict(input_images)

    # 对预测结果进行后处理（根据具体需求进行调整）
    processed_predictions = postprocess_predictions(predictions)

    return processed_predictions

# 在预测之前，需要对图像进行预处理，以符合模型的输入格式
def preprocess_images(images):
    # 根据模型的输入尺寸调整图像大小
    resized_images = [cv2.resize(image, (100, 100)) for image in images]
    # 将图像转换为模型所需的数据类型并归一化
    processed_images = np.array(resized_images, dtype=np.float32) / 255.0
    return processed_images

# 对模型输出进行后处理，例如根据具体任务的需要转换格式、应用阈值、反归一化等
def postprocess_predictions(predictions):
    # 这里简单返回预测结果，可以根据具体情况进行后续处理
    return predictions

if __name__ == '__main__':
    # 载入模型
    model = Dave_orig(load_weights=True)

    # 加载图像
    adv_image_folder = './generated_inputs1'  # 对抗样本图像文件夹的路径
    adv_images = load_images(adv_image_folder)

    # 使用模型进行预测
    predictions = predict_with_model(model, adv_images)

    print("predictions",predictions)

    # 获取真实标签
    true_labels = []
    for file_name in os.listdir(adv_image_folder):
        match = re.search(r'\[([-+]?\d*\.\d+)\]', file_name)  # 使用正则表达式提取真实标签
        if match:
            true_label_str = match.group(1)
            true_label = float(true_label_str)
            true_labels.append(true_label)
    true_labels = np.array(true_labels)
    print("true_labels", true_labels)

    # 计算评价指标
    mse = mean_squared_error(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)
    accuracy = accuracy_score(np.round(true_labels), np.round(predictions))

    # 打印评价结果
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print(f"Accuracy: {accuracy}")
