import numpy as np
import time
import argparse
import cv2
from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from sa1 import fetch_dsa, fetch_lsa, get_sc
from utils import *
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

CLIP_MIN = -0.5
CLIP_MAX = 0.5

def load_images(image_folder):
    image_list = []
    for filename in os.listdir(image_folder):
        img = Image.open(os.path.join(image_folder, filename))
        img = img.resize((64, 64))  # Resize the image
        img = np.array(img, dtype=np.float32)  # Convert image to float32
        image_list.append(img)
    return np.array(image_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="udacity")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="deepxplore",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp2"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-11,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", "udacity"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa, "Select either 'lsa' or 'dsa'"
    print(args)

    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Load pre-trained model.
        model = load_model("./model/model_mnist.h5")
        model.summary()

        # You can select some layers you want to test.
        # layer_names = ["activation_1"]
        # layer_names = ["activation_2"]
        layer_names = ["activation_3"]

        # Load target set.
        x_target = np.load("./adv/adv_mnist_{}.npy".format(args.target))
        print("target",x_target)
        print("target",x_target.size)

    elif args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        model = load_model("./model/model_cifar.h5")
        model.summary()

        # layer_names = [
        #     layer.name
        #     for layer in model.layers
        #     if ("activation" in layer.name or "pool" in layer.name)
        #     and "activation_9" not in layer.name
        # ]
        layer_names = ["activation_6"]

        x_target = np.load("./adv/adv_cifar_{}.npy".format(args.target))

    elif args.d == "udacity":
        data = pd.read_csv('./Ch2_001/final_example.csv')
        label_column = 'steering_angle'  # 修改为 'steering_angle' 列作为标签列
        image_folder = './Ch2_001/center'  # 图像文件夹的路径

        # 加载图像并将其调整为模型所需的大小
        images = load_images(image_folder)
        # 提取标签
        labels = data[label_column].values

        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        model = load_model("./model/model_Dave_orig.h5")
        model.summary()

        # 选择目标层
        layer_names = ["block1_conv3"]  # 选择block1_conv3作为目标层

        # 加载对抗样本集
        adv_image_folder = './generated_inputs1'  # 对抗样本图像文件夹的路径
        adv_images = load_images(adv_image_folder)
        # x_target 作为对抗样本集
        x_target = adv_images
        x_target = x_target.astype("float32")
        x_target = (x_target / 255.0) - (1.0 - CLIP_MAX)
        print("x_target",x_target[1][1])
        print("x_target",x_target.shape)


    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    if args.lsa:
        test_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)

        target_lsa = fetch_lsa(model, x_train, x_target, args.target, layer_names, args)

        print("test_lsa：",test_lsa)
        print("target_lsa：",target_lsa)#none

        sorted_indexes = np.argsort(target_lsa)[::-1]
        # 创建一个新的文件夹来存储筛选后的图像
        output_folder = "./final_images"
        os.makedirs(output_folder, exist_ok=True)

        # 保存前三个具有最高 LSA 值的图像
        N = 10  # 保存前三个图像
        for i in range(N):
            index = sorted_indexes[i]
            image = images[index]
            lsa_value = target_lsa[index]
            image_filename = os.path.join(output_folder, f"image_{lsa_value:.6f}_{i}.jpg")
            cv2.imwrite(image_filename, image)

        print(f"前 {N} 个具有最高 LSA 值的图像已保存到 {output_folder}")

        # 处理 target_lsa 中的 None 值
        target_lsa = [val for val in target_lsa if val is not None]
        if target_lsa:
            min_value = np.amin(target_lsa)
            target_cov = get_sc(min_value, args.upper_bound, args.n_bucket, target_lsa)
        else:
            print("Error: target_lsa is empty.")

        auc = compute_roc_auc(test_lsa, target_lsa)
        print(infog("ROC-AUC: " + str(auc * 100)))

    if args.dsa:
        print("11111")
        test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)
        print("22222")
        target_dsa = fetch_dsa(model, x_train, x_target, args.target, layer_names, args)
        target_cov = get_sc(
            np.amin(target_dsa), args.upper_bound, args.n_bucket, target_dsa
        )

        auc = compute_roc_auc(test_dsa, target_dsa)
        print(infog("ROC-AUC: " + str(auc * 100)))

    print(infog("{} coverage: ".format(args.target) + str(target_cov)))
