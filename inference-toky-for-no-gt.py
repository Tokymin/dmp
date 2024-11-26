import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from PIL import Image
# from Utils.calculate_depth_metrics import compute_errors
import matplotlib.pyplot as plt
import os
from config.config_test_for_clinical_dataset import hyperparameters as param
import scipy.stats as stats
from skimage.metrics import structural_similarity as ssim



# 定义计算SSIM的函数
def calculate_ssim(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score


# 定义计算边缘一致性的函数
def calculate_edge_consistency(imageA, imageB):
    edgesA = cv2.Canny(imageA, 100, 200)
    edgesB = cv2.Canny(imageB, 100, 200)
    consistency = np.mean(np.abs(edgesA - edgesB))
    return consistency


# 定义计算直方图相似度的函数
def calculate_histogram_similarity(imageA, imageB):
    histA = cv2.calcHist([imageA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histB = cv2.calcHist([imageB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histA = cv2.normalize(histA, histA).flatten()
    histB = cv2.normalize(histB, histB).flatten()
    similarity = cv2.compareHist(histA, histB, cv2.HISTCMP_INTERSECT)
    return similarity


# 计算统计值（均值、标准差、95%置信区间）
def calculate_statistics(values):
    mean_val = np.mean(values)
    std_val = np.std(values)
    ci95_val = stats.norm.interval(0.95, loc=mean_val, scale=std_val / np.sqrt(len(values)))
    return mean_val, std_val, ci95_val


def is_nan_judge(value):
    return np.isnan(value)


def load_and_normalize_image(file_path):
    # 仅仅进行灰度值归一化发现gt 和pred的值还是有数量级差距
    """Load an image, convert to grayscale, normalize, and return as numpy array."""
    with Image.open(file_path) as img:
        img = (np.array(img.convert('L'), dtype=np.float32)) / 255  # / 255.0
        if img.shape != (320, 320):
            img = cv2.resize(img, (320, 320))
        return img


def load_and_normalize_image_serv(file_path):
    # 仅仅进行灰度值归一化发现gt 和pred的值还是有数量级差距
    """Load an image, convert to grayscale, normalize, and return as numpy array."""
    with Image.open(file_path) as img:
        original_img = np.array(img)
        normalized_img = (original_img - np.min(original_img)) / (np.max(original_img) - np.min(original_img))
        # img = np.array(img.convert('L'), dtype=np.float32) / 255.0 # / 255.0
        if normalized_img.shape != (320, 320):
            normalized_img = cv2.resize(normalized_img, (320, 320))
        return normalized_img


def load_image_color(file_path):
    """Load an image in full color."""
    with Image.open(file_path) as img:
        img = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
        if img.shape != (320, 320, 3):
            img = cv2.resize(img, (320, 320))
        return img
# 调整大小为 320x320
def resize_image(image, size=(320, 320)):
    """
    调整图像大小。
    :param image: 输入图像，类型为 numpy array
    :param size: 目标大小，默认 (320, 320)
    :return: 调整大小后的图像
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)



def load_matched_images(tgt_folder, pred_folder, start, num_images, prefix):


    """Load images based on filenames in the tgt_folder."""
    tgt_files = sorted([f for f in os.listdir(tgt_folder) if f.endswith(".png")])[start:num_images]
    ssim_scores = []
    edge_consistency_scores = []
    hist_similarity_scores = []
    pred_depth = []
    tgt_images = []
    for index, tgt_file in zip(range(len(tgt_files)), tgt_files):
        base_name = tgt_file.split('.')[0]
        pred_file = f"{base_name}{prefix}.png"  # 必要时需要修改
        tgt_image_path = os.path.join(tgt_folder, tgt_file)
        pred_image_path = os.path.join(pred_folder, pred_file)

        if os.path.exists(pred_image_path) and os.path.exists(tgt_image_path):
            tgt_images.append(load_image_color(tgt_image_path))
            pred_depth.append(load_and_normalize_image(pred_image_path))
        elif not os.path.exists(pred_image_path):
            print(f"{pred_image_path}file does not exist")
        elif not os.path.exists(tgt_image_path):
            print(f"{tgt_image_path}file does not exist")

        if os.path.exists(pred_image_path):
            imageA = cv2.imread(tgt_image_path)
            imageB = cv2.imread(pred_image_path)
            imageB = resize_image(imageB)  # 调整大小为 320x320
            # 计算三个指标
            ssim_score = calculate_ssim(imageA, imageB)
            ssim_scores.append(ssim_score)

            edge_consistency_score = calculate_edge_consistency(imageA, imageB)
            edge_consistency_scores.append(edge_consistency_score)

            hist_similarity_score = calculate_histogram_similarity(imageA, imageB)
            hist_similarity_scores.append(hist_similarity_score)

        # 计算统计值
        ssim_mean, ssim_std, ssim_ci95 = calculate_statistics(ssim_scores)
        edge_mean, edge_std, edge_ci95 = calculate_statistics(edge_consistency_scores)
        hist_mean, hist_std, hist_ci95 = calculate_statistics(hist_similarity_scores)
        # 创建一个DataFrame并保存到Excel
        data = {
            'Model': [param["model_name"]],
            'SSIM Mean': [ssim_mean],
            'SSIM Std': [ssim_std],
            'SSIM 95% CI Lower': [ssim_ci95[0]],
            'SSIM 95% CI Upper': [ssim_ci95[1]],
            'Edge Consistency Mean': [edge_mean],
            'Edge Consistency Std': [edge_std],
            'Edge Consistency 95% CI Lower': [edge_ci95[0]],
            'Edge Consistency 95% CI Upper': [edge_ci95[1]],
            'Histogram Similarity Mean': [hist_mean],
            'Histogram Similarity Std': [hist_std],
            'Histogram Similarity 95% CI Lower': [hist_ci95[0]],
            'Histogram Similarity 95% CI Upper': [hist_ci95[1]]
        }

    df = pd.DataFrame(data)
    os.makedirs(save_metric_file_path, exist_ok=True)
    save_path = os.path.join(save_metric_file_path, f'{param["model_name"]}.xlsx')
    df.to_excel(save_path, index=False)

    print(f"统计值已保存到 {save_path}")
    return np.array(tgt_images), np.array(pred_depth)


def run_script(tgt_img_folder, output_dir, num_images_to_load):
    import os
    os.makedirs(output_dir, exist_ok=True)
    import subprocess
    """--lora-ckpt="ckpt/normal-scene100-notext" --src="/mnt/share/toky/Projects/dmp/test_Images/"  
    --config=config.yaml --num-workers=0"""
    # 配置参数
    ckpt = "ckpt/normal-scene100-notext"
    inference_script = "infer.py"  # py 的路径
    config = "config.yaml"  # 检查点路径
    num_workers = 0
    input_rgb_dir = tgt_img_folder
    output_dir = output_dir

    # 遍历目录下的所有 PNG 图像
    if not os.path.exists(tgt_img_folder):
        raise ValueError(f"Target image folder does not exist: {tgt_img_folder}")

    # png_files = sorted([f for f in os.listdir(tgt_img_folder) if f.endswith(".png")])

    tgt_files = sorted([f for f in os.listdir(tgt_img_folder) if f.endswith(".png")])[start:num_images_to_load]
    if not tgt_files:
        raise ValueError(f"No PNG files found in {tgt_img_folder}")
    # 遍历并调用 inference.py

    command = [
        "python",
        inference_script,
        "--lora-ckpt", ckpt,
        "--config", config,
        "--num-workers", str(num_workers),
        "--src", input_rgb_dir,
        "--output", output_dir,
    ]
    try:
        # 执行命令
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 输出结果
        if result.returncode == 0:
            print(f"Success")
        else:
            print(f"Error processing: {result.stderr}")
    except Exception as e:
        print(f"Failed to process: {str(e)}")


def visualize_and_save_images(tgt_images, gt_images, pred_images, output_folder):
    """Create visualizations of images and save them."""
    os.makedirs(output_folder, exist_ok=True)
    for i in range(tgt_images.shape[0]):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(tgt_images[i], cmap='hot')
        plt.title('Target Image')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(gt_images[i], cmap='hot')
        plt.title('Ground Truth Depth')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(pred_images[i], cmap='hot')
        plt.title('Predicted Depth')
        plt.axis('off')
        plt.suptitle(f'Visualization {i}')
        plt.savefig(os.path.join(output_folder, f"vis_image_{i:04d}.png"))
        plt.close()


def save_results_to_excel(results, file_path, model_name):
    os.makedirs(file_path, exist_ok=True)
    data = {'Model Name': model_name}
    for metric, mean_value in results.items():
        data[f'{metric}_Mean'] = [mean_value]
    df = pd.DataFrame(data)
    df.to_excel(file_path + f"/results-{model_name}.xlsx", index=False)  # 保存到Excel
    print(f"Metrics saved to {file_path}")


if __name__ == '__main__':
    # step1: 加载图像
    start = param["start"]
    num_images_to_load = param["num_images_to_load"]  # Example number, change as needed
    is_visualize = param["is_visualize"]
    prefix = param["prefix"]  # 深度图存的前缀，例如depth_aov_
    input_folder = param['input_folder']
    save_metric_file_path = param["metrics_path"] + param["model_name"]
    output_folder = param["pred_depth_img_path"] + param["model_name"]
    vis_folder = param["vis_path"] + param["model_name"] + "/"
    # step2: 运行脚本
    if param['is_run_script']:
        run_script(input_folder, output_folder, num_images_to_load)
    # step3: 调用匹配函数找到GT、Compute metrics
    if param['is_save_metric']:
        tgt_images, pred_images = load_matched_images(input_folder, output_folder, start, num_images_to_load, prefix)
    # step4: 计算可视化以及保存metric
    if is_visualize:
        visualize_and_save_images(tgt_images, tgt_images, pred_images, vis_folder)
