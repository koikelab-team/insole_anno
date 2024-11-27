from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import io

def plot_insole_heatmap_gif(sensor_values_y, sensor_values_output, output_path):
    """
    Create a heatmap GIF comparing ground truth (y) and predicted (output) sensor values.

    Parameters:
    - insole_image: Base insole image (PIL.Image object).
    - sensor_values_y: Ground truth sensor values, shape (frames, sensors).
    - sensor_values_output: Predicted sensor values, shape (frames, sensors).
    - output_path: Path to save the output GIF.
    """
    # Adjusted sensor positions based on the image dimensions (296x666)
    sensor_positions = [
        (159, 598), (95, 604), (163, 512), (89, 515),
        (167, 419), (89, 428), (212, 308), (62, 334),
        (240, 182), (187, 183), (145, 187), (105, 197),
        (61, 207), (223, 71), (167, 79), (102, 92)
    ]

    sensor_values_y = sensor_values_y.cpu().detach().numpy()
    sensor_values_output = sensor_values_output.cpu().detach().numpy()

    # Ensure the image size
    insole_image = Image.open('insole.png')
    width, height = insole_image.size

    sensor_values_y = denormalize_output(sensor_values_y, min_npy='./dataset/insole_min.npy', max_npy='./dataset/insole_max.npy')
    sensor_values_output = denormalize_output(sensor_values_output, min_npy='./dataset/insole_min.npy', max_npy='./dataset/insole_max.npy')
    sensor_values_y = sensor_values_y[0]
    sensor_values_output = sensor_values_output[0]
    
    frames = []
    for frame in range(sensor_values_y.shape[0]):  # Iterate over sequence length
        # Create a composite image for the current frame
        composite_image = Image.new('RGBA', (width * 2, height * 2))
        
        # Ground truth (y) - Left foot
        img_y_left = insole_image.copy()
        draw_y_left = ImageDraw.Draw(img_y_left, "RGBA")
        for i, (x, y) in enumerate(sensor_positions):
            radius = 10
            color = plt.cm.plasma(sensor_values_y[frame, i])  # Left foot: channels 0-15
            color = tuple(int(255 * c) for c in color[:3]) + (int(255 * 0.8),)
            draw_y_left.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        composite_image.paste(img_y_left, (0, 0))

        # Ground truth (y) - Right foot
        img_y_right = insole_image.copy()
        draw_y_right = ImageDraw.Draw(img_y_right, "RGBA")
        for i, (x, y) in enumerate(sensor_positions):
            radius = 10
            color = plt.cm.plasma(sensor_values_y[frame, i + 16])  # Right foot: channels 16-31
            color = tuple(int(255 * c) for c in color[:3]) + (int(255 * 0.8),)
            draw_y_right.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        composite_image.paste(img_y_right.transpose(Image.FLIP_LEFT_RIGHT), (width, 0))

        # Predicted (output) - Left foot
        img_output_left = insole_image.copy()
        draw_output_left = ImageDraw.Draw(img_output_left, "RGBA")
        for i, (x, y) in enumerate(sensor_positions):
            radius = 10
            color = plt.cm.plasma(sensor_values_output[frame, i])  # Left foot: channels 0-15
            color = tuple(int(255 * c) for c in color[:3]) + (int(255 * 0.8),)
            draw_output_left.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        composite_image.paste(img_output_left, (0, height))

        # Predicted (output) - Right foot
        img_output_right = insole_image.copy()
        draw_output_right = ImageDraw.Draw(img_output_right, "RGBA")
        for i, (x, y) in enumerate(sensor_positions):
            radius = 10
            color = plt.cm.plasma(sensor_values_output[frame, i + 16])  # Right foot: channels 16-31
            color = tuple(int(255 * c) for c in color[:3]) + (int(255 * 0.8),)
            draw_output_right.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        composite_image.paste(img_output_right.transpose(Image.FLIP_LEFT_RIGHT), (width, height))

        # Append frame
        frames.append(composite_image)

    # Save frames as a gif
    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=100)

# Function to denormalize the model output
def denormalize_output(output, min_npy, max_npy):
    """
    Denormalize the output using saved min and max values.

    Parameters:
    - output: Normalized model output, shape (bs, seq_len, channel).
    - min_values: Minimum values used during normalization (array).
    - max_values: Maximum values used during normalization (array).

    Returns:
    - Denormalized output (array of same shape as input).
    """
    min_values = np.load(min_npy)
    max_values = np.load(max_npy)
    return (output + 1) / 2 * (max_values - min_values) + min_values

# def log_results(insole, pred, epoch, is_test=False):
#     insole = insole.cpu().detach().numpy()
#     pred = pred.cpu().detach().numpy()
    
#     insole = denormalize_output(insole, min_npy='./dataset/insole_min.npy', max_npy='./dataset/insole_max.npy')
#     pred = denormalize_output(pred, min_npy='./dataset/insole_min.npy', max_npy='./dataset/insole_max.npy')
  
#     insole = insole[0]
#     pred = pred[0]
#     fig, axs = plt.subplots(2, 1)
#     axs[0].imshow(insole.T, aspect='auto')
#     axs[1].imshow(pred.T, aspect='auto')
#     if is_test:
#         plt.savefig(f"results/test_{epoch}.png")
#     else:
#         plt.savefig(f"results/train_{epoch}.png")
#     plt.close()


def log_results(insole, pred, epoch, is_test=False):
    """
    可视化真实值与预测值的时序曲线图。

    Parameters:
    - insole: 真实值 Tensor, shape (batch_size, seq_len, channels)
    - pred: 模型预测值 Tensor, shape (batch_size, seq_len, channels)
    - epoch: 当前 epoch
    - is_test: 是否为测试集结果
    """
    insole = insole.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    
    # 反归一化
    insole = denormalize_output(insole, min_npy='./dataset/insole_min.npy', max_npy='./dataset/insole_max.npy')
    pred = denormalize_output(pred, min_npy='./dataset/insole_min.npy', max_npy='./dataset/insole_max.npy')

    # 取第一个样本进行可视化
    insole = insole[0]  # shape (seq_len, channels)
    pred = pred[0]      # shape (seq_len, channels)

    # 创建曲线图
    fig, axs = plt.subplots(insole.shape[1], 1, figsize=(10, 20), sharex=True, sharey=True)
    fig.suptitle(f"{'Test' if is_test else 'Train'} Results - Epoch {epoch}", fontsize=16)

    for i in range(insole.shape[1]):  # 遍历每个通道
        axs[i].plot(insole[:, i], label='Ground Truth', color='blue', linestyle='-')
        axs[i].plot(pred[:, i], label='Prediction', color='red', linestyle='--')
        axs[i].set_ylabel(f"Channel {i+1}")
        axs[i].legend(loc='upper right')

    axs[-1].set_xlabel("Time Step")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图像
    if is_test:
        plt.savefig(f"results/test_{epoch}.png")
    else:
        plt.savefig(f"results/train_{epoch}.png")
    plt.close()