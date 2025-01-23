import numpy as np
import matplotlib.pyplot as plt
import torch

# 显示图像
def show_image(image, title=''):
    '''
    显示图像

    Args:
        :param image: 图像数据 [H, W, 3]
        :param title: 图像标题
    '''
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')

