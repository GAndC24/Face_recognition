from Hyparam_optimizer_MCVGAN import Hyparam_optimizer_MCVGAN
import torch
from Trainer_MCVGAN import Trainer_MCVGAN
from Model_MCVGAN import Masked_ConViT_GAN_Generator, Masked_ConViT_GAN_Discriminator

# 超参数
img_size = 128
NP = 15
G = 10
select_ratio = 0.8
L = 18
Pc = 0.8
Pm = 0.05
train_mini_epochs = 3
epochs = 1000

if __name__ == '__main__':
    # 两点交叉
    # 初始化 Hyperparameter optimizer
    Hyparam_optimizer = Hyparam_optimizer_MCVGAN(img_size=img_size, NP=NP, select_ratio=select_ratio, G=G, L=L,
                                                 Pc=Pc, Pm=Pm, train_mini_epochs=train_mini_epochs)

    # 获取 best Hyperparameters
    Hyparam_best = Hyparam_optimizer.get_best_hyperparameters()


