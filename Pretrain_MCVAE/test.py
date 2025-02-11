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
train_mini_epochs = 5
epochs = 1000

if __name__ == '__main__':
    # 两点交叉
    # 初始化 Hyperparameter optimizer
    Hyparam_optimizer = Hyparam_optimizer_MCVGAN(img_size=img_size, NP=NP, select_ratio=select_ratio, G=G, L=L,
                                                 Pc=Pc, Pm=Pm, train_mini_epochs=train_mini_epochs)

    # 获取 best Hyperparameters
    Hyparam_best = Hyparam_optimizer.get_best_hyperparameters()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 cuda
    #
    # generator = Masked_ConViT_GAN_Generator(img_size=img_size, patch_size=16, in_chans=3, num_classes=1, embed_dim=1024,depth=24,
    #                                         num_heads=16, mlp_ratio=8., qkv_bias=False, qk_scale=None, drop_rate=0.5,
    #                                         attn_drop_rate=0.3, drop_path_rate=0.3, local_up_to_layer=22,locality_strength=2.,
    #                                         use_pos_embed=True, decoder_embed_dim=512, decoder_depth=8,decoder_num_heads=16,
    #                                         norm_pix_loss=False).to(device)
    #
    # discriminator = Masked_ConViT_GAN_Discriminator(img_size=img_size, filter_size=7, num_filters=128).to(device)
    #
    # trainer = Trainer_MCVGAN(generator=generator, discriminator=discriminator, img_size=img_size,
    #                          lr=0.0003556041369588326, warmup_proportion=8.414984455744593e-05, weight_decay=0.0005195866419309716,
    #                          batch_size=128, epochs=train_mini_epochs,)
    # trainer.train_HP_optim(0)


