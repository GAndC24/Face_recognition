from Hyparam_optimizer_MCVGAN import Hyparam_optimizer_MCVGAN
import torch
from Trainer_MCVGAN import Trainer_MCVGAN
from Model_MCVGAN import Masked_ConViT_GAN_Generator, Masked_ConViT_GAN_Discriminator

# 超参数
img_size = 128
NP = 60
G = 20
select_ratio = 0.8
L = 18
Pc = 0.8
Pm = 0.05
train_mini_epochs = 1
epochs = 1000

if __name__ == '__main__':
    # 初始化 Hyperparameter optimizer
    Hyparam_optimizer = Hyparam_optimizer_MCVGAN(img_size=img_size, NP=NP, select_ratio=select_ratio, G=G, L=L,
                                       Pc=Pc, Pm=Pm, train_mini_epochs=train_mini_epochs)

    # 获取 best Hyperparameters
    Hyparam_best = Hyparam_optimizer.get_best_hyperparameters()

    lr = Hyparam_best[0]
    warmup_proportion = Hyparam_best[1]
    weight_decay = Hyparam_best[2]
    batch_size = Hyparam_best[3]
    embed_dim = Hyparam_best[4]
    depth = Hyparam_best[5]
    num_heads = Hyparam_best[6]
    mlp_ratio = Hyparam_best[7]
    drop_rate = Hyparam_best[8]
    attn_drop_rate = Hyparam_best[9]
    drop_path_rate = Hyparam_best[10]
    local_up_to_layer = Hyparam_best[11]
    locality_strength = Hyparam_best[12]
    decoder_embed_dim = Hyparam_best[13]
    decoder_depth = Hyparam_best[14]
    decoder_num_heads = Hyparam_best[15]
    filter_size = Hyparam_best[16]
    num_filters = Hyparam_best[17]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 cuda

    # 初始化 generator
    generator = Masked_ConViT_GAN_Generator(
        img_size=img_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        local_up_to_layer=local_up_to_layer,
        locality_strength=locality_strength,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads
    ).to(device)

    # 初始化 discriminator
    discriminator = Masked_ConViT_GAN_Discriminator(
        img_size=img_size,
        filter_size=filter_size,
        num_filters=num_filters
    ).to(device)

    # 初始化 trainer
    trainer = Trainer_MCVGAN(
        generator=generator,
        discriminator=discriminator,
        lr=lr,
        warmup_proportion=warmup_proportion,
        weight_decay=weight_decay,
        batch_size=batch_size,
        img_size=img_size,
        epochs=epochs
    )

    trainer.train()
    trainer.save_generator()
    trainer.save_MCVAE()






