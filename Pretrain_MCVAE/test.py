import torch
from Trainer_MCVGAN import Trainer_MCVGAN
from Model_MCVGAN import Masked_ConViT_GAN_Generator, Masked_ConViT_GAN_Discriminator

train_mini_epochs = 20
img_size = 128
lr = 0.000003989694086403787
weight_decay = 0.000009529054823440733
warmup_proportion = 0.000035623267961890455
batch_size = 32
embed_dim = 512
depth = 10
num_heads = 16
mlp_ratio = 2.0
drop_rate = 0.5
attn_drop_rate = 0.1
drop_path_rate = 0.3
local_up_to_layer = 10
locality_strength = 1.5
decoder_embed_dim = 768
decoder_depth = 4
decoder_num_heads = 8
filter_size = 3
num_filters = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # 使用 cuda
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
discriminator = Masked_ConViT_GAN_Discriminator(
                    img_size=img_size,
                    filter_size=filter_size,
                    num_filters=num_filters
).to(device)
trainer = Trainer_MCVGAN(
                    generator=generator,
                    discriminator=discriminator,
                    lr=lr,
                    warmup_proportion=warmup_proportion,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    img_size=img_size,
                    epochs=train_mini_epochs
)

if __name__ == '__main__':
    trainer.train_HP_optim(0)

