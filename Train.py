from Hyparam_optimizer_MCV import Hyparam_optimizer_MCV
import torch
from Trainer_MCV import Trainer_MCV
from Model_MCV import Masked_ConViT

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
    Hyparam_optimizer = Hyparam_optimizer_MCV(img_size=img_size, NP=NP, select_ratio=select_ratio, G=G, L=L,
                                       Pc=Pc, Pm=Pm, train_mini_epochs=train_mini_epochs)

    # 获取 best Hyperparameters
    Hyparam_best = Hyparam_optimizer.get_best_hyperparameters()

    lr = Hyparam_best[0]
    warmup_proportion = Hyparam_best[1]
    weight_decay = Hyparam_best[2]
    batch_size = Hyparam_best[3]
    hidden_dim = Hyparam_best[4]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 cuda

    # 初始化 Masked_ConViT
    model = Masked_ConViT(hidden_dim=hidden_dim, pretrained_MCVAE_path="Pretrain_MCVAE/pretrain_models/best_pretrain_MCVAE.pth").to(device)

    # 初始化 Trainer
    trainer = Trainer_MCV(model=model, lr=lr, warmup_proportion=warmup_proportion, weight_decay=weight_decay, batch_size=batch_size, img_size=img_size, epochs=epochs)

    # 训练
    trainer.train()
    trainer.save_model()