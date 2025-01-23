from HP_optimizer_MCVGAN import HP_optimizer_MCVGAN

# 超参数
img_size = 128
NP = 60
G = 20
select_ratio = 0.8
L = 18
Pc = 0.8
Pm = 0.05
train_mini_epochs = 20
epochs = 500

if __name__ == '__main__':
    # 初始化 Hyperparameter optimizer
    HP_optimizer = HP_optimizer_MCVGAN(img_size=img_size, NP=NP, select_ratio=select_ratio, G=G, L=L,
                                       Pc=Pc, Pm=Pm, train_mini_epochs=train_mini_epochs)

    HP_best = HP_optimizer.get_best_hyperparameters()






