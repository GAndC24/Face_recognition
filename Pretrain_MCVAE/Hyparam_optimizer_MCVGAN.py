import numpy as np
import matplotlib.pyplot as plt
from Pretrain_MCVAE.Trainer_MCVGAN import Trainer_MCVGAN
from Model_MCVGAN import *
from datetime import datetime


class Hyparam_optimizer_MCVGAN():
    '''MAE 超参数优化器（基于遗传算法）'''
    def __init__(self, img_size=128, NP=60, select_ratio=0.8, L=18, G=20, Pc=0.8, Pm=0.05, train_mini_epochs=20):
        '''
        初始化超参数优化器
        :param NP: 种群数目
        :param select_ratio: 每一代选择比例
        :param L: 染色体长度
        :param G: 进化代数
        :param Pc: 交叉概率
        :param Pm: 变异概率
        :param train_mini_epochs: 训练迭代次数

        染色体:
        0: lr = [1e-6, 1e-3)
        1: warmup_proportion = [1e-5, 1e-2)
        2: weight_decay = [1e-6, 1e-3)
        3: batch_size = (16, 32)
        4: embed_dim = (256, 512, 768, 1024)
        5: depth = [6, 48]
        6: num_heads = (8, 16, 32)
        7: mlp_ratio = (2, 4, 8)
        8: drop_rate = (0.1, 0.2, 0.3, 0.4, 0.5)
        9: attn_drop_rate = (0.1, 0.2, 0.3)
        10: drop_path_rate = (0.1, 0.2, 0.3)
        11: local_up_to_layer = (6, 8, 10, 12)
        12: locality_strength = (0.5, 1.0, 1.5)
        13: decoder_embed_dim = (256, 512, 768)
        14: decoder_depth = (4, 8, 12)
        15: decoder_num_heads = (4, 8, 16)
        16: filter_size = (3, 5, 7)
        17: num_filters = (32, 64, 128)
        '''
        self.img_size = img_size
        self.NP = NP
        self.select_ratio = select_ratio
        self.L = L
        self.G = G
        self.Pc = Pc
        self.Pm = Pm
        self.train_mini_epochs = train_mini_epochs

        self.initialize_population()

    def initialize_population(self):
        '''
        初始化种群
        '''
        # 初始化种群
        self.population = np.zeros((self.NP, self.L))
        for i in range(self.NP):
            self.population[i, 0] = np.random.uniform(1E-6, 1E-3)  # lr
            self.population[i, 1] = np.random.uniform(1E-5, 1E-2)  # warmup_proportion
            self.population[i, 2] = np.random.uniform(1E-6, 1E-3)    # weight_decay
            self.population[i, 3] = np.random.choice([16, 32], size=1)  # batch_size
            self.population[i, 4] = np.random.choice([256, 512, 768, 1024], size=1)  # embed_dim
            self.population[i, 5] = np.random.randint(6, 49)  # depth [6, 48]
            self.population[i, 6] = np.random.choice([8, 16, 32], size=1)  # num_heads
            self.population[i, 7] = np.random.choice([2, 4, 8], size=1)  # mlp_ratio
            self.population[i, 8] = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5], size=1)  # drop_rate
            self.population[i, 9] = np.random.choice([0.1, 0.2, 0.3], size=1)  # attn_drop_rate
            self.population[i, 10] = np.random.choice([0.1, 0.2, 0.3], size=1)  # drop_path_rate
            self.population[i, 11] = np.random.choice([6, 8, 10, 12], size=1)  # local_up_to_layer
            self.population[i, 12] = np.random.choice([0.5, 1.0, 1.5], size=1)  # locality_strength
            self.population[i, 13] = np.random.choice([256, 512, 768], size=1)  # decoder_embed_dim
            self.population[i, 14] = np.random.choice([4, 8, 12], size=1)  # decoder_depth
            self.population[i, 15] = np.random.choice([4, 8, 16], size=1)  # decoder_num_heads
            self.population[i, 16] = np.random.choice([3, 5], size=1)  # filter_size
            self.population[i, 17] = np.random.choice([32, 64, 128], size=1)  # num_filters

    def get_best_hyperparameters(self):
        '''
        获取最优超参数

        :return: 最优超参数组合
        '''
        # 日志记录
        with open("pretrain_log.txt", "a") as f:
            current_time = datetime.now()  # 获取当前日期和时间
            f.write(f"--------------------Start Hyperparameter optimize--------------------\n"
                    f"Start Time : {current_time}\n"
            )

        average_fitness_list = []  # 平均适应度列表
        best_fitness_list = []  # 最优适应度列表
        best_fitness = np.inf  # 最优适应度值
        x_best = None  # 最优个体
        count_gen = 0   # 记录优化代数
        self.fitness = np.zeros(self.NP)  # 种群适应度值
        for i in range(self.NP):
            self.fitness[i] = np.inf

        # 进化迭代
        for gen in range(self.G):
            # 日志记录
            with open("pretrain_log.txt", "a") as f:
                current_time = datetime.now()
                f.write(f"\nGeneration: {gen + 1}\n"
                        f"Start Time : {current_time}\n"
                )

            # 计算适应度值

            # deep learning
            for i in range(self.NP):
                # 判断是否已经计算过适应度值
                if self.fitness[i] < np.inf:
                    continue

                lr = self.population[i, 0]
                warmup_proportion = self.population[i, 1]
                weight_decay = self.population[i, 2]
                batch_size = int(self.population[i, 3])
                embed_dim = int(self.population[i, 4])
                depth = int(self.population[i, 5])
                num_heads = int(self.population[i, 6])
                mlp_ratio = self.population[i, 7]
                drop_rate = self.population[i, 8]
                attn_drop_rate = self.population[i, 9]
                drop_path_rate = self.population[i, 10]
                local_up_to_layer = int(self.population[i, 11])
                locality_strength = self.population[i, 12]
                decoder_embed_dim = int(self.population[i, 13])
                decoder_depth = int(self.population[i, 14])
                decoder_num_heads = int(self.population[i, 15])
                filter_size = int(self.population[i, 16])
                num_filters = int(self.population[i, 17])

                torch.cuda.empty_cache()        # 清除 GPU 显存缓存
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # 使用 cuda
                generator = Masked_ConViT_GAN_Generator(
                    img_size=self.img_size,
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
                    img_size=self.img_size,
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
                    img_size=self.img_size,
                    epochs=self.train_mini_epochs
                )

                fid = trainer.train_HP_optim(i)
                # 记录适应度值
                self.fitness[i] = fid        # 以 FID score 作为适应度值

            # 记录平均适应度值
            average_fitness = np.mean(self.fitness)
            average_fitness_list.append(average_fitness)

            # 计算当代最优适应度值并记录
            index = np.argmin(self.fitness)      # 最小适应度值为最优
            current_x_best = self.population[index]      # 当代最优个体
            current_best_fitness = self.fitness[index].item()        # 当代最优适应度值
            best_fitness_list.append(current_best_fitness)      # 记录当代最优适应度值

            # 记录日志：记录每一代平均适应度值、最优适应度值
            with open("pretrain_log.txt", "a") as f:
                f.write(f"\nAverage Fitness: {average_fitness:.4f}\n")
                f.write(f"Best Fitness: {current_best_fitness:.4f}\n")

            # 更新全局最优
            if current_best_fitness < best_fitness:
                x_best = current_x_best
                best_fitness = current_best_fitness

            # 轮盘赌选择
            self.roulette_wheel_selection()

            # # 单点交叉
            # self.single_point_crossover()

            # 两点交叉
            self.two_point_crossover()

            # # 均匀交叉
            # self.uniform_crossover()

            # 变异
            self.mutation()

            # 精英策略：将最优个体加入新种群
            reshaped_current_x_best = current_x_best.reshape(1, self.L)
            new_population = np.append(self.population, reshaped_current_x_best, axis=0)
            # 更新适应度值列表
            self.fitness = np.append(self.fitness, current_best_fitness)
            # 更新种群数目
            self.NP = new_population.shape[0]

            # 更新种群
            self.population = new_population

            # 更新优化代数
            count_gen += 1

        # 输出结果

        # 将结果写入到文件中进行记录
        with open("pretrain_log.txt", "a") as f:
            f.write(f"\n最优适应度值: {best_fitness}\n"
                    "最优超参数:\n"
                    f"lr = {x_best[0]}\n"
                    f"warmup_proportion = {x_best[1]}\n"
                    f"weight_decay = {x_best[2]}\n"
                    f"batch_size = {x_best[3]}\n"
                    f"embed_dim = {x_best[4]}\n"
                    f"depth = {x_best[5]}\n"
                    f"num_heads = {x_best[6]}\n"
                    f"mlp_ratio = {x_best[7]}\n"
                    f"drop_rate = {x_best[8]}\n"
                    f"attn_drop_rate = {x_best[9]}\n"
                    f"drop_path_rate = {x_best[10]}\n"
                    f"local_up_to_layer = {x_best[11]}\n"
                    f"locality_strength = {x_best[12]}\n"
                    f"decoder_embed_dim = {x_best[13]}\n"
                    f"decoder_depth = {x_best[14]}\n"
                    f"decoder_num_heads = {x_best[15]}\n"
                    f"filter_size = {x_best[16]}\n"
                    f"num_filters = {x_best[17]}\n"
                    f"--------------------End--------------------\n"
            )

        print(f"Best Fitness: {best_fitness}\n"
              "Best Hyperparameters:\n"
              f"lr = {x_best[0]}\n"
              f"warmup_proportion = {x_best[1]}\n"
              f"weight_decay = {x_best[2]}\n"
              f"batch_size = {x_best[3]}\n"
              f"embed_dim = {x_best[4]}\n"
              f"depth = {x_best[5]}\n"
              f"num_heads = {x_best[6]}\n"
              f"mlp_ratio = {x_best[7]}\n"
              f"drop_rate = {x_best[8]}\n"
              f"attn_drop_rate = {x_best[9]}\n"
              f"drop_path_rate = {x_best[10]}\n"
              f"local_up_to_layer = {x_best[11]}\n"
              f"locality_strength = {x_best[12]}\n"
              f"decoder_embed_dim = {x_best[13]}\n"
              f"decoder_depth = {x_best[14]}\n"
              f"decoder_num_heads = {x_best[15]}\n"
              f"filter_size = {x_best[16]}\n"
              f"num_filters = {x_best[17]}\n"
        )

        # 绘制适应度曲线
        x = np.arange(start=1, stop=count_gen + 1)
        plt.plot(x, best_fitness_list, label='best', markevery=2)
        plt.plot(x, average_fitness_list, label='average', markevery=2)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

        return x_best

    def roulette_wheel_selection(self):
        '''
        轮盘赌选择（同步更新选择后种群适应度值）
        '''
        # 归一化
        max_fitness = np.max(self.fitness)
        min_fitness = np.min(self.fitness)
        fitness_norm = (max_fitness - self.fitness) / (max_fitness - min_fitness)        # 归一化后的种群适应度值

        # 计算选择概率
        P = fitness_norm / np.sum(fitness_norm)
        P = P.flatten()     # 展平为一维

        # 选择
        selected_indices = np.random.choice(np.arange(self.NP), size=int(self.NP * self.select_ratio), replace=False, p=P)
        selected_individuals = self.population[selected_indices]

        # 更新种群
        self.population = selected_individuals

        # 更新种群数目
        self.NP = self.population.shape[0]

        # 更新种群适应度值
        self.fitness = self.fitness[selected_indices]

    def single_point_crossover(self):
        '''
        单点交叉
        '''
        selected_crossover_indices = np.random.choice(np.arange(self.NP), size=int(self.NP // 2 * self.Pc), replace=False)      # 被选中进行交叉的个体索引

        # 交叉
        for i in range(0, len(selected_crossover_indices), 2):
            # 随机选择交叉点
            point = np.random.randint(1, self.L)

            # 交叉
            offspring1 = self.population[selected_crossover_indices[i], point:]     # 子代个体 1
            offspring2 = self.population[selected_crossover_indices[i + 1], point:]     # 子代个体 2
            self.population[selected_crossover_indices[i], point:], self.population[selected_crossover_indices[i + 1], point:] = offspring2, offspring1

            # 更新适应度值
            self.fitness[selected_crossover_indices[i]] = self.fitness[selected_crossover_indices[i + 1]] = np.inf

    def two_point_crossover(self):
        '''
        两点交叉
        '''
        selected_crossover_indices = np.random.choice(np.arange(self.NP), size=int(self.NP // 2 * self.Pc),replace=False)       # 被选中进行交叉的个体索引

        # 交叉
        for i in range(0, len(selected_crossover_indices), 2):
            # 随机选择交叉点
            point_start = np.random.randint(1, self.L)
            point_end = np.random.randint(point_start, self.L)

            # 交叉
            offspring1 = self.population[selected_crossover_indices[i], point_start:point_end]      # 子代个体 1
            offspring2 = self.population[selected_crossover_indices[i + 1], point_start:point_end]      # 子代个体 2
            self.population[selected_crossover_indices[i], point_start:point_end], self.population[selected_crossover_indices[i + 1], point_start:point_end] = offspring2, offspring1

            # 更新适应度值
            self.fitness[selected_crossover_indices[i]] = self.fitness[selected_crossover_indices[i + 1]] = np.inf

    def uniform_crossover(self, P=0.5):
        '''
        均匀交叉

        Args:
            :param: P: 随机从两个父代个体选择基因的概率
        '''
        selected_crossover_indices = np.random.choice(np.arange(self.NP), size=int(self.NP // 2 * self.Pc), replace=False)      # 被选中进行交叉的个体索引

        for i in range(0, len(selected_crossover_indices), 2):
            offspring1 = np.zeros((1, self.L))      # 子代个体 1
            offspring2 = np.zeros((1, self.L))      # 子代个体 2
            for point in range(self.L):
                # 从父代个体 1 中获取基因
                if np.random.rand() < P:
                    offspring1[0, point] = self.population[selected_crossover_indices[i], point]
                    offspring2[0, point] = self.population[selected_crossover_indices[i], point]
                # 从父代个体 2 中获取基因
                else:
                    offspring1[0, point] = self.population[selected_crossover_indices[i + 1], point]
                    offspring2[0, point] = self.population[selected_crossover_indices[i + 1], point]

            self.population[selected_crossover_indices[i]], self.population[selected_crossover_indices[i + 1]] = offspring2, offspring1

            # 更新适应度值
            self.fitness[selected_crossover_indices[i]] = self.fitness[selected_crossover_indices[i + 1]] = np.inf

    def mutation(self):
        '''
        变异
        '''
        selected_mutation_indices = np.random.choice(np.arange(self.NP), size=int(self.NP * self.Pm), replace=False)      # 被选中进行变异的个体索引

        for i in range(len(selected_mutation_indices)):
            # 随机选择变异位
            point = np.random.randint(0, self.L)

            # 变异
            if point == 0:  # lr 变异
                self.population[selected_mutation_indices[i], point] = np.random.uniform(1E-6, 1E-3)
            elif point == 1:  # warmup_proportion 变异
                self.population[selected_mutation_indices[i], point] = np.random.uniform(1E-5, 1E-2)
            elif point == 2:  # weight_decay 变异
                self.population[selected_mutation_indices[i], point] = np.random.uniform(1E-6, 1E-3)
            elif point == 3:  # batch_size 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([16, 32], size=1)
            elif point == 4:  # embed_dim 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([256, 512, 768, 1024], size=1)
            elif point == 5:  # depth 变异
                self.population[selected_mutation_indices[i], point] = np.random.randint(6, 49)
            elif point == 6:  # num_heads 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([8, 16, 32], size=1)
            elif point == 7:  # mlp_ratio 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([2, 4, 8], size=1)
            elif point == 8:  # drop_rate 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5], size=1)
            elif point == 9:  # attn_drop_rate 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([0.1, 0.2, 0.3], size=1)
            elif point == 10:  # drop_path_rate 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([0.1, 0.2, 0.3], size=1)
            elif point == 11:  # local_up_to_layer 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([6, 8, 10, 12], size=1)
            elif point == 12:  # locality_strength 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([0.5, 1.0, 1.5], size=1)
            elif point == 13:  # decoder_embed_dim 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([256, 512, 768], size=1)
            elif point == 14:  # decoder_depth 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([4, 8, 12], size=1)
            elif point == 15:  # decoder_num_heads 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([4, 8, 16], size=1)
            elif point == 16:  # filter_size 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([3, 5, 7], size=1)
            elif point == 17:  # num_filters 变异
                self.population[selected_mutation_indices[i], point] = np.random.choice([32, 64, 128], size=1)

            # 更新适应度值缓存
            self.fitness[selected_mutation_indices[i]] = np.inf



