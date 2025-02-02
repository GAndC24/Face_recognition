import torch
import torch.nn as nn
import torch.optim as optim
from FaceMask_Dataset import FaceMask_Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Trainer_MCV():
    def __init__(self, model, lr, warmup_proportion, weight_decay, batch_size, img_size, epochs=15):
        '''
        初始化训练器

        Args:
            :param model: 模型
            :param lr: 学习率
            :param warmup_proportion: 学习率预热比例
            :param weight_decay: 学习率衰减系数
            :param batch_size: 批次大小
            :param img_size: 图像大小
            :param train_mini_epochs: mini_train 迭代次数
            :param epochs: train 迭代次数
        '''
        self.model = model
        self.lr = lr
        self.warmup_proportion = warmup_proportion
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.img_size = img_size
        self.epochs = epochs
        self.num_warmup_epochs = int(self.warmup_proportion * self.epochs)  # 预热轮数

        # 定义数据预处理
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),  # 调整图像大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # # 定义数据增强
        # transform_extend = transforms.Compose([
        #     transforms.Resize(img_size),  # 调整图像大小为 224 x 224
        #     transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # 图像标准化
        # ])

        # # 定义数据预处理(label)
        # transform_label = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((224, 224)),  # 调整图像大小为 224 x 224
        # ])

        # 加载数据集
        train_dataset = FaceMask_Dataset(img_dir="FaceMask/train", transform=transform)
        mini_train_dataset = FaceMask_Dataset(img_dir='FaceMask/train_mini', transform=transform)
        # train_dataset_extend = FaceMask_Dataset(img_dir='FaceMask/train', transform=transform_extend)
        # train_dataset += train_dataset_extend
        validation_dataset = FaceMask_Dataset(img_dir='FaceMask/validation', transform=transform)
        mini_validation_dataset = FaceMask_Dataset(img_dir='FaceMask/validation_mini', transform=transform)
        test_dataset = FaceMask_Dataset(img_dir='FaceMask/test', transform=transform)

        # 数据加载
        self.mini_train_loader = DataLoader(dataset=mini_train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(dataset=validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.mini_validation_loader = DataLoader(dataset=mini_validation_dataset, batch_size=self.batch_size,shuffle=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        # 使用 cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.model = self.model.to(self.device)

        # 定义损失函数
        self.loss_func = nn.BCELoss()

        # 定义优化器
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 定义学习率预热调度器
        self.lr_warmup_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self.lr_lambda
        )

        # 定义学习率衰减调度器
        self.lr_decay_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=self.weight_decay
        )

    # 学习率调整函数(预热)
    def lr_lambda(self, current_epoch):
        '''
        学习率调整函数(预热)
        :param current_epoch: 当前训练轮数
        :return:
        '''
        # 初期线性预热，从 0 开始到学习率
        if current_epoch < self.num_warmup_epochs:
            return float(current_epoch) / float(max(1, self.num_warmup_epochs))
        else:
            return 1  # 预热结束后，返回学习率不变

    # 训练(超参数优化)
    def train_HP_optim(self, index):
        '''
        训练
        :param index: 种群个体序号

        :return: 最终 Accuracy
        '''
        # 记录日志
        with open("train_log.txt", "a") as f:
            current_time = datetime.now()
            f.write(f"\nIndex: {index + 1}\n"
                    f"\n--------------------Start train-------------------\n"
                    f"Start time: {current_time}\n\n"
                    )
        print(
            f"\nIndex: {index + 1}\n"
            f"img_size : {self.img_size}\n"
            f"lr : {self.lr}\n"
            f"weight_decay : {self.weight_decay}\n"
            f"warmup_proportion : {self.warmup_proportion}\n"
            f"batch_size : {self.batch_size}\n"
            f"hidden_dim : {self.model.hidden_dim}\n"
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for step, (X, y) in enumerate(self.mini_train_loader):
                X, y = X.to(self.device), y.to(self.device)
                outputs = torch.sigmoid(self.model(X))
                loss = self.loss_func(outputs, y)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_warmup_scheduler.step()
                self.lr_decay_scheduler.step()

                print(f"Epoch: {epoch}, Step: {step + 1}, Loss: {loss.item():.4f}, ")

            average_loss = epoch_loss / len(self.mini_train_loader)     # 每个 epoch 的平均 loss

            train_acc = self.get_accuracy(self.model, self.mini_train_loader)  # mini 训练集准确率
            val_acc = self.get_accuracy(self.model, self.validation_loader)  # 验证集准确率

            # 记录日志
            with open("train_log.txt", "a") as f:
                f.write(f"Epoch: {epoch}, Loss: {average_loss:.4f}, Train acc: {train_acc * 100:.2f}%, Val acc: {val_acc * 100:.2f}%\n")

            print(f"Epoch: {epoch}, Loss: {average_loss:.4f}, Train acc: {train_acc * 100:.2f}%, Val acc: {val_acc * 100:.2f}%\n")

        final_acc = self.get_accuracy(self.model, self.validation_loader)  # 最终 accuracy
        print(f"Final accuracy: {final_acc * 100:.2f}")

        # 记录日志
        with open("train_log.txt", "a") as f:
            current_time = datetime.now()
            f.write(f"\nresult:\n"
                    f"Final accuracy: {final_acc:.4f}\n"
                    f"\nHyperparameters: \n"
                    f"lr : {self.lr}\n"
                    f"weight_decay : {self.weight_decay}\n"
                    f"warmup_proportion : {self.warmup_proportion}\n"
                    f"batch_size : {self.batch_size}\n"
                    f"hidden_dim : {self.model.hidden_dim}\n"
                    f"\nEnd time: {current_time}\n"
                    f"--------------------End--------------------\n"
                    )

        return final_acc

    # 获取准确率
    def get_accuracy(self, model, data_loader):
        '''
        获取准确率
        Args:
            :param model:
            :param data_loader:

        :return: 准确率
        '''
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)

                outputs = torch.sigmoid(model(X))
                predicted = (outputs > 0.5).float()

                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = correct / total  # 计算准确率

        return accuracy

    # 训练
    def train(self):
        '''
        训练
        '''
        # 记录日志
        with open("train_log.txt", "a") as f:
            current_time = datetime.now()
            f.write(f"\n--------------------Start train-------------------\n"
                    f"Start time: {current_time}\n"
                    f"\nHyperparameters:\n"
                    f"lr : {self.lr}\n"
                    f"weight_decay : {self.weight_decay}\n"
                    f"warmup_proportion : {self.warmup_proportion}\n"
                    f"batch_size : {self.batch_size}\n"
                    f"hidden_dim : {self.model.hidden_dim}\n"
                    )

        print(f"lr : {self.lr}\n"
              f"weight_decay : {self.weight_decay}\n"
              f"warmup_proportion : {self.warmup_proportion}\n"
              f"batch_size : {self.batch_size}\n"
              f"hidden_dim : {self.model.hidden_dim}\n"
        )

        train_acc_list = []
        test_acc_list = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for step, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                outputs = torch.sigmoid(self.model(X))
                loss = self.loss_func(outputs, y)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_warmup_scheduler.step()
                self.lr_decay_scheduler.step()

                print(f"Epoch: {epoch}, Step: {step + 1}, Loss: {loss.item():.4f}, ")

            average_loss = epoch_loss / len(self.train_loader)  # 每个 epoch 的平均 loss

            train_acc = self.get_accuracy(self.model, self.train_loader)  # 训练集准确率
            test_acc = self.get_accuracy(self.model, self.test_loader)  # 测试集准确率
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            # 记录日志
            with open("train_log.txt", "a") as f:
                f.write(f"Epoch: {epoch}, Loss: {average_loss:.4f}, Train acc: {train_acc * 100:.2f}%, Test acc: {test_acc * 100:.2f}%\n")

            print(f"Epoch: {epoch}, Loss: {average_loss:.4f}, Train acc: {train_acc * 100:.2f}%, Test acc: {test_acc * 100:.2f}%\n")

        final_acc = self.get_accuracy(self.model, self.test_loader)  # 最终 accuracy
        print(f"Final accuracy: {final_acc * 100:.2f}")

        # 记录日志
        with open("train_log.txt", "a") as f:
            current_time = datetime.now()
            f.write(f"\nresult:\n"
                    f"Final accuracy: {final_acc:.4f}\n"
                    f"\nHyperparameters: \n"
                    f"lr : {self.lr}\n"
                    f"weight_decay : {self.weight_decay}\n"
                    f"warmup_proportion : {self.warmup_proportion}\n"
                    f"batch_size : {self.batch_size}\n"
                    f"hidden_dim : {self.model.hidden_dim}\n"
                    f"\nEnd time: {current_time}\n"
                    f"--------------------End--------------------\n"
                    )

        # 绘制 accuracy 曲线
        x = np.arange(self.epochs)
        plt.plot(x, train_acc_list, label='train', markevery=2)
        plt.plot(x, test_acc_list, label='test', markevery=2)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='lower right')
        plt.show()

    # 保存 model 参数
    def save_model(self):
        '''
        保存 model 模型参数
        '''
        torch.save(self.model.state_dict(), "models/best_model.pth")
        file_path = "models/best_model.pth"
        print(f"Model parameters saved to {file_path}")



