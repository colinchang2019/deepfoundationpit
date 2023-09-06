# -*- encoding: utf-8 -*-
"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn, optim
import torch
from config import config

from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
from model4 import Informer4 as DFPTransformer
from utils.earlyStopping import EarlyStopping
from utils.dataset import TimeSeriesDataset

LR = 0.00001  # 0.0001
STEP_SIZE = 2000  # 20000
GAMMA = 0.7

# 固定随机数种子，保证结果可重复


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    # cudnn.benchmark = False
    # cudnn.enabled = False

setup_seed(2019)

def _init_fn(worker_id):
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)


import logging

# 1.显示创建
logging.basicConfig(filename='logger.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# 2.定义logger,设定setLevel，FileHandler，setFormatter
logger = logging.getLogger(__name__)  # 定义一次就可以，其他地方需要调用logger,只需要直接使用logger就行了
logger.setLevel(level=logging.INFO)  # 定义过滤级别

pathlog = "log_train.txt"
filehandler = logging.FileHandler(pathlog)  # Handler用于将日志记录发送至合适的目的地，如文件、终端等
filehandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)

console = logging.StreamHandler()  # 日志信息显示在终端terminal
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(filehandler)
logger.addHandler(console)

logger.info("Start log")

logger.info("Parametes:　LR: {}, STEP_SIZE:{}, GAMMA: {}".format(LR, STEP_SIZE, GAMMA))


def main(path="./dataVertical"):
    """
    :param path: "C:/Users/chesley/Pictures/ne7/second" as default
    :param case:
    :param n_train:15076 for train
    :param n_test: 2542 for test
    :param dt: 1000 or 3000
    :return:
    """
    logger.info("Run in file: {}".format(path))

    device = torch.device("cuda")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    pathtrainx = path + "/train" + str(config.sequences_in) + "_x.npz"
    pathtrainy = path + "/train" + str(config.sequences_in) + "_y.npz"
    pathtestx = path + "/test" + str(config.sequences_in) + "_x.npz"
    pathtesty = path + "/test" + str(config.sequences_in) + "_y.npz"

    train_x = np.load(pathtrainx)["sequence"]
    train_y = np.load(pathtrainy)["sequence"]
    test_x = np.load(pathtestx)["sequence"]
    test_y = np.load(pathtesty)["sequence"]

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()
    print(train_x.shape)

    # create dataloader
    print("Laoding dataset to torch.")
    trainDataset = TensorDataset(train_x, train_y)  # 合并训练数据和目标数据
    trainDataloader = DataLoader(
        dataset=trainDataset,
        batch_size=config.batch,
        shuffle=True,
        num_workers=config.num_workers
    )
    testDataset = TensorDataset(test_x, test_y)
    testDataloader = DataLoader(
        dataset=testDataset,
        batch_size=config.batch,
        shuffle=False,
        num_workers=config.num_workers
    )
    print("Dataset prepared.")

    torch.cuda.empty_cache()  # 清理显存

    model = DFPTransformer().to(device)
    # model = EFN(config).to(device)  # 建立模型

    loss_fn = nn.MSELoss()  # 定义均方差作为损失函数
    loss_fn.to(device)

    # loss_fn_p = precision()
    # loss_fn_r = recall()

    # optimiser = optim.Adam(params=model.parameters(), lr=LR)  # 定义优化方法
    optimiser = optim.AdamW(params=model.parameters(), lr=LR)  # 定义优化方法
    # weight_decay=0.01 参数设置能让优化器自动带有L2正则


    print("Start Train.")
    num_epochs = config.num_epochs
    total_step = len(trainDataloader)
    loss_List = []
    loss_test_list = []
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=STEP_SIZE, gamma=GAMMA)  # StepLR
    """
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.8, patience=10,
                                                              verbose=False, threshold=0.0001,
                                                              threshold_mode='rel', cooldown=0,
                                                              min_lr=0, eps=1e-08)  # ReduceLROnPlateau
    """

    # save_path for model in different case
    path_m = path + "/" + "model_" + str(config.sequences_in) +".pth"

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=config.patience, verbose=False, path=path_m)

    if os.path.exists(path_m):
        print(path_m)
        model.load_state_dict(torch.load(path_m)["state_dict"])

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        totalLoss = 0  # 计算训练集的平均loss
        for i, (images, target) in enumerate(trainDataloader):
            images = images.to(device)
            target = target.to(device)
            # print("batch 数据集形状", images.shape, target.shape)
            pred = model(images)
            loss = loss_fn(pred, target)
            # loss = loss_fn(pred, target)
            # loss += 0.001 * torch.norm(model.transformer_encoder.weight, p=2)  # L2正则化

            # 反向传播
            optimiser.zero_grad()
            loss.backward()
            # loss_with_penalty.backward()  # 加入L1正则化
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)  # 控制梯度爆炸，Shi et al.(2015)
            optimiser.step()

            lr_now = optimiser.state_dict()['param_groups'][0]['lr']

            # lr_scheduler.step(loss)
            lr_scheduler.step()
            # optimiser.step()

            # 计算平均loss
            totalLoss = totalLoss + loss.item()

            # 打印结果
            if i % 30 == 0:
                tem = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("Time {}, Epoch [{}/{}], Step [{}/{}], loss: {:.8f}, lr: {}".format(tem, epoch+1, num_epochs, i+1, total_step, totalLoss/(i+1), lr_now))
        loss_List.append(totalLoss/(i+1))
        logger.info("Time {}, Epoch [{}/{}], Step [{}/{}], loss: {:.8f}, lr: {}".format(tem, epoch+1, num_epochs, i+1, total_step, totalLoss/(i+1), lr_now))

        # 每一次epoch都对测试集进行测试
        model.eval()
        with torch.no_grad():
            loss_t = 0
            for j, (images, target) in enumerate(testDataloader):
                images, target = images.to(device), target.to(device)
                pred = model(images)
                loss_test = loss_fn(pred, target)
                loss_t += loss_test.item()
                # loss_prec = loss_fn_p(preds, target, threshold=100).item()
                # loss_reca = loss_fn_r(preds, target, threshold=100).item()
        # logger.info("Loss in Test dataset: {}, precision: {}, recall: {}".format(loss_t / (j + 1), loss_prec, loss_reca))
        logger.info("Loss in Test dataset: {}".format(loss_t/(j+1)))

        checkpoint = {
            "state_dict": model.state_dict(),
            "opt_state_dict": optimiser.state_dict(),
            "epoch": epoch
        }
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping.checkpoint = checkpoint
        early_stopping(loss_t/(j+1), model)

        loss_test_list.append(loss_t/(j+1))
        print("_"*10)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    df = pd.DataFrame(data=np.array(loss_List), columns=["loss_train"])
    df["loss_test"] = np.array(loss_test_list)
    pathdf = path + "/_loss.xlsx"

    df.to_excel(pathdf)

    plt.figure(figsize=(8, 8))
    plt.plot(loss_List, color='red', linewidth=1.5, linestyle='-', label="loss_train")
    plt.plot(loss_test_list, color='black', linewidth=1.5, linestyle='-', label="loss_test")
    plt.legend(loc="upper right")
    pathpic = path + "/_loss.jpg"
    plt.savefig(pathpic, dpi=100)
    plt.close()


if __name__ == "__main__":
    path = "./dataVertical"

    torch.cuda.empty_cache()
    train = True
    if train:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        main(path) 
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
