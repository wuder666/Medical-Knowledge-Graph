# coding:utf-8
from model.CasrelModel import *
from utils.process import *
from utils.data_loader import *
from config import *
import pandas as pd
from tqdm import tqdm
import math


# 定义主训练方法
def mode2train(model, train_iter, dev_iter, optimizer, conf):
    # 1.设置训练轮数
    epochs = conf.epochs
    # 初始化最佳Triple F1（核心：仅保留这个指标最高的模型）
    best_triple_f1 = 0.0
    # 计算训练集总步数（用于进度展示）
    total_train_steps = len(train_iter)

    print(f"===== 开始训练，总轮数：{epochs}，每轮步数：{total_train_steps} =====")

    for epoch in range(epochs):
        # 打印当前轮数
        print(f"\n===== Epoch [{epoch + 1}/{epochs}] 开始训练 =====")
        # 训练单轮并更新最佳F1
        best_triple_f1 = train_epoch(
            model=model,
            train_iter=train_iter,
            dev_iter=dev_iter,
            optimizer=optimizer,
            best_triple_f1=best_triple_f1,
            epoch=epoch,
            total_train_steps=total_train_steps
        )

    # 可选：保存最后一轮模型（若不需要，直接删除这两行）
    print(f"\n===== 训练结束，保存最后一轮模型（兜底） =====")
    torch.save(model.state_dict(), './save_model/last_model.pth')
    print(f"最终最佳Triple F1: {best_triple_f1:.4f}，对应的模型已保存在 ./save_model/ai22_best_f1.pth")


def train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch, total_train_steps):
    # 为tqdm添加epoch信息，进度条显示当前轮数
    pbar = tqdm(
        train_iter,
        desc=f'Casrel模型训练 - Epoch [{epoch + 1}/{conf.epochs}]'
    )

    for step, (inputs, labels) in enumerate(pbar):
        model.train()
        # 模型前向传播
        logits = model(**inputs)
        # 计算损失
        loss = model.compute_loss(**logits, **labels)
        # 梯度清零 + 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 实时更新进度条描述，显示当前step、epoch和loss
        pbar.set_postfix({
            'Epoch': f'{epoch + 1}',
            'Step': f'{step + 1}/{total_train_steps}',
            'Loss': f'{loss.item():.4f}',
            'Best_F1': f'{best_triple_f1:.4f}'  # 进度条显示当前最佳F1
        })

        # 每500步验证一次，判断是否更新最佳模型
        if (step + 1) % 500 == 0:
            print(f"\n----- Epoch [{epoch + 1}] Step [{step + 1}/{total_train_steps}] 开始验证 -----")
            # 验证模型并获取指标
            results = model2dev(model, dev_iter)
            current_triple_f1 = results[5]  # results[5]是triple_f1
            print(f"----- Epoch [{epoch + 1}] Step [{step + 1}] 验证完成 -----")
            print(f"当前最佳Triple F1: {best_triple_f1:.4f} | 本次验证Triple F1: {current_triple_f1:.4f}")

            # 核心逻辑：仅当本次Triple F1 > 历史最佳时，才更新并保存最佳模型
            if current_triple_f1 > best_triple_f1:
                best_triple_f1 = current_triple_f1
                # 覆盖保存最佳模型（始终只保留1个最优模型）
                torch.save(model.state_dict(), './save_model/ai22_best_f1.pth')
                print('===== 最佳模型已更新！=====')
                # 打印详细指标
                print(
                    'epoch:{}/{},'
                    'step:{}/{},'
                    'sub_precision:{:.4f}, '
                    'sub_recall:{:.4f}, '
                    'sub_f1:{:.4f}, '
                    'triple_precision:{:.4f}, '
                    'triple_recall:{:.4f}, '
                    'triple_f1:{:.4f},'
                    'train loss:{:.4f}'.format(
                        epoch + 1, conf.epochs,
                        step + 1, total_train_steps,
                        results[0], results[1], results[2],
                        results[3], results[4], results[5],
                        loss.item()
                    )
                )
            else:
                # 未更新最佳模型时，仅打印基础信息
                print(
                    'epoch:{}/{},'
                    'step:{}/{},'
                    'sub_precision:{:.4f}, '
                    'sub_recall:{:.4f}, '
                    'sub_f1:{:.4f}, '
                    'triple_precision:{:.4f}, '
                    'triple_recall:{:.4f}, '
                    'triple_f1:{:.4f},'
                    'train loss:{:.4f}'.format(
                        epoch + 1, conf.epochs,
                        step + 1, total_train_steps,
                        results[0], results[1], results[2],
                        results[3], results[4], results[5],
                        loss.item()
                    )
                )
    return best_triple_f1


def model2dev(model, dev_iter):
    '''
    验证模型效果，返回各类指标
    :param model: 训练的Casrel模型
    :param dev_iter: 验证集迭代器
    :return: sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df
    '''
    model.eval()
    # 定义指标统计DataFrame
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)

    # 验证进度条
    dev_pbar = tqdm(dev_iter, desc='Casrel模型验证')
    with torch.no_grad():  # 验证阶段禁用梯度计算，节省显存
        for inputs, labels in dev_pbar:
            logist = model(**inputs)
            # 转换预测/标签为0-1值（阈值筛选）
            pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
            pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
            sub_heads = convert_score_to_zero_one(labels['sub_heads'])
            sub_tails = convert_score_to_zero_one(labels['sub_tails'])
            batch_size = inputs['input_ids'].shape[0]
            obj_heads = convert_score_to_zero_one(labels['obj_heads'])
            obj_tails = convert_score_to_zero_one(labels['obj_tails'])
            pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
            pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])

            # 逐样本统计指标
            for batch_idx in range(batch_size):
                # 提取主实体
                pred_subs = extract_sub(pred_sub_heads[batch_idx].squeeze(), pred_sub_tails[batch_idx].squeeze())
                true_subs = extract_sub(sub_heads[batch_idx].squeeze(), sub_tails[batch_idx].squeeze())
                # 提取客体和关系（三元组）
                pred_objs = extract_obj_and_rel(pred_obj_heads[batch_idx], pred_obj_tails[batch_idx])
                true_objs = extract_obj_and_rel(obj_heads[batch_idx], obj_tails[batch_idx])

                # 统计主实体指标
                df.loc["sub", "PRED"] += len(pred_subs)
                df.loc["sub", "REAL"] += len(true_subs)
                for true_sub in true_subs:
                    if true_sub in pred_subs:
                        df.loc['sub', 'TP'] += 1

                # 统计三元组指标
                df.loc["triple", "PRED"] += len(pred_objs)
                df.loc["triple", "REAL"] += len(true_objs)
                for true_obj in true_objs:
                    if true_obj in pred_objs:
                        df.loc["triple", 'TP'] += 1

    # 计算主实体的精确率、召回率、F1
    df.loc["sub", "p"] = df.loc['sub', 'TP'] / (df.loc["sub", "PRED"] + 1e-9)  # 加1e-9避免除0
    df.loc["sub", "r"] = df.loc['sub', 'TP'] / (df.loc["sub", "REAL"] + 1e-9)
    df.loc["sub", 'f1'] = 2 * df.loc["sub", "p"] * df.loc["sub", "r"] / (df.loc["sub", "p"] + df.loc["sub", "r"] + 1e-9)

    # 计算三元组的精确率、召回率、F1
    df.loc["triple", "p"] = df.loc['triple', 'TP'] / (df.loc["triple", "PRED"] + 1e-9)
    df.loc["triple", "r"] = df.loc['triple', 'TP'] / (df.loc["triple", "REAL"] + 1e-9)
    df.loc["triple", 'f1'] = 2 * df.loc["triple", "p"] * df.loc["triple", "r"] / (df.loc["triple", "p"] + df.loc["triple", "r"] + 1e-9)

    # 提取最终指标
    sub_precision = df.loc["sub", "p"]
    sub_recall = df.loc["sub", "r"]
    sub_f1 = df.loc["sub", 'f1']
    triple_precision = df.loc["triple", "p"]
    triple_recall = df.loc["triple", "r"]
    triple_f1 = df.loc["triple", 'f1']

    # 打印验证结果摘要
    print(f"\n验证结果摘要：")
    print(f"主实体 - 精确率：{sub_precision:.4f}，召回率：{sub_recall:.4f}，F1：{sub_f1:.4f}")
    print(f"三元组 - 精确率：{triple_precision:.4f}，召回率：{triple_recall:.4f}，F1：{triple_f1:.4f}")

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


if __name__ == '__main__':
    # 加载配置、模型、优化器、数据
    conf = Config()
    model, optimizer, sheduler, device = load_model(conf)  # sheduler未使用，可根据需要添加学习率调度
    train_iter, dev_iter, _ = get_data()

    # 启动训练（核心：仅保留Triple F1最高的1个模型）
    mode2train(model, train_iter, dev_iter, optimizer, conf)