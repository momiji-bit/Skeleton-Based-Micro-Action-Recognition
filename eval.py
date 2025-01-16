# -*- coding: utf-8 -*-

"""
该脚本用于验证指定模型参数(.pt文件)的准确率，主要流程包括：
1. 加载模型并初始化
2. 加载测试数据
3. 在测试集上进行推理并计算准确率
4. 可输出混淆矩阵和各类别准确率等信息
5. 新增：添加F1 Macro和F1 Micro的计算并显示

使用示例：
python eval_model.py --config ./config/test.yaml --weights ./checkpoints/model_weights.pt

请在配置文件config或命令行参数中修改自己的数据加载、模型构建等参数。
"""

import argparse
import os
import sys
import time
import pickle
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shutil
import traceback
import csv
from collections import OrderedDict
# --------------------------
# 新增：导入 f1_score
# --------------------------
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
import numpy as np
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def init_seed(seed):
    """初始化随机种子."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    """通过字符串导入类或函数."""
    mod_str, _, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'无法在 {mod_str} 中找到类 {class_str}.\n{traceback.format_exc()}')


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_parser():
    """获取命令行参数解析器."""

    parser = argparse.ArgumentParser(description='验证指定模型参数的准确率')


    # parser.add_argument('--config', default='config/SkateFormer_j_NEW.yaml', help='配置文件路径')
    # parser.add_argument('--weights', default="output/original_48/runs-84-73752.pt", help='模型权重文件(.pt)的路径')

    # parser.add_argument('--config', default='config/SkateFormer_4w_s_j_NEW.yaml', help='配置文件路径')
    # parser.add_argument('--weights', default="output/original_48_4w_s/runs-89-22339.pt", help='模型权重文件(.pt)的路径')

    # parser.add_argument('--config', default='config/SkateFormer_4w_j_NEW.yaml', help='配置文件路径')
    # parser.add_argument('--weights', default="output/original_48_4w/runs-82-31980.pt", help='模型权重文件(.pt)的路径')

    # parser.add_argument('--config', default='config/SkateFormer_4w_SM_j_NEW.yaml', help='配置文件路径')
    # parser.add_argument('--weights', default="output/original_48_4w_SM/runs-75-29250.pt", help='模型权重文件(.pt)的路径')
    # parser.add_argument('--weights', default="output/original_48_4w_SM_p0.5/runs-62-24180.pt", help='模型权重文件(.pt)的路径')

    # parser.add_argument('--config', default='config/SkateFormer_6w_j_NEW.yaml', help='配置文件路径')
    # parser.add_argument('--weights', default="output/original_48_6w/runs-80-31200.pt", help='模型权重文件(.pt)的路径')

    parser.add_argument('--config', default='config/SkateFormer_6w_s_j_NEW.yaml', help='配置文件路径')
    parser.add_argument('--weights', default="output/original_48_6w_s/runs-56-49168.pt", help='模型权重文件(.pt)的路径')

    # parser.add_argument('--config', default='config/SkateFormer_j_-TGconv2_NEW.yaml', help='配置文件路径')
    # parser.add_argument('--weights', default="output/original_48_-TGconv2/runs-89-34710.pt", help='模型权重文件(.pt)的路径')

    # parser.add_argument('--config', default='config/SkateFormer_j_s_NEW.yaml', help='配置文件路径')
    # parser.add_argument('--weights', default="output/original_48_s/runs-58-14558.pt",
    #                     help='模型权重文件(.pt)的路径')


    # parser.add_argument('--config', default='./config/SkateFormer_j_SE3_NEW.yaml', help='path to the configuration file')
    # parser.add_argument('--weights', default="output/original_48_SE3/runs-35-15365.pt",
    #                     help='模型权重文件(.pt)的路径')

    parser.add_argument('--work-dir', default='./output', help='日志或输出等文件的保存路径')
    parser.add_argument('--feeder', default='feeder.feeder', help='数据加载类(含路径)的名称')
    parser.add_argument('--test-feeder-args', type=dict, default={}, help='测试集DataLoader的相关参数')
    parser.add_argument('--model', default=None, help='模型类(含路径)的名称')
    parser.add_argument('--model-args', type=dict, default={}, help='模型初始化所需的参数')
    parser.add_argument('--ignore-weights', type=str, nargs='+', default=[], help='需要在初始化时忽略的权重名')
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='GPU设备ID列表')
    parser.add_argument('--seed', type=int, default=1, help='随机数种子')
    parser.add_argument('--show-topk', type=int, nargs='+', default=[1, 5], help='显示top-k准确率')
    parser.add_argument('--save-score', action='store_true', help='是否保存推理输出score到.pkl文件')
    parser.add_argument('--print-log', action='store_true', default=True, help='是否打印日志')
    return parser


class Evaluator:
    def __init__(self, arg):
        self.arg = arg
        self.output_device = "cuda:0"
        self._print_log(f"使用GPU: {self.output_device}")
        self.load_model()
        self.load_data()

    def load_model(self):
        """加载模型结构和参数."""
        if self.arg.model is None:
            raise ValueError("请指定 --model，例如 'models.MyModel'")

        ModelClass = import_class(self.arg.model)
        shutil.copy2(sys.modules[ModelClass.__module__].__file__, self.arg.work_dir)  # 备份模型结构文件
        self.model = ModelClass(**self.arg.model_args).cuda(self.output_device)

        if self.arg.weights:
            self._print_log(f"加载模型权重: {self.arg.weights}")
            weights = torch.load(self.arg.weights, map_location='cpu')
            weights = OrderedDict([[k.replace('module.', ''), v] for k, v in weights.items()])

            if self.arg.ignore_weights:
                for iw in self.arg.ignore_weights:
                    weights = {k: v for k, v in weights.items() if iw not in k}

            model_state = self.model.state_dict()
            diff_keys = set(model_state.keys()) - set(weights.keys())
            if diff_keys:
                self._print_log("以下权重未被加载:")
                for k in diff_keys:
                    self._print_log(f"  {k}")
            model_state.update({k: v for k, v in weights.items() if k in model_state})
            self.model.load_state_dict(model_state)

        # 如果有多块GPU
        # if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)

    def load_data(self):
        """仅加载测试数据."""
        FeederClass = import_class(self.arg.feeder)
        self.data_loader = torch.utils.data.DataLoader(
            dataset=FeederClass(**self.arg.test_feeder_args),
            batch_size=self.arg.test_feeder_args.get('batch_size', 64),
            pin_memory=True,
            drop_last=False,
            worker_init_fn=init_seed,
            shuffle=False,
            num_workers=self.arg.test_feeder_args.get('num_workers', 0),
        )
        self._print_log("测试集加载完毕。")

    @torch.no_grad()
    def eval(self):
        """在测试集中验证模型的准确率."""
        self.model.eval()
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(self.output_device)

        score_list = []
        label_list = []
        losses = []

        self._print_log("开始在测试集上计算准确率...")

        process_bar = tqdm(self.data_loader, ncols=80)
        for batch_idx, (data, index_t, label, index) in enumerate(process_bar):
            data = data.float().cuda(self.output_device)
            index_t = index_t.float().cuda(self.output_device)
            label = label.long().cuda(self.output_device)

            output = self.model(data, index_t)
            loss = loss_fn(output, label)
            losses.append(loss.item())

            score_list.append(output.cpu().numpy())
            label_list.append(label.cpu().numpy())

        scores = np.concatenate(score_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        avg_loss = np.mean(losses)

        if self.arg.save_score:
            score_dict = {i: score for i, score in enumerate(scores)}
            score_path = os.path.join(self.arg.work_dir, 'test_score.pkl')
            with open(score_path, 'wb') as f:
                pickle.dump(score_dict, f)
            self._print_log(f"推理分数已保存至: {score_path}")

        # 计算并输出混淆矩阵及各类准确率
        each_class_acc, confusion = self.compute_confusion_matrix(scores, labels)
        print("==="*10)
        # 计算top-k准确率
        top = self._compute_accuracy(scores, labels, avg_loss)

        # --------------------------
        # 新增：计算并打印F1 Macro和F1 Micro
        # --------------------------
        pred_label = np.argmax(scores, axis=1)
        macro_f1 = f1_score(labels, pred_label, average='macro')
        micro_f1 = f1_score(labels, pred_label, average='micro')
        self._print_log(f"F1 Macro: {macro_f1*100:.2f}%")
        self._print_log(f"F1 Micro: {micro_f1*100:.2f}%")

        # 绘制每类准确率柱状图
        self.plot_class_accuracy(each_class_acc, top, labels)
        # 绘制混淆矩阵热力图
        self.plot_confusion_matrix(confusion)

    def _compute_accuracy(self, scores, labels, avg_loss):
        """计算并输出top-k准确率."""
        num_samples = len(labels)
        self._print_log(f"测试样本总数: {num_samples}")
        self._print_log(f"平均loss: {avg_loss:.4f}")

        top = []
        for k in self.arg.show_topk:
            hit_top_k = 0
            pred = np.argsort(scores, axis=1)[:, -k:]
            for i in range(num_samples):
                if labels[i] in pred[i, :]:
                    hit_top_k += 1
            acc = hit_top_k / num_samples * 100.0
            self._print_log(f"Top-{k} 准确率: {acc:.2f}%")
            top.append(acc)
        return top

    def compute_confusion_matrix(self, scores, labels):
        """输出混淆矩阵及各类别准确率."""
        pred_label = np.argmax(scores, axis=1)
        confusion = confusion_matrix(labels, pred_label)
        diag = np.diag(confusion)
        raw_sum = np.sum(confusion, axis=1)
        each_class_acc = diag / (raw_sum + 1e-10)  # 防止除0

        for i, acc in enumerate(each_class_acc):
            print(f"类别{i}的准确率: {acc * 100:.2f}%")

        csv_path = os.path.join(self.arg.work_dir, 'test_confusion_matrix.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Class_Accuracy"] + list(range(len(each_class_acc))))
            writer.writerow(["Value"] + list(each_class_acc))
            writer.writerow([])
            writer.writerow(["Confusion_Matrix"])
            writer.writerows(confusion)
        print(f"混淆矩阵已保存至: {csv_path}")
        return each_class_acc, confusion

    def plot_class_accuracy(self, each_class_acc, top, labels):
        """绘制每类准确率的柱状图，并根据类别占比调整柱状图颜色深浅，同时添加平均准确率的横线."""
        num_classes = len(each_class_acc)
        total_labels = len(labels)
        times = [0]*num_classes
        for i in labels:
            times[i] += 1
        label_ratios = np.array(times) / total_labels
        min_ratio, max_ratio = np.min(label_ratios), np.max(label_ratios)
        normalized_ratios = (label_ratios - min_ratio) / (max_ratio - min_ratio)
        cmap = plt.cm.magma
        colors = cmap(normalized_ratios)

        plt.figure(figsize=(15, 6))
        plt.bar(range(num_classes), each_class_acc * 100, color=colors, label='Class Accuracy')
        plt.axhline(y=top[0], color='red', linestyle='--', label=f'Top-1 ({top[0]:.2f}%)')
        if len(top) > 1:
            plt.axhline(y=top[1], color='orange', linestyle='--', label=f'Top-{self.arg.show_topk[1]} ({top[1]:.2f}%)')
        plt.xticks(range(num_classes), [f'{i}' for i in range(num_classes)], rotation=45)
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Class')
        plt.title('Per-Class Accuracy (Color by Label Proportion)')
        plt.ylim(0, 100)
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label='Label Proportion')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, confusion):
        """绘制混淆矩阵的热力图."""
        row_sums = confusion.sum(axis=1, keepdims=True)
        percentage_matrix = confusion / row_sums

        num_classes = confusion.shape[0]
        plt.figure(figsize=(8, 6))
        sns.heatmap(percentage_matrix, annot=False)
        plt.ylabel('True Class')
        plt.gca().invert_yaxis()
        plt.xlabel('Predicted Class')
        plt.title('Confusion Matrix Heatmap (Percentage per Row)')
        plt.show()

    def _compute_confusion_matrix(self, scores, labels):
        """输出混淆矩阵及各类别准确率（仅内部使用的版本）."""
        pred_label = np.argmax(scores, axis=1)
        confusion = confusion_matrix(labels, pred_label)
        diag = np.diag(confusion)
        raw_sum = np.sum(confusion, axis=1)
        each_class_acc = diag / (raw_sum + 1e-10)

        for i, acc in enumerate(each_class_acc):
            self._print_log(f"类别{i}的准确率: {acc*100:.2f}%")

        csv_path = os.path.join(self.arg.work_dir, 'test_confusion_matrix.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Class_Accuracy"] + list(range(len(each_class_acc))))
            writer.writerow(["Value"] + list(each_class_acc))
            writer.writerow([])
            writer.writerow(["Confusion_Matrix"])
            writer.writerows(confusion)
        self._print_log(f"混淆矩阵已保存至: {csv_path}")

    def _print_log(self, msg):
        """打印日志到屏幕和log文件."""
        if self.arg.print_log:
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            info = f"[{localtime}] {msg}"
            print(info)
            log_path = os.path.join(self.arg.work_dir, 'eval_log.txt')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(info + '\n')


def main():
    parser = get_parser()
    p = parser.parse_args()

    if p.config and os.path.exists(p.config):
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)
        for k, v in default_arg.items():
            if getattr(p, k, None) == parser.get_default(k):
                setattr(p, k, v)

    init_seed(p.seed)

    if not os.path.exists(p.work_dir):
        os.makedirs(p.work_dir)

    evaluator = Evaluator(p)
    evaluator.eval()


if __name__ == '__main__':
    main()
