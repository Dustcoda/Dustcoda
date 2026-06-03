#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证码识别模型训练脚本
针对 svg-captcha (150x48, noise:2, 白底黑字) 专用 CNN+BiLSTM+CTC 模型

环境依赖:
    pip install torch torchvision pillow numpy

使用:
    python train.py
"""

import os
import sys
import time
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import numpy as np

# ============================================================
# 字符集配置（必须与 svg-captcha 的 ignoreChars + toUpperCase 完全一致）
# ============================================================
CHARS = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ'
NUM_CLASSES = len(CHARS) + 1          # 32 个有效字符 + 1 个 CTC blank
BLANK_IDX = len(CHARS)                # blank 索引为 32
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}


def parse_args():
    parser = argparse.ArgumentParser(description='训练验证码识别模型')
    parser.add_argument('--data-dir', type=str, default='./dataset',
                        help='数据集根目录（包含 train/val/test 子目录）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='初始学习率')
    parser.add_argument('--img-h', type=int, default=48,
                        help='图像高度')
    parser.add_argument('--img-w', type=int, default=150,
                        help='图像宽度')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备: auto/cpu/cuda')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    return parser.parse_args()


# ============================================================
# 数据集
# ============================================================

class CaptchaDataset(Dataset):
    def __init__(self, img_dir: str, labels_file: str, img_h: int, img_w: int, augment: bool = False):
        self.img_dir = img_dir
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.samples: List[Tuple[str, str]] = []

        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) == 2:
                    self.samples.append((parts[0], parts[1]))

        print(f'[Dataset] 加载 {len(self.samples)} 条样本 from {labels_file}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)

        # 读取为灰度图并缩放到固定尺寸
        img = Image.open(img_path).convert('L')
        img = img.resize((self.img_w, self.img_h), Image.Resampling.BILINEAR)

        # 数据增强（仅训练集）
        if self.augment:
            # 随机对比度微调
            if np.random.rand() < 0.3:
                factor = 0.8 + np.random.rand() * 0.4
                img = ImageEnhance.Contrast(img).enhance(factor)
            # 随机添加高斯噪声
            if np.random.rand() < 0.3:
                arr = np.array(img, dtype=np.float32)
                noise = np.random.normal(0, 5, arr.shape)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)

        # 归一化到 [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

        # 标签编码
        target = torch.tensor([CHAR2IDX[c] for c in label], dtype=torch.long)
        return tensor, target, len(target)


def collate_fn(batch):
    """CTC Loss 需要 targets 拼接为 1D，并配合 target_lengths"""
    imgs = torch.stack([item[0] for item in batch])
    targets = torch.cat([item[1] for item in batch])
    target_lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return imgs, targets, target_lengths


# ============================================================
# 模型：CNN + BiLSTM + CTC
# ============================================================

class CaptchaModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # 输入: (N, 1, 48, 150)
        self.cnn = nn.Sequential(
            # 48x150 -> 24x75
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 24x75 -> 12x37
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 12x37 -> 6x18
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 6x18 -> 6x18 (保持空间尺寸，增加通道)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # CNN 输出: (N, 256, 6, 18)
        # 按宽度 18 作为时间步 T，每个时间步特征维度 = 256 * 6 = 1536
        self.rnn = nn.LSTM(
            input_size=256 * 6,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=False
        )

        self.fc = nn.Linear(512, num_classes)  # 256*2 双向

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W)
        x = self.cnn(x)                       # (N, 256, 6, 18)
        x = x.permute(3, 0, 1, 2)             # (18, N, 256, 6)
        x = x.reshape(x.size(0), x.size(1), -1)  # (18, N, 1536)
        x, _ = self.rnn(x)                    # (18, N, 512)
        x = self.fc(x)                        # (18, N, num_classes)
        return x


# ============================================================
# 浏览器兼容模型（手动 LSTM，零 LSTM 节点）
# ============================================================

class ManualLSTMCell(nn.Module):
    """单个 LSTM 单元，纯基础算子实现（sigmoid/tanh/matmul）"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.linear(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ManualBiLSTM(nn.Module):
    """
    手动双向多层 LSTM（纯基础算子实现）
    参数布局与 nn.LSTM 完全一致，可直接加载训练权重。
    """
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input = input_size if layer == 0 else hidden_size * self.num_directions
                self.cells.append(ManualLSTMCell(layer_input, hidden_size))

    def forward(self, x):
        T, N, _ = x.shape
        h_fwd = [[None] * T for _ in range(self.num_layers)]
        h_bwd = [[None] * T for _ in range(self.num_layers)]

        for layer in range(self.num_layers):
            fwd_idx = layer * self.num_directions + 0
            bwd_idx = layer * self.num_directions + 1

            fwd_h = torch.zeros(N, self.hidden_size, device=x.device, dtype=x.dtype)
            fwd_c = torch.zeros(N, self.hidden_size, device=x.device, dtype=x.dtype)
            for t in range(T):
                inp = x[t] if layer == 0 else torch.cat([h_fwd[layer-1][t], h_bwd[layer-1][t]], dim=1)
                fwd_h, fwd_c = self.cells[fwd_idx](inp, (fwd_h, fwd_c))
                h_fwd[layer][t] = fwd_h

            bwd_h = torch.zeros(N, self.hidden_size, device=x.device, dtype=x.dtype)
            bwd_c = torch.zeros(N, self.hidden_size, device=x.device, dtype=x.dtype)
            for t in range(T - 1, -1, -1):
                inp = x[t] if layer == 0 else torch.cat([h_fwd[layer-1][t], h_bwd[layer-1][t]], dim=1)
                bwd_h, bwd_c = self.cells[bwd_idx](inp, (bwd_h, bwd_c))
                h_bwd[layer][t] = bwd_h

        last = self.num_layers - 1
        result = torch.stack([
            torch.cat([h_fwd[last][t], h_bwd[last][t]], dim=1)
            for t in range(T)
        ], dim=0)
        return result


class BrowserCaptchaModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.rnn = ManualBiLSTM(
            input_size=256 * 6,
            hidden_size=256,
            num_layers=2,
            bidirectional=True
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(3, 0, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.rnn(x)
        x = self.fc(x)
        return x


def load_lstm_weights(manual_lstm: ManualBiLSTM, state_dict: dict, prefix: str):
    """将 nn.LSTM 的权重映射到 ManualBiLSTM"""
    for layer in range(manual_lstm.num_layers):
        for d in range(manual_lstm.num_directions):
            cell_idx = layer * manual_lstm.num_directions + d
            suffix = '_reverse' if d == 1 else ''
            w_ih_key = f'{prefix}weight_ih_l{layer}{suffix}'
            w_hh_key = f'{prefix}weight_hh_l{layer}{suffix}'
            b_ih_key = f'{prefix}bias_ih_l{layer}{suffix}'
            b_hh_key = f'{prefix}bias_hh_l{layer}{suffix}'

            input_size = manual_lstm.cells[cell_idx].linear.in_features - manual_lstm.hidden_size
            hidden_size = manual_lstm.hidden_size
            w_ih = state_dict[w_ih_key]
            w_hh = state_dict[w_hh_key]
            b_ih = state_dict[b_ih_key]
            b_hh = state_dict[b_hh_key]

            combined_weight = torch.cat([w_ih, w_hh], dim=1)
            combined_bias = b_ih + b_hh

            manual_lstm.cells[cell_idx].linear.weight.data.copy_(combined_weight)
            manual_lstm.cells[cell_idx].linear.bias.data.copy_(combined_bias)


# ============================================================
# CTC 解码
# ============================================================

def ctc_greedy_decode(outputs: torch.Tensor) -> List[str]:
    """
    outputs: (T, N, C) 已经过 softmax 或 argmax 前的 logits
    """
    preds = outputs.argmax(dim=2).cpu().numpy()  # (T, N)
    batch_size = preds.shape[1]
    results = []
    for n in range(batch_size):
        seq = preds[:, n].tolist()
        decoded = []
        last = -1
        for p in seq:
            if p != BLANK_IDX and p != last:
                decoded.append(IDX2CHAR.get(p, '?'))
            last = p
        results.append(''.join(decoded))
    return results


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    all_loss = 0.0
    count = 0

    with torch.no_grad():
        for imgs, targets, target_lengths in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)  # (T, N, C)
            log_probs = F.log_softmax(outputs, dim=2)
            T, N, _ = log_probs.shape
            input_lengths = torch.full((N,), T, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, targets.to(device), input_lengths, target_lengths.to(device))
            all_loss += loss.item()
            count += 1

            preds = ctc_greedy_decode(outputs)
            # 获取真实标签
            # 需要从 targets 和 target_lengths 中还原
            idx = 0
            for i, length in enumerate(target_lengths):
                label_indices = targets[idx: idx + length].tolist()
                label = ''.join(IDX2CHAR.get(j, '?') for j in label_indices)
                idx += length
                if preds[i] == label:
                    correct += 1
                total += 1

    acc = correct / total if total > 0 else 0
    avg_loss = all_loss / count if count > 0 else 0
    return acc, avg_loss


# ============================================================
# 训练主流程
# ============================================================

def main():
    args = parse_args()

    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'使用设备: {device}')

    # 数据集路径
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')

    train_labels = os.path.join(train_dir, 'labels.txt')
    val_labels = os.path.join(val_dir, 'labels.txt')
    test_labels = os.path.join(test_dir, 'labels.txt')

    for p in [train_labels, val_labels]:
        if not os.path.exists(p):
            print(f'错误: 未找到标签文件 {p}，请先运行 node generate_dataset.js')
            sys.exit(1)

    # 构建 DataLoader
    train_ds = CaptchaDataset(train_dir, train_labels, args.img_h, args.img_w, augment=True)
    val_ds = CaptchaDataset(val_dir, val_labels, args.img_h, args.img_w, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    # 模型
    model = CaptchaModel(NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params / 1e6:.2f}M')

    # 优化器与调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    # 训练状态
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, 'best.pth')
    best_acc = 0.0
    global_step = 0

    print('\n开始训练...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        start_time = time.time()

        for imgs, targets, target_lengths in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)  # (T, N, C)
            log_probs = F.log_softmax(outputs, dim=2)
            T, N, _ = log_probs.shape
            input_lengths = torch.full((N,), T, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            # 梯度裁剪，防止 RNN 梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1
            global_step += 1

        scheduler.step()
        avg_train_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0

        # 验证
        val_acc, val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - start_time

        print(f'Epoch [{epoch:02d}/{args.epochs}]  '
              f'TrainLoss: {avg_train_loss:.4f}  '
              f'ValLoss: {val_loss:.4f}  '
              f'ValAcc: {val_acc*100:.2f}%  '
              f'LR: {scheduler.get_last_lr()[0]:.6f}  '
              f'Time: {elapsed:.1f}s')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'chars': CHARS
            }, best_path)
            print(f'  -> 最佳模型已保存 (acc={val_acc*100:.2f}%)')

    # 测试集评估
    if os.path.exists(test_labels):
        test_ds = CaptchaDataset(test_dir, test_labels, args.img_h, args.img_w, augment=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
        test_acc, test_loss = evaluate(model, test_loader, device)
        print(f'\n测试集结果: Loss={test_loss:.4f}, Acc={test_acc*100:.2f}%')

    # 导出 ONNX（动态 batch）
    print('\n导出 ONNX...')
    model.eval()
    dummy_input = torch.randn(1, 1, args.img_h, args.img_w).to(device)
    onnx_path = os.path.join(args.save_dir, 'captcha_model.onnx')

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {1: 'batch_size'}
        },
        opset_version=15,
        do_constant_folding=True
    )
    print(f'ONNX 模型已保存: {onnx_path}')
    print(f'模型大小: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB')

    # 导出浏览器兼容 ONNX
    print('\n导出浏览器兼容 ONNX...')
    browser_model = BrowserCaptchaModel(NUM_CLASSES)
    checkpoint = torch.load(best_path, map_location=device)
    ckpt_state = checkpoint['model_state_dict']

    cnn_state = {k.replace('cnn.', ''): v for k, v in ckpt_state.items() if k.startswith('cnn.')}
    browser_model.cnn.load_state_dict(cnn_state)
    load_lstm_weights(browser_model.rnn, ckpt_state, 'rnn.')
    browser_model.fc.load_state_dict({
        'weight': ckpt_state['fc.weight'],
        'bias': ckpt_state['fc.bias'],
    })
    browser_model.eval()
    browser_model.to(device)

    browser_onnx_path = os.path.join(args.save_dir, 'captcha_model_browser.onnx')
    torch.onnx.export(
        browser_model,
        dummy_input,
        browser_onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {1: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True,
        dynamo=False
    )
    print(f'浏览器兼容 ONNX 已保存: {browser_onnx_path}')
    print(f'模型大小: {os.path.getsize(browser_onnx_path) / 1024 / 1024:.2f} MB')

    # 验证
    try:
        import onnx
        onnx_model = onnx.load(browser_onnx_path)
        lstm_nodes = [n for n in onnx_model.graph.node if n.op_type == 'LSTM']
        if lstm_nodes:
            print(f'警告: 模型仍有 {len(lstm_nodes)} 个 LSTM 节点')
        else:
            print('验证通过: 零 LSTM 节点，浏览器兼容')
        op_types = sorted(set(n.op_type for n in onnx_model.graph.node))
        print(f'算子类型 ({len(op_types)} 个): {", ".join(op_types)}')
        opset = onnx_model.opset_import[0].version if onnx_model.opset_import else '?'
        print(f'Opset 版本: {opset}  IR 版本: {onnx_model.ir_version}')
    except ImportError:
        pass

    print('\n下一步: 将 captcha_model_browser.onnx 部署到可访问的 URL，')
    print('        然后修改油猴脚本中的 CAPTCHA_MODEL_URL 配置即可。')


if __name__ == '__main__':
    main()
