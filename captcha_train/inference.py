#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 模型本地验证脚本
支持：
  1) 在测试集上批量验证准确率
  2) 对单张图片实时推理

依赖：
    pip install onnxruntime pillow numpy

使用示例：
    # 批量验证测试集
    python inference.py --model ./checkpoints/captcha_model.onnx --mode test --data-dir ./dataset/test

    # 单张图片推理
    python inference.py --model ./checkpoints/captcha_model.onnx --mode single --image ./test_sample.png
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
import onnxruntime as ort

# 字符集必须与训练时完全一致
CHARS = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ'
BLANK_IDX = len(CHARS)
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

IMG_H = 48
IMG_W = 150


def preprocess(image_path: str) -> np.ndarray:
    """
    图像预处理：灰度化 → resize → 归一化 → 增加 batch 维度
    输出形状: (1, 1, 48, 150)
    """
    img = Image.open(image_path).convert('L')
    img = img.resize((IMG_W, IMG_H), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, np.newaxis, ...]


def ctc_decode(pred: np.ndarray) -> str:
    """
    CTC 贪心解码
    pred: (T, C) 每个时间步的类别概率分布
    """
    seq = pred.argmax(axis=1)
    decoded = []
    last = -1
    for p in seq:
        if p != BLANK_IDX and p != last:
            decoded.append(IDX2CHAR.get(int(p), '?'))
        last = p
    return ''.join(decoded)


def batch_validate(session: ort.InferenceSession, data_dir: str) -> None:
    """在数据集上批量验证"""
    labels_file = os.path.join(data_dir, 'labels.txt')
    if not os.path.exists(labels_file):
        print(f'错误: 未找到标签文件 {labels_file}')
        sys.exit(1)

    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0
    errors: List[Tuple[str, str, str]] = []

    with open(labels_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f'开始验证 {len(lines)} 条样本...')
    for line in lines:
        fname, label = line.split(',')
        img_path = os.path.join(data_dir, fname)

        if not os.path.exists(img_path):
            print(f'  警告: 图片不存在 {img_path}，跳过')
            continue

        x = preprocess(img_path)
        outputs = session.run(None, {input_name: x})[0]  # (T, 1, C)
        pred = ctc_decode(outputs[:, 0, :])

        total += 1
        if pred == label:
            correct += 1
        else:
            errors.append((fname, label, pred))

    acc = correct / total * 100 if total > 0 else 0
    print(f'\n准确率: {acc:.2f}% ({correct}/{total})')

    if errors:
        print(f'\n识别错误的样本 (共 {len(errors)} 个，显示前 15 个):')
        for fname, truth, pred in errors[:15]:
            print(f'  {fname}: 真实="{truth}", 预测="{pred}"')


def single_inference(session: ort.InferenceSession, image_path: str) -> None:
    """单张图片推理"""
    if not os.path.exists(image_path):
        print(f'错误: 图片不存在 {image_path}')
        sys.exit(1)

    input_name = session.get_inputs()[0].name
    x = preprocess(image_path)

    # 推理计时
    start = os.times()[0]
    outputs = session.run(None, {input_name: x})[0]
    elapsed = (os.times()[0] - start) * 1000

    pred = ctc_decode(outputs[:, 0, :])
    print(f'\n识别结果: {pred}')
    print(f'推理耗时: {elapsed:.2f} ms')

    # 同时输出各时间步的最高概率字符（调试用）
    seq = outputs[:, 0, :].argmax(axis=1)
    probs = outputs[:, 0, :].max(axis=1)
    debug_chars = []
    for t, (idx, prob) in enumerate(zip(seq, probs)):
        ch = IDX2CHAR.get(int(idx), '_') if idx != BLANK_IDX else '-'
        debug_chars.append(f'{ch}:{prob:.2f}')
    print(f'时间步解码: {" | ".join(debug_chars)}')


def main():
    parser = argparse.ArgumentParser(description='ONNX 验证码模型本地验证')
    parser.add_argument('--model', type=str, default='./checkpoints/captcha_model_browser.onnx',
                        help='ONNX 模型路径')
    parser.add_argument('--mode', choices=['test', 'single'], default='test',
                        help='验证模式: test=批量验证, single=单张推理')
    parser.add_argument('--image', type=str, default='',
                        help='单张图片路径 (single 模式必填)')
    parser.add_argument('--data-dir', type=str, default='./dataset/test',
                        help='测试集目录 (test 模式)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'错误: 模型文件不存在 {args.model}')
        print('请先运行 python train.py 完成训练并导出 ONNX')
        sys.exit(1)

    # 加载 ONNX Runtime（优先 GPU，否则 CPU）
    providers = ort.get_available_providers()
    preferred = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
    selected = [p for p in preferred if p in providers]
    session = ort.InferenceSession(args.model, providers=selected)
    print(f'加载模型: {args.model}')
    print(f'推理后端: {selected[0] if selected else "CPU"}')

    if args.mode == 'test':
        batch_validate(session, args.data_dir)
    else:
        if not args.image:
            print('错误: single 模式需要指定 --image 参数')
            sys.exit(1)
        single_inference(session, args.image)


if __name__ == '__main__':
    main()
