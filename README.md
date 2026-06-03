# EasyLink 智能自动登录助手

基于 Tampermonkey 的油猴脚本，提供多账号管理、自动表单填充、本地 ONNX 验证码识别功能。针对 SVG 验证码做了专门优化。

---

## 🚀 功能特性

- **多账号管理**：支持保存多个平台账号，随时切换登录
- **智能元素定位**：可自定义 CSS 选择器，适配不同页面结构
- **本地验证码识别**：使用 PyTorch 训练 CNN+BiLSTM+CTC 模型，通过 ONNX Runtime Web 在浏览器内直接推理，无需调用远程 API
- **可配置自动弹窗**：开关控制是否在登录页自动弹出账号选择窗口
- **SPA 路由兼容**：支持 hashchange 和 DOM 变化监听，兼容退出登录后重绘登录页场景
- **干净的 UI**：深色玻璃拟态风格，右下角悬浮按钮可随时唤起

---

## 📦 快速开始

### 1. 安装油猴脚本

1. 在浏览器中安装 [Tampermonkey](https://www.tampermonkey.net/) 扩展
2. 打开 Tampermonkey 管理面板 →  utilities →  Import from URL
3. 输入脚本文件路径（或直接复制粘贴 `auto-login.user.js` 的内容）
4. 保存安装

### 2. 安装后使用

- 打开你的登录页面（例如 `https://debug.easylink-iot.com/#/login`）
- 右下角会出现一个钥匙形状悬浮按钮
- 点击按钮 → 添加账号 → 填写用户名和密码后保存
- 勾选「打开登录页时自动弹出此窗口」，则每次进入登录页都会自动弹出选择
- 鼠标悬停在钥匙按钮上，点击右上角小红叉可关闭助手

---

## 🔧 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 用户名输入框 | CSS 选择器，定位用户名输入框 | `input[name="username"], input[placeholder*="用户" i], ...` |
| 密码输入框 | CSS 选择器，定位密码输入框 | `input[name="password"], input[type="password"], ...` |
| 验证码输入框 | CSS 选择器，定位验证码文本输入框 | `input[name="imageCode"], input[placeholder*="验证码" i], ...` |
| 验证码图片 | CSS 选择器，定位验证码 img 元素 | `img[src*="captcha" i], .captcha-img, ...` |
| 验证码 SVG | CSS 选择器，定位验证码 SVG 元素 | `.imageCodeStyle svg` |
| 验证码刷新按钮 | CSS 选择器，定位刷新按钮 | `.captcha-refresh, .el-icon-refresh, ...` |
| 登录按钮 | CSS 选择器，定位登录提交按钮 | `button[type="submit"], .el-button--primary, ...` |
| 打开登录页自动弹窗 | 是否自动弹出账号选择窗口 | `否` |

---

## 🧠 模型训练与导出

如果你需要重新训练验证码识别模型，可以按照以下步骤操作：

### 环境准备

```bash
# 生成数据集依赖
cd captcha_train
npm install

# 训练依赖
pip install torch torchvision pillow numpy onnxruntime
```

### 生成数据集

```bash
node generate_dataset.js
```

脚本会在 `captcha_train/dataset/` 下生成训练、验证、测试集，每张图片大小为 `150x48`。

### 开始训练

```bash
python train.py
```

训练完成后会自动输出：
- `checkpoints/best.pth` - PyTorch 权重文件
- `checkpoints/captcha_model_browser.onnx` - **浏览器兼容版本 ONNX 模型**（已手动替换 LSTM 为基础算子，适配 ONNX Runtime Web）

### 本地验证

```bash
# 批量验证测试集准确率
python inference.py --model ./checkpoints/captcha_model_browser.onnx --mode test --data-dir ./dataset/test

# 单张图片推理
python inference.py --model ./checkpoints/captcha_model_browser.onnx --mode single --image ./test.png
```

### 部署脚本

将训练好的 `captcha_model_browser.onnx` 上传到你自己的静态文件服务器，然后修改脚本中的 `CAPTCHA_MODEL_URL` 地址。

当前示例中使用的模型托管在：
```javascript
const CAPTCHA_MODEL_URL = 'https://dustcoda.github.io/Dustcoda/captcha_train/checkpoints/captcha_model_browser.onnx';
```

---

## 📝 项目结构

```
monkey/
├── autoLogin/
│   └── auto-login.user.js    # 油猴脚本主文件
├── captcha_train/
│   ├── checkpoints/
│   │   ├── best.pth          # PyTorch 训练权重
│   │   └── captcha_model_browser.onnx  # 浏览器兼容 ONNX
│   ├── generate_dataset.js   # 数据集生成器
│   ├── train.py              # 训练 + 自动导出浏览器兼容模型
│   ├── inference.py          # 本地 ONNX 验证脚本
│   └── package.json
└── README.md
```

---

## ⚠️ 安全提示

- 本脚本使用 `GM_setValue` 在浏览器本地存储用户名和密码，**密码以明文形式保存**
- **请勿在公共或共享设备上使用**，避免敏感信息泄露
- 脚本只在匹配的域名下运行，不会向第三方服务器发送你的账号信息
- 验证码识别推理完全在浏览器本地完成

---

## 📄 License

MIT
