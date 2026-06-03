// ==UserScript==
// @name         EasyLink 自动登录助手
// @namespace    http://tampermonkey.net/
// @version      1.0.0
// @description  自动用户选择、表单填充与 SVG 验证码识别（使用本地训练的 ONNX 模型）
// @author       Dustcoda
// @match        https://*.easylink-iot.com/*
// @match        http://localhost:9528/*
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_registerMenuCommand
// @grant        GM_addStyle
// @require      https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js
// @run-at       document-end
// ==/UserScript==

(function () {
  'use strict';

  // 安全提示：本脚本使用 GM_setValue 在本地存储账号密码，密码以明文形式保存。
  // 请勿在公共或共享设备上使用，避免敏感信息泄露风险。

  // ============================================================
  // 配置与常量
  // ============================================================
  const STORAGE_KEY = 'easylink_auto_login_users';
  const CONFIG_KEY = 'easylink_auto_login_config';
  const SITE_HOST = location.host;

  const DEFAULT_SELECTORS = {
    username: 'input[name="username"], input[placeholder*="用户" i], input[placeholder*="账号" i], input[placeholder*="User" i], input[type="text"]:first-of-type',
    password: 'input[name="password"], input[type="password"], input[placeholder*="密码" i], input[placeholder*="Pass" i]',
    captchaInput: 'input[name="imageCode"], input[placeholder*="验证码" i], input[placeholder*="图形码" i]',
    captchaImage: 'img[src*="captcha" i], .captcha-img, .login-captcha, .el-form-item__content img, .verify-code img',
    captchaSvg: '.imageCodeStyle svg',
    captchaRefresh: '.captcha-refresh, .refresh-captcha, .captcha-img + i, .el-icon-refresh, [class*="refresh" i]',
    loginBtn: 'button[type="submit"], .el-button--primary, button.login-btn, .login-form button',
    autoPopup: false
  };

  const CAPTCHA_MODEL_URL = 'https://dustcoda.github.io/Dustcoda/captcha_train/checkpoints/captcha_model_browser.onnx';

  const CHARS_ONNX = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ';
  const IDX2CHAR_ONNX = CHARS_ONNX.split('');
  const BLANK_IDX_ONNX = CHARS_ONNX.length;

  let onnxSession = null;
  let isProcessing = false;

  // ============================================================
  // 工具函数
  // ============================================================

  /**
   * 安全读取存储数据
   */
  function getUsers() {
    try {
      const raw = GM_getValue(STORAGE_KEY, '[]');
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      console.warn('[AutoLogin] 读取用户数据失败:', e);
      return [];
    }
  }

  function saveUsers(users) {
    GM_setValue(STORAGE_KEY, JSON.stringify(users));
  }

  function getConfig() {
    try {
      const raw = GM_getValue(CONFIG_KEY, '{}');
      return Object.assign({}, DEFAULT_SELECTORS, JSON.parse(raw));
    } catch (e) {
      return { ...DEFAULT_SELECTORS };
    }
  }

  function saveConfig(cfg) {
    GM_setValue(CONFIG_KEY, JSON.stringify(cfg));
  }

  /**
   * 等待元素出现
   */
  function waitForElement(selector, timeout = 10000) {
    return new Promise((resolve) => {
      const el = document.querySelector(selector);
      if (el) return resolve(el);

      const observer = new MutationObserver(() => {
        const found = document.querySelector(selector);
        if (found) {
          observer.disconnect();
          resolve(found);
        }
      });

      observer.observe(document.body, { childList: true, subtree: true });

      setTimeout(() => {
        observer.disconnect();
        resolve(null);
      }, timeout);
    });
  }

  /**
   * 触发输入事件（兼容 Vue / React 受控组件）
   */
  function simulateInput(el, value) {
    if (!el) return;
    el.focus();
    el.value = value;

    const events = ['input', 'change', 'blur', 'keyup'];
    events.forEach((type) => {
      const evt = new Event(type, { bubbles: true, cancelable: true });
      el.dispatchEvent(evt);
    });

    // 针对 Vue 的自定义事件
    const tracker = el._valueTracker;
    if (tracker) tracker.setValue('');
  }

  /**
   * 查找元素（支持多个选择器回退）
   */
  function queryElement(selectors) {
    if (typeof selectors === 'string') selectors = selectors.split(',');
    for (const sel of selectors) {
      const s = sel.trim();
      if (!s) continue;
      try {
        const el = document.querySelector(s);
        if (el) return el;
      } catch (e) {
        // 忽略非法选择器
      }
    }
    return null;
  }

  /**
   * 文本模糊匹配按钮
   */
  function findButtonByText(patterns) {
    const btns = document.querySelectorAll('button, input[type="submit"], a.btn');
    for (const btn of btns) {
      const text = (btn.textContent || btn.value || '').toLowerCase();
      for (const p of patterns) {
        if (text.includes(p.toLowerCase())) return btn;
      }
    }
    return null;
  }

  // ============验证码处理模块============

  // 内存缓存：页面生命周期内复用已下载的模型数据
  let _cachedModelBuffer = null;

  /**
   * 初始化 ONNX 模型推理会话
   */
  async function initOnnxSession() {
    if (onnxSession) return onnxSession;
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime Web 未加载');

    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'error';

    // 从网络下载模型（浏览器 HTTP 缓存 + CDN 自动处理重复请求）
    if (!_cachedModelBuffer) {
      console.log('[AutoLogin] 从网络加载 ONNX 模型...');
      const resp = await fetch(CAPTCHA_MODEL_URL);
      if (!resp.ok) throw new Error('模型下载失败: ' + resp.status);
      _cachedModelBuffer = await resp.arrayBuffer();
      console.log('[AutoLogin] 模型下载完成, 大小:', _cachedModelBuffer.byteLength);
    } else {
      console.log('[AutoLogin] 使用内存缓存的 ONNX 模型, 大小:', _cachedModelBuffer.byteLength);
    }

    console.log('[AutoLogin] 正在创建 InferenceSession...');

    // 使用 WASM 后端
    onnxSession = await ort.InferenceSession.create(_cachedModelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled'
    });

    console.log('[AutoLogin] ONNX 模型加载成功');

    // 预计算额外输入（如 LSTM 初始隐藏状态）的形状，供推理时使用
    onnxSession._extraInputDims = {};
    for (const name of onnxSession.inputNames) {
      if (name !== 'input' && onnxSession.inputMetadata && onnxSession.inputMetadata[name]) {
        const dims = onnxSession.inputMetadata[name].dimensions;
        onnxSession._extraInputDims[name] = dims;
      }
    }

    return onnxSession;
  }

  /**
   * 验证码图片转换为 ONNX 输入张式
   */
  function captchaImageToTensor(sourceCanvas, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(sourceCanvas, 0, 0, width, height);
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    const floatData = new Float32Array(1 * 1 * height * width);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        floatData[y * width + x] = gray / 255.0;
      }
    }
    return new ort.Tensor('float32', floatData, [1, 1, height, width]);
  }

  /**
   * CTC 贪婪解码
   */
  function ctcGreedyDecode(probs, T, N, C, blankIdx) {
    const results = [];
    for (let n = 0; n < N; n++) {
      const chars = [];
      let last = -1;
      for (let t = 0; t < T; t++) {
        let maxIdx = 0, maxVal = -Infinity;
        const offset = (t * N + n) * C;
        for (let c = 0; c < C; c++) {
          if (probs[offset + c] > maxVal) {
            maxVal = probs[offset + c];
            maxIdx = c;
          }
        }
        if (maxIdx !== blankIdx && maxIdx !== last) {
          chars.push(IDX2CHAR_ONNX[maxIdx]);
        }
        last = maxIdx;
      }
      results.push(chars.join(''));
    }
    return results;
  }

  /**
   * ONNX 模型推理识别验证码
   */
  async function recognizeCaptchaOnnx(capEl) {
    try {
      const { canvas } = await elementToCanvas(capEl);
      const inputCanvas = document.createElement('canvas');
      inputCanvas.width = 150;
      inputCanvas.height = 48;
      const ictx = inputCanvas.getContext('2d');
      ictx.drawImage(canvas, 0, 0, 150, 48);

      const tensor = captchaImageToTensor(inputCanvas, 150, 48);
      const session = await initOnnxSession();

      const feeds = { input: tensor };
      if (session._extraInputDims && Object.keys(session._extraInputDims).length > 0) {
        for (const [name, dims] of Object.entries(session._extraInputDims)) {
          const resolvedDims = dims.map(d => (typeof d === 'string') ? 1 : d);
          const size = resolvedDims.reduce((a, b) => a * b, 1);
          feeds[name] = new ort.Tensor('float32', new Float32Array(size), resolvedDims);
        }
      }

      const results = await session.run(feeds);
      const output = results.output;
      if (!output) {
        throw new Error('模型输出 "output" 不存在, 可用输出: ' + Object.keys(results).join(', '));
      }
      const [T, N, C] = output.dims;
      const probs = output.data;

      const preds = ctcGreedyDecode(probs, T, N, C, BLANK_IDX_ONNX);
      const code = preds[0];
      if (code) console.log('[AutoLogin] ONNX 识别结果:', code);
      return code;
    } catch (e) {
      console.error('[AutoLogin] ONNX 推理失败:', e.message || e);
      return null;
    }
  }

  /**
   * 将元素转换为 Canvas
   */
  async function elementToCanvas(el) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (el.tagName.toLowerCase() === 'img') {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      await new Promise((resolve, reject) => {
        img.onload = () => { img.onload = null; img.onerror = null; resolve(); };
        img.onerror = () => { img.onload = null; img.onerror = null; reject(new Error('验证码图片加载失败')); };
        img.src = el.src;
      });
      canvas.width = img.naturalWidth || img.width || 120;
      canvas.height = img.naturalHeight || img.height || 40;
      ctx.drawImage(img, 0, 0);
    } else if (el.tagName.toLowerCase() === 'svg') {
      const svgData = new XMLSerializer().serializeToString(el);
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(svgBlob);
      const img = new Image();
      const rect = el.getBoundingClientRect();
      try {
        await new Promise((resolve, reject) => {
          img.onload = () => { img.onload = null; img.onerror = null; resolve(); };
          img.onerror = () => { img.onload = null; img.onerror = null; reject(new Error('验证码 SVG 加载失败')); };
          img.src = url;
        });
        canvas.width = Math.max(rect.width, 120);
        canvas.height = Math.max(rect.height, 40);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      } finally {
        URL.revokeObjectURL(url);
      }
    } else if (el.tagName.toLowerCase() === 'canvas') {
      canvas.width = el.width;
      canvas.height = el.height;
      ctx.drawImage(el, 0, 0);
    } else {
      throw new Error('不支持的验证码元素类型: ' + el.tagName);
    }

    return { canvas, ctx };
  }

  /**
   * 识别验证码
   */
  async function recognizeCaptcha(retryCount = 0) {
    const config = getConfig();
    const MAX_RETRY = 3;

    // 轮询等待验证码元素（SVG 可能由接口异步返回）
    let capEl = null;
    let attempts = 0;
    const maxAttempts = 10; // 最多轮询 10 次，每次 300ms，共约 3 秒
    while (!capEl && attempts < maxAttempts) {
      capEl = queryElement(config.captchaImage) || queryElement(config.captchaSvg);
      if (!capEl) {
        const imgs = document.querySelectorAll('img');
        const svgs = document.querySelectorAll('svg');
        if (imgs.length === 1) capEl = imgs[0];
        else if (svgs.length === 1) capEl = svgs[0];
      }
      if (!capEl) {
        attempts++;
        await delay(300);
      }
    }

    if (!capEl) {
      console.warn('[AutoLogin] 未找到验证码元素');
      return null;
    }

    try {
      const code = await recognizeCaptchaOnnx(capEl);
      if (code && code.length >= 4) {
        return code;
      }

      // 识别结果过短，尝试刷新重试
      if (retryCount < MAX_RETRY) {
        console.log('[AutoLogin] 识别结果过短，尝试刷新重试 (' + (retryCount + 1) + '/' + MAX_RETRY + ')');
        const refreshBtn = queryElement(config.captchaRefresh);
        if (refreshBtn) {
          refreshBtn.click();
          await delay(800);
          return recognizeCaptcha(retryCount + 1);
        }
      }

      console.warn('[AutoLogin] ONNX 识别结果不可靠，等待手动输入');
      return null;
    } catch (e) {
      console.error('[AutoLogin] 验证码识别失败:', e);
      return null;
    }
  }

  // ============================================================
  // UI 样式（玻璃拟态 + 极简深色）
  // ============================================================
  GM_addStyle(`
        .alm-overlay {
            position: fixed;
            inset: 0;
            background: rgba(10, 12, 18, 0.55);
            backdrop-filter: blur(8px);
            z-index: 99998;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.35s ease;
        }
        .alm-overlay.active { opacity: 1; }

        .alm-modal {
            width: 420px;
            max-width: 92vw;
            background: linear-gradient(160deg, #1a1d26 0%, #11131a 100%);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            box-shadow: 0 24px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04);
            padding: 28px 32px 32px;
            color: #e8eaed;
            font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            transform: translateY(16px) scale(0.98);
            transition: transform 0.4s cubic-bezier(0.22, 1, 0.36, 1);
        }
        .alm-overlay.active .alm-modal { transform: translateY(0) scale(1); }

        .alm-title {
            font-size: 20px;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin: 0 0 6px;
            background: linear-gradient(90deg, #a78bfa 0%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .alm-subtitle {
            font-size: 13px;
            color: #7d8597;
            margin-bottom: 22px;
        }

        .alm-user-list { max-height: 260px; }
        .alm-user-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 16px;
            margin-bottom: 10px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.25s ease;
        }
        .alm-user-item:hover {
            background: rgba(255,255,255,0.06);
            border-color: rgba(167, 139, 250, 0.25);
            transform: translateX(4px);
        }
        .alm-user-info { display: flex; align-items: center; gap: 12px; }
        .alm-avatar {
            width: 36px; height: 36px; border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; font-weight: 700; color: #fff;
            flex-shrink: 0;
        }
        .alm-name { font-size: 15px; font-weight: 500; color: #f1f5f9; }
        .alm-desc { font-size: 12px; color: #64748b; margin-top: 2px; }
        .alm-actions { display: flex; gap: 8px; }
        .alm-btn-icon {
            width: 30px; height: 30px; border-radius: 8px;
            border: none; background: rgba(255,255,255,0.05);
            color: #94a3b8; cursor: pointer; display: flex;
            align-items: center; justify-content: center;
            transition: all 0.2s ease;
        }
        .alm-btn-icon:hover { background: rgba(255,255,255,0.1); color: #e2e8f0; }
        .alm-btn-icon.danger:hover { background: rgba(239,68,68,0.15); color: #f87171; }

        .alm-empty {
            text-align: center; padding: 40px 0; color: #475569; font-size: 14px;
        }

        .alm-form-row { margin-bottom: 16px; }
        .alm-label {
            display: block; font-size: 12px; font-weight: 500;
            color: #94a3b8; margin-bottom: 6px; text-transform: uppercase;
            letter-spacing: 0.6px;
        }
        .alm-input {
            width: 100%; box-sizing: border-box;
            padding: 11px 14px; background: rgba(0,0,0,0.25);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px; color: #f8fafc; font-size: 14px;
            outline: none; transition: border-color 0.2s, box-shadow 0.2s;
        }
        .alm-input:focus {
            border-color: rgba(139, 92, 246, 0.5);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
        }
        .alm-input::placeholder { color: #475569; }

        .alm-footer {
            display: flex; justify-content: flex-end; gap: 10px;
            margin-top: 20px;
        }
        .alm-btn {
            padding: 10px 20px; border-radius: 10px; border: none;
            font-size: 14px; font-weight: 500; cursor: pointer;
            transition: all 0.2s ease;
        }
        .alm-btn-primary {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: #fff; box-shadow: 0 4px 14px rgba(99,102,241,0.35);
        }
        .alm-btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99,102,241,0.45); }
        .alm-btn-ghost {
            background: rgba(255,255,255,0.04);
            color: #94a3b8; border: 1px solid rgba(255,255,255,0.06);
        }
        .alm-btn-ghost:hover { background: rgba(255,255,255,0.08); color: #e2e8f0; }

        .alm-fab {
            position: fixed; bottom: 28px; right: 28px;
            width: 52px; height: 52px; border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: #fff; border: none; font-size: 22px;
            cursor: pointer; z-index: 99997;
            box-shadow: 0 8px 28px rgba(99,102,241,0.4);
            display: flex; align-items: center; justify-content: center;
            transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s;
        }
        .alm-fab:hover { transform: scale(1.1) rotate(90deg); box-shadow: 0 12px 36px rgba(99,102,241,0.5); }
        .alm-fab-close {
            position: absolute; top: -2px; left: -2px;
            width: 16px; height: 16px; border-radius: 50%;
            background: rgba(239, 68, 68, 0.9); color: #fff;
            border: none; font-size: 10px; line-height: 1;
            cursor: pointer; display: flex;
            align-items: center; justify-content: center;
            opacity: 0; transition: opacity 0.2s;
            pointer-events: auto; z-index: 99998;
        }
        .alm-fab:hover .alm-fab-close { opacity: 1; }

        .alm-close {
            position: absolute; top: 18px; right: 20px;
            width: 28px; height: 28px; border-radius: 8px;
            border: none; background: rgba(255,255,255,0.04);
            color: #64748b; cursor: pointer; font-size: 18px;
            line-height: 1; display: flex; align-items: center; justify-content: center;
            transition: all 0.2s;
        }
        .alm-close:hover { background: rgba(255,255,255,0.1); color: #e2e8f0; }

        .alm-tag {
            display: inline-block; padding: 2px 8px; border-radius: 6px;
            font-size: 11px; font-weight: 600; margin-left: 8px;
            background: rgba(56, 189, 248, 0.12); color: #38bdf8;
        }
    `);

  // ============================================================
  // UI 渲染
  // ============================================================

  function createOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'alm-overlay';
    return overlay;
  }

  function createModal(title, subtitle) {
    const modal = document.createElement('div');
    modal.className = 'alm-modal';
    modal.innerHTML = `
            <button class="alm-close">&times;</button>
            <div class="alm-title">${title}</div>
            <div class="alm-subtitle">${subtitle}</div>
            <div class="alm-body"></div>
            <div class="alm-footer"></div>
        `;
    modal.querySelector('.alm-close').onclick = () => closeOverlay();
    return modal;
  }

  let _closeTimer = null;
  function closeOverlay() {
    if (_closeTimer) { clearTimeout(_closeTimer); _closeTimer = null; }
    document.querySelectorAll('.alm-overlay').forEach(ov => {
      ov.classList.remove('active');
      _closeTimer = setTimeout(() => ov.remove(), 400);
    });
  }

  function showOverlay(contentBuilder) {
    const ov = createOverlay();
    const modal = createModal('', '');
    contentBuilder(modal);
    ov.appendChild(modal);
    document.body.appendChild(ov);
    // 强制重排以触发过渡
    requestAnimationFrame(() => ov.classList.add('active'));
  }

  // ============================================================
  // 用户选择弹窗
  // ============================================================

  function showUserSelector(onSelect) {
    const users = getUsers();
    showOverlay((modal) => {
      const titleEl = modal.querySelector('.alm-title');
      const subEl = modal.querySelector('.alm-subtitle');
      const body = modal.querySelector('.alm-body');
      const footer = modal.querySelector('.alm-footer');

      titleEl.textContent = '选择登录账号';
      subEl.textContent = users.length ? `已保存 ${users.length} 个账号` : '暂无保存的账号，请添加';

      if (!users.length) {
        body.innerHTML = '<div class="alm-empty">点击右下角按钮或下方「添加账号」开始使用</div>';
      } else {
        const list = document.createElement('div');
        list.className = 'alm-user-list';
        users.forEach((u, idx) => {
          const item = document.createElement('div');
          item.className = 'alm-user-item';
          const avatarLetter = (u.name || u.username || '?').charAt(0).toUpperCase();
          item.innerHTML = `
                        <div class="alm-user-info">
                            <div class="alm-avatar">${avatarLetter}</div>
                            <div>
                                <div class="alm-name">${escapeHtml(u.name || u.username)}</div>
                                <div class="alm-desc">${escapeHtml(u.username)}</div>
                            </div>
                        </div>
                        <div class="alm-actions">
                            <button class="alm-btn-icon" title="编辑">&#9998;</button>
                            <button class="alm-btn-icon danger" title="删除">&#128465;</button>
                        </div>
                    `;
          item.onclick = (e) => {
            if (e.target.closest('.alm-btn-icon')) return;
            closeOverlay();
            onSelect(u);
          };
          item.querySelector('.alm-btn-icon:not(.danger)').onclick = (e) => {
            e.stopPropagation();
            closeOverlay();
            setTimeout(() => showUserEditor(u, idx, () => showUserSelector(onSelect)), 200);
          };
          item.querySelector('.alm-btn-icon.danger').onclick = (e) => {
            e.stopPropagation();
            if (!confirm(`确定删除账号「${u.name || u.username}」吗？`)) return;
            users.splice(idx, 1);
            saveUsers(users);
            closeOverlay();
            setTimeout(() => showUserSelector(onSelect), 200);
          };
          list.appendChild(item);
        });
        body.appendChild(list);
      }

      // 自动弹窗开关
      const config = getConfig();
      const toggleRow = document.createElement('div');
      toggleRow.style.cssText = 'margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.06);';
      toggleRow.innerHTML = `
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:13px;color:#94a3b8;">
            <input type="checkbox" id="alm-autoPopup-toggle" ${config.autoPopup ? 'checked' : ''} style="accent-color:#8b5cf6;">
            打开登录页时自动弹出此窗口
        </label>
      `;
      body.appendChild(toggleRow);
      toggleRow.querySelector('input').addEventListener('change', (e) => {
        const cfg = getConfig();
        cfg.autoPopup = e.target.checked;
        saveConfig(cfg);
      });

      footer.innerHTML = `
                <button class="alm-btn alm-btn-ghost" id="alm-btn-config">配置选择器</button>
                <button class="alm-btn alm-btn-primary" id="alm-btn-add">添加账号</button>
            `;
      footer.querySelector('#alm-btn-add').onclick = () => {
        closeOverlay();
        setTimeout(() => showUserEditor(null, -1, () => showUserSelector(onSelect)), 200);
      };
      footer.querySelector('#alm-btn-config').onclick = () => {
        closeOverlay();
        setTimeout(() => showSelectorConfig(() => showUserSelector(onSelect)), 200);
      };
    });
  }

  // ============================================================
  // 添加/编辑用户
  // ============================================================

  function showUserEditor(user, index, onDone) {
    const isEdit = index >= 0;
    showOverlay((modal) => {
      modal.querySelector('.alm-title').textContent = isEdit ? '编辑账号' : '添加账号';
      modal.querySelector('.alm-subtitle').textContent = isEdit ? '修改已保存的登录信息' : '填写用户名与密码，系统将自动填充';
      const body = modal.querySelector('.alm-body');
      body.innerHTML = `
                <div class="alm-form-row">
                    <label class="alm-label">显示名称</label>
                    <input class="alm-input" id="alm-in-name" placeholder="如：测试账号、生产环境" value="${escapeHtml(user?.name || '')}">
                </div>
                <div class="alm-form-row">
                    <label class="alm-label">用户名</label>
                    <input class="alm-input" id="alm-in-user" placeholder="登录用户名" value="${escapeHtml(user?.username || '')}">
                </div>
                <div class="alm-form-row">
                    <label class="alm-label">密码</label>
                    <input class="alm-input" id="alm-in-pass" type="password" placeholder="登录密码" value="${escapeHtml(user?.password || '')}">
                </div>
            `;
      const footer = modal.querySelector('.alm-footer');
      footer.innerHTML = `
                <button class="alm-btn alm-btn-ghost" id="alm-btn-cancel">取消</button>
                <button class="alm-btn alm-btn-primary" id="alm-btn-save">保存</button>
            `;
      footer.querySelector('#alm-btn-cancel').onclick = () => { closeOverlay(); setTimeout(onDone, 200); };
      footer.querySelector('#alm-btn-save').onclick = () => {
        const name = document.getElementById('alm-in-name').value.trim();
        const username = document.getElementById('alm-in-user').value.trim();
        const password = document.getElementById('alm-in-pass').value;
        if (!username || !password) {
          alert('用户名和密码不能为空');
          return;
        }
        const users = getUsers();
        const record = { name: name || username, username, password };
        if (isEdit) users[index] = record;
        else users.push(record);
        saveUsers(users);
        closeOverlay();
        setTimeout(onDone, 200);
      };
    });
  }

  // ============================================================
  // 选择器配置
  // ============================================================

  function showSelectorConfig(onDone) {
    const cfg = getConfig();
    showOverlay((modal) => {
      modal.querySelector('.alm-title').textContent = '脚本配置';
      modal.querySelector('.alm-subtitle').textContent = '调整自动填充行为与元素定位规则';
      const body = modal.querySelector('.alm-body');
      const fields = [
        { key: 'username', label: '用户名输入框', val: cfg.username },
        { key: 'password', label: '密码输入框', val: cfg.password },
        { key: 'captchaInput', label: '验证码输入框', val: cfg.captchaInput },
        { key: 'captchaImage', label: '验证码图片 (img)', val: cfg.captchaImage },
        { key: 'captchaSvg', label: '验证码 SVG', val: cfg.captchaSvg },
        { key: 'captchaRefresh', label: '验证码刷新按钮', val: cfg.captchaRefresh },
        { key: 'loginBtn', label: '登录按钮', val: cfg.loginBtn }
      ];
      body.innerHTML = fields.map(f => `
                <div class="alm-form-row">
                    <label class="alm-label">${f.label}</label>
                    <input class="alm-input" id="alm-cfg-${f.key}" value="${escapeHtml(f.val)}">
                </div>
            `).join('');
      const footer = modal.querySelector('.alm-footer');
      footer.innerHTML = `
                <button class="alm-btn alm-btn-ghost" id="alm-btn-cancel">返回</button>
                <button class="alm-btn alm-btn-primary" id="alm-btn-save">保存</button>
            `;
      footer.querySelector('#alm-btn-cancel').onclick = () => { closeOverlay(); setTimeout(onDone, 200); };
      footer.querySelector('#alm-btn-save').onclick = () => {
        const newCfg = { ...DEFAULT_SELECTORS };
        fields.forEach(f => {
          newCfg[f.key] = document.getElementById(`alm-cfg-${f.key}`).value.trim() || DEFAULT_SELECTORS[f.key];
        });
        saveConfig(newCfg);
        closeOverlay();
        setTimeout(onDone, 200);
      };
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
  }

  // ============================================================
  // 悬浮按钮
  // ============================================================

  function injectFab() {
    if (document.getElementById('alm-fab')) return;
    const fab = document.createElement('button');
    fab.id = 'alm-fab';
    fab.className = 'alm-fab';
    fab.innerHTML = '&#128273;<span class="alm-fab-close" title="关闭助手">&#10005;</span>';
    fab.title = '自动登录助手';
    fab.onclick = (e) => {
      if (e.target.closest('.alm-fab-close')) {
        e.stopPropagation();
        fab.remove();
        return;
      }
      showUserSelector((user) => performAutoLogin(user, true));
    };
    document.body.appendChild(fab);
  }

  // ============================================================
  // 核心登录逻辑
  // ============================================================

  /**
   * 执行自动登录核心流程
   * @param {Object} user 用户信息
   * @param {boolean} interactive 是否交互模式（用户手动触发）
   */
  async function performAutoLogin(user, interactive = false) {
    if (isProcessing) return;
    isProcessing = true;

    const config = getConfig();
    console.log('[AutoLogin] 开始自动登录流程:', user.name);

    try {
      // 1. 等待并填充用户名
      const userEl = await waitForElement(config.username.split(',')[0]);
      if (userEl) simulateInput(userEl, user.username);
      await delay(200);

      // 2. 填充密码
      const passEl = queryElement(config.password);
      if (passEl) simulateInput(passEl, user.password);
      await delay(200);

      // 3. 识别并填充验证码
      let capReady = true;
      const capInput = queryElement(config.captchaInput);
      let code = null;
      if (capInput) {
        code = await recognizeCaptcha();
        if (code && code.length >= 3) {
          simulateInput(capInput, code);
        } else {
          capReady = false;
          console.warn('[AutoLogin] 未能可靠识别验证码，等待手动输入或刷新');
        }
      }

      await delay(200);

      // 4. 自动提交策略
      // - 交互模式下：总是自动点击登录（用户明确选择了账号）
      // - 自动模式下：仅当验证码识别成功时才自动提交，否则等待用户确认
      const shouldSubmit = interactive || capReady;
      if (shouldSubmit) {
        const btn = queryElement(config.loginBtn) || findButtonByText(['登录', 'login', '登入', 'sign in']);
        if (btn) {
          btn.click();
          console.log('[AutoLogin] 已自动点击登录按钮');
        } else {
          console.warn('[AutoLogin] 未找到登录按钮');
        }
      }
    } catch (e) {
      console.error('[AutoLogin] 自动登录异常:', e);
    } finally {
      isProcessing = false;
    }
  }

  function delay(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  // ============================================================
  // 入口与初始化
  // ============================================================

  function isLoginPage() {
    const path = location.hash || location.pathname;
    return /login|signin|auth/i.test(path);
  }

  let _popupTimer = null;
  function init() {
    injectFab();

    if (!isLoginPage()) return;

    const config = getConfig();
    if (!config.autoPopup) return;

    if (_popupTimer) clearTimeout(_popupTimer);

    const users = getUsers();
    if (!users.length) {
      // 无用户时，延迟弹出添加引导
      _popupTimer = setTimeout(() => {
        _popupTimer = null;
        showUserSelector((u) => performAutoLogin(u, true));
      }, 600);
    } else {
      // 有用户时弹出选择
      _popupTimer = setTimeout(() => {
        _popupTimer = null;
        showUserSelector((u) => performAutoLogin(u, true));
      }, 600);
    }
  }

  // SPA 路由兼容：监听 hashchange
  window.addEventListener('hashchange', () => {
    if (isLoginPage()) setTimeout(init, 400);
  });

  // 监听 DOM 变化，兼容 history 路由或同 hash 下的页面重绘（如退出登录后重绘登录页）
  let _lastWasLoginPage = isLoginPage();
  const _spaObserver = new MutationObserver(() => {
    const nowLogin = isLoginPage();
    if (nowLogin && !_lastWasLoginPage) {
      _lastWasLoginPage = true;
      setTimeout(init, 400);
    } else if (!nowLogin) {
      _lastWasLoginPage = false;
    }
  });
  if (document.body) {
    _spaObserver.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  GM_registerMenuCommand('打开账号管理', () => {
    showUserSelector((u) => performAutoLogin(u, true));
  });

  GM_registerMenuCommand('配置 DOM 选择器', () => {
    showSelectorConfig(() => { });
  });

  console.log('[AutoLogin] 智能登录助手已加载');
})();
