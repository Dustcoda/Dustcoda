/**
 * 验证码数据集生成器
 * 使用 svg-captcha 库，参数与后端完全一致，确保训练/推理分布相同
 *
 * 依赖安装：npm install svg-captcha sharp
 * 运行：node generate_dataset.js
 */

const fs = require('fs');
const path = require('path');
const svgCaptcha = require('svg-captcha');
const sharp = require('sharp');

const CONFIG = {
  total: 12000,        // 总样本数（建议 8000 训练 + 2000 验证 + 2000 测试）
  trainRatio: 0.7,
  valRatio: 0.15,
  outputDir: path.join(__dirname, 'dataset'),
  // 与后端 gen_image_code 完全一致的参数
  captchaOptions: {
    inverse: false,
    ignoreChars: '0oO1iIl',
    fontSize: 48,
    noise: 2,
    width: 150,
    height: 48,
    color: false,
    size: 4
  }
};

async function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

async function generateSample(index) {
  const codeInfo = svgCaptcha.create(CONFIG.captchaOptions);
  const label = codeInfo.text.toUpperCase();

  // 将 SVG 转为 PNG（150x48）
  const pngBuffer = await sharp(Buffer.from(codeInfo.data))
    .png()
    .toBuffer();

  return { buffer: pngBuffer, label, index };
}

async function main() {
  await ensureDir(CONFIG.outputDir);

  const trainDir = path.join(CONFIG.outputDir, 'train');
  const valDir = path.join(CONFIG.outputDir, 'val');
  const testDir = path.join(CONFIG.outputDir, 'test');
  await ensureDir(trainDir);
  await ensureDir(valDir);
  await ensureDir(testDir);

  const trainList = [];
  const valList = [];
  const testList = [];

  console.log(`开始生成 ${CONFIG.total} 张验证码样本...`);
  const startTime = Date.now();

  for (let i = 0; i < CONFIG.total; i++) {
    const { buffer, label } = await generateSample(i);

    let subset, list, name;
    const r = Math.random();
    if (r < CONFIG.trainRatio) {
      subset = trainDir;
      list = trainList;
      name = `train_${String(i).padStart(6, '0')}.png`;
    } else if (r < CONFIG.trainRatio + CONFIG.valRatio) {
      subset = valDir;
      list = valList;
      name = `val_${String(i).padStart(6, '0')}.png`;
    } else {
      subset = testDir;
      list = testList;
      name = `test_${String(i).padStart(6, '0')}.png`;
    }

    fs.writeFileSync(path.join(subset, name), buffer);
    list.push(`${name},${label}`);

    if ((i + 1) % 500 === 0) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`  进度: ${i + 1}/${CONFIG.total}  (${elapsed}s)`);
    }
  }

  // 写入标签文件
  fs.writeFileSync(path.join(trainDir, 'labels.txt'), trainList.join('\n'));
  fs.writeFileSync(path.join(valDir, 'labels.txt'), valList.join('\n'));
  fs.writeFileSync(path.join(testDir, 'labels.txt'), testList.join('\n'));

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log('\n生成完成！');
  console.log(`  训练集: ${trainList.length} 张  -> ${trainDir}`);
  console.log(`  验证集: ${valList.length} 张   -> ${valDir}`);
  console.log(`  测试集: ${testList.length} 张   -> ${testDir}`);
  console.log(`  耗时: ${totalTime}s`);
  console.log('\n下一步: 运行 python train.py');
}

main().catch(err => {
  console.error('生成失败:', err.message);
  process.exit(1);
});
