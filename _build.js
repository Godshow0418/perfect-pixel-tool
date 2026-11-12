// 构建：把 pixcore.js 注入模板，输出单文件 HTML
const fs = require('fs');
const path = require('path');
const dir = __dirname;
const core = fs.readFileSync(path.join(dir, 'pixcore.js'), 'utf8');
const tpl = fs.readFileSync(path.join(dir, '_template.html'), 'utf8');
if (!tpl.includes('{{CORE}}')) throw new Error('模板里没有 {{CORE}} 占位符');
const out = tpl.replace('{{CORE}}', core);
fs.writeFileSync(path.join(dir, '完美像素工具.html'), out);
console.log('已生成 完美像素工具.html', (out.length / 1024).toFixed(1) + ' KB');
