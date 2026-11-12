// 回归测试：完美像素 / 像素块 / 色表对标（不依赖浏览器，node 直接跑）
const path = require('path');
const P = require(path.join(__dirname, 'pixcore.js'));
const img = (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) });
let pass = 0, fail = 0;
const check = (ok, label, extra = '') => { ok ? pass++ : fail++; console.log(`${ok ? 'OK ' : 'BAD'} ${label} ${extra}`); };

const PAL = [[35, 29, 48], [103, 127, 84], [113, 142, 74], [200, 60, 60], [240, 220, 160], [20, 20, 24], [80, 160, 200], [210, 120, 60]];
function rnd(seed) { let s = seed >>> 0; return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }

/** 造一张"逻辑像素"图，再按 pitch 放大 —— 模拟 AI 出的大图 */
function makeArt(logicalW, logicalH, pitch, seed) {
  const W = logicalW * pitch, H = logicalH * pitch;
  const im = img(W, H), R = rnd(seed);
  const cells = [];
  for (let cy = 0; cy < logicalH; cy++) for (let cx = 0; cx < logicalW; cx++) {
    const inBody = ((cx - logicalW / 2) / (logicalW * 0.42)) ** 2 + ((cy - logicalH / 2) / (logicalH * 0.45)) ** 2 < 1;
    cells.push(!inBody ? null : (R() < 0.35 ? [...PAL[0], 255] : [...PAL[1 + Math.floor(R() * 6)], 255]));
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = cells[Math.floor(y / pitch) * logicalW + Math.floor(x / pitch)];
    const i = (y * W + x) * 4;
    if (c) { im.data[i] = c[0]; im.data[i + 1] = c[1]; im.data[i + 2] = c[2]; im.data[i + 3] = 255; }
  }
  return im;
}
function blur(src, t = 1) {
  let cur = src;
  for (let k = 0; k < t; k++) {
    const o = img(cur.width, cur.height);
    for (let y = 0; y < cur.height; y++) for (let x = 0; x < cur.width; x++) {
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = Math.min(cur.width - 1, Math.max(0, x + dx)), yy = Math.min(cur.height - 1, Math.max(0, y + dy));
        const i = (yy * cur.width + xx) * 4;
        r += cur.data[i]; g += cur.data[i + 1]; b += cur.data[i + 2]; a += cur.data[i + 3]; n++;
      }
      const i = (y * cur.width + x) * 4;
      o.data[i] = r / n; o.data[i + 1] = g / n; o.data[i + 2] = b / n; o.data[i + 3] = a / n;
    }
    cur = o;
  }
  return cur;
}
function sameImage(a, b, tol = 4) {
  if (a.width !== b.width || a.height !== b.height) return false;
  for (let i = 0; i < a.data.length; i += 4) {
    if (a.data[i + 3] < 128 && b.data[i + 3] < 128) continue;
    if (Math.abs(a.data[i] - b.data[i]) > tol || Math.abs(a.data[i + 1] - b.data[i + 1]) > tol) return false;
  }
  return true;
}

/* ============ A. 完美像素：各种尺寸/倍数都要还原 ============ */
console.log('--- A. 完美像素 ---');
const cases = [[24, 32, 8, 7], [50, 97, 8, 11], [64, 64, 6, 13], [48, 48, 16, 17], [20, 20, 4, 19], [32, 40, 12, 23]];
for (const [lw, lh, pitch, seed] of cases) {
  const art = makeArt(lw, lh, pitch, seed);
  for (const [tag, src] of [['硬边', art], ['糊边', blur(art, 1)]]) {
    const t = Date.now();
    const det = P.detectGrid(src);
    const o = P.perfectPixel(src, { fixSquare: false });
    // 已知限制：4px 块叠 3x3 模糊时自动检测不可靠，此时只要求「安全地不动图」
    const hardCase = pitch <= 4 && tag === '糊边';
    const ok = hardCase
      ? (o.width === lw && o.height === lh) || (o.noGrid === true && o.width === src.width)
      : o.width === lw && o.height === lh;
    check(ok, `${lw}x${lh} @${pitch}x ${tag}${hardCase ? '（已知限制：可退回原图）' : ''}`,
      `检测 ${det.cols}x${det.rows} px=${det.pixelX.toFixed(2)}/${det.pixelY.toFixed(2)} → ${o.width}x${o.height} ${Date.now() - t}ms`);
  }
}

// 超 512 / 超 2048
const huge = makeArt(50, 97, 32, 31);   // 1600 x 3104
const t1 = Date.now();
const oh = P.perfectPixel(huge, { fixSquare: false });
check(oh.width === 50 && oh.height === 97, `${huge.width}x${huge.height} (远超 2048) -> ${oh.width}x${oh.height}`, `${Date.now() - t1}ms`);

// 已是 1:1 像素图：必须原样返回
const native = makeArt(50, 97, 1, 37);
const on = P.perfectPixel(native, { fixSquare: false });
check(on.noGrid === true && sameImage(on, native), `1:1 像素图原样放过`, `(${on.width}x${on.height})`);

// 写实图（无周期）：必须原样返回，不能切碎
const photo = img(320, 240);
{ let s = 12345; const R = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  for (let y = 0; y < 240; y++) for (let x = 0; x < 320; x++) {
    const i = (y * 320 + x) * 4;
    const v = 40 + 160 * Math.sin(x / 37) * Math.cos(y / 53) + R() * 30;
    photo.data[i] = v; photo.data[i + 1] = v * 0.9; photo.data[i + 2] = 200 - v * 0.4; photo.data[i + 3] = 255;
  } }
const op = P.perfectPixel(photo, { fixSquare: false });
check(op.noGrid === true && op.width === 320 && op.height === 240, `写实图不被切碎 -> ${op.width}x${op.height}`);

// 手动网格
const manual = P.perfectPixel(makeArt(50, 97, 8, 41), { gridSize: { cols: 25, rows: 48 }, fixSquare: false });
check(manual.width === 25 && manual.height === 48, `手动网格 25x48 -> ${manual.width}x${manual.height}`);

/* ============ B. 像素块 ============ */
console.log('\n--- B. 像素块 ---');
const art48 = makeArt(48, 48, 8, 43);      // 384x384
const keep = P.pixelateImage(art48, { block: 8, keepSize: true });
check(keep.width === 384 && keep.height === 384, `保持原尺寸 384x384 -> ${keep.width}x${keep.height}`);
const small = P.pixelateImage(art48, { block: 8, keepSize: false });
check(small.width === 48 && small.height === 48, `块 8 缩回逻辑尺寸 -> ${small.width}x${small.height}`);
const ref48 = makeArt(48, 48, 1, 43);
check(sameImage(small, ref48, 6), `块 8 缩回后与母版一致`);
// 块中心取色
const centerMode = P.pixelateImage(art48, { block: 8, keepSize: false, resample: 'center' });
check(sameImage(centerMode, ref48, 6), `块中心取色也一致`);
// 透明边缘不能变黑
const alphaArt = makeArt(48, 48, 8, 47);
let dark = 0, semi = 0;
for (let i = 0; i < small.data.length; i += 4) {
  if (small.data[i + 3] === 0 && (small.data[i] > 8 || small.data[i + 1] > 8)) dark++;
  if (small.data[i + 3] > 0 && small.data[i + 3] < 250) semi++;
}
check(dark === 0, `透明区不带黑底`, `(透明像素颜色非黑: ${dark}, 半透明: ${semi})`);
// 极端块
const b2 = P.pixelateImage(art48, { block: 2, keepSize: false });
check(b2.width === 192, `块 2 -> ${b2.width}x${b2.height}`);

/* ============ C. 色表对标 ============ */
console.log('\n--- C. 色表对标 ---');
const grad = img(128, 128);
for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) {
  const i = (y * 128 + x) * 4;
  grad.data[i] = x * 2; grad.data[i + 1] = y * 2; grad.data[i + 2] = 128; grad.data[i + 3] = 255;
}
for (const n of [8, 16, 32, 64, 128, 256]) {
  const pal = P.medianCutPalette(grad, n);
  const m = P.mapToPalette(grad, pal, { method: 'lab', dither: 0 });
  check(Math.abs(P.countColors(m).length - n) <= 1, `自动提 ${n} 色 -> ${P.countColors(m).length} 色`);
}
const custom = PAL.slice();
const mc = P.mapToPalette(grad, custom, { method: 'lab', dither: 0 });
check(P.countColors(mc).length <= custom.length, `自定义 ${custom.length} 色 -> ${P.countColors(mc).length} 色`);
const mf = P.mapToPalette(grad, custom, { method: 'lab', dither: 1 });
check(P.countColors(mf).length <= custom.length, `抖动后不超色表 -> ${P.countColors(mf).length} 色`);
// 抖动确实改变了画面（不是空开关）
let changed = 0;
for (let i = 0; i < mc.data.length; i += 4) if (mc.data[i] !== mf.data[i]) changed++;
check(changed > 500, `抖动生效（${changed} 个像素变化）`);
// 透明保护
const mAlpha = P.mapToPalette(art48, custom, { method: 'lab', dither: 0 });
let trans = 0;
for (let i = 3; i < mAlpha.data.length; i += 4) if (mAlpha.data[i] === 0) trans++;
check(trans > 1000, `透明底保持（${trans} 个透明像素）`);
// 色表数量上限
const bigPal = P.medianCutPalette(grad, 256);
check(bigPal.length === 256, `256 色表完整提取 -> ${bigPal.length} 色`);

/* ============ D. 导出 ============ */
console.log('\n--- D. 导出 ---');
const up = P.nearestScale(small, 4);
check(up.width === small.width * 4 && up.height === small.height * 4, `最近邻 4x -> ${up.width}x${up.height}`);
let blocky = true;
for (let y = 0; y < small.height && blocky; y++) for (let x = 1; x < 4; x++) {
  const a = (y * up.width + x) * 4, b = (y * up.width + 0) * 4;
  if (up.data[a] !== up.data[b]) { blocky = false; break; }
}
check(blocky, `放大后像素块保持硬边`);

/* ================= E. 抠图 ================= */
console.log('\n--- E. 抠图 ---');

// E1 白底 + 中间色块
{
  const im = img(20, 20);
  for (let y = 0; y < 20; y++) for (let x = 0; x < 20; x++) {
    const i = (y * 20 + x) * 4;
    const inner = x >= 6 && x < 14 && y >= 6 && y < 14;
    const c = inner ? [180, 40, 40] : [255, 255, 255];
    im.data[i] = c[0]; im.data[i + 1] = c[1]; im.data[i + 2] = c[2]; im.data[i + 3] = 255;
  }
  const edge = P.sampleEdgeColors(im, 2);
  check(edge.length === 1 && edge[0].rgb.join(',') === '255,255,255', `四边取样白底 -> ${edge.map(e => e.rgb.join(',')).join('|')}`);
  const r = P.removeBackground(im, { tolerance: 20 });
  const px = (x, y) => r.imageData.data[(y * 20 + x) * 4 + 3];
  check(px(0, 0) === 0 && px(19, 19) === 0, `四角变透明 (${px(0, 0)},${px(19, 19)})`);
  check(px(10, 10) === 255, `中间色块保留 alpha=${px(10, 10)}`);
  check(r.removed === 400 - 64, `抠掉像素数 ${r.removed}（应为 336）`);
}

// E2 棋盘格背景（两色）
{
  const im = img(24, 24);
  for (let y = 0; y < 24; y++) for (let x = 0; x < 24; x++) {
    const i = (y * 24 + x) * 4;
    const inner = x >= 8 && x < 16 && y >= 8 && y < 16;
    const check2 = (Math.floor(x / 4) + Math.floor(y / 4)) % 2 === 1;
    const c = inner ? [40, 90, 160] : (check2 ? [222, 222, 222] : [255, 255, 255]);
    im.data[i] = c[0]; im.data[i + 1] = c[1]; im.data[i + 2] = c[2]; im.data[i + 3] = 255;
  }
  const edge = P.sampleEdgeColors(im, 2);
  check(edge.length === 2, `棋盘格四边取样出 2 色：${edge.map(e => e.rgb.join(',')).join(' | ')}`);
  const r = P.removeBackground(im, { tolerance: 20 });
  let left = 0;
  for (let y = 8; y < 16; y++) for (let x = 8; x < 16; x++) if (r.imageData.data[(y * 24 + x) * 4 + 3] === 0) left++;
  check(left === 0, `棋盘格全部抠净，色块残留透明像素 ${left}`);
  check(r.imageData.data[(0 * 24 + 0) * 4 + 3] === 0, '棋盘格角透明');
}

// E3 边缘连通：内部同色洞不该被挖掉
{
  const im = img(20, 20);
  for (let y = 0; y < 20; y++) for (let x = 0; x < 20; x++) {
    const i = (y * 20 + x) * 4;
    const ring = x < 4 || x > 15 || y < 4 || y > 15;         // 外圈白
    const hole = x >= 9 && x < 11 && y >= 9 && y < 11;       // 内部 2x2 白点
    const c = (ring || hole) ? [255, 255, 255] : [30, 120, 60];
    im.data[i] = c[0]; im.data[i + 1] = c[1]; im.data[i + 2] = c[2]; im.data[i + 3] = 255;
  }
  const on = P.removeBackground(im, { tolerance: 20, edgeConnected: true, lightFringe: false });
  check(on.imageData.data[(9 * 20 + 9) * 4 + 3] === 255, '边缘连通模式下，内部同色洞被保留');
  const off = P.removeBackground(im, { tolerance: 20, edgeConnected: false, lightFringe: false, despeckle: false });
  check(off.imageData.data[(9 * 20 + 9) * 4 + 3] === 0, '关掉边缘连通后，内部同色洞也被挖掉');
}

// E4 容差=0 只吃完全同色的像素
{
  const im = img(10, 10);
  for (let y = 0; y < 10; y++) for (let x = 0; x < 10; x++) {
    const i = (y * 10 + x) * 4;
    const c = x < 5 ? [255, 255, 255] : [252, 252, 252];
    im.data[i] = c[0]; im.data[i + 1] = c[1]; im.data[i + 2] = c[2]; im.data[i + 3] = 255;
  }
  const r = P.removeBackground(im, { tolerance: 0, lightFringe: false });
  check(r.imageData.data[(5 * 10 + 8) * 4 + 3] === 255, '容差 0：252,252,252 不被当白底');
}


/* ---------- 去除杂色 ---------- */
{
  const w = 20, h = 20, m = img(w, h);
  const put = (x, y, c) => { const i = (y * w + x) * 4; m.data[i] = c[0]; m.data[i + 1] = c[1]; m.data[i + 2] = c[2]; m.data[i + 3] = 255; };
  const al = (im, x, y) => im.data[(y * w + x) * 4 + 3];
  for (let y = 5; y < 15; y++) for (let x = 5; x < 15; x++) put(x, y, [100, 100, 100]);   // 主体 100px
  for (let y = 1; y < 3; y++) for (let x = 1; x < 3; x++) put(x, y, [200, 50, 50]);        // 4px 孤立碎块
  put(18, 18, [0, 255, 255]);                                                             // 1px 孤立碎块

  const r1 = P.removeNoise(m, { minArea: 16, dots: false });
  check(r1.blocks === 2 && r1.pixels === 5 && al(r1.imageData, 1, 1) === 0 && al(r1.imageData, 10, 10) === 255,
    `去杂色：删 ${r1.blocks} 块 / ${r1.pixels}px，主体完好`);

  const r0 = P.removeNoise(m, { minArea: 0, dots: false });
  check(r0.pixels === 0 && al(r0.imageData, 1, 1) === 255, `去杂色 minArea=0 不删任何块`);

  // 主体内部一个异色单点：邻居全都不像它 -> 当杂点删掉
  const m2 = img(w, h);
  for (let y = 5; y < 15; y++) for (let x = 5; x < 15; x++) { const i = (y * w + x) * 4; m2.data[i] = 100; m2.data[i + 1] = 100; m2.data[i + 2] = 100; m2.data[i + 3] = 255; }
  { const i = (10 * w + 10) * 4; m2.data[i] = 255; m2.data[i + 1] = 0; m2.data[i + 2] = 255; m2.data[i + 3] = 255; }
  const r2 = P.removeNoise(m2, { minArea: 0, dots: true, dotTol: 36 });
  check(r2.dotPixels === 1 && al(r2.imageData, 10, 10) === 0 && al(r2.imageData, 9, 10) === 255,
    `去杂色（单像素）：清掉 1 个离群色点，四周保留`);

  // 主体边缘不该被当成杂点
  const m3 = img(w, h);
  for (let y = 5; y < 15; y++) for (let x = 5; x < 15; x++) { const i = (y * w + x) * 4; m3.data[i] = 100; m3.data[i + 1] = 100; m3.data[i + 2] = 100; m3.data[i + 3] = 255; }
  const r3 = P.removeNoise(m3, { minArea: 0, dots: true });
  check(r3.dotPixels === 0 && al(r3.imageData, 5, 5) === 255, `去杂色（单像素）：纯色主体一个点都不动`);
}

/* ---------- 描边一圈 ---------- */
{
  const w = 11, h = 11;
  const mk = () => { const m = img(w, h); for (let y = 4; y < 7; y++) for (let x = 4; x < 7; x++) { const i = (y * w + x) * 4; m.data[i] = 100; m.data[i + 1] = 100; m.data[i + 2] = 100; m.data[i + 3] = 255; } return m; };
  const A = (im, x, y) => im.data[(y * w + x) * 4 + 3];
  const RGB = (im, x, y) => { const i = (y * w + x) * 4; return im.data[i] + ',' + im.data[i + 1] + ',' + im.data[i + 2]; };

  const m0 = mk();
  const r1 = P.strokeOutline(m0);
  check(r1.pixels === 16, `描边：3×3 方块外描边出 16 像素（5×5 − 3×3）`, `实得 ${r1.pixels}`);
  check(A(r1.imageData, 3, 3) === 255 && RGB(r1.imageData, 3, 3) === '26,26,34', `描边：默认色是近黑 #1a1a22`);
  check(RGB(r1.imageData, 5, 5) === '100,100,100' && A(r1.imageData, 0, 0) === 0, `描边：主体本身一个像素没动，远处仍透明`);

  const r2 = P.strokeOutline(r1.imageData);
  check(r2.pixels === 24, `描边：连点两次 = 2 像素粗（7×7 − 5×5 = 24）`, `实得 ${r2.pixels}`);

  const r3 = P.strokeOutline(mk(), { color: [200, 60, 60], mode: 'in' });
  check(r3.pixels === 8 && RGB(r3.imageData, 4, 4) === '200,60,60' && RGB(r3.imageData, 5, 5) === '100,100,100',
    `描边（往内）：3×3 只剩中心是原色，外圈 8 个变描边色`, `实得 ${r3.pixels}`);

  // 选区限制
  const mask = new Uint8Array(w * h); for (let y = 0; y < 6; y++) for (let x = 0; x < w; x++) mask[y * w + x] = 1;
  const r4 = P.strokeOutline(mk(), { mask });
  check(r4.pixels === 9 && A(r4.imageData, 3, 8) === 0, `描边：有选区时只描选区里（上半 9 个，下半不动）`, `实得 ${r4.pixels}`);

  const blank = P.strokeOutline(img(w, h));
  check(blank.pixels === 0, `描边：全透明图描不出任何东西`);
}

/* ---------- 静态检查：JS 里 $('#id') 引用的元素必须真的存在 ---------- */
/* 删 DOM 元素时忘了删 JS 引用会抛 TypeError，把整个 loadFile 打断（原图/结果窗口全空） */
{
  const fs = require('fs');
  const html = fs.readFileSync(path.join(__dirname, '完美像素工具.html'), 'utf8');
  const ids = new Set();
  for (const m of html.matchAll(/\sid="([^"]+)"/g)) ids.add(m[1]);
  const refs = new Set();
  for (const m of html.matchAll(/\$\('#([A-Za-z0-9_-]+)'\)/g)) refs.add(m[1]);
  const missing = [...refs].filter((x) => !ids.has(x));
  check(missing.length === 0, `DOM 引用完整：${refs.size} 个 $('#id') 全部有对应元素`,
    missing.length ? '← 悬空：' + missing.join(', ') : '');
}


/* ================= F. GIF 编码 ================= */
{
  const frames = [];
  for (let f = 0; f < 3; f++) {
    const im = img(32, 24);
    for (let y = 0; y < 24; y++) for (let x = 0; x < 32; x++) {
      const i = (y * 32 + x) * 4;
      const inside = x > f * 4 && x < 28 && y > 6 && y < 18;
      if (inside) { const c = PAL[1 + (f % 6)]; im.data[i] = c[0]; im.data[i + 1] = c[1]; im.data[i + 2] = c[2]; im.data[i + 3] = 255; }
    }
    frames.push({ imageData: im, delay: 80 });
  }
  const gif = P.encodeGIF(frames, { loop: 0 });
  check(gif.length > 100, 'GIF 编码输出非空', gif.length + 'B');
  check(String.fromCharCode(...gif.slice(0, 6)) === 'GIF89a', 'GIF 签名 GIF89a');
  check(gif[gif.length - 1] === 0x3b, 'GIF 结尾 0x3B');
  check(gif[6] === 32 && gif[7] === 0 && gif[8] === 24 && gif[9] === 0, 'GIF 宽高字段 32×24');
  let gce = 0;
  for (let i = 0; i + 8 < gif.length; i++) if (gif[i] === 0x21 && gif[i + 1] === 0xf9 && gif[i + 2] === 0x04) gce++;
  check(gce === 3, 'GIF 图形控制块数 = 帧数', gce + ' 个');
  let desc = 0;
  for (let i = 0; i + 9 < gif.length; i++) if (gif[i] === 0x2c && gif[i + 1] === 0 && gif[i + 2] === 0) desc++;
  check(desc === 3, 'GIF 图像描述符数 = 帧数', desc + ' 个');

  // 不透明帧（没有透明像素 → 不该有透明索引标记）
  const solid = [];
  for (let f = 0; f < 2; f++) {
    const im = img(8, 8);
    for (let i = 0; i < im.data.length; i += 4) {
      const c = PAL[f];
      im.data[i] = c[0]; im.data[i + 1] = c[1]; im.data[i + 2] = c[2]; im.data[i + 3] = 255;
    }
    solid.push({ imageData: im, delay: 100 });
  }
  const g2 = P.encodeGIF(solid, { loop: 0 });
  const packed1 = g2[g2.indexOf(0x21, 30)];
  check(g2.length > 50, '不透明 GIF 也能编', g2.length + 'B');

  // 尺寸不一致必须报错
  let threw = false;
  try { P.encodeGIF([{ imageData: img(8, 8), delay: 100 }, { imageData: img(8, 4), delay: 100 }]); } catch (e) { threw = true; }
  check(threw, '不同尺寸的帧会报错');

  // LZW 压缩率
  const idx = new Uint8Array(4000);
  for (let i = 0; i < idx.length; i++) idx[i] = i % 16;
  const lzw = P.gifLZW(idx, 4);
  check(lzw.length > 0 && lzw.length < idx.length / 4, 'LZW 压缩率正常', `${idx.length} → ${lzw.length} 字节`);

  // ── 背景色自动取样（白边污染时退到画面主色） ──
  // 造一张：四周 6px 白边，中间一大块洋红，正中放个小方块当主体
  const border = img(120, 120);
  for (let y = 0; y < 120; y++) {
    for (let x = 0; x < 120; x++) {
      const i = (y * 120 + x) * 4;
      const inBorder = x < 6 || y < 6 || x >= 114 || y >= 114;
      const inSubject = x >= 50 && x < 70 && y >= 50 && y < 70;
      const c = inSubject ? [40, 40, 50] : inBorder ? [255, 255, 255] : [249, 6, 241];
      border.data[i] = c[0]; border.data[i + 1] = c[1]; border.data[i + 2] = c[2]; border.data[i + 3] = 255;
    }
  }
  const guess = P.guessBackgroundColors(border);
  check(guess.guessed === true, '白边图：判定要用画面主色');
  check(guess.colors.length === 1 && guess.colors[0][0] > 200 && guess.colors[0][1] < 60 && guess.colors[0][2] > 200,
    '白边图：猜到的背景是洋红', JSON.stringify(guess.colors));

  // 纯白底图不该改判（四边色 = 画面主色）
  const white = img(80, 80);
  for (let i = 0; i < white.data.length; i += 4) {
    white.data[i] = 255; white.data[i + 1] = 255; white.data[i + 2] = 255; white.data[i + 3] = 255;
  }
  const g3 = P.guessBackgroundColors(white);
  check(g3.guessed === false, '纯白底：不改判');
  check(g3.colors.length && g3.colors[0][0] > 240, '纯白底：取到白色');

  // 洋红有多个色差变体（真实素材就长这样）—— 合并后仍要能认出是背景
  const noisy = img(140, 140);
  const VARIANTS = [[249, 6, 241], [244, 10, 237], [245, 6, 238]];
  for (let y = 0; y < 140; y++) {
    for (let x = 0; x < 140; x++) {
      const i = (y * 140 + x) * 4;
      const inBorder = x < 5 || y < 5 || x >= 135 || y >= 135;
      const inSubject = x >= 60 && x < 80 && y >= 60 && y < 80;
      let c;
      if (inSubject) c = [44, 46, 38];
      else if (inBorder) c = [255, 255, 255];
      else c = VARIANTS[(x + y) % 3];
      noisy.data[i] = c[0]; noisy.data[i + 1] = c[1]; noisy.data[i + 2] = c[2]; noisy.data[i + 3] = 255;
    }
  }
  const gn = P.guessBackgroundColors(noisy);
  check(gn.guessed === true, '多色变体洋红：仍判定要用画面主色');
  check(gn.colors.length === 1 && gn.colors[0][0] > 200 && gn.colors[0][1] < 60 && gn.colors[0][2] > 200,
    '多色变体洋红：猜到的仍是洋红', JSON.stringify(gn.colors));
  const dom = P.sampleDominantColors(noisy, 1);
  check(dom[0].share > 0.6, '多色变体合并后主色占比过 60%', (dom[0].share * 100).toFixed(1) + '%');

  // 白边 + 洋红：自动抠图应当自己认出洋红才是背景（不靠人填）
  const rb = P.removeBackground(border, { tolerance: 32 });
  check(rb.bgColors.length === 1 && rb.bgColors[0][0] > 200 && rb.bgColors[0][1] < 60,
    '自动抠图自动认出洋红背景', JSON.stringify(rb.bgColors));
  check(rb.bgGuessed === true, '自动抠图会标记 bgGuessed');
}

console.log(`\n===== ${pass} 通过 / ${fail} 失败 =====`);
process.exit(fail ? 1 : 0);
