/**
 * pixcore.js —— 全部核心算法。纯 JS、无 DOM 依赖，可以直接在 Node 里 require 做测试。
 *
 * 【第三方来源 · MIT】
 *   网格检测（FFT 估周期 → Sobel 细化 → 格内 center/median/majority 采样）
 *   算法与流程来自 theamusing/perfectPixel（Python · MIT）
 *     https://github.com/theamusing/perfectPixel
 *   本文件是对照其 src/perfect_pixel/perfect_pixel_noCV2.py 用 JavaScript
 *   重新实现的（原版是 numpy 向量化写法，这里是显式循环）。
 *   完整声明见 LICENSE 末尾「第三方代码声明」。
 *
 * 【本项目独立实现】
 *   抠图、削边、描边、杂色清理、色表对标、视频转帧、GIF 编解码、ZIP 打包；
 *   以及 detectGrid / detectPitchAxis 这套「整数块一致性」检测——它不在上游算法里，
 *   是为处理 AI 出图的糊边自己加的，FFT 判不了时才回落到上游方案。
 *   其中色表对标用的「中位切分」(median cut) 是 Paul Heckbert 1982 年发表的通用算法，本实现为自写。
 */
'use strict';

/* ============================ ImageData 兜底 ============================ */
if (typeof ImageData === 'undefined') {
  globalThis.ImageData = class ImageData {
    constructor(data, width, height) {
      if (typeof data === 'number') {
        this.width = data; this.height = width;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      } else {
        this.data = data; this.width = width; this.height = height;
      }
    }
  };
}

/* ==================================================================
 * 完美像素 · 网格检测与重采样
 *
 * 算法来源：theamusing/perfectPixel（MIT License · © theamusing）
 *   https://github.com/theamusing/perfectPixel
 *
 * 本段是对照其 src/perfect_pixel/perfect_pixel_noCV2.py 用 JavaScript
 * 重新实现的：FFT 求周期 → Sobel 细化网格线 → 格内采样（center / median
 * / majority 三选一）。原版是 Python + numpy，这里按同样的步骤用显式
 * 循环写，没有 numpy 的切片/向量化语法可用。
 *
 * 与本文件后半段 detectGrid/detectPitchAxis 的关系：
 *   那两个是本项目自研的「整数块一致性」检测，专门对付 AI 出图那种
 *   「块内是纯色、但边缘被糊掉 1-2px」的情况——纯 FFT 在这种图上会把
 *   糊边的渐变当成高频噪声，估出错误的周期。所以优先用自研检测，
 *   它判不了（返回 null）才回落到这里的 FFT/梯度方案。
 * ================================================================== */
const GRID_ESTIMATE_MAX_SIDE = 512;      // 估网格只在 ≤512 的缩略图上做（原版也是这么设计的）
const ALPHA_CUTOFF = 8;    // alpha 低于此值视为全透明

/* ---------------------------- 像素解包 ---------------------------- */

/** 把 ImageData 拆成 r/g/b/a 四个 Float32 平面（后续所有计算都吃平面） */
function unpackRGBA(imageData) {
  const w = imageData.width;
  const h = imageData.height;
  const n = w * h;
  const src = imageData.data;
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  const a = new Float32Array(n);
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    r[p] = src[i];
    g[p] = src[i + 1];
    b[p] = src[i + 2];
    a[p] = src[i + 3];
  }
  return { r, g, b, a, w, h };
}

/** RGB 三平面 → 灰度平面，用 ITU-R BT.601 亮度权重（和原版一致） */
function rgbPlanesToGray(r, g, b, n) {
  const gray = new Float32Array(n);
  for (let p = 0; p < n; p++) {
    gray[p] = 0.299 * r[p] + 0.587 * g[p] + 0.114 * b[p];
  }
  return gray;
}

/** ImageData → 灰度平面（一次搞定解包和转灰度，省一遍遍历） */
function imageToGray(imageData) {
  const w = imageData.width;
  const h = imageData.height;
  const n = w * h;
  const src = imageData.data;
  const gray = new Float32Array(n);
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    gray[p] = 0.299 * src[i] + 0.587 * src[i + 1] + 0.114 * src[i + 2];
  }
  return { gray, w, h };
}

/** 最近邻缩放灰度平面（估网格前缩到 512 以内用） */
function downscaleGrayNearest(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.floor((y * sh) / dh));
    const rowSrc = sy * sw;
    const rowDst = y * dw;
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.floor((x * sw) / dw));
      out[rowDst + x] = src[rowSrc + sx];
    }
  }
  return out;
}

/* ---------------------------- 基础数学 ---------------------------- */

/** 线性映射到 [a, b]；整幅平坦时返回常量 a（原版 normalize_minmax） */
function normalizeMinMax(x, a = 0, b = 1) {
  const n = x.length;
  const out = new Float32Array(n);
  if (n === 0) return out;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = x[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  if (span < 1e-8) {
    out.fill(a);
    return out;
  }
  const scale = (b - a) / span;
  for (let i = 0; i < n; i++) out[i] = a + (x[i] - lo) * scale;
  return out;
}

/** 高斯核一维平滑，核长 k（自动取奇数），sigma = k / 6 */
function smooth1d(v, k = 17) {
  const n = v.length;
  let kw = Math.floor(k);
  if (kw < 3 || n === 0) return Float32Array.from(v);
  if (kw % 2 === 0) kw += 1;
  const sigma = kw / 6;
  const half = kw >> 1;
  const ker = new Float32Array(kw);
  let sum = 1e-8;
  for (let i = 0; i < kw; i++) {
    const d = i - half;
    const val = Math.exp(-(d * d) / (2 * sigma * sigma));
    ker[i] = val;
    sum += val;
  }
  for (let i = 0; i < kw; i++) ker[i] /= sum;

  // 边界按端点值外推（等价于 numpy 的 mode='same' 在两端补零再归一）
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let j = 0; j < kw; j++) {
      let idx = i + j - half;
      if (idx < 0) idx = 0;
      else if (idx >= n) idx = n - 1;
      acc += ker[j] * v[idx];
    }
    out[i] = acc;
  }
  return out;
}

/* ---------------------------- 卷积 / Sobel ---------------------------- */

/** 3×3 卷积，same 尺寸，边界按镜像反射取值 */
function convolve3x3(plane, w, h, kernel) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let ky = -1; ky <= 1; ky++) {
        let sy = y + ky;
        if (sy < 0) sy = -sy;
        else if (sy >= h) sy = 2 * (h - 1) - sy;
        if (sy < 0) sy = 0;
        else if (sy >= h) sy = h - 1;
        const rowOff = sy * w;
        const kRow = (ky + 1) * 3;
        for (let kx = -1; kx <= 1; kx++) {
          const weight = kernel[kRow + kx + 1];
          if (weight === 0) continue;
          let sx = x + kx;
          if (sx < 0) sx = -sx;
          else if (sx >= w) sx = 2 * (w - 1) - sx;
          if (sx < 0) sx = 0;
          else if (sx >= w) sx = w - 1;
          acc += weight * plane[rowOff + sx];
        }
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

const SOBEL_X3 = new Float32Array([-1, 0, 1, -2, 0, 2, -1, 0, 1]);
const SOBEL_Y3 = new Float32Array([-1, -2, -1, 0, 0, 0, 1, 2, 1]);

/** Sobel 梯度（3×3） */
function sobelXY(gray, w, h) {
  return {
    gx: convolve3x3(gray, w, h, SOBEL_X3),
    gy: convolve3x3(gray, w, h, SOBEL_Y3),
  };
}

/* ---------------------------- FFT ---------------------------- */

/** 原地基 2 复数 FFT（长度必须是 2 的幂；不做逆变换，只用于求幅度谱） */
function fftInPlace(re, im) {
  const n = re.length;
  // 位反转重排
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  // 蝶形
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const ang = (-2 * Math.PI) / len;
    const wStepR = Math.cos(ang);
    const wStepI = Math.sin(ang);
    for (let base = 0; base < n; base += len) {
      let wR = 1, wI = 0;
      for (let k = 0; k < halfLen; k++) {
        const i0 = base + k;
        const i1 = i0 + halfLen;
        const vR = re[i1] * wR - im[i1] * wI;
        const vI = re[i1] * wI + im[i1] * wR;
        re[i1] = re[i0] - vR;
        im[i1] = im[i0] - vI;
        re[i0] += vR;
        im[i0] += vI;
        const nR = wR * wStepR - wI * wStepI;
        wI = wR * wStepI + wI * wStepR;
        wR = nR;
      }
    }
  }
}

/** 不小于 v 的最小 2 的幂 */
function pow2Ceil(v) {
  let p = 1;
  while (p < v) p <<= 1;
  return p;
}

/** 边界复制填充到 pw×ph（pw/ph 必须是 2 的幂） */
function padEdge(plane, w, h, pw, ph) {
  if (pw === w && ph === h) return Float32Array.from(plane);
  const out = new Float32Array(pw * ph);
  for (let y = 0; y < ph; y++) {
    const sy = y < h ? y : h - 1;
    const srcRow = sy * w;
    const dstRow = y * pw;
    for (let x = 0; x < pw; x++) {
      const sx = x < w ? x : w - 1;
      out[dstRow + x] = plane[srcRow + sx];
    }
  }
  return out;
}

/**
 * 幅度谱 → 1 - log(1 + |F|) → 归一化。
 * 原版先做 fftshift 再取模，但后面只按行/列整幅求和，
 * 平移不改变投影结果，所以这里省掉 shift 直接求模。
 */
function fftMagnitude(plane, w, h) {
  const pw = pow2Ceil(w);
  const ph = pow2Ceil(h);
  const padded = padEdge(plane, w, h, pw, ph);

  const re = Float64Array.from(padded);
  const im = new Float64Array(pw * ph);
  // 先行后列做 2D FFT（可分离）
  const rowR = new Float64Array(pw);
  const rowI = new Float64Array(pw);
  for (let y = 0; y < ph; y++) {
    const off = y * pw;
    for (let x = 0; x < pw; x++) {
      rowR[x] = re[off + x];
      rowI[x] = im[off + x];
    }
    fftInPlace(rowR, rowI);
    for (let x = 0; x < pw; x++) {
      re[off + x] = rowR[x];
      im[off + x] = rowI[x];
    }
  }
  const colR = new Float64Array(ph);
  const colI = new Float64Array(ph);
  for (let x = 0; x < pw; x++) {
    for (let y = 0; y < ph; y++) {
      const off = y * pw + x;
      colR[y] = re[off];
      colI[y] = im[off];
    }
    fftInPlace(colR, colI);
    for (let y = 0; y < ph; y++) {
      const off = y * pw + x;
      re[off] = colR[y];
      im[off] = colI[y];
    }
  }

  const mag = new Float32Array(pw * ph);
  for (let i = 0; i < mag.length; i++) {
    const m = Math.hypot(re[i], im[i]);
    mag[i] = 1 - Math.log1p(m);
  }
  return { mag, pw, ph };
}

/* ---------------------------- 峰值检测 ---------------------------- */

/**
 * 以谱中心为对称轴，找左右两侧最强的峰，返回「半间距」= 周期长度。
 * 原版 detect_peak：峰的判据是「peakWidth 范围内单调」，峰高要过 0.35 倍最大值。
 */
function detectPeak(proj, peakWidth = 6, relThr = 0.35, minDist = 6) {
  const n = proj.length;
  if (n < 4) return null;
  const center = n >> 1;
  let peakMax = -Infinity;
  for (let i = 0; i < n; i++) if (proj[i] > peakMax) peakMax = proj[i];
  if (peakMax < 1e-6) return null;
  const thr = peakMax * relThr;

  const candidates = [];
  for (let i = 1; i < n - 1; i++) {
    let isPeak = true;
    for (let j = 1; j < peakWidth; j++) {
      const leftA = i - j, leftB = i - j + 1;
      const rightA = i + j - 1, rightB = i + j;
      if (leftA < 0 || rightB >= n) continue;
      if (proj[leftB] < proj[leftA] || proj[rightA] < proj[rightB]) {
        isPeak = false;
        break;
      }
    }
    if (!isPeak || proj[i] < thr) continue;

    // 向左爬升的落差
    let climb = 0;
    for (let k = i; k > 0; k--) {
      if (proj[k] > proj[k - 1]) climb = Math.abs(proj[i] - proj[k - 1]);
      else break;
    }
    // 向右回落的落差
    let fall = 0;
    for (let k = i; k < n - 1; k++) {
      if (proj[k] > proj[k + 1]) fall = Math.abs(proj[i] - proj[k + 1]);
      else break;
    }
    candidates.push({ index: i, score: Math.max(climb, fall) });
  }
  if (candidates.length === 0) return null;

  let bestLeft = null;
  let bestRight = null;
  for (const c of candidates) {
    if (c.index < center - minDist && c.index > center * 0.25) {
      if (!bestLeft || c.score > bestLeft.score) bestLeft = c;
    } else if (c.index > center + minDist && c.index < center * 1.75) {
      if (!bestRight || c.score > bestRight.score) bestRight = c;
    }
  }
  if (!bestLeft || !bestRight) return null;
  return Math.abs(bestRight.index - bestLeft.index) / 2;
}

/** 在 origin ± range 范围内找梯度最强的局部极大，作为网格线位置 */
function findBestGrid(origin, rangeMin, rangeMax, gradMag, thr = 0) {
  const n = gradMag.length;
  let best = Math.round(origin);
  let gradMax = 0;
  for (let i = 0; i < n; i++) if (gradMag[i] > gradMax) gradMax = gradMag[i];
  if (gradMax < 1e-6) return best;
  const floor = gradMax * thr;

  const lo = -Math.round(rangeMin);
  const hi = Math.round(rangeMax);
  let bestVal = -Infinity;
  let bestIdx = -1;
  for (let d = lo; d <= hi; d++) {
    const c = Math.round(origin + d);
    if (c <= 0 || c >= n - 1) continue;
    if (!(gradMag[c] > gradMag[c - 1] && gradMag[c] > gradMag[c + 1])) continue;
    if (gradMag[c] < floor) continue;
    if (gradMag[c] > bestVal) {
      bestVal = gradMag[c];
      bestIdx = c;
    }
  }
  return bestIdx >= 0 ? bestIdx : best;
}

/* ---------------------------- 格内采样 ---------------------------- */

/** 取格子中心那一个像素 */
function sampleCenter(r, g, b, a, W, H, xCoords, yCoords) {
  const nx = xCoords.length - 1;
  const ny = yCoords.length - 1;
  const out = new Uint8ClampedArray(nx * ny * 4);
  let o = 0;
  for (let j = 0; j < ny; j++) {
    // 原版用 astype(int32) 截断，不是四舍五入
    let cy = ((yCoords[j] + yCoords[j + 1]) * 0.5) | 0;
    if (cy < 0) cy = 0;
    else if (cy > H - 1) cy = H - 1;
    const rowOff = cy * W;
    for (let i = 0; i < nx; i++) {
      let cx = ((xCoords[i] + xCoords[i + 1]) * 0.5) | 0;
      if (cx < 0) cx = 0;
      else if (cx > W - 1) cx = W - 1;
      const si = rowOff + cx;
      const alpha = a[si];
      if (alpha <= ALPHA_CUTOFF) {
        o += 4;
        continue;
      }
      out[o++] = r[si];
      out[o++] = g[si];
      out[o++] = b[si];
      out[o++] = alpha;
    }
  }
  return new ImageData(out, nx, ny);
}

/** 取格子内各通道中位数（透明像素不参与算色，但参与算 alpha 中位数） */
function sampleMedian(r, g, b, a, W, H, xCoords, yCoords) {
  const nx = xCoords.length - 1;
  const ny = yCoords.length - 1;
  const out = new Uint8ClampedArray(nx * ny * 4);
  const rs = [];
  const gs = [];
  const bs = [];
  const as = [];
  let o = 0;

  for (let j = 0; j < ny; j++) {
    let y0 = Math.floor(yCoords[j]);
    let y1 = Math.floor(yCoords[j + 1]);
    if (y0 < 0) y0 = 0;
    if (y1 > H) y1 = H;
    if (y1 <= y0) y1 = Math.min(y0 + 1, H);

    for (let i = 0; i < nx; i++) {
      let x0 = Math.floor(xCoords[i]);
      let x1 = Math.floor(xCoords[i + 1]);
      if (x0 < 0) x0 = 0;
      if (x1 > W) x1 = W;
      if (x1 <= x0) x1 = Math.min(x0 + 1, W);

      rs.length = 0; gs.length = 0; bs.length = 0; as.length = 0;
      for (let y = y0; y < y1; y++) {
        const rowOff = y * W;
        for (let x = x0; x < x1; x++) {
          const si = rowOff + x;
          as.push(a[si]);
          if (a[si] > ALPHA_CUTOFF) {
            rs.push(r[si]); gs.push(g[si]); bs.push(b[si]);
          }
        }
      }
      if (as.length === 0 || rs.length === 0) {
        o += 4;
        continue;
      }
      as.sort((p, q) => p - q);
      const alpha = Math.round(medianOf(as));
      if (alpha <= ALPHA_CUTOFF) {
        o += 4;
        continue;
      }
      rs.sort((p, q) => p - q);
      gs.sort((p, q) => p - q);
      bs.sort((p, q) => p - q);
      out[o++] = Math.round(medianOf(rs));
      out[o++] = Math.round(medianOf(gs));
      out[o++] = Math.round(medianOf(bs));
      out[o++] = alpha;
    }
  }
  return new ImageData(out, nx, ny);
}

function medianOf(sortedArr) {
  const n = sortedArr.length;
  if (n === 0) return 0;
  const mid = n >> 1;
  return n % 2 ? sortedArr[mid] : (sortedArr[mid - 1] + sortedArr[mid]) / 2;
}

/**
 * 格子内取「占多数的那一类颜色」。
 * 原版做法：格内像素跑 2-means（k=2，6 轮），初始两个中心分别是
 * 「第一个像素」和「离它最远的像素」，最后取人数多的那一边的中心。
 */
function sampleMajority(r, g, b, a, W, H, xCoords, yCoords, maxSamples = 128, iters = 6) {
  const nx = xCoords.length - 1;
  const ny = yCoords.length - 1;
  const out = new Uint8ClampedArray(nx * ny * 4);
  const as = [];
  let o = 0;

  for (let j = 0; j < ny; j++) {
    let y0 = Math.floor(yCoords[j]);
    let y1 = Math.floor(yCoords[j + 1]);
    if (y0 < 0) y0 = 0;
    if (y1 > H) y1 = H;
    if (y1 <= y0) y1 = Math.min(y0 + 1, H);

    for (let i = 0; i < nx; i++) {
      let x0 = Math.floor(xCoords[i]);
      let x1 = Math.floor(xCoords[i + 1]);
      if (x0 < 0) x0 = 0;
      if (x1 > W) x1 = W;
      if (x1 <= x0) x1 = Math.min(x0 + 1, W);

      let visible = 0;
      let total = 0;
      for (let y = y0; y < y1; y++) {
        const rowOff = y * W;
        for (let x = x0; x < x1; x++) {
          total++;
          if (a[rowOff + x] > ALPHA_CUTOFF) visible++;
        }
      }
      // 格子一半以上是透明 → 整格透明（原版没有 alpha，这条是本项目为像素画加的）
      if (total === 0 || visible === 0 || visible * 2 < total) {
        o += 4;
        continue;
      }

      // 收集可见像素，超过上限就等距抽样（原版是随机抽，这里改成等距，结果可复现）
      const stride = visible > maxSamples ? visible / maxSamples : 1;
      const sampleN = visible > maxSamples ? maxSamples : visible;
      const cr = new Float64Array(sampleN);
      const cg = new Float64Array(sampleN);
      const cb = new Float64Array(sampleN);
      let seen = 0;
      let slot = 0;
      let nextAt = 0;
      for (let y = y0; y < y1 && slot < sampleN; y++) {
        const rowOff = y * W;
        for (let x = x0; x < x1 && slot < sampleN; x++) {
          const si = rowOff + x;
          if (a[si] <= ALPHA_CUTOFF) continue;
          if (seen >= nextAt) {
            cr[slot] = r[si]; cg[slot] = g[si]; cb[slot] = b[si];
            slot++;
            nextAt += stride;
          }
          seen++;
        }
      }
      const n = slot;
      if (n === 0) {
        o += 4;
        continue;
      }

      // 2-means：c0 取第一个点，c1 取离 c0 最远的点
      let c0r = cr[0], c0g = cg[0], c0b = cb[0];
      let far = -1, farD = -1;
      for (let p = 0; p < n; p++) {
        const dr = cr[p] - c0r, dg = cg[p] - c0g, db = cb[p] - c0b;
        const d = dr * dr + dg * dg + db * db;
        if (d > farD) { farD = d; far = p; }
      }
      let c1r = cr[far], c1g = cg[far], c1b = cb[far];

      let cnt1 = 0;
      for (let it = 0; it < iters; it++) {
        let s0r = 0, s0g = 0, s0b = 0, k0 = 0;
        let s1r = 0, s1g = 0, s1b = 0, k1 = 0;
        for (let p = 0; p < n; p++) {
          const d0r = cr[p] - c0r, d0g = cg[p] - c0g, d0b = cb[p] - c0b;
          const d1r = cr[p] - c1r, d1g = cg[p] - c1g, d1b = cb[p] - c1b;
          if (d1r * d1r + d1g * d1g + d1b * d1b < d0r * d0r + d0g * d0g + d0b * d0b) {
            s1r += cr[p]; s1g += cg[p]; s1b += cb[p]; k1++;
          } else {
            s0r += cr[p]; s0g += cg[p]; s0b += cb[p]; k0++;
          }
        }
        if (k0) { c0r = s0r / k0; c0g = s0g / k0; c0b = s0b / k0; }
        if (k1) { c1r = s1r / k1; c1g = s1g / k1; c1b = s1b / k1; }
        cnt1 = k1;
      }

      const pick = cnt1 * 2 >= n
        ? [Math.round(c1r), Math.round(c1g), Math.round(c1b)]
        : [Math.round(c0r), Math.round(c0g), Math.round(c0b)];

      // alpha 用格内中位数
      as.length = 0;
      for (let y = y0; y < y1; y++) {
        const rowOff = y * W;
        for (let x = x0; x < x1; x++) as.push(a[rowOff + x]);
      }
      as.sort((p, q) => p - q);

      out[o++] = pick[0];
      out[o++] = pick[1];
      out[o++] = pick[2];
      out[o++] = Math.round(medianOf(as));
    }
  }
  return new ImageData(out, nx, ny);
}

/* ---------------------------- 网格线细化 ---------------------------- */

/**
 * 从图像中心出发，按 cell 步长向左右/上下推，每一步在 ±cell*intensity
 * 范围内把网格线吸附到梯度最强的位置。
 */
function refineGrids(gray, W, H, gridX, gridY, refineIntensity = 0.25) {
  const cellW = W / gridX;
  const cellH = H / gridY;
  const { gx, gy } = sobelXY(gray, W, H);

  // 按列/行累加梯度绝对值 → 网格线在这里会形成尖峰
  const gradXSum = new Float32Array(W);
  const gradYSum = new Float32Array(H);
  for (let y = 0; y < H; y++) {
    const rowOff = y * W;
    for (let x = 0; x < W; x++) {
      const v = Math.abs(gx[rowOff + x]);
      gradXSum[x] += v;
      gradYSum[y] += Math.abs(gy[rowOff + x]);
    }
  }

  const xCoords = [];
  const yCoords = [];
  const stepX = cellW * refineIntensity;
  const stepY = cellH * refineIntensity;

  let x = findBestGrid(W / 2, cellW, cellW, gradXSum);
  while (x < W + cellW / 2) {
    x = findBestGrid(x, stepX, stepX, gradXSum);
    xCoords.push(x);
    x += cellW;
  }
  x = findBestGrid(W / 2, cellW, cellW, gradXSum) - cellW;
  while (x > -cellW / 2) {
    x = findBestGrid(x, stepX, stepX, gradXSum);
    xCoords.push(x);
    x -= cellW;
  }

  let y = findBestGrid(H / 2, cellH, cellH, gradYSum);
  while (y < H + cellH / 2) {
    y = findBestGrid(y, stepY, stepY, gradYSum);
    yCoords.push(y);
    y += cellH;
  }
  y = findBestGrid(H / 2, cellH, cellH, gradYSum) - cellH;
  while (y > -cellH / 2) {
    y = findBestGrid(y, stepY, stepY, gradYSum);
    yCoords.push(y);
    y -= cellH;
  }

  xCoords.sort((p, q) => p - q);
  yCoords.sort((p, q) => p - q);
  return { xCoords, yCoords };
}

/* ---------------------------- 尺度估计 ---------------------------- */

/** FFT 方案：幅度谱按行/列投影 → 平滑 → 找对称峰 → 半间距就是周期 */
function estimateGridFft(gray, w, h, peakWidth = 6) {
  const { mag, pw, ph } = fftMagnitude(gray, w, h);

  const rowSum = new Float32Array(ph);
  const colSum = new Float32Array(pw);
  for (let y = 0; y < ph; y++) {
    const rowOff = y * pw;
    for (let x = 0; x < pw; x++) {
      const v = mag[rowOff + x];
      rowSum[y] += v;
      colSum[x] += v;
    }
  }

  const rows = smooth1d(normalizeMinMax(rowSum), 17);
  const cols = smooth1d(normalizeMinMax(colSum), 17);

  const periodRow = detectPeak(rows, peakWidth);
  const periodCol = detectPeak(cols, peakWidth);
  if (periodRow == null || periodCol == null || periodCol <= 0) return null;

  // 周期是在填充后的尺寸上量出来的，要按原尺寸折算格子数
  const gridW = Math.round(w / (periodCol * (w / pw)));
  const gridH = Math.round(h / (periodRow * (h / ph)));
  if (!(gridW > 0) || !(gridH > 0)) return null;
  return { gridW, gridH };
}

/** 梯度方案：投影找峰 → 相邻峰间隔取中位数 → 反算格子数 */
function estimateGridGradient(gray, w, h, relThr = 0.2) {
  const { gx, gy } = sobelXY(gray, w, h);
  const gradXSum = new Float32Array(w);
  const gradYSum = new Float32Array(h);
  let maxX = 0, maxY = 0;
  for (let y = 0; y < h; y++) {
    const rowOff = y * w;
    for (let x = 0; x < w; x++) {
      const ax = Math.abs(gx[rowOff + x]);
      const ay = Math.abs(gy[rowOff + x]);
      gradXSum[x] += ax;
      gradYSum[y] += ay;
      if (ax > maxX) maxX = ax;
      if (ay > maxY) maxY = ay;
    }
  }

  const MIN_INTERVAL = 4;
  const collectPeaks = (sum, n, thr) => {
    const peaks = [];
    for (let i = 1; i < n - 1; i++) {
      if (!(sum[i] > sum[i - 1] && sum[i] > sum[i + 1])) continue;
      if (sum[i] < thr) continue;
      if (peaks.length && i - peaks[peaks.length - 1] < MIN_INTERVAL) continue;
      peaks.push(i);
    }
    return peaks;
  };

  const peakX = collectPeaks(gradXSum, w, relThr * maxX);
  const peakY = collectPeaks(gradYSum, h, relThr * maxY);
  if (peakX.length < 4 || peakY.length < 4) return null;

  const medianGap = (peaks) => {
    const gaps = [];
    for (let i = 1; i < peaks.length; i++) gaps.push(peaks[i] - peaks[i - 1]);
    gaps.sort((p, q) => p - q);
    return medianOf(gaps);
  };

  const gapX = medianGap(peakX);
  const gapY = medianGap(peakY);
  if (!(gapX > 0) || !(gapY > 0)) return null;

  return { gridW: Math.round(w / gapX), gridH: Math.round(h / gapY) };
}

/**
 * 估出 cols×rows。优先 FFT，估出来的格子尺寸不合理（太小/太大/长宽比失衡）
 * 就退回梯度方案；最后统一用「平均像素尺寸」反算一次，保证 x/y 用同一个 pixelSize。
 */
function detectGridScale(gray, w, h, peakWidth = 6, maxRatio = 1.5, minSize = 4) {
  const tw = Math.min(w, GRID_ESTIMATE_MAX_SIDE);
  const th = Math.min(h, GRID_ESTIMATE_MAX_SIDE);
  const thumb = (tw === w && th === h) ? gray : downscaleGrayNearest(gray, w, h, tw, th);

  let gridW = null;
  let gridH = null;

  const fft = estimateGridFft(thumb, tw, th, peakWidth);
  if (fft) {
    const pxX = w / fft.gridW;
    const pxY = h / fft.gridH;
    const maxPixel = 20;
    const sane = Math.min(pxX, pxY) >= minSize
      && Math.max(pxX, pxY) <= maxPixel
      && pxX / pxY <= maxRatio
      && pxY / pxX <= maxRatio;
    if (sane) {
      gridW = fft.gridW;
      gridH = fft.gridH;
    }
  }
  if (gridW == null) {
    const grad = estimateGridGradient(gray, w, h);
    if (!grad) return null;
    gridW = grad.gridW;
    gridH = grad.gridH;
  }
  if (!(gridW > 0) || !(gridH > 0)) return null;

  const pxX = w / gridW;
  const pxY = h / gridH;
  const pixelSize = (pxX / pxY > maxRatio || pxY / pxX > maxRatio)
    ? Math.min(pxX, pxY)
    : (pxX + pxY) / 2;
  if (!(pixelSize > 0)) return null;

  return {
    gridW: Math.round(w / pixelSize),
    gridH: Math.round(h / pixelSize),
  };
}

/* ---------------------------- 输出修正 ---------------------------- */

/** 宽高只差 1px 时掰成正方形（像素画习惯） */
function forceSquareOutput(img, fixSquare) {
  if (!fixSquare) return img;
  const w = img.width;
  const h = img.height;
  if (Math.abs(w - h) !== 1) return img;
  const src = img.data;

  const cropCol = (newW) => {
    const out = new Uint8ClampedArray(newW * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < newW; x++) {
        const si = (y * w + x) * 4;
        const di = (y * newW + x) * 4;
        out[di] = src[si]; out[di + 1] = src[si + 1];
        out[di + 2] = src[si + 2]; out[di + 3] = src[si + 3];
      }
    }
    return new ImageData(out, newW, h);
  };
  const cropRow = (newH) => {
    const out = new Uint8ClampedArray(w * newH * 4);
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < w; x++) {
        const si = (y * w + x) * 4;
        const di = (y * w + x) * 4;
        out[di] = src[si]; out[di + 1] = src[si + 1];
        out[di + 2] = src[si + 2]; out[di + 3] = src[si + 3];
      }
    }
    return new ImageData(out, w, newH);
  };
  const dupCol = (newW) => {
    // 复制第 0 列到最左，其余右移
    const out = new Uint8ClampedArray(newW * h * 4);
    for (let y = 0; y < h; y++) {
      const s0 = y * w * 4;
      const d0 = y * newW * 4;
      out[d0] = src[s0]; out[d0 + 1] = src[s0 + 1];
      out[d0 + 2] = src[s0 + 2]; out[d0 + 3] = src[s0 + 3];
      for (let x = 0; x < w; x++) {
        const si = (y * w + x) * 4;
        const di = (y * newW + x + 1) * 4;
        out[di] = src[si]; out[di + 1] = src[si + 1];
        out[di + 2] = src[si + 2]; out[di + 3] = src[si + 3];
      }
    }
    return new ImageData(out, newW, h);
  };
  const dupRow = (newH) => {
    // 复制第 0 行到最上，其余下移
    const out = new Uint8ClampedArray(w * newH * 4);
    for (let x = 0; x < w; x++) {
      out[x * 4] = src[x * 4];
      out[x * 4 + 1] = src[x * 4 + 1];
      out[x * 4 + 2] = src[x * 4 + 2];
      out[x * 4 + 3] = src[x * 4 + 3];
    }
    for (let y = 1; y < newH; y++) {
      for (let x = 0; x < w; x++) {
        const si = ((y - 1) * w + x) * 4;
        const di = (y * w + x) * 4;
        out[di] = src[si]; out[di + 1] = src[si + 1];
        out[di + 2] = src[si + 2]; out[di + 3] = src[si + 3];
      }
    }
    return new ImageData(out, w, newH);
  };

  if (w > h) return w % 2 === 1 ? cropCol(w - 1) : dupRow(h + 1);
  return h % 2 === 1 ? cropRow(h - 1) : dupCol(w + 1);
}

/* ==================================================================
 * 以下是本项目自研的检测：整数块一致性
 * 目的：AI 放大的像素图，块内是纯色、但边缘被插值糊了 1-2px。
 * 这种图 FFT 会把糊边的渐变当噪声，估不准周期；而「按 k 切竖条、
 * 看每条内部颜色散不散」对糊边不敏感——真周期处一定出现明显低谷。
 * ================================================================== */

function rgbaToGrayArray(rgba, n) {
  const { r, g, b } = rgba;
  const gray = new Float32Array(n);
  for (let i = 0; i < n; i++) gray[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
  return gray;
}

function gradProjections(gray, W, H) {
  const projX = new Float64Array(W);
  const projY = new Float64Array(H);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const dx = -gray[i - W - 1] + gray[i - W + 1] - 2 * gray[i - 1] + 2 * gray[i + 1] - gray[i + W - 1] + gray[i + W + 1];
      const dy = -gray[i - W - 1] - 2 * gray[i - W] - gray[i - W + 1] + gray[i + W - 1] + 2 * gray[i + W] + gray[i + W + 1];
      const v = Math.abs(dx) + Math.abs(dy);
      projX[x] += v;
      projY[y] += v;
    }
  }
  return { projX, projY };
}

function gradProjection(gray, W, H, axis) {
  const p = gradProjections(gray, W, H);
  return axis === 'x' ? p.projX : p.projY;
}

function gridLineEnrichment(proj, p, halfW = 0.5) {
  const N = proj.length;
  let total = 0;
  for (let i = 0; i < N; i++) total += proj[i];
  if (total <= 0) return 0;
  let onSum = 0, onCnt = 0;
  const n = Math.floor((N - 1) / p);
  for (let k = 0; k <= n; k++) {
    const c = k * p;
    const lo = Math.max(0, Math.ceil(c - halfW));
    const hi = Math.min(N - 1, Math.floor(c + halfW));
    for (let x = lo; x <= hi; x++) {
      onSum += proj[x];
      onCnt++;
    }
  }
  if (onCnt === 0) return 0;
  return (onSum / onCnt) / Math.max(total / N, 1e-9);
}

/** 格内颜色离散度：pitch 取真值时接近 0，取 2 倍 pitch 时明显变大 */
function withinCellVariance(rgba, W, H, p, axis) {
  const { r, g, b, a } = rgba;
  let acc = 0;
  let cnt = 0;
  const stepX = Math.max(1, Math.floor(W / 96));
  const stepY = Math.max(1, Math.floor(H / 96));
  const cells = Math.floor((axis === 'x' ? W : H) / p);
  for (let ci = 0; ci < cells; ci++) {
    const c0 = Math.round(ci * p);
    const c1 = Math.min(axis === 'x' ? W : H, Math.round((ci + 1) * p));
    if (c1 <= c0) continue;
    const outerEnd = axis === 'x' ? H : W;
    const outerStep = axis === 'x' ? stepY : stepX;
    for (let o = 0; o < outerEnd; o += outerStep) {
      let vis = 0, n = 0, sr = 0, sg = 0, sb = 0;
      for (let c = c0; c < c1; c++) {
        const i = axis === 'x' ? o * W + c : c * W + o;
        vis++;
        if (a[i] > 8) {
          sr += r[i]; sg += g[i]; sb += b[i];
          n++;
        }
      }
      if (n < 2 || n * 2 < vis) continue;
      const mr = sr / n, mg = sg / n, mb = sb / n;
      for (let c2 = c0; c2 < c1; c2++) {
        const i = axis === 'x' ? o * W + c2 : c2 * W + o;
        if (a[i] > 8) acc += Math.abs(r[i] - mr) + Math.abs(g[i] - mg) + Math.abs(b[i] - mb);
      }
      cnt += n;
    }
  }
  return cnt ? acc / cnt : Infinity;
}

/**
 * 检测一个方向上的像素宽度。
 *
 * 判据是「整数块一致性曲线」v(k)：把图按 k 宽切成竖条，看每条内部颜色有多散。
 *  - 图是 k 倍整数放大的 → v(k) ≈ 0（块内颜色完全一致）
 *  - 真周期的整数倍（2k、3k…）也是 0，所以取「v≈0 的最大 k」
 *  - 糊边（AI 出图）不会归零，但会在真周期处出现明显的局部低谷
 *  - 写实图 v(k) 单调上升、没有低谷 → 判无网格
 */
function detectPitchAxis(rgba, W, H, axis, minP, maxP) {
  const span = axis === 'x' ? W : H;
  const kMax = Math.min(maxP, Math.floor(span / 4));
  const kMin = Math.max(2, Math.floor(minP));
  if (kMax < kMin) return null;

  const curve = [];
  for (let k = kMin; k <= kMax; k++) curve.push({ k, v: withinCellVariance(rgba, W, H, k, axis) });
  const vOf = (k) => {
    const e = curve.find((c) => c.k === k);
    return e ? e.v : Infinity;
  };

  // 1) 精确整数放大：取 v 近似为零的最大 k
  let zeroMax = null;
  for (const c of curve) if (c.v <= 0.5) zeroMax = c.k;
  if (zeroMax != null) return { p: zeroMax, s: 1, v: vOf(zeroMax) };

  // 2) 糊边：找显著的低谷（比两侧都低得多）
  let best = null;
  for (let i = 1; i < curve.length - 1; i++) {
    const { k, v } = curve[i];
    const left = curve[i - 1].v;
    const right = curve[i + 1].v;
    if (!(v < left && v < right)) continue;
    const worse = Math.max(left, right);
    const sharp = v / Math.max(worse, 1e-6);
    if (sharp > 0.85) continue;
    if (worse - v < 2.5) continue;
    if (!best || sharp < best.sharp - 1e-9) best = { p: k, s: 1 / Math.max(sharp, 1e-6), v, sharp };
  }
  if (best) return best;

  // 3) 没有低谷 → 无周期网格
  return { p: 1, s: 1, v: Infinity };
}

/** 自动检测网格（返回 cols/rows 与像素尺寸），失败返回 null */
function detectGrid(imageData, opts = {}) {
  const W = imageData.width;
  const H = imageData.height;
  const minP = opts.minPixel ?? 2;
  const maxP = opts.maxPixel ?? Math.min(64, Math.floor(Math.min(W, H) / 2));
  const rgba = unpackRGBA(imageData);
  const px = detectPitchAxis(rgba, W, H, 'x', minP, maxP);
  const py = detectPitchAxis(rgba, W, H, 'y', minP, maxP);
  if (!px || !py) return null;
  // 任一方向没有周期 → 不能当网格（像素块是二维的），写实图才会被放过
  if (!(px.v < Infinity) || !(py.v < Infinity)) {
    return { cols: W, rows: H, pixelX: 1, pixelY: 1, scoreX: px.s, scoreY: py.s, noGrid: true, flat: true };
  }
  const cols = Math.max(2, Math.round(W / px.p));
  const rows = Math.max(2, Math.round(H / py.p));
  return {
    cols, rows,
    pixelX: W / cols,
    pixelY: H / rows,
    scoreX: px.s, scoreY: py.s,
    gridX: px.p, gridY: py.p,
    noGrid: cols >= W && rows >= H,
  };
}

/* ---------------------------- 主流程 ---------------------------- */

/**
 * 完美像素：把「AI 出的糊图」还原成逻辑像素图。
 * 与原版区别：不做整体缩图上限（原版 UI 会先把图缩到 2048），
 * 网格估计仍在 ≤512 缩略图上做（算法本来的设计），细化与采样按全分辨率走。
 */
function perfectPixel(imageData, opts = {}) {
  const W = imageData.width;
  const H = imageData.height;
  if (W < 4 || H < 4) return new ImageData(new Uint8ClampedArray(W * H * 4), W, H);

  const sampleMethod = opts.sampleMethod ?? 'center';
  const peakWidth = opts.peakWidth ?? 6;
  const refineIntensity = opts.refineIntensity ?? 0.25;
  const fixSquare = opts.fixSquare ?? true;

  const ch = unpackRGBA(imageData);
  const gray = rgbPlanesToGray(ch.r, ch.g, ch.b, W * H);

  let sizeX, sizeY;
  if (opts.gridSize != null) {
    sizeX = Math.max(2, Math.floor(opts.gridSize.cols));
    sizeY = Math.max(2, Math.floor(opts.gridSize.rows));
  } else if (opts.detector !== 'legacy') {
    // 优先自研检测（整数块一致性），它对付 AI 糊边更准
    const det = detectGrid(imageData, {
      minPixel: opts.minSize ?? 2,
      maxPixel: opts.maxPixel ?? 64,
    });
    if (det != null && det.noGrid) {
      // 已经是 1:1 像素图：原样返回，不切、不重采样
      const copy = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
      copy.noGrid = true;
      copy.gridCols = imageData.width;
      copy.gridRows = imageData.height;
      return copy;
    }
    if (det != null) {
      sizeX = det.cols;
      sizeY = det.rows;
    } else {
      // 自研判不了 → 回落到上游的 FFT/梯度方案
      const legacy = detectGridScale(gray, W, H, peakWidth, 1.5, opts.minSize ?? 4);
      if (legacy == null) throw new Error('ERR_PERFECT_PIXEL_GRID');
      sizeX = legacy.gridW;
      sizeY = legacy.gridH;
    }
  } else {
    const legacy = detectGridScale(gray, W, H, peakWidth, 1.5, opts.minSize ?? 4);
    if (legacy == null) throw new Error('ERR_PERFECT_PIXEL_GRID');
    sizeX = legacy.gridW;
    sizeY = legacy.gridH;
  }

  // 手动网格 / 视频序列锁定的网格：用严格等分。
  // refineGrids 是「从中心向外每步再搜一次」，误差会逐步累积成漂移，
  // 同一套 cols×rows 在不同画面上返回的边界数会不一样，输出尺寸就不固定。
  let xCoords, yCoords;
  if (opts.gridSize != null && opts.uniformGrid !== false) {
    xCoords = [];
    yCoords = [];
    for (let i = 0; i <= sizeX; i++) xCoords.push((i * W) / sizeX);
    for (let j = 0; j <= sizeY; j++) yCoords.push((j * H) / sizeY);
  } else {
    ({ xCoords, yCoords } = refineGrids(gray, W, H, sizeX, sizeY, refineIntensity));
    // 细化是从中心向外「每步再搜一次峰」，某一步没搜到峰就会回落到整数位置，
    // 误差一路累积下去会让坐标数比 sizeX+1 少一个 —— 输出尺寸就跟着漂。
    // 数量对不上时直接等分，保证输出一定是检测到的 cols×rows。
    if (xCoords.length !== sizeX + 1 || yCoords.length !== sizeY + 1) {
      xCoords = [];
      yCoords = [];
      for (let i = 0; i <= sizeX; i++) xCoords.push((i * W) / sizeX);
      for (let j = 0; j <= sizeY; j++) yCoords.push((j * H) / sizeY);
    }
  }

  let out;
  if (sampleMethod === 'majority') out = sampleMajority(ch.r, ch.g, ch.b, ch.a, W, H, xCoords, yCoords);
  else if (sampleMethod === 'median') out = sampleMedian(ch.r, ch.g, ch.b, ch.a, W, H, xCoords, yCoords);
  else out = sampleCenter(ch.r, ch.g, ch.b, ch.a, W, H, xCoords, yCoords);

  return forceSquareOutput(out, fixSquare);
}


/* ============================ 色表 / 调色 ============================ */
function rgbToLab(r, g, b) {
  let x = r / 255, y = g / 255, z = b / 255;
  x = x > 0.04045 ? ((x + 0.055) / 1.055) ** 2.4 : x / 12.92;
  y = y > 0.04045 ? ((y + 0.055) / 1.055) ** 2.4 : y / 12.92;
  z = z > 0.04045 ? ((z + 0.055) / 1.055) ** 2.4 : z / 12.92;
  x *= 100; y *= 100; z *= 100;
  const f = (t) => (t > 0.008856 ? t ** (1 / 3) : 7.787 * t + 16 / 116);
  const l = 116 * f(y / 100) - 16;
  const aa = 500 * (f(x / 95.047) - f(y / 100));
  const bb = 200 * (f(y / 100) - f(z / 108.883));
  return [l, aa, bb];
}

function labToRgb(l, a, b) {
  const fInv = (ft) => { const cube = ft * ft * ft; return cube > 0.008856 ? cube : (ft - 16 / 116) / 7.787; };
  const fy = (l + 16) / 116, fx = a / 500 + fy, fz = fy - b / 200;
  const rLin = fInv(fx) * (95.047 / 100), gLin = fInv(fy), bLin = fInv(fz) * (108.883 / 100);
  const toSrgbByte = (u) => {
    const c = Math.max(0, Math.min(1, u));
    const x = c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
    return Math.max(0, Math.min(255, Math.round(x * 255)));
  };
  return [toSrgbByte(rLin), toSrgbByte(gLin), toSrgbByte(bLin)];
}

/** LAB 网格对齐合并相近色（站内「合并相近色」，strength 0~100） */
function mergeNearbyLab(imageData, strength) {
  if (strength <= 0) return imageData;
  const t = strength / 100;
  const stepL = 2 + t * 26, stepA = 1.5 + t * 14, stepB = 1.5 + t * 14;
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] < 128) continue;
    let [L, la, lb] = rgbToLab(d[i], d[i + 1], d[i + 2]);
    L = Math.round(L / stepL) * stepL;
    la = Math.round(la / stepA) * stepA;
    lb = Math.round(lb / stepB) * stepB;
    const [r, g, b] = labToRgb(L, la, lb);
    d[i] = r; d[i + 1] = g; d[i + 2] = b;
  }
  return imageData;
}

/** 中位切分提色（站内 16 色那套，色数可调） */
function medianCutPalette(imageData, n = 16, maxSample = 10000) {
  const d = imageData.data;
  const pixels = [];
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] < 128) continue;
    pixels.push([d[i], d[i + 1], d[i + 2]]);
  }
  if (pixels.length === 0) pixels.push([255, 255, 255]);
  const step = pixels.length > maxSample ? Math.ceil(pixels.length / maxSample) : 1;
  const sampled = step > 1 ? pixels.filter((_, i) => i % step === 0) : pixels;

  let boxes = [[...sampled]];
  while (boxes.length < n) {
    let maxRange = -1, splitIdx = 0, channel = 0;
    for (let i = 0; i < boxes.length; i++) {
      const box = boxes[i];
      if (box.length <= 1) continue;
      const r1 = Math.max(...box.map((p) => p[0])), r0 = Math.min(...box.map((p) => p[0]));
      const g1 = Math.max(...box.map((p) => p[1])), g0 = Math.min(...box.map((p) => p[1]));
      const b1 = Math.max(...box.map((p) => p[2])), b0 = Math.min(...box.map((p) => p[2]));
      const dr = r1 - r0, dg = g1 - g0, db = b1 - b0;
      const max = Math.max(dr, dg, db);
      if (max > maxRange) { maxRange = max; splitIdx = i; channel = dr >= dg && dr >= db ? 0 : dg >= db ? 1 : 2; }
    }
    const box = boxes[splitIdx];
    if (!box || box.length <= 1) break;
    box.sort((a, b) => a[channel] - b[channel]);
    const mid = Math.floor(box.length / 2);
    boxes = [...boxes.slice(0, splitIdx), box.slice(0, mid), box.slice(mid), ...boxes.slice(splitIdx + 1)];
  }
  return boxes.filter((bx) => bx.length > 0).map((box) => [
    Math.round(box.reduce((s, p) => s + p[0], 0) / box.length),
    Math.round(box.reduce((s, p) => s + p[1], 0) / box.length),
    Math.round(box.reduce((s, p) => s + p[2], 0) / box.length),
  ]);
}

/**
 * 把图映射到指定色表。
 * @param palette  [[r,g,b], ...]
 * @param method   'rgb' | 'lab'
 * @param dither   0 = 不抖动；>0 为 Floyd–Steinberg 强度（1 = 原版强度）
 */
function mapToPalette(imageData, palette, opts = {}) {
  const method = opts.method ?? 'lab';
  const dither = opts.dither ?? 0;
  const alphaThreshold = opts.alphaThreshold ?? 128;
  const w = imageData.width, h = imageData.height;
  const data = new Uint8ClampedArray(imageData.data);
  const pal = palette.length ? palette : [[0, 0, 0]];
  const palLab = method === 'lab' ? pal.map((p) => rgbToLab(p[0], p[1], p[2])) : null;

  const nearest = (rgb) => {
    const r = rgb[0], g = rgb[1], b = rgb[2];
    let best = pal[0], bestD = Infinity;
    if (method === 'lab' && palLab) {
      const [l, a, bb] = rgbToLab(r, g, b);
      for (let i = 0; i < pal.length; i++) {
        const [pl, pa, pb] = palLab[i];
        const d = (l - pl) ** 2 + (a - pa) ** 2 + (bb - pb) ** 2;
        if (d < bestD) { bestD = d; best = pal[i]; }
      }
    } else {
      for (let i = 0; i < pal.length; i++) {
        const p = pal[i];
        const d = (r - p[0]) ** 2 + (g - p[1]) ** 2 + (b - p[2]) ** 2;
        if (d < bestD) { bestD = d; best = p; }
      }
    }
    return best;
  };

  if (dither > 0) {
    const errBuf = new Float32Array((w + 2) * (h + 2) * 3);
    const ei = (y, x, c) => ((y + 1) * (w + 2) + (x + 1)) * 3 + c;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        if (data[i + 3] < alphaThreshold) { data[i] = 0; data[i + 1] = 0; data[i + 2] = 0; data[i + 3] = 0; continue; }
        const r = Math.max(0, Math.min(255, data[i] + errBuf[ei(y, x, 0)]));
        const g = Math.max(0, Math.min(255, data[i + 1] + errBuf[ei(y, x, 1)]));
        const b = Math.max(0, Math.min(255, data[i + 2] + errBuf[ei(y, x, 2)]));
        const [pr, pg, pb] = nearest([r, g, b]);
        data[i] = pr; data[i + 1] = pg; data[i + 2] = pb;
        const k = 16 / dither;   // dither=1 → 与原版 /16 一致
        const er = (r - pr) / k, eg = (g - pg) / k, eb = (b - pb) / k;
        errBuf[ei(y, x + 1, 0)] += er * 7; errBuf[ei(y, x + 1, 1)] += eg * 7; errBuf[ei(y, x + 1, 2)] += eb * 7;
        errBuf[ei(y + 1, x - 1, 0)] += er * 3; errBuf[ei(y + 1, x - 1, 1)] += eg * 3; errBuf[ei(y + 1, x - 1, 2)] += eb * 3;
        errBuf[ei(y + 1, x, 0)] += er * 5; errBuf[ei(y + 1, x, 1)] += eg * 5; errBuf[ei(y + 1, x, 2)] += eb * 5;
        errBuf[ei(y + 1, x + 1, 0)] += er * 1; errBuf[ei(y + 1, x + 1, 1)] += eg * 1; errBuf[ei(y + 1, x + 1, 2)] += eb * 1;
      }
    }
  } else {
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] < alphaThreshold) { data[i] = 0; data[i + 1] = 0; data[i + 2] = 0; data[i + 3] = 0; continue; }
      const [r, g, b] = nearest([data[i], data[i + 1], data[i + 2]]);
      data[i] = r; data[i + 1] = g; data[i + 2] = b;
    }
  }
  return new ImageData(data, w, h);
}

/** 统计图里实际用到的颜色及像素数 */
function countColors(imageData, alphaThreshold = 128) {
  const d = imageData.data, m = new Map();
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] < alphaThreshold) continue;
    const key = (d[i] << 16) | (d[i + 1] << 8) | d[i + 2];
    m.set(key, (m.get(key) || 0) + 1);
  }
  return [...m.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([key, n]) => ({ rgb: [(key >> 16) & 255, (key >> 8) & 255, key & 255], count: n }));
}

/* ============================ 抠图：背景 → 透明 ============================ */
/**
 * 从四条边取样猜背景色（棋盘格会出两种），按出现次数降序。
 * @returns [{ rgb:[r,g,b], count }]
 */
function sampleEdgeColors(imageData, maxColors = 2, band = 3) {
  const { width: W, height: H, data: d } = imageData;
  const m = new Map();
  const bandY = Math.min(band, Math.floor(H / 2));
  const bandX = Math.min(band, Math.floor(W / 2));
  const add = (x, y) => {
    const i = (y * W + x) * 4;
    if (d[i + 3] < 20) return;
    const key = ((d[i] >> 3) << 10) | ((d[i + 1] >> 3) << 5) | (d[i + 2] >> 3);
    let e = m.get(key);
    if (!e) { e = { r: 0, g: 0, b: 0, n: 0 }; m.set(key, e); }
    e.r += d[i]; e.g += d[i + 1]; e.b += d[i + 2]; e.n++;
  };
  for (let y = 0; y < bandY; y++) for (let x = 0; x < W; x++) { add(x, y); add(x, H - 1 - y); }
  for (let x = 0; x < bandX; x++) for (let y = 0; y < H; y++) { add(x, y); add(W - 1 - x, y); }

  const total = [...m.values()].reduce((a, e) => a + e.n, 0);
  const sampled = (bandY * W * 2) + (bandX * H * 2);      // 四边一共看了多少像素
  if (!total || total < sampled * 0.03) return [];        // 四边几乎全透明 → 这图已经是透明底
  return [...m.values()]
    .sort((a, b) => b.n - a.n)
    .slice(0, maxColors)
    .map((e) => ({ rgb: [Math.round(e.r / e.n), Math.round(e.g / e.n), Math.round(e.b / e.n)], count: e.n }));
}

/**
 * 全图主色：抽样统计出现最多的颜色（量化到 5 位）。用来兜底边缘取样失败的情况。
 * @param maxColors 取前几名
 * @returns [{ rgb:[r,g,b], count, share }]，share 是占全图不透明像素的比例
 */
function sampleDominantColors(imageData, maxColors = 2) {
  const { width: W, height: H, data: d } = imageData;
  const m = new Map();
  let total = 0;
  const step = Math.max(1, Math.round(Math.max(W, H) / 512));   // 大图抽样，控制在 ~26 万点以内
  for (let y = 0; y < H; y += step) {
    for (let x = 0; x < W; x += step) {
      const i = (y * W + x) * 4;
      if (d[i + 3] < 20) continue;
      const key = ((d[i] >> 3) << 10) | ((d[i + 1] >> 3) << 5) | (d[i + 2] >> 3);
      let e = m.get(key);
      if (!e) { e = { r: 0, g: 0, b: 0, n: 0 }; m.set(key, e); }
      e.r += d[i]; e.g += d[i + 1]; e.b += d[i + 2]; e.n++;
      total++;
    }
  }
  if (!total) return [];
  // 同一个背景色在编码后常有一堆微差变体（#f906f1/#f40aed/#f506ee…），
  // 先按 LAB 距离把相邻桶合并，再比占比，否则每个变体都够不到门槛
  const top = [...m.values()]
    .map((e) => ({ rgb: [Math.round(e.r / e.n), Math.round(e.g / e.n), Math.round(e.b / e.n)], count: e.n }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 16);
  const merged = [];
  for (const it of top) {
    const l = rgbToLab(it.rgb[0], it.rgb[1], it.rgb[2]);
    let hit = null;
    for (const g of merged) {
      const gl = rgbToLab(g.rgb[0], g.rgb[1], g.rgb[2]);
      if (Math.hypot(l[0] - gl[0], l[1] - gl[1], l[2] - gl[2]) < 12) { hit = g; break; }
    }
    if (hit) {
      const n = hit.count + it.count;
      hit.rgb = [0, 1, 2].map((k) => Math.round((hit.rgb[k] * hit.count + it.rgb[k] * it.count) / n));
      hit.count = n;
    } else {
      merged.push({ rgb: it.rgb.slice(), count: it.count });
    }
  }
  return merged
    .sort((a, b) => b.count - a.count)
    .slice(0, maxColors)
    .map((e) => ({ rgb: e.rgb, count: e.count, share: e.count / total }));
}

/**
 * 猜背景色：先看四边，四边被白边/杂边污染时退到「全图主色」。
 * 判据：全图有一种颜色占了 ≥35%，且跟四边取到的色差得远（LAB ΔE > 18）。
 * @returns { colors:[[r,g,b],…], guessed:boolean }
 */
function guessBackgroundColors(imageData) {
  const edge = sampleEdgeColors(imageData, 2).map((c) => c.rgb);
  const dom = sampleDominantColors(imageData, 1);
  if (dom.length && dom[0].share >= 0.35) {
    const dc = dom[0].rgb, dl = rgbToLab(dc[0], dc[1], dc[2]);
    const far = !edge.length || edge.every((c) => {
      const cl = rgbToLab(c[0], c[1], c[2]);
      return Math.hypot(dl[0] - cl[0], dl[1] - cl[1], dl[2] - cl[2]) > 18;
    });
    if (far) return { colors: [dc], guessed: true };
  }
  return { colors: edge, guessed: false };
}

/* RGB → HSV，色相 0–360 */
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), v = mx;
  const s = mx ? (mx - mn) / mx : 0;
  let h = 0;
  if (mx !== mn) {
    const dd = mx - mn;
    if (mx === r) h = (g - b) / dd + (g < b ? 6 : 0);
    else if (mx === g) h = (b - r) / dd + 2;
    else h = (r - g) / dd + 4;
    h *= 60;
  }
  return [h, s, v];
}

/* 色相环上的最短距离（0–180） */
function hueDist(a, b) {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

/**
 * 描边一圈：在主体外侧的透明像素上画 1 像素边 —— 像素画标准的外描边。
 * 只往透明处画，不覆盖主体本身；判据用原始像素，所以点三次是 3 像素粗，不会一次连锁加粗。
 * @param opts.color  描边色 [r,g,b] 或 [r,g,b,a]，默认近黑 #1a1a22
 * @param opts.mask   可选 Uint8Array（长度 = W*H），只处理 mask 里为 1 的像素
 * @param opts.mode   'out'(默认，往外描) | 'in'(往里描，吃掉主体最外一圈)
 * @returns { imageData, pixels }
 */
function strokeOutline(imageData, opts = {}) {
  const W = imageData.width, H = imageData.height, d = imageData.data, N = W * H;
  const col = opts.color || [26, 26, 34, 255];
  const alpha = col.length > 3 ? col[3] : 255;
  const mask = opts.mask || null;
  const inward = opts.mode === 'in';
  const out = new Uint8ClampedArray(d);
  let n = 0;
  for (let p = 0; p < N; p++) {
    if (mask && !mask[p]) continue;
    const i = p * 4;
    const solid = d[i + 3] >= 20;
    if (inward ? !solid : solid) continue;     // 外描边只看透明点，内描边只看不透明点
    const x = p % W, y = (p - x) / W;
    let hit = false;
    for (let dy = -1; dy <= 1 && !hit; dy++) {
      const yy = y + dy;
      if (yy < 0 || yy >= H) continue;
      for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const xx = x + dx;
        if (xx < 0 || xx >= W) continue;
        const a = d[(yy * W + xx) * 4 + 3] >= 20;
        if (inward ? !a : a) { hit = true; break; }
      }
    }
    if (!hit) continue;
    out[i] = col[0]; out[i + 1] = col[1]; out[i + 2] = col[2];
    out[i + 3] = inward ? 255 : alpha;
    n++;
  }
  return {
    imageData: typeof ImageData !== 'undefined' ? new ImageData(out, W, H) : { width: W, height: H, data: out },
    pixels: n,
  };
}

/**
 * 收缩一圈：把「透明区」往里扩 px 个像素 —— 等价于不透明轮廓整体瘦一圈。
 * 纯几何，不看颜色（对比 erodeEdgeOnce 只吃像背景色的像素）。
 * @param opts.px  收缩几个像素，默认 1
 * @param opts.mask 可选 Uint8Array（长度 = W*H），只处理 mask 里为 1 的像素
 * @returns { imageData, removed }
 */
function shrinkAlphaOnce(imageData, opts = {}) {
  const W = imageData.width, H = imageData.height, d = imageData.data, N = W * H;
  const px = Math.max(1, Math.min(16, opts.px ?? 1));
  const mask = opts.mask || null;
  let src = new Uint8ClampedArray(d);
  let out = new Uint8ClampedArray(d);
  let removed = 0;
  for (let pass = 0; pass < px; pass++) {
    let n = 0;
    for (let p = 0; p < N; p++) {
      if (mask && !mask[p]) continue;
      const i = p * 4;
      if (src[i + 3] < 20) continue;
      const x = p % W, y = (p - x) / W;
      let touch = false;
      for (let dy = -1; dy <= 1 && !touch; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= W) continue;
          if (src[(yy * W + xx) * 4 + 3] < 20) { touch = true; break; }
        }
      }
      if (touch) { out[i + 3] = 0; n++; }
    }
    removed += n;
    if (!n) break;
    src = new Uint8ClampedArray(out);
  }
  return {
    imageData: typeof ImageData !== 'undefined' ? new ImageData(out, W, H) : { width: W, height: H, data: out },
    removed,
  };
}

/**
 * 把画面边缘再削掉一圈（1 像素）：紧贴着透明区、而且颜色接近背景色的像素 → 变透明。
 * 点一次削一圈，反复点就是一圈一圈往外推。适合清掉抠图后残留的背景色描边 / 光晕。
 * @param opts.bgColors  背景色 [[r,g,b],…]，空则从四边取样
 * @param opts.tolerance 0–100，与抠图同一个容差
 * @param opts.mult      边缘阈值放宽倍数，默认 1.8（边缘像素是混色，比纯背景色偏得多）
 * @returns { imageData, removed }
 */
function erodeEdgeOnce(imageData, opts = {}) {
  const W = imageData.width, H = imageData.height, d = imageData.data, N = W * H;
  const tol = Math.max(0, Math.min(100, opts.tolerance ?? 32));
  const thr = tol * 0.6 * (opts.mult ?? 1.8);
  const bgColors = opts.bgColors && opts.bgColors.length
    ? opts.bgColors.map((c) => c.slice(0, 3))
    : sampleEdgeColors(imageData, 2).map((c) => c.rgb);
  const bgLab = bgColors.map((c) => rgbToLab(c[0], c[1], c[2]));
  const near = (r, g, b) => {
    for (let k = 0; k < bgColors.length; k++) {
      const dr = r - bgColors[k][0], dg = g - bgColors[k][1], db = b - bgColors[k][2];
      if (dr * dr + dg * dg + db * db > (thr * 3.4) ** 2) continue;
      const L = rgbToLab(r, g, b), q = bgLab[k];
      if (Math.sqrt((L[0] - q[0]) ** 2 + (L[1] - q[1]) ** 2 + (L[2] - q[2]) ** 2) <= thr) return true;
    }
    return false;
  };
  const out = new Uint8ClampedArray(d);
  let removed = 0;
  for (let p = 0; p < N; p++) {
    const i = p * 4;
    if (d[i + 3] < 20) continue;                       // 本来就是透明的
    const x = p % W, y = (p - x) / W;
    let touch = false;
    if (x > 0 && d[i - 1] < 20) touch = true;
    else if (x < W - 1 && d[i + 7] < 20) touch = true;
    else if (y > 0 && d[i - W * 4 + 3] < 20) touch = true;
    else if (y < H - 1 && d[i + W * 4 + 3] < 20) touch = true;
    if (!touch) continue;
    if (near(d[i], d[i + 1], d[i + 2])) { out[i + 3] = 0; removed++; }
  }
  return {
    imageData: typeof ImageData !== 'undefined' ? new ImageData(out, W, H) : { width: W, height: H, data: out },
    removed,
  };
}

/**
 * 把背景变全透明。背景 = 与给定背景色的 LAB 距离小于阈值的像素（可限定只吃跟画布边缘连通的）。
 * @param opts.bgColors     背景色数组 [[r,g,b],…]，空则自动从四边取样
 * @param opts.tolerance    0–100，换算成 LAB ΔE 阈值 = tolerance × 0.6
 * @param opts.edgeConnected 默认 true：只有跟边缘连着的背景才抠，角色身上的同色块不会被挖穿
 * @param opts.despeckle    默认 true：去掉抠出来的孤立小洞（< 24 像素）
 * @param opts.lightFringe  默认 true：紧贴透明区、亮度 > 215 的白边像素一起去掉
 * @returns { imageData, removed, total, bgColors }
 */
function removeBackground(imageData, opts = {}) {
  const W = imageData.width, H = imageData.height, d = imageData.data;
  const N = W * H;
  const tol = Math.max(0, Math.min(100, opts.tolerance ?? 32));
  const baseThr = tol * 0.6;
  const edgeConnected = opts.edgeConnected ?? true;
  const despeckle = opts.despeckle ?? true;
  const lightFringe = opts.lightFringe ?? true;
  const cleanPasses = Math.max(0, Math.min(4, opts.cleanPasses ?? 0));   // 边缘毛刺清理（默认关，见下）
  const cleanSpecks = opts.cleanSpecks ?? true;                           // 清掉与主体分离的小碎屑
  const keepLargest = opts.keepLargest ?? false;                          // 只留最大主体（更狠）
  const binAlpha = opts.binAlpha ?? true;                                 // 硬边：不留半透明灰边
  const autoRetry = opts.autoRetry ?? false;                             // 残留多就再推两轮边缘
  const bgIslands = opts.bgIslands ?? false;                             // 清掉被轮廓夹住的背景色孤岛
  const islandMinArea = Math.max(8, opts.islandMinArea ?? 48);           // 孤岛面积阈值：够大就是背景，不用非挨着透明区

  /* ---- Alpha 通道模式：图本身就是透明底，直接按 alpha 抠，一个像素都不会误伤 ---- */
  if (opts.alphaMode) {
    const thr = Math.max(1, Math.min(255, opts.alphaThr ?? 128));   // 低于它算背景
    const out = new Uint8ClampedArray(d);
    let removed = 0, opaque = 0;
    for (let p = 0; p < N; p++) {
      const i = p * 4, a = d[i + 3];
      if (a >= 20) opaque++;
      if (a < thr) { out[i + 3] = 0; if (a >= 20) removed++; }
      else if (binAlpha) out[i + 3] = 255;                          // 硬边
    }
    return {
      imageData: typeof ImageData !== 'undefined' ? new ImageData(out, W, H) : { width: W, height: H, data: out },
      removed, opaque, total: N, bgColors: [], alphaMode: true, alphaThr: thr,
    };
  }

  let bgColors, bgGuessed = false;
  if (opts.bgColors && opts.bgColors.length) {
    bgColors = opts.bgColors.map((c) => c.slice(0, 3));
  } else {
    const g = guessBackgroundColors(imageData);
    bgColors = g.colors; bgGuessed = g.guessed;
  }
  if (!bgColors.length) {
    // 四边都是透明的：这张图已经是透明底，没什么可抠的，原样返回
    const img0 = typeof ImageData !== 'undefined'
      ? new ImageData(new Uint8ClampedArray(d), W, H)
      : { width: W, height: H, data: new Uint8ClampedArray(d) };
    let op0 = 0;
    for (let p = 0; p < N; p++) if (d[p * 4 + 3] >= 20) op0++;
    return { imageData: img0, removed: 0, opaque: op0, total: N, bgColors: [], noBg: true };
  }
  const bgLab = bgColors.map((c) => rgbToLab(c[0], c[1], c[2]));
  // 纯色幕（洋红 / 绿幕 / 蓝幕）：背景色的饱和度很高时，同色相不同明暗的像素也算背景
  const bgHsv = bgColors.map((c) => rgbToHsv(c[0], c[1], c[2]));
  const isKeyColor = bgHsv.some((h) => h[1] >= 0.45);
  const sameHue = (r, g, b) => {
    const h = rgbToHsv(r, g, b);
    if (h[1] < 0.30) return false;                       // 灰 / 黑白不算
    for (const q of bgHsv) {
      if (q[1] < 0.45) continue;
      if (hueDist(h[0], q[0]) <= 22 && Math.abs(h[2] - q[2]) < 0.5) return true;
    }
    return false;
  };

  const near = (r, g, b, mult = 1, thrArg) => {
    const t = (thrArg === undefined ? baseThr : thrArg) * mult;
    for (let k = 0; k < bgColors.length; k++) {
      const dr = r - bgColors[k][0], dg = g - bgColors[k][1], db = b - bgColors[k][2];
      if (dr * dr + dg * dg + db * db > (t * 3.4) * (t * 3.4)) continue;  // 粗筛：明显不是背景
      const L = rgbToLab(r, g, b), q = bgLab[k];
      const dE = Math.sqrt((L[0] - q[0]) ** 2 + (L[1] - q[1]) ** 2 + (L[2] - q[2]) ** 2);
      if (dE <= t) return true;
    }
    return false;
  };

  const neighbors = (p, fn) => {
    const x = p % W, y = (p - x) / W;
    if (x > 0) fn(p - 1);
    if (x < W - 1) fn(p + 1);
    if (y > 0) fn(p - W);
    if (y < H - 1) fn(p + W);
  };

  /* 给定阈值生成一张遮罩（1 = 要抠掉） */
  function buildMask(thr, passes = cleanPasses) {
    const mask = new Uint8Array(N);

    // 1) 颜色候选
    for (let p = 0; p < N; p++) {
      const i = p * 4;
      const a = d[i + 3];
      if (a < 20) { mask[p] = 1; continue; }
      // 半透明边缘像素多半是抗锯齿物，阈值放宽一点，否则会留一圈背景色
      if (near(d[i], d[i + 1], d[i + 2], a < 252 ? 1.8 : 1, thr)) mask[p] = 1;
    }

    // 2) 只保留与画布边缘连通的背景（4 邻域 BFS）
    if (edgeConnected) {
      const keep = new Uint8Array(N);
      const stack = [];
      const push = (p) => { if (mask[p] && !keep[p]) { keep[p] = 1; stack.push(p); } };
      for (let x = 0; x < W; x++) { push(x); push((H - 1) * W + x); }
      for (let y = 0; y < H; y++) { push(y * W); push(y * W + W - 1); }
      while (stack.length) {
        const p = stack.pop();
        neighbors(p, (q) => push(q));
      }
      mask.set(keep);
    }

    // 3) 孤岛去噪：小于 24 像素的透明块补回来（AI 图里的碎噪点）
    if (despeckle) {
      const seen = new Uint8Array(N);
      for (let p = 0; p < N; p++) {
        if (!mask[p] || seen[p]) continue;
        const comp = [p]; seen[p] = 1;
        for (let h = 0; h < comp.length; h++) {
          neighbors(comp[h], (q) => { if (mask[q] && !seen[q]) { seen[q] = 1; comp.push(q); } });
        }
        const touchesEdge = comp.some((q) => {
          const x = q % W, y = (q - x) / W;
          return x === 0 || y === 0 || x === W - 1 || y === H - 1;
        });
        if (comp.length < 24 && !touchesEdge) for (const q of comp) mask[q] = 0;
      }
    }

    // 4) 白边：紧贴透明区、亮度很高的像素一起去掉
    if (lightFringe && thr > 0) {
      const add = [];
      for (let p = 0; p < N; p++) {
        if (mask[p]) continue;
        const i = p * 4;
        const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        if (lum < 215) continue;
        let adj = false;
        neighbors(p, (q) => { if (mask[q]) adj = true; });
        if (adj) add.push(p);
      }
      for (const p of add) mask[p] = 1;
    }

    // 5) 种子 + 背景色洪泛：先找"贴着已透明区、颜色又像背景"的种子像素，
    //    再从种子沿同色像素整块扩开。卡在轮廓凹槽里的背景色块（棋盘格白块最典型）
    //    内部不挨着透明区，只清 1px 边缘永远清不掉 —— 洪泛才能一次吃掉整块。
    //    洪泛走的是严格阈值，机体色挡得住，白色贴花/高光被轮廓包着没有种子，也不会被误吃。
    for (let pass = 0; pass < passes; pass++) {
      const mult = 1.3 + pass * 0.25;
      const stack = [];
      for (let p = 0; p < N; p++) {
        if (mask[p]) continue;
        const x = p % W, y = (p - x) / W;
        let t = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= H) { t += 3; continue; }
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = x + dx;
            if (nx < 0 || nx >= W) { t++; continue; }
            if (mask[ny * W + nx]) t++;
          }
        }
        if (t < 2) continue;
        const i = p * 4;
        if (near(d[i], d[i + 1], d[i + 2], mult, thr)) stack.push(p);
      }
      if (!stack.length) break;
      for (let h = 0; h < stack.length; h++) {
        const p = stack[h];
        if (mask[p]) continue;
        mask[p] = 1;
        neighbors(p, (q) => {
          if (mask[q]) return;
          const i = q * 4;
          if (near(d[i], d[i + 1], d[i + 2], 1, thr)) stack.push(q);   // 严格阈值，只吃真·背景色
        });
      }
    }

    // 5.5) 背景色孤岛：被机体轮廓夹住、但贴着轮廓边缘的背景色块（棋盘格白块最典型）
    //      判据 = 整块里只要有像素落在"距透明区 3px"的带内就整块清掉；
    //      真正被机体包在中间深处的白色贴花离透明区很远，不会被误吃。
    if (bgIslands) {
      const band = new Uint8Array(N);
      let frontier = [];
      for (let p = 0; p < N; p++) if (mask[p]) frontier.push(p);
      for (let lvl = 0; lvl < 3 && frontier.length; lvl++) {
        const next = [];
        for (const p of frontier) {
          neighbors(p, (q) => {
            if (mask[q] || band[q]) return;
            band[q] = 1; next.push(q);
          });
        }
        frontier = next;
      }
      const seen2 = new Uint8Array(N);
      for (let p = 0; p < N; p++) {
        if (mask[p] || seen2[p]) continue;
        const i = p * 4;
        if (d[i + 3] < 20 || !(near(d[i], d[i + 1], d[i + 2], 1, thr * 1.5) || (isKeyColor && sameHue(d[i], d[i + 1], d[i + 2])))) { seen2[p] = 1; continue; }
        const comp = [p]; seen2[p] = 1;
        let touch = band[p] === 1;
        for (let h = 0; h < comp.length; h++) {
          neighbors(comp[h], (r) => {
            if (mask[r] || seen2[r]) return;
            const j = r * 4;
            // 孤岛用放宽 1.5 倍的阈值：被机身包住的洋红/绿幕通常被压暗或带阴影，比纯背景色偏得多
            if (!(near(d[j], d[j + 1], d[j + 2], 1, thr * 1.5) || (isKeyColor && sameHue(d[j], d[j + 1], d[j + 2])))) { seen2[r] = 1; return; }
            seen2[r] = 1; comp.push(r);
            if (band[r]) touch = true;
          });
        }
        // 挨着轮廓边缘（3px 带内）的整块清掉；没挨着的，只要面积够大（默认 48 像素）也当背景
        // —— 洋红 / 绿幕底最典型的残法就是机身里透出来一大块，离边缘远但明显是背景
        if ((touch || comp.length >= islandMinArea) && comp.length <= 60000) {
          for (const q of comp) mask[q] = 1;
        }
      }
    }

    // 6) 清掉和主体分离的小碎屑（阈值按主体面积算，分离的大零件会保留）
    if (cleanSpecks || keepLargest) {
      const seen = new Uint8Array(N);
      const comps = [];
      for (let p = 0; p < N; p++) {
        if (mask[p] || seen[p]) continue;
        const comp = [p]; seen[p] = 1;
        for (let h = 0; h < comp.length; h++) {
          neighbors(comp[h], (q) => { if (!mask[q] && !seen[q]) { seen[q] = 1; comp.push(q); } });
        }
        comps.push(comp);
      }
      let maxA = 0;
      for (const c of comps) if (c.length > maxA) maxA = c.length;
      // 碎屑阈值取小值：分离的零件（尾翼、炸弹、起落架）通常几百像素，碎点只有个位数
      const minA = keepLargest ? Infinity : 24;   // 只清 24 像素以下的碎点，分离的零件一块都不动
      for (const c of comps) {
        if (keepLargest ? c.length < maxA : c.length < minA) for (const p of c) mask[p] = 1;
      }
    }

    return mask;
  }

  /* 抠完之后还剩多少"像背景"的不透明像素 —— 用来自动决定要不要加大容差重来 */
  function residual(mask, thr) {
    let leftBg = 0, op = 0;
    for (let p = 0; p < N; p++) {
      const i = p * 4;
      if (d[i + 3] < 20 || mask[p]) { if (d[i + 3] >= 20) op++; continue; }
      op++;
      if (near(d[i], d[i + 1], d[i + 2], 1, thr)) leftBg++;
    }
    return { leftBg, op };
  }

  let thr = baseThr;
  let mask = buildMask(thr);
  let retries = 0;
  if (autoRetry && baseThr > 0) {
    // 残留超过 2% 就自动把"边缘清理"多推 2 轮（只吃紧贴透明区的那圈，
    // 绝不放宽全局容差 —— 深色机体挨着深色背景时，放宽全局会把机身啃掉）
    const r = residual(mask, thr);
    if (r.op > 0 && r.leftBg / r.op > 0.02 && cleanPasses < 3) {
      mask = buildMask(thr, Math.min(4, cleanPasses + 2));
      retries = cleanPasses + 2;
    }
  }

  const out = new Uint8ClampedArray(d);
  let removed = 0, opaque = 0;
  for (let p = 0; p < N; p++) {
    const i = p * 4;
    const wasOpaque = d[i + 3] >= 20;
    if (wasOpaque) opaque++;
    if (mask[p]) {
      out[i + 3] = 0;
      if (wasOpaque) removed++;
    } else if (binAlpha && out[i + 3] < 255) {
      out[i + 3] = 255;          // 硬边：像素画不需要半透明灰边
    }
  }
  // 浏览器里返回真 ImageData，才能直接 putImageData；Node 里回落到普通对象
  const img = typeof ImageData !== 'undefined' ? new ImageData(out, W, H) : { width: W, height: H, data: out };
  return { imageData: img, removed, opaque, total: N, bgColors, thr, retries, keepLargest, bgGuessed };
}


/* ============================ 像素块（站内 pixelateImage 的等价实现 + 可选项） ============================ */
/**
 * 像素块风格：把图按 block×block 求平均压成小图，再用最近邻放回。
 * @param block     块尺寸（站内范围 2–128）
 * @param keepSize  true = 输出仍为原尺寸（站内默认，边缘保持硬块）；false = 输出缩小后的逻辑尺寸
 * @param resample  'box' 平均（站内 canvas 缩小等价）| 'center' 取块中心像素（更保色）
 */
function pixelateImage(imageData, opts = {}) {
  const block = Math.max(2, Math.round(opts.block ?? 8));
  const keepSize = opts.keepSize ?? true;
  const resample = opts.resample ?? 'box';
  const W = imageData.width, H = imageData.height;
  const d = imageData.data;
  const nw = Math.max(1, Math.round(W / block));
  const nh = Math.max(1, Math.round(H / block));
  const small = new Uint8ClampedArray(nw * nh * 4);

  for (let gy = 0; gy < nh; gy++) {
    const y0 = Math.floor((gy * H) / nh), y1 = Math.max(y0 + 1, Math.floor(((gy + 1) * H) / nh));
    for (let gx = 0; gx < nw; gx++) {
      const x0 = Math.floor((gx * W) / nw), x1 = Math.max(x0 + 1, Math.floor(((gx + 1) * W) / nw));
      const o = (gy * nw + gx) * 4;
      if (resample === 'center') {
        const cx = Math.min(W - 1, Math.floor((x0 + x1) / 2)), cy = Math.min(H - 1, Math.floor((y0 + y1) / 2));
        const i = (cy * W + cx) * 4;
        small[o] = d[i]; small[o + 1] = d[i + 1]; small[o + 2] = d[i + 2]; small[o + 3] = d[i + 3];
        continue;
      }
      let sr = 0, sg = 0, sb = 0, sa = 0, n = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = (y * W + x) * 4;
          const al = d[i + 3];
          sr += d[i] * al; sg += d[i + 1] * al; sb += d[i + 2] * al;  // 按 alpha 加权，避免透明区拉黑
          sa += al; n++;
        }
      }
      if (sa > 0) { small[o] = sr / sa; small[o + 1] = sg / sa; small[o + 2] = sb / sa; }
      small[o + 3] = n ? sa / n : 0;
    }
  }

  if (!keepSize) return new ImageData(small, nw, nh);

  const out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    const sy = Math.min(nh - 1, Math.floor((y * nh) / H));
    for (let x = 0; x < W; x++) {
      const sx = Math.min(nw - 1, Math.floor((x * nw) / W));
      const si = (sy * nw + sx) * 4, di = (y * W + x) * 4;
      out[di] = small[si]; out[di + 1] = small[si + 1]; out[di + 2] = small[si + 2]; out[di + 3] = small[si + 3];
    }
  }
  return new ImageData(out, W, H);
}

/** 最近邻放大倍率输出（导出/预览用，像素不糊） */
function nearestScale(imageData, factor) {
  const f = Math.max(1, Math.round(factor));
  if (f === 1) return imageData;
  const W = imageData.width, H = imageData.height, d = imageData.data;
  const out = new Uint8ClampedArray(W * f * H * f * 4);
  for (let y = 0; y < H * f; y++) {
    const sy = Math.floor(y / f);
    for (let x = 0; x < W * f; x++) {
      const sx = Math.floor(x / f);
      const si = (sy * W + sx) * 4, di = (y * W * f + x) * 4;
      out[di] = d[si]; out[di + 1] = d[si + 1]; out[di + 2] = d[si + 2]; out[di + 3] = d[si + 3];
    }
  }
  return new ImageData(out, W * f, H * f);
}


/* ---------------- ZIP 打包（仅存储，不压缩；PNG 本来已压缩） ---------------- */
const _CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(u8) {
  let c = 0xffffffff;
  for (let i = 0; i < u8.length; i++) c = _CRC_TABLE[(c ^ u8[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/**
 * 把 [{name, data:Uint8Array}] 打成一个 zip（存储模式）。
 * 不依赖任何库，浏览器 / node 通用。
 */
function zipStore(files) {
  const enc = new TextEncoder();
  const now = new Date();
  const dosTime = ((now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1)) & 0xffff;
  const dosDate = (((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate()) & 0xffff;

  const locals = [];
  const centrals = [];
  let offset = 0;

  for (const f of files) {
    const name = enc.encode(f.name);
    const data = f.data;
    const crc = crc32(data);

    const lh = new Uint8Array(30 + name.length);
    const lv = new DataView(lh.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);        // version needed
    lv.setUint16(6, 0x0800, true);    // UTF-8 文件名
    lv.setUint16(8, 0, true);         // method: store
    lv.setUint16(10, dosTime, true);
    lv.setUint16(12, dosDate, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, data.length, true);
    lv.setUint32(22, data.length, true);
    lv.setUint16(26, name.length, true);
    lv.setUint16(28, 0, true);
    lh.set(name, 30);
    locals.push(lh, data);

    const ch = new Uint8Array(46 + name.length);
    const cv = new DataView(ch.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0x0800, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, dosTime, true);
    cv.setUint16(14, dosDate, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, data.length, true);
    cv.setUint32(24, data.length, true);
    cv.setUint16(28, name.length, true);
    cv.setUint16(30, 0, true);
    cv.setUint16(32, 0, true);
    cv.setUint16(34, 0, true);
    cv.setUint16(36, 0, true);
    cv.setUint32(38, 0, true);
    cv.setUint32(42, offset, true);
    ch.set(name, 46);
    centrals.push(ch);

    offset += lh.length + data.length;
  }

  let cdSize = 0;
  for (const c of centrals) cdSize += c.length;
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, cdSize, true);
  ev.setUint32(16, offset, true);

  const total = offset + cdSize + 22;
  const out = new Uint8Array(total);
  let p = 0;
  for (const part of locals) { out.set(part, p); p += part.length; }
  for (const c of centrals) { out.set(c, p); p += c.length; }
  out.set(eocd, p);
  return out;
}


/**
 * 去除杂色：删掉孤立的零星碎块，以及和四周颜色都对不上的单像素杂点。
 * @param {ImageData} imageData
 * @param {object} opts
 * @param {number} opts.minArea  连通块面积小于它整块删掉（0 = 不做面积过滤）
 * @param {boolean} opts.dots    是否顺便清掉单像素离群色点
 * @param {number} opts.dotTol   单像素判据：和八个邻居逐个比（RGB 差之和），一个都不小于它就当杂点
 * @returns {{imageData: ImageData, blocks: number, pixels: number, dotPixels: number}}
 */
function removeNoise(imageData, opts = {}) {
  const minArea = opts.minArea ?? 32;
  const dots = opts.dots ?? true;
  const dotTol = opts.dotTol ?? 36;
  const W = imageData.width, H = imageData.height, N = W * H;
  const out = new Uint8ClampedArray(imageData.data);
  let blocks = 0, pixels = 0, dotPixels = 0;

  // 1) 连通块面积过滤：只按 alpha 连通，和主体相连的部分不会被误删
  if (minArea > 0) {
    const seen = new Uint8Array(N);
    const stack = [];
    for (let p0 = 0; p0 < N; p0++) {
      if (seen[p0] || out[p0 * 4 + 3] < 20) continue;
      const comp = [p0]; seen[p0] = 1;
      for (let h = 0; h < comp.length; h++) {
        const p = comp[h], x = p % W, y = (p / W) | 0;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy; if (ny < 0 || ny >= H) continue;
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = x + dx; if (nx < 0 || nx >= W) continue;
            const q = ny * W + nx;
            if (seen[q] || out[q * 4 + 3] < 20) continue;
            seen[q] = 1; comp.push(q);
          }
        }
      }
      if (comp.length < minArea) {
        for (const q of comp) out[q * 4 + 3] = 0;
        blocks++; pixels += comp.length;
      }
    }
  }

  // 2) 单像素离群点：八个邻居里一个都不像它 → 当杂点
  if (dots && W > 2 && H > 2) {
    const a0 = new Uint8ClampedArray(out);
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const p = y * W + x, i = p * 4;
        if (a0[i + 3] < 20) continue;
        const r = a0[i], g = a0[i + 1], b = a0[i + 2];
        let alike = 0, neigh = 0;
        for (let dy = -1; dy <= 1 && !alike; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const j = ((y + dy) * W + (x + dx)) * 4;
            if (a0[j + 3] < 20) continue;
            neigh++;
            if (Math.abs(a0[j] - r) + Math.abs(a0[j + 1] - g) + Math.abs(a0[j + 2] - b) < dotTol) { alike = 1; break; }
          }
        }
        if (neigh > 0 && !alike) { out[i + 3] = 0; dotPixels++; }
      }
    }
  }

  return { imageData: new ImageData(out, W, H), blocks, pixels, dotPixels };
}


/* ===================== GIF 编码（GIF89a，纯 JS，无依赖） ===================== */

/** GIF 用的 LZW 压缩：LSB-first，按 12 位上限自动重置字典 */
function gifLZW(indices, minCodeSize) {
  const clearCode = 1 << minCodeSize;
  const endCode = clearCode + 1;
  let codeSize = minCodeSize + 1;
  let nextCode = endCode + 1;
  let dict = new Map();
  const out = [];
  let bitBuf = 0, bitLen = 0;
  const emit = (code) => {
    bitBuf |= code << bitLen;
    bitLen += codeSize;
    while (bitLen >= 8) { out.push(bitBuf & 0xff); bitBuf >>>= 8; bitLen -= 8; }
  };
  emit(clearCode);
  if (!indices.length) { emit(endCode); if (bitLen > 0) out.push(bitBuf & 0xff); return out; }
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const k = indices[i];
    const key = (prefix << 8) | k;
    const hit = dict.get(key);
    if (hit !== undefined) { prefix = hit; continue; }
    emit(prefix);
    if (nextCode < 4096) {
      dict.set(key, nextCode++);
      if (nextCode > (1 << codeSize) && codeSize < 12) codeSize++;
    } else {
      emit(clearCode);
      dict = new Map();
      codeSize = minCodeSize + 1;
      nextCode = endCode + 1;
    }
    prefix = k;
  }
  emit(prefix);
  emit(endCode);
  if (bitLen > 0) out.push(bitBuf & 0xff);
  return out;
}

/** 把字节流切成 GIF 的 255 字节子块（末尾 0x00 结束） */
function gifSubBlocks(bytes) {
  const out = [];
  for (let i = 0; i < bytes.length; i += 255) {
    const chunk = bytes.slice(i, i + 255);
    out.push(chunk.length, ...chunk);
  }
  out.push(0);
  return out;
}

/**
 * 帧序列 → GIF89a 字节流。
 * @param frames [{ imageData, delay }]  delay 单位毫秒
 * @param opts   { loop: 0=无限循环, maxColors: 256 }
 */
function encodeGIF(frames, opts = {}) {
  if (!frames || !frames.length) throw new Error('没有帧可编码');
  const w = frames[0].imageData.width, h = frames[0].imageData.height;
  for (let i = 0; i < frames.length; i++) {
    const im = frames[i].imageData;
    if (im.width !== w || im.height !== h) throw new Error('所有帧必须同尺寸（第 ' + (i + 1) + ' 帧是 ' + im.width + '×' + im.height + '）');
  }
  const maxColors = Math.max(2, Math.min(256, opts.maxColors || 256));
  const loop = opts.loop === undefined ? 0 : opts.loop;

  // 有没有透明像素 —— 有的话留一个索引当透明色
  let hasAlpha = false;
  for (const f of frames) {
    const d = f.imageData.data;
    for (let i = 3; i < d.length; i += 4) { if (d[i] < 128) { hasAlpha = true; break; } }
    if (hasAlpha) break;
  }
  const palN = hasAlpha ? Math.min(maxColors - 1, 255) : maxColors;

  // 所有帧合并采样提色（逐帧各取一部分，避免一帧吃满采样预算）
  const totalPx = w * h * frames.length;
  const step = Math.max(1, Math.floor(totalPx / 20000));
  const buf = [];
  let seen = 0;
  for (const f of frames) {
    const d = f.imageData.data;
    for (let i = 0; i < d.length; i += 4) {
      if ((seen++ % step) !== 0) continue;
      if (d[i + 3] < 128) continue;
      buf.push(d[i], d[i + 1], d[i + 2], 255);
    }
  }
  let palette;
  if (buf.length >= 4) {
    const n = buf.length / 4;
    palette = medianCutPalette(new ImageData(new Uint8ClampedArray(buf), n, 1), palN, 20000);
  } else {
    palette = [[0, 0, 0]];
  }
  if (!palette.length) palette = [[0, 0, 0]];
  const transIdx = hasAlpha ? palette.length : -1;
  if (hasAlpha) palette = palette.concat([[0, 0, 0]]);     // 末尾这格专供透明

  // 全局色表尺寸必须是 2 的幂
  let tableSize = 2, exp = 0;
  while (tableSize < palette.length) { tableSize *= 2; exp++; }
  while (palette.length < tableSize) palette.push([0, 0, 0]);
  const minCodeSize = Math.max(2, exp + 1);

  // 最近色查找（缓存 RGB 三元组，大序列也不会慢）
  const cache = new Map();
  const nearest = (r, g, b) => {
    const key = (r << 16) | (g << 8) | b;
    const hit = cache.get(key);
    if (hit !== undefined) return hit;
    let best = 0, bestD = Infinity;
    for (let i = 0; i < palette.length; i++) {
      if (i === transIdx) continue;
      const q = palette[i];
      const dr = r - q[0], dg = g - q[1], db = b - q[2];
      const dd = dr * dr + dg * dg + db * db;
      if (dd < bestD) { bestD = dd; best = i; if (dd === 0) break; }
    }
    cache.set(key, best);
    return best;
  };

  const o = [];
  const u16 = (v) => o.push(v & 0xff, (v >> 8) & 0xff);
  // 头 + 逻辑屏幕描述符
  o.push(0x47, 0x49, 0x46, 0x38, 0x39, 0x61);              // "GIF89a"
  u16(w); u16(h);
  o.push(0x80 | (7 << 4) | exp, 0, 0);                      // 全局色表 / 8 位色深
  for (const c of palette) o.push(c[0], c[1], c[2]);
  // Netscape 循环扩展
  o.push(0x21, 0xff, 0x0b, 0x4e, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2e, 0x30);
  o.push(0x03, 0x01); u16(loop); o.push(0x00);

  for (const f of frames) {
    const im = f.imageData, d = im.data;
    const indices = new Uint8Array(w * h);
    for (let p = 0, i = 0; p < indices.length; p++, i += 4) {
      indices[p] = (transIdx >= 0 && d[i + 3] < 128) ? transIdx : nearest(d[i], d[i + 1], d[i + 2]);
    }
    const delay = Math.max(2, Math.round((f.delay === undefined ? 100 : f.delay) / 10));  // 1/100 秒，浏览器会把 0/1 当 10
    // 图形控制扩展：有透明就把 disposal 设成 2（还原背景），否则 1（保留）
    o.push(0x21, 0xf9, 0x04, ((transIdx >= 0 ? 2 : 1) << 2) | (transIdx >= 0 ? 1 : 0));
    u16(delay);
    o.push(transIdx >= 0 ? transIdx : 0, 0x00);
    // 图像描述符（整屏一块，无局部色表、不交错）
    o.push(0x2c); u16(0); u16(0); u16(w); u16(h); o.push(0x00);
    o.push(minCodeSize & 0xff);
    o.push(...gifSubBlocks(gifLZW(indices, minCodeSize)));
  }
  o.push(0x3b);                                             // trailer
  return new Uint8Array(o);
}

const nextPow2 = pow2Ceil;   // 兼容旧导出名

const PixCore = {
  perfectPixel, pixelateImage, nearestScale, detectGrid,
  mapToPalette, medianCutPalette, countColors,
  mergeNearbyLab, rgbToLab, labToRgb, nextPow2,
  removeBackground, removeNoise, erodeEdgeOnce, shrinkAlphaOnce, strokeOutline, rgbToHsv, hueDist, sampleEdgeColors,
  sampleDominantColors, guessBackgroundColors,
  zipStore, crc32, encodeGIF, gifLZW,
};
if (typeof module !== 'undefined' && module.exports) module.exports = PixCore;
if (typeof window !== 'undefined') window.PixCore = PixCore;
