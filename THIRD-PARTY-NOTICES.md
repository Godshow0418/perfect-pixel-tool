# 第三方代码声明 · Third-party notices

第三方代码声明 / Third-party notices

本项目「完美像素」模块（pixcore.js 中的网格检测与格内采样部分：FFT 估算网格周期、
Sobel 边缘细化、center / median / majority 三种采样）是下面这个项目的衍生作品，
按 MIT 许可使用。MIT 全文见同目录的 `LICENSE`。

1. theamusing/perfectPixel
   https://github.com/theamusing/perfectPixel
   Copyright (c) theAmusing
   License: MIT（仓库根目录未放置 LICENSE 文件，但 pyproject.toml 中
   `license = { text = "MIT" }`、README 徽章与 PyPI 分类器均声明为 MIT）
   算法：FFT 估算网格周期 + Sobel 边缘细化 + 格内采样

   本项目的 JavaScript 实现是对照其 src/perfect_pixel/perfect_pixel_noCV2.py
   重新编写的（原版为 Python + numpy 的向量化写法，本项目改为显式循环）。
   算法与流程一致，代码为本项目所写。

本项目的其余部分——抠图、削边、描边、杂色清理、色表对标（含中位切分的实现）、
视频转帧、GIF 编解码、ZIP 打包，以及 detectGrid / detectPitchAxis 这套
「整数块一致性」检测（上游算法中不存在，为本项目为处理 AI 出图糊边而自研）
——为本项目独立实现，同样以 MIT 许可提供。

「中位切分」（median cut）为 Paul Heckbert 于 1982 年发表的通用色彩量化算法，
属于公有领域的算法思想，不涉及任何第三方代码授权。
