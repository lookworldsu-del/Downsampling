# DownsamplingBenchmark

iOS 实时摄像头采集场景下，CPU vs GPU Downsampling 性能对比工具。

## 项目背景

在移动端图像处理中，Downsampling（降采样）是最常见的操作之一——从相机预览、缩略图生成到视频流预处理都会用到。本项目在 **真实的实时摄像头采集场景** 下，分别实现了 CPU 和 GPU 两大类共 **6 种** Downsampling 方案，并通过全面的性能指标进行实时对比。

## 架构设计

```
┌─────────────────────────────────────────────────┐
│              AVCaptureSession                    │
│          (1080p / 4K 实时采集)                    │
└────────────────────┬────────────────────────────┘
                     │ CVPixelBuffer
            ┌────────┴────────┐
            ▼                 ▼
    ┌──────────────┐  ┌──────────────┐
    │     CPU      │  │     GPU      │
    │ Downsampling │  │ Downsampling │
    ├──────────────┤  ├──────────────┤
    │ • vImage     │  │ • Metal      │
    │ • CGContext   │  │   Compute    │
    │              │  │ • MPS        │
    └──────┬───────┘  └──────┬───────┘
           │                 │
           ▼                 ▼
    ┌─────────────────────────────────┐
    │      BenchmarkEngine            │
    │  (耗时/FPS/CPU/内存/电量/热状态)  │
    └────────────────┬────────────────┘
                     ▼
    ┌─────────────────────────────────┐
    │         UIKit 界面               │
    │  实时对比 │ 性能仪表盘 │ 设置     │
    └─────────────────────────────────┘
```

## 技术方案

### CPU Downsampling（3 种）

| 方案 | 框架 | 原理 |
|------|------|------|
| **vImage** | Accelerate | 基于 SIMD/NEON 指令集的高性能图像缩放，使用 `vImageScale_ARGB8888` + 高质量重采样 |
| **CGContext** | Core Graphics | 传统 `CGContext.draw(in:)` 双线性插值，作为 CPU 基准对照 |
| **UIGraphicsImageRenderer** | UIKit | UIKit 高层封装，代码最简，适合快速降采样场景 |

### GPU Downsampling（3 种）

| 方案 | 框架 | 原理 |
|------|------|------|
| **Metal Compute** | Metal | 自定义 compute kernel，使用 GPU 硬件纹理采样单元进行双线性插值 |
| **MPS** | MetalPerformanceShaders | Apple 官方 `MPSImageBilinearScale`，GPU 硬件加速最优实现 |
| **Core Image (CILanczos)** | CoreImage | `CILanczosScaleTransform` 滤镜，系统级 GPU 加速，Lanczos 插值质量最高 |

### Metal Shader 核心

```metal
kernel void downsample_bilinear(
    texture2d<float, access::sample> inTexture  [[texture(0)]],
    texture2d<float, access::write>  outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(outTexture.get_width(), outTexture.get_height());
    outTexture.write(inTexture.sample(s, uv), gid);
}
```

## 性能指标

采集维度覆盖 **时间、资源、能耗、稳定性** 四大方面：

| 维度 | 指标 | 采集方式 |
|------|------|----------|
| 时间 | 帧处理耗时 (ms) | `mach_absolute_time` 高精度计时 |
| 时间 | 实时 FPS | 1/平均帧耗时 |
| 时间 | P99 耗时 | 滑动窗口排序取 99 分位 |
| 资源 | CPU 占用率 (%) | `thread_basic_info` 遍历进程线程 |
| 资源 | 内存占用 (MB) | `mach_task_basic_info.resident_size` |
| 资源 | GPU 指令耗时 (ms) | Metal `addCompletedHandler` 差值 |
| 能耗 | 电池电量变化 (%) | `UIDevice.batteryLevel` 周期采样 |
| 能耗 | 消耗速率 (%/min) | 电量差 / 运行时长 |
| 能耗 | 热状态 | `ProcessInfo.thermalState` |
| 稳定性 | 丢帧率 | `captureOutput(_:didDrop:)` 计数 |
| 稳定性 | 帧耗时方差 | 衡量处理抖动 |

## 项目结构

```
DownsamplingBenchmark/
├── AppDelegate.swift                  // UIApplication 生命周期
├── SceneDelegate.swift                // UIWindow + UITabBarController
├── Camera/
│   └── CameraManager.swift            // AVCaptureSession 封装
├── Downsampling/
│   ├── DownsamplerProtocol.swift      // 统一协议 + 数据模型
│   ├── CPU/
│   │   ├── VImageDownsampler.swift        // Accelerate/vImage
│   │   ├── CGContextDownsampler.swift     // Core Graphics
│   │   └── UIGraphicsDownsampler.swift    // UIGraphicsImageRenderer
│   └── GPU/
│       ├── MetalDownsampler.swift         // Metal Compute Shader
│       ├── MPSDownsampler.swift           // Metal Performance Shaders
│       ├── CoreImageDownsampler.swift     // Core Image CILanczos
│       └── Downsampling.metal             // GPU Shader 源码
├── Benchmark/
│   ├── BenchmarkEngine.swift          // 性能统计引擎
│   └── BenchmarkResult.swift          // 指标数据模型
├── ViewControllers/
│   ├── ComparisonViewController.swift // 实时 CPU vs GPU 对比
│   ├── DashboardViewController.swift  // 性能仪表盘
│   └── SettingsViewController.swift   // 算法/分辨率设置
└── Views/
    ├── MetricsOverlayView.swift       // 半透明指标叠加层
    └── LineChartView.swift            // Core Graphics 自绘折线图
```

## 技术栈

- **语言**: Swift 5 / Metal Shading Language
- **UI**: UIKit 纯代码布局（无 Storyboard）
- **摄像头**: AVFoundation (AVCaptureSession + AVCaptureVideoDataOutput)
- **CPU 图像处理**: Accelerate (vImage) / Core Graphics / UIGraphicsImageRenderer
- **GPU 图像处理**: Metal Compute Pipeline / MetalPerformanceShaders / Core Image (CILanczos)
- **性能采集**: mach_task_info / mach_absolute_time / UIDevice Battery API

## 环境要求

- iOS 17.0+
- Xcode 16.0+
- 真机运行（摄像头 + Metal GPU）

## 运行方式

1. 用 Xcode 打开 `DownsamplingBenchmark.xcodeproj`
2. 选择真机设备，配置 Signing Team
3. Build & Run
4. 授权摄像头权限后即可看到实时对比

## 关键知识点

- **AVFoundation 实时采集管线**: CMSampleBuffer → CVPixelBuffer 零拷贝流转
- **Metal Compute Pipeline**: device → library → function → pipelineState → commandBuffer → encoder → dispatch
- **CVMetalTextureCache**: CVPixelBuffer 到 MTLTexture 的高效桥接，避免 CPU-GPU 数据拷贝
- **vImage vs CGContext**: SIMD 加速 vs 通用路径的性能差异
- **MPS vs 自定义 Shader**: Apple 硬件优化内核 vs 手写通用 kernel 的对比
- **iOS 性能监控**: mach_task_info 线程级 CPU 采集、resident_size 内存跟踪、thermalState 降频检测

## 真机实测数据

> 测试设备：iPhone 16 (iPhone18,1) · iOS 26.3.1 · A18 芯片 · 每算法 60 帧
>
> 6 组配置：1080p / 4K × 1/8 / 1/4 / 1/2 缩放

### 冠军矩阵一览

| 配置 | 输出尺寸 | 冠军 | 耗时 | FPS |
|------|---------|------|------|-----|
| 1080p × 1/8 | 240×135 | **MPS** | 0.93 ms | 1077 |
| 1080p × 1/4 | 480×270 | **vImage** | 0.98 ms | 1023 |
| 1080p × 1/2 | 960×540 | **vImage** | 1.04 ms | 964 |
| 4K × 1/8 | 480×270 | **MPS** | 1.32 ms | 759 |
| 4K × 1/4 | 960×540 | **MPS** | 3.27 ms | 306 |
| 4K × 1/2 | 1920×1080 | **Core Image** | 10.99 ms | 91 |

---

### 1080p × 1/8（1920×1080 → 240×135）

| 算法 | 类型 | 平均耗时 | 最小 | 最大 | P99 | FPS | 方差 | CPU 占用 | 峰值内存 |
|------|------|---------|------|------|-----|-----|------|---------|---------|
| **MPS** | GPU | **0.93 ms** | 0.69 | 1.42 | 1.42 | **1077.5** | 0.015 | 50.1% | 220.2 MB |
| vImage | CPU | 1.03 ms | 0.81 | 3.76 | 3.76 | 973.0 | 0.146 | 50.3% | 212.4 MB |
| Metal Compute | GPU | 1.30 ms | 1.00 | 1.67 | 1.67 | 768.9 | 0.015 | 50.1% | 220.2 MB |
| UIGraphicsImageRenderer | CPU | 3.44 ms | 3.07 | 5.89 | 5.89 | 291.1 | 0.144 | 50.2% | 220.4 MB |
| CGContext | CPU | 3.73 ms | 3.19 | 7.02 | 7.02 | 267.8 | 0.239 | 50.3% | 212.4 MB |
| Core Image | GPU | 3.74 ms | 3.13 | 10.55 | 10.55 | 267.4 | 0.875 | 50.1% | 220.3 MB |

### 1080p × 1/4（1920×1080 → 480×270）

| 算法 | 类型 | 平均耗时 | 最小 | 最大 | P99 | FPS | 方差 | CPU 占用 | 峰值内存 |
|------|------|---------|------|------|-----|-----|------|---------|---------|
| **vImage** | CPU | **0.98 ms** | 0.89 | 1.61 | 1.61 | **1023.4** | 0.009 | 44.2% | 202.7 MB |
| MPS | GPU | 1.45 ms | 1.19 | 1.78 | 1.78 | 691.3 | 0.016 | 44.0% | 210.7 MB |
| Metal Compute | GPU | 1.92 ms | 1.54 | 2.33 | 2.33 | 519.6 | 0.031 | 44.1% | 210.7 MB |
| Core Image | GPU | 3.26 ms | 2.93 | 6.79 | 6.79 | 306.7 | 0.231 | 44.0% | 211.2 MB |
| UIGraphicsImageRenderer | CPU | 3.56 ms | 3.40 | 3.94 | 3.94 | 280.6 | 0.016 | 44.1% | 210.7 MB |
| CGContext | CPU | 3.89 ms | 3.56 | 4.25 | 4.25 | 256.8 | 0.023 | 44.1% | 202.7 MB |

### 1080p × 1/2（1920×1080 → 960×540）

| 算法 | 类型 | 平均耗时 | 最小 | 最大 | P99 | FPS | 方差 | CPU 占用 | 峰值内存 |
|------|------|---------|------|------|-----|-----|------|---------|---------|
| **vImage** | CPU | **1.04 ms** | 0.89 | 1.75 | 1.75 | **963.8** | 0.013 | 49.0% | 207.2 MB |
| MPS | GPU | 2.82 ms | 2.40 | 3.63 | 3.63 | 355.0 | 0.076 | 48.9% | 215.2 MB |
| Core Image | GPU | 3.20 ms | 2.99 | 4.19 | 4.19 | 312.4 | 0.021 | 48.9% | 217.2 MB |
| Metal Compute | GPU | 3.30 ms | 2.72 | 4.48 | 4.48 | 302.9 | 0.085 | 48.9% | 215.2 MB |
| UIGraphicsImageRenderer | CPU | 3.75 ms | 3.60 | 4.40 | 4.40 | 266.7 | 0.014 | 48.9% | 215.2 MB |
| CGContext | CPU | 4.08 ms | 3.84 | 4.85 | 4.85 | 245.3 | 0.023 | 49.0% | 207.2 MB |

### 4K × 1/8（3840×2160 → 480×270）

| 算法 | 类型 | 平均耗时 | 最小 | 最大 | P99 | FPS | 方差 | CPU 占用 | 峰值内存 |
|------|------|---------|------|------|-----|-----|------|---------|---------|
| **MPS** | GPU | **1.32 ms** | 0.97 | 2.09 | 2.09 | **758.5** | 0.047 | 87.4% | 637.5 MB |
| Metal Compute | GPU | 2.19 ms | 1.77 | 3.38 | 3.38 | 456.0 | 0.129 | 87.4% | 637.5 MB |
| Core Image | GPU | 8.18 ms | 7.63 | 9.33 | 9.33 | 122.3 | 0.179 | 87.4% | 637.9 MB |
| UIGraphicsImageRenderer | CPU | 8.84 ms | 7.87 | 10.34 | 10.34 | 113.1 | 0.165 | 87.3% | 637.4 MB |
| CGContext | CPU | 9.07 ms | 8.31 | 9.86 | 9.86 | 110.3 | 0.096 | 87.2% | 605.8 MB |
| **vImage** | CPU | **14.31 ms** | 13.02 | 16.19 | 16.19 | 69.9 | 0.633 | 86.9% | 605.8 MB |

### 4K × 1/4（3840×2160 → 960×540）

| 算法 | 类型 | 平均耗时 | 最小 | 最大 | P99 | FPS | 方差 | CPU 占用 | 峰值内存 |
|------|------|---------|------|------|-----|-----|------|---------|---------|
| **MPS** | GPU | **3.27 ms** | 2.80 | 5.41 | 5.41 | **305.8** | 0.168 | 86.9% | 646.2 MB |
| Metal Compute | GPU | 4.34 ms | 3.62 | 5.09 | 5.09 | 230.3 | 0.111 | 86.9% | 646.2 MB |
| Core Image | GPU | 8.93 ms | 8.28 | 10.19 | 10.19 | 112.0 | 0.326 | 86.8% | 648.1 MB |
| UIGraphicsImageRenderer | CPU | 10.81 ms | 9.96 | 12.84 | 12.84 | 92.5 | 0.301 | 86.6% | 646.1 MB |
| CGContext | CPU | 11.05 ms | 10.11 | 14.92 | 14.92 | 90.5 | 0.364 | 86.4% | 614.5 MB |
| vImage | CPU | 12.06 ms | 10.76 | 13.91 | 13.91 | 82.9 | 0.328 | 86.2% | 614.5 MB |

### 4K × 1/2（3840×2160 → 1920×1080）

| 算法 | 类型 | 平均耗时 | 最小 | 最大 | P99 | FPS | 方差 | CPU 占用 | 峰值内存 |
|------|------|---------|------|------|-----|-----|------|---------|---------|
| **Core Image** | GPU | **10.99 ms** | 10.15 | 12.81 | 12.81 | **91.0** | 0.444 | 91.4% | 659.7 MB |
| MPS | GPU | 11.19 ms | 9.34 | 13.23 | 13.23 | 89.4 | 1.056 | 91.5% | 651.8 MB |
| Metal Compute | GPU | 12.04 ms | 10.18 | 14.27 | 14.27 | 83.1 | 1.466 | 91.5% | 651.8 MB |
| vImage | CPU | 12.98 ms | 12.10 | 16.50 | 16.50 | 77.0 | 0.436 | 91.0% | 620.1 MB |
| CGContext | CPU | 16.74 ms | 15.67 | 18.63 | 18.63 | 59.8 | 0.268 | 91.0% | 620.1 MB |
| UIGraphicsImageRenderer | CPU | 17.01 ms | 16.03 | 19.41 | 19.41 | 58.8 | 0.335 | 91.3% | 651.8 MB |

---

### 各算法全配置耗时矩阵（ms）

| 算法 | 1080p ×1/8 | 1080p ×1/4 | 1080p ×1/2 | 4K ×1/8 | 4K ×1/4 | 4K ×1/2 |
|------|-----------|-----------|-----------|---------|---------|---------|
| vImage | 1.03 | **0.98** | **1.04** | 14.31 | 12.06 | 12.98 |
| MPS | **0.93** | 1.45 | 2.82 | **1.32** | **3.27** | 11.19 |
| Metal Compute | 1.30 | 1.92 | 3.30 | 2.19 | 4.34 | 12.04 |
| Core Image | 3.74 | 3.26 | 3.20 | 8.18 | 8.93 | **10.99** |
| UIGraphics | 3.44 | 3.56 | 3.75 | 8.84 | 10.81 | 17.01 |
| CGContext | 3.73 | 3.89 | 4.08 | 9.07 | 11.05 | 16.74 |

### 各算法全配置 FPS 矩阵

| 算法 | 1080p ×1/8 | 1080p ×1/4 | 1080p ×1/2 | 4K ×1/8 | 4K ×1/4 | 4K ×1/2 |
|------|-----------|-----------|-----------|---------|---------|---------|
| vImage | 973 | **1023** | **964** | 70 | 83 | 77 |
| MPS | **1078** | 691 | 355 | **759** | **306** | 89 |
| Metal Compute | 769 | 520 | 303 | 456 | 230 | 83 |
| Core Image | 267 | 307 | 312 | 122 | 112 | **91** |
| UIGraphics | 291 | 281 | 267 | 113 | 93 | 59 |
| CGContext | 268 | 257 | 245 | 110 | 91 | 60 |

### 各算法峰值内存矩阵（MB）

| 算法 | 1080p ×1/8 | 1080p ×1/4 | 1080p ×1/2 | 4K ×1/8 | 4K ×1/4 | 4K ×1/2 |
|------|-----------|-----------|-----------|---------|---------|---------|
| vImage | 212 | **203** | **207** | 606 | **615** | 620 |
| CGContext | 212 | **203** | **207** | **606** | **615** | 620 |
| MPS | 220 | 211 | 215 | 638 | 646 | 652 |
| Metal Compute | 220 | 211 | 215 | 638 | 646 | 652 |
| UIGraphics | 220 | 211 | 215 | 637 | 646 | 652 |
| Core Image | 220 | 211 | 217 | 638 | 648 | 660 |

---

### 数据分析与结论

**1. 没有银弹：冠军随配置切换**

6 组数据揭示了一个关键事实——**没有任何方案在所有场景下都是最优的**。冠军分布如下：

- **vImage** 赢下 1080p × 1/4 和 1080p × 1/2（中低输出 + 小输入）
- **MPS** 赢下 1080p × 1/8、4K × 1/8 和 4K × 1/4（极小/中等输出 + 任意输入）
- **Core Image** 赢下 4K × 1/2（最大输出 + 最大输入）

**2. 核心发现：vImage 性能取决于输入尺寸，GPU 方案取决于输出尺寸**

这是整个 benchmark 最重要的结论。相同输出尺寸 480×270 的对比：

| 方案 | 1080p × 1/4（输入 2M 像素）| 4K × 1/8（输入 8.3M 像素）| 增长 |
|------|--------------------------|--------------------------|------|
| vImage | 0.98 ms | **14.31 ms** | **14.6x** |
| MPS | 1.45 ms | 1.32 ms | **0.91x（几乎不变）** |
| Metal Compute | 1.92 ms | 2.19 ms | 1.14x |

vImage 在输入像素量增长 4 倍时，耗时暴涨 14.6 倍——这是因为 `vImageScale_ARGB8888` 需要遍历所有输入像素进行加权采样，受 CPU 内存带宽瓶颈限制。而 MPS/Metal 利用 GPU 纹理采样单元硬件插值，性能几乎只与输出纹理大小相关。

**3. MPS 的线性扩展规律**

MPS 耗时与输出像素量近似线性关系：

| 输出像素量 | 耗时范围 |
|-----------|---------|
| 3.24 万（240×135） | 0.93 ms |
| 12.96 万（480×270） | 1.32 ~ 1.45 ms |
| 51.84 万（960×540） | 2.82 ~ 3.27 ms |
| 207.36 万（1920×1080） | 11.19 ms |

输出像素量每增长 4 倍，耗时约增长 2~3 倍，符合 GPU 并行计算的特征。

**4. Core Image 的"高启动、低边际"模型**

Core Image 在所有 1080p 配置下耗时几乎恒定（3.20 ~ 3.74 ms），不受缩放比例影响。这是 CIFilter 的 GPU filter graph 编译和调度开销所致。但进入 4K 后，其边际增长极低：

| 配置 | 耗时 | 相对 1080p 增长 |
|------|------|----------------|
| 1080p × 1/2 | 3.20 ms | 基准 |
| 4K × 1/2 | 10.99 ms | 3.4x（输入像素增长 4x） |

对比 MPS 同场景增长 3.97x，Core Image 的延迟求值 + GPU 自动分块在高负载下优势明显。

**5. vImage 在 4K 下从冠军变末位——缓存失效是根因**

vImage 在 1080p 下表现惊艳（0.98~1.04ms），但在 4K 下跌至末位：

| 配置 | vImage 排名 | 耗时 | 冠军差距 |
|------|-----------|------|---------|
| 1080p × 1/4 | **第 1** | 0.98 ms | — |
| 1080p × 1/2 | **第 1** | 1.04 ms | — |
| 4K × 1/8 | **第 6（末位）** | 14.31 ms | MPS 快 10.8x |
| 4K × 1/4 | 第 6（末位） | 12.06 ms | MPS 快 3.7x |
| 4K × 1/2 | 第 4 | 12.98 ms | Core Image 快 1.2x |

4K 帧（3840×2160×4 = 33.2 MB）远超 A18 芯片 L2 缓存容量（16 MB），导致 SIMD 流水线频繁 cache miss，性能急剧下降。1080p 帧（8.3 MB）恰好在缓存范围内，因此 vImage 的 NEON 指令优化得以全速运行。

**6. CPU 占用率：1080p ~44~50%，4K ~87~91%**

| 输入分辨率 | CPU 占用率范围 | 解读 |
|-----------|---------------|------|
| 1080p | 44% ~ 50% | 各方案差异极小，主线程仍有充足余量 |
| 4K | 86% ~ 91% | 接近单核满载，算法间差异被 I/O 瓶颈抹平 |

4K 实时处理已触及 CPU 调度瓶颈。工程优化方向：**异步化 + 双缓冲 + 多线程流水线**。

**7. 内存规律：由输入分辨率决定，与算法和缩放比例无关**

| 输入分辨率 | 内存范围 | 帧缓冲区大小 |
|-----------|---------|------------|
| 1080p | 203 ~ 220 MB | ~8 MB/帧 |
| 4K | 606 ~ 660 MB | ~33 MB/帧 |

同一分辨率下，6 种算法的内存差异不超过 18 MB（GPU 方案略多纹理缓存开销）。缩放比例对内存几乎无影响——因为输入帧缓冲区占了绝对大头。

**8. 稳定性（方差）分析**

| 配置 | 最稳定 | 方差 | 最不稳定 | 方差 |
|------|--------|------|---------|------|
| 1080p × 1/8 | Metal / MPS | 0.015 | Core Image | 0.875 |
| 1080p × 1/4 | vImage | 0.009 | Core Image | 0.231 |
| 1080p × 1/2 | vImage | 0.013 | Metal Compute | 0.085 |
| 4K × 1/8 | MPS | 0.047 | vImage | 0.633 |
| 4K × 1/4 | Metal Compute | 0.111 | CGContext | 0.364 |
| 4K × 1/2 | CGContext | 0.268 | Metal Compute | 1.466 |

Core Image 在 1080p 小输出时方差较大（JIT 编译 + filter graph 抖动），但随负载增大方差趋于收敛。vImage 在 4K 下方差陡增（缓存行为不稳定）。

### 方案选择建议

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 1080p 实时处理（AI 推理等） | **vImage** | 0.98~1.04ms，CPU 方案无 GPU 初始化开销，内存最省 |
| 1080p 极小输出（缩略图等） | **MPS** | 0.93ms（1078 FPS），GPU 硬件极致性能 |
| 4K 实时处理（中小输出） | **MPS** | 1.32~3.27ms，输出像素量线性扩展 |
| 4K 大输出（4K→1080p 等） | **Core Image** | 10.99ms，延迟求值 + 自动分块最适合大负载 |
| 实时 AI 推理前处理 | **Metal Compute** | 可在同一 Shader 合并降采样+归一化+色彩转换 |
| CPU 资源敏感 / 后台省电 | **vImage**（1080p） | CPU 方案之王，1080p 内存仅 203 MB |
| 简单场景 / 快速原型 | **UIGraphicsImageRenderer** | 3 行代码搞定 |
| 最大兼容 / 基准对照 | **CGContext** | 无依赖，方差始终稳定 |

## License

MIT
