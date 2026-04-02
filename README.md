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

> 测试设备：iPhone 16 (iPhone18,1)，iOS 26.3.1，缩放比例 1/4，每算法 60 帧

### 6 种算法全量对比（1080p → 270×480）

| 算法 | 类型 | 平均耗时 | 最小 | 最大 | P99 | FPS | 方差 | CPU 占用 | 峰值内存 |
|------|------|---------|------|------|-----|-----|------|---------|---------|
| **MPS** | GPU | **0.83 ms** | 0.74 | 1.21 | 1.21 | **1202.8** | 0.008 | 79.5% | 1051.5 MB |
| **Metal Compute** | GPU | 1.02 ms | 0.82 | 2.23 | 2.23 | 983.2 | 0.062 | 75.7% | 1050.6 MB |
| **vImage** | CPU | 1.20 ms | 0.63 | 5.15 | 5.15 | 833.8 | 0.940 | **13.4%** | **569.7 MB** |
| Core Image (CILanczos) | GPU | 2.22 ms | 1.12 | 48.94 | 48.94 | 449.6 | 37.305 | 69.2% | 1054.2 MB |
| CGContext | CPU | 2.54 ms | 2.40 | 3.12 | 3.12 | 393.1 | 0.021 | 39.2% | 569.7 MB |
| UIGraphicsImageRenderer | CPU | 2.80 ms | 2.66 | 3.23 | 3.23 | 357.5 | 0.008 | 66.1% | 1049.7 MB |

### 性能排名（按平均耗时）

```
MPS            ████░░░░░░░░░░░░░░░░  0.83 ms  (最快)
Metal Compute  █████░░░░░░░░░░░░░░░  1.02 ms  (1.23x)
vImage         ██████░░░░░░░░░░░░░░  1.20 ms  (1.45x)
Core Image     ███████████░░░░░░░░░  2.22 ms  (2.67x)
CGContext      █████████████░░░░░░░  2.54 ms  (3.06x)
UIGraphics     ██████████████░░░░░░  2.80 ms  (3.37x)
```

### 数据分析与结论

**1. GPU 方案整体领先，MPS 是绝对王者**

MPS 以 0.83ms 的平均耗时（1202 FPS）大幅领先所有方案，这是 Apple 针对自研芯片深度优化的结果。Metal Compute 紧随其后（1.02ms），手写 Shader 性能已接近 MPS 的 81%。

**2. vImage 是最高性价比的 CPU 方案**

vImage 平均仅 1.20ms（833 FPS），逼近 Metal Compute，且 **CPU 占用率仅 13.4%**——是所有方案中最低的。这得益于 Accelerate 框架的 SIMD/NEON 指令优化，在不启用 GPU 的前提下提供了极优性能。

**3. Core Image 首帧 JIT 编译导致严重尾部延迟**

Core Image 的最大耗时高达 48.94ms（方差 37.3），是因为 CIFilter 首次执行时需要 JIT 编译 GPU shader。剔除首帧后实际性能约 1.1ms，但 **P99 不可控**使其不适合严格实时场景。

**4. 内存分水岭：CPU 原生方案省一半内存**

vImage 和 CGContext 峰值内存约 570 MB，而 GPU 方案和 UIGraphicsImageRenderer 约 1050 MB（多出约 480 MB）。GPU 方案需要额外的纹理缓存、command buffer 等资源开销。

**5. CPU 占用率与方案选择的权衡**

| 方案 | CPU 占用率 | 解读 |
|------|-----------|------|
| vImage | 13.4% | CPU 侧 SIMD 高效执行，主线程几乎无影响 |
| CGContext | 39.2% | 中等负载，可接受 |
| UIGraphicsImageRenderer | 66.1% | UIKit 高层封装引入额外开销 |
| Core Image | 69.2% | GPU 调度 + CPU 侧 filter graph 构建 |
| Metal Compute | 75.7% | CPU 侧 command buffer 编码 + 同步等待 |
| MPS | 79.5% | 同 Metal，CPU 侧同步等待 GPU 完成 |

GPU 方案虽然帧处理快，但 `waitUntilCompleted()` 同步等待会占用 CPU 线程。实际项目中应使用异步回调避免阻塞。

**6. 稳定性对比**

CGContext（0.021）和 UIGraphicsImageRenderer（0.008）方差最小，帧间抖动最稳定。vImage 方差较大（0.940）是因为偶发缓存未命中导致个别帧偏慢。MPS 方差极低（0.008），说明硬件加速路径非常稳定。

### 方案选择建议

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 极致性能（实时 30/60fps） | **MPS** | 0.83ms，硬件级最优，iPhone 芯片专属优化 |
| 实时 AI 推理前处理 | **Metal Compute** | 可在同一 Shader 合并降采样+归一化+色彩转换 |
| CPU 资源敏感（后台处理/省电） | **vImage** | 仅 13.4% CPU，内存最省（570 MB），性能接近 GPU |
| 简单场景/快速原型 | **UIGraphicsImageRenderer** | 3 行代码搞定，方差最稳 |
| 通用 GPU 加速（非实时） | **Core Image** | 代码简洁，但需预热避免首帧卡顿 |
| 基准对照/兼容低端设备 | **CGContext** | 无依赖，CPU 占用适中，稳定性优秀 |

## License

MIT
