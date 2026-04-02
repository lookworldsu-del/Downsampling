# DownsamplingBenchmark

iOS 实时摄像头采集场景下，CPU vs GPU Downsampling 性能对比工具。

## 项目背景

在移动端图像处理中，Downsampling（降采样）是最常见的操作之一——从相机预览、缩略图生成到视频流预处理都会用到。本项目在 **真实的实时摄像头采集场景** 下，分别实现了 CPU 和 GPU 两大类共 4 种 Downsampling 方案，并通过全面的性能指标进行实时对比。

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

### CPU Downsampling

| 方案 | 框架 | 原理 |
|------|------|------|
| **vImage** | Accelerate | 基于 SIMD/NEON 指令集的高性能图像缩放，使用 `vImageScale_ARGB8888` + 高质量重采样 |
| **CGContext** | Core Graphics | 传统 `CGContext.draw(in:)` 双线性插值，作为 CPU 基准对照 |

### GPU Downsampling

| 方案 | 框架 | 原理 |
|------|------|------|
| **Metal Compute** | Metal | 自定义 compute kernel，使用 GPU 硬件纹理采样单元进行双线性插值 |
| **MPS** | MetalPerformanceShaders | Apple 官方 `MPSImageBilinearScale`，GPU 硬件加速最优实现 |

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
│   │   ├── VImageDownsampler.swift    // Accelerate/vImage
│   │   └── CGContextDownsampler.swift // Core Graphics
│   └── GPU/
│       ├── MetalDownsampler.swift     // Metal Compute Shader
│       ├── MPSDownsampler.swift       // Metal Performance Shaders
│       └── Downsampling.metal         // GPU Shader 源码
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
- **CPU 图像处理**: Accelerate (vImage) / Core Graphics
- **GPU 图像处理**: Metal Compute Pipeline / MetalPerformanceShaders
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

## 性能对比数据

> 以下为预期参考值，实际数据请在真机上运行后截图补充。

| 方案 | 1080p→270p 平均耗时 | FPS |
|------|---------------------|-----|
| vImage (CPU) | ~2-4 ms | ~250-500 |
| CGContext (CPU) | ~4-8 ms | ~125-250 |
| Metal Compute (GPU) | ~0.5-1 ms | ~1000+ |
| MPS (GPU) | ~0.3-0.8 ms | ~1200+ |

## License

MIT
