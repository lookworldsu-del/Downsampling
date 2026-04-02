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

> 测试设备：iPhone，缩放比例 1/4，算法组合 UIGraphicsImageRenderer (CPU) vs Core Image CILanczos (GPU)

### 1080p 采集（1920×1080 → 480×270）

| 指标 | CPU (UIGraphicsImageRenderer) | GPU (Core Image CILanczos) |
|------|------|------|
| 平均耗时 | 3.99 ms | 3.49 ms |
| 最小耗时 | 3.74 ms | 3.19 ms |
| 最大耗时 | 4.89 ms | 4.07 ms |
| P99 耗时 | 4.34 ms | 4.07 ms |
| FPS | 250.8 | 286.6 |
| 方差 | 0.0231 | 0.0411 |
| CPU 占用率 | 40.3% | 40.3% |
| 内存占用 | 145.2 MB | 145.2 MB |
| 丢帧率 | 0.00% | 0.00% |

### 4K 采集（3840×2160 → 960×540）

| 指标 | CPU (UIGraphicsImageRenderer) | GPU (Core Image CILanczos) |
|------|------|------|
| 平均耗时 | 11.07 ms | 9.22 ms |
| 最小耗时 | 10.40 ms | 6.06 ms |
| 最大耗时 | 15.51 ms | 13.81 ms |
| P99 耗时 | 11.86 ms | 12.39 ms |
| FPS | 90.3 | 108.4 |
| 方差 | 0.1961 | 0.7160 |
| CPU 占用率 | 63.9% | 63.9% |
| 内存占用 | 152.1 MB | 152.1 MB |
| 丢帧率 | 0.00% | 0.00% |

### 1080p vs 4K 倍率变化

| 指标 | CPU 4K/1080p | GPU 4K/1080p |
|------|-------------|-------------|
| 平均耗时 | 2.77x | 2.64x |
| FPS | 0.36x | 0.38x |
| CPU 占用率 | 1.59x | 1.59x |
| 内存增长 | +6.9 MB | +6.9 MB |

### 数据分析与结论

1. **GPU 始终优于 CPU**：1080p 下 GPU 快 14%（3.49 vs 3.99 ms），4K 下快 17%（9.22 vs 11.07 ms）。分辨率越高，GPU 加速优势越明显。

2. **4K 像素量是 1080p 的 4 倍**（3840×2160 vs 1920×1080），但耗时仅增长约 2.7 倍，说明两种方案都有良好的缓存局部性和流水线利用率。

3. **CPU 占用率从 40% 飙升到 64%**：4K 下系统负载显著增大。在实际项目中，如果需要同时运行 AI 推理等 CPU 密集任务，GPU 降采样能有效卸载 CPU 压力。

4. **稳定性差异**：1080p 下两者方差都很小，表现稳定；4K 下 GPU 方差（0.716）明显高于 CPU（0.196），说明 GPU 调度存在偶发延迟波动，但不影响整体吞吐。

5. **内存表现优秀**：1080p 到 4K 仅增加约 7 MB，两种方案都避免了大量临时内存分配。

6. **零丢帧**：无论 1080p 还是 4K，丢帧率均为 0.00%，说明处理速度始终能跟上摄像头采集帧率。

### 方案选择建议

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 实时预览降采样 | Core Image (CILanczos) | GPU 加速，CPU 占用低，代码简洁 |
| 实时 AI 推理前处理 | Metal Compute Shader | 可合并降采样+归一化+色彩转换，零拷贝极致性能 |
| 批量图片缩略图 | vImage (Accelerate) | SIMD 优化，无 GPU 初始化开销 |
| 简单场景/快速原型 | UIGraphicsImageRenderer | 代码量最少，兼容性最好 |
| 硬件级最优缩放 | MPS (MPSImageBilinearScale) | Apple 芯片专属优化，性能天花板 |

## License

MIT
