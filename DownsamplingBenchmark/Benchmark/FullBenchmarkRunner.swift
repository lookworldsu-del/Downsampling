import Foundation
import UIKit
import CoreVideo

struct AlgorithmResult: Codable {
    let name: String
    let type: String
    let avgTime: Double
    let minTime: Double
    let maxTime: Double
    let p99Time: Double
    let fps: Double
    let variance: Double
    let avgCPU: Double
    let peakMemory: Double
    let frameCount: Int
}

struct FullBenchmarkReport: Codable {
    let device: String
    let osVersion: String
    let capturePreset: String
    let scaleFactor: String
    let frameCount: Int
    let date: String
    let results: [AlgorithmResult]
}

final class FullBenchmarkRunner {

    private let allDownsamplers: [Downsampler]
    private let framesToCollect: Int
    private let scaleFactor: Float

    private var frameBuffers: [CVPixelBuffer] = []
    private var isCollecting = false
    private(set) var isRunning = false
    private(set) var progress: String = ""

    var onProgress: ((String) -> Void)?
    var onComplete: ((FullBenchmarkReport) -> Void)?

    init(scaleFactor: Float, framesToCollect: Int = 60) {
        self.scaleFactor = scaleFactor
        self.framesToCollect = framesToCollect

        var downsamplers: [Downsampler] = [
            VImageDownsampler(),
            CGContextDownsampler(),
            UIGraphicsDownsampler(),
        ]
        if let metal = MetalDownsampler() { downsamplers.append(metal) }
        if let mps = MPSDownsampler() { downsamplers.append(mps) }
        downsamplers.append(CoreImageDownsampler())

        self.allDownsamplers = downsamplers
    }

    func collectFrame(_ pixelBuffer: CVPixelBuffer) {
        guard isCollecting, frameBuffers.count < framesToCollect else { return }

        var copy: CVPixelBuffer?
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &copy)

        if let dst = copy {
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            CVPixelBufferLockBaseAddress(dst, [])
            if let srcPtr = CVPixelBufferGetBaseAddress(pixelBuffer),
               let dstPtr = CVPixelBufferGetBaseAddress(dst) {
                let bytes = CVPixelBufferGetBytesPerRow(pixelBuffer) * height
                memcpy(dstPtr, srcPtr, bytes)
            }
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            frameBuffers.append(dst)
        }

        let count = frameBuffers.count
        DispatchQueue.main.async { [weak self] in
            self?.onProgress?("采集帧: \(count)/\(self?.framesToCollect ?? 0)")
        }

        if frameBuffers.count >= framesToCollect {
            isCollecting = false
            runBenchmark()
        }
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true
        frameBuffers.removeAll()
        isCollecting = true
        DispatchQueue.main.async { [weak self] in
            self?.onProgress?("采集帧中...")
        }
    }

    private func runBenchmark() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }

            var results: [AlgorithmResult] = []

            for (index, ds) in self.allDownsamplers.enumerated() {
                DispatchQueue.main.async {
                    self.onProgress?("测试中 (\(index + 1)/\(self.allDownsamplers.count)): \(ds.name)")
                }

                var times: [Double] = []
                var cpuUsages: [Double] = []
                var peakMem: UInt64 = 0

                for buffer in self.frameBuffers {
                    let cpuBefore = BenchmarkEngine.currentCPUUsage()
                    let output = ds.downsample(buffer, scaleFactor: self.scaleFactor)
                    let cpuAfter = BenchmarkEngine.currentCPUUsage()

                    times.append(output.processingTime)
                    cpuUsages.append((cpuBefore + cpuAfter) / 2.0)

                    let mem = BenchmarkEngine.currentMemoryUsage()
                    if mem > peakMem { peakMem = mem }
                }

                let sorted = times.sorted()
                let avg = times.reduce(0, +) / Double(times.count)
                let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count)
                let p99Index = min(Int(Double(sorted.count) * 0.99), sorted.count - 1)

                let result = AlgorithmResult(
                    name: ds.name,
                    type: ds.type.rawValue,
                    avgTime: avg * 1000,
                    minTime: (sorted.first ?? 0) * 1000,
                    maxTime: (sorted.last ?? 0) * 1000,
                    p99Time: sorted[p99Index] * 1000,
                    fps: avg > 0 ? 1.0 / avg : 0,
                    variance: variance * 1_000_000,
                    avgCPU: cpuUsages.reduce(0, +) / Double(cpuUsages.count),
                    peakMemory: Double(peakMem) / 1_048_576,
                    frameCount: times.count
                )
                results.append(result)
            }

            let formatter = DateFormatter()
            formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"

            let report = FullBenchmarkReport(
                device: Self.deviceName(),
                osVersion: UIDevice.current.systemVersion,
                capturePreset: self.frameBuffers.isEmpty ? "unknown" :
                    "\(CVPixelBufferGetWidth(self.frameBuffers[0]))x\(CVPixelBufferGetHeight(self.frameBuffers[0]))",
                scaleFactor: String(format: "%.3f", self.scaleFactor),
                frameCount: self.framesToCollect,
                date: formatter.string(from: Date()),
                results: results
            )

            self.frameBuffers.removeAll()
            self.isRunning = false

            DispatchQueue.main.async {
                self.onComplete?(report)
            }
        }
    }

    static func reportToJSON(_ report: FullBenchmarkReport) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(report) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private static func deviceName() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machine = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0) ?? "unknown"
            }
        }
        return "\(machine) (\(UIDevice.current.name))"
    }
}
