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
    private let target: DownsampleTarget

    private var collectedFrames: Int = 0
    private var timings: [[Double]] = []
    private var cpuUsages: [[Double]] = []
    private var peakMemories: [UInt64] = []

    private(set) var isRunning = false
    private var captureSize: String = "unknown"

    var onProgress: ((String) -> Void)?
    var onComplete: ((FullBenchmarkReport) -> Void)?

    init(target: DownsampleTarget, framesToCollect: Int = 60) {
        self.target = target
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

    func start() {
        guard !isRunning else { return }
        isRunning = true
        collectedFrames = 0
        captureSize = "unknown"

        let count = allDownsamplers.count
        timings = Array(repeating: [], count: count)
        cpuUsages = Array(repeating: [], count: count)
        peakMemories = Array(repeating: 0, count: count)

        DispatchQueue.main.async { [weak self] in
            self?.onProgress?("跑分中: 0/\(self?.framesToCollect ?? 0)")
        }
    }

    /// Process one frame through all algorithms immediately — no pixel buffer storage.
    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard isRunning, collectedFrames < framesToCollect else { return }

        if captureSize == "unknown" {
            let w = CVPixelBufferGetWidth(pixelBuffer)
            let h = CVPixelBufferGetHeight(pixelBuffer)
            captureSize = "\(w)x\(h)"
        }

        for (i, ds) in allDownsamplers.enumerated() {
            let cpuBefore = BenchmarkEngine.currentCPUUsage()
            let output = ds.downsample(pixelBuffer, target: target)
            let cpuAfter = BenchmarkEngine.currentCPUUsage()

            timings[i].append(output.processingTime * 1000)
            cpuUsages[i].append((cpuBefore + cpuAfter) / 2.0)

            let mem = BenchmarkEngine.currentMemoryUsage()
            if mem > peakMemories[i] { peakMemories[i] = mem }
        }

        collectedFrames += 1

        let current = collectedFrames
        let total = framesToCollect
        DispatchQueue.main.async { [weak self] in
            self?.onProgress?("跑分中: \(current)/\(total)")
        }

        if collectedFrames >= framesToCollect {
            finalize()
        }
    }

    private func finalize() {
        isRunning = false

        var results: [AlgorithmResult] = []

        for (i, ds) in allDownsamplers.enumerated() {
            let times = timings[i]
            guard !times.isEmpty else { continue }

            let sorted = times.sorted()
            let avg = times.reduce(0, +) / Double(times.count)
            let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count)
            let p99Index = min(Int(Double(sorted.count) * 0.99), sorted.count - 1)
            let avgCPU = cpuUsages[i].reduce(0, +) / Double(cpuUsages[i].count)

            let result = AlgorithmResult(
                name: ds.name,
                type: ds.type.rawValue,
                avgTime: avg,
                minTime: sorted.first ?? 0,
                maxTime: sorted.last ?? 0,
                p99Time: sorted[p99Index],
                fps: avg > 0 ? 1000.0 / avg : 0,
                variance: variance,
                avgCPU: avgCPU,
                peakMemory: Double(peakMemories[i]) / 1_048_576,
                frameCount: times.count
            )
            results.append(result)
        }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"

        let report = FullBenchmarkReport(
            device: Self.deviceName(),
            osVersion: UIDevice.current.systemVersion,
            capturePreset: captureSize,
            scaleFactor: target.displayName,
            frameCount: framesToCollect,
            date: formatter.string(from: Date()),
            results: results
        )

        timings.removeAll()
        cpuUsages.removeAll()
        peakMemories.removeAll()

        DispatchQueue.main.async { [weak self] in
            self?.onComplete?(report)
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
