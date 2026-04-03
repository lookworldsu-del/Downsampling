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
    let peakThermalState: Int
    let thermalBefore: Int
    let thermalAfter: Int
    let batteryBefore: Double
    let batteryAfter: Double
    let durationSeconds: Double
}

struct ThermalSnapshot: Codable {
    let frameIndex: Int
    let thermalState: Int
    let batteryLevel: Double
    let timestamp: Double
    let algorithmIndex: Int
}

struct FullBenchmarkReport: Codable {
    let device: String
    let osVersion: String
    let capturePreset: String
    let scaleFactor: String
    let frameCount: Int
    let date: String
    let isolatedMode: Bool
    let cooldownSeconds: Double
    let results: [AlgorithmResult]

    let thermalStateBefore: Int
    let thermalStateAfter: Int
    let peakThermalState: Int
    let batteryBefore: Double
    let batteryAfter: Double
    let batteryDrainPercent: Double
    let durationSeconds: Double
    let estimatedPowerMW: Double
    let cpuEnergyScore: Double
    let thermalTimeline: [ThermalSnapshot]
}

// MARK: - Isolated Benchmark Runner

final class FullBenchmarkRunner {

    private enum Phase {
        case idle
        case running(algorithmIndex: Int)
        case cooling(nextAlgorithmIndex: Int, startTime: TimeInterval)
        case done
    }

    private let allDownsamplers: [Downsampler]
    private let framesToCollect: Int
    private let target: DownsampleTarget
    private let cooldownDuration: TimeInterval

    private var phase: Phase = .idle
    private var captureSize: String = "unknown"

    // Per-algorithm isolated data
    private var currentFrameCount: Int = 0
    private var currentTimings: [Double] = []
    private var currentCPUUsages: [Double] = []
    private var currentPeakMemory: UInt64 = 0
    private var currentThermalPeak: Int = 0
    private var algorithmStartTime: TimeInterval = 0
    private var algorithmStartBattery: Float = 0
    private var algorithmStartThermal: Int = 0

    // Accumulated results
    private var completedResults: [AlgorithmResult] = []

    // Global tracking
    private var benchmarkStartTime: TimeInterval = 0
    private var globalStartBattery: Float = -1
    private var globalStartThermal: Int = 0
    private var thermalTimeline: [ThermalSnapshot] = []
    private var lastSampledThermal: Int = -1
    private var totalCPUSeconds: Double = 0

    private(set) var isRunning = false

    var onProgress: ((String) -> Void)?
    var onComplete: ((FullBenchmarkReport) -> Void)?

    init(target: DownsampleTarget, framesToCollect: Int = 60, cooldownSeconds: TimeInterval = 3.0) {
        self.target = target
        self.framesToCollect = framesToCollect
        self.cooldownDuration = cooldownSeconds

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

        UIDevice.current.isBatteryMonitoringEnabled = true
        benchmarkStartTime = CACurrentMediaTime()
        globalStartBattery = UIDevice.current.batteryLevel
        globalStartThermal = ProcessInfo.processInfo.thermalState.rawValue
        thermalTimeline = []
        lastSampledThermal = -1
        completedResults = []
        totalCPUSeconds = 0
        captureSize = "unknown"

        beginAlgorithm(index: 0)
    }

    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard isRunning else { return }

        if captureSize == "unknown" {
            captureSize = "\(CVPixelBufferGetWidth(pixelBuffer))x\(CVPixelBufferGetHeight(pixelBuffer))"
        }

        switch phase {
        case .idle, .done:
            return

        case .cooling(let nextIdx, let startTime):
            let elapsed = CACurrentMediaTime() - startTime
            if elapsed >= cooldownDuration {
                beginAlgorithm(index: nextIdx)
                runFrame(algorithmIndex: nextIdx, pixelBuffer: pixelBuffer)
            } else {
                let remaining = ceil(cooldownDuration - elapsed)
                let nextName = allDownsamplers[nextIdx].name
                let total = allDownsamplers.count
                DispatchQueue.main.async { [weak self] in
                    self?.onProgress?("⏸ 冷却中 \(Int(remaining))s → [\(nextIdx + 1)/\(total)] \(nextName)")
                }
            }

        case .running(let algIdx):
            runFrame(algorithmIndex: algIdx, pixelBuffer: pixelBuffer)
        }
    }

    // MARK: - Algorithm Lifecycle

    private func beginAlgorithm(index: Int) {
        currentFrameCount = 0
        currentTimings = []
        currentCPUUsages = []
        currentPeakMemory = 0
        currentThermalPeak = 0
        algorithmStartTime = CACurrentMediaTime()
        algorithmStartBattery = UIDevice.current.batteryLevel
        algorithmStartThermal = ProcessInfo.processInfo.thermalState.rawValue

        phase = .running(algorithmIndex: index)

        let name = allDownsamplers[index].name
        let total = allDownsamplers.count
        DispatchQueue.main.async { [weak self] in
            self?.onProgress?("▶ [\(index + 1)/\(total)] \(name) - 0/\(self?.framesToCollect ?? 0)")
        }
    }

    private func runFrame(algorithmIndex algIdx: Int, pixelBuffer: CVPixelBuffer) {
        guard currentFrameCount < framesToCollect else { return }

        let ds = allDownsamplers[algIdx]

        // Sample thermal state
        let thermal = ProcessInfo.processInfo.thermalState.rawValue
        if thermal != lastSampledThermal || thermalTimeline.isEmpty {
            thermalTimeline.append(ThermalSnapshot(
                frameIndex: currentFrameCount,
                thermalState: thermal,
                batteryLevel: Double(UIDevice.current.batteryLevel),
                timestamp: CACurrentMediaTime() - benchmarkStartTime,
                algorithmIndex: algIdx
            ))
            lastSampledThermal = thermal
        }

        let cpuBefore = BenchmarkEngine.currentCPUUsage()
        let output = ds.downsample(pixelBuffer, target: target)
        let cpuAfter = BenchmarkEngine.currentCPUUsage()

        let timeMs = output.processingTime * 1000
        let avgCPU = (cpuBefore + cpuAfter) / 2.0
        currentTimings.append(timeMs)
        currentCPUUsages.append(avgCPU)
        totalCPUSeconds += (avgCPU / 100.0) * output.processingTime

        let mem = BenchmarkEngine.currentMemoryUsage()
        if mem > currentPeakMemory { currentPeakMemory = mem }
        if thermal > currentThermalPeak { currentThermalPeak = thermal }

        currentFrameCount += 1

        let name = ds.name
        let total = allDownsamplers.count
        let frame = currentFrameCount
        let target = framesToCollect
        DispatchQueue.main.async { [weak self] in
            self?.onProgress?("▶ [\(algIdx + 1)/\(total)] \(name) - \(frame)/\(target)")
        }

        if currentFrameCount >= framesToCollect {
            finalizeAlgorithm(index: algIdx)
        }
    }

    private func finalizeAlgorithm(index: Int) {
        let ds = allDownsamplers[index]
        let now = CACurrentMediaTime()

        let sorted = currentTimings.sorted()
        let avg = currentTimings.reduce(0, +) / Double(currentTimings.count)
        let variance = currentTimings.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(currentTimings.count)
        let p99Index = min(Int(Double(sorted.count) * 0.99), sorted.count - 1)
        let avgCPU = currentCPUUsages.reduce(0, +) / Double(currentCPUUsages.count)

        let endBattery = UIDevice.current.batteryLevel
        let endThermal = ProcessInfo.processInfo.thermalState.rawValue

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
            peakMemory: Double(currentPeakMemory) / 1_048_576,
            frameCount: currentTimings.count,
            peakThermalState: currentThermalPeak,
            thermalBefore: algorithmStartThermal,
            thermalAfter: endThermal,
            batteryBefore: Double(algorithmStartBattery) * 100,
            batteryAfter: Double(endBattery) * 100,
            durationSeconds: now - algorithmStartTime
        )
        completedResults.append(result)

        let nextIdx = index + 1
        if nextIdx < allDownsamplers.count {
            phase = .cooling(nextAlgorithmIndex: nextIdx, startTime: CACurrentMediaTime())
            let nextName = allDownsamplers[nextIdx].name
            let total = allDownsamplers.count
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.onProgress?("✅ \(ds.name) 完成 → 冷却 \(Int(self.cooldownDuration))s → [\(nextIdx + 1)/\(total)] \(nextName)")
            }
        } else {
            finalizeAll()
        }
    }

    // MARK: - Final Report

    private func finalizeAll() {
        phase = .done
        isRunning = false

        let endTime = CACurrentMediaTime()
        let duration = endTime - benchmarkStartTime
        let batteryAtEnd = UIDevice.current.batteryLevel
        let thermalAtEnd = ProcessInfo.processInfo.thermalState.rawValue

        thermalTimeline.append(ThermalSnapshot(
            frameIndex: 0, thermalState: thermalAtEnd,
            batteryLevel: Double(batteryAtEnd),
            timestamp: duration, algorithmIndex: -1
        ))

        let batteryDrain = max(0, Double(globalStartBattery - batteryAtEnd)) * 100.0
        let batteryCapacityWh = 13.85
        let powerMW: Double
        if duration > 0 && batteryDrain > 0 {
            powerMW = (batteryDrain / 100.0) * batteryCapacityWh * 1000.0 / (duration / 3600.0)
        } else {
            powerMW = 0
        }

        let peakThermal = thermalTimeline.map { $0.thermalState }.max() ?? globalStartThermal

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"

        let report = FullBenchmarkReport(
            device: Self.deviceName(),
            osVersion: UIDevice.current.systemVersion,
            capturePreset: captureSize,
            scaleFactor: target.displayName,
            frameCount: framesToCollect,
            date: formatter.string(from: Date()),
            isolatedMode: true,
            cooldownSeconds: cooldownDuration,
            results: completedResults,
            thermalStateBefore: globalStartThermal,
            thermalStateAfter: thermalAtEnd,
            peakThermalState: peakThermal,
            batteryBefore: Double(globalStartBattery) * 100,
            batteryAfter: Double(batteryAtEnd) * 100,
            batteryDrainPercent: batteryDrain,
            durationSeconds: duration,
            estimatedPowerMW: powerMW,
            cpuEnergyScore: totalCPUSeconds,
            thermalTimeline: thermalTimeline
        )

        completedResults.removeAll()

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
