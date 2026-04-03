import Foundation
import UIKit
import CoreVideo
import QuartzCore

// MARK: - Data Models

struct ConfigurationResult: Codable {
    let capturePreset: String
    let scaleFactor: String
    let frameCount: Int
    let results: [AlgorithmResult]
    let thermalStateBefore: Int
    let thermalStateAfter: Int
    let peakThermalState: Int
    let batteryBefore: Double
    let batteryAfter: Double
    let durationSeconds: Double
}

struct ComprehensiveReport: Codable {
    let device: String
    let osVersion: String
    let date: String
    let isolatedMode: Bool
    let cooldownSeconds: Double
    let presetCooldownSeconds: Double
    let framesPerAlgorithm: Int
    let totalAlgorithmRuns: Int
    let totalDurationSeconds: Double
    let batteryBefore: Double
    let batteryAfter: Double
    let batteryDrainPercent: Double
    let estimatedPowerMW: Double
    let cpuEnergyScore: Double
    let thermalStateBefore: Int
    let thermalStateAfter: Int
    let peakThermalState: Int
    let thermalTimeline: [ThermalSnapshot]
    let configurations: [ConfigurationResult]
}

// MARK: - Runner

final class ComprehensiveBenchmarkRunner {

    struct BenchmarkConfig: Equatable {
        let preset: CameraManager.Preset
        let target: DownsampleTarget
    }

    private enum Phase {
        case idle
        case waitingForPresetSwitch
        case presetCooldown(startTime: TimeInterval)
        case running(algorithmIndex: Int)
        case cooling(nextAlgorithmIndex: Int, startTime: TimeInterval)
        case done
    }

    private let allConfigs: [BenchmarkConfig]
    private let allDownsamplers: [Downsampler]
    private let framesToCollect: Int
    private let cooldownDuration: TimeInterval
    private let presetCooldownDuration: TimeInterval

    private var phase: Phase = .idle
    private var currentConfigIndex: Int = 0
    private var captureSize: String = "unknown"

    private var currentFrameCount: Int = 0
    private var currentTimings: [Double] = []
    private var currentCPUUsages: [Double] = []
    private var currentPeakMemory: UInt64 = 0
    private var currentThermalPeak: Int = 0
    private var algorithmStartTime: TimeInterval = 0
    private var algorithmStartBattery: Float = 0
    private var algorithmStartThermal: Int = 0

    private var configAlgorithmResults: [AlgorithmResult] = []
    private var configStartThermal: Int = 0
    private var configStartBattery: Float = 0
    private var configStartTime: TimeInterval = 0
    private var allConfigResults: [ConfigurationResult] = []

    private var benchmarkStartTime: TimeInterval = 0
    private var globalStartBattery: Float = -1
    private var globalStartThermal: Int = 0
    private var thermalTimeline: [ThermalSnapshot] = []
    private var lastSampledThermal: Int = -1
    private var totalCPUSeconds: Double = 0

    private(set) var isRunning = false

    var onProgress: ((String) -> Void)?
    var onComplete: ((ComprehensiveReport) -> Void)?
    var onSwitchPreset: ((CameraManager.Preset, @escaping () -> Void) -> Void)?

    init(framesToCollect: Int = 60,
         cooldownSeconds: TimeInterval = 5.0,
         presetCooldownSeconds: TimeInterval = 8.0) {
        self.framesToCollect = framesToCollect
        self.cooldownDuration = cooldownSeconds
        self.presetCooldownDuration = presetCooldownSeconds

        let presets: [CameraManager.Preset] = [.hd1080p, .uhd4K]
        let targets = DownsampleTarget.allPresets

        var configs: [BenchmarkConfig] = []
        for preset in presets {
            for target in targets {
                configs.append(BenchmarkConfig(preset: preset, target: target))
            }
        }
        self.allConfigs = configs

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

    var totalRuns: Int { allConfigs.count * allDownsamplers.count }

    var completedRuns: Int {
        allConfigResults.count * allDownsamplers.count + configAlgorithmResults.count
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
        allConfigResults = []
        totalCPUSeconds = 0
        currentConfigIndex = 0

        beginConfig(index: 0)
    }

    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard isRunning else { return }

        if captureSize == "unknown" {
            captureSize = "\(CVPixelBufferGetWidth(pixelBuffer))x\(CVPixelBufferGetHeight(pixelBuffer))"
        }

        switch phase {
        case .idle, .done, .waitingForPresetSwitch:
            return

        case .presetCooldown(let startTime):
            let elapsed = CACurrentMediaTime() - startTime
            if elapsed >= presetCooldownDuration {
                captureSize = "\(CVPixelBufferGetWidth(pixelBuffer))x\(CVPixelBufferGetHeight(pixelBuffer))"
                startConfigAlgorithms()
            } else {
                let remaining = Int(ceil(presetCooldownDuration - elapsed))
                let cfg = allConfigs[currentConfigIndex]
                let progress = configProgressPrefix()
                DispatchQueue.main.async { [weak self] in
                    self?.onProgress?("\(progress) ⏸ 分辨率切换冷却 \(remaining)s → \(cfg.preset.displayName) \(cfg.target.displayName)")
                }
            }

        case .cooling(let nextIdx, let startTime):
            let elapsed = CACurrentMediaTime() - startTime
            if elapsed >= cooldownDuration {
                beginAlgorithm(index: nextIdx)
                runFrame(algorithmIndex: nextIdx, pixelBuffer: pixelBuffer)
            } else {
                let remaining = Int(ceil(cooldownDuration - elapsed))
                let nextName = allDownsamplers[nextIdx].name
                let algoTotal = allDownsamplers.count
                let progress = configProgressPrefix()
                DispatchQueue.main.async { [weak self] in
                    self?.onProgress?("\(progress) ⏸ 冷却 \(remaining)s → [\(nextIdx + 1)/\(algoTotal)] \(nextName)")
                }
            }

        case .running(let algIdx):
            runFrame(algorithmIndex: algIdx, pixelBuffer: pixelBuffer)
        }
    }

    // MARK: - Configuration Lifecycle

    private func configProgressPrefix() -> String {
        let cfgNum = currentConfigIndex + 1
        let cfgTotal = allConfigs.count
        let done = completedRuns
        let total = totalRuns
        return "[\(cfgNum)/\(cfgTotal)] (\(done)/\(total))"
    }

    private func beginConfig(index: Int) {
        currentConfigIndex = index
        let config = allConfigs[index]

        configAlgorithmResults = []
        configStartThermal = ProcessInfo.processInfo.thermalState.rawValue
        configStartBattery = UIDevice.current.batteryLevel
        configStartTime = CACurrentMediaTime()
        captureSize = "unknown"

        let needSwitch: Bool
        if index == 0 {
            needSwitch = true
        } else {
            needSwitch = allConfigs[index - 1].preset != config.preset
        }

        if needSwitch {
            phase = .waitingForPresetSwitch
            let progress = configProgressPrefix()
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.onProgress?("\(progress) 🔄 切换至 \(config.preset.displayName)...")
                self.onSwitchPreset?(config.preset) { [weak self] in
                    guard let self else { return }
                    if index > 0 {
                        self.phase = .presetCooldown(startTime: CACurrentMediaTime())
                    } else {
                        self.startConfigAlgorithms()
                    }
                }
            }
        } else {
            startConfigAlgorithms()
        }
    }

    private func startConfigAlgorithms() {
        beginAlgorithm(index: 0)
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
        let algoTotal = allDownsamplers.count
        let config = allConfigs[currentConfigIndex]
        let progress = configProgressPrefix()
        DispatchQueue.main.async { [weak self] in
            self?.onProgress?("\(progress) ▶ \(config.preset.displayName) \(config.target.displayName) — [\(index + 1)/\(algoTotal)] \(name) 0/\(self?.framesToCollect ?? 0)")
        }
    }

    private func runFrame(algorithmIndex algIdx: Int, pixelBuffer: CVPixelBuffer) {
        guard currentFrameCount < framesToCollect else { return }

        let ds = allDownsamplers[algIdx]
        let config = allConfigs[currentConfigIndex]

        let thermal = ProcessInfo.processInfo.thermalState.rawValue
        if thermal != lastSampledThermal || thermalTimeline.isEmpty {
            thermalTimeline.append(ThermalSnapshot(
                frameIndex: currentFrameCount,
                thermalState: thermal,
                batteryLevel: Double(UIDevice.current.batteryLevel),
                timestamp: CACurrentMediaTime() - benchmarkStartTime,
                algorithmIndex: completedRuns + algIdx
            ))
            lastSampledThermal = thermal
        }

        let cpuBefore = BenchmarkEngine.currentCPUUsage()
        let output = ds.downsample(pixelBuffer, target: config.target)
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
        let algoTotal = allDownsamplers.count
        let frame = currentFrameCount
        let target = framesToCollect
        let progress = configProgressPrefix()
        DispatchQueue.main.async { [weak self] in
            self?.onProgress?("\(progress) ▶ \(config.preset.displayName) \(config.target.displayName) — [\(algIdx + 1)/\(algoTotal)] \(name) \(frame)/\(target)")
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
        configAlgorithmResults.append(result)

        let nextIdx = index + 1
        if nextIdx < allDownsamplers.count {
            phase = .cooling(nextAlgorithmIndex: nextIdx, startTime: CACurrentMediaTime())
        } else {
            finalizeConfig()
        }
    }

    private func finalizeConfig() {
        let config = allConfigs[currentConfigIndex]
        let endThermal = ProcessInfo.processInfo.thermalState.rawValue
        let endBattery = UIDevice.current.batteryLevel
        let now = CACurrentMediaTime()

        let peakT = configAlgorithmResults.map { $0.peakThermalState }.max() ?? configStartThermal

        let configResult = ConfigurationResult(
            capturePreset: captureSize,
            scaleFactor: config.target.displayName,
            frameCount: framesToCollect,
            results: configAlgorithmResults,
            thermalStateBefore: configStartThermal,
            thermalStateAfter: endThermal,
            peakThermalState: peakT,
            batteryBefore: Double(configStartBattery) * 100,
            batteryAfter: Double(endBattery) * 100,
            durationSeconds: now - configStartTime
        )
        allConfigResults.append(configResult)

        let nextCfg = currentConfigIndex + 1
        if nextCfg < allConfigs.count {
            let needPresetSwitch = allConfigs[nextCfg].preset != config.preset
            if needPresetSwitch {
                beginConfig(index: nextCfg)
            } else {
                phase = .cooling(nextAlgorithmIndex: 0, startTime: CACurrentMediaTime())
                currentConfigIndex = nextCfg
                configAlgorithmResults = []
                configStartThermal = ProcessInfo.processInfo.thermalState.rawValue
                configStartBattery = UIDevice.current.batteryLevel
                configStartTime = CACurrentMediaTime()
                captureSize = "unknown"

                let nextConfig = allConfigs[nextCfg]
                let progress = configProgressPrefix()
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.onProgress?("\(progress) ⏸ 冷却中 → \(nextConfig.preset.displayName) \(nextConfig.target.displayName)")
                }
            }
        } else {
            finalizeAll()
        }
    }

    // MARK: - Cooling between configs (same preset)

    // The cooling state reuses the algorithm cooling state machine.
    // When nextAlgorithmIndex == 0 and we've finalized the config, the cooling
    // callback in processFrame will call beginAlgorithm(0) which starts fresh.
    // We override beginAlgorithm to detect this via captureSize == "unknown".

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

        let report = ComprehensiveReport(
            device: Self.deviceName(),
            osVersion: UIDevice.current.systemVersion,
            date: formatter.string(from: Date()),
            isolatedMode: true,
            cooldownSeconds: cooldownDuration,
            presetCooldownSeconds: presetCooldownDuration,
            framesPerAlgorithm: framesToCollect,
            totalAlgorithmRuns: totalRuns,
            totalDurationSeconds: duration,
            batteryBefore: Double(globalStartBattery) * 100,
            batteryAfter: Double(batteryAtEnd) * 100,
            batteryDrainPercent: batteryDrain,
            estimatedPowerMW: powerMW,
            cpuEnergyScore: totalCPUSeconds,
            thermalStateBefore: globalStartThermal,
            thermalStateAfter: thermalAtEnd,
            peakThermalState: peakThermal,
            thermalTimeline: thermalTimeline,
            configurations: allConfigResults
        )

        allConfigResults.removeAll()

        DispatchQueue.main.async { [weak self] in
            self?.onComplete?(report)
        }
    }

    static func reportToJSON(_ report: ComprehensiveReport) -> String? {
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
