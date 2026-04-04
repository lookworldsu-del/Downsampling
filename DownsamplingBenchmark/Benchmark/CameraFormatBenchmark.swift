import Foundation
import UIKit
import CoreVideo
import QuartzCore
import Accelerate

// MARK: - Report Model

struct CameraFormatReport: Codable {
    let device: String
    let osVersion: String
    let date: String
    let framesPerConfig: Int
    let results: [CameraFormatResult]
}

struct CameraFormatResult: Codable {
    let resolution: String
    let pixelFormat: String
    let actualSize: String
    let frameCount: Int
    let droppedFrames: Int
    let bufferBytes: Int
    let bytesPerPixel: Double
    let avgFrameIntervalMs: Double
    let minFrameIntervalMs: Double
    let maxFrameIntervalMs: Double
    let deliveredFPS: Double
    let avgBufferAccessMs: Double
    let avgYUVConvertMs: Double
}

// MARK: - Runner

final class CameraFormatBenchmark {

    struct Config {
        let preset: CameraManager.Preset
        let format: CameraManager.PixelFormat
    }

    private enum Phase {
        case idle
        case waitingForSwitch
        case cooldown(startTime: TimeInterval)
        case warmup(remaining: Int)
        case measuring
        case done
    }

    private let configs: [Config] = [
        Config(preset: .hd1080p, format: .yuv),
        Config(preset: .hd1080p, format: .bgra),
        Config(preset: .uhd4K,   format: .yuv),
        Config(preset: .uhd4K,   format: .bgra),
    ]

    private let framesToCollect: Int
    private let warmupFrames = 10
    private let cooldownDuration: TimeInterval = 3.0

    private var phase: Phase = .idle
    private var currentConfigIndex = 0

    private var frameIntervals: [Double] = []
    private var bufferAccessTimes: [Double] = []
    private var yuvConvertTimes: [Double] = []
    private var lastFrameTime: TimeInterval = 0
    private var measuredFrameCount = 0
    private var bufferBytes = 0
    private var droppedFrameCount = 0
    private var actualWidth = 0
    private var actualHeight = 0

    private var allResults: [CameraFormatResult] = []

    private let yuvConverter = YUVConverter()

    private(set) var isRunning = false

    var onProgress: ((String) -> Void)?
    var onComplete: ((CameraFormatReport) -> Void)?
    var onSwitchPreset: ((CameraManager.Preset, @escaping () -> Void) -> Void)?
    var onSwitchFormat: ((CameraManager.PixelFormat, @escaping () -> Void) -> Void)?

    init(framesToCollect: Int = 120) {
        self.framesToCollect = framesToCollect
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true
        allResults = []
        currentConfigIndex = 0
        beginConfig(index: 0)
    }

    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard isRunning else { return }

        switch phase {
        case .idle, .done, .waitingForSwitch:
            return

        case .cooldown(let startTime):
            let elapsed = CACurrentMediaTime() - startTime
            if elapsed >= cooldownDuration {
                phase = .warmup(remaining: warmupFrames)
                lastFrameTime = 0
            } else {
                let remaining = Int(ceil(cooldownDuration - elapsed))
                let cfg = configs[currentConfigIndex]
                DispatchQueue.main.async { [weak self] in
                    self?.onProgress?("[\(cfg.format.displayName) \(cfg.preset.displayName)] 冷却 \(remaining)s...")
                }
            }

        case .warmup(let remaining):
            lastFrameTime = CACurrentMediaTime()
            if remaining <= 1 {
                phase = .measuring
                let cfg = configs[currentConfigIndex]
                DispatchQueue.main.async { [weak self] in
                    self?.onProgress?("[\(cfg.format.displayName) \(cfg.preset.displayName)] 采集中 0/\(self?.framesToCollect ?? 0)")
                }
            } else {
                phase = .warmup(remaining: remaining - 1)
            }

        case .measuring:
            measureFrame(pixelBuffer)
        }
    }

    func recordDroppedFrame() {
        if case .measuring = phase {
            droppedFrameCount += 1
        }
    }

    // MARK: - Internal

    private func beginConfig(index: Int) {
        currentConfigIndex = index
        let config = configs[index]

        frameIntervals = []
        bufferAccessTimes = []
        yuvConvertTimes = []
        lastFrameTime = 0
        measuredFrameCount = 0
        bufferBytes = 0
        droppedFrameCount = 0
        actualWidth = 0
        actualHeight = 0

        phase = .waitingForSwitch
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            let cfgNum = index + 1
            let cfgTotal = self.configs.count
            self.onProgress?("[\(cfgNum)/\(cfgTotal)] 切换至 \(config.format.displayName) \(config.preset.displayName)...")

            self.onSwitchFormat?(config.format) { [weak self] in
                guard let self else { return }
                self.onSwitchPreset?(config.preset) { [weak self] in
                    guard let self else { return }
                    if index > 0 {
                        self.phase = .cooldown(startTime: CACurrentMediaTime())
                    } else {
                        self.phase = .warmup(remaining: self.warmupFrames)
                    }
                }
            }
        }
    }

    private func measureFrame(_ pixelBuffer: CVPixelBuffer) {
        let now = CACurrentMediaTime()

        if actualWidth == 0 {
            actualWidth = CVPixelBufferGetWidth(pixelBuffer)
            actualHeight = CVPixelBufferGetHeight(pixelBuffer)
            bufferBytes = CVPixelBufferGetDataSize(pixelBuffer)
        }

        if lastFrameTime > 0 {
            let interval = (now - lastFrameTime) * 1000.0
            frameIntervals.append(interval)
        }
        lastFrameTime = now

        let accessStart = CACurrentMediaTime()
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        let isYUV = isYUVFormat(pixelBuffer)
        if isYUV {
            let yW = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
            let yH = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
            if let yBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) {
                _ = yBase.load(fromByteOffset: yW * yH / 2, as: UInt8.self)
            }
            if let uvBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) {
                let uvW = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1)
                let uvH = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1)
                _ = uvBase.load(fromByteOffset: uvW * uvH - 1, as: UInt8.self)
            }
        } else {
            if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
                let totalBytes = CVPixelBufferGetDataSize(pixelBuffer)
                _ = base.load(fromByteOffset: totalBytes / 2, as: UInt8.self)
                _ = base.load(fromByteOffset: totalBytes - 1, as: UInt8.self)
            }
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        let accessTime = (CACurrentMediaTime() - accessStart) * 1000.0
        bufferAccessTimes.append(accessTime)

        if isYUV {
            let convertStart = CACurrentMediaTime()
            _ = yuvConverter.convert(pixelBuffer)
            let convertTime = (CACurrentMediaTime() - convertStart) * 1000.0
            yuvConvertTimes.append(convertTime)
        }

        measuredFrameCount += 1

        let cfg = configs[currentConfigIndex]
        let cfgNum = currentConfigIndex + 1
        let cfgTotal = self.configs.count
        if measuredFrameCount % 10 == 0 {
            let count = measuredFrameCount
            let total = framesToCollect
            DispatchQueue.main.async { [weak self] in
                self?.onProgress?("[\(cfgNum)/\(cfgTotal)] \(cfg.format.displayName) \(cfg.preset.displayName) — \(count)/\(total)")
            }
        }

        if measuredFrameCount >= framesToCollect {
            finalizeConfig()
        }
    }

    private func finalizeConfig() {
        let config = configs[currentConfigIndex]
        let px = actualWidth * actualHeight

        let avgInterval = frameIntervals.isEmpty ? 0 : frameIntervals.reduce(0, +) / Double(frameIntervals.count)
        let sortedIntervals = frameIntervals.sorted()
        let minInterval = sortedIntervals.first ?? 0
        let maxInterval = sortedIntervals.last ?? 0
        let deliveredFPS = avgInterval > 0 ? 1000.0 / avgInterval : 0

        let avgAccess = bufferAccessTimes.isEmpty ? 0 : bufferAccessTimes.reduce(0, +) / Double(bufferAccessTimes.count)
        let avgConvert = yuvConvertTimes.isEmpty ? 0 : yuvConvertTimes.reduce(0, +) / Double(yuvConvertTimes.count)

        let result = CameraFormatResult(
            resolution: config.preset.displayName,
            pixelFormat: config.format.displayName,
            actualSize: "\(actualWidth)x\(actualHeight)",
            frameCount: measuredFrameCount,
            droppedFrames: droppedFrameCount,
            bufferBytes: bufferBytes,
            bytesPerPixel: px > 0 ? Double(bufferBytes) / Double(px) : 0,
            avgFrameIntervalMs: avgInterval,
            minFrameIntervalMs: minInterval,
            maxFrameIntervalMs: maxInterval,
            deliveredFPS: deliveredFPS,
            avgBufferAccessMs: avgAccess,
            avgYUVConvertMs: avgConvert
        )
        allResults.append(result)

        let nextIdx = currentConfigIndex + 1
        if nextIdx < configs.count {
            beginConfig(index: nextIdx)
        } else {
            finalizeAll()
        }
    }

    private func finalizeAll() {
        phase = .done
        isRunning = false

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"

        var systemInfo = utsname()
        uname(&systemInfo)
        let machine = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0) ?? "unknown"
            }
        }

        let report = CameraFormatReport(
            device: "\(machine) (\(UIDevice.current.name))",
            osVersion: UIDevice.current.systemVersion,
            date: formatter.string(from: Date()),
            framesPerConfig: framesToCollect,
            results: allResults
        )

        DispatchQueue.main.async { [weak self] in
            self?.onComplete?(report)
        }
    }

    static func reportToJSON(_ report: CameraFormatReport) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(report) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
