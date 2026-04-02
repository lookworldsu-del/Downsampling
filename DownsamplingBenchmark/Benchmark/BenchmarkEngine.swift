import Foundation
import UIKit
import Darwin

final class BenchmarkEngine {

    private let windowSize: Int
    private var cpuHistory: [FrameMetrics] = []
    private var gpuHistory: [FrameMetrics] = []

    private var startBatteryLevel: Float = -1
    private var startTime: TimeInterval = 0

    init(windowSize: Int = 120) {
        self.windowSize = windowSize
        UIDevice.current.isBatteryMonitoringEnabled = true
    }

    func reset() {
        cpuHistory.removeAll()
        gpuHistory.removeAll()
        startBatteryLevel = UIDevice.current.batteryLevel
        startTime = CACurrentMediaTime()
    }

    func recordCPU(processingTime: TimeInterval) {
        let metrics = buildMetrics(processingTime: processingTime, gpuTime: nil)
        cpuHistory.append(metrics)
        if cpuHistory.count > windowSize { cpuHistory.removeFirst() }
    }

    func recordGPU(processingTime: TimeInterval, gpuTime: TimeInterval?) {
        let metrics = buildMetrics(processingTime: processingTime, gpuTime: gpuTime)
        gpuHistory.append(metrics)
        if gpuHistory.count > windowSize { gpuHistory.removeFirst() }
    }

    var cpuFrameTimes: [TimeInterval] { cpuHistory.map { $0.processingTime } }
    var gpuFrameTimes: [TimeInterval] { gpuHistory.map { $0.processingTime } }

    func aggregatedCPU(droppedFrameRatio: Double, totalFrames: Int) -> AggregatedMetrics {
        aggregate(cpuHistory, droppedFrameRatio: droppedFrameRatio, totalFrames: totalFrames)
    }

    func aggregatedGPU(droppedFrameRatio: Double, totalFrames: Int) -> AggregatedMetrics {
        aggregate(gpuHistory, droppedFrameRatio: droppedFrameRatio, totalFrames: totalFrames)
    }

    // MARK: - System Metrics

    static func currentCPUUsage() -> Double {
        var threadsList: thread_act_array_t?
        var threadCount = mach_msg_type_number_t(0)

        let result = task_threads(mach_task_self_, &threadsList, &threadCount)
        guard result == KERN_SUCCESS, let threads = threadsList else { return 0 }

        var totalUsage: Double = 0
        for i in 0..<Int(threadCount) {
            var info = thread_basic_info()
            var infoCount = mach_msg_type_number_t(MemoryLayout<thread_basic_info_data_t>.size / MemoryLayout<natural_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(infoCount)) {
                    thread_info(threads[i], thread_flavor_t(THREAD_BASIC_INFO), $0, &infoCount)
                }
            }
            if kr == KERN_SUCCESS && (info.flags & TH_FLAGS_IDLE) == 0 {
                totalUsage += Double(info.cpu_usage) / Double(TH_USAGE_SCALE) * 100.0
            }
        }

        vm_deallocate(mach_task_self_, vm_address_t(bitPattern: threads), vm_size_t(threadCount) * vm_size_t(MemoryLayout<thread_t>.stride))
        return totalUsage
    }

    static func currentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.resident_size : 0
    }

    // MARK: - Private

    private func buildMetrics(processingTime: TimeInterval, gpuTime: TimeInterval?) -> FrameMetrics {
        FrameMetrics(
            timestamp: CACurrentMediaTime(),
            processingTime: processingTime,
            gpuTime: gpuTime,
            cpuUsage: Self.currentCPUUsage(),
            memoryUsage: Self.currentMemoryUsage(),
            thermalState: ProcessInfo.processInfo.thermalState,
            batteryLevel: UIDevice.current.batteryLevel
        )
    }

    private func aggregate(_ history: [FrameMetrics], droppedFrameRatio: Double, totalFrames: Int) -> AggregatedMetrics {
        guard !history.isEmpty else { return .zero }

        let times = history.map { $0.processingTime }
        let sorted = times.sorted()
        let avg = times.reduce(0, +) / Double(times.count)
        let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count)
        let p99Index = min(Int(Double(sorted.count) * 0.99), sorted.count - 1)

        let fps: Double = avg > 0 ? 1.0 / avg : 0

        let currentBattery = UIDevice.current.batteryLevel
        let elapsed = CACurrentMediaTime() - startTime
        let batteryDrain = max(0, startBatteryLevel - currentBattery) * 100
        let drainRate: Float = elapsed > 0 ? batteryDrain / Float(elapsed / 60.0) : 0

        let peakMem = history.map { $0.memoryUsage }.max() ?? 0
        let avgCPU = history.map { $0.cpuUsage }.reduce(0, +) / Double(history.count)

        return AggregatedMetrics(
            averageTime: avg,
            minTime: sorted.first ?? 0,
            maxTime: sorted.last ?? 0,
            p99Time: sorted[p99Index],
            fps: fps,
            variance: variance,
            averageCPU: avgCPU,
            peakMemory: peakMem,
            currentMemory: history.last?.memoryUsage ?? 0,
            batteryDrain: batteryDrain,
            batteryDrainRate: drainRate,
            thermalState: history.last?.thermalState ?? .nominal,
            droppedFrameRatio: droppedFrameRatio,
            totalFrames: totalFrames
        )
    }
}
