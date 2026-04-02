import Foundation

struct FrameMetrics {
    let timestamp: TimeInterval
    let processingTime: TimeInterval   // seconds
    let gpuTime: TimeInterval?         // GPU-only
    let cpuUsage: Double               // percent 0-100
    let memoryUsage: UInt64            // bytes
    let thermalState: ProcessInfo.ThermalState
    let batteryLevel: Float            // 0.0 - 1.0
}

struct AggregatedMetrics {
    let averageTime: TimeInterval
    let minTime: TimeInterval
    let maxTime: TimeInterval
    let p99Time: TimeInterval
    let fps: Double
    let variance: Double
    let averageCPU: Double
    let peakMemory: UInt64
    let currentMemory: UInt64
    let batteryDrain: Float            // percentage points consumed
    let batteryDrainRate: Float        // %/min
    let thermalState: ProcessInfo.ThermalState
    let droppedFrameRatio: Double
    let totalFrames: Int

    static let zero = AggregatedMetrics(
        averageTime: 0, minTime: 0, maxTime: 0, p99Time: 0,
        fps: 0, variance: 0, averageCPU: 0,
        peakMemory: 0, currentMemory: 0,
        batteryDrain: 0, batteryDrainRate: 0,
        thermalState: .nominal, droppedFrameRatio: 0, totalFrames: 0
    )
}

extension ProcessInfo.ThermalState {
    var displayName: String {
        switch self {
        case .nominal:  return "正常"
        case .fair:     return "微热"
        case .serious:  return "过热"
        case .critical: return "严重"
        @unknown default: return "未知"
        }
    }

    var emoji: String {
        switch self {
        case .nominal:  return "🟢"
        case .fair:     return "🟡"
        case .serious:  return "🟠"
        case .critical: return "🔴"
        @unknown default: return "⚪"
        }
    }
}
