import CoreVideo
import CoreGraphics
import Metal

enum DownsamplerType: String, CaseIterable {
    case cpu = "CPU"
    case gpu = "GPU"
}

enum DownsamplerID: String, CaseIterable {
    case vImage = "vImage (Accelerate)"
    case cgContext = "CGContext (Core Graphics)"
    case uiGraphics = "UIGraphicsImageRenderer"
    case metalCompute = "Metal Compute Shader"
    case mps = "MPS (Metal Performance Shaders)"
    case coreImage = "Core Image (CILanczos)"

    var type: DownsamplerType {
        switch self {
        case .vImage, .cgContext, .uiGraphics: return .cpu
        case .metalCompute, .mps, .coreImage: return .gpu
        }
    }
}

enum ScaleFactor: Float, CaseIterable {
    case half = 0.5
    case quarter = 0.25
    case eighth = 0.125

    var displayName: String {
        switch self {
        case .half: return "1/2"
        case .quarter: return "1/4"
        case .eighth: return "1/8"
        }
    }
}

struct DownsampleOutput {
    let image: CGImage?
    let processingTime: TimeInterval  // seconds
    let gpuTime: TimeInterval?        // GPU command buffer time, nil for CPU
    let outputWidth: Int
    let outputHeight: Int
}

protocol Downsampler: AnyObject {
    var id: DownsamplerID { get }
    var name: String { get }
    var type: DownsamplerType { get }
    func downsample(_ pixelBuffer: CVPixelBuffer, scaleFactor: Float) -> DownsampleOutput
}
