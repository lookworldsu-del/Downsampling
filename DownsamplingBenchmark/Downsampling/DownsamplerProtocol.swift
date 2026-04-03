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

struct LetterboxLayout: Equatable {
    let canvasWidth: Int
    let canvasHeight: Int
    let innerX: Int
    let innerY: Int
    let innerWidth: Int
    let innerHeight: Int
}

enum DownsampleTarget: Equatable {
    case scale(Float)
    case fixedSize(width: Int, height: Int)
    case letterbox(width: Int, height: Int)

    static let allPresets: [DownsampleTarget] = [
        .scale(0.5), .scale(0.25), .scale(0.125),
        .fixedSize(width: 416, height: 416),
        .letterbox(width: 416, height: 416),
    ]

    func outputSize(inputWidth: Int, inputHeight: Int) -> (width: Int, height: Int) {
        switch self {
        case .scale(let factor):
            return (max(1, Int(Float(inputWidth) * factor)),
                    max(1, Int(Float(inputHeight) * factor)))
        case .fixedSize(let w, let h):
            return (w, h)
        case .letterbox(let w, let h):
            return (w, h)
        }
    }

    func letterboxLayout(inputWidth: Int, inputHeight: Int) -> LetterboxLayout? {
        guard case .letterbox(let cw, let ch) = self else { return nil }
        let scale = min(Float(cw) / Float(inputWidth), Float(ch) / Float(inputHeight))
        let innerW = max(1, Int(Float(inputWidth) * scale))
        let innerH = max(1, Int(Float(inputHeight) * scale))
        return LetterboxLayout(
            canvasWidth: cw, canvasHeight: ch,
            innerX: (cw - innerW) / 2, innerY: (ch - innerH) / 2,
            innerWidth: innerW, innerHeight: innerH
        )
    }

    var displayName: String {
        switch self {
        case .scale(let f):
            if f == 0.5 { return "1/2" }
            if f == 0.25 { return "1/4" }
            if f == 0.125 { return "1/8" }
            return String(format: "%.3f", f)
        case .fixedSize(let w, let h):
            return "\(w)×\(h)"
        case .letterbox(let w, let h):
            return "\(w)×\(h) Letterbox"
        }
    }

    var persistenceKey: String {
        switch self {
        case .scale(let f): return String(format: "scale_%.3f", f)
        case .fixedSize(let w, let h): return "fixed_\(w)x\(h)"
        case .letterbox(let w, let h): return "letterbox_\(w)x\(h)"
        }
    }

    static func from(persistenceKey key: String) -> DownsampleTarget? {
        if key.hasPrefix("scale_"), let f = Float(String(key.dropFirst(6))) {
            return .scale(f)
        }
        if key.hasPrefix("fixed_") {
            let parts = key.dropFirst(6).split(separator: "x")
            if parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]) {
                return .fixedSize(width: w, height: h)
            }
        }
        if key.hasPrefix("letterbox_") {
            let parts = key.dropFirst(10).split(separator: "x")
            if parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]) {
                return .letterbox(width: w, height: h)
            }
        }
        return nil
    }
}

func letterboxComposite(innerImage: CGImage, layout: LetterboxLayout) -> CGImage? {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
    guard let ctx = CGContext(
        data: nil, width: layout.canvasWidth, height: layout.canvasHeight,
        bitsPerComponent: 8, bytesPerRow: layout.canvasWidth * 4,
        space: colorSpace, bitmapInfo: bitmapInfo
    ) else { return nil }
    ctx.setFillColor(gray: 0.5, alpha: 1.0)
    ctx.fill(CGRect(x: 0, y: 0, width: layout.canvasWidth, height: layout.canvasHeight))
    ctx.draw(innerImage, in: CGRect(x: layout.innerX, y: layout.innerY,
                                     width: layout.innerWidth, height: layout.innerHeight))
    return ctx.makeImage()
}

struct DownsampleOutput {
    let image: CGImage?
    let processingTime: TimeInterval
    let gpuTime: TimeInterval?
    let outputWidth: Int
    let outputHeight: Int
}

protocol Downsampler: AnyObject {
    var id: DownsamplerID { get }
    var name: String { get }
    var type: DownsamplerType { get }
    func downsample(_ pixelBuffer: CVPixelBuffer, target: DownsampleTarget) -> DownsampleOutput
}
