import CoreVideo
import CoreGraphics
import Metal
import Accelerate

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

/// AI inference preprocessing final output format:
/// - `image`: CGImage for UI display (BGRA uint8)
/// - `rgbTensor`: Float32 RGB [H,W,3] normalized to [0,1] for model input
/// - `processingTime`: End-to-end wall time from YUV input to tensor output
struct DownsampleOutput {
    let image: CGImage?
    let rgbTensor: [Float]?
    let processingTime: TimeInterval
    let gpuTime: TimeInterval?
    let outputWidth: Int
    let outputHeight: Int
}

/// Convert BGRA uint8 CGImage → Float32 RGB [H,W,3] tensor normalized to [0,1].
/// Used by all non-Metal algorithms as the final conversion step.
func cgImageToRGBTensor(_ image: CGImage) -> [Float] {
    let w = image.width, h = image.height
    let bpr = w * 4
    var bgra = [UInt8](repeating: 0, count: h * bpr)
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: &bgra, width: w, height: h,
        bitsPerComponent: 8, bytesPerRow: bpr, space: cs,
        bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
    ) else { return [] }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

    let px = w * h
    var tensor = [Float](repeating: 0, count: px * 3)
    for i in 0..<px {
        let off = i * 4
        tensor[i * 3]     = Float(bgra[off + 2]) / 255.0
        tensor[i * 3 + 1] = Float(bgra[off + 1]) / 255.0
        tensor[i * 3 + 2] = Float(bgra[off])     / 255.0
    }
    return tensor
}

protocol Downsampler: AnyObject {
    var id: DownsamplerID { get }
    var name: String { get }
    var type: DownsamplerType { get }
    func downsample(_ pixelBuffer: CVPixelBuffer, target: DownsampleTarget) -> DownsampleOutput
}

// MARK: - YUV→BGRA Conversion Utility (for CPU downsamplers)

final class YUVConverter {

    struct BGRABuffer {
        let data: UnsafeMutableRawPointer
        let width: Int
        let height: Int
        let bytesPerRow: Int
    }

    private var conversionInfo = vImage_YpCbCrToARGB()
    private var infoReady = false
    private var bgraData: UnsafeMutableRawPointer?
    private var cachedWidth = 0
    private var cachedHeight = 0

    deinit {
        if let d = bgraData { free(d) }
    }

    func convert(_ pixelBuffer: CVPixelBuffer) -> BGRABuffer? {
        let fmt = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
              fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange else {
            return nil
        }

        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)

        if !infoReady {
            var pixelRange = vImage_YpCbCrPixelRange(
                Yp_bias: 0, CbCr_bias: 128,
                YpRangeMax: 255, CbCrRangeMax: 255,
                YpMax: 255, YpMin: 0, CbCrMax: 255, CbCrMin: 0
            )
            vImageConvert_YpCbCrToARGB_GenerateConversion(
                kvImage_YpCbCrToARGBMatrix_ITU_R_601_4,
                &pixelRange, &conversionInfo,
                kvImage420Yp8_CbCr8, kvImageARGB8888,
                vImage_Flags(kvImageNoFlags)
            )
            infoReady = true
        }

        let bytesPerRow = width * 4
        if cachedWidth != width || cachedHeight != height {
            if let old = bgraData { free(old) }
            bgraData = malloc(height * bytesPerRow)
            cachedWidth = width
            cachedHeight = height
        }
        guard let dst = bgraData else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) else { return nil }

        let yBPR = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let uvBPR = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)

        var yBuf = vImage_Buffer(data: yBase, height: vImagePixelCount(height),
                                 width: vImagePixelCount(width), rowBytes: yBPR)
        var uvBuf = vImage_Buffer(data: uvBase, height: vImagePixelCount(height / 2),
                                  width: vImagePixelCount(width / 2), rowBytes: uvBPR)
        var dstBuf = vImage_Buffer(data: dst, height: vImagePixelCount(height),
                                   width: vImagePixelCount(width), rowBytes: bytesPerRow)

        var permuteMap: [UInt8] = [3, 2, 1, 0]
        vImageConvert_420Yp8_CbCr8ToARGB8888(
            &yBuf, &uvBuf, &dstBuf, &conversionInfo,
            &permuteMap, 255, vImage_Flags(kvImageNoFlags)
        )

        return BGRABuffer(data: dst, width: width, height: height, bytesPerRow: bytesPerRow)
    }
}

func isYUVFormat(_ pixelBuffer: CVPixelBuffer) -> Bool {
    let fmt = CVPixelBufferGetPixelFormatType(pixelBuffer)
    return fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
           fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
}
