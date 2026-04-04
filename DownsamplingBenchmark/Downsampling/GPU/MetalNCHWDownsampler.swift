import Metal
import CoreVideo
import CoreGraphics
import QuartzCore

/// Zero-copy NCHW direct output path for AI inference.
/// GPU writes float32 NCHW tensor into shared memory — no CGImage, no getBytes, no tensor conversion.
final class MetalNCHWDownsampler: Downsampler {

    let id: DownsamplerID = .metalNCHW
    let name = "Metal NCHW Direct"
    let type: DownsamplerType = .gpu

    struct ConvertParam {
        /// yuv_to_rgb already outputs [0, 1] float. Default identity keeps [0, 1].
        /// For TNN models expecting specific normalization, set scale/bias accordingly.
        /// e.g. ImageNet: scale = {1, 1, 1}, bias = {-0.485, -0.456, -0.406}
        var scale: SIMD3<Float> = SIMD3(1.0, 1.0, 1.0)
        var bias: SIMD3<Float> = .zero
    }

    var convertParam = ConvertParam()

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private let yuvBilinearPipeline: MTLComputePipelineState
    private let yuvLetterboxPipeline: MTLComputePipelineState
    private let bgraBilinearPipeline: MTLComputePipelineState
    private let bgraLetterboxPipeline: MTLComputePipelineState

    private var textureCache: CVMetalTextureCache?
    private var outputBuffer: MTLBuffer?
    private var lastBufferFloatCount: Int = 0

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let fn1 = library.makeFunction(name: "downsample_yuv_bilinear_nchw"),
              let p1 = try? device.makeComputePipelineState(function: fn1),
              let fn2 = library.makeFunction(name: "downsample_yuv_letterbox_nchw"),
              let p2 = try? device.makeComputePipelineState(function: fn2),
              let fn3 = library.makeFunction(name: "downsample_bilinear_nchw"),
              let p3 = try? device.makeComputePipelineState(function: fn3),
              let fn4 = library.makeFunction(name: "downsample_letterbox_nchw"),
              let p4 = try? device.makeComputePipelineState(function: fn4)
        else { return nil }

        self.device = device
        self.commandQueue = queue
        self.yuvBilinearPipeline = p1
        self.yuvLetterboxPipeline = p2
        self.bgraBilinearPipeline = p3
        self.bgraLetterboxPipeline = p4

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        self.textureCache = cache
    }

    func downsample(_ pixelBuffer: CVPixelBuffer, target: DownsampleTarget) -> DownsampleOutput {
        let wallStart = CACurrentMediaTime()

        let srcWidth = CVPixelBufferGetWidth(pixelBuffer)
        let srcHeight = CVPixelBufferGetHeight(pixelBuffer)
        let (dstWidth, dstHeight) = target.outputSize(inputWidth: srcWidth, inputHeight: srcHeight)

        guard let cache = textureCache else { return fail(wallStart) }

        let isYUV = isYUVFormat(pixelBuffer)
        let layout = target.letterboxLayout(inputWidth: srcWidth, inputHeight: srcHeight)
        let buffer = getOrCreateOutputBuffer(width: dstWidth, height: dstHeight)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return fail(wallStart)
        }

        var params = NCHWParamsStruct(
            offsetX: 0, offsetY: 0,
            innerSizeX: Float(dstWidth), innerSizeY: Float(dstHeight),
            width: Int32(dstWidth), height: Int32(dstHeight),
            scaleX: convertParam.scale.x, scaleY: convertParam.scale.y, scaleZ: convertParam.scale.z,
            biasX: convertParam.bias.x, biasY: convertParam.bias.y, biasZ: convertParam.bias.z
        )

        if let lb = layout {
            params.offsetX = Float(lb.innerX)
            params.offsetY = Float(lb.innerY)
            params.innerSizeX = Float(lb.innerWidth)
            params.innerSizeY = Float(lb.innerHeight)
        }

        if isYUV {
            guard let (yTex, uvTex) = makeYUVTextures(pixelBuffer: pixelBuffer, cache: cache) else {
                encoder.endEncoding()
                return fail(wallStart)
            }
            encoder.setComputePipelineState(layout != nil ? yuvLetterboxPipeline : yuvBilinearPipeline)
            encoder.setTexture(yTex, index: 0)
            encoder.setTexture(uvTex, index: 1)
        } else {
            guard let inputTexture = makeBGRATexture(pixelBuffer: pixelBuffer, cache: cache,
                                                     width: srcWidth, height: srcHeight) else {
                encoder.endEncoding()
                return fail(wallStart)
            }
            encoder.setComputePipelineState(layout != nil ? bgraLetterboxPipeline : bgraBilinearPipeline)
            encoder.setTexture(inputTexture, index: 0)
        }

        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBytes(&params, length: MemoryLayout<NCHWParamsStruct>.size, index: 1)

        let tgs = MTLSize(width: 16, height: 16, depth: 1)
        let tgc = MTLSize(width: (dstWidth + 15) / 16, height: (dstHeight + 15) / 16, depth: 1)
        encoder.dispatchThreadgroups(tgc, threadsPerThreadgroup: tgs)
        encoder.endEncoding()

        var gpuStartTime: TimeInterval = 0
        var gpuEndTime: TimeInterval = 0
        commandBuffer.addScheduledHandler { _ in gpuStartTime = CACurrentMediaTime() }
        commandBuffer.addCompletedHandler { _ in gpuEndTime = CACurrentMediaTime() }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let gpuTime = gpuEndTime - gpuStartTime
        let floatCount = dstWidth * dstHeight * 3

        let tensorPtr = buffer.contents().assumingMemoryBound(to: Float.self)
        let tensor = Array(UnsafeBufferPointer(start: tensorPtr, count: floatCount))

        let image = makePreviewImage(from: tensor, width: dstWidth, height: dstHeight)

        let wallTime = CACurrentMediaTime() - wallStart
        return DownsampleOutput(image: image, rgbTensor: tensor,
                                processingTime: wallTime, gpuTime: gpuTime,
                                outputWidth: dstWidth, outputHeight: dstHeight)
    }

    // MARK: - Buffer Management

    private func getOrCreateOutputBuffer(width: Int, height: Int) -> MTLBuffer {
        let floatCount = width * height * 3
        if let buf = outputBuffer, lastBufferFloatCount == floatCount { return buf }

        let byteCount = floatCount * MemoryLayout<Float>.size
        let buf = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        outputBuffer = buf
        lastBufferFloatCount = floatCount
        return buf
    }

    // MARK: - Texture Helpers

    private func makeYUVTextures(pixelBuffer: CVPixelBuffer, cache: CVMetalTextureCache) -> (MTLTexture, MTLTexture)? {
        let yW = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let yH = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let uvW = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1)
        let uvH = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1)

        var yCV: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, pixelBuffer, nil,
                                                  .r8Unorm, yW, yH, 0, &yCV)
        var uvCV: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, pixelBuffer, nil,
                                                  .rg8Unorm, uvW, uvH, 1, &uvCV)

        guard let y = yCV, let uv = uvCV,
              let yTex = CVMetalTextureGetTexture(y),
              let uvTex = CVMetalTextureGetTexture(uv) else { return nil }
        return (yTex, uvTex)
    }

    private func makeBGRATexture(pixelBuffer: CVPixelBuffer, cache: CVMetalTextureCache,
                                 width: Int, height: Int) -> MTLTexture? {
        var cvTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, pixelBuffer, nil,
                                                  .bgra8Unorm, width, height, 0, &cvTex)
        guard let cv = cvTex else { return nil }
        return CVMetalTextureGetTexture(cv)
    }

    /// Reconstruct a CGImage from NCHW float tensor for UI preview only.
    /// In production this is unnecessary — the tensor goes directly to TNN.
    private func makePreviewImage(from tensor: [Float], width: Int, height: Int) -> CGImage? {
        let pixelCount = width * height
        guard tensor.count == pixelCount * 3 else { return nil }

        var bgra = [UInt8](repeating: 255, count: pixelCount * 4)
        let rPlane = tensor
        for i in 0..<pixelCount {
            let r = rPlane[i]
            let g = tensor[pixelCount + i]
            let b = tensor[2 * pixelCount + i]
            bgra[i * 4 + 0] = UInt8(clamping: Int(b * 255.0))
            bgra[i * 4 + 1] = UInt8(clamping: Int(g * 255.0))
            bgra[i * 4 + 2] = UInt8(clamping: Int(r * 255.0))
            bgra[i * 4 + 3] = 255
        }

        let bytesPerRow = width * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &bgra, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return nil }
        return context.makeImage()
    }

    private func fail(_ wallStart: TimeInterval) -> DownsampleOutput {
        DownsampleOutput(image: nil, rgbTensor: nil,
                         processingTime: CACurrentMediaTime() - wallStart,
                         gpuTime: nil, outputWidth: 0, outputHeight: 0)
    }
}

// MARK: - Params struct matching Metal side NCHWParams

private struct NCHWParamsStruct {
    var offsetX: Float
    var offsetY: Float
    var innerSizeX: Float
    var innerSizeY: Float
    var width: Int32
    var height: Int32
    var scaleX: Float
    var scaleY: Float
    var scaleZ: Float
    var biasX: Float
    var biasY: Float
    var biasZ: Float
}
