import Metal
import CoreVideo
import CoreGraphics
import QuartzCore

final class MetalDownsampler: Downsampler {

    let id: DownsamplerID = .metalCompute
    let name = "Metal Compute Shader"
    let type: DownsamplerType = .gpu

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private let pipelineState: MTLComputePipelineState
    private let letterboxPipeline: MTLComputePipelineState
    private let yuvPipeline: MTLComputePipelineState
    private let yuvLetterboxPipeline: MTLComputePipelineState

    private var textureCache: CVMetalTextureCache?
    private var floatTexture: MTLTexture?
    private var lastOutputSize: (Int, Int) = (0, 0)

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let fn1 = library.makeFunction(name: "downsample_bilinear"),
              let p1 = try? device.makeComputePipelineState(function: fn1),
              let fn2 = library.makeFunction(name: "downsample_letterbox"),
              let p2 = try? device.makeComputePipelineState(function: fn2),
              let fn3 = library.makeFunction(name: "downsample_yuv_bilinear"),
              let p3 = try? device.makeComputePipelineState(function: fn3),
              let fn4 = library.makeFunction(name: "downsample_yuv_letterbox"),
              let p4 = try? device.makeComputePipelineState(function: fn4)
        else { return nil }

        self.device = device
        self.commandQueue = queue
        self.pipelineState = p1
        self.letterboxPipeline = p2
        self.yuvPipeline = p3
        self.yuvLetterboxPipeline = p4

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
        let outTex = getOrCreateFloatTexture(width: dstWidth, height: dstHeight)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return fail(wallStart)
        }

        if isYUV {
            guard let (yTex, uvTex) = makeYUVTextures(pixelBuffer: pixelBuffer, cache: cache) else {
                encoder.endEncoding()
                return fail(wallStart)
            }
            if let lb = layout {
                encoder.setComputePipelineState(yuvLetterboxPipeline)
                encoder.setTexture(yTex, index: 0)
                encoder.setTexture(uvTex, index: 1)
                encoder.setTexture(outTex, index: 2)
                var params = SIMD4<Float>(Float(lb.innerX), Float(lb.innerY),
                                          Float(lb.innerWidth), Float(lb.innerHeight))
                encoder.setBytes(&params, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
            } else {
                encoder.setComputePipelineState(yuvPipeline)
                encoder.setTexture(yTex, index: 0)
                encoder.setTexture(uvTex, index: 1)
                encoder.setTexture(outTex, index: 2)
            }
        } else {
            guard let inputTexture = makeBGRATexture(pixelBuffer: pixelBuffer, cache: cache,
                                                     width: srcWidth, height: srcHeight) else {
                encoder.endEncoding()
                return fail(wallStart)
            }
            if let lb = layout {
                encoder.setComputePipelineState(letterboxPipeline)
                encoder.setTexture(inputTexture, index: 0)
                encoder.setTexture(outTex, index: 1)
                var params = SIMD4<Float>(Float(lb.innerX), Float(lb.innerY),
                                          Float(lb.innerWidth), Float(lb.innerHeight))
                encoder.setBytes(&params, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
            } else {
                encoder.setComputePipelineState(pipelineState)
                encoder.setTexture(inputTexture, index: 0)
                encoder.setTexture(outTex, index: 1)
            }
        }

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

        // Read back float RGBA data — no uint8 quantization, no extra conversion
        let floatBytesPerRow = dstWidth * 16
        var floatData = [Float](repeating: 0, count: dstWidth * dstHeight * 4)
        outTex.getBytes(&floatData, bytesPerRow: floatBytesPerRow,
                        from: MTLRegionMake2D(0, 0, dstWidth, dstHeight), mipmapLevel: 0)

        // Extract RGB tensor directly from float RGBA (skip alpha channel)
        let px = dstWidth * dstHeight
        var tensor = [Float](repeating: 0, count: px * 3)
        for i in 0..<px {
            tensor[i * 3]     = floatData[i * 4]
            tensor[i * 3 + 1] = floatData[i * 4 + 1]
            tensor[i * 3 + 2] = floatData[i * 4 + 2]
        }

        // Derive display CGImage from float data (cheap at 416×416)
        let image = makeImageFromFloat(floatData, width: dstWidth, height: dstHeight)

        let wallTime = CACurrentMediaTime() - wallStart
        return DownsampleOutput(image: image, rgbTensor: tensor,
                                processingTime: wallTime, gpuTime: gpuTime,
                                outputWidth: dstWidth, outputHeight: dstHeight)
    }

    // MARK: - Texture helpers

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

    /// Float32 RGBA output texture — shader writes float precision, no uint8 quantization
    private func getOrCreateFloatTexture(width: Int, height: Int) -> MTLTexture {
        if let tex = floatTexture, lastOutputSize == (width, height) { return tex }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderWrite, .shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        floatTexture = tex
        lastOutputSize = (width, height)
        return tex
    }

    /// Convert float RGBA data → CGImage for display
    private func makeImageFromFloat(_ floatData: [Float], width: Int, height: Int) -> CGImage? {
        let bytesPerRow = width * 4
        var uint8Data = [UInt8](repeating: 0, count: height * bytesPerRow)
        let count = width * height
        for i in 0..<count {
            let src = i * 4
            let dst = i * 4
            // Float RGBA → BGRA uint8 for CGContext
            uint8Data[dst]     = UInt8(max(0, min(255, floatData[src + 2] * 255.0))) // B
            uint8Data[dst + 1] = UInt8(max(0, min(255, floatData[src + 1] * 255.0))) // G
            uint8Data[dst + 2] = UInt8(max(0, min(255, floatData[src]     * 255.0))) // R
            uint8Data[dst + 3] = UInt8(max(0, min(255, floatData[src + 3] * 255.0))) // A
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &uint8Data, width: width, height: height,
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
