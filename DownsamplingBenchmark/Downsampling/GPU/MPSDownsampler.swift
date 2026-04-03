import Metal
import MetalPerformanceShaders
import CoreVideo
import CoreGraphics
import QuartzCore

final class MPSDownsampler: Downsampler {

    let id: DownsamplerID = .mps
    let name = "MPS (Metal Performance Shaders)"
    let type: DownsamplerType = .gpu

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let yuvConvertPipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?

    private var outputTexture: MTLTexture?
    private var lastOutputSize: (Int, Int) = (0, 0)
    private var innerTexture: MTLTexture?
    private var lastInnerSize: (Int, Int) = (0, 0)
    private var fullResBGRATexture: MTLTexture?
    private var lastFullResSize: (Int, Int) = (0, 0)

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let fn = library.makeFunction(name: "yuv_to_bgra"),
              let pipeline = try? device.makeComputePipelineState(function: fn)
        else { return nil }

        self.device = device
        self.commandQueue = queue
        self.yuvConvertPipeline = pipeline

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        self.textureCache = cache
    }

    func downsample(_ pixelBuffer: CVPixelBuffer, target: DownsampleTarget) -> DownsampleOutput {
        let wallStart = CACurrentMediaTime()

        let srcWidth = CVPixelBufferGetWidth(pixelBuffer)
        let srcHeight = CVPixelBufferGetHeight(pixelBuffer)
        let (dstWidth, dstHeight) = target.outputSize(inputWidth: srcWidth, inputHeight: srcHeight)
        let layout = target.letterboxLayout(inputWidth: srcWidth, inputHeight: srcHeight)

        guard let cache = textureCache else { return fail(wallStart) }

        let isYUV = isYUVFormat(pixelBuffer)

        // Step 1: Get or create a BGRA source texture for MPS
        let bgraSource: MTLTexture
        if isYUV {
            guard let (yTex, uvTex) = makeYUVTextures(pixelBuffer: pixelBuffer, cache: cache) else {
                return fail(wallStart)
            }
            let fullRes = getOrCreateFullResBGRA(width: srcWidth, height: srcHeight)

            guard let cb = commandQueue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { return fail(wallStart) }
            enc.setComputePipelineState(yuvConvertPipeline)
            enc.setTexture(yTex, index: 0)
            enc.setTexture(uvTex, index: 1)
            enc.setTexture(fullRes, index: 2)
            let tgs = MTLSize(width: 16, height: 16, depth: 1)
            let tgc = MTLSize(width: (srcWidth + 15) / 16, height: (srcHeight + 15) / 16, depth: 1)
            enc.dispatchThreadgroups(tgc, threadsPerThreadgroup: tgs)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            bgraSource = fullRes
        } else {
            var cvTex: CVMetalTexture?
            CVMetalTextureCacheCreateTextureFromImage(nil, cache, pixelBuffer, nil,
                                                      .bgra8Unorm, srcWidth, srcHeight, 0, &cvTex)
            guard let cv = cvTex, let tex = CVMetalTextureGetTexture(cv) else { return fail(wallStart) }
            bgraSource = tex
        }

        // Step 2: MPS bilinear scale
        let scaleTargetW: Int
        let scaleTargetH: Int
        if let lb = layout {
            scaleTargetW = lb.innerWidth
            scaleTargetH = lb.innerHeight
        } else {
            scaleTargetW = dstWidth
            scaleTargetH = dstHeight
        }

        let scaleDst: MTLTexture
        if layout != nil {
            scaleDst = getOrCreateInnerTexture(width: scaleTargetW, height: scaleTargetH)
        } else {
            scaleDst = getOrCreateOutputTexture(width: dstWidth, height: dstHeight)
        }

        let scaleX = Double(scaleTargetW) / Double(srcWidth)
        let scaleY = Double(scaleTargetH) / Double(srcHeight)
        var transform = MPSScaleTransform(scaleX: scaleX, scaleY: scaleY, translateX: 0, translateY: 0)
        let bilinearScale = MPSImageBilinearScale(device: device)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return fail(wallStart) }

        withUnsafePointer(to: &transform) { ptr in
            bilinearScale.scaleTransform = ptr
            bilinearScale.encode(commandBuffer: commandBuffer, sourceTexture: bgraSource, destinationTexture: scaleDst)
        }

        // Step 3: Letterbox compositing
        if let lb = layout {
            let outTex = getOrCreateOutputTexture(width: dstWidth, height: dstHeight)

            if let blit = commandBuffer.makeBlitCommandEncoder() {
                let region = MTLRegionMake2D(0, 0, dstWidth, dstHeight)
                let bytesPerRow = dstWidth * 4
                let totalBytes = bytesPerRow * dstHeight
                var grayPixels = [UInt8](repeating: 0, count: totalBytes)
                for i in stride(from: 0, to: totalBytes, by: 4) {
                    grayPixels[i]     = 128
                    grayPixels[i + 1] = 128
                    grayPixels[i + 2] = 128
                    grayPixels[i + 3] = 255
                }
                outTex.replace(region: region, mipmapLevel: 0, withBytes: &grayPixels, bytesPerRow: bytesPerRow)

                blit.copy(from: scaleDst, sourceSlice: 0, sourceLevel: 0,
                          sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                          sourceSize: MTLSize(width: lb.innerWidth, height: lb.innerHeight, depth: 1),
                          to: outTex, destinationSlice: 0, destinationLevel: 0,
                          destinationOrigin: MTLOrigin(x: lb.innerX, y: lb.innerY, z: 0))
                blit.endEncoding()
            }
        }

        var gpuStartTime: TimeInterval = 0
        var gpuEndTime: TimeInterval = 0
        commandBuffer.addScheduledHandler { _ in gpuStartTime = CACurrentMediaTime() }
        commandBuffer.addCompletedHandler { _ in gpuEndTime = CACurrentMediaTime() }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let gpuTime = gpuEndTime - gpuStartTime
        let finalTex = layout != nil ? getOrCreateOutputTexture(width: dstWidth, height: dstHeight) : scaleDst
        let image = makeImage(from: finalTex, width: dstWidth, height: dstHeight)
        let wallTime = CACurrentMediaTime() - wallStart

        return DownsampleOutput(image: image, processingTime: wallTime, gpuTime: gpuTime,
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

    private func getOrCreateFullResBGRA(width: Int, height: Int) -> MTLTexture {
        if let tex = fullResBGRATexture, lastFullResSize == (width, height) { return tex }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderWrite, .shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        fullResBGRATexture = tex
        lastFullResSize = (width, height)
        return tex
    }

    private func getOrCreateInnerTexture(width: Int, height: Int) -> MTLTexture {
        if let tex = innerTexture, lastInnerSize == (width, height) { return tex }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderWrite, .shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        innerTexture = tex
        lastInnerSize = (width, height)
        return tex
    }

    private func getOrCreateOutputTexture(width: Int, height: Int) -> MTLTexture {
        if let tex = outputTexture, lastOutputSize == (width, height) { return tex }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderWrite, .shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        outputTexture = tex
        lastOutputSize = (width, height)
        return tex
    }

    private func makeImage(from texture: MTLTexture, width: Int, height: Int) -> CGImage? {
        let bytesPerRow = width * 4
        var pixelData = [UInt8](repeating: 0, count: bytesPerRow * height)
        texture.getBytes(&pixelData, bytesPerRow: bytesPerRow,
                         from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return nil }
        return context.makeImage()
    }

    private func fail(_ wallStart: TimeInterval) -> DownsampleOutput {
        DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart,
                         gpuTime: nil, outputWidth: 0, outputHeight: 0)
    }
}
