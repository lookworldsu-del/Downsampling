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
    private var textureCache: CVMetalTextureCache?

    private var outputTexture: MTLTexture?
    private var lastOutputSize: (Int, Int) = (0, 0)

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { return nil }

        self.device = device
        self.commandQueue = queue

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        self.textureCache = cache
    }

    func downsample(_ pixelBuffer: CVPixelBuffer, target: DownsampleTarget) -> DownsampleOutput {
        let wallStart = CACurrentMediaTime()

        let srcWidth = CVPixelBufferGetWidth(pixelBuffer)
        let srcHeight = CVPixelBufferGetHeight(pixelBuffer)
        let (dstWidth, dstHeight) = target.outputSize(inputWidth: srcWidth, inputHeight: srcHeight)

        guard let cache = textureCache else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        var cvTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            nil, cache, pixelBuffer, nil,
            .bgra8Unorm, srcWidth, srcHeight, 0, &cvTexture
        )
        guard let cvTex = cvTexture, let inputTexture = CVMetalTextureGetTexture(cvTex) else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        let outTex = getOrCreateOutputTexture(width: dstWidth, height: dstHeight)

        let scaleX = Double(dstWidth) / Double(srcWidth)
        let scaleY = Double(dstHeight) / Double(srcHeight)
        let translateX = 0.0
        let translateY = 0.0

        var transform = MPSScaleTransform(scaleX: scaleX, scaleY: scaleY, translateX: translateX, translateY: translateY)

        let bilinearScale = MPSImageBilinearScale(device: device)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        withUnsafePointer(to: &transform) { ptr in
            bilinearScale.scaleTransform = ptr
            bilinearScale.encode(commandBuffer: commandBuffer, sourceTexture: inputTexture, destinationTexture: outTex)
        }

        var gpuStartTime: TimeInterval = 0
        var gpuEndTime: TimeInterval = 0

        commandBuffer.addScheduledHandler { _ in
            gpuStartTime = CACurrentMediaTime()
        }
        commandBuffer.addCompletedHandler { _ in
            gpuEndTime = CACurrentMediaTime()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let gpuTime = gpuEndTime - gpuStartTime
        let image = makeImage(from: outTex, width: dstWidth, height: dstHeight)
        let wallTime = CACurrentMediaTime() - wallStart

        return DownsampleOutput(image: image, processingTime: wallTime, gpuTime: gpuTime, outputWidth: dstWidth, outputHeight: dstHeight)
    }

    private func getOrCreateOutputTexture(width: Int, height: Int) -> MTLTexture {
        if let tex = outputTexture, lastOutputSize == (width, height) {
            return tex
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderWrite, .shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        outputTexture = tex
        lastOutputSize = (width, height)
        return tex
    }

    private func makeImage(from texture: MTLTexture, width: Int, height: Int) -> CGImage? {
        let bytesPerRow = width * 4
        let totalBytes = bytesPerRow * height
        var pixelData = [UInt8](repeating: 0, count: totalBytes)

        texture.getBytes(&pixelData, bytesPerRow: bytesPerRow,
                         from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return nil }

        return context.makeImage()
    }
}
