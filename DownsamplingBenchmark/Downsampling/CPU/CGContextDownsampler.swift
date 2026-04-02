import CoreGraphics
import CoreVideo
import QuartzCore

final class CGContextDownsampler: Downsampler {

    let id: DownsamplerID = .cgContext
    let name = "CGContext (Core Graphics)"
    let type: DownsamplerType = .cpu

    func downsample(_ pixelBuffer: CVPixelBuffer, scaleFactor: Float) -> DownsampleOutput {
        let start = CACurrentMediaTime()

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let srcWidth = CVPixelBufferGetWidth(pixelBuffer)
        let srcHeight = CVPixelBufferGetHeight(pixelBuffer)
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - start, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue

        guard let srcContext = CGContext(
            data: srcBase,
            width: srcWidth,
            height: srcHeight,
            bitsPerComponent: 8,
            bytesPerRow: srcBytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let srcImage = srcContext.makeImage() else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - start, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        let dstWidth = Int(Float(srcWidth) * scaleFactor)
        let dstHeight = Int(Float(srcHeight) * scaleFactor)

        guard let dstContext = CGContext(
            data: nil,
            width: dstWidth,
            height: dstHeight,
            bitsPerComponent: 8,
            bytesPerRow: dstWidth * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - start, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        dstContext.interpolationQuality = .high
        dstContext.draw(srcImage, in: CGRect(x: 0, y: 0, width: dstWidth, height: dstHeight))

        let image = dstContext.makeImage()
        let elapsed = CACurrentMediaTime() - start
        return DownsampleOutput(image: image, processingTime: elapsed, gpuTime: nil, outputWidth: dstWidth, outputHeight: dstHeight)
    }
}
