import CoreGraphics
import CoreVideo
import QuartzCore

final class CGContextDownsampler: Downsampler {

    let id: DownsamplerID = .cgContext
    let name = "CGContext (Core Graphics)"
    let type: DownsamplerType = .cpu

    private let yuvConverter = YUVConverter()

    func downsample(_ pixelBuffer: CVPixelBuffer, target: DownsampleTarget) -> DownsampleOutput {
        let start = CACurrentMediaTime()

        let srcBase: UnsafeMutableRawPointer
        let srcWidth: Int
        let srcHeight: Int
        let srcBytesPerRow: Int
        var needsUnlock = false

        if isYUVFormat(pixelBuffer) {
            guard let bgra = yuvConverter.convert(pixelBuffer) else {
                return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - start,
                                        gpuTime: nil, outputWidth: 0, outputHeight: 0)
            }
            srcBase = bgra.data
            srcWidth = bgra.width
            srcHeight = bgra.height
            srcBytesPerRow = bgra.bytesPerRow
        } else {
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            needsUnlock = true
            guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
                return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - start,
                                        gpuTime: nil, outputWidth: 0, outputHeight: 0)
            }
            srcBase = base
            srcWidth = CVPixelBufferGetWidth(pixelBuffer)
            srcHeight = CVPixelBufferGetHeight(pixelBuffer)
            srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue

        guard let srcContext = CGContext(
            data: srcBase, width: srcWidth, height: srcHeight,
            bitsPerComponent: 8, bytesPerRow: srcBytesPerRow,
            space: colorSpace, bitmapInfo: bitmapInfo
        ), let srcImage = srcContext.makeImage() else {
            if needsUnlock { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - start,
                                    gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        let (dstWidth, dstHeight) = target.outputSize(inputWidth: srcWidth, inputHeight: srcHeight)
        let layout = target.letterboxLayout(inputWidth: srcWidth, inputHeight: srcHeight)

        guard let dstContext = CGContext(
            data: nil, width: dstWidth, height: dstHeight,
            bitsPerComponent: 8, bytesPerRow: dstWidth * 4,
            space: colorSpace, bitmapInfo: bitmapInfo
        ) else {
            if needsUnlock { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - start,
                                    gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        dstContext.interpolationQuality = .high
        if let lb = layout {
            dstContext.setFillColor(gray: 0.5, alpha: 1.0)
            dstContext.fill(CGRect(x: 0, y: 0, width: dstWidth, height: dstHeight))
            dstContext.draw(srcImage, in: CGRect(x: lb.innerX, y: lb.innerY,
                                                 width: lb.innerWidth, height: lb.innerHeight))
        } else {
            dstContext.draw(srcImage, in: CGRect(x: 0, y: 0, width: dstWidth, height: dstHeight))
        }

        let image = dstContext.makeImage()
        if needsUnlock { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        let elapsed = CACurrentMediaTime() - start
        return DownsampleOutput(image: image, processingTime: elapsed, gpuTime: nil,
                                outputWidth: dstWidth, outputHeight: dstHeight)
    }
}
