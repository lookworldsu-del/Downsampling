import UIKit
import CoreVideo
import QuartzCore

final class UIGraphicsDownsampler: Downsampler {

    let id: DownsamplerID = .uiGraphics
    let name = "UIGraphicsImageRenderer"
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
                return fail(start)
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
                return fail(start)
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
        ), let srcCGImage = srcContext.makeImage() else {
            if needsUnlock { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            return fail(start)
        }

        let srcImage = UIImage(cgImage: srcCGImage)
        let (dstWidth, dstHeight) = target.outputSize(inputWidth: srcWidth, inputHeight: srcHeight)
        let targetSize = CGSize(width: dstWidth, height: dstHeight)
        let layout = target.letterboxLayout(inputWidth: srcWidth, inputHeight: srcHeight)

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        format.prefersExtendedRange = false

        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let resultImage = renderer.image { ctx in
            if let lb = layout {
                UIColor(white: 0.5, alpha: 1.0).setFill()
                ctx.fill(CGRect(origin: .zero, size: targetSize))
                srcImage.draw(in: CGRect(x: lb.innerX, y: lb.innerY,
                                         width: lb.innerWidth, height: lb.innerHeight))
            } else {
                srcImage.draw(in: CGRect(origin: .zero, size: targetSize))
            }
        }

        if needsUnlock { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let cgImage = resultImage.cgImage
        let tensor: [Float]? = cgImage.map { cgImageToRGBTensor($0) }

        let elapsed = CACurrentMediaTime() - start
        return DownsampleOutput(image: cgImage, rgbTensor: tensor,
                                processingTime: elapsed, gpuTime: nil,
                                outputWidth: dstWidth, outputHeight: dstHeight)
    }

    private func fail(_ start: TimeInterval) -> DownsampleOutput {
        DownsampleOutput(image: nil, rgbTensor: nil,
                         processingTime: CACurrentMediaTime() - start,
                         gpuTime: nil, outputWidth: 0, outputHeight: 0)
    }
}
