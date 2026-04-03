import Accelerate
import CoreVideo
import CoreGraphics
import QuartzCore

final class VImageDownsampler: Downsampler {

    let id: DownsamplerID = .vImage
    let name = "vImage (Accelerate)"
    let type: DownsamplerType = .cpu

    func downsample(_ pixelBuffer: CVPixelBuffer, target: DownsampleTarget) -> DownsampleOutput {
        let start = CACurrentMediaTime()

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let srcWidth = CVPixelBufferGetWidth(pixelBuffer)
        let srcHeight = CVPixelBufferGetHeight(pixelBuffer)
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - start, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        let layout = target.letterboxLayout(inputWidth: srcWidth, inputHeight: srcHeight)
        let scaleW = layout?.innerWidth ?? target.outputSize(inputWidth: srcWidth, inputHeight: srcHeight).width
        let scaleH = layout?.innerHeight ?? target.outputSize(inputWidth: srcWidth, inputHeight: srcHeight).height

        var srcBuffer = vImage_Buffer(
            data: srcBase,
            height: vImagePixelCount(srcHeight),
            width: vImagePixelCount(srcWidth),
            rowBytes: srcBytesPerRow
        )

        let dstBytesPerRow = scaleW * 4
        guard let dstData = malloc(scaleH * dstBytesPerRow) else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - start, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        var dstBuffer = vImage_Buffer(
            data: dstData, height: vImagePixelCount(scaleH),
            width: vImagePixelCount(scaleW), rowBytes: dstBytesPerRow
        )

        let error = vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(kvImageHighQualityResampling))

        var image: CGImage?
        if error == kvImageNoError {
            if let lb = layout {
                if let inner = createCGImage(from: &dstBuffer, width: scaleW, height: scaleH, bytesPerRow: dstBytesPerRow) {
                    image = letterboxComposite(innerImage: inner, layout: lb)
                }
            } else {
                image = createCGImage(from: &dstBuffer, width: scaleW, height: scaleH, bytesPerRow: dstBytesPerRow)
            }
        }

        free(dstData)

        let outW = layout?.canvasWidth ?? scaleW
        let outH = layout?.canvasHeight ?? scaleH
        let elapsed = CACurrentMediaTime() - start
        return DownsampleOutput(image: image, processingTime: elapsed, gpuTime: nil, outputWidth: outW, outputHeight: outH)
    }

    private func createCGImage(from buffer: inout vImage_Buffer, width: Int, height: Int, bytesPerRow: Int) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: buffer.data,
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
