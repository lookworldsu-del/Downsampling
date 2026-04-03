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

        let (dstWidth, dstHeight) = target.outputSize(inputWidth: srcWidth, inputHeight: srcHeight)
        let dstBytesPerRow = dstWidth * 4

        var srcBuffer = vImage_Buffer(
            data: srcBase,
            height: vImagePixelCount(srcHeight),
            width: vImagePixelCount(srcWidth),
            rowBytes: srcBytesPerRow
        )

        guard let dstData = malloc(dstHeight * dstBytesPerRow) else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - start, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        var dstBuffer = vImage_Buffer(
            data: dstData,
            height: vImagePixelCount(dstHeight),
            width: vImagePixelCount(dstWidth),
            rowBytes: dstBytesPerRow
        )

        let error = vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(kvImageHighQualityResampling))

        var image: CGImage?
        if error == kvImageNoError {
            image = createCGImage(from: &dstBuffer, width: dstWidth, height: dstHeight, bytesPerRow: dstBytesPerRow)
        }

        free(dstData)

        let elapsed = CACurrentMediaTime() - start
        return DownsampleOutput(image: image, processingTime: elapsed, gpuTime: nil, outputWidth: dstWidth, outputHeight: dstHeight)
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
