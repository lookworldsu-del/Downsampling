import CoreImage
import CoreVideo
import CoreGraphics
import QuartzCore

final class CoreImageDownsampler: Downsampler {

    let id: DownsamplerID = .coreImage
    let name = "Core Image (CILanczos)"
    let type: DownsamplerType = .gpu

    private let ciContext: CIContext

    init() {
        let options: [CIContextOption: Any] = [
            .useSoftwareRenderer: false,
            .cacheIntermediates: false,
            .priorityRequestLow: false,
        ]
        self.ciContext = CIContext(options: options)
    }

    func downsample(_ pixelBuffer: CVPixelBuffer, target: DownsampleTarget) -> DownsampleOutput {
        let wallStart = CACurrentMediaTime()

        let srcWidth = CVPixelBufferGetWidth(pixelBuffer)
        let srcHeight = CVPixelBufferGetHeight(pixelBuffer)
        let (dstWidth, dstHeight) = target.outputSize(inputWidth: srcWidth, inputHeight: srcHeight)

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        guard let filter = CIFilter(name: "CILanczosScaleTransform") else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        let scaleY = Float(dstHeight) / Float(srcHeight)
        let scaleX = Float(dstWidth) / Float(srcWidth)
        let aspectRatio = scaleX / scaleY

        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(scaleY, forKey: kCIInputScaleKey)
        filter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)

        guard let outputImage = filter.outputImage else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        let outputRect = CGRect(x: 0, y: 0, width: dstWidth, height: dstHeight)

        let gpuStart = CACurrentMediaTime()
        guard let cgImage = ciContext.createCGImage(outputImage, from: outputRect) else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }
        let gpuTime = CACurrentMediaTime() - gpuStart

        let wallTime = CACurrentMediaTime() - wallStart
        return DownsampleOutput(image: cgImage, processingTime: wallTime, gpuTime: gpuTime, outputWidth: dstWidth, outputHeight: dstHeight)
    }
}
