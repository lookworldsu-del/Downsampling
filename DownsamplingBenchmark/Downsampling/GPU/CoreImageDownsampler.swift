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
        let layout = target.letterboxLayout(inputWidth: srcWidth, inputHeight: srcHeight)

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        guard let filter = CIFilter(name: "CILanczosScaleTransform") else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }

        let finalImage: CIImage

        if let lb = layout {
            let scaleY = Float(lb.innerHeight) / Float(srcHeight)
            let scaleX = Float(lb.innerWidth) / Float(srcWidth)
            let aspectRatio = scaleX / scaleY

            filter.setValue(ciImage, forKey: kCIInputImageKey)
            filter.setValue(scaleY, forKey: kCIInputScaleKey)
            filter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)

            guard let scaled = filter.outputImage else {
                return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart, gpuTime: nil, outputWidth: 0, outputHeight: 0)
            }

            let translated = scaled.transformed(by: CGAffineTransform(translationX: CGFloat(lb.innerX), y: CGFloat(lb.innerY)))
            let grayBg = CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5))
                .cropped(to: CGRect(x: 0, y: 0, width: dstWidth, height: dstHeight))
            finalImage = translated.composited(over: grayBg)
        } else {
            let scaleY = Float(dstHeight) / Float(srcHeight)
            let scaleX = Float(dstWidth) / Float(srcWidth)
            let aspectRatio = scaleX / scaleY

            filter.setValue(ciImage, forKey: kCIInputImageKey)
            filter.setValue(scaleY, forKey: kCIInputScaleKey)
            filter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)

            guard let output = filter.outputImage else {
                return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart, gpuTime: nil, outputWidth: 0, outputHeight: 0)
            }
            finalImage = output
        }

        let outputRect = CGRect(x: 0, y: 0, width: dstWidth, height: dstHeight)

        let gpuStart = CACurrentMediaTime()
        guard let cgImage = ciContext.createCGImage(finalImage, from: outputRect) else {
            return DownsampleOutput(image: nil, processingTime: CACurrentMediaTime() - wallStart, gpuTime: nil, outputWidth: 0, outputHeight: 0)
        }
        let gpuTime = CACurrentMediaTime() - gpuStart

        let wallTime = CACurrentMediaTime() - wallStart
        return DownsampleOutput(image: cgImage, processingTime: wallTime, gpuTime: gpuTime, outputWidth: dstWidth, outputHeight: dstHeight)
    }
}
