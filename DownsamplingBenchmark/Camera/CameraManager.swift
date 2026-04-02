import AVFoundation
import UIKit

protocol CameraManagerDelegate: AnyObject {
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime)
    func cameraManager(_ manager: CameraManager, didDrop sampleBuffer: CMSampleBuffer, reason: String)
}

final class CameraManager: NSObject {

    weak var delegate: CameraManagerDelegate?

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "com.downsamplingbenchmark.camera", qos: .userInteractive)

    private(set) var isRunning = false
    private(set) var totalFrames: Int = 0
    private(set) var droppedFrames: Int = 0

    var droppedFrameRatio: Double {
        guard totalFrames > 0 else { return 0 }
        return Double(droppedFrames) / Double(totalFrames)
    }

    enum Preset {
        case hd1080p
        case uhd4K

        var sessionPreset: AVCaptureSession.Preset {
            switch self {
            case .hd1080p: return .hd1920x1080
            case .uhd4K: return .hd4K3840x2160
            }
        }

        var displayName: String {
            switch self {
            case .hd1080p: return "1080p"
            case .uhd4K: return "4K"
            }
        }
    }

    func configure(preset: Preset = .hd1080p) {
        session.beginConfiguration()
        session.sessionPreset = preset.sessionPreset

        session.inputs.forEach { session.removeInput($0) }

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            session.commitConfiguration()
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = false
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)

        session.outputs.forEach { session.removeOutput($0) }
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        if let connection = videoOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
        }

        session.commitConfiguration()
    }

    func start() {
        guard !isRunning else { return }
        totalFrames = 0
        droppedFrames = 0
        processingQueue.async { [weak self] in
            self?.session.startRunning()
        }
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        processingQueue.async { [weak self] in
            self?.session.stopRunning()
        }
        isRunning = false
    }

    func makePreviewLayer() -> AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        return layer
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        totalFrames += 1
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        delegate?.cameraManager(self, didOutput: pixelBuffer, timestamp: timestamp)
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        droppedFrames += 1

        var reason = "unknown"
        if let attachment = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_DroppedFrameReason, attachmentModeOut: nil) {
            reason = attachment as? String ?? "unknown"
        }
        delegate?.cameraManager(self, didDrop: sampleBuffer, reason: reason)
    }
}
