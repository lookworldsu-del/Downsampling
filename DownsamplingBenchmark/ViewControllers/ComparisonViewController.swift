import UIKit
import AVFoundation

final class ComparisonViewController: UIViewController {

    private let cameraManager = CameraManager()
    private let benchmarkEngine = BenchmarkEngine()

    private var cpuDownsampler: Downsampler = VImageDownsampler()
    private var gpuDownsampler: Downsampler?

    private let cpuImageView = UIImageView()
    private let gpuImageView = UIImageView()
    private let cpuOverlay = MetricsOverlayView()
    private let gpuOverlay = MetricsOverlayView()
    private let systemBar = UIView()
    private let thermalLabel = UILabel()
    private let batteryLabel = UILabel()
    private let droppedLabel = UILabel()

    private var scaleFactor: Float = 0.25
    private var isProcessing = false

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupDownsamplers()
        setupUI()
        setupCamera()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        loadSettings()
        benchmarkEngine.reset()
        cameraManager.start()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cameraManager.stop()
    }

    // MARK: - Setup

    private func setupDownsamplers() {
        gpuDownsampler = MetalDownsampler() ?? nil
    }

    private func setupUI() {
        cpuImageView.contentMode = .scaleAspectFit
        gpuImageView.contentMode = .scaleAspectFit
        cpuImageView.backgroundColor = .black
        gpuImageView.backgroundColor = .black

        cpuOverlay.title = "CPU"
        gpuOverlay.title = "GPU"

        [cpuImageView, gpuImageView, cpuOverlay, gpuOverlay].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview($0)
        }

        let divider = UIView()
        divider.backgroundColor = .darkGray
        divider.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(divider)

        setupSystemBar()

        let safeArea = view.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            cpuImageView.topAnchor.constraint(equalTo: safeArea.topAnchor),
            cpuImageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            cpuImageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            cpuImageView.heightAnchor.constraint(equalTo: safeArea.heightAnchor, multiplier: 0.44),

            divider.topAnchor.constraint(equalTo: cpuImageView.bottomAnchor),
            divider.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            divider.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            divider.heightAnchor.constraint(equalToConstant: 1),

            gpuImageView.topAnchor.constraint(equalTo: divider.bottomAnchor),
            gpuImageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            gpuImageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            gpuImageView.heightAnchor.constraint(equalTo: cpuImageView.heightAnchor),

            cpuOverlay.topAnchor.constraint(equalTo: cpuImageView.topAnchor, constant: 8),
            cpuOverlay.leadingAnchor.constraint(equalTo: cpuImageView.leadingAnchor, constant: 8),

            gpuOverlay.topAnchor.constraint(equalTo: gpuImageView.topAnchor, constant: 8),
            gpuOverlay.leadingAnchor.constraint(equalTo: gpuImageView.leadingAnchor, constant: 8),

            systemBar.topAnchor.constraint(equalTo: gpuImageView.bottomAnchor, constant: 4),
            systemBar.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 8),
            systemBar.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -8),
            systemBar.bottomAnchor.constraint(lessThanOrEqualTo: safeArea.bottomAnchor, constant: -4),
        ])
    }

    private func setupSystemBar() {
        systemBar.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(systemBar)

        [thermalLabel, batteryLabel, droppedLabel].forEach {
            $0.font = .monospacedDigitSystemFont(ofSize: 11, weight: .medium)
            $0.textColor = .lightGray
            $0.translatesAutoresizingMaskIntoConstraints = false
            systemBar.addSubview($0)
        }

        NSLayoutConstraint.activate([
            thermalLabel.leadingAnchor.constraint(equalTo: systemBar.leadingAnchor),
            thermalLabel.centerYAnchor.constraint(equalTo: systemBar.centerYAnchor),

            batteryLabel.centerXAnchor.constraint(equalTo: systemBar.centerXAnchor),
            batteryLabel.centerYAnchor.constraint(equalTo: systemBar.centerYAnchor),

            droppedLabel.trailingAnchor.constraint(equalTo: systemBar.trailingAnchor),
            droppedLabel.centerYAnchor.constraint(equalTo: systemBar.centerYAnchor),

            systemBar.heightAnchor.constraint(equalToConstant: 20),
        ])
    }

    private func setupCamera() {
        cameraManager.delegate = self
        cameraManager.configure(preset: .hd1080p)
    }

    // MARK: - Settings

    func loadSettings() {
        let defaults = UserDefaults.standard

        let cpuID = DownsamplerID(rawValue: defaults.string(forKey: "cpuAlgorithm") ?? "") ?? .vImage
        switch cpuID {
        case .vImage: cpuDownsampler = VImageDownsampler()
        case .cgContext: cpuDownsampler = CGContextDownsampler()
        default: cpuDownsampler = VImageDownsampler()
        }

        let gpuID = DownsamplerID(rawValue: defaults.string(forKey: "gpuAlgorithm") ?? "") ?? .metalCompute
        switch gpuID {
        case .metalCompute: gpuDownsampler = MetalDownsampler()
        case .mps: gpuDownsampler = MPSDownsampler()
        default: gpuDownsampler = MetalDownsampler()
        }

        scaleFactor = defaults.float(forKey: "scaleFactor")
        if scaleFactor <= 0 { scaleFactor = 0.25 }

        let presetRaw = defaults.string(forKey: "cameraPreset") ?? ""
        let preset: CameraManager.Preset = presetRaw == "4K" ? .uhd4K : .hd1080p
        cameraManager.configure(preset: preset)

        cpuOverlay.title = "CPU: \(cpuDownsampler.name)"
        gpuOverlay.title = "GPU: \(gpuDownsampler?.name ?? "N/A")"

        benchmarkEngine.reset()
    }

    // MARK: - Processing

    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard !isProcessing else { return }
        isProcessing = true

        let sf = scaleFactor
        let cpuDS = cpuDownsampler
        let gpuDS = gpuDownsampler

        let cpuResult = cpuDS.downsample(pixelBuffer, scaleFactor: sf)
        let gpuResult = gpuDS?.downsample(pixelBuffer, scaleFactor: sf)

        benchmarkEngine.recordCPU(processingTime: cpuResult.processingTime)
        if let gr = gpuResult {
            benchmarkEngine.recordGPU(processingTime: gr.processingTime, gpuTime: gr.gpuTime)
        }

        let cpuAgg = benchmarkEngine.aggregatedCPU(
            droppedFrameRatio: cameraManager.droppedFrameRatio,
            totalFrames: cameraManager.totalFrames
        )
        let gpuAgg = benchmarkEngine.aggregatedGPU(
            droppedFrameRatio: cameraManager.droppedFrameRatio,
            totalFrames: cameraManager.totalFrames
        )

        let cpuImg = cpuResult.image.map { UIImage(cgImage: $0) }
        let gpuImg = gpuResult?.image.map { UIImage(cgImage: $0) }

        let thermal = ProcessInfo.processInfo.thermalState
        let battery = UIDevice.current.batteryLevel
        let dropRatio = cameraManager.droppedFrameRatio

        DispatchQueue.main.async { [weak self] in
            self?.cpuImageView.image = cpuImg
            self?.gpuImageView.image = gpuImg

            self?.cpuOverlay.update(
                processingTime: cpuResult.processingTime,
                fps: cpuAgg.fps,
                width: cpuResult.outputWidth,
                height: cpuResult.outputHeight
            )

            if let gr = gpuResult {
                self?.gpuOverlay.update(
                    processingTime: gr.processingTime,
                    fps: gpuAgg.fps,
                    width: gr.outputWidth,
                    height: gr.outputHeight,
                    gpuTime: gr.gpuTime
                )
            }

            self?.thermalLabel.text = "\(thermal.emoji) \(thermal.displayName)"
            self?.batteryLabel.text = battery >= 0 ? String(format: "🔋 %.0f%%", battery * 100) : "🔋 N/A"
            self?.droppedLabel.text = String(format: "丢帧: %.1f%%", dropRatio * 100)

            self?.isProcessing = false
        }
    }
}

extension ComparisonViewController: CameraManagerDelegate {
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        processFrame(pixelBuffer)
    }

    func cameraManager(_ manager: CameraManager, didDrop sampleBuffer: CMSampleBuffer, reason: String) {}
}
