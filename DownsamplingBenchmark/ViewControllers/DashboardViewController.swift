import UIKit
import AVFoundation

final class DashboardViewController: UIViewController {

    private let benchmarkEngine = BenchmarkEngine()
    private let cameraManager = CameraManager()

    private var cpuDownsampler: Downsampler = VImageDownsampler()
    private var gpuDownsampler: Downsampler?
    private var scaleFactor: Float = 0.25
    private var isProcessing = false

    private let chartView = LineChartView()
    private let scrollView = UIScrollView()
    private let contentStack = UIStackView()

    private let cpuMetricsCard = MetricCardView()
    private let gpuMetricsCard = MetricCardView()
    private let systemMetricsCard = MetricCardView()

    // MARK: - Full Benchmark

    private let fullBenchmarkButton = UIButton(type: .system)
    private let exportButton = UIButton(type: .system)
    private let progressLabel = UILabel()
    private var fullRunner: FullBenchmarkRunner?
    private var lastReport: FullBenchmarkReport?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
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

    private func setupDownsamplers() {
        gpuDownsampler = MetalDownsampler()
    }

    private func setupUI() {
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(scrollView)

        contentStack.axis = .vertical
        contentStack.spacing = 12
        contentStack.translatesAutoresizingMaskIntoConstraints = false
        scrollView.addSubview(contentStack)

        chartView.translatesAutoresizingMaskIntoConstraints = false
        chartView.layer.cornerRadius = 8
        chartView.clipsToBounds = true

        let chartContainer = UIView()
        chartContainer.translatesAutoresizingMaskIntoConstraints = false
        chartContainer.addSubview(chartView)

        let chartTitle = UILabel()
        chartTitle.text = "帧处理耗时趋势 (ms)"
        chartTitle.font = .systemFont(ofSize: 15, weight: .semibold)
        chartTitle.translatesAutoresizingMaskIntoConstraints = false
        chartContainer.addSubview(chartTitle)

        NSLayoutConstraint.activate([
            chartTitle.topAnchor.constraint(equalTo: chartContainer.topAnchor),
            chartTitle.leadingAnchor.constraint(equalTo: chartContainer.leadingAnchor),
            chartView.topAnchor.constraint(equalTo: chartTitle.bottomAnchor, constant: 4),
            chartView.leadingAnchor.constraint(equalTo: chartContainer.leadingAnchor),
            chartView.trailingAnchor.constraint(equalTo: chartContainer.trailingAnchor),
            chartView.bottomAnchor.constraint(equalTo: chartContainer.bottomAnchor),
            chartView.heightAnchor.constraint(equalToConstant: 200),
        ])

        contentStack.addArrangedSubview(chartContainer)

        cpuMetricsCard.titleText = "CPU 指标"
        gpuMetricsCard.titleText = "GPU 指标"
        systemMetricsCard.titleText = "系统指标"

        [cpuMetricsCard, gpuMetricsCard, systemMetricsCard].forEach {
            contentStack.addArrangedSubview($0)
        }

        setupFullBenchmarkUI()

        let safeArea = view.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: safeArea.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: safeArea.bottomAnchor),

            contentStack.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 12),
            contentStack.leadingAnchor.constraint(equalTo: scrollView.leadingAnchor, constant: 16),
            contentStack.trailingAnchor.constraint(equalTo: scrollView.trailingAnchor, constant: -16),
            contentStack.bottomAnchor.constraint(equalTo: scrollView.bottomAnchor, constant: -12),
            contentStack.widthAnchor.constraint(equalTo: scrollView.widthAnchor, constant: -32),
        ])
    }

    private func setupFullBenchmarkUI() {
        let separator = UIView()
        separator.backgroundColor = .separator
        separator.translatesAutoresizingMaskIntoConstraints = false
        separator.heightAnchor.constraint(equalToConstant: 1).isActive = true
        contentStack.addArrangedSubview(separator)

        let titleLabel = UILabel()
        titleLabel.text = "全量跑分（6 种算法对比）"
        titleLabel.font = .systemFont(ofSize: 16, weight: .bold)
        contentStack.addArrangedSubview(titleLabel)

        fullBenchmarkButton.setTitle("  开始全量跑分", for: .normal)
        fullBenchmarkButton.setImage(UIImage(systemName: "play.circle.fill"), for: .normal)
        fullBenchmarkButton.titleLabel?.font = .systemFont(ofSize: 15, weight: .semibold)
        fullBenchmarkButton.backgroundColor = .systemBlue
        fullBenchmarkButton.setTitleColor(.white, for: .normal)
        fullBenchmarkButton.tintColor = .white
        fullBenchmarkButton.layer.cornerRadius = 10
        fullBenchmarkButton.translatesAutoresizingMaskIntoConstraints = false
        fullBenchmarkButton.heightAnchor.constraint(equalToConstant: 48).isActive = true
        fullBenchmarkButton.addTarget(self, action: #selector(startFullBenchmark), for: .touchUpInside)
        contentStack.addArrangedSubview(fullBenchmarkButton)

        progressLabel.font = .monospacedDigitSystemFont(ofSize: 13, weight: .medium)
        progressLabel.textColor = .secondaryLabel
        progressLabel.textAlignment = .center
        progressLabel.isHidden = true
        contentStack.addArrangedSubview(progressLabel)

        exportButton.setTitle("  复制 JSON 到剪贴板", for: .normal)
        exportButton.setImage(UIImage(systemName: "doc.on.clipboard"), for: .normal)
        exportButton.titleLabel?.font = .systemFont(ofSize: 15, weight: .semibold)
        exportButton.backgroundColor = .systemGreen
        exportButton.setTitleColor(.white, for: .normal)
        exportButton.tintColor = .white
        exportButton.layer.cornerRadius = 10
        exportButton.translatesAutoresizingMaskIntoConstraints = false
        exportButton.heightAnchor.constraint(equalToConstant: 48).isActive = true
        exportButton.addTarget(self, action: #selector(exportJSON), for: .touchUpInside)
        exportButton.isHidden = true
        contentStack.addArrangedSubview(exportButton)
    }

    @objc private func startFullBenchmark() {
        guard fullRunner == nil || !(fullRunner!.isRunning) else { return }

        fullBenchmarkButton.isEnabled = false
        fullBenchmarkButton.setTitle("  跑分中...", for: .normal)
        fullBenchmarkButton.backgroundColor = .systemGray
        exportButton.isHidden = true
        progressLabel.isHidden = false
        lastReport = nil

        let sf = scaleFactor
        let runner = FullBenchmarkRunner(scaleFactor: sf, framesToCollect: 60)

        runner.onProgress = { [weak self] msg in
            self?.progressLabel.text = msg
        }

        runner.onComplete = { [weak self] report in
            guard let self else { return }
            self.lastReport = report
            self.fullRunner = nil

            self.fullBenchmarkButton.isEnabled = true
            self.fullBenchmarkButton.setTitle("  重新跑分", for: .normal)
            self.fullBenchmarkButton.backgroundColor = .systemBlue
            self.progressLabel.text = "跑分完成！共测试 \(report.results.count) 种算法"
            self.exportButton.isHidden = false

            self.showResultsSummary(report)
        }

        fullRunner = runner
        runner.start()
    }

    @objc private func exportJSON() {
        guard let report = lastReport,
              let json = FullBenchmarkRunner.reportToJSON(report) else { return }

        UIPasteboard.general.string = json

        let alert = UIAlertController(
            title: "已复制到剪贴板",
            message: "JSON 数据（\(json.count) 字符）已复制，可直接粘贴给 AI 分析并写入 README。",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "好", style: .default))
        present(alert, animated: true)
    }

    private func showResultsSummary(_ report: FullBenchmarkReport) {
        let existingCards = contentStack.arrangedSubviews.filter { $0.tag == 999 }
        existingCards.forEach { $0.removeFromSuperview() }

        for r in report.results {
            let card = MetricCardView()
            card.tag = 999
            card.titleText = "\(r.type): \(r.name)"
            let lines = [
                String(format: "平均: %.2f ms  |  最小: %.2f ms  |  最大: %.2f ms", r.avgTime, r.minTime, r.maxTime),
                String(format: "P99: %.2f ms  |  方差: %.4f  |  FPS: %.1f", r.p99Time, r.variance, r.fps),
                String(format: "CPU占用: %.1f%%  |  峰值内存: %.1f MB", r.avgCPU, r.peakMemory),
            ]
            card.setDetail(lines.joined(separator: "\n"))
            contentStack.addArrangedSubview(card)
        }
    }

    private func setupCamera() {
        cameraManager.delegate = self
        cameraManager.configure(preset: .hd1080p)
    }

    func loadSettings() {
        let defaults = UserDefaults.standard
        let cpuID = DownsamplerID(rawValue: defaults.string(forKey: "cpuAlgorithm") ?? "") ?? .vImage
        switch cpuID {
        case .vImage: cpuDownsampler = VImageDownsampler()
        case .cgContext: cpuDownsampler = CGContextDownsampler()
        case .uiGraphics: cpuDownsampler = UIGraphicsDownsampler()
        default: cpuDownsampler = VImageDownsampler()
        }
        let gpuID = DownsamplerID(rawValue: defaults.string(forKey: "gpuAlgorithm") ?? "") ?? .metalCompute
        switch gpuID {
        case .metalCompute: gpuDownsampler = MetalDownsampler()
        case .mps: gpuDownsampler = MPSDownsampler()
        case .coreImage: gpuDownsampler = CoreImageDownsampler()
        default: gpuDownsampler = MetalDownsampler()
        }
        scaleFactor = defaults.float(forKey: "scaleFactor")
        if scaleFactor <= 0 { scaleFactor = 0.25 }

        let presetRaw = defaults.string(forKey: "cameraPreset") ?? ""
        let preset: CameraManager.Preset = presetRaw == "4K" ? .uhd4K : .hd1080p
        cameraManager.configure(preset: preset)
        benchmarkEngine.reset()
    }

    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        if let runner = fullRunner, runner.isRunning {
            runner.collectFrame(pixelBuffer)
            return
        }

        guard !isProcessing else { return }
        isProcessing = true

        let cpuResult = cpuDownsampler.downsample(pixelBuffer, scaleFactor: scaleFactor)
        let gpuResult = gpuDownsampler?.downsample(pixelBuffer, scaleFactor: scaleFactor)

        benchmarkEngine.recordCPU(processingTime: cpuResult.processingTime)
        if let gr = gpuResult {
            benchmarkEngine.recordGPU(processingTime: gr.processingTime, gpuTime: gr.gpuTime)
        }

        let cpuTimes = benchmarkEngine.cpuFrameTimes.map { $0 * 1000 }
        let gpuTimes = benchmarkEngine.gpuFrameTimes.map { $0 * 1000 }

        let cpuAgg = benchmarkEngine.aggregatedCPU(
            droppedFrameRatio: cameraManager.droppedFrameRatio,
            totalFrames: cameraManager.totalFrames
        )
        let gpuAgg = benchmarkEngine.aggregatedGPU(
            droppedFrameRatio: cameraManager.droppedFrameRatio,
            totalFrames: cameraManager.totalFrames
        )

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }

            self.chartView.dataSets = [
                LineChartView.DataSet(label: "CPU", color: .systemBlue, values: cpuTimes),
                LineChartView.DataSet(label: "GPU", color: .systemGreen, values: gpuTimes),
            ]

            self.cpuMetricsCard.update(with: cpuAgg, label: self.cpuDownsampler.name)
            self.gpuMetricsCard.update(with: gpuAgg, label: self.gpuDownsampler?.name ?? "N/A")
            self.systemMetricsCard.updateSystem(cpu: cpuAgg, gpu: gpuAgg)

            self.isProcessing = false
        }
    }
}

extension DashboardViewController: CameraManagerDelegate {
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        processFrame(pixelBuffer)
    }
    func cameraManager(_ manager: CameraManager, didDrop sampleBuffer: CMSampleBuffer, reason: String) {}
}

// MARK: - MetricCardView

final class MetricCardView: UIView {

    var titleText: String = "" {
        didSet { titleLabel.text = titleText }
    }

    private let titleLabel = UILabel()
    private let detailLabel = UILabel()

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .secondarySystemBackground
        layer.cornerRadius = 10

        titleLabel.font = .systemFont(ofSize: 14, weight: .bold)
        titleLabel.textColor = .label

        detailLabel.font = .monospacedDigitSystemFont(ofSize: 12, weight: .regular)
        detailLabel.textColor = .secondaryLabel
        detailLabel.numberOfLines = 0

        let stack = UIStackView(arrangedSubviews: [titleLabel, detailLabel])
        stack.axis = .vertical
        stack.spacing = 4
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)

        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 10),
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -12),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -10),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    func setDetail(_ text: String) {
        detailLabel.text = text
    }

    func update(with m: AggregatedMetrics, label: String) {
        titleLabel.text = "\(titleText) — \(label)"
        let lines = [
            String(format: "平均: %.2f ms  |  最小: %.2f ms  |  最大: %.2f ms", m.averageTime * 1000, m.minTime * 1000, m.maxTime * 1000),
            String(format: "P99: %.2f ms  |  方差: %.4f  |  FPS: %.1f", m.p99Time * 1000, m.variance * 1e6, m.fps),
            String(format: "CPU占用: %.1f%%  |  内存: %.1f MB", m.averageCPU, Double(m.currentMemory) / 1_048_576),
            String(format: "总帧数: %d  |  丢帧率: %.2f%%", m.totalFrames, m.droppedFrameRatio * 100),
        ]
        detailLabel.text = lines.joined(separator: "\n")
    }

    func updateSystem(cpu: AggregatedMetrics, gpu: AggregatedMetrics) {
        let thermal = ProcessInfo.processInfo.thermalState
        let battery = UIDevice.current.batteryLevel
        let latest = gpu.totalFrames > 0 ? gpu : cpu
        let lines = [
            "\(thermal.emoji) 热状态: \(thermal.displayName)",
            battery >= 0 ? String(format: "🔋 电量: %.0f%%  |  消耗: %.2f%%  |  速率: %.2f%%/min", battery * 100, latest.batteryDrain, latest.batteryDrainRate) : "🔋 电量: 模拟器不可用",
            String(format: "峰值内存: %.1f MB  |  可用: %.1f MB", Double(latest.peakMemory) / 1_048_576, Double(os_proc_available_memory()) / 1_048_576),
        ]
        detailLabel.text = lines.joined(separator: "\n")
    }
}
