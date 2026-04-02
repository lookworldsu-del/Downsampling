import UIKit

final class MetricsOverlayView: UIView {

    private let titleLabel = UILabel()
    private let timeLabel = UILabel()
    private let fpsLabel = UILabel()
    private let resolutionLabel = UILabel()
    private let gpuTimeLabel = UILabel()
    private let stackView = UIStackView()

    var title: String = "" {
        didSet { titleLabel.text = title }
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupUI() {
        backgroundColor = UIColor.black.withAlphaComponent(0.6)
        layer.cornerRadius = 8
        clipsToBounds = true

        titleLabel.font = .systemFont(ofSize: 14, weight: .bold)
        titleLabel.textColor = .white

        [timeLabel, fpsLabel, resolutionLabel, gpuTimeLabel].forEach {
            $0.font = .monospacedDigitSystemFont(ofSize: 12, weight: .medium)
            $0.textColor = .white
        }

        gpuTimeLabel.isHidden = true

        stackView.axis = .vertical
        stackView.spacing = 2
        stackView.alignment = .leading
        [titleLabel, timeLabel, fpsLabel, resolutionLabel, gpuTimeLabel].forEach {
            stackView.addArrangedSubview($0)
        }

        addSubview(stackView)
        stackView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            stackView.topAnchor.constraint(equalTo: topAnchor, constant: 6),
            stackView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            stackView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            stackView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -6),
        ])
    }

    func update(processingTime: TimeInterval, fps: Double, width: Int, height: Int, gpuTime: TimeInterval? = nil) {
        timeLabel.text = String(format: "耗时: %.2f ms", processingTime * 1000)
        fpsLabel.text = String(format: "FPS: %.1f", fps)
        resolutionLabel.text = "输出: \(width)×\(height)"

        if let gt = gpuTime {
            gpuTimeLabel.isHidden = false
            gpuTimeLabel.text = String(format: "GPU: %.2f ms", gt * 1000)
        } else {
            gpuTimeLabel.isHidden = true
        }

        let color: UIColor = processingTime < 0.008 ? .systemGreen :
                              processingTime < 0.016 ? .systemYellow : .systemRed
        timeLabel.textColor = color
    }
}
