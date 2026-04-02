import UIKit

final class LineChartView: UIView {

    struct DataSet {
        let label: String
        let color: UIColor
        var values: [Double]
    }

    var dataSets: [DataSet] = [] {
        didSet { setNeedsDisplay() }
    }

    var yAxisLabel: String = "ms"
    var maxVisiblePoints: Int = 120

    private let legendHeight: CGFloat = 24
    private let padding = UIEdgeInsets(top: 8, left: 40, bottom: 24, right: 12)

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .systemBackground
        isOpaque = true
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        let chartRect = CGRect(
            x: padding.left,
            y: padding.top,
            width: bounds.width - padding.left - padding.right,
            height: bounds.height - padding.top - padding.bottom - legendHeight
        )

        drawGrid(ctx: ctx, rect: chartRect)
        drawLines(ctx: ctx, rect: chartRect)
        drawLegend(ctx: ctx, rect: chartRect)
    }

    private func drawGrid(ctx: CGContext, rect: CGRect) {
        ctx.setStrokeColor(UIColor.separator.cgColor)
        ctx.setLineWidth(0.5)

        let gridLines = 4
        for i in 0...gridLines {
            let y = rect.minY + rect.height * CGFloat(i) / CGFloat(gridLines)
            ctx.move(to: CGPoint(x: rect.minX, y: y))
            ctx.addLine(to: CGPoint(x: rect.maxX, y: y))
        }
        ctx.strokePath()

        let allValues = dataSets.flatMap { $0.values }
        let maxVal = allValues.max() ?? 1
        let attrs: [NSAttributedString.Key: Any] = [
            .font: UIFont.monospacedDigitSystemFont(ofSize: 9, weight: .regular),
            .foregroundColor: UIColor.secondaryLabel
        ]
        for i in 0...gridLines {
            let val = maxVal * Double(gridLines - i) / Double(gridLines)
            let y = rect.minY + rect.height * CGFloat(i) / CGFloat(gridLines)
            let str = String(format: "%.1f", val) as NSString
            str.draw(at: CGPoint(x: 2, y: y - 6), withAttributes: attrs)
        }
    }

    private func drawLines(ctx: CGContext, rect: CGRect) {
        let allValues = dataSets.flatMap { $0.values }
        guard !allValues.isEmpty else { return }
        let maxVal = max(allValues.max() ?? 1, 0.001)

        for ds in dataSets {
            guard ds.values.count > 1 else { continue }
            let count = ds.values.count

            ctx.setStrokeColor(ds.color.cgColor)
            ctx.setLineWidth(1.5)
            ctx.setLineJoin(.round)

            let xStep = rect.width / CGFloat(max(maxVisiblePoints - 1, 1))

            for (i, val) in ds.values.enumerated() {
                let x = rect.minX + CGFloat(i) * xStep
                let y = rect.maxY - CGFloat(val / maxVal) * rect.height
                if i == 0 {
                    ctx.move(to: CGPoint(x: x, y: y))
                } else {
                    ctx.addLine(to: CGPoint(x: x, y: y))
                }
            }
            ctx.strokePath()
        }
    }

    private func drawLegend(ctx: CGContext, rect: CGRect) {
        let y = rect.maxY + 8
        var x: CGFloat = rect.minX

        let attrs: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 10, weight: .medium),
            .foregroundColor: UIColor.label
        ]

        for ds in dataSets {
            ctx.setFillColor(ds.color.cgColor)
            ctx.fill(CGRect(x: x, y: y + 2, width: 10, height: 10))
            x += 14

            let str = ds.label as NSString
            str.draw(at: CGPoint(x: x, y: y), withAttributes: attrs)
            x += str.size(withAttributes: attrs).width + 16
        }
    }
}
