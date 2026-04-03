import UIKit

final class SettingsViewController: UIViewController {

    private let tableView = UITableView(frame: .zero, style: .insetGrouped)

    private enum Section: Int, CaseIterable {
        case cpuAlgorithm
        case gpuAlgorithm
        case scaleFactor
        case cameraPreset
    }

    private let cpuOptions: [DownsamplerID] = [.vImage, .cgContext, .uiGraphics]
    private let gpuOptions: [DownsamplerID] = [.metalCompute, .mps, .coreImage]
    private let targetOptions: [DownsampleTarget] = DownsampleTarget.allPresets
    private let presetOptions: [CameraManager.Preset] = [.hd1080p, .uhd4K]

    private var selectedCPU: DownsamplerID = .vImage
    private var selectedGPU: DownsamplerID = .metalCompute
    private var selectedTarget: DownsampleTarget = .scale(0.25)
    private var selectedPreset: CameraManager.Preset = .hd1080p

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        loadCurrentSettings()
        setupTableView()
    }

    private func loadCurrentSettings() {
        let defaults = UserDefaults.standard
        selectedCPU = DownsamplerID(rawValue: defaults.string(forKey: "cpuAlgorithm") ?? "") ?? .vImage
        selectedGPU = DownsamplerID(rawValue: defaults.string(forKey: "gpuAlgorithm") ?? "") ?? .metalCompute

        if let key = defaults.string(forKey: "downsampleTarget"),
           let t = DownsampleTarget.from(persistenceKey: key) {
            selectedTarget = t
        } else {
            let sf = defaults.float(forKey: "scaleFactor")
            selectedTarget = sf > 0 ? .scale(sf) : .scale(0.25)
        }

        let preset = defaults.string(forKey: "cameraPreset") ?? ""
        selectedPreset = preset == "4K" ? .uhd4K : .hd1080p
    }

    private func saveSettings() {
        let defaults = UserDefaults.standard
        defaults.set(selectedCPU.rawValue, forKey: "cpuAlgorithm")
        defaults.set(selectedGPU.rawValue, forKey: "gpuAlgorithm")
        defaults.set(selectedTarget.persistenceKey, forKey: "downsampleTarget")
        defaults.set(selectedPreset.displayName, forKey: "cameraPreset")
        defaults.synchronize()
    }

    private func setupTableView() {
        tableView.dataSource = self
        tableView.delegate = self
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        tableView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(tableView)

        let safeArea = view.safeAreaLayoutGuide
        NSLayoutConstraint.activate([
            tableView.topAnchor.constraint(equalTo: safeArea.topAnchor),
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tableView.bottomAnchor.constraint(equalTo: safeArea.bottomAnchor),
        ])
    }
}

extension SettingsViewController: UITableViewDataSource, UITableViewDelegate {

    func numberOfSections(in tableView: UITableView) -> Int {
        Section.allCases.count
    }

    func tableView(_ tableView: UITableView, titleForHeaderInSection section: Int) -> String? {
        switch Section(rawValue: section)! {
        case .cpuAlgorithm: return "CPU 算法"
        case .gpuAlgorithm: return "GPU 算法"
        case .scaleFactor: return "降采样目标"
        case .cameraPreset: return "采集分辨率"
        }
    }

    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        switch Section(rawValue: section)! {
        case .cpuAlgorithm: return cpuOptions.count
        case .gpuAlgorithm: return gpuOptions.count
        case .scaleFactor: return targetOptions.count
        case .cameraPreset: return presetOptions.count
        }
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        cell.tintColor = .systemBlue

        switch Section(rawValue: indexPath.section)! {
        case .cpuAlgorithm:
            let opt = cpuOptions[indexPath.row]
            cell.textLabel?.text = opt.rawValue
            cell.accessoryType = opt == selectedCPU ? .checkmark : .none

        case .gpuAlgorithm:
            let opt = gpuOptions[indexPath.row]
            cell.textLabel?.text = opt.rawValue
            cell.accessoryType = opt == selectedGPU ? .checkmark : .none

        case .scaleFactor:
            let opt = targetOptions[indexPath.row]
            cell.textLabel?.text = opt.displayName
            cell.accessoryType = opt == selectedTarget ? .checkmark : .none

        case .cameraPreset:
            let opt = presetOptions[indexPath.row]
            cell.textLabel?.text = opt.displayName
            cell.accessoryType = opt == selectedPreset ? .checkmark : .none
        }

        return cell
    }

    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)

        switch Section(rawValue: indexPath.section)! {
        case .cpuAlgorithm:
            selectedCPU = cpuOptions[indexPath.row]
        case .gpuAlgorithm:
            selectedGPU = gpuOptions[indexPath.row]
        case .scaleFactor:
            selectedTarget = targetOptions[indexPath.row]
        case .cameraPreset:
            selectedPreset = presetOptions[indexPath.row]
        }

        saveSettings()
        tableView.reloadSections(IndexSet(integer: indexPath.section), with: .none)
    }
}
