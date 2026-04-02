import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {

    var window: UIWindow?

    func scene(
        _ scene: UIScene,
        willConnectTo session: UISceneSession,
        options connectionOptions: UIScene.ConnectionOptions
    ) {
        guard let windowScene = scene as? UIWindowScene else { return }

        let window = UIWindow(windowScene: windowScene)

        let tabBar = UITabBarController()
        tabBar.viewControllers = [
            makeNav(ComparisonViewController(), title: "实时对比", icon: "camera.viewfinder"),
            makeNav(DashboardViewController(), title: "性能仪表盘", icon: "chart.xyaxis.line"),
            makeNav(SettingsViewController(), title: "设置", icon: "gearshape"),
        ]
        tabBar.tabBar.backgroundColor = .systemBackground

        window.rootViewController = tabBar
        window.makeKeyAndVisible()
        self.window = window
    }

    private func makeNav(_ vc: UIViewController, title: String, icon: String) -> UINavigationController {
        vc.title = title
        vc.tabBarItem = UITabBarItem(title: title, image: UIImage(systemName: icon), selectedImage: nil)
        return UINavigationController(rootViewController: vc)
    }
}
