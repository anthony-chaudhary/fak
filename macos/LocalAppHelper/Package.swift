// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FakLocalApp",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "FakLocalAppSDK", targets: ["FakLocalAppSDK"]),
        .executable(name: "FakLocalAppHelper", targets: ["FakLocalAppHelper"]),
    ],
    targets: [
        .target(name: "FakLocalAppSDK"),
        .executableTarget(name: "FakLocalAppHelper", dependencies: ["FakLocalAppSDK"]),
    ]
)
