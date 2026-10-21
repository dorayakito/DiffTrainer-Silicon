// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DiffTrainerNative",
    defaultLocalization: "en",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "DiffTrainer", targets: ["DiffTrainer"])],
    targets: [.executableTarget(name: "DiffTrainer", path: "Sources/DiffTrainer", resources: [.process("Resources")])]
)
