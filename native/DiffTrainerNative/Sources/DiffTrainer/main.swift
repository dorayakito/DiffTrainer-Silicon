import SwiftUI
import AppKit

enum Area: String, CaseIterable, Identifiable {
    case overview, preparation, configuration, training, export, inference
    var id: String { rawValue }
    var icon: String {
        switch self { case .overview: "square.grid.2x2"; case .preparation: "waveform.path"; case .configuration: "slider.horizontal.3"; case .training: "cpu"; case .export: "shippingbox"; case .inference: "waveform" }
    }
}

@MainActor final class AppModel: ObservableObject {
    @Published var area: Area = .overview
    @Published var checkout = UserDefaults.standard.string(forKey: "checkout") ?? ""
    @Published var python = UserDefaults.standard.string(forKey: "python") ?? ""
    @Published var language = UserDefaults.standard.string(forKey: "language") ?? "English"
    @Published var fields: [String: String] = [:]
    @Published var flags: [String: Bool] = [:]
    @Published var logs: [String] = []
    @Published var status: [String: Any] = [:]
    @Published var busy = false
    @Published var error: String?
    private var process: Process?

    func chooseCheckout() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url { checkout = url.path; UserDefaults.standard.set(checkout, forKey: "checkout"); refreshStatus() }
    }
    func choosePython() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.canChooseFiles = true
        if panel.runModal() == .OK, let url = panel.url { python = url.path; UserDefaults.standard.set(python, forKey: "python"); refreshStatus() }
    }
    func refreshStatus() { run("status", payload: [:]) }
    func runCurrent() {
        let op: String
        switch area {
        case .preparation: op = "prepare"
        case .configuration: op = fields["mode"] == "Dictionary" ? "dictionary" : "configure"
        case .training: op = fields["task"] == "Binarize" ? "binarize" : "train"
        case .export: op = flags["advanced"] == true ? "export_advanced" : "export_basic"
        case .inference: op = "inference"
        case .overview: return
        }
        var payload: [String: Any] = fields
        flags.forEach { payload[$0.key] = $0.value }
        run(op, payload: payload)
    }
    func cancel() { process?.terminate(); logs.append("Cancellation requested") }
    func run(_ operation: String, payload: [String: Any]) {
        guard !busy else { return }
        guard !checkout.isEmpty else { error = "Choose the repository checkout first."; return }
        let root = URL(fileURLWithPath: checkout)
        let interpreter = python.isEmpty ? root.appending(path: ".venv/bin/python3").path : python
        guard FileManager.default.fileExists(atPath: interpreter) else { error = "Python interpreter not found: \(interpreter)"; return }
        let bridge = root.appending(path: "src/difftrainer_app/bridge.py")
        guard FileManager.default.fileExists(atPath: bridge.path) else { error = "Python bridge not found: \(bridge.path)"; return }
        do {
            let request = try JSONSerialization.data(withJSONObject: ["operation": operation, "payload": payload])
            let task = Process(); task.executableURL = URL(fileURLWithPath: interpreter); task.arguments = [bridge.path]
            task.currentDirectoryURL = root
            var env = ProcessInfo.processInfo.environment; env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"; task.environment = env
            let input = Pipe(), output = Pipe(); task.standardInput = input; task.standardOutput = output; task.standardError = output
            busy = true; error = nil; logs = []; process = task
            task.terminationHandler = { [weak self] proc in
                Task { @MainActor in
                    self?.busy = false; self?.process = nil
                    if proc.terminationStatus != 0 && proc.terminationStatus != 15 { self?.logs.append("Bridge exited with status \(proc.terminationStatus)") }
                }
            }
            try task.run(); input.fileHandleForWriting.write(request + Data([10])); try? input.fileHandleForWriting.close()
            let handle = output.fileHandleForReading
            handle.readabilityHandler = { [weak self] pipe in
                let data = pipe.availableData
                if data.isEmpty { pipe.readabilityHandler = nil; return }
                let text = String(decoding: data, as: UTF8.self)
                Task { @MainActor in self?.consume(text) }
            }
        } catch { busy = false; self.error = error.localizedDescription }
    }
    private func consume(_ text: String) {
        for line in text.split(separator: "\n") {
            guard let data = String(line).data(using: .utf8), let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { logs.append(String(line)); continue }
            switch event["event"] as? String {
            case "log", "progress": logs.append(event["message"] as? String ?? "")
            case "result":
                let result = event["result"] as? [String: Any] ?? [:]
                if result["checkout"] != nil { status = result }
                if let content = result["content"] as? String { fields["content"] = content }
                logs.append("Completed: \(result.map { "\($0.key)=\($0.value)" }.joined(separator: " · "))")
            case "error": self.error = event["message"] as? String ?? "Operation failed"; logs.append(self.error ?? "")
            default: logs.append(String(line))
            }
        }
    }
}

@main struct DiffTrainerApp: App {
    @StateObject private var model = AppModel()
    var body: some Scene { WindowGroup { MainView().environmentObject(model).frame(minWidth: 980, minHeight: 680).task { if model.checkout.isEmpty { model.chooseCheckout() } else { model.refreshStatus() } } } }
}

struct MainView: View {
    @EnvironmentObject private var model: AppModel
    var body: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 8) {
                Label("DiffTrainer", systemImage: "waveform.badge.mic").font(.title2.bold()).padding(.bottom, 18)
                ForEach(Area.allCases) { area in
                    Button { model.area = area } label: { Label(tr(titleKey(area)), systemImage: area.icon).frame(maxWidth: .infinity, alignment: .leading).padding(8).background(model.area == area ? Color.accentColor.opacity(0.14) : .clear, in: RoundedRectangle(cornerRadius: 8)) }.buttonStyle(.plain)
                }
                Spacer()
                Picker("Language", selection: $model.language) { Text("English").tag("English"); Text("Português (Brasil)").tag("Português (Brasil)"); Text("Deutsch").tag("Deutsch") }.onChange(of: model.language) { _, v in UserDefaults.standard.set(v, forKey: "language") }
            }.padding().frame(minWidth: 210)
        } detail: {
            VStack(alignment: .leading, spacing: 18) {
                HStack { VStack(alignment: .leading) { Text(tr(titleKey(model.area))).font(.largeTitle.bold()); Text(tr(subtitleKey(model.area))).foregroundStyle(.secondary) }; Spacer(); if model.busy { ProgressView().controlSize(.small); Button("Cancel") { model.cancel() } } }
                if model.area == .overview { overview } else { operationForm }
                if let error = model.error { Label(error, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red).textSelection(.enabled) }
                GroupBox("Operation output") { ScrollView { VStack(alignment: .leading, spacing: 5) { ForEach(Array(model.logs.enumerated()), id: \.offset) { _, line in Text(line).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) } }.padding(6) }.frame(minHeight: 120) }
                Spacer(minLength: 0)
            }.padding(28)
        }.preferredColorScheme(nil)
    }
    private var overview: some View {
        VStack(alignment: .leading, spacing: 14) {
            GroupBox("Local environment") {
                VStack(alignment: .leading, spacing: 12) {
                    HStack { Text("Repository checkout"); Spacer(); Text(model.checkout.isEmpty ? "Not selected" : model.checkout).lineLimit(1).foregroundStyle(.secondary); Button("Choose…", action: model.chooseCheckout) }
                    HStack { Text("Python interpreter"); Spacer(); Text(model.python.isEmpty ? "Automatic (.venv/bin/python3)" : model.python).lineLimit(1).foregroundStyle(.secondary); Button("Choose…", action: model.choosePython) }
                    HStack { Text("DiffSinger"); Spacer(); Text((model.status["diffsinger"] as? Bool == true) ? "Available" : "Missing").foregroundStyle(model.status["diffsinger"] as? Bool == true ? .green : .orange) }
                    HStack { Text("Python"); Spacer(); Text("\(String(describing: model.status["python_version"] ?? "Not checked")) · \(String(describing: model.status["device"] ?? "Device unknown"))").foregroundStyle(.secondary) }
                    HStack { Text("Version"); Spacer(); Text(String(describing: model.status["app_version"] ?? "—")).foregroundStyle(.secondary) }
                }.padding(6)
            }
            HStack { Button("Refresh environment") { model.refreshStatus() }; Button("Update tools…") { model.error = "Use the existing Python interface to update tools. Review its replacement warning before starting an update." }.disabled(model.busy) }
            Text("Update tools can replace files inside DiffSinger/. Review your local models and checkpoints before proceeding.").font(.caption).foregroundStyle(.secondary)
        }
    }
    private var operationForm: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                switch model.area {
                case .preparation:
                    field("input_folder", "Raw data folder", "Folder containing audio and .lab files")
                    toggle("segment", "Segment long recordings")
                    if model.flags["segment"] == true { field("max_length", "Maximum segment length (seconds)", "5") }
                    toggle("estimate_notes", "Estimate notes with SOME")
                    toggle("make_ds", "Create .ds files (requires estimated notes)")
                    field("pitch_extractor", "Pitch extractor", "rmvpe")
                case .configuration:
                    Picker("Configuration task", selection: Binding(get: { model.fields["mode"] ?? "Create config" }, set: { model.fields["mode"] = $0 })) { Text("Create config").tag("Create config"); Text("Edit dictionary").tag("Dictionary") }
                    if model.fields["mode"] == "Dictionary" {
                        Picker("Dictionary action", selection: Binding(get: { model.fields["action"] ?? "read" }, set: { model.fields["action"] = $0 })) { Text("Read").tag("read"); Text("Save edits").tag("write") }
                        field("dictionary", "Dictionary file", "DiffSinger/dictionaries/en-phonemes.txt")
                        VStack(alignment: .leading) { Text("Dictionary content").font(.headline); TextEditor(text: Binding(get: { model.fields["content"] ?? "" }, set: { model.fields["content"] = $0 })).font(.system(.body, design: .monospaced)).frame(minHeight: 220).border(.quaternary) }
                    } else {
                        field("data_folder", "Prepared data folder", "Contains speaker folders and transcriptions.csv")
                        field("output_folder", "Experiment output folder", "DiffSinger/checkpoints/experiment")
                        field("model_type", "Model type", "acoustic or variance")
                        field("config_name", "Configuration name", "acoustic_native")
                    }
                case .training:
                    Picker("Task", selection: Binding(get: { model.fields["task"] ?? "Train" }, set: { model.fields["task"] = $0 })) { Text("Binarize").tag("Binarize"); Text("Train").tag("Train") }
                    field("config", "Configuration YAML", "DiffSinger/configs/acoustic.yaml")
                    field("experiment", "Experiment name", "experiment")
                case .export:
                    toggle("advanced", "Advanced export (acoustic, duration, variance and pitch)")
                    field("singer_name", "Singer name", "Singer")
                    field("output_folder", "OpenUtau export folder", "")
                    field("acoustic_checkpoint", "Acoustic checkpoint folder", "")
                    if model.flags["advanced"] == true { field("duration_checkpoint", "Duration checkpoint folder", ""); field("variance_checkpoint", "Variance checkpoint folder", ""); field("pitch_checkpoint", "Pitch checkpoint folder", "") } else { field("variance_checkpoint", "Optional variance checkpoint folder", "") }
                    field("vocoder", "Optional custom vocoder (.onnx)", "")
                    toggle("make_dictionary", "Generate dsdict files")
                case .inference:
                    field("input_ds", "Input .ds file", "")
                    field("acoustic_checkpoint", "Acoustic experiment folder", "")
                    field("speaker", "Speaker", "singer")
                    field("language", "Language code", "en")
                    field("output_folder", "Output folder", "")
                    field("title", "Output title", "native_inference")
                case .overview: EmptyView()
                }
                HStack { Button("Run operation") { model.runCurrent() }.keyboardShortcut(.defaultAction).disabled(model.busy); Button("Choose checkout…", action: model.chooseCheckout) }
            }.padding(.vertical, 4)
        }
    }
    private func field(_ key: String, _ label: String, _ hint: String) -> some View {
        let directoryFields: Set<String> = ["input_folder", "data_folder", "output_folder", "experiment", "acoustic_checkpoint", "duration_checkpoint", "variance_checkpoint", "pitch_checkpoint"]
        let fileFields: Set<String> = ["config", "input_ds", "vocoder", "dictionary"]
        return VStack(alignment: .leading, spacing: 4) {
            Text(tr(label)).font(.headline)
            HStack {
                TextField(hint, text: Binding(get: { model.fields[key] ?? "" }, set: { model.fields[key] = $0 })).textFieldStyle(.roundedBorder)
                if directoryFields.contains(key) || fileFields.contains(key) {
                    Button("Choose…") {
                        let panel = NSOpenPanel(); panel.canChooseDirectories = directoryFields.contains(key); panel.canChooseFiles = fileFields.contains(key)
                        if key == "config" { panel.allowedContentTypes = [.yaml] }
                        if key == "input_ds" { panel.allowedContentTypes = [.init(filenameExtension: "ds")!] }
                        if key == "vocoder" { panel.allowedContentTypes = [.init(filenameExtension: "onnx")!] }
                        if key == "dictionary" { panel.allowedContentTypes = [.plainText] }
                        if panel.runModal() == .OK, let url = panel.url { model.fields[key] = url.path }
                    }
                }
            }
        }
    }
    private func toggle(_ key: String, _ label: String) -> some View { Toggle(label, isOn: Binding(get: { model.flags[key] ?? false }, set: { model.flags[key] = $0 })) }
    private func titleKey(_ area: Area) -> String { switch area { case .overview: "Overview"; case .preparation: "Data Preparation"; case .configuration: "Configuration"; case .training: "Binarize & Train"; case .export: "Export"; case .inference: "Quick Inference" } }
    private func subtitleKey(_ area: Area) -> String { switch area { case .overview: "Project environment and tools"; case .preparation: "Convert labeled audio into DiffSinger training data"; case .configuration: "Create a project configuration from prepared speakers"; case .training: "Run DiffSinger preprocessing and training scripts"; case .export: "Create OpenUtau singer files from exported checkpoints"; case .inference: "Render a .ds file with a trained acoustic model" } }
    private func tr(_ value: String) -> String {
        let locale = model.language == "Português (Brasil)" ? "pt-BR" : model.language == "Deutsch" ? "de" : "en"
        let localizedBundle = Bundle.module.path(forResource: locale, ofType: "lproj").flatMap(Bundle.init(path:)) ?? .module
        return NSLocalizedString(value, bundle: localizedBundle, comment: "")
    }
}
