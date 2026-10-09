import AudioCommon
import CoreML
import Foundation

/// A Whisper Core ML bundle on the Hugging Face Hub, in Argmax's layout: at
/// the repository root, or in one folder of a repository that holds several.
public struct WhisperBundle: Sendable, Hashable {
    public let repository: String
    public let folder: String?
    /// The bundles carry no tokenizer; this repository's matches the model.
    public let tokenizerRepository: String

    public init(repository: String, folder: String? = nil, tokenizerRepository: String) {
        self.repository = repository
        self.folder = folder
        self.tokenizerRepository = tokenizerRepository
    }

    public static let largeV3Turbo = WhisperBundle(
        repository: WhisperASRModel.defaultModelId,
        tokenizerRepository: WhisperASRModel.tokenizerModelId)

    /// A folder of `argmaxinc/whisperkit-coreml`, such as `openai_whisper-small_216MB`.
    public static func argmax(_ folder: String, tokenizerRepository: String) -> WhisperBundle {
        WhisperBundle(repository: "argmaxinc/whisperkit-coreml", folder: folder, tokenizerRepository: tokenizerRepository)
    }

    /// Where the bundle's files sit inside a download directory.
    public func modelFolder(in cacheDir: URL) -> URL {
        folder.map { cacheDir.appendingPathComponent($0, isDirectory: true) } ?? cacheDir
    }

    func repositoryPath(_ file: String) -> String {
        folder.map { "\($0)/\(file)" } ?? file
    }
}

/// Where each stage runs. iOS refuses the GPU to a backgrounded app, so a
/// caller that transcribes there keeps every stage off it.
public struct WhisperComputeUnits: Sendable {
    public var melSpectrogram: MLComputeUnits
    public var encoder: MLComputeUnits
    public var decoder: MLComputeUnits

    public init(melSpectrogram: MLComputeUnits, encoder: MLComputeUnits, decoder: MLComputeUnits) {
        self.melSpectrogram = melSpectrogram
        self.encoder = encoder
        self.decoder = decoder
    }

    public static let `default` = WhisperComputeUnits(
        melSpectrogram: .cpuAndGPU, encoder: .cpuAndNeuralEngine, decoder: .cpuAndNeuralEngine)
}

/// Whisper ASR through speech-swift's native CoreML runtime.
public final class WhisperASRModel: SpeechRecognitionModel, @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Whisper-Large-v3-Turbo-CoreML"
    public static let defaultModelVariant = "large-v3-v20240930_turbo"
    public static let tokenizerModelId = "openai/whisper-large-v3"

    static let prefillModelName = "TextDecoderContextPrefill.mlmodelc"

    /// Files that make a published Core ML bundle usable without touching
    /// the network. Checking the model payloads (not just the `.mlmodelc`
    /// directories) avoids accepting a cache left incomplete by an interrupted
    /// snapshot download.
    static let requiredBundleFiles = [
        "MelSpectrogram.mlmodelc/metadata.json",
        "MelSpectrogram.mlmodelc/model.mil",
        "MelSpectrogram.mlmodelc/weights/weight.bin",
        "AudioEncoder.mlmodelc/metadata.json",
        "AudioEncoder.mlmodelc/model.mil",
        "AudioEncoder.mlmodelc/weights/weight.bin",
        "TextDecoder.mlmodelc/metadata.json",
        "TextDecoder.mlmodelc/model.mil",
        "TextDecoder.mlmodelc/weights/weight.bin",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    /// Optional, but checked like the rest when a bundle has it.
    static let prefillBundleFiles = [
        "\(prefillModelName)/metadata.json",
        "\(prefillModelName)/model.mil",
        "\(prefillModelName)/weights/weight.bin",
    ]

    static let downloadPatterns = [
        "MelSpectrogram.mlmodelc/**",
        "AudioEncoder.mlmodelc/**",
        "TextDecoder.mlmodelc/**",
        "\(prefillModelName)/**",
        "generation_config.json",
    ]

    public let inputSampleRate = WhisperCoreMLRuntime.sampleRate
    public let modelId: String
    public let modelFolder: URL

    private let runtime: WhisperCoreMLRuntime

    private init(modelId: String, modelFolder: URL, runtime: WhisperCoreMLRuntime) {
        self.modelId = modelId
        self.modelFolder = modelFolder
        self.runtime = runtime
    }

    /// Load Whisper large-v3 turbo from the published CoreML bundle.
    public static func fromPretrained(
        modelId: String? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> WhisperASRModel {
        let bundle = WhisperBundle(repository: modelId ?? defaultModelId, tokenizerRepository: tokenizerModelId)
        return try await fromPretrained(
            bundle: bundle, cacheDir: cacheDir, offlineMode: offlineMode, progressHandler: progressHandler)
    }

    /// Downloads `bundle` into `cacheDir` unless it is already there, then
    /// loads it, which compiles it for `computeUnits` on first use.
    public static func fromPretrained(
        bundle: WhisperBundle,
        cacheDir: URL? = nil,
        computeUnits: WhisperComputeUnits = .default,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> WhisperASRModel {
        let effectiveModelId = bundle.folder.map { "\(bundle.repository)/\($0)" } ?? bundle.repository
        let resolvedCacheDir: URL
        do {
            resolvedCacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: bundle.repository)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId,
                reason: "Failed to resolve cache directory",
                underlying: error)
        }
        let modelFolder = bundle.modelFolder(in: resolvedCacheDir)

        progressHandler?(0.0, "Preparing Whisper CoreML bundle...")
        do {
            if !hasCompleteCachedBundle(in: modelFolder, includingTokenizer: false) {
                try await HuggingFaceDownloader.downloadFiles(
                    modelId: bundle.repository,
                    to: resolvedCacheDir,
                    files: downloadPatterns.map(bundle.repositoryPath),
                    offlineMode: offlineMode
                ) { fraction in
                    progressHandler?(fraction * 0.75, "Downloading Whisper CoreML bundle...")
                }
            } else {
                progressHandler?(0.75, "Using cached Whisper CoreML bundle")
            }

            try await ensureTokenizer(
                in: modelFolder,
                repository: bundle.tokenizerRepository,
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(0.75 + fraction * 0.10, "Downloading Whisper tokenizer...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId,
                reason: "Download failed",
                underlying: error)
        }

        progressHandler?(0.90, "Loading Whisper CoreML runtime...")
        do {
            let runtime = try WhisperCoreMLRuntime(modelFolder: modelFolder, computeUnits: computeUnits)
            progressHandler?(1.0, "Model loaded")
            return WhisperASRModel(modelId: effectiveModelId, modelFolder: modelFolder, runtime: runtime)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId,
                reason: "Failed to initialize native Whisper runtime",
                underlying: error)
        }
    }

    /// Whether `bundle` is fully downloaded into `cacheDir`, tokenizer included.
    public static func isDownloaded(_ bundle: WhisperBundle, in cacheDir: URL) -> Bool {
        hasCompleteCachedBundle(in: bundle.modelFolder(in: cacheDir))
    }

    static func hasCompleteCachedBundle(in directory: URL, includingTokenizer: Bool = true) -> Bool {
        let fileManager = FileManager.default
        let tokenizerFiles: Set<String> = ["tokenizer.json", "tokenizer_config.json"]
        var files = requiredBundleFiles.filter { includingTokenizer || !tokenizerFiles.contains($0) }
        if fileManager.fileExists(atPath: directory.appendingPathComponent(prefillModelName).path) {
            files += prefillBundleFiles
        }
        for relativePath in files {
            let url = directory.appendingPathComponent(relativePath)
            guard fileManager.fileExists(atPath: url.path),
                  let attributes = try? fileManager.attributesOfItem(atPath: url.path),
                  let size = attributes[.size] as? NSNumber,
                  size.int64Value > 0
            else {
                return false
            }
        }
        return true
    }

    /// The most prompt tokens a transcription keeps. A longer prompt loses its start.
    public var maximumPromptTokens: Int {
        runtime.maximumPromptTokens
    }

    /// How many tokens `prompt` takes, to fit a prompt under ``maximumPromptTokens``.
    public func promptTokenCount(_ prompt: String) -> Int {
        runtime.promptTokens(for: prompt).count
    }

    /// Transcribes 16 kHz audio. `prompt` reads to the model as text spoken
    /// just before the audio, so names and spellings in it steer the
    /// transcript. A nil `language` is detected from the audio. Checks for
    /// cancellation between decoder steps.
    public func transcribe(audio: [Float], language: String?, prompt: String?) throws -> AudioCommon.TranscriptionResult {
        let result = try runtime.transcribe(audio: audio, languageHint: normalizedLanguage(language), prompt: prompt)
        return AudioCommon.TranscriptionResult(text: result.text, language: result.language, confidence: 0)
    }

    /// Transcribe PCM Float32 audio. Input is resampled to 16 kHz when needed.
    public func transcribeAudio(_ audio: [Float], sampleRate: Int, language: String? = nil) async throws -> String {
        let normalizedAudio: [Float]
        if sampleRate == inputSampleRate {
            normalizedAudio = audio
        } else {
            normalizedAudio = AudioFileLoader.resample(audio, from: sampleRate, to: inputSampleRate)
        }

        return try runtime.transcribe(audio: normalizedAudio, languageHint: normalizedLanguage(language)).text
    }

    /// Transcribe and return the detected or hinted Whisper language code when available.
    public func transcribeWithLanguageAsync(
        audio: [Float],
        sampleRate: Int,
        language: String? = nil
    ) async throws -> AudioCommon.TranscriptionResult {
        let normalizedAudio: [Float]
        if sampleRate == inputSampleRate {
            normalizedAudio = audio
        } else {
            normalizedAudio = AudioFileLoader.resample(audio, from: sampleRate, to: inputSampleRate)
        }

        let result = try runtime.transcribe(audio: normalizedAudio, languageHint: normalizedLanguage(language))
        return AudioCommon.TranscriptionResult(text: result.text, language: result.language, confidence: 0)
    }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try runBlocking {
                try await self.transcribeAudio(audio, sampleRate: sampleRate, language: language)
            }
        } catch {
            AudioLog.inference.error("Whisper transcription failed: \(error)")
            return ""
        }
    }

    public func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> AudioCommon.TranscriptionResult {
        do {
            return try runBlocking {
                try await self.transcribeWithLanguageAsync(audio: audio, sampleRate: sampleRate, language: language)
            }
        } catch {
            AudioLog.inference.error("Whisper transcription failed: \(error)")
            return AudioCommon.TranscriptionResult(text: "")
        }
    }

    private static func ensureTokenizer(
        in modelFolder: URL,
        repository: String,
        offlineMode: Bool,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        let files = ["tokenizer.json", "tokenizer_config.json"]
        let missingFiles = files.filter {
            !FileManager.default.fileExists(atPath: modelFolder.appendingPathComponent($0).path)
        }
        if missingFiles.isEmpty {
            progressHandler?(1.0)
            return
        }
        if offlineMode {
            throw AudioModelError.modelLoadFailed(
                modelId: repository,
                reason: "Offline tokenizer cache miss: \(missingFiles.joined(separator: ", "))")
        }

        try FileManager.default.createDirectory(at: modelFolder, withIntermediateDirectories: true)
        for (index, file) in missingFiles.enumerated() {
            let escaped = file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file
            let endpoint = (HuggingFaceDownloader.resolvedEndpoint() ?? "https://huggingface.co")
                .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            guard let url = URL(string: "\(endpoint)/\(repository)/resolve/main/\(escaped)") else {
                throw AudioModelError.modelLoadFailed(modelId: repository, reason: "Invalid tokenizer URL for \(file)")
            }

            var request = URLRequest(url: url)
            let env = ProcessInfo.processInfo.environment
            if let token = env["HF_TOKEN"] ?? env["HUGGING_FACE_HUB_TOKEN"],
               !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }

            let (data, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse,
               !(200..<300).contains(http.statusCode) {
                throw AudioModelError.modelLoadFailed(
                    modelId: repository,
                    reason: "\(file): HTTP \(http.statusCode)")
            }
            try data.write(to: modelFolder.appendingPathComponent(file), options: .atomic)
            progressHandler?(Double(index + 1) / Double(missingFiles.count))
        }
    }

    private func normalizedLanguage(_ language: String?) -> String? {
        guard let language else { return nil }
        let trimmed = language.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return trimmed
    }
}

private func runBlocking<T>(_ operation: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<T, Error>?

    Task {
        do {
            result = .success(try await operation())
        } catch {
            result = .failure(error)
        }
        semaphore.signal()
    }

    semaphore.wait()
    return try result!.get()
}
