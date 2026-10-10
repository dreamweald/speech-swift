import Foundation
@testable import WhisperASR
import XCTest

final class WhisperBundleCacheTests: XCTestCase {
    private var temporaryDirectory: URL!

    override func setUpWithError() throws {
        temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: temporaryDirectory)
        temporaryDirectory = nil
    }

    func testCompleteBundleIsAccepted() throws {
        try createCompleteBundle()

        XCTAssertTrue(WhisperASRModel.hasCompleteCachedBundle(in: temporaryDirectory))
    }

    func testMissingPayloadIsRejected() throws {
        try createCompleteBundle()
        try FileManager.default.removeItem(
            at: temporaryDirectory.appendingPathComponent("AudioEncoder.mlmodelc/model.mil")
        )

        XCTAssertFalse(WhisperASRModel.hasCompleteCachedBundle(in: temporaryDirectory))
    }

    func testEmptyPayloadIsRejected() throws {
        try createCompleteBundle()
        try Data().write(
            to: temporaryDirectory.appendingPathComponent("TextDecoder.mlmodelc/weights/weight.bin")
        )

        XCTAssertFalse(WhisperASRModel.hasCompleteCachedBundle(in: temporaryDirectory))
    }

    func testBundleWithoutPrefillIsAccepted() throws {
        try createCompleteBundle()
        try FileManager.default.removeItem(
            at: temporaryDirectory.appendingPathComponent(WhisperASRModel.prefillModelName)
        )

        XCTAssertTrue(WhisperASRModel.hasCompleteCachedBundle(in: temporaryDirectory))
    }

    func testIncompletePrefillIsRejected() throws {
        try createCompleteBundle()
        try FileManager.default.removeItem(
            at: temporaryDirectory.appendingPathComponent("\(WhisperASRModel.prefillModelName)/model.mil")
        )

        XCTAssertFalse(WhisperASRModel.hasCompleteCachedBundle(in: temporaryDirectory))
    }

    func testMissingTokenizerOnlyBlocksTheFullCheck() throws {
        try createCompleteBundle()
        try FileManager.default.removeItem(at: temporaryDirectory.appendingPathComponent("tokenizer.json"))

        XCTAssertFalse(WhisperASRModel.hasCompleteCachedBundle(in: temporaryDirectory))
        XCTAssertTrue(WhisperASRModel.hasCompleteCachedBundle(in: temporaryDirectory, includingTokenizer: false))
    }

    func testFolderBundleLivesInItsFolder() throws {
        let bundle = WhisperBundle.argmax("openai_whisper-small_216MB", tokenizerRepository: "openai/whisper-small")
        let folder = temporaryDirectory.appendingPathComponent("openai_whisper-small_216MB", isDirectory: true)
        try createCompleteBundle(in: folder)

        XCTAssertEqual(bundle.modelFolder(in: temporaryDirectory), folder)
        XCTAssertEqual(bundle.repositoryPath("TextDecoder.mlmodelc/**"), "openai_whisper-small_216MB/TextDecoder.mlmodelc/**")
        XCTAssertTrue(WhisperASRModel.isDownloaded(bundle, in: temporaryDirectory))
        XCTAssertFalse(WhisperASRModel.isDownloaded(.largeV3Turbo, in: temporaryDirectory))
    }

    private func createCompleteBundle(in directory: URL? = nil) throws {
        let directory = directory ?? temporaryDirectory!
        for relativePath in WhisperASRModel.requiredBundleFiles + WhisperASRModel.prefillBundleFiles {
            let fileURL = directory.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data([0x01]).write(to: fileURL)
        }
    }
}
