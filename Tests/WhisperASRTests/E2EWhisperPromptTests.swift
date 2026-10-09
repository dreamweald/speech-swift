import AudioCommon
import Foundation
@testable import WhisperASR
import XCTest

/// Whisper Small from Argmax's repository, which ships no prefill model, so
/// both paths here step the opening tokens through the decoder.
final class E2EWhisperPromptTests: XCTestCase {
    private static let bundle = WhisperBundle.argmax(
        "openai_whisper-small_216MB", tokenizerRepository: "openai/whisper-small")

    private static func loadModel() async throws -> WhisperASRModel {
        try await WhisperASRModel.fromPretrained(bundle: bundle)
    }

    private func audio() throws -> [Float] {
        // "I asked Aditya Dhar to move the Maia pill into the Dreamweald repo."
        guard let url = Bundle.module.url(forResource: "custom_words", withExtension: "wav", subdirectory: "Resources") else {
            throw XCTSkip("Test audio not in bundle resources")
        }
        return try AudioFileLoader.load(url: url, targetSampleRate: 16_000)
    }

    /// Reference IDs from OpenAI's tiktoken tokenizer.
    func testPromptEncodingMatchesOpenAI() async throws {
        let model = try await Self.loadModel()
        let tokenizer = try WhisperByteLevelTokenizer(modelFolder: model.modelFolder)

        XCTAssertEqual(
            tokenizer.encode(" Vocabulary: Maia, Dreamweald, kubectl."),
            [8993, 19189, 25, 4042, 654, 11, 12105, 826, 3976, 11, 350, 836, 557, 75, 13])
        XCTAssertEqual(
            tokenizer.encode(" Aditya Dhar's 3 Nemotron-runs\u{2014}don't stop!  Caf\u{e9}"),
            [1999, 507, 64, 413, 5854, 311, 805, 22210, 310, 2044, 12, 12997, 82, 2958, 13966, 380, 1590, 0, 220, 46701, 526])
        XCTAssertEqual(tokenizer.startOfPreviousToken, 50_361)
    }

    func testPromptSteersSpelling() async throws {
        let model = try await Self.loadModel()
        let samples = try audio()

        let plain = try model.transcribe(audio: samples, language: "en", prompt: nil).text
        let prompted = try model.transcribe(
            audio: samples, language: "en", prompt: "Vocabulary: Aditya Dhar, Maia, Dreamweald.").text

        XCTAssertTrue(plain.lowercased().contains("pill"), plain)
        XCTAssertTrue(prompted.contains("Dhar"), prompted)
        XCTAssertTrue(prompted.contains("Maia"), prompted)
        XCTAssertTrue(prompted.contains("Dreamweald"), prompted)
        XCTAssertFalse(prompted.contains("Vocabulary"), prompted)
    }

    func testOverlongPromptKeepsItsEnd() async throws {
        let model = try await Self.loadModel()
        let filler = Array(repeating: "Zephyrine", count: 400).joined(separator: ", ")
        XCTAssertGreaterThan(model.promptTokenCount(filler), model.maximumPromptTokens)

        let text = try model.transcribe(
            audio: try audio(), language: "en", prompt: filler + ", Dreamweald.").text

        XCTAssertTrue(text.contains("Dreamweald"), text)
    }
}
