import Foundation
@testable import WhisperASR
import XCTest

final class WhisperGenerationConfigTests: XCTestCase {
    /// The prefill model indexes tasks translate first, and the token IDs
    /// shift by one between the large-v2 and large-v3 vocabularies.
    func testTranscribeTaskIndexHoldsAcrossVocabularies() throws {
        XCTAssertEqual(try config(transcribe: 50_359, translate: 50_358).transcribeTaskIndex, 1)
        XCTAssertEqual(try config(transcribe: 50_360, translate: 50_359).transcribeTaskIndex, 1)
    }

    private func config(transcribe: Int, translate: Int) throws -> WhisperGenerationConfig {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let json: [String: Any] = [
            "lang_to_id": ["<|en|>": 50_259],
            "task_to_id": ["transcribe": transcribe, "translate": translate],
        ]
        try JSONSerialization.data(withJSONObject: json)
            .write(to: folder.appendingPathComponent("generation_config.json"))
        return try WhisperGenerationConfig.load(from: folder)
    }
}
