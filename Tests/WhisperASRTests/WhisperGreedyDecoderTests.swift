import AudioCommon
import Foundation
@testable import WhisperASR
import XCTest

final class WhisperGreedyDecoderTests: XCTestCase {
    private var folder: URL!
    private var config: WhisperGenerationConfig!
    private var tokenizer: WhisperByteLevelTokenizer!

    override func setUpWithError() throws {
        folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data(#"{"lang_to_id":{"<|en|>":50259},"task_to_id":{"transcribe":50360,"translate":50359}}"#.utf8)
            .write(to: folder.appendingPathComponent("generation_config.json"))
        try Data(#"{"model":{"vocab":{"Ġgo":1,"ĠGO":2,".":3,"Ġnow":4,"Ġg":5,"o":6,"<|startofprev|>":50361},"merges":[]}}"#.utf8)
            .write(to: folder.appendingPathComponent("tokenizer.json"))
        config = try WhisperGenerationConfig.load(from: folder)
        tokenizer = try WhisperByteLevelTokenizer(modelFolder: folder)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: folder)
    }

    func testRepeatedSpeechKeepsTheThirdWordAndFollowingSpeech() throws {
        let cases: [(tokens: [Int], text: String)] = [
            ([1, 1, 1, 4, 3], " go go go now."),
            ([2, 3, 1, 3, 1, 3, 4, 3], " GO. go. go. now."),
            ([5, 6, 5, 6, 5, 6, 4, 3], " go go go now."),
        ]
        for test in cases {
            let generated = try decode(test.tokens + [config.endToken])
            XCTAssertEqual(tokenizer.decode(generated), test.text)
        }
    }

    func testLargePromptCannotTurnContextExhaustionIntoPartialSuccess() {
        // 223 vocabulary tokens, start-of-previous, and three opening tokens use 227 positions.
        let start = 223 + 1 + 3
        var indices: [Int] = []
        XCTAssertThrowsError(try WhisperGreedyDecoder.decode(from: start, cacheLength: 448, config: config) { _, index, _ in
            indices.append(index)
            return 1
        }) { error in
            self.assertContextExhaustion(error)
        }
        XCTAssertEqual(indices, Array(227..<447))
        XCTAssertEqual(indices.count, 220)
    }

    func testEndTokenAtLastAvailableStepCompletesTranscript() throws {
        let spoken = Array(repeating: 1, count: 219)
        let generated = try decode(spoken + [config.endToken], from: 227)
        XCTAssertEqual(generated, spoken)
    }

    func testNoDecoderRoomFailsWithoutSampling() {
        XCTAssertThrowsError(try WhisperGreedyDecoder.decode(from: 447, cacheLength: 448, config: config) { _, _, _ in
            XCTFail("An exhausted decoder must not sample outside its context")
            return self.config.endToken
        }) { error in
            self.assertContextExhaustion(error)
        }
    }

    func testEndTokenStopsSamplingForPromptedAndUnpromptedOpenings() throws {
        for start in [3, 227] {
            XCTAssertEqual(try decode([1, 4, config.endToken], from: start), [1, 4])
        }
    }

    func testCancellationFromDecoderStepIsPreservedAfterPartialSpeech() {
        var calls = 0
        XCTAssertThrowsError(try WhisperGreedyDecoder.decode(from: 3, cacheLength: 448, config: config) { _, _, _ in
            calls += 1
            if calls == 2 { throw CancellationError() }
            return 1
        }) { error in
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertEqual(calls, 2)
    }

    func testTaskCancellationWinsOverEndTokenOrContextExhaustion() async {
        let config = self.config!
        for sampled in [1, config.endToken] {
            let task = Task {
                try WhisperGreedyDecoder.decode(from: 446, cacheLength: 448, config: config) { _, _, _ in
                    withUnsafeCurrentTask { $0?.cancel() }
                    return sampled
                }
            }
            do {
                _ = try await task.value
                XCTFail("Cancellation must not become success or context exhaustion")
            } catch {
                XCTAssertTrue(error is CancellationError)
            }
        }
    }

    private func decode(_ tokens: [Int], from start: Int = 3) throws -> [Int] {
        var cursor = 0
        let generated = try WhisperGreedyDecoder.decode(from: start, cacheLength: 448, config: config) { token, index, _ in
            XCTAssertEqual(index, start + cursor)
            XCTAssertEqual(token, cursor == 0 ? self.config.noTimestampsToken : tokens[cursor - 1])
            let sampled = try XCTUnwrap(tokens.dropFirst(cursor).first, "Decoder sampled past EOT")
            cursor += 1
            return sampled
        }
        XCTAssertEqual(cursor, tokens.count)
        return generated
    }

    private func assertContextExhaustion(_ error: Error, file: StaticString = #filePath, line: UInt = #line) {
        guard case AudioModelError.inferenceFailed(let operation, let reason) = error else {
            return XCTFail("Expected inference failure, got \(error)", file: file, line: line)
        }
        XCTAssertEqual(operation, "Whisper decoder", file: file, line: line)
        XCTAssertTrue(reason.contains("context exhausted"), file: file, line: line)
    }
}
