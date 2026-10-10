import AudioCommon
import Foundation

enum WhisperGreedyDecoder {
    static func decode(
        from start: Int,
        cacheLength: Int,
        config: WhisperGenerationConfig,
        sample: (_ token: Int, _ index: Int, _ generated: [Int]) throws -> Int
    ) throws -> [Int] {
        var generated: [Int] = []
        var nextToken = config.noTimestampsToken
        var tokenIndex = start

        while tokenIndex < cacheLength - 1 {
            try Task.checkCancellation()
            let sampled = try sample(nextToken, tokenIndex, generated)
            try Task.checkCancellation()
            if sampled == config.endToken {
                return generated
            }
            if sampled < config.specialTokenBegin {
                generated.append(sampled)
            }
            nextToken = sampled
            tokenIndex += 1
        }

        // Only EOT confirms a complete transcript; a full cache can hold a partial one.
        try Task.checkCancellation()
        throw AudioModelError.inferenceFailed(
            operation: "Whisper decoder",
            reason: "Decoder context exhausted before the end token")
    }
}
