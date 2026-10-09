import AudioCommon
import Foundation

struct WhisperByteLevelTokenizer: Sendable {
    private let idToToken: [Int: String]
    private let tokenToId: [String: Int]
    private let specialIds: Set<Int>
    private let mergeRanks: [String: Int]
    private let byteDecoder: [UInt32: UInt8]
    private let byteEncoder: [UInt8: Character]
    let startOfPreviousToken: Int

    init(modelFolder: URL) throws {
        let url = modelFolder.appendingPathComponent("tokenizer.json")
        let raw: RawTokenizer
        do {
            raw = try JSONDecoder().decode(RawTokenizer.self, from: Data(contentsOf: url))
        } catch {
            throw AudioModelError.weightLoadingFailed(path: url.path, underlying: error)
        }

        var tokens = raw.model.vocab.reduce(into: [Int: String]()) { result, entry in
            result[entry.value] = entry.key
        }
        var ids = raw.model.vocab
        var specials = Set<Int>()
        for token in raw.addedTokens ?? [] {
            tokens[token.id] = token.content
            ids[token.content] = token.id
            if token.special == true {
                specials.insert(token.id)
            }
        }

        guard let startOfPrevious = ids["<|startofprev|>"] else {
            throw AudioModelError.weightLoadingFailed(
                path: url.path, underlying: TokenizerError.invalidFormat("No <|startofprev|> token"))
        }
        self.startOfPreviousToken = startOfPrevious
        self.idToToken = tokens
        self.tokenToId = ids
        self.specialIds = specials
        self.mergeRanks = Dictionary(
            raw.model.merges.enumerated().map { ($0.element, $0.offset) },
            uniquingKeysWith: { first, _ in first })
        self.byteDecoder = Self.makeByteDecoder()
        self.byteEncoder = Dictionary(uniqueKeysWithValues: byteDecoder.map {
            ($0.value, Character(UnicodeScalar($0.key)!))
        })
    }

    func tokenId(for token: String) -> Int? {
        tokenToId[token]
    }

    func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count * 4)

        for id in ids {
            guard !specialIds.contains(id),
                  let token = idToToken[id],
                  !token.isEmpty
            else {
                continue
            }

            for scalar in token.unicodeScalars {
                if let byte = byteDecoder[scalar.value] {
                    bytes.append(byte)
                } else {
                    bytes.append(contentsOf: String(scalar).utf8)
                }
            }
        }

        return String(decoding: bytes, as: UTF8.self)
    }

    /// Byte-level BPE with GPT-2's pre-tokenizer, which Whisper's tokenizer
    /// uses. Special tokens in `text` are encoded as plain text.
    func encode(_ text: String) -> [Int] {
        let range = NSRange(text.startIndex..., in: text)
        return Self.preTokenizer.matches(in: text, range: range).flatMap { match -> [Int] in
            guard let wordRange = Range(match.range, in: text) else { return [] }
            let word = String(text[wordRange].utf8.compactMap { byteEncoder[$0] })
            return bpe(word).compactMap { tokenToId[$0] }
        }
    }

    private func bpe(_ word: String) -> [String] {
        var pieces = word.map(String.init)
        while pieces.count > 1 {
            var best: (index: Int, rank: Int)?
            for index in 0..<(pieces.count - 1) {
                if let rank = mergeRanks["\(pieces[index]) \(pieces[index + 1])"],
                   rank < best?.rank ?? .max {
                    best = (index, rank)
                }
            }
            guard let best else { break }
            let first = pieces[best.index]
            let second = pieces[best.index + 1]
            var merged: [String] = []
            var index = 0
            while index < pieces.count {
                if index < pieces.count - 1, pieces[index] == first, pieces[index + 1] == second {
                    merged.append(first + second)
                    index += 2
                } else {
                    merged.append(pieces[index])
                    index += 1
                }
            }
            pieces = merged
        }
        return pieces
    }

    private static let preTokenizer = try! NSRegularExpression(
        pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)

    private static func makeByteDecoder() -> [UInt32: UInt8] {
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var chars = bytes
        var next = 0

        for byte in 0...255 where !bytes.contains(byte) {
            bytes.append(byte)
            chars.append(256 + next)
            next += 1
        }

        var decoder: [UInt32: UInt8] = [:]
        for (byte, char) in zip(bytes, chars) {
            decoder[UInt32(char)] = UInt8(byte)
        }
        return decoder
    }
}

private struct RawTokenizer: Decodable {
    let model: RawTokenizerModel
    let addedTokens: [RawAddedToken]?

    enum CodingKeys: String, CodingKey {
        case model
        case addedTokens = "added_tokens"
    }
}

private struct RawTokenizerModel: Decodable {
    let vocab: [String: Int]
    /// Each merge as `"left right"`. Newer exports write pairs as arrays.
    let merges: [String]

    enum CodingKeys: String, CodingKey {
        case vocab
        case merges
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocab = try container.decode([String: Int].self, forKey: .vocab)
        if let pairs = try? container.decode([[String]].self, forKey: .merges) {
            merges = pairs.map { $0.joined(separator: " ") }
        } else {
            merges = try container.decodeIfPresent([String].self, forKey: .merges) ?? []
        }
    }
}

private struct RawAddedToken: Decodable {
    let id: Int
    let content: String
    let special: Bool?
}
