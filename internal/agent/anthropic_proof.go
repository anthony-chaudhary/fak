package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
)

// AnthropicRewriteProof describes the byte regions protected by the actual
// rewrite selector. It owns no request bytes. Tokens use EstimateAnthropicTokens,
// rather than an exact tokenizer or provider usage measurement.
type AnthropicRewriteProof struct {
	PrefixBytes      int
	SuffixBytes      int
	PrePrefixSHA256  string
	PostPrefixSHA256 string
	PreSuffixSHA256  string
	PostSuffixSHA256 string
	PreTokens        int
	PostTokens       int
	ShedBytes        int
}

var (
	ErrAnthropicProofAnchor = errors.New("ANTHROPIC_PROOF_ANCHOR_UNAVAILABLE")
	ErrAnthropicProofBody   = errors.New("ANTHROPIC_PROOF_BODY_INVALID")
	ErrAnthropicProofPrefix = errors.New("ANTHROPIC_PROOF_PREFIX_MISMATCH")
	ErrAnthropicProofDelta  = errors.New("ANTHROPIC_PROOF_DELTA_INVALID")
)

// VerifyAnthropicRewrite verifies a compaction using the same anchor selector
// and splice verifier as CompactAnthropicHistoryWithOptions. Head mode protects
// the serialized tail too: system/tools may follow the messages array.
func VerifyAnthropicRewrite(before, after []byte, anchor CompactAnchor) (AnthropicRewriteProof, error) {
	if anchor != CompactAnchorFirstBP && anchor != CompactAnchorHead {
		return AnthropicRewriteProof{}, ErrAnthropicProofAnchor
	}
	_, spans, prefix, _, ok := anchorCompactablePrefixMode(before, 3, anchor)
	if !ok {
		return AnthropicRewriteProof{}, ErrAnthropicProofAnchor
	}
	return verifyAnthropicProofSplice(before, after, spans, prefix)
}

// VerifyAnthropicElision uses the elision selector, whose deeper breakpoint
// search protects cache_control nested inside tool results. A compaction anchor
// is not a substitute for this proof.
func VerifyAnthropicElision(before, after []byte) (AnthropicRewriteProof, error) {
	_, spans, prefix, reason := anchorElidableMessages(before, elideAnchorReasons{
		nonJSON: "invalid", noMsgsKey: "invalid", decodeFailed: "invalid",
		tooFewMsgs: "invalid", noBreakpoint: "invalid",
	})
	if reason != "" {
		return AnthropicRewriteProof{}, ErrAnthropicProofAnchor
	}
	return verifyAnthropicProofSplice(before, after, spans, prefix)
}

func verifyAnthropicProofSplice(before, after []byte, spans []elementSpan, prefix int) (AnthropicRewriteProof, error) {
	if len(after) >= len(before) {
		return AnthropicRewriteProof{}, ErrAnthropicProofDelta
	}
	pre, err := DecodeAnthropicMessagesRequest(before)
	if err != nil {
		return AnthropicRewriteProof{}, ErrAnthropicProofBody
	}
	switch verifySplicedBody(before, after, spans, prefix) {
	case spliceVerdictPrefixMismatch:
		return AnthropicRewriteProof{}, ErrAnthropicProofPrefix
	case spliceVerdictOK:
	default:
		return AnthropicRewriteProof{}, ErrAnthropicProofBody
	}
	post, err := DecodeAnthropicMessagesRequest(after)
	if err != nil {
		return AnthropicRewriteProof{}, ErrAnthropicProofBody
	}
	prefixBytes := arrayContentStart(spans)
	if prefix >= 0 {
		prefixBytes = spans[prefix].end
	}
	suffixBytes := len(before) - spans[len(spans)-1].end
	if prefixBytes > len(after)-suffixBytes {
		return AnthropicRewriteProof{}, ErrAnthropicProofPrefix
	}
	return AnthropicRewriteProof{
		PrefixBytes: prefixBytes, SuffixBytes: suffixBytes,
		PrePrefixSHA256:  anthropicProofHash(before[:prefixBytes]),
		PostPrefixSHA256: anthropicProofHash(after[:prefixBytes]),
		PreSuffixSHA256:  anthropicProofHash(before[len(before)-suffixBytes:]),
		PostSuffixSHA256: anthropicProofHash(after[len(after)-suffixBytes:]),
		PreTokens:        EstimateAnthropicTokens(pre), PostTokens: EstimateAnthropicTokens(post),
		ShedBytes: len(before) - len(after),
	}, nil
}

func anthropicProofHash(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
