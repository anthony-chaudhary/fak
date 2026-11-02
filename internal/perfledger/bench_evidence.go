// Adapted from TensorFold tools/bench_evidence.py at 9e8c368b0b4c2c7892a0c9f0f302d7c164016209.
// Copyright 2026 TensorFold contributors. Apache-2.0; see THIRD_PARTY_NOTICES.md.
// Changed: Go receipt types and strict JSON count parsing.
package perfledger

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

const MaxBenchEvidenceOutputBytes = 4 << 20

type BenchEvidence struct {
	PromptTokens *int   `json:"prompt_tokens"`
	CachedTokens *int   `json:"cached_tokens"`
	Tokens       *int   `json:"tokens"`
	CacheState   string `json:"cache_state"`
	Complete     bool   `json:"complete"`
	TokenSHA     string `json:"token_sha,omitempty"`
	TokenSHA256  string `json:"token_sha256,omitempty"`
	OutputSHA256 string `json:"output_sha256,omitempty"`
}

func NewBenchEvidence(usageRaw []byte, content, reasoning, tokenSHA string, complete bool) *BenchEvidence {
	return NewBenchEvidenceWithTokenSHA256(usageRaw, content, reasoning, tokenSHA, "", complete)
}

func NewBenchEvidenceWithTokenSHA256(usageRaw []byte, content, reasoning, tokenSHA, tokenSHA256 string, complete bool) *BenchEvidence {
	complete = complete && len(content) <= MaxBenchEvidenceOutputBytes && len(reasoning) <= MaxBenchEvidenceOutputBytes-len(content)
	e := &BenchEvidence{CacheState: "unverified", Complete: complete}
	var usage struct {
		Prompt     json.RawMessage `json:"prompt_tokens"`
		Completion json.RawMessage `json:"completion_tokens"`
		Details    json.RawMessage `json:"prompt_tokens_details"`
	}
	if json.Unmarshal(usageRaw, &usage) == nil {
		e.PromptTokens = benchCount(usage.Prompt)
		e.Tokens = benchCount(usage.Completion)
		var details struct {
			Cached json.RawMessage `json:"cached_tokens"`
		}
		if json.Unmarshal(usage.Details, &details) == nil {
			e.CachedTokens = benchCount(details.Cached)
		}
	}
	if !complete {
		return e
	}
	if e.PromptTokens != nil && e.CachedTokens != nil && *e.CachedTokens <= *e.PromptTokens {
		e.CacheState = "cold"
		if *e.CachedTokens > 0 {
			e.CacheState = "reused"
		}
	}
	if validBenchTokenSHA(tokenSHA) {
		e.TokenSHA = tokenSHA
	}
	if len(tokenSHA256) == 64 && validBenchTokenSHA(tokenSHA256) && (e.TokenSHA == "" || tokenSHA256[:12] == e.TokenSHA[:12]) {
		e.TokenSHA256 = tokenSHA256
	}
	contentDigest := sha256.Sum256([]byte(content))
	reasoningDigest := sha256.Sum256([]byte(reasoning))
	var joined [2 * sha256.Size]byte
	copy(joined[:sha256.Size], contentDigest[:])
	copy(joined[sha256.Size:], reasoningDigest[:])
	outputDigest := sha256.Sum256(joined[:])
	e.OutputSHA256 = hex.EncodeToString(outputDigest[:])
	return e
}

func benchCount(raw json.RawMessage) *int {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var n int
	if json.Unmarshal(raw, &n) != nil || n < 0 {
		return nil
	}
	return &n
}

func validBenchTokenSHA(value string) bool {
	if len(value) != 12 && len(value) != 64 {
		return false
	}
	for _, c := range value {
		if !(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
