package perfledger

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func benchEvidenceFingerprint(content, reasoning string) string {
	contentDigest := sha256.Sum256([]byte(content))
	reasoningDigest := sha256.Sum256([]byte(reasoning))
	digest := sha256.Sum256(append(contentDigest[:], reasoningDigest[:]...))
	return hex.EncodeToString(digest[:])
}

func benchEvidenceUsage(prompt, cached, completion string) []byte {
	return []byte(fmt.Sprintf("{\"prompt_tokens\":%s,\"completion_tokens\":%s,\"prompt_tokens_details\":{\"cached_tokens\":%s}}", prompt, completion, cached))
}

// fak-test:justify why=contract when=changed:internal/perfledger/**
// fak-test:runtime fast est=200ms lane=default
func TestBenchEvidenceStrictUsagePreservesKnownSiblings(t *testing.T) {
	for _, invalid := range []string{"true", "false", "1.0", "1.5", "\"1\"", "-1", "null", "{}", "[]", "9223372036854775808"} {
		for _, field := range []string{"prompt", "cached", "completion"} {
			t.Run(field+"_"+invalid, func(t *testing.T) {
				values := map[string]string{"prompt": "10", "cached": "4", "completion": "3"}
				values[field] = invalid
				got := NewBenchEvidence(benchEvidenceUsage(values["prompt"], values["cached"], values["completion"]), "answer", "reason", "", true)
				if got == nil {
					t.Fatal("strict usage observation disappeared")
				}
				counts := map[string]*int{"prompt": got.PromptTokens, "cached": got.CachedTokens, "completion": got.Tokens}
				expected := map[string]int{"prompt": 10, "cached": 4, "completion": 3}
				for name, count := range counts {
					if name == field {
						if count != nil {
							t.Errorf("invalid %s count became known: %d", name, *count)
						}
					} else if count == nil || *count != expected[name] {
						t.Errorf("invalid sibling erased known %s count: %v", name, count)
					}
				}
				if field != "completion" && got.CacheState != "unverified" {
					t.Errorf("invalid cache accounting became verified: %q", got.CacheState)
				}
			})
		}
	}
}

// fak-test:justify why=regression when=changed:internal/perfledger/**
// fak-test:runtime fast est=200ms lane=default
func TestBenchEvidenceCacheNeedsExplicitInclusiveAccounting(t *testing.T) {
	for _, tc := range []struct {
		name, prompt, cached, completion, state string
		wantPrompt, wantCached, wantTokens      int
	}{
		{"explicit-zero", "0", "0", "0", "cold", 0, 0, 0},
		{"cold", "10", "0", "3", "cold", 10, 0, 3},
		{"partial-reuse", "10", "4", "3", "reused", 10, 4, 3},
		{"inclusive-boundary", "10", "10", "3", "reused", 10, 10, 3},
		{"inconsistent-cache", "10", "11", "3", "unverified", 10, 11, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := NewBenchEvidence(benchEvidenceUsage(tc.prompt, tc.cached, tc.completion), "", "", "", true)
			if got == nil || got.CacheState != tc.state {
				t.Fatalf("cache evidence state = %+v, want %q", got, tc.state)
			}
			if got.PromptTokens == nil || *got.PromptTokens != tc.wantPrompt || got.CachedTokens == nil || *got.CachedTokens != tc.wantCached || got.Tokens == nil || *got.Tokens != tc.wantTokens {
				t.Fatalf("explicit usage counts not preserved: %+v", got)
			}
		})
	}
	for _, raw := range [][]byte{
		nil,
		[]byte("{"),
		[]byte("null"),
		[]byte("[]"),
		[]byte("{}"),
		[]byte("{\"prompt_tokens\":10,\"completion_tokens\":3}"),
		[]byte("{\"prompt_tokens\":10,\"completion_tokens\":3,\"cached_tokens\":0}"),
	} {
		got := NewBenchEvidence(raw, "", "", "", true)
		if got == nil || got.CachedTokens != nil || got.CacheState != "unverified" {
			t.Fatalf("missing nested cache count fabricated cold evidence: %+v", got)
		}
	}
}

// fak-test:justify why=contract when=changed:internal/perfledger/**
// fak-test:runtime fast est=200ms lane=default
func TestBenchEvidenceFingerprintSeparatesContentAndReasoning(t *testing.T) {
	content, reasoning := "héllo 世界", "考える!"
	got := NewBenchEvidence(nil, content, reasoning, "", true)
	if got == nil || !got.Complete || got.OutputSHA256 != benchEvidenceFingerprint(content, reasoning) {
		t.Fatalf("completed UTF8 output fingerprint violates the paired-digest contract: %+v", got)
	}
	swapped := NewBenchEvidence(nil, reasoning, content, "", true)
	one := NewBenchEvidence(nil, "ab", "c", "", true)
	two := NewBenchEvidence(nil, "a", "bc", "", true)
	if got.OutputSHA256 == swapped.OutputSHA256 || one.OutputSHA256 == two.OutputSHA256 {
		t.Fatal("fingerprint collapsed content/reasoning boundaries or order")
	}
	if got.TokenSHA != "" {
		t.Fatal("text fingerprint was substituted for token proof")
	}
}

// fak-test:justify why=contract when=changed:internal/perfledger/**
// fak-test:runtime fast est=200ms lane=default
func TestBenchEvidenceTokenProofRequiresItsClosedShape(t *testing.T) {
	for _, tokenSHA := range []string{strings.Repeat("a", 12), strings.Repeat("9", 64)} {
		got := NewBenchEvidence(nil, "answer", "", tokenSHA, true)
		if got == nil || got.TokenSHA != tokenSHA {
			t.Fatalf("valid token proof not retained: %+v", got)
		}
	}
	for _, tokenSHA := range []string{"", strings.Repeat("a", 11), strings.Repeat("a", 13), strings.Repeat("a", 63), strings.Repeat("a", 65), strings.Repeat("A", 12), strings.Repeat("g", 64), "sha256:" + strings.Repeat("a", 64)} {
		got := NewBenchEvidence(nil, "answer", "", tokenSHA, true)
		if got == nil || got.TokenSHA != "" {
			t.Fatalf("invalid or missing token proof became known: %+v", got)
		}
	}
}

// fak-test:justify why=regression when=changed:internal/perfledger/**
// fak-test:runtime fast est=200ms lane=default
func TestBenchEvidenceIncompleteOutputCannotClaimEquivalence(t *testing.T) {
	got := NewBenchEvidence(benchEvidenceUsage("10", "4", "3"), "partial answer", "partial reasoning", strings.Repeat("a", 64), false)
	if got == nil || got.Complete || got.OutputSHA256 != "" || got.TokenSHA != "" || got.CacheState != "unverified" {
		t.Fatalf("interrupted output retained complete evidence: %+v", got)
	}
	if got.PromptTokens == nil || *got.PromptTokens != 10 || got.CachedTokens == nil || *got.CachedTokens != 4 || got.Tokens == nil || *got.Tokens != 3 {
		t.Fatal("completion uncertainty erased explicit server usage")
	}
}

// fak-test:justify why=contract when=changed:internal/perfledger/**
// fak-test:runtime fast est=200ms lane=default
func TestBenchEvidenceLegacyRecordOmitsOptionalEvidence(t *testing.T) {
	raw, err := json.Marshal(Record{})
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	if _, exists := fields["bench_evidence"]; exists {
		t.Fatal("legacy row fabricated the new evidence field")
	}
}

// fak-test:justify why=regression when=changed:internal/perfledger/**
// fak-test:runtime fast est=200ms lane=default
func TestBenchEvidenceOutputBoundPreservesCountsAndRevokesProof(t *testing.T) {
	usage := []byte(`{"prompt_tokens":10,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":4}}`)
	half := MaxBenchEvidenceOutputBytes / 2
	atLimit := NewBenchEvidence(usage, strings.Repeat("a", half), strings.Repeat("b", MaxBenchEvidenceOutputBytes-half), "abcdef123456", true)
	if !atLimit.Complete || atLimit.OutputSHA256 == "" || atLimit.CacheState != "reused" || atLimit.TokenSHA != "abcdef123456" {
		t.Fatalf("declared output boundary rejected a complete observation: %+v", atLimit)
	}
	for _, tc := range []struct {
		name, content, reasoning string
	}{
		{"content", strings.Repeat("a", MaxBenchEvidenceOutputBytes+1), ""},
		{"reasoning", "", strings.Repeat("b", MaxBenchEvidenceOutputBytes+1)},
		{"combined", strings.Repeat("a", half), strings.Repeat("b", MaxBenchEvidenceOutputBytes-half+1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			evidence := NewBenchEvidence(usage, tc.content, tc.reasoning, "abcdef123456", true)
			if evidence.Complete || evidence.CacheState != "unverified" || evidence.OutputSHA256 != "" || evidence.TokenSHA != "" {
				t.Fatalf("over-budget output promoted an equivalence/cache claim: %+v", evidence)
			}
			if evidence.PromptTokens == nil || *evidence.PromptTokens != 10 || evidence.CachedTokens == nil || *evidence.CachedTokens != 4 || evidence.Tokens == nil || *evidence.Tokens != 3 {
				t.Fatalf("output bound erased explicit usage evidence: %+v", evidence)
			}
		})
	}
}
