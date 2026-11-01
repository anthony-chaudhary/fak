package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

func perfEvidenceFingerprint(content, reasoning string) string {
	contentDigest := sha256.Sum256([]byte(content))
	reasoningDigest := sha256.Sum256([]byte(reasoning))
	digest := sha256.Sum256(append(contentDigest[:], reasoningDigest[:]...))
	return hex.EncodeToString(digest[:])
}

func perfEvidenceCompletion(raw []byte) *agent.Completion {
	return &agent.Completion{
		Message:      agent.Message{Role: "assistant", Content: "héllo", ReasoningContent: "think!"},
		FinishReason: "stop",
		Usage: agent.Usage{
			PromptTokens:        99,
			CompletionTokens:    8,
			PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 77},
		},
		Raw: raw,
	}
}

func perfEvidenceSSE(split bool) []byte {
	frames := []string{
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":79,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\n",
	}
	if split {
		frames = append(frames,
			"data: {\"choices\":[{\"delta\":{\"content\":\"hé\",\"reasoning_content\":\"th\"},\"finish_reason\":null}]}\n\n",
			"data: {\"choices\":[{\"delta\":{\"content\":\"llo\",\"reasoning_content\":\"ink!\"},\"finish_reason\":\"stop\"}]}\n\n")
	} else {
		frames = append(frames, "data: {\"choices\":[{\"delta\":{\"content\":\"héllo\",\"reasoning_content\":\"think!\"},\"finish_reason\":\"stop\"}]}\n\n")
	}
	frames = append(frames,
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\n",
		"data: [DONE]\n\n")
	return []byte(strings.Join(frames, ""))
}

func perfEvidenceBuffered() []byte {
	return []byte("{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"héllo\",\"reasoning_content\":\"think!\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":4}}}")
}

func perfEvidenceObserve(t *testing.T, completion *agent.Completion) perfledger.Record {
	t.Helper()
	metrics := newGatewayMetrics(time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC))
	metrics.observeCompletionServedStreamFrom(perfSource{}, localitySelfHosted, completion, 2*time.Second, 250*time.Millisecond)
	records, _, _ := metrics.perfRecordsSnapshot()
	if len(records) != 1 {
		t.Fatalf("production observation did not reach the bounded ring: %d rows", len(records))
	}
	return records[0]
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=200ms lane=default
func TestPerfEvidenceCompleteStreamIgnoresChunkingAndUsesLastUsage(t *testing.T) {
	first := perfEvidenceObserve(t, perfEvidenceCompletion(perfEvidenceSSE(false)))
	second := perfEvidenceObserve(t, perfEvidenceCompletion(perfEvidenceSSE(true)))
	for _, row := range []perfledger.Record{first, second} {
		evidence := row.BenchEvidence
		if evidence == nil || !evidence.Complete || evidence.CacheState != "reused" || evidence.OutputSHA256 != perfEvidenceFingerprint("héllo", "think!") {
			t.Fatalf("completed raw stream lost trustworthy evidence: %+v", evidence)
		}
		if evidence.PromptTokens == nil || *evidence.PromptTokens != 10 || evidence.CachedTokens == nil || *evidence.CachedTokens != 4 || evidence.Tokens == nil || *evidence.Tokens != 3 {
			t.Fatalf("stream did not retain actual last usage: %+v", evidence)
		}
		if evidence.TokenSHA != "" {
			t.Fatal("assembled text fabricated token proof")
		}
		if row.PromptTokens != 99-77 || row.CompletionTokens != 8 || row.CachedTokens != 77 {
			t.Fatalf("new evidence silently rewrote legacy accounting: %+v", row)
		}
	}
	if first.BenchEvidence.OutputSHA256 != second.BenchEvidence.OutputSHA256 {
		t.Fatal("SSE transport chunking changed complete output identity")
	}
}

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=200ms lane=default
func TestPerfEvidenceCompleteStreamRequiresDoneAndNoError(t *testing.T) {
	for _, tc := range []struct {
		name, raw, finish string
	}{
		{"partial", strings.TrimSuffix(string(perfEvidenceSSE(true)), "data: [DONE]\n\n"), "stop"},
		{"error-frame", strings.Replace(string(perfEvidenceSSE(true)), "data: [DONE]", "data: {\"error\":{\"type\":\"server_error\",\"message\":\"fixture failure\"}}\n\ndata: [DONE]", 1), "stop"},
		{"malformed-frame", strings.Replace(string(perfEvidenceSSE(true)), "data: [DONE]", "data: {\n\ndata: [DONE]", 1), "stop"},
		{"unknown-finish", string(perfEvidenceSSE(true)), "server_crashed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			completion := perfEvidenceCompletion([]byte(tc.raw))
			completion.FinishReason = tc.finish
			evidence := perfEvidenceObserve(t, completion).BenchEvidence
			if evidence == nil || evidence.Complete || evidence.OutputSHA256 != "" || evidence.TokenSHA != "" || evidence.CacheState != "unverified" {
				t.Fatalf("partial or failed stream became complete cache/fingerprint evidence: %+v", evidence)
			}
		})
	}
}

// fak-test:justify why=contract when=changed:internal/gateway/**
// fak-test:runtime fast est=200ms lane=default
func TestPerfEvidenceCompleteStreamDistinguishesMissingRawAndInvalidUsage(t *testing.T) {
	withoutRaw := perfEvidenceObserve(t, perfEvidenceCompletion(nil)).BenchEvidence
	if withoutRaw == nil || !withoutRaw.Complete || withoutRaw.OutputSHA256 != perfEvidenceFingerprint("héllo", "think!") || withoutRaw.PromptTokens != nil || withoutRaw.CachedTokens != nil || withoutRaw.Tokens != nil || withoutRaw.CacheState != "unverified" {
		t.Fatalf("native completion fabricated raw usage or lost complete output: %+v", withoutRaw)
	}
	for _, raw := range []string{
		strings.Replace(string(perfEvidenceBuffered()), "\"cached_tokens\":4", "\"cached_tokens\":true", 1),
		strings.Replace(string(perfEvidenceBuffered()), "\"cached_tokens\":4", "\"cached_tokens\":11", 1),
		strings.Replace(string(perfEvidenceBuffered()), "\"prompt_tokens\":10", "\"prompt_tokens\":10.0", 1),
	} {
		evidence := perfEvidenceObserve(t, perfEvidenceCompletion([]byte(raw))).BenchEvidence
		if evidence == nil || !evidence.Complete || evidence.CacheState != "unverified" || evidence.OutputSHA256 == "" {
			t.Fatalf("invalid cache usage became verified or erased completed text identity: %+v", evidence)
		}
	}
}

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=200ms lane=default
func TestPerfEvidenceCompleteStreamWriteFailureCannotReattachSuccess(t *testing.T) {
	metrics := newGatewayMetrics(time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC))
	metrics.setHTTPWriteTimeout(time.Millisecond)
	metrics.observeCompletionServedStreamFrom(perfSource{}, localitySelfHosted, perfEvidenceCompletion(perfEvidenceBuffered()), 2*time.Millisecond, 0)
	records, _, _ := metrics.perfRecordsSnapshot()
	if len(records) != 1 || records[0].Error == "" {
		t.Fatalf("known client write failure did not produce a failure row: %+v", records)
	}
	evidence := records[0].BenchEvidence
	if evidence != nil && (evidence.Complete || evidence.OutputSHA256 != "" || evidence.TokenSHA != "" || evidence.CacheState != "unverified") {
		t.Fatalf("failure row reattached complete evidence after classification: %+v", evidence)
	}
}

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=200ms lane=default
func TestPerfEvidenceCompleteStreamOutputBoundAtObservedSeam(t *testing.T) {
	for _, field := range []string{"content", "reasoning"} {
		t.Run(field, func(t *testing.T) {
			completion := perfEvidenceCompletion(perfEvidenceBuffered())
			oversized := strings.Repeat("x", perfledger.MaxBenchEvidenceOutputBytes+1)
			if field == "content" {
				completion.Message.Content = oversized
			} else {
				completion.Message.ReasoningContent = oversized
			}
			evidence := perfEvidenceObserve(t, completion).BenchEvidence
			if evidence == nil || evidence.Complete || evidence.CacheState != "unverified" || evidence.OutputSHA256 != "" || evidence.TokenSHA != "" {
				t.Fatalf("bounded production observer promoted oversized assembled output: %+v", evidence)
			}
			if evidence.PromptTokens == nil || *evidence.PromptTokens != 10 || evidence.CachedTokens == nil || *evidence.CachedTokens != 4 || evidence.Tokens == nil || *evidence.Tokens != 3 {
				t.Fatalf("bounded production observer erased raw usage: %+v", evidence)
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=200ms lane=default
func TestPerfEvidenceCompleteStreamTokenProofUsesLastExplicitValue(t *testing.T) {
	const tokenProof = "abcdef123456"
	for _, tc := range []struct {
		name, object, want string
	}{
		{"valid", `{"token_sha":"abcdef123456"}`, tokenProof},
		{"duplicate-null", `{"token_sha":"abcdef123456","token_sha":null}`, ""},
		{"duplicate-bool", `{"token_sha":"abcdef123456","token_sha":true}`, ""},
		{"null-object", `null`, ""},
		{"bool-object", `true`, ""},
	} {
		t.Run("buffered/"+tc.name, func(t *testing.T) {
			raw := strings.TrimSuffix(string(perfEvidenceBuffered()), "}") + `,"tensorfold":` + tc.object + "}"
			evidence := perfEvidenceObserve(t, perfEvidenceCompletion([]byte(raw))).BenchEvidence
			if evidence == nil || !evidence.Complete || evidence.OutputSHA256 != perfEvidenceFingerprint("héllo", "think!") || evidence.TokenSHA != tc.want {
				t.Fatalf("last explicit token proof was not kept honest: %+v", evidence)
			}
		})
	}
	for _, tc := range []struct {
		name, final, want string
	}{
		{"valid", `{"token_sha":"abcdef123456"}`, tokenProof},
		{"null", `{"token_sha":null}`, ""},
		{"bool", `{"token_sha":true}`, ""},
		{"missing", `{}`, ""},
	} {
		t.Run("stream/"+tc.name, func(t *testing.T) {
			frames := `data: {"tensorfold":{"token_sha":"abcdef123456"}}` + "\n\n" + `data: {"tensorfold":` + tc.final + "}\n\n" + "data: [DONE]"
			raw := strings.Replace(string(perfEvidenceSSE(true)), "data: [DONE]", frames, 1)
			evidence := perfEvidenceObserve(t, perfEvidenceCompletion([]byte(raw))).BenchEvidence
			if evidence == nil || !evidence.Complete || evidence.TokenSHA != tc.want || evidence.OutputSHA256 != perfEvidenceFingerprint("héllo", "think!") {
				t.Fatalf("earlier stream token proof survived an unknown final observation: %+v", evidence)
			}
		})
	}
}

type perfEvidencePlanner struct {
	completion agent.Completion
}

func (p *perfEvidencePlanner) Model() string { return "test-model" }

func (p *perfEvidencePlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	result := p.completion
	result.Raw = append([]byte(nil), p.completion.Raw...)
	return &result, nil
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=200ms lane=default
func TestPerfEvidenceCompleteStreamReachesHTTPAndDurableLedger(t *testing.T) {
	server := newTestServer(t)
	server.planner = &perfEvidencePlanner{completion: *perfEvidenceCompletion(perfEvidenceBuffered())}
	path := filepath.Join(t.TempDir(), "perf.jsonl")
	writer := perfledger.OpenWriter(path, perfledger.DefaultMaxBytes)
	closed := false
	t.Cleanup(func() {
		if !closed {
			if err := writer.Close(); err != nil {
				t.Errorf("close ledger: %v", err)
			}
		}
	})
	server.SetPerfLedger(writer, nil, false)
	httpServer := httptest.NewServer(server.Handler())
	t.Cleanup(httpServer.Close)
	var response ChatResponse
	postJSON(t, httpServer.URL+"/v1/chat/completions", ChatRequest{Model: "test-model", Messages: []agent.Message{{Role: "user", Content: "fixture prompt"}}}, &response)
	report := decodePerfReport(t, getPerfRecent(t, server, ""))
	if len(report.Records) != 1 || report.Records[0].Error != "" || report.Records[0].BenchEvidence == nil || !report.Records[0].BenchEvidence.Complete {
		t.Fatalf("actual route did not emit complete evidence into its bounded reader: %+v", report.Records)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	records, _ := perfledger.ReadTail(path)
	if len(records) != 1 || records[0].BenchEvidence == nil || records[0].BenchEvidence.OutputSHA256 != perfEvidenceFingerprint("héllo", "think!") {
		t.Fatalf("actual gateway receipt did not persist output identity: %+v", records)
	}
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=500ms lane=default
func TestPerfEvidenceFullTokenDigestUsesLastExplicitValue(t *testing.T) {
	full := "abcdef123456" + strings.Repeat("a", 52)
	valid := `{"token_sha":"abcdef123456","token_sha256":"` + full + `"}`
	for _, tc := range []struct{ name, object, want string }{
		{"valid", valid, full},
		{"full-alone", `{"token_sha256":"` + full + `"}`, full},
		{"legacy-only", `{"token_sha":"abcdef123456"}`, ""},
		{"null", `{"token_sha256":null}`, ""},
		{"boolean", `{"token_sha256":true}`, ""},
		{"number", `{"token_sha256":7}`, ""},
		{"object", `{"token_sha256":{}}`, ""},
		{"array", `{"token_sha256":[]}`, ""},
		{"duplicate-null", `{"token_sha256":"` + full + `","token_sha256":null}`, ""},
		{"duplicate-bool", `{"token_sha256":"` + full + `","token_sha256":false}`, ""},
		{"mismatch", `{"token_sha":"123456abcdef","token_sha256":"` + full + `"}`, ""},
		{"empty", `{}`, ""},
	} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, stream), func(t *testing.T) {
				raw := strings.TrimSuffix(string(perfEvidenceBuffered()), "}") + `,"tensorfold":` + tc.object + `}`
				if stream {
					frames := `data: {"tensorfold":` + valid + "}\n\n" + `data: {"tensorfold":` + tc.object + "}\n\n" + "data: [DONE]"
					raw = strings.Replace(string(perfEvidenceSSE(true)), "data: [DONE]", frames, 1)
				}
				got := perfEvidenceObserve(t, perfEvidenceCompletion([]byte(raw))).BenchEvidence
				if got == nil || !got.Complete || got.TokenSHA256 != tc.want || got.OutputSHA256 != perfEvidenceFingerprint("héllo", "think!") {
					t.Fatalf("raw runtime field did not preserve last explicit identity: %+v", got)
				}
			})
		}
	}
	for _, raw := range []string{
		strings.Replace(string(perfEvidenceSSE(true)), "data: [DONE]", `data: {"tensorfold":`+valid+"}\n\n", 1),
		strings.Replace(string(perfEvidenceSSE(true)), "data: [DONE]", `data: {"tensorfold":`+valid+"}\n\n"+`data: {"error":{"message":"fixture failure"}}`+"\n\ndata: [DONE]", 1),
	} {
		got := perfEvidenceObserve(t, perfEvidenceCompletion([]byte(raw))).BenchEvidence
		if got == nil || got.Complete || got.TokenSHA256 != "" {
			t.Fatal("incomplete/error stream retained full token proof")
		}
	}
}

// fak-test:justify why=integration when=changed:internal/gateway/**
// fak-test:runtime fast est=1s lane=default
func TestPerfEvidenceFullTokenDigestReachesHTTPAndDurableLedger(t *testing.T) {
	prefix := "abcdef123456"
	for _, tc := range []struct {
		name, full, want string
		complete         bool
	}{
		{"first", prefix + strings.Repeat("a", 52), prefix + strings.Repeat("a", 52), true},
		{"same-prefix-different-full", prefix + strings.Repeat("b", 52), prefix + strings.Repeat("b", 52), true},
		{"short-is-not-full", prefix, "", true},
		{"incomplete", prefix + strings.Repeat("a", 52), "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := newTestServer(t)
			metadata := `data: {"tensorfold":{"token_sha":"` + prefix + `","token_sha256":"` + tc.full + "\"}}\n\n"
			raw := strings.Replace(string(perfEvidenceSSE(true)), "data: [DONE]", metadata+"data: [DONE]", 1)
			if !tc.complete {
				raw = strings.Replace(raw, "data: [DONE]", "", 1)
			}
			server.planner = &perfEvidencePlanner{completion: *perfEvidenceCompletion([]byte(raw))}
			path := filepath.Join(t.TempDir(), "perf.jsonl")
			writer := perfledger.OpenWriter(path, perfledger.DefaultMaxBytes)
			t.Cleanup(func() { _ = writer.Close() })
			server.SetPerfLedger(writer, nil, false)
			httpServer := httptest.NewServer(server.Handler())
			t.Cleanup(httpServer.Close)
			var response ChatResponse
			postJSON(t, httpServer.URL+"/v1/chat/completions", ChatRequest{Model: "test-model", Messages: []agent.Message{{Role: "user", Content: "fixture prompt"}}}, &response)
			report := decodePerfReport(t, getPerfRecent(t, server, ""))
			if len(report.Records) != 1 {
				t.Fatal("ordinary HTTP path did not record request")
			}
			if err := writer.Close(); err != nil {
				t.Fatal(err)
			}
			persisted, _ := perfledger.ReadTail(path)
			if len(persisted) != 1 {
				t.Fatal("ordinary HTTP evidence did not reach durable ledger")
			}
			for _, record := range []perfledger.Record{report.Records[0], persisted[0]} {
				if record.BenchEvidence == nil || record.BenchEvidence.TokenSHA256 != tc.want || record.BenchEvidence.Complete != tc.complete {
					t.Fatalf("ordinary caller changed token identity: %+v", record.BenchEvidence)
				}
				encoded, err := json.Marshal(record.BenchEvidence)
				if err != nil {
					t.Fatal(err)
				}
				var fields map[string]json.RawMessage
				if json.Unmarshal(encoded, &fields) != nil {
					t.Fatal("invalid serialized evidence")
				}
				if (len(fields["token_sha256"]) > 0) != (tc.want != "") {
					t.Fatal("serialization manufactured or dropped full token proof")
				}
			}
		})
	}
}
