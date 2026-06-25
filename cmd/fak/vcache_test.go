package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

func TestRunVCacheStatusReportsM5AndRemainingIssues(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"status"}); code != 0 {
		t.Fatalf("status exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	for _, want := range []string{
		"vCache M5 governor: up",
		"codex-like star proof: PROVEN",
		"codex/openai verifier: ready",
		"codex/openai cached-token sample: PROVEN saved 1728.0 / 2006.0",
		"codex/openai zero-cache sample: REFUTED saved 0.0 / 2006.0",
		"remaining: #716 #717 #718 #719 #727",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("status missing %q:\n%s", want, s)
		}
	}
}

func TestRunVCacheStatusJSONIncludesCodexOpenAIProofs(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"status", "--json"}); code != 0 {
		t.Fatalf("status --json exit=%d stderr=%s", code, errb.String())
	}
	var rep vcacheStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if rep.CodexOpenAI.Verifier != "ready" || rep.CodexOpenAI.Issue == "" {
		t.Fatalf("codex_openai status missing verifier/issue: %+v", rep.CodexOpenAI)
	}
	if rep.CodexOpenAI.CachedSampleProof.Status != vcachegov.ProofProven ||
		rep.CodexOpenAI.CachedSampleProof.SavedTokenEquiv != 1728 {
		t.Fatalf("cached sample proof = %+v, want OpenAI cached-token savings", rep.CodexOpenAI.CachedSampleProof)
	}
	if rep.CodexOpenAI.NoCacheRefutation.Status != vcachegov.ProofRefuted ||
		rep.CodexOpenAI.NoCacheRefutation.SavedTokenEquiv != 0 {
		t.Fatalf("no-cache proof = %+v, want zero-cache refutation", rep.CodexOpenAI.NoCacheRefutation)
	}
}

func TestRunVCacheProveDefaultCodexLikeWorkload(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"prove"}); code != 0 {
		t.Fatalf("prove exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	s := out.String()
	if !strings.Contains(s, "status: PROVEN") || !strings.Contains(s, "saved token-equiv:") {
		t.Fatalf("prove output unexpected:\n%s", s)
	}
	if !strings.Contains(s, "correctness depends on cache hit: false") {
		t.Fatalf("prove must print the Law-A correctness fence:\n%s", s)
	}
}

func TestRunVCacheProveRefutesBelowMinimum(t *testing.T) {
	var out, errb bytes.Buffer
	code := runVCache(&out, &errb, []string{"prove", "--anchor-tokens", "512"})
	if code != 1 {
		t.Fatalf("refuted proof exit=%d, want 1; stderr=%s output=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "status: REFUTED") ||
		!strings.Contains(out.String(), "below the provider minimum") {
		t.Fatalf("refuted output unexpected:\n%s", out.String())
	}
}

func TestRunVCacheProveJSON(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"prove", "--json"}); code != 0 {
		t.Fatalf("json prove exit=%d stderr=%s", code, errb.String())
	}
	var proof vcachegov.StarSavingsProof
	if err := json.Unmarshal(out.Bytes(), &proof); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if proof.Status != vcachegov.ProofProven || proof.SavedTokenEquiv <= 0 {
		t.Fatalf("proof = %+v, want proven positive savings", proof)
	}
}

func TestRunVCacheProveTelemetryClaudeProbe(t *testing.T) {
	path := filepath.Join(t.TempDir(), "claude.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join([]string{
		`{"input_tokens":10098,"cache_creation_input_tokens":59400,"cache_read_input_tokens":0,"ephemeral_1h_input_tokens":59400,"ephemeral_5m_input_tokens":0}`,
		`{"input_tokens":10065,"cache_creation_input_tokens":15411,"cache_read_input_tokens":43995,"ephemeral_1h_input_tokens":15411,"ephemeral_5m_input_tokens":0}`,
		`{"input_tokens":10065,"cache_creation_input_tokens":15410,"cache_read_input_tokens":43995,"ephemeral_1h_input_tokens":15410,"ephemeral_5m_input_tokens":0}`,
		`{"input_tokens":10065,"cache_creation_input_tokens":15424,"cache_read_input_tokens":43995,"ephemeral_1h_input_tokens":15424,"ephemeral_5m_input_tokens":0}`,
	}, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"prove-telemetry", "--file", path}); code != 0 {
		t.Fatalf("prove-telemetry exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	s := out.String()
	for _, want := range []string{
		"status: PROVEN",
		"saved token-equiv: 13141.5",
		"first positive request: 4",
		"correctness depends on cache hit: false",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("prove-telemetry missing %q:\n%s", want, s)
		}
	}
}

func TestRunVCacheProveTelemetryOpenAIResponsesUsage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "openai-responses.jsonl")
	if err := os.WriteFile(path, []byte(`{"usage":{"input_tokens":2006,"input_tokens_details":{"cached_tokens":1920}}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"prove-telemetry", "--file", path, "--read-mult", "0.1"}); code != 0 {
		t.Fatalf("openai prove-telemetry exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	s := out.String()
	for _, want := range []string{
		"status: PROVEN",
		"baseline token-equiv: 2006.0",
		"actual token-equiv: 278.0",
		"saved token-equiv: 1728.0",
		"cache read/write tokens: 1920 / 0",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("openai prove-telemetry missing %q:\n%s", want, s)
		}
	}
}

func TestReadVCacheTelemetryOpenAIChatUsage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "openai-chat.jsonl")
	if err := os.WriteFile(path, []byte(`{"usage":{"prompt_tokens":2006,"prompt_tokens_details":{"cached_tokens":1920}}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	rows, err := readVCacheTelemetry(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows=%d, want 1", len(rows))
	}
	if rows[0].InputTokens != 86 || rows[0].CacheReadInputTokens != 1920 || rows[0].CacheCreationInputTokens != 0 {
		t.Fatalf("row=%+v, want OpenAI total split into uncached=86 cached=1920", rows[0])
	}
}

func TestRunVCacheProveTelemetryCodexTokenCountEvents(t *testing.T) {
	path := filepath.Join(t.TempDir(), "codex-session.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join([]string{
		`{"type":"response_item","payload":{"type":"message","content":[]}}`,
		`{"type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":2006,"cached_input_tokens":1920,"output_tokens":12}}}}`,
		`{"type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":2006,"cached_input_tokens":0,"output_tokens":12}}}}`,
	}, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"prove-telemetry", "--file", path, "--read-mult", "0.1"}); code != 0 {
		t.Fatalf("codex prove-telemetry exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	s := out.String()
	for _, want := range []string{
		"status: PROVEN",
		"requests: 2",
		"baseline token-equiv: 4012.0",
		"actual token-equiv: 2284.0",
		"saved token-equiv: 1728.0",
		"cache read/write tokens: 1920 / 0",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("codex prove-telemetry missing %q:\n%s", want, s)
		}
	}
}

func TestReadVCacheTelemetrySkipsNonTelemetryJSONL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join([]string{
		`{"type":"response_item","payload":{"type":"message"}}`,
		`{"type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":100,"cached_input_tokens":50}}}}`,
	}, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}

	rows, err := readVCacheTelemetry(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows=%d, want only telemetry rows", len(rows))
	}
	if rows[0].InputTokens != 50 || rows[0].CacheReadInputTokens != 50 {
		t.Fatalf("row=%+v, want Codex cached_input_tokens split", rows[0])
	}
}

func TestRunVCacheProveTelemetryRefutesFirstThreeClaudeProbeTurns(t *testing.T) {
	path := filepath.Join(t.TempDir(), "claude-three.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join([]string{
		`{"input_tokens":10098,"cache_creation_input_tokens":59400,"cache_read_input_tokens":0,"ephemeral_1h_input_tokens":59400}`,
		`{"input_tokens":10065,"cache_creation_input_tokens":15411,"cache_read_input_tokens":43995,"ephemeral_1h_input_tokens":15411}`,
		`{"input_tokens":10065,"cache_creation_input_tokens":15410,"cache_read_input_tokens":43995,"ephemeral_1h_input_tokens":15410}`,
	}, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runVCache(&out, &errb, []string{"prove-telemetry", "--file", path})
	if code != 1 {
		t.Fatalf("three-turn prove-telemetry exit=%d, want 1; stderr=%s output=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "status: REFUTED") ||
		!strings.Contains(out.String(), "did not repay cache write cost") {
		t.Fatalf("three-turn output unexpected:\n%s", out.String())
	}
}
