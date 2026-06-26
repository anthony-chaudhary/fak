package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcachechain"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
)

func TestRunVCacheStatusReportsM5AndRemainingIssues(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"status"}); code != 0 {
		t.Fatalf("status exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	for _, want := range []string{
		"vCache M5 governor: up",
		"vCache M4 chains & recall: up",
		"M4 recall cost-gate proof: refuted",
		"codex-like star proof: PROVEN",
		"codex/openai verifier: ready",
		"codex/openai live telemetry: proven (Codex CLI replay artifact)",
		"codex/openai cached-token sample: PROVEN saved 1728.0 / 2006.0",
		"codex/openai zero-cache sample: REFUTED saved 0.0 / 2006.0",
		// #719 (M4 chains & recall) and #727 (Codex/OpenAI telemetry probe) are closed.
		"remaining: #716 #717 #718",
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

func TestRunVCacheProveRecallRefusesSingleUnit(t *testing.T) {
	// The §11.0 headline default: a 30k-token prefix replayed at r=0.1 to recall one
	// 10-token unit is a 300× LOSS, so the cost gate REFUSES it (exit 1).
	var out, errb bytes.Buffer
	code := runVCache(&out, &errb, []string{"prove-recall"})
	if code != 1 {
		t.Fatalf("single-unit prove-recall exit=%d, want 1; stderr=%s output=%s", code, errb.String(), out.String())
	}
	s := out.String()
	for _, want := range []string{
		"status: refuted",
		"decision: cold_prefill",
		"replay cost (P·r): 3000.0 token-equiv",
		"single-unit loss ratio (P·r/U): 300.0x",
		"break-even siblings: 301",
		"correctness depends on cache hit: false",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("prove-recall missing %q:\n%s", want, s)
		}
	}
}

func TestRunVCacheProveRecallAmortizedProven(t *testing.T) {
	// The amortized-fan-out exception: 401 sibling units clear the 301 break-even,
	// so rebuild is net-positive (exit 0).
	var out, errb bytes.Buffer
	code := runVCache(&out, &errb, []string{"prove-recall", "--siblings", "401"})
	if code != 0 {
		t.Fatalf("amortized prove-recall exit=%d, want 0; stderr=%s output=%s", code, errb.String(), out.String())
	}
	s := out.String()
	if !strings.Contains(s, "status: proven") || !strings.Contains(s, "decision: rebuild") {
		t.Fatalf("amortized prove-recall unexpected:\n%s", s)
	}
}

func TestRunVCacheProveRecallJSON(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"prove-recall", "--json", "--siblings", "301"}); code != 0 {
		t.Fatalf("json prove-recall exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var proof vcachechain.RecallProof
	if err := json.Unmarshal(out.Bytes(), &proof); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if proof.Status != vcachechain.ProofProven || proof.Decision != vcachechain.DecisionRebuild {
		t.Fatalf("proof = %+v, want proven/rebuild at the 301 break-even", proof)
	}
	if proof.BreakEvenSiblings != 301 || proof.LossRatio != 300 {
		t.Fatalf("economics = %+v, want break-even 301 / loss 300", proof)
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

func TestReadVCacheTelemetryCodexExecUsage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "codex-exec.jsonl")
	if err := os.WriteFile(path, []byte(`{"type":"turn.completed","usage":{"input_tokens":24763,"cached_input_tokens":24448,"output_tokens":122}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	rows, err := readVCacheTelemetry(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows=%d, want 1", len(rows))
	}
	if rows[0].InputTokens != 315 || rows[0].CacheReadInputTokens != 24448 || rows[0].CacheCreationInputTokens != 0 {
		t.Fatalf("row=%+v, want Codex exec total split into uncached=315 cached=24448", rows[0])
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

func TestRunVCacheScoreDefaultPassesTwoXGate(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"score", "--json"}); code != 0 {
		t.Fatalf("score exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if rep.Schema != "fak.vcache.score.v1" || rep.Status != "2x_ready" || !rep.TwoXBetter {
		t.Fatalf("score = %+v, want default 2x-ready report", rep)
	}
}

func TestRunVCacheScoreTelemetryJSONAndOut(t *testing.T) {
	dir := t.TempDir()
	telemetry := filepath.Join(dir, "openai.jsonl")
	if err := os.WriteFile(telemetry, []byte(`{"usage":{"input_tokens":2006,"input_tokens_details":{"cached_tokens":1920}}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	artifact := filepath.Join(dir, "nested", "score.json")
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"score", "--json", "--telemetry", telemetry, "--out", artifact}); code != 0 {
		t.Fatalf("score telemetry exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var stdoutRep vcachescore.Report
	if err := json.Unmarshal(out.Bytes(), &stdoutRep); err != nil {
		t.Fatalf("stdout is invalid json: %v\n%s", err, out.String())
	}
	if !stdoutRep.TwoXBetter || stdoutRep.ActiveSource != "telemetry" || stdoutRep.Observed == nil {
		t.Fatalf("stdout report = %+v, want telemetry-backed 2x report", stdoutRep)
	}
	var fileRep vcachescore.Report
	b, err := os.ReadFile(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &fileRep); err != nil {
		t.Fatalf("artifact is invalid json: %v\n%s", err, string(b))
	}
	if fileRep.ActiveSource != "telemetry" || fileRep.ActiveMultiplier != stdoutRep.ActiveMultiplier {
		t.Fatalf("artifact report = %+v, stdout multiplier=%g", fileRep, stdoutRep.ActiveMultiplier)
	}
}

func TestRunVCacheScoreTelemetryHumanReportsEconomics(t *testing.T) {
	telemetry := filepath.Join(t.TempDir(), "openai.jsonl")
	if err := os.WriteFile(telemetry, []byte(`{"usage":{"input_tokens":2006,"input_tokens_details":{"cached_tokens":1920}}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"score", "--telemetry", telemetry}); code != 0 {
		t.Fatalf("score telemetry exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	s := out.String()
	for _, want := range []string{
		"active source: telemetry",
		"2x gate: pass",
		"economics (telemetry, observed): hit 95.71%",
		"read 1920 cached (write 0)",
		"rebate 1728.0 (86.14%)",
		"cost 278.0 / 2006.0 baseline",
		"7.22x",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("score telemetry output missing %q:\n%s", want, s)
		}
	}
}

func TestRunVCacheScoreAnchorsFileWritesIndexArtifact(t *testing.T) {
	dir := t.TempDir()
	anchors := filepath.Join(dir, "anchors.jsonl")
	if err := os.WriteFile(anchors, []byte(strings.Join([]string{
		`{"key":"tail","weight":10}`,
		`{"key":"head","weight":60}`,
		`{"key":"mid","weight":30}`,
	}, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}
	index := filepath.Join(dir, "nested", "anchor-index.json")
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"score", "--json", "--anchors-file", anchors, "--index-out", index}); code != 0 {
		t.Fatalf("score anchors exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("stdout is invalid json: %v\n%s", err, out.String())
	}
	if rep.Index.AnchorCount != 2 || rep.Index.Coverage != 0.9 {
		t.Fatalf("report index=%+v, want top-2 covering 90%%", rep.Index)
	}
	b, err := os.ReadFile(index)
	if err != nil {
		t.Fatal(err)
	}
	var artifact vcachescore.AnchorIndexArtifact
	if err := json.Unmarshal(b, &artifact); err != nil {
		t.Fatalf("index artifact is invalid json: %v\n%s", err, string(b))
	}
	if artifact.Schema != "fak.vcache.anchor_index.v1" || len(artifact.Entries) != 2 {
		t.Fatalf("artifact=%+v, want v1 top-2 index", artifact)
	}
	if artifact.Entries[0].Key != "head" || artifact.Entries[1].Key != "mid" {
		t.Fatalf("entries=%+v, want sorted hot-anchor keys", artifact.Entries)
	}
}

func TestRunVCacheBenchAliasTelemetryJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "openai-responses.jsonl")
	if err := os.WriteFile(path, []byte(`{"usage":{"input_tokens":2006,"input_tokens_details":{"cached_tokens":1920}}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"bench", "--telemetry", path, "--json"}); code != 0 {
		t.Fatalf("bench --json exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep struct {
		Schema       string  `json:"schema"`
		ActiveSource string  `json:"active_source"`
		TwoXBetter   bool    `json:"two_x_better"`
		Multiplier   float64 `json:"active_multiplier"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if rep.Schema != "fak.vcache.score.v1" || rep.ActiveSource != "telemetry" || !rep.TwoXBetter || rep.Multiplier < 7 {
		t.Fatalf("score json = %+v, want telemetry-backed 2x score", rep)
	}
}
