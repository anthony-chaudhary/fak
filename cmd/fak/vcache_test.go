package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
	"github.com/anthony-chaudhary/fak/internal/vcachechain"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
)

func TestRunVCacheStatusReportsM5AndRemainingIssues(t *testing.T) {
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(t.TempDir(), "missing.jsonl"))
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"status"}); code != 0 {
		t.Fatalf("status exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	for _, want := range []string{
		"vCache M5 governor: decision witness live",
		"pin/lazy/evict actions not registered",
		"vCache M4 chains & recall: implemented",
		"gated OFF by default; off-path",
		"M4 recall cost-gate proof: refuted",
		"context API: ready (GET /v1/fak/ctxvalue; MCP fak_context_value; advice_only=true)",
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
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(t.TempDir(), "missing.jsonl"))
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
	if rep.ContextAPI.Verifier != "ready" ||
		rep.ContextAPI.HTTP != "GET /v1/fak/ctxvalue" ||
		rep.ContextAPI.MCPTool != "fak_context_value" ||
		!rep.ContextAPI.AdviceOnly ||
		!strings.Contains(rep.ContextAPI.ScoreIntegration, "after a guard/serve context event fires") {
		t.Fatalf("context_api status missing live API contract: %+v", rep.ContextAPI)
	}
	if rep.CodexOpenAI.CachedSampleProof.Status != vcachegov.ProofProven ||
		rep.CodexOpenAI.CachedSampleProof.SavedTokenEquiv != 1728 {
		t.Fatalf("cached sample proof = %+v, want OpenAI cached-token savings", rep.CodexOpenAI.CachedSampleProof)
	}
	if !strings.Contains(rep.Governor, "decision witness live") || strings.Contains(rep.Governor, "up (pin/lazy/evict") {
		t.Fatalf("governor status should distinguish live witness from registered actions: %q", rep.Governor)
	}
	if strings.Contains(rep.Chains, "up") || !strings.Contains(rep.Chains, "off-path") {
		t.Fatalf("chains status should distinguish implemented code from a live engine: %q", rep.Chains)
	}
	if rep.CodexOpenAI.NoCacheRefutation.Status != vcachegov.ProofRefuted ||
		rep.CodexOpenAI.NoCacheRefutation.SavedTokenEquiv != 0 {
		t.Fatalf("no-cache proof = %+v, want zero-cache refutation", rep.CodexOpenAI.NoCacheRefutation)
	}
}

func TestRunVCacheStatusReadsRecentSnapshot(t *testing.T) {
	snap := filepath.Join(t.TempDir(), "vcache-turns.jsonl")
	body := `{"family":"head","unix_millis":1,"input_tokens":86,"cache_read_input_tokens":1920,"fak_context_events":1,"fak_context_shed_tokens":200}` + "\n" +
		`{"family":"head","unix_millis":2,"input_tokens":86,"cache_read_input_tokens":1920}` + "\n"
	if err := os.WriteFile(snap, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_VCACHE_SNAPSHOT", snap)

	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"status", "--json"}); code != 0 {
		t.Fatalf("status --json exit=%d stderr=%s", code, errb.String())
	}
	var rep vcacheStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if rep.RecentObservation == nil {
		t.Fatalf("status did not attach recent observation:\n%s", out.String())
	}
	recent := rep.RecentObservation
	if recent.Turns != 2 || recent.ProviderStatus != string(vcachegov.ProofProven) || recent.CacheReadTokens != 3840 {
		t.Fatalf("recent observation = %+v, want two proven cached turns", recent)
	}
	if recent.FalseWarmRate != 0 || recent.GovernorDecision == "" || recent.ContextEvents != 1 || recent.ContextShedTokens != 200 {
		t.Fatalf("recent observation missed prediction/governor/context evidence: %+v", recent)
	}
	if recent.ContextStatus != "WITNESSED" || !strings.Contains(recent.ContextReason, "fak_context_* counters") {
		t.Fatalf("recent context status = %q (%q), want WITNESSED with reason", recent.ContextStatus, recent.ContextReason)
	}

	out.Reset()
	errb.Reset()
	if code := runVCache(&out, &errb, []string{"status"}); code != 0 {
		t.Fatalf("status exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "recent snapshot: 2 turns, provider PROVEN") ||
		!strings.Contains(out.String(), "context WITNESSED (1 events)") {
		t.Fatalf("text status did not report recent snapshot:\n%s", out.String())
	}
}

func TestRunVCacheStatusExplainsProviderOnlySnapshotContextGap(t *testing.T) {
	snap := filepath.Join(t.TempDir(), "vcache-turns.jsonl")
	body := `{"family":"head","unix_millis":1,"input_tokens":86,"cache_read_input_tokens":1920}` + "\n" +
		`{"family":"head","unix_millis":2,"input_tokens":86,"cache_read_input_tokens":1920}` + "\n"
	if err := os.WriteFile(snap, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_VCACHE_SNAPSHOT", snap)

	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"status", "--json"}); code != 0 {
		t.Fatalf("status --json exit=%d stderr=%s", code, errb.String())
	}
	var rep vcacheStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if rep.RecentObservation == nil {
		t.Fatalf("status did not attach recent observation:\n%s", out.String())
	}
	recent := rep.RecentObservation
	if recent.ProviderStatus != string(vcachegov.ProofProven) || recent.ContextStatus != "MISSING" ||
		!strings.Contains(recent.ContextReason, "no fak_context_* counters") {
		t.Fatalf("provider-only context status = %+v, want provider proven with MISSING context reason", recent)
	}

	out.Reset()
	errb.Reset()
	if code := runVCache(&out, &errb, []string{"status"}); code != 0 {
		t.Fatalf("status exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "context MISSING (0 events)") {
		t.Fatalf("text status did not explain missing context counters:\n%s", out.String())
	}
}

func TestRunVCacheStatusIncludesRecentSessionSummary(t *testing.T) {
	cfg := t.TempDir()
	workspace := filepath.Join(t.TempDir(), "work", "fak")
	if err := os.MkdirAll(workspace, 0o777); err != nil {
		t.Fatal(err)
	}
	ns := sessionaudit.ProjectNamespace(workspace)
	writeSessionAuditJSONL(t, filepath.Join(cfg, "projects", ns, "heavy.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("opus", 200, 0, 900_000, 50_000, "claude-opus-4-8", ""),
	})
	writeSessionAuditJSONL(t, filepath.Join(cfg, "projects", ns, "fable.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("fable", 300, 0, 20_000, 1_000, "claude-fable-5", ""),
	})
	t.Setenv("CLAUDE_CONFIG_DIR", cfg)
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(t.TempDir(), "missing.jsonl"))
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(workspace); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(oldwd)
	})

	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"status", "--json", "--sessions", "--session-days", "-1", "--session-max", "2"}); code != 0 {
		t.Fatalf("status --sessions --json exit=%d stderr=%s", code, errb.String())
	}
	var rep vcacheStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if rep.RecentSessions == nil {
		t.Fatalf("recent session summary missing:\n%s", out.String())
	}
	summary := rep.RecentSessions
	if summary.Scope.NamespaceFilter != ns || summary.Scope.Audited != 2 || summary.Totals.TotalContextTokens != 971_000 {
		t.Fatalf("recent session summary = %+v", summary)
	}
	tiers := map[string]sessionaudit.CompactTier{}
	for _, tier := range summary.Tiers {
		tiers[tier.Tier] = tier
	}
	if tiers["fable"].OutputTokens != 300 || tiers["opus"].OutputTokens != 200 {
		t.Fatalf("recent session tiers = %+v", summary.Tiers)
	}
	if len(summary.TopLongContext) == 0 || summary.TopLongContext[0].Session != "heavy" {
		t.Fatalf("recent long-context rows = %+v", summary.TopLongContext)
	}

	out.Reset()
	errb.Reset()
	if code := runVCache(&out, &errb, []string{"status", "--sessions", "--session-days", "-1", "--session-max", "2"}); code != 0 {
		t.Fatalf("status --sessions exit=%d stderr=%s", code, errb.String())
	}
	text := out.String()
	for _, want := range []string{"recent sessions:", "fable: output 300", "opus: output 200", "top long-context: heavy"} {
		if !strings.Contains(text, want) {
			t.Fatalf("status --sessions missing %q:\n%s", want, text)
		}
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
	if proof.Schema != vcachechain.RecallProofSchema {
		t.Fatalf("schema = %q, want %q", proof.Schema, vcachechain.RecallProofSchema)
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
	if proof.Schema != vcachegov.StarSavingsSchema {
		t.Fatalf("schema = %q, want %q", proof.Schema, vcachegov.StarSavingsSchema)
	}
	if proof.Status != vcachegov.ProofProven || proof.SavedTokenEquiv <= 0 {
		t.Fatalf("proof = %+v, want proven positive savings", proof)
	}
}

func TestRunVCacheProveTelemetryJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "openai.jsonl")
	if err := os.WriteFile(path, []byte(`{"usage":{"input_tokens":2006,"input_tokens_details":{"cached_tokens":1920}}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"prove-telemetry", "--file", path, "--json"}); code != 0 {
		t.Fatalf("json prove-telemetry exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var proof vcachegov.TelemetrySavingsProof
	if err := json.Unmarshal(out.Bytes(), &proof); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if proof.Schema != vcachegov.TelemetrySavingsSchema || proof.Status != vcachegov.ProofProven {
		t.Fatalf("proof = %+v, want schema %q and proven status", proof, vcachegov.TelemetrySavingsSchema)
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
	if code := runVCache(&out, &errb, []string{"score", "--json", "--snapshot", "off"}); code != 0 {
		t.Fatalf("score exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if rep.Schema != "fak.vcache.score.v1" || rep.Status != "2x_ready" || !rep.TwoXBetter {
		t.Fatalf("score = %+v, want default 2x-ready report", rep)
	}
	if rep.AnchorSource != vcachescore.AnchorSourceSynthetic || rep.TurnsObserved != 0 {
		t.Fatalf("anchor source/turns = %q/%d, want synthetic/0", rep.AnchorSource, rep.TurnsObserved)
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
	if stdoutRep.AnchorSource != vcachescore.AnchorSourceSynthetic || stdoutRep.TurnsObserved != 1 {
		t.Fatalf("stdout anchor source/turns = %q/%d, want synthetic/1", stdoutRep.AnchorSource, stdoutRep.TurnsObserved)
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

func TestRunVCacheScoreAgenticActivationFlags(t *testing.T) {
	telemetry := filepath.Join(t.TempDir(), "openai.jsonl")
	if err := os.WriteFile(telemetry, []byte(`{"usage":{"input_tokens":2006,"input_tokens_details":{"cached_tokens":1920}}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runVCache(&out, &errb, []string{
		"score", "--json", "--telemetry", telemetry,
		"--kernel-kv-events", "2",
		"--kernel-kv-prompt-tokens", "1000",
		"--kernel-kv-reused-tokens", "900",
		"--context-events", "1",
		"--context-shed-tokens", "800",
		"--context-resident-tokens", "1200",
		"--provider-vcache-decisions", "3",
		"--external-engine-events", "4",
		"--external-engine-hit-rate", "0.67",
	})
	if code != 0 {
		t.Fatalf("score activation exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("stdout is invalid json: %v\n%s", err, out.String())
	}
	if !rep.AgenticActivation.Active || rep.AgenticActivation.Total != 10 {
		t.Fatalf("activation report=%+v, want supplied fak-authored events counted", rep.AgenticActivation)
	}
	if rep.AgenticActivation.KernelKVEvents != 2 ||
		rep.AgenticActivation.ContextEvents != 1 ||
		rep.AgenticActivation.ProviderVCacheDecisions != 3 ||
		rep.AgenticActivation.ExternalEngineEvents != 4 {
		t.Fatalf("activation counters=%+v, want exact CLI values", rep.AgenticActivation)
	}
	if rep.DefaultUsefulness.Facets.AgenticActivation != 20 {
		t.Fatalf("agentic activation facet=%d, want full credit when fak-authored activation is supplied", rep.DefaultUsefulness.Facets.AgenticActivation)
	}
	if !rep.Planes.ProviderObserved.Available || !rep.Planes.KernelWitnessed.Available || !rep.Planes.ContextWitnessed.Available {
		t.Fatalf("provider, kernel, and context planes should be available: %+v", rep.Planes)
	}
	if rep.Planes.KernelWitnessed.Provenance != "WITNESSED" || rep.Planes.KernelWitnessed.SavedTokenEquiv != 900 {
		t.Fatalf("kernel plane=%+v, want witnessed 900-token pure-fak reuse", rep.Planes.KernelWitnessed)
	}
	if rep.Planes.ContextWitnessed.Provenance != "WITNESSED" ||
		rep.Planes.ContextWitnessed.SavedTokenEquiv != 800 ||
		rep.Planes.ContextWitnessed.BaselineTokenEquiv != 2000 ||
		rep.Planes.ContextWitnessed.CostTokenEquiv != 1200 {
		t.Fatalf("context plane=%+v, want 800 saved / 2000 baseline / 1200 cost", rep.Planes.ContextWitnessed)
	}
	if !rep.Planes.ExternalEngineObserved.Available || rep.Planes.ExternalEngineObserved.HitRate != 0.67 {
		t.Fatalf("external plane=%+v, want observed 0.67 hit-rate evidence", rep.Planes.ExternalEngineObserved)
	}
}

func TestRunVCacheScoreContextSavedOnlyDoesNotInferBaseline(t *testing.T) {
	var out, errb bytes.Buffer
	code := runVCache(&out, &errb, []string{
		"score", "--json", "--snapshot", "off",
		"--context-shed-tokens", "800",
	})
	if code != 0 {
		t.Fatalf("score context exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("stdout is invalid json: %v\n%s", err, out.String())
	}
	if !rep.Planes.ContextWitnessed.Available || rep.Planes.ContextWitnessed.BaselineTokenEquiv != 0 {
		t.Fatalf("context plane=%+v, want saved-only witness without inferred baseline", rep.Planes.ContextWitnessed)
	}
	if rep.DefaultUsefulness.Facets.NetRealizedValue != 0 {
		t.Fatalf("saved-only context evidence must not earn net-value credit: %+v", rep.DefaultUsefulness)
	}
	if rep.AgenticActivation.ContextEvents != 1 {
		t.Fatalf("context events=%d, want auto event when shed tokens supplied", rep.AgenticActivation.ContextEvents)
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
		"planes: provider=OBSERVED kernel=MISSING context=MISSING external=MISSING forecast=FORECAST",
		"agentic activation: 0 events",
		"default usefulness: partial",
		"provider rebate observed",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("score telemetry output missing %q:\n%s", want, s)
		}
	}
}

// TestRunVCacheScoreObservedByDefaultFromSnapshot pins #1090: with no --telemetry, the
// score reads the persisted observed cache window (the per-turn snapshot a finished
// guard/serve session leaves) and reports the REALIZED multiplier — active source
// "telemetry", not the synthetic-Zipf "planned" forecast.
func TestRunVCacheScoreObservedByDefaultFromSnapshot(t *testing.T) {
	snap := filepath.Join(t.TempDir(), "vcache-turns.jsonl")
	// Two turns with heavy cache_read reuse — the observed window vcacheobserve.Rows folds.
	body := `{"family":"head","unix_millis":1,"input_tokens":86,"cache_read_input_tokens":1920}` + "\n" +
		`{"family":"head","unix_millis":2,"input_tokens":86,"cache_read_input_tokens":1920}` + "\n"
	if err := os.WriteFile(snap, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"score", "--json", "--snapshot", snap}); code != 0 && code != 1 {
		// exit 0/1 is the 2x gate verdict, both valid; a 2 is a real error.
		t.Fatalf("score --snapshot exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("stdout is invalid json: %v\n%s", err, out.String())
	}
	if rep.ActiveSource != "telemetry" || rep.Observed == nil {
		t.Fatalf("a snapshot with turns must flip active_source to telemetry (observed), got %+v", rep)
	}
	if rep.AnchorSource != vcachescore.AnchorSourceMeasured || rep.TurnsObserved != 2 {
		t.Fatalf("snapshot anchor source/turns = %q/%d, want measured/2", rep.AnchorSource, rep.TurnsObserved)
	}
	if rep.Index.TotalAnchors != 1 {
		t.Fatalf("snapshot index total anchors=%d, want measured one-family workload", rep.Index.TotalAnchors)
	}
	if rep.Prediction.Total != 2 || rep.Prediction.TrueWarm != 1 || rep.Prediction.FalseCold != 1 {
		t.Fatalf("snapshot prediction = %+v, want total=2 true_warm=1 false_cold=1", rep.Prediction)
	}

	// --snapshot off forces the planned FORECAST even when a snapshot exists.
	var out2, errb2 bytes.Buffer
	if code := runVCache(&out2, &errb2, []string{"score", "--json", "--snapshot", "off"}); code != 0 && code != 1 {
		t.Fatalf("score --snapshot off exit=%d stderr=%s", code, errb2.String())
	}
	var rep2 vcachescore.Report
	if err := json.Unmarshal(out2.Bytes(), &rep2); err != nil {
		t.Fatalf("stdout invalid json: %v\n%s", err, out2.String())
	}
	if rep2.ActiveSource != "planned" {
		t.Fatalf("--snapshot off must fall back to the planned forecast, got active_source=%q", rep2.ActiveSource)
	}
	if rep2.AnchorSource != vcachescore.AnchorSourceSynthetic || rep2.TurnsObserved != 0 {
		t.Fatalf("--snapshot off anchor source/turns = %q/%d, want synthetic/0", rep2.AnchorSource, rep2.TurnsObserved)
	}
}

func TestRunVCacheScoreSnapshotCarriesContextEvidence(t *testing.T) {
	snap := filepath.Join(t.TempDir(), "vcache-turns.jsonl")
	body := `{"family":"head","unix_millis":1,"input_tokens":86,"cache_read_input_tokens":1920,"fak_context_events":2,"fak_context_shed_tokens":1200,"fak_context_dropped_turns":3,"fak_context_baseline_tokens":2000,"fak_context_cost_tokens":800}` + "\n" +
		`{"family":"head","unix_millis":2,"input_tokens":86,"cache_read_input_tokens":1920}` + "\n"
	if err := os.WriteFile(snap, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"score", "--json", "--snapshot", snap}); code != 0 && code != 1 {
		t.Fatalf("score --snapshot exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("stdout is invalid json: %v\n%s", err, out.String())
	}
	if !rep.Planes.ContextWitnessed.Available || rep.Planes.ContextWitnessed.Provenance != "WITNESSED" {
		t.Fatalf("context plane = %+v, want WITNESSED evidence from snapshot", rep.Planes.ContextWitnessed)
	}
	if rep.Planes.ContextWitnessed.SavedTokenEquiv != 1200 {
		t.Fatalf("context saved token-equiv=%g, want 1200", rep.Planes.ContextWitnessed.SavedTokenEquiv)
	}
	if rep.Planes.ContextWitnessed.BaselineTokenEquiv != 2000 || rep.Planes.ContextWitnessed.CostTokenEquiv != 800 {
		t.Fatalf("context baseline/cost=%g/%g, want 2000/800",
			rep.Planes.ContextWitnessed.BaselineTokenEquiv, rep.Planes.ContextWitnessed.CostTokenEquiv)
	}
	if rep.AgenticActivation.ContextEvents != 2 || !rep.AgenticActivation.Active {
		t.Fatalf("agentic activation = %+v, want two context events active", rep.AgenticActivation)
	}
	if rep.DefaultUsefulness.Facets.AgenticActivation == 0 {
		t.Fatalf("default-usefulness missed context activation: %+v", rep.DefaultUsefulness)
	}
}

func TestRunVCacheScoreSnapshotEnvFeedsMeasuredSource(t *testing.T) {
	snap := filepath.Join(t.TempDir(), "vcache-turns.jsonl")
	body := `{"family":"head","unix_millis":1,"input_tokens":86,"cache_read_input_tokens":1920}` + "\n"
	if err := os.WriteFile(snap, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_VCACHE_SNAPSHOT", snap)
	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"score", "--json"}); code != 0 && code != 1 {
		t.Fatalf("score with FAK_VCACHE_SNAPSHOT exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("stdout is invalid json: %v\n%s", err, out.String())
	}
	if rep.AnchorSource != vcachescore.AnchorSourceMeasured || rep.TurnsObserved != 1 || rep.ActiveSource != "telemetry" {
		t.Fatalf("env snapshot report = %+v, want measured one-turn telemetry", rep)
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
	if rep.AnchorSource != vcachescore.AnchorSourceMeasured || rep.TurnsObserved != 0 {
		t.Fatalf("anchors-file source/turns = %q/%d, want measured/0", rep.AnchorSource, rep.TurnsObserved)
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
