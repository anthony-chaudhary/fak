package main

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

func TestValueChainSupportSpine(t *testing.T) {
	root := findRepoRoot("../..")
	var out, errOut bytes.Buffer
	code := runValueChain(&out, &errOut, []string{"audit", "--manifest", filepath.Join(root, "examples", "value-chain", "support-manifest.json"), "--observations", filepath.Join(root, "examples", "value-chain", "support-observations.json")})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"arm=baseline", "$/ticket_resolved=2.000000", "arm=shared", "sessions=2", "$/ticket_resolved=0.600000", "stage=gpu kind=hardware status=ABSENT", "design=paired"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("missing %q in\n%s", want, out.String())
		}
	}
}
func TestValueChainAgenticPacketAdapter(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "manifest.json")
	observations := filepath.Join(dir, "observations.json")
	packet := filepath.Join(dir, "packet.json")
	mustWrite := func(path, body string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	mustWrite(manifest, `{"schema":"fak-value-chain/1","name":"latest-harness","stages":[{"id":"benchmark","kind":"agentic-benchmark"}],"arms":[{"id":"raw","default":true},{"id":"fak"}],"outcomes":[{"id":"safe_success","unit":"case"}]}`)
	mustWrite(observations, `{"schema":"fak-value-chain/1","observations":[]}`)
	mustWrite(packet, `{"schema":"fak.agentic-benchmark-result-packet.v1","status":"PASS_RESULT","result_claim_allowed":true,"benchmark_native":true,"same_task_ids":true,"same_model":true,"same_budget":true,"official_grader":{"available":true},"arms":[{"role":"raw"},{"role":"fak"}],"metric_categories":{"task_success":true,"safe_success":true,"cost_or_token_budget":true,"latency":true,"policy_events":true,"evidence_completeness":true},"artifacts":["raw.json","fak.json"],"value_chain":[{"role":"raw","trace_id":"r","pair_id":"case-1","turns":5,"outcomes":{"safe_success":1},"provenance":"official-grader"},{"role":"fak","trace_id":"f","pair_id":"case-1","turns":3,"cost_usd":0.3,"outcomes":{"safe_success":1},"provenance":"official-grader+bill"}]}`)
	var out, errOut bytes.Buffer
	if code := runValueChain(&out, &errOut, []string{"audit", "--manifest", manifest, "--observations", observations, "--agentic-packet", packet}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	for _, want := range []string{"arm=raw", "$/turn=UNKNOWN", "arm=fak", "$/safe_success=0.300000", "design=paired"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("missing %q in %s", want, out.String())
		}
	}
}

func TestValueChainUsageLedgerAndWeeklyFold(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "usage.jsonl")
	root := repoRoot()
	args := []string{"audit", "--manifest", filepath.Join(root, "examples", "value-chain", "support-manifest.json"), "--observations", filepath.Join(root, "examples", "value-chain", "support-observations.json"), "--ledger", ledger}
	for range 2 {
		var out, errOut bytes.Buffer
		if code := runValueChain(&out, &errOut, args); code != 0 {
			t.Fatalf("audit code=%d stderr=%s", code, errOut.String())
		}
	}
	rows, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if got := bytes.Count(rows, []byte("\n")); got != 2 {
		t.Fatalf("ledger rows=%d, want 2; ledger=%s", got, rows)
	}
	if bytes.Contains(rows, []byte(root)) {
		t.Fatalf("ledger leaks repository path: %s", rows)
	}
	var out, errOut bytes.Buffer
	if code := runValueChain(&out, &errOut, []string{"usage", "--ledger", ledger}); code != 0 {
		t.Fatalf("usage code=%d stderr=%s", code, errOut.String())
	}
	if got := out.String(); !regexp.MustCompile(`^week=\d{4}-W\d{2} invocations=2\n$`).MatchString(got) {
		t.Fatalf("usage fold=%q", got)
	}
}

func TestValueChainLedgerIgnoresLegacyEnvironment(t *testing.T) {
	legacy := filepath.Join(t.TempDir(), "legacy.jsonl")
	t.Setenv("FAK_VALUE_CHAIN_LEDGER", legacy)
	if got := defaultValueChainLedger(); got == legacy {
		t.Fatalf("legacy environment still controls ledger path: %s", got)
	}
}

func TestValueChainSelfcheckMatchesCapturedWitness(t *testing.T) {
	root := repoRoot()
	var out, errOut bytes.Buffer
	code := runValueChain(&out, &errOut, []string{
		"audit",
		"--manifest", filepath.Join(root, "examples", "value-chain", "support-manifest.json"),
		"--observations", filepath.Join(root, "examples", "value-chain", "support-observations.json"),
		"--selfcheck",
		"--expect", filepath.Join(root, "examples", "value-chain", "support-witness.txt"),
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "comparison=baseline->shared design=paired paired=1") {
		t.Fatalf("captured output missing paired result: %s", out.String())
	}
}

func TestValueChainSelfcheckRejectsStaleWitness(t *testing.T) {
	root := repoRoot()
	stale := filepath.Join(t.TempDir(), "stale.txt")
	if err := os.WriteFile(stale, []byte("stale\n"), 0644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	code := runValueChain(&out, &errOut, []string{
		"audit",
		"--manifest", filepath.Join(root, "examples", "value-chain", "support-manifest.json"),
		"--observations", filepath.Join(root, "examples", "value-chain", "support-observations.json"),
		"--selfcheck", "--expect", stale,
	})
	if code != 1 || !strings.Contains(errOut.String(), "selfcheck mismatch") {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}

func TestDispatchTickAutomaticallyRecordsValueChainUsage(t *testing.T) {
	root := t.TempDir()
	payload := map[string]any{"action": "would_spawn"}
	receipt, err := recordDispatchValueChainUsage(root, payload, time.Date(2026, 8, 26, 8, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if receipt["automatic"] != true || receipt["outcome"] != "would_spawn" {
		t.Fatalf("receipt = %#v", receipt)
	}
	ledger, _ := receipt["ledger"].(string)
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"outcome":"would_spawn"`) {
		t.Fatalf("ledger = %s", data)
	}
}
