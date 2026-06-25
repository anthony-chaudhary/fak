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
	for _, want := range []string{"vCache M5 governor: up", "codex-like star proof: PROVEN", "remaining: #716 #717 #718 #719 #727"} {
		if !strings.Contains(s, want) {
			t.Fatalf("status missing %q:\n%s", want, s)
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
