package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

const svLedger = `{"schema":"fak-skill-value-ledger/1","session_id":"s1","task_class":"refactor","skills":["helper","harmful"],"pass":true,"cost_usd":1,"latency_ms":100}
{"schema":"fak-skill-value-ledger/1","session_id":"s2","task_class":"refactor","skills":["helper"],"pass":true,"cost_usd":1,"latency_ms":100}
{"schema":"fak-skill-value-ledger/1","session_id":"s3","task_class":"refactor","skills":["harmful"],"pass":false,"cost_usd":3,"latency_ms":500}
{"schema":"fak-skill-value-ledger/1","session_id":"s4","task_class":"refactor","skills":[],"pass":false,"cost_usd":2,"latency_ms":300}
`

func TestRunSkillValueReport(t *testing.T) {
	ledger := writeTemp(t, "skill-value.jsonl", svLedger)
	basis := writeTemp(t, "basis.jsonl", `{"skill_id":"helper","valuation_basis":"ablation:matched-pass-delta"}`+"\n")

	var out, errb bytes.Buffer
	code := runSkillValue(&out, &errb, []string{"--ledger", ledger, "--basis", basis})
	if code != 0 {
		t.Fatalf("report exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "helper") || !strings.Contains(s, "harmful") {
		t.Fatalf("report missing skills:\n%s", s)
	}
	// harmful has measured lift < 0 -> must be in the auto-revert line.
	if !strings.Contains(s, "auto-revert") || !strings.Contains(s, "harmful") {
		t.Fatalf("harmful should be auto-reverted:\n%s", s)
	}
	// harmful carries no basis -> ungrounded gate line names it.
	if !strings.Contains(s, "ungrounded") {
		t.Fatalf("expected an ungrounded gate line:\n%s", s)
	}
}

func TestRunSkillValueGateFails(t *testing.T) {
	ledger := writeTemp(t, "skill-value.jsonl", svLedger)
	// No --basis file: every active skill is ungrounded, so --gate must fail.
	var out, errb bytes.Buffer
	code := runSkillValue(&out, &errb, []string{"--ledger", ledger, "--gate"})
	if code != 1 {
		t.Fatalf("gate should fail with exit 1, got %d\nstderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "valuation-basis gate FAILED") {
		t.Fatalf("expected gate-failure message, got: %s", errb.String())
	}
}

func TestRunSkillValueGatePasses(t *testing.T) {
	ledger := writeTemp(t, "skill-value.jsonl",
		`{"schema":"fak-skill-value-ledger/1","session_id":"s1","task_class":"a","skills":["k"],"pass":true}`+"\n"+
			`{"schema":"fak-skill-value-ledger/1","session_id":"s2","task_class":"a","skills":[],"pass":true}`+"\n")
	basis := writeTemp(t, "basis.jsonl", `{"skill_id":"k","valuation_basis":"ablation"}`+"\n")
	var out, errb bytes.Buffer
	code := runSkillValue(&out, &errb, []string{"--ledger", ledger, "--basis", basis, "--gate"})
	if code != 0 {
		t.Fatalf("gate should pass, got %d\nstderr=%s", code, errb.String())
	}
}

func TestRunSkillValueNotYet(t *testing.T) {
	var out, errb bytes.Buffer
	code := runSkillValue(&out, &errb, []string{"--ledger", filepath.Join(t.TempDir(), "absent.jsonl")})
	if code != 0 {
		t.Fatalf("missing ledger should be not-yet (exit 0), got %d\nstderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "not yet") {
		t.Fatalf("expected not-yet report, got: %s", out.String())
	}
}

func TestRunSkillValueJSON(t *testing.T) {
	ledger := writeTemp(t, "skill-value.jsonl", svLedger)
	var out, errb bytes.Buffer
	code := runSkillValue(&out, &errb, []string{"--ledger", ledger, "--json"})
	if code != 0 {
		t.Fatalf("json exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "fak-skill-value-ledger/1") || !strings.Contains(s, "\"pass_lift\"") {
		t.Fatalf("json payload missing expected fields:\n%s", s)
	}
}
