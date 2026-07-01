package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFusedExplainShowsBothFamiliesOnOneFloor(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runFused(&out, &errb, []string{"explain"}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}
	s := out.String()
	// One turn spawned both families...
	for _, want := range []string{"classical", "weight", "fused: true"} {
		if !strings.Contains(s, want) {
			t.Errorf("explain output missing %q:\n%s", want, s)
		}
	}
	// ...and the SAME floor allowed a benign op of each family and denied the destructive one.
	if !strings.Contains(s, "read_file") || !strings.Contains(s, "chat_completion") {
		t.Errorf("explain missing the demo ops:\n%s", s)
	}
	if !strings.Contains(s, "allow") || !strings.Contains(s, "deny") {
		t.Errorf("explain should show both an allow and a deny from one floor:\n%s", s)
	}
	if !strings.Contains(s, "[classical weight]") {
		t.Errorf("explain should report both families governed by the same kernel:\n%s", s)
	}
}

func TestFusedClassifyJSONSummary(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runFused(&out, &errb, []string{"classify", "--json"}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}
	var s struct {
		Schema    string `json:"schema"`
		Ops       int    `json:"ops"`
		Classical int    `json:"classical"`
		Weight    int    `json:"weight"`
		Fused     bool   `json:"fused"`
	}
	if err := json.Unmarshal(out.Bytes(), &s); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out.String())
	}
	if s.Schema != "fak.fusedturn.v1" {
		t.Errorf("schema = %q", s.Schema)
	}
	if !s.Fused || s.Classical < 1 || s.Weight < 1 {
		t.Errorf("built-in demo turn should be fused with both families: %+v", s)
	}
}

func TestFusedClassifyFromFile(t *testing.T) {
	// A turn of only classical ops is NOT fused — the CLI must say so.
	dir := t.TempDir()
	path := filepath.Join(dir, "ops.json")
	if err := os.WriteFile(path, []byte(`{"ops":[
		{"tool":"read_file","class":"classical"},
		{"tool":"git_commit","class":"classical"}
	]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runFused(&out, &errb, []string{"classify", "--file", path, "--json"}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}
	var s struct {
		Classical int  `json:"classical"`
		Weight    int  `json:"weight"`
		Fused     bool `json:"fused"`
	}
	if err := json.Unmarshal(out.Bytes(), &s); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out.String())
	}
	if s.Fused || s.Classical != 2 || s.Weight != 0 {
		t.Errorf("classical-only turn should not be fused: %+v", s)
	}
}

func TestFusedRunExecutesBothFamiliesThroughKernel(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runFused(&out, &errb, []string{"run", "--json"}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}

	var report struct {
		Summary struct {
			Classical int  `json:"classical"`
			Weight    int  `json:"weight"`
			Fused     bool `json:"fused"`
		} `json:"summary"`
		EngineCalls int `json:"engine_calls"`
		Ops         []struct {
			Tool    string `json:"tool"`
			Class   string `json:"class"`
			Verdict string `json:"verdict"`
			Status  string `json:"status"`
			Result  string `json:"result"`
		} `json:"ops"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out.String())
	}
	if !report.Summary.Fused || report.Summary.Classical < 1 || report.Summary.Weight < 1 {
		t.Fatalf("run should report a fused turn spanning both families: %+v", report.Summary)
	}
	if report.EngineCalls < 2 {
		t.Fatalf("run should dispatch both calls through Engine.Complete, engine_calls=%d", report.EngineCalls)
	}

	var classicalResult, weightResult bool
	for _, op := range report.Ops {
		if op.Verdict != "allow" {
			t.Errorf("%s verdict = %q, want allow", op.Tool, op.Verdict)
		}
		if op.Status != "ok" {
			t.Errorf("%s status = %q, want ok", op.Tool, op.Status)
		}
		if op.Result == "" {
			t.Errorf("%s should carry a real result payload", op.Tool)
		}
		switch op.Class {
		case "classical":
			classicalResult = classicalResult || strings.Contains(op.Result, "classical result:")
		case "weight":
			weightResult = weightResult || strings.Contains(op.Result, "weight result:")
		}
	}
	if !classicalResult || !weightResult {
		t.Fatalf("run should produce real results for both families; classical=%v weight=%v output=%s",
			classicalResult, weightResult, out.String())
	}
}

func TestFusedUnknownSubcommand(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runFused(&out, &errb, []string{"bogus"}); rc != 2 {
		t.Fatalf("rc = %d, want 2 for an unknown subcommand", rc)
	}
	if rc := runFused(&out, &errb, nil); rc != 2 {
		t.Fatalf("rc = %d, want 2 for no subcommand", rc)
	}
}
