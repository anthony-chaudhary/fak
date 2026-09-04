// Package main tests for the behavior the caveman-pairwise-judge CLI drives:
// the cmd's own helpers (bindSafety, exitFor, writeJSON, mustRead), the strict
// judge-output parser the run depends on, and the protocol-2 flip diagnosis
// over the frozen v1 receipt. No network, no model calls.
package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cavemanpairwise"
)

const frozenV1Receipt = "../../docs/_witnesses/caveman-pairwise-judge/receipt.json"

func TestParseJudgment(t *testing.T) {
	valid := `{"verdict":"A","scores":{"factual_correctness":{"A":3,"B":1},"required_constraints":{"A":2,"B":2},"instruction_adherence":{"A":4,"B":0},"safety":{"A":4,"B":4},"justified_answering":{"A":1,"B":3}},"evidence":["a is grounded","b contradicts the prompt"]}`
	tests := []struct {
		name    string
		input   string
		wantErr string
	}{
		{"well-formed judgment accepted", valid, ""},
		{"invalid verdict refused", strings.Replace(valid, `"verdict":"A"`, `"verdict":"C"`, 1), "invalid verdict"},
		{"missing criterion refused", strings.Replace(valid, `"safety":{"A":4,"B":4},`, ``, 1), "exactly five criteria"},
		{"out-of-range score refused", strings.Replace(valid, `"A":4,"B":4}`, `"A":5,"B":4}`, 1), "invalid score safety"},
		{"zero evidence refused", strings.Replace(valid, `"evidence":["a is grounded","b contradicts the prompt"]`, `"evidence":[]`, 1), "evidence count"},
		{"six evidence entries refused", strings.Replace(valid, `"evidence":["a is grounded","b contradicts the prompt"]`, `"evidence":["1","2","3","4","5","6"]`, 1), "evidence count"},
		{"overlong evidence refused", strings.Replace(valid, `"a is grounded"`, `"`+strings.Repeat("x", 401)+`"`, 1), "evidence too long"},
		{"unknown field refused", strings.Replace(valid, `{"verdict`, `{"confidence":1,"verdict`, 1), "unknown field"},
		{"trailing JSON refused", valid + ` {"verdict":"A"}`, "trailing JSON"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			j, err := cavemanpairwise.ParseJudgment(tt.input)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("ParseJudgment error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseJudgment unexpected error: %v", err)
			}
			if j.Verdict != "A" {
				t.Fatalf("Verdict = %q, want A", j.Verdict)
			}
			if len(j.Scores) != len(cavemanpairwise.Criteria) {
				t.Fatalf("Scores = %d criteria, want %d", len(j.Scores), len(cavemanpairwise.Criteria))
			}
		})
	}
}

func TestDeterministicProvenanceHelpers(t *testing.T) {
	// Hash is the content binding every receipt and diagnosis depends on.
	if got := cavemanpairwise.Hash([]byte("abc")); got != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" {
		t.Fatalf("Hash(abc) = %s", got)
	}
	// Order and Blind derive from the source hash so the same input always
	// yields the same presentation order and arm labels.
	if cavemanpairwise.Order("src", "p1") != cavemanpairwise.Order("src", "p1") {
		t.Fatal("Order is not deterministic for a fixed input")
	}
	a, b := cavemanpairwise.Blind("src", "p1", "A"), cavemanpairwise.Blind("src", "p1", "B")
	if a == "" || a == b {
		t.Fatalf("Blind arms collide or are empty: %q vs %q", a, b)
	}
	if got := cavemanpairwise.EndpointClass("anything"); got != "openai-compatible-chat-completions" {
		t.Fatalf("EndpointClass = %q", got)
	}
}

func v2ReadyReceipt(sourceSHA string, semanticPass, nonInferior bool) cavemanpairwise.Receipt {
	r := cavemanpairwise.Receipt{TokenEligible: false}
	r.Provenance.SourceSHA256 = sourceSHA
	r.Deterministic.SemanticPass = semanticPass
	r.Application.NonInferiority = &nonInferior
	return r
}

func TestBindSafetyThroughCmdWrapper(t *testing.T) {
	const sourceSHA = "src-sha-123"
	safety := []byte(`{"source_sha256":"src-sha-123","verdict":{"safety_gate_pass":true}}`)
	source := []byte(`{"upstream":{"saved_percent":37.5}}`)

	t.Run("all gates pass makes tokens eligible with the saved percent", func(t *testing.T) {
		r := v2ReadyReceipt(sourceSHA, true, true)
		if err := bindSafety(&r, source, safety); err != nil {
			t.Fatalf("bindSafety: %v", err)
		}
		if !r.Deterministic.SafetyPass {
			t.Fatal("SafetyPass = false, want true")
		}
		if r.Deterministic.SafetySHA256 != cavemanpairwise.Hash(safety) {
			t.Fatalf("SafetySHA256 = %q, want the receipt content hash", r.Deterministic.SafetySHA256)
		}
		if !r.TokenEligible {
			t.Fatalf("TokenEligible = false, verdict %q", r.TokenVerdict)
		}
		if r.TokenSavedPercent == nil || *r.TokenSavedPercent != 37.5 {
			t.Fatalf("TokenSavedPercent = %v, want 37.5", r.TokenSavedPercent)
		}
	})
	t.Run("mismatched safety receipt source is refused", func(t *testing.T) {
		r := v2ReadyReceipt("different-source", true, true)
		err := bindSafety(&r, source, safety)
		if err == nil || !strings.Contains(err.Error(), "source mismatch") {
			t.Fatalf("bindSafety error = %v, want source mismatch", err)
		}
		if r.TokenEligible {
			t.Fatal("tokens stayed eligible after a refused safety bind")
		}
	})
	t.Run("failed safety gate suppresses tokens", func(t *testing.T) {
		r := v2ReadyReceipt(sourceSHA, true, true)
		failing := []byte(`{"source_sha256":"src-sha-123","verdict":{"safety_gate_pass":false}}`)
		if err := bindSafety(&r, source, failing); err != nil {
			t.Fatalf("bindSafety: %v", err)
		}
		if r.Deterministic.SafetyPass || r.TokenEligible {
			t.Fatalf("gates did not hold: safetyPass=%v tokenEligible=%v", r.Deterministic.SafetyPass, r.TokenEligible)
		}
		if !strings.Contains(r.TokenVerdict, "suppressed") {
			t.Fatalf("TokenVerdict = %q, want suppressed", r.TokenVerdict)
		}
	})
	t.Run("malformed safety receipt refused", func(t *testing.T) {
		r := v2ReadyReceipt(sourceSHA, true, true)
		if err := bindSafety(&r, source, []byte(`{broken`)); err == nil || !strings.Contains(err.Error(), "parse deterministic safety receipt") {
			t.Fatalf("bindSafety error = %v, want parse failure", err)
		}
	})
	t.Run("semantic gate still binds tokens", func(t *testing.T) {
		r := v2ReadyReceipt(sourceSHA, false, true)
		if err := bindSafety(&r, source, safety); err != nil {
			t.Fatalf("bindSafety: %v", err)
		}
		if r.TokenEligible {
			t.Fatal("tokens eligible without the semantic gate")
		}
	})
}

func TestExitForHappyPathReturns(t *testing.T) {
	// The exit paths terminate the process, which cannot be exercised
	// in-process; the success path must simply return without exiting.
	pass := true
	exitFor(true, &pass, true, nil)
	if !pass {
		t.Fatal("expected pass to remain true")
	}
}

func TestDiagnoseV1OverFrozenReceipt(t *testing.T) {
	v1, err := os.ReadFile(frozenV1Receipt)
	if err != nil {
		t.Fatalf("read frozen v1 receipt: %v", err)
	}
	d, err := cavemanpairwise.DiagnoseV1(v1)
	if err != nil {
		t.Fatalf("DiagnoseV1: %v", err)
	}
	if d.Schema != "caveman-pairwise-diagnosis/1" {
		t.Fatalf("Schema = %q", d.Schema)
	}
	if d.V1ReceiptSHA256 != cavemanpairwise.FrozenV1ReceiptSHA {
		t.Fatalf("diagnosis is not bound to the frozen receipt: %q", d.V1ReceiptSHA256)
	}
	if d.Count != len(d.Flips) || d.Count == 0 {
		t.Fatalf("Count = %d with %d flips", d.Count, len(d.Flips))
	}
	classifications := map[string]bool{"parse_schema": true, "output_truncation": true, "tie_boundary_ambiguity": true, "actual_asymmetric_content": true}
	for _, f := range d.Flips {
		if f.PairID == "" || f.Comparison == "" {
			t.Fatalf("flip without identity: %+v", f)
		}
		if !classifications[f.Classification] {
			t.Fatalf("pair %s classified %q", f.PairID, f.Classification)
		}
		if len(f.Verdicts) == 0 {
			t.Fatalf("pair %s has no verdicts recorded", f.PairID)
		}
	}
	t.Run("tampered receipt refused", func(t *testing.T) {
		_, err := cavemanpairwise.DiagnoseV1(append(v1[:len(v1):len(v1)], 'x'))
		if err == nil || !strings.Contains(err.Error(), "hash mismatch") {
			t.Fatalf("DiagnoseV1 error = %v, want hash mismatch", err)
		}
	})
}

func TestWriteJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "receipt.json")
	writeJSON(path, map[string]int{"application_calls": 4})
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read written receipt: %v", err)
	}
	if !strings.HasSuffix(string(b), "\n") {
		t.Fatal("writeJSON must end the file with a newline")
	}
	var got map[string]int
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("written file is not JSON: %v", err)
	}
	if got["application_calls"] != 4 {
		t.Fatalf("round trip lost data: %v", got)
	}
	if !strings.Contains(string(b), "\n  ") {
		t.Fatal("writeJSON output is not indented")
	}
}

func TestMustRead(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "in.json")
	if err := os.WriteFile(path, []byte("payload"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := string(mustRead(path)); got != "payload" {
		t.Fatalf("mustRead = %q", got)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("mustRead did not panic on a missing file")
			}
		}()
		mustRead(filepath.Join(dir, "missing.json"))
	}()
}
