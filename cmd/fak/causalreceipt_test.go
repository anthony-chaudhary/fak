package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRunCausalReceiptSelfTest(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCausalReceipt(&stdout, &stderr, strings.NewReader(""), []string{"--self-test", "--json"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d (stderr: %s)", code, stderr.String())
	}

	var out causalReceiptOutput
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		t.Fatalf("json unmarshal failed: %v (raw: %s)", err, stdout.String())
	}
	if !out.Valid || out.Schema != "fak.causal-receipt/1" {
		t.Fatalf("unexpected self-test output: %+v", out)
	}
}

func TestRunCausalReceiptStdinValidation(t *testing.T) {
	validJSON := `{
		"schema": "fak.causal-receipt/1",
		"ids": {
			"work": "w1",
			"turn": "t1",
			"graph": "g1",
			"request": "r1"
		},
		"phases": [
			{
				"id": "p1",
				"kind": "agent",
				"engine": "fak-native",
				"backend": "offline",
				"outcome": "completed",
				"tokens": 42
			}
		],
		"resources": [
			{
				"id": "res1",
				"kind": "weights",
				"state": "released",
				"released": true
			}
		],
		"decisions": [
			{
				"id": "d1",
				"kind": "policy",
				"actual": "allow"
			}
		]
	}`

	var stdout, stderr bytes.Buffer
	code := runCausalReceipt(&stdout, &stderr, strings.NewReader(validJSON), []string{"--validate", "-"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d (stderr: %s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "OK (valid)") {
		t.Fatalf("expected valid message, got: %s", stdout.String())
	}

	// Now invalid JSON (missing turn)
	invalidJSON := `{"schema":"fak.causal-receipt/1","ids":{"work":"w1"}}`
	stdout.Reset()
	stderr.Reset()
	code = runCausalReceipt(&stdout, &stderr, strings.NewReader(invalidJSON), []string{"--validate", "-"})
	if code != 1 {
		t.Fatalf("expected exit code 1 for invalid receipt, got %d", code)
	}
}
