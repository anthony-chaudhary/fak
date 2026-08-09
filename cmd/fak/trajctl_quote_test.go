package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestTrajctlQuoteCLIAndColdStartRefusal(t *testing.T) {
	corpus := filepath.Join("..", "..", "internal", "trajctl", "testdata", "repo_question_corpus.jsonl")
	args := []string{"quote", "--corpus", corpus, "--capability-version", "cap-v3", "--policy-version", "pol-v2", "--index-version", "idx-v3", "--index-coverage", "0.95", "--quality-min", "0.8", "--quality-witness", "fixture-grader-v1", "--at", "2026-01-07T00:00:00Z"}
	var out, errOut bytes.Buffer
	if code := runTrajctl(&out, &errOut, args); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !bytes.Contains(out.Bytes(), []byte(`"request_class": "repo_question"`)) {
		t.Fatalf("out=%s", out.String())
	}
	cold := filepath.Join(t.TempDir(), "cold.jsonl")
	if err := os.WriteFile(cold, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	args[2] = cold
	out.Reset()
	errOut.Reset()
	if code := runTrajctl(&out, &errOut, args); code != 3 || !bytes.Contains(errOut.Bytes(), []byte("REFUSED")) {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}

func TestTrajctlQuoteBacktestCLI(t *testing.T) {
	corpus := filepath.Join("..", "..", "internal", "trajctl", "testdata", "repo_question_corpus.jsonl")
	var out, errOut bytes.Buffer
	if code := runTrajctl(&out, &errOut, []string{"quote-backtest", "--corpus", corpus}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, needle := range []string{`"quantile": "p50"`, `"quantile": "p80"`, `"quantile": "p95"`, `"cold_start_refusals": 2`} {
		if !bytes.Contains(out.Bytes(), []byte(needle)) {
			t.Fatalf("missing %s in %s", needle, out.String())
		}
	}
}
