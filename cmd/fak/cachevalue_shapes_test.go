package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

// writeShapeLedger drops a WITNESSED Track-1 ledger with a mix of session shapes and
// returns its path: one single-turn cold run, two short warm sessions, and one long
// partial session — enough to populate three distinct (length × outcome) clusters.
func writeShapeLedger(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "cache-value.jsonl")
	content := `{"date":"2026-06-15","session_type":"run","turns":1,"prompt_tokens":100,"reused_tokens":0,"reuse_ratio":0}
{"date":"2026-06-16","session_type":"guard","turns":3,"prompt_tokens":1000,"reused_tokens":700,"reuse_ratio":0.70}
{"date":"2026-06-17","session_type":"guard","turns":4,"prompt_tokens":500,"reused_tokens":300,"reuse_ratio":0.60}
{"date":"2026-06-18","session_type":"serve","turns":8,"prompt_tokens":2000,"reused_tokens":600,"reuse_ratio":0.30}
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestCachevalueShapesPrintsClusters is the witness at the CLI seam: `fak cachevalue
// shapes` prints the length × outcome cluster table with the #1066 fence label.
func TestCachevalueShapesPrintsClusters(t *testing.T) {
	dir := t.TempDir()
	ledger := writeShapeLedger(t, dir)

	var out, errb bytes.Buffer
	code := runCachevalueShapes(&out, &errb, []string{"--ledger", ledger, "--since", "2026-06-01"})
	if code != 0 {
		t.Fatalf("shapes exit = %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"session shapes", "WITNESSED",
		"length", "outcome", "health", "reuse-tok%",
		"single", "short", "long",
		"warm", "partial",
		"marginal-over-tuned-warm-KV", // the #1066 fence
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("shapes table missing %q:\n%s", want, got)
		}
	}
}

// TestCachevalueShapesJSONReproducesFold is the recompute witness: `--json` emits a
// ShapeReport that re-folds from the same ledger, and single-turn rows land under the
// n/a outcome (never cold).
func TestCachevalueShapesJSONReproducesFold(t *testing.T) {
	dir := t.TempDir()
	ledger := writeShapeLedger(t, dir)

	var out, errb bytes.Buffer
	code := runCachevalueShapes(&out, &errb, []string{"--ledger", ledger, "--json"})
	if code != 0 {
		t.Fatalf("shapes --json exit = %d, stderr=%s", code, errb.String())
	}
	var rep cachevaluereport.ShapeReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("shapes --json is not a ShapeReport: %v\n%s", err, out.String())
	}
	if rep.Verdict != "MEASURED" {
		t.Fatalf("mixed multi-turn corpus should be MEASURED, got %q", rep.Verdict)
	}
	if rep.TotalSessions != 4 || rep.MultiTurnSessions != 3 || rep.SingleTurnSessions != 1 {
		t.Fatalf("session counts = total %d / multi %d / single %d, want 4/3/1",
			rep.TotalSessions, rep.MultiTurnSessions, rep.SingleTurnSessions)
	}
	var single *cachevaluereport.ShapeCluster
	for i := range rep.Clusters {
		if rep.Clusters[i].Length == cachevaluereport.LengthSingle {
			single = &rep.Clusters[i]
		}
		if rep.Clusters[i].Outcome == cachevaluereport.OutcomeCold && rep.Clusters[i].Length == cachevaluereport.LengthSingle {
			t.Fatalf("single-turn rows must not land in a cold cluster: %+v", rep.Clusters[i])
		}
	}
	if single == nil || single.Outcome != cachevaluereport.OutcomeNA {
		t.Fatalf("single-turn cluster outcome = %+v, want n/a", single)
	}
}

// TestCachevalueShapesBadSince rejects a malformed --since with exit 2.
func TestCachevalueShapesBadSince(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runCachevalueShapes(&out, &errb, []string{"--since", "last-tuesday"}); code != 2 {
		t.Fatalf("bad --since exit = %d, want 2", code)
	}
}
