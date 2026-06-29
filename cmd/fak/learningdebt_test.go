package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestLearningDebtDispatchDryRunJSONDoesNotWriteCache(t *testing.T) {
	dir := t.TempDir()
	scorecard := filepath.Join(dir, "learning.json")
	cache := filepath.Join(dir, "seen", "seen.json")
	raw := `{
	  "schema":"fleet-learning-scorecard/1",
	  "corpus":{"priorities":[{"path":"docs/fak/tutorial.md","priority":1}]},
	  "docs":[{"path":"docs/fak/tutorial.md","score":50,"grade":"D",
	    "defects":["orientation: no orientation signpost"]}],
	  "coverage":{"defects":[]},
	  "stamp_freshness":{"stale_stamp":false}
	}`
	if err := os.WriteFile(scorecard, []byte(raw), 0o644); err != nil {
		t.Fatalf("write scorecard: %v", err)
	}

	var out, errb bytes.Buffer
	code := runLearningDebtDispatch(&out, &errb, []string{
		"--scorecard", scorecard,
		"--cache", cache,
		"--cap", "1",
		"--json",
	})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if _, err := os.Stat(cache); !os.IsNotExist(err) {
		t.Fatalf("dry-run wrote cache, stat err=%v", err)
	}
	var result map[string]any
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatalf("json output: %v\n%s", err, out.String())
	}
	if result["mode"] != "dry-run" {
		t.Fatalf("mode=%v, want dry-run", result["mode"])
	}
	planned := result["planned"].([]any)
	if len(planned) != 1 {
		t.Fatalf("planned len=%d, want 1", len(planned))
	}
}
