package conflationscore

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// parity_test.go -- the differential harness proving the Go conflation port matches its Python
// oracle (tools/conflation_scorecard.py). It runs BOTH over the SAME live tree:
// `python tools/conflation_scorecard.py --json` and Go Build(root), then asserts the
// load-bearing control-pane scalars agree -- conflation_debt (exact int), grade (exact
// string), score (within rounding), ok (exact). This is stronger than the hooks harness
// (which compares one block/clean bit) because a scorecard's contract IS the number the
// control-pane folds.
//
// It is the PORT-TIME proof: it runs green against the still-present .py in CI/WSL before the
// Python card is deleted. Once tools/conflation_scorecard.py is gone there is no oracle to
// diff, so the test self-skips (and is removed in the same commit that deletes the .py).
// Skipped under -short or when python/git is absent -- so a box without Python skips cleanly
// rather than reddening (the internal/hooks/parity_test.go precedent).

func pyExe() (string, []string) {
	for _, c := range []struct {
		bin  string
		args []string
	}{
		{"python3", nil}, {"python", nil}, {"py", []string{"-3"}},
	} {
		if _, err := exec.LookPath(c.bin); err == nil {
			return c.bin, c.args
		}
	}
	return "", nil
}

func repoRootForParity(t *testing.T) string {
	t.Helper()
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Skip("not in a git repo")
	}
	return strings.TrimSpace(string(out))
}

func TestConflationParityVsPythonOracle(t *testing.T) {
	if testing.Short() {
		t.Skip("parity harness skipped under -short")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	py, pyArgs := pyExe()
	if py == "" {
		t.Skip("python not on PATH")
	}
	root := repoRootForParity(t)
	script := filepath.Join(root, "tools", "conflation_scorecard.py")
	if _, err := os.Stat(script); err != nil {
		t.Skipf("oracle conflation_scorecard.py not found (deleted post-port): %v", err)
	}

	// 1. Python oracle over the live tree.
	args := append(append([]string{}, pyArgs...), script, "--json")
	out, err := exec.Command(py, args...).Output()
	if err != nil {
		// exit 1 is "has debt" (still a valid payload on stdout); only a non-payload failure skips.
		if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() > 1 {
			t.Skipf("python oracle could not run: %v", err)
		}
	}
	var pyPayload map[string]any
	if err := json.Unmarshal(out, &pyPayload); err != nil {
		t.Fatalf("oracle did not emit JSON: %v\n%s", err, out)
	}
	pyCorpus, _ := pyPayload["corpus"].(map[string]any)
	if pyCorpus == nil {
		t.Fatalf("oracle payload missing corpus: %v", pyPayload)
	}

	// 2. Go port over the same tree.
	goPayload := Build(root)
	goCorpus := goPayload.Corpus

	// 3. Assert the load-bearing scalars agree.
	if pd, gd := toInt(pyCorpus[DebtKey]), toInt(goCorpus[DebtKey]); pd != gd {
		t.Errorf("conflation_debt MISMATCH: python=%d go=%d", pd, gd)
	}
	if pg, gg := toStr(pyCorpus["grade"]), toStr(goCorpus["grade"]); pg != gg {
		t.Errorf("grade MISMATCH: python=%q go=%q", pg, gg)
	}
	if ps, gs := toFloat(pyCorpus["score"]), toFloat(goCorpus["score"]); math.Abs(ps-gs) > 0.05 {
		t.Errorf("score MISMATCH: python=%v go=%v", ps, gs)
	}
	if pok, gok := boolOf(pyPayload["ok"]), goPayload.OK; pok != gok {
		t.Errorf("ok MISMATCH: python=%v go=%v", pok, gok)
	}
	// surfaces / external_values_seen are also part of the corpus contract -- diff them too so a
	// drift in extraction (not just scoring) is caught.
	for _, k := range []string{"surfaces", "external_values_seen"} {
		if pv, gv := toInt(pyCorpus[k]), toInt(goCorpus[k]); pv != gv {
			t.Errorf("%s MISMATCH: python=%d go=%d", k, pv, gv)
		}
	}
}

func toInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	default:
		return -1
	}
}

func toFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	default:
		return math.NaN()
	}
}

func toStr(v any) string {
	s, _ := v.(string)
	return s
}

func boolOf(v any) bool {
	b, _ := v.(bool)
	return b
}
