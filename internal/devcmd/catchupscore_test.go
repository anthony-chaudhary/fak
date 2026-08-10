package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/catchupscore"
)

// runCatchUpScore in an empty workspace (no dos.toml -> index level excluded) with
// explicit level flags emits a control-pane JSON carrying the unbounded backlog, the
// debt integer `fak scoreboard post --debt-key` targets, and the worst-first worklist.
func TestRunCatchUpScoreJSON(t *testing.T) {
	empty := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := RunCatchUpScore(&stdout, &stderr, []string{
		"--workspace", empty,
		"--intake-behind", "20", "--intake-total", "50", // 0.60 caught up -> behind
		"--trunk-behind", "0", "--trunk-total", "8", // 1.00 caught up -> clean
		"--json",
	})
	if code != 1 { // a behind level -> not ok -> exit 1
		t.Fatalf("exit = %d (stderr=%q), want 1 (intake behind)", code, stderr.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("payload is not JSON: %v\n%s", err, stdout.String())
	}
	if payload["schema"] != catchupscore.Schema {
		t.Errorf("schema = %v, want %v", payload["schema"], catchupscore.Schema)
	}
	corpus, ok := payload["corpus"].(map[string]any)
	if !ok {
		t.Fatalf("corpus missing/!map: %v", payload["corpus"])
	}
	if got := jsonInt(corpus, catchupscore.DebtKey); got != 1 {
		t.Errorf("%s = %d, want 1 (one behind level)", catchupscore.DebtKey, got)
	}
	if got := jsonInt(corpus, catchupscore.BacklogKey); got != 20 {
		t.Errorf("%s = %d, want 20 (raw untriaged backlog)", catchupscore.BacklogKey, got)
	}
	if got := jsonInt(corpus, "levels_present"); got != 2 {
		t.Errorf("levels_present = %d, want 2 (index excluded, no dos.toml)", got)
	}
	worklist, ok := corpus["catchup_worklist"].([]any)
	if !ok || len(worklist) != 2 {
		t.Fatalf("worklist = %v, want 2 rows", corpus["catchup_worklist"])
	}
	first, _ := worklist[0].(map[string]any)
	if first["level"] != catchupscore.LevelIntake {
		t.Errorf("worst-first row = %v, want intake (the behind level)", first["level"])
	}
}

// The measurement level is folded from a captured control-pane payload via --scores-from.
func TestRunCatchUpScoreScoresFrom(t *testing.T) {
	dir := t.TempDir()
	pane := filepath.Join(dir, "pane.json")
	if err := os.WriteFile(pane, []byte(`{"measured": 30, "errored": 9}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := RunCatchUpScore(&stdout, &stderr, []string{
		"--workspace", dir, "--scores-from", pane, "--json",
	})
	// 9 errored of 39 -> 30/39 = 0.769 caught up < 0.8 -> behind -> exit 1.
	if code != 1 {
		t.Fatalf("exit = %d (stderr=%q), want 1", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), catchupscore.LevelMeasurement) {
		t.Errorf("payload missing measurement level:\n%s", stdout.String())
	}
	var payload map[string]any
	json.Unmarshal(stdout.Bytes(), &payload)
	corpus := payload["corpus"].(map[string]any)
	if got := jsonInt(corpus, catchupscore.BacklogKey); got != 9 {
		t.Errorf("%s = %d, want 9 (errored cards)", catchupscore.BacklogKey, got)
	}
}

// With no evidence at all (empty workspace, no flags, no --scores-from) the card is a
// clean INSUFFICIENT read: ok, exit 0, catchup 1.0.
func TestRunCatchUpScoreNoEvidenceIsClean(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunCatchUpScore(&stdout, &stderr, []string{
		"--workspace", t.TempDir(), "--no-index", "--json",
	})
	if code != 0 {
		t.Fatalf("exit = %d (stderr=%q), want 0 (no evidence -> clean)", code, stderr.String())
	}
	var payload map[string]any
	json.Unmarshal(stdout.Bytes(), &payload)
	if payload["ok"] != true {
		t.Errorf("ok = %v, want true", payload["ok"])
	}
	corpus := payload["corpus"].(map[string]any)
	if jsonInt(corpus, "levels_present") != 0 {
		t.Errorf("levels_present = %v, want 0", corpus["levels_present"])
	}
}
