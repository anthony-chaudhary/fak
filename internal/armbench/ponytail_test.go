package armbench

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestPonytailDryRunPinsSuiteAndCounterbalances(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("pinned upstream selftest has a Windows process-handle cleanup bug; WSL/Linux is the supported test path")
	}
	root := repoRootForPonytailTest(t)
	checkout := filepath.Join(root, "_scratch", "issue-6688-upstream", "ponytail")
	if _, err := exec.LookPath("python"); err != nil {
		t.Skip("python unavailable")
	}
	if _, err := exec.Command("git", "-C", checkout, "rev-parse", "HEAD").Output(); err != nil {
		t.Skip("pinned comparator checkout unavailable")
	}
	p, err := Ponytail(PonytailOptions{Checkout: checkout, Python: "python", Model: "haiku", Trials: 1})
	if err != nil {
		t.Fatal(err)
	}
	if p.Mode != "dry-run" || p.Revision != PonytailRevision || len(p.Tasks) != 39 || len(p.Arms) != 3 {
		t.Fatalf("packet = %+v", p)
	}
	if p.TimeoutSeconds != 300 || p.JudgeModel == "" || p.Models["haiku"] != "claude-haiku-4-5-20251001" {
		t.Fatalf("pins = %+v", p)
	}
	if !reflect.DeepEqual(p.Arms, []string{"baseline", "caveman", "ponytail"}) {
		t.Fatalf("arms=%v", p.Arms)
	}
	joined := strings.Join(p.Commands, "\n")
	for _, order := range []string{"baseline,caveman,ponytail", "caveman,ponytail,baseline", "ponytail,baseline,caveman"} {
		if !strings.Contains(joined, order) {
			t.Fatalf("missing order %s", order)
		}
	}
}

func TestPonytailLiveRequiresAccountIdentity(t *testing.T) {
	_, err := Ponytail(PonytailOptions{Checkout: "missing", Live: true})
	if err == nil {
		t.Fatal("expected refusal")
	}
}

func TestSummarizePonytailEvidenceSuccessFirst(t *testing.T) {
	root := t.TempDir()
	d := filepath.Join(root, "run")
	if err := os.MkdirAll(d, 0755); err != nil {
		t.Fatal(err)
	}
	doc := map[string]any{"results": []map[string]any{{"arm": "baseline", "correct": 1, "safe": 0, "cost": 0.5, "duration_ms": 1200, "denials": 2, "in_tokens": 10, "out_tokens": 3, "cache_tokens": 4}, {"arm": "ponytail", "correct": 0, "safe": 1, "error": "timeout"}}}
	b, _ := json.Marshal(doc)
	if err := os.WriteFile(filepath.Join(d, "results.json"), b, 0644); err != nil {
		t.Fatal(err)
	}
	got, err := SummarizePonytailEvidence(root)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Successes != 1 || got[0].Tokens != 17 || got[0].Denials != 2 || got[2].Failures != 1 || got[0].Retries != 0 {
		t.Fatalf("report=%+v", got)
	}
}

func TestRunDirsUsesAgenticRunRoot(t *testing.T) {
	root := t.TempDir()
	want := filepath.Join(root, "benchmarks", "agentic", "runs", "stamp")
	if err := os.MkdirAll(want, 0755); err != nil {
		t.Fatal(err)
	}
	got, err := runDirs(root)
	if err != nil || !got[want] {
		t.Fatalf("got=%v err=%v", got, err)
	}
}

func repoRootForPonytailTest(t *testing.T) string {
	t.Helper()
	p, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return p
}
