package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

func gitInitRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	run("init", "-q")
	run("config", "user.email", "t@example.com")
	run("config", "user.name", "t")
	if err := os.WriteFile(filepath.Join(dir, "seed.txt"), []byte("seed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	run("add", "-A")
	run("commit", "-q", "-m", "seed")
	return dir
}

// End-to-end CLI witness: the real --since path over a real git repo. Two fast fake
// cards (missing scripts => the "measure" path returns a fast error, no subprocess)
// swap in for the compiled roster; a hand-written baseline pins both. A clean tree
// (empty diff) carries both and the fold reproduces the baseline; an untracked file in
// one card's corpus measures only that card. Proves flag parsing, baseline load,
// scdiff.ChangedPaths git integration, CollectSince, Fold, and the incremental JSON.
func TestControlPaneSinceJSONWitness(t *testing.T) {
	orig := scorecardpane.Cards
	t.Cleanup(func() { scorecardpane.Cards = orig })
	scorecardpane.Cards = []scorecardpane.Card{
		{Key: "a", Debt: "a_debt", Label: "alpha", Script: "missing_a.py", Corpus: []string{"docs/"}},
		{Key: "b", Debt: "b_debt", Label: "beta", Script: "missing_b.py", Corpus: []string{"src/"}},
	}
	dir := gitInitRepo(t)
	baseline := scorecardpane.Baseline{
		Schema: scorecardpane.BaselineSchema, Commit: "seedsha", TotalDebt: 8, GradeDebt: 3,
		Metrics: map[string]int{"a": 3, "b": 5}, GradeWeights: map[string]int{"a": 1, "b": 2},
	}
	if err := os.MkdirAll(filepath.Join(dir, "tools"), 0o755); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(baseline)
	if err := os.WriteFile(filepath.Join(dir, filepath.FromSlash(scorecardpane.BaselineRel)), b, 0o644); err != nil {
		t.Fatal(err)
	}

	decode := func(out bytes.Buffer) scorecardpane.Payload {
		t.Helper()
		var p scorecardpane.Payload
		if err := json.Unmarshal(out.Bytes(), &p); err != nil {
			t.Fatalf("decode payload: %v\n%s", err, out.String())
		}
		if p.Incremental == nil {
			t.Fatalf("payload missing incremental block: %s", out.String())
		}
		return p
	}

	t.Run("clean tree carries all, fold equals baseline", func(t *testing.T) {
		var out, errb bytes.Buffer
		code := runScorecardControlPane(&out, &errb, []string{"--workspace", dir, "--since", "HEAD", "--json"})
		p := decode(out)
		if p.Incremental.Carried != 2 || p.Incremental.Measured != 0 {
			t.Fatalf("carried=%d measured=%d want 2/0", p.Incremental.Carried, p.Incremental.Measured)
		}
		if p.TotalDebt != 8 {
			t.Fatalf("total_debt=%d want 8 (baseline reproduced)", p.TotalDebt)
		}
		// The carried fold reproduces the debt-8 baseline exactly, so the existing
		// contract (exit 0 only at zero debt) correctly reports outstanding debt.
		if code != 1 || p.Finding != "scorecard_debt" {
			t.Fatalf("carried debt-8 fold exit=%d finding=%q want 1/scorecard_debt", code, p.Finding)
		}
	})

	t.Run("untracked corpus file measures only that card", func(t *testing.T) {
		if err := os.MkdirAll(filepath.Join(dir, "src"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "src", "x.go"), []byte("package x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		var out, errb bytes.Buffer
		runScorecardControlPane(&out, &errb, []string{"--workspace", dir, "--since", "HEAD", "--json"})
		p := decode(out)
		if p.Incremental.Carried != 1 || p.Incremental.Measured != 1 {
			t.Fatalf("carried=%d measured=%d want 1/1", p.Incremental.Carried, p.Incremental.Measured)
		}
		if len(p.Incremental.CarriedKeys) != 1 || p.Incremental.CarriedKeys[0] != "a" {
			t.Fatalf("carried_keys=%v want [a] (b's corpus touched)", p.Incremental.CarriedKeys)
		}
	})
}

// The --since incremental fold is an approximate read, so it must refuse the two flags
// that would launder a partial measurement into a durable claim (--pin seeds a floor,
// --post publishes a number). Both guards return 2 BEFORE any card runs.
func TestControlPaneSinceRefusesPinAndPost(t *testing.T) {
	dir := t.TempDir()
	for _, extra := range []string{"--pin", "--post"} {
		var out, errb bytes.Buffer
		code := runScorecardControlPane(&out, &errb, []string{"--workspace", dir, "--since", "HEAD", extra})
		if code != 2 {
			t.Fatalf("--since %s exit=%d want 2", extra, code)
		}
		if !strings.Contains(errb.String(), "cannot be combined with --pin or --post") {
			t.Fatalf("--since %s stderr missing guard message: %q", extra, errb.String())
		}
	}
}

// --since is meaningless without a pinned baseline to carry from; it must say so
// rather than silently degrade to a full run.
func TestControlPaneSinceRequiresBaseline(t *testing.T) {
	dir := t.TempDir() // no baseline file here
	var out, errb bytes.Buffer
	code := runScorecardControlPane(&out, &errb, []string{"--workspace", dir, "--since", "HEAD"})
	if code != 2 {
		t.Fatalf("exit=%d want 2", code)
	}
	if !strings.Contains(errb.String(), "requires a pinned baseline") {
		t.Fatalf("stderr missing baseline-required message: %q", errb.String())
	}
}
