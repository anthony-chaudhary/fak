package conceptcatalog

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestProbeFreshnessErrorOutranksAFreshFlag pins the invariant the whole verdict rests
// on: whatever the result says about itself, a check that returned an error proved
// nothing and must not render as fresh (#5962).
//
// The literal below is the exact pair CheckFresh hands back when it cannot read a
// generated artifact -- it seeds the result with Fresh:true before the comparison loop
// and returns THAT value alongside the read error. Encoding it is what printed
// `{"fresh":true}` for a failed check.
func TestProbeFreshnessErrorOutranksAFreshFlag(t *testing.T) {
	p := ProbeFreshness(
		FreshnessResult{Fresh: true, Regenerate: RegenerateCommand},
		errors.New("read generated README.md: is a directory"),
	)
	if p.Verdict != VerdictUnknown {
		t.Fatalf("verdict = %q, want %q: a check that errored proved neither fresh nor stale", p.Verdict, VerdictUnknown)
	}
	if p.Fresh {
		t.Fatal("a failed check must never render Fresh, however the half-filled result flagged itself")
	}
	if !strings.Contains(p.Unchecked, "is a directory") {
		t.Errorf("Unchecked = %q, want the read error that stopped the check", p.Unchecked)
	}
	if p.Regenerate != RegenerateCommand {
		t.Errorf("Regenerate = %q, want the cure preserved so an unknown verdict still has a next step", p.Regenerate)
	}
	if raw := string(p.JSON()); strings.Contains(raw, `"fresh":true`) {
		t.Errorf("probe JSON = %s, must not carry a true fresh flag", raw)
	}
}

// TestProbeFreshnessKeepsTheRanVerdicts: adding the third verdict must not blur the two
// that already worked.
func TestProbeFreshnessKeepsTheRanVerdicts(t *testing.T) {
	fresh := ProbeFreshness(FreshnessResult{Fresh: true, Regenerate: RegenerateCommand}, nil)
	if fresh.Verdict != VerdictFresh || !fresh.Fresh {
		t.Errorf("a clean check = %+v, want the fresh verdict", fresh)
	}
	stale := ProbeFreshness(FreshnessResult{StalePaths: []string{GeneratedIndex}, Regenerate: RegenerateCommand}, nil)
	if stale.Verdict != VerdictStale || stale.Fresh || len(stale.StalePaths) != 1 {
		t.Errorf("a drifting check = %+v, want the stale verdict with its paths", stale)
	}
}

// TestRenderFreshnessOnAnUnreadableArtifactNeverPrintsFreshTrue is the issue's named
// witness: make the generated-artifact read fail inside the real CheckFresh, render the
// answer the way `fak concept freshness --check --json` does, and assert stdout does NOT
// contain `"fresh":true`.
//
// The fixture makes the read fail rather than the generation: a stub generator writes
// README.md and INDEX.md as DIRECTORIES, so generate()'s "every artifact exists" check
// passes and CheckFresh reaches the os.ReadFile that returns the error. That is the
// precise seam where the check hands back a result still flagged fresh.
func TestRenderFreshnessOnAnUnreadableArtifactNeverPrintsFreshTrue(t *testing.T) {
	root := unreadableArtifactRepo(t)
	res, err := CheckFresh(root)
	if err == nil {
		t.Fatalf("fixture did not make the generated-artifact read fail: CheckFresh = %+v", res)
	}

	var stdout, stderr bytes.Buffer
	code := RenderFreshness(&stdout, &stderr, "fak concept freshness", "", res, err, true)
	if code != 1 {
		t.Errorf("exit code = %d, want 1: a check that could not run is not a pass", code)
	}
	if strings.Contains(stdout.String(), `"fresh":true`) {
		t.Fatalf("stdout reported a fresh check that never ran: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), `"verdict":"unknown"`) {
		t.Errorf("stdout = %s, want the unknown verdict a --json consumer can branch on", stdout.String())
	}
	if !strings.Contains(stderr.String(), "UNKNOWN") {
		t.Errorf("stderr = %q, want the unknown verdict named for the operator too", stderr.String())
	}
}

// TestRenderFreshnessRanVerdicts pins the human and exit-code contract the cmd/fak
// surfaces used to spell inline, so folding them onto this renderer cannot silently
// change what an operator sees.
func TestRenderFreshnessRanVerdicts(t *testing.T) {
	var freshOut, freshErr bytes.Buffer
	if code := RenderFreshness(&freshOut, &freshErr, "fak concept freshness", "", FreshnessResult{Fresh: true, Regenerate: RegenerateCommand}, nil, false); code != 0 {
		t.Errorf("fresh exit = %d, want 0", code)
	}
	if !strings.Contains(freshOut.String(), "concept generated artifacts fresh") {
		t.Errorf("fresh stdout = %q", freshOut.String())
	}

	var staleOut, staleErr bytes.Buffer
	code := RenderFreshness(&staleOut, &staleErr, "fak concept freshness --staged", " in the staged tree",
		FreshnessResult{StalePaths: []string{GeneratedReadme}, Regenerate: RegenerateStagedCommand}, nil, false)
	if code != 1 {
		t.Errorf("stale exit = %d, want 1", code)
	}
	for _, want := range []string{"stale generated concept artifacts in the staged tree:", GeneratedReadme, RegenerateStagedCommand} {
		if !strings.Contains(staleErr.String(), want) {
			t.Errorf("stale stderr = %q, want it to name %q", staleErr.String(), want)
		}
	}
	if staleOut.Len() != 0 {
		t.Errorf("stale stdout = %q, want the human report on stderr only", staleOut.String())
	}
}

// unreadableArtifactRepo is a root whose generator succeeds but whose output cannot be
// read back: the stub writes each tracked artifact name as a directory. generate() only
// stats the artifacts, so it reports success and CheckFresh proceeds to the read that
// fails -- the "could not check" path, reached through the production code.
func unreadableArtifactRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	tools := filepath.Join(root, "tools")
	if err := os.MkdirAll(tools, 0755); err != nil {
		t.Fatal(err)
	}
	stub := "import os, sys\n" +
		"d = sys.argv[sys.argv.index('--markdown-dir') + 1]\n" +
		"for name in ('README.md', 'INDEX.md'):\n" +
		"    os.makedirs(os.path.join(d, name), exist_ok=True)\n"
	if err := os.WriteFile(filepath.Join(tools, "concept_disambiguation_scorecard.py"), []byte(stub), 0644); err != nil {
		t.Fatal(err)
	}
	return root
}
