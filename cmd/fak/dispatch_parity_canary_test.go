package main

// Hermetic tests for dispatch_parity_canary.go — the Go port of
// tools/dispatch_parity_canary_test.py. The gate composes six rungs; rungs 1
// (shipped) and 2 (witnessed) come from git + `dos commit-audit`, rungs 3-6 are
// pure functions of the commit subject and the worker's git-command stream. Here
// we drive the pure predicates and the grade fold directly — every PARITY_PROVEN /
// PARITY_UNPROVEN_* / PARITY_UNOBSERVED verdict — plus the run-log git extractor
// and an evaluate() pass over injected git/witness seams. No git, no subprocess.
//
// The fixtures encode the REAL glm-5.2 canary (#420's evidence): opencode shipped
// 7528df3 for #545 by explicit pathspec, signed off, #545 in the subject, lane
// tree clean — so the proven-path fixture is the one the operator actually relies
// on, not a synthetic happy path.

import (
	"bytes"
	"strings"
	"testing"
)

// The real glm-5.2 canary's git stream, prose stripped to git cmds, as it appears
// in .dispatch-runs/resolve-545-20260623-162209.log.
var realGLMGitCmds = strings.Join([]string{
	"git status --porcelain",
	"git log --oneline -5",
	"git config user.name",
	"git commit -s -- internal/recall/ctxplan_store.go internal/recall/ctxplan_store_test.go",
	"git fetch origin main",
	"git push origin main",
	"git rev-parse --short HEAD",
}, "\n")

const realGLMSubject = "feat(ctxplan): add recall.Session -> ctxplan.Store adapter, " +
	"the first real-store backing (#545) (fak recall)"

func TestParityIssueBound(t *testing.T) {
	if !parityIssueBound(realGLMSubject, 545, true) {
		t.Error("exact issue #545 should bind")
	}
	if parityIssueBound(realGLMSubject, 999, true) {
		t.Error("wrong issue #999 should not bind")
	}
	if !parityIssueBound("fix(x): a thing (#12)", 0, false) {
		t.Error("any #N should bind when issue unset")
	}
	if parityIssueBound("fix(x): a thing", 0, false) {
		t.Error("no #N should not bind when issue unset")
	}
	// #54 must NOT match a subject that only carries #545.
	if parityIssueBound("feat: thing (#545)", 54, true) {
		t.Error("#54 must not match a subject carrying only #545")
	}
}

func TestParityByPathspec(t *testing.T) {
	if !parityByPathspec(realGLMGitCmds) {
		t.Error("explicit `git commit -- <paths>` should pass")
	}
	if parityByPathspec("git add -A\ngit commit -s -- foo.go") {
		t.Error("blanket `git add -A` should refute")
	}
	if parityByPathspec("git add --all\ngit commit -s -- foo") {
		t.Error("blanket `git add --all` should refute")
	}
	if parityByPathspec("git add .\ngit commit -s -- foo") {
		t.Error("blanket `git add .` should refute")
	}
	if !parityByPathspec("git add tools/x.py\ngit commit -s -- tools/x.py") {
		t.Error("`git add <named-path>` is not blanket")
	}
	if parityByPathspec("git commit -s -m 'thing'") {
		t.Error("a commit with no `-- <path>` should fail")
	}
}

func TestParitySignedOff(t *testing.T) {
	if !paritySignedOff(realGLMGitCmds) {
		t.Error("`git commit -s` should be signed off")
	}
	if !paritySignedOff("git commit --signoff -- foo") {
		t.Error("`git commit --signoff` should be signed off")
	}
	if paritySignedOff("git commit -- foo bar") {
		t.Error("a commit without -s/--signoff should not be signed off")
	}
}

func TestParityGitCmdsFromLog(t *testing.T) {
	// The opencode shell glyph: "\x1b[0m$ \x1b[0m<command>".
	log := "\x1b[0mI'll commit now.\n" +
		"\x1b[0m$ \x1b[0mgit commit -s -- a.go b.go\n" +
		"\x1b[0mPush landed.\n" +
		"\x1b[0m$ \x1b[0mgit push origin main\n"
	cmds := parityGitCmdsFromLog(log)
	if !strings.Contains(cmds, "git commit -s -- a.go b.go") {
		t.Errorf("expected commit cmd extracted, got %q", cmds)
	}
	if !strings.Contains(cmds, "git push origin main") {
		t.Errorf("expected push cmd extracted, got %q", cmds)
	}
	if strings.Contains(cmds, "Push landed") {
		t.Errorf("prose without git should be dropped, got %q", cmds)
	}
}

func provenInputs() parityGradeInput {
	cmds := realGLMGitCmds
	clean := true
	return parityGradeInput{
		ShippedAUnit: true,
		Witnessed:    true,
		Subject:      realGLMSubject,
		Issue:        545,
		IssueSet:     true,
		GitCmds:      &cmds,
		LaneClean:    &clean,
	}
}

func TestParityGradeRealGLMCanaryProven(t *testing.T) {
	g := parityGradeFold(provenInputs())
	if !g.Proven {
		t.Fatalf("real glm canary should be proven, got %+v", g)
	}
	if g.Verdict != "PARITY_PROVEN" {
		t.Errorf("verdict = %q, want PARITY_PROVEN", g.Verdict)
	}
	if g.FailedRung != nil {
		t.Errorf("failed_rung = %v, want nil", *g.FailedRung)
	}
}

func TestParityGradeFirstFailingRung(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*parityGradeInput)
		verdict string
		rung    string
	}{
		{"no_unit", func(in *parityGradeInput) { in.ShippedAUnit = false }, "PARITY_UNPROVEN_NO_UNIT", "shipped_a_unit"},
		{"unwitnessed", func(in *parityGradeInput) { in.Witnessed = false }, "PARITY_UNPROVEN_UNWITNESSED", "witnessed"},
		{"unbound", func(in *parityGradeInput) { in.Issue = 999 }, "PARITY_UNPROVEN_UNBOUND", "issue_bound"},
		{"blanket_add", func(in *parityGradeInput) {
			s := "git add -A\ngit commit -s -- foo.go"
			in.GitCmds = &s
		}, "PARITY_UNPROVEN_BLANKET_ADD", "by_pathspec"},
		{"no_signoff", func(in *parityGradeInput) {
			s := "git commit -- foo.go bar.go"
			in.GitCmds = &s
		}, "PARITY_UNPROVEN_NO_SIGNOFF", "signed_off"},
		{"dirty_tree", func(in *parityGradeInput) {
			dirty := false
			in.LaneClean = &dirty
		}, "PARITY_UNPROVEN_TREE_DIRTY", "lane_tree_clean"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := provenInputs()
			tc.mutate(&in)
			g := parityGradeFold(in)
			if g.Verdict != tc.verdict {
				t.Errorf("verdict = %q, want %q", g.Verdict, tc.verdict)
			}
			if g.FailedRung == nil || *g.FailedRung != tc.rung {
				t.Errorf("failed_rung = %v, want %q", g.FailedRung, tc.rung)
			}
		})
	}
}

func TestParityGradeNoLogIsUnobserved(t *testing.T) {
	in := provenInputs()
	in.GitCmds = nil // no run log
	g := parityGradeFold(in)
	if g.Proven {
		t.Error("no-log should not be proven")
	}
	if g.Verdict != "PARITY_UNOBSERVED" {
		t.Errorf("verdict = %q, want PARITY_UNOBSERVED", g.Verdict)
	}
	if g.FailedRung == nil || *g.FailedRung != "by_pathspec" {
		t.Errorf("failed_rung = %v, want by_pathspec", g.FailedRung)
	}
	if g.Rungs["by_pathspec"] != nil {
		t.Error("by_pathspec rung should be nil (unobserved) with no log")
	}
}

func TestParityGradeUnknownLaneTreeIsUnobserved(t *testing.T) {
	in := provenInputs()
	in.LaneClean = nil // unknown lane tree
	g := parityGradeFold(in)
	if g.Proven {
		t.Error("unknown lane tree should not be proven")
	}
	if g.Verdict != "PARITY_UNOBSERVED" {
		t.Errorf("verdict = %q, want PARITY_UNOBSERVED", g.Verdict)
	}
	if g.FailedRung == nil || *g.FailedRung != "lane_tree_clean" {
		t.Errorf("failed_rung = %v, want lane_tree_clean", g.FailedRung)
	}
}

func TestParityGradeFirstFailureWinsOrdering(t *testing.T) {
	// Both unwitnessed AND unbound — the earlier rung (witnessed) reports.
	in := provenInputs()
	in.Witnessed = false
	in.Issue = 999
	g := parityGradeFold(in)
	if g.Verdict != "PARITY_UNPROVEN_UNWITNESSED" {
		t.Errorf("verdict = %q, want PARITY_UNPROVEN_UNWITNESSED (earliest failing rung)", g.Verdict)
	}
}

func TestParityBarShape(t *testing.T) {
	rungs := map[string]bool{}
	tokens := map[string]bool{}
	for _, bar := range parityBar {
		if rungs[bar.rung] {
			t.Errorf("duplicate rung name %q", bar.rung)
		}
		if tokens[bar.token] {
			t.Errorf("duplicate verdict token %q", bar.token)
		}
		rungs[bar.rung] = true
		tokens[bar.token] = true
		if !strings.HasPrefix(bar.token, "PARITY_UNPROVEN_") {
			t.Errorf("token %q must start with PARITY_UNPROVEN_", bar.token)
		}
	}
}

// --- evaluate() over injected git/witness seams (the I/O layer, hermetic) ---

// installParitySeams swaps the git/witness/lane seams for the duration of a test.
func installParitySeams(t *testing.T, gitRunner func(string, ...string) (int, string), wit parityWitnessResult) {
	t.Helper()
	oldGit, oldWit, oldLane := parityGitRunner, parityWitnessRunner, parityLaneCleanCheck
	parityGitRunner = gitRunner
	parityWitnessRunner = func(string, string) parityWitnessResult { return wit }
	parityLaneCleanCheck = parityLaneTreeCleanGit // exercised via the (seamed) git runner
	t.Cleanup(func() {
		parityGitRunner, parityWitnessRunner, parityLaneCleanCheck = oldGit, oldWit, oldLane
	})
}

func TestParityEvaluateProvenOverSeams(t *testing.T) {
	git := func(root string, args ...string) (int, string) {
		switch {
		case len(args) >= 1 && args[0] == "rev-parse":
			return 0, "7528df3\n"
		case len(args) >= 1 && args[0] == "show":
			return 0, realGLMSubject + "\n"
		case len(args) >= 1 && args[0] == "diff-tree":
			return 0, "internal/recall/ctxplan_store.go\n"
		case len(args) >= 1 && args[0] == "status":
			return 0, "" // clean lane tree
		}
		return 1, ""
	}
	installParitySeams(t, git, parityWitnessResult{Witnessed: true, Verdict: strPtr("OK"), Witness: strPtr("diff-witnessed")})

	// No --log ⇒ behavioral rungs unobserved ⇒ PARITY_UNOBSERVED, not proven.
	p := parityEvaluate("/repo", parityEvalArgs{Commit: "7528df3", Issue: 545, IssueSet: true, Backend: "opencode", LaneTree: "internal/recall"})
	if p.Verdict != "PARITY_UNOBSERVED" {
		t.Errorf("without --log, verdict = %q, want PARITY_UNOBSERVED", p.Verdict)
	}
	if p.Commit != "7528df3" {
		t.Errorf("commit short sha = %q, want 7528df3", p.Commit)
	}
	if p.OK {
		t.Error("unobserved must not be OK")
	}
}

func TestParityEvaluateUnwitnessedRefutes(t *testing.T) {
	git := func(root string, args ...string) (int, string) {
		switch {
		case len(args) >= 1 && args[0] == "rev-parse":
			return 0, "dead\n"
		case len(args) >= 1 && args[0] == "show":
			return 0, realGLMSubject + "\n"
		case len(args) >= 1 && args[0] == "diff-tree":
			return 0, "internal/recall/ctxplan_store.go\n"
		}
		return 1, ""
	}
	installParitySeams(t, git, parityWitnessResult{Witnessed: false, Verdict: strPtr("CLAIM_UNWITNESSED"), Witness: strPtr("subject-only")})

	p := parityEvaluate("/repo", parityEvalArgs{Commit: "dead", Issue: 545, IssueSet: true, Backend: "opencode"})
	if p.Verdict != "PARITY_UNPROVEN_UNWITNESSED" {
		t.Errorf("verdict = %q, want PARITY_UNPROVEN_UNWITNESSED", p.Verdict)
	}
	if p.OK {
		t.Error("an unwitnessed canary must not be OK")
	}
}

func TestRunDispatchParityCanaryRequiresCommit(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDispatchParityCanary(&stdout, &stderr, nil); code != 2 {
		t.Errorf("missing --commit should exit 2, got %d", code)
	}
	if !strings.Contains(stderr.String(), "--commit is required") {
		t.Errorf("stderr should name the missing flag, got %q", stderr.String())
	}
}

// TestRunDispatchParityCanaryIssueProvidedVsOmitted pins the Python-faithful
// issue: int | None semantics: a PROVIDED --issue (even 0) is "set" and binds to
// that literal #N; an omitted --issue falls back to the any-#N branch (JSON null).
func TestRunDispatchParityCanaryIssueProvidedVsOmitted(t *testing.T) {
	git := func(root string, args ...string) (int, string) {
		switch {
		case len(args) >= 1 && args[0] == "rev-parse":
			return 0, "abc123\n"
		case len(args) >= 1 && args[0] == "show":
			return 0, "chore: bump #0 placeholder (#0)\n"
		case len(args) >= 1 && args[0] == "diff-tree":
			return 0, "x.go\n"
		}
		return 1, ""
	}
	installParitySeams(t, git, parityWitnessResult{Witnessed: true, Verdict: strPtr("OK"), Witness: strPtr("diff-witnessed")})

	// --issue 0 is PROVIDED ⇒ set ⇒ JSON carries "issue": 0 (not null).
	var out0, err0 bytes.Buffer
	runDispatchParityCanary(&out0, &err0, []string{"--commit", "abc123", "--issue", "0", "--workspace", "/repo", "--json"})
	if !strings.Contains(out0.String(), "\"issue\": 0") {
		t.Errorf("--issue 0 must be treated as SET (\"issue\": 0), got:\n%s", out0.String())
	}

	// --issue omitted ⇒ unset ⇒ JSON carries "issue": null.
	var out1, err1 bytes.Buffer
	runDispatchParityCanary(&out1, &err1, []string{"--commit", "abc123", "--workspace", "/repo", "--json"})
	if !strings.Contains(out1.String(), "\"issue\": null") {
		t.Errorf("omitted --issue must be unset (\"issue\": null), got:\n%s", out1.String())
	}
}

func TestRunDispatchParityCanaryJSONShape(t *testing.T) {
	git := func(root string, args ...string) (int, string) {
		switch {
		case len(args) >= 1 && args[0] == "rev-parse":
			return 0, "7528df3\n"
		case len(args) >= 1 && args[0] == "show":
			return 0, realGLMSubject + "\n"
		case len(args) >= 1 && args[0] == "diff-tree":
			return 0, "internal/recall/ctxplan_store.go\n"
		}
		return 1, ""
	}
	installParitySeams(t, git, parityWitnessResult{Witnessed: true, Verdict: strPtr("OK"), Witness: strPtr("diff-witnessed")})

	var stdout, stderr bytes.Buffer
	code := runDispatchParityCanary(&stdout, &stderr, []string{"--commit", "7528df3", "--issue", "545", "--workspace", "/repo", "--json"})
	// No --log/--lane-tree ⇒ unobserved ⇒ exit 1.
	if code != 1 {
		t.Errorf("unobserved canary should exit 1, got %d (stderr=%q)", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{parityCanarySchema, "\"verdict\": \"PARITY_UNOBSERVED\"", "\"backend\": \"opencode\""} {
		if !strings.Contains(out, want) {
			t.Errorf("json output missing %q; got:\n%s", want, out)
		}
	}
}
