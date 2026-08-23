package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// TestHandoffToWitnessedCloseChain (#1462) is the smoke the repo has never had:
// every existing per-stage test mocks the boundary to its neighbor (routing
// fixtures a hand-built dispatchtick.Issue; issue_resolve_witnessed_test.py
// monkeypatches mod.reverify so the real `dos commit-audit` subprocess is never
// exercised). This test instead threads ONE fixture's real data through every
// real function in the chain -- handoff -> issue plan -> route -> prompt render
// -> a real git commit -> a real `dos commit-audit` invocation -> a real
// `tools/issue_resolve_witnessed.py` dry-run -- and asserts the same in-scope/
// out-of-scope/done-condition/witness/acceptance-gate text survives intact end
// to end. It skips gracefully (not fails) when git/dos/python are unavailable,
// matching the exec.LookPath("dos") pattern already used in cmd/fak/dojorsi.go.
func TestHandoffToWitnessedCloseChain(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dosPath, err := exec.LookPath("dos")
	if err != nil {
		t.Skip("dos not on PATH")
	}
	pythonPath := ""
	for _, cand := range []string{"python3", "python"} {
		if p, err := exec.LookPath(cand); err == nil {
			pythonPath = p
			break
		}
	}
	if pythonPath == "" {
		t.Skip("no python3/python on PATH")
	}

	const (
		inScope       = "Add the smoke test that threads real handoff data through routing and prompt rendering (#1462)."
		outOfScope    = "No fleet dispatch, no GPU/model work; one issue, one lane only."
		doneCondition = "go test ./cmd/fak/... -run TestHandoffToWitnessedCloseChain -v is green."
		witnessText   = "A hermetic test proves data survives handoff, issue-plan, route, and prompt render without mocking commit-audit."
		acceptGate    = "The rendered agent issue brief contains the same in-scope/out-of-scope/done-condition/witness text as the fixture."
	)

	// Step 1: build a Handoff fixture in-process (pure Go, no gh).
	handoff := taskmgr.Handoff{
		Schema:       taskmgr.SchemaHandoff,
		CurrentState: "Prior stages of #1462 are wired; the chain has never been proven end to end.",
		Task: taskmgr.HandoffTask{
			TaskID: "task_1462_chain_smoke",
			Title:  "Prove the handoff-to-close chain",
			State:  taskmgr.StateDone,
			Witness: &taskmgr.WitnessRecord{
				VerifiedState: taskmgr.VerifiedDone,
				Source:        "manual",
			},
		},
		NextSteps: []taskmgr.HandoffNextStep{{
			Key:            "task_1462_chain_smoke/prove-chain",
			Title:          "test(cmd): prove handoff-to-close chain end to end",
			Reason:         "No single witnessed run has yet proven create, route, dispatch, commit, audit, close end to end.",
			InScope:        inScope,
			OutOfScope:     outOfScope,
			DoneCondition:  doneCondition,
			Witness:        witnessText,
			AcceptanceGate: acceptGate,
			Lane:           "cmd",
			Paths:          []string{"cmd/fak/handoff_chain_smoke_test.go"},
			Labels:         []string{"priority/P1"},
		}},
	}

	// Step 2: BuildHandoffIssuePlan renders the real issue body (real
	// HandoffIssueBody markdown, not a hand-authored fixture body).
	plan := taskmgr.BuildHandoffIssuePlan(handoff, nil)
	if len(plan) != 1 {
		t.Fatalf("plan rows = %d, want 1", len(plan))
	}
	row := plan[0]
	if row.Action != "create" {
		t.Fatalf("plan action = %q, want create", row.Action)
	}
	for _, want := range []string{inScope, outOfScope, doneCondition, witnessText, acceptGate} {
		if !strings.Contains(row.Body, want) {
			t.Fatalf("rendered issue body missing %q:\n%s", want, row.Body)
		}
	}

	// Step 3: construct a dispatchtick.Issue from the plan row's real body.
	issueNumber := 194613 // synthetic; never a real GitHub issue number
	labels := make([]dispatchtick.IssueLabel, 0, len(row.Labels))
	for _, l := range row.Labels {
		labels = append(labels, dispatchtick.IssueLabel{Name: l})
	}
	issue := dispatchtick.Issue{
		Number: issueNumber,
		Title:  row.Title,
		Body:   row.Body,
		Labels: labels,
	}

	// Step 4: route the issue with a taxonomy fixture (pure Go, no gh).
	taxonomy := dispatchtick.LaneTaxonomy{
		Concurrent: []string{"cmd", "docs", "tools"},
		Trees: map[string][]string{
			"cmd":   {"cmd/**"},
			"docs":  {"docs/**"},
			"tools": {"tools/**"},
		},
	}
	route := dispatchtick.RouteIssue(issue, taxonomy, dispatchtick.RouteOptions{})
	if route.Lane != "cmd" {
		t.Fatalf("routed lane = %q, want cmd (route=%+v)", route.Lane, route)
	}

	// Step 5: render the agent-facing prompt from the routed issue's real data.
	prompt := dispatchtick.RenderIssuePrompt(dispatchtick.IssuePromptInput{
		Number:            issue.Number,
		Title:             issue.Title,
		Body:              issue.Body,
		Labels:            row.Labels,
		Lane:              route.Lane,
		Workspace:         ".",
		DevelopmentBranch: "main",
	})

	// Step 6: the rendered "agent issue brief" must carry the SAME text the
	// fixture set on the HandoffNextStep -- proving nothing was dropped or
	// mangled across handoff -> issue-plan -> route -> prompt.
	for _, want := range []string{inScope, outOfScope, doneCondition, witnessText, acceptGate} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("rendered prompt missing %q:\n%s", want, prompt)
		}
	}
	if !strings.Contains(prompt, "agent issue brief") {
		t.Fatalf("rendered prompt missing the agent issue brief section:\n%s", prompt)
	}

	// Step 7: a real scratch git repo, seeded, then one commit whose SUBJECT
	// follows the prompt's own commit-binding rule -- citing the SAME issue
	// number/lane the router/prompt just produced, not hardcoded.
	repo := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}
	if _, err := exec.Command("git", "-C", repo, "init", "-q", "-b", "main").CombinedOutput(); err != nil {
		git("init", "-q")
		git("symbolic-ref", "HEAD", "refs/heads/main")
	}
	git("config", "user.email", "t@t")
	git("config", "user.name", "t")
	seedPath := filepath.Join(repo, "seed.txt")
	if err := os.WriteFile(seedPath, []byte("seed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "seed.txt")
	git("commit", "-qm", "seed")

	target := filepath.Join("cmd", "fak", "chain_smoke.go")
	if err := os.MkdirAll(filepath.Join(repo, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	src := "package fak\n\n// chainSmoke is a throwaway marker for the #1462 chain-smoke fixture commit.\nvar chainSmoke = true\n"
	if err := os.WriteFile(filepath.Join(repo, target), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", filepath.ToSlash(target))
	subject := "feat(cmd): add handoff chain smoke fixture (#194613) (fak cmd)"
	git("commit", "-qm", subject)
	sha := strings.TrimSpace(git("rev-parse", "HEAD"))

	// Step 8: shell to the REAL dos binary -- not a mock -- and assert the
	// real commit-audit verdict. requireDosCommitAuditOK proves the commit is
	// non-empty with git FIRST, so an "EMPTY" verdict can only be dos's read
	// (#5519), never this fixture's setup.
	requireDosCommitAuditOK(t, dosPath, repo, sha, filepath.ToSlash(target))

	// Step 9: build an issue_closure_audit-shaped fixture referencing the REAL
	// sha + subject computed above.
	auditFixture := map[string]any{
		"issues": []map[string]any{{
			"number": issueNumber,
			"bucket": "OPEN_WITNESSED",
			"witnessed_commits": []map[string]any{{
				"sha":     sha,
				"subject": subject,
			}},
		}},
	}
	fixtureBytes, err := json.Marshal(auditFixture)
	if err != nil {
		t.Fatal(err)
	}
	fixturePath := filepath.Join(t.TempDir(), "closure_audit.json")
	if err := os.WriteFile(fixturePath, fixtureBytes, 0o644); err != nil {
		t.Fatal(err)
	}

	// Step 10: dry-run tools/issue_resolve_witnessed.py against the REAL
	// commit-audit verdict computed above -- proving it flows through to a
	// real "would-close" decision, not a hand-faked reverify().
	requireWouldCloseWitness(t, pythonPath, repo, fixturePath)
}

// --------------------------------------------------------------------------
// Shared smoke plumbing (#5519). Both this smoke and the #1908 rotation smoke
// (relay_handoff_rotate_close_test.go) shell to the same two real subprocesses --
// `dos commit-audit` and `tools/issue_resolve_witnessed.py` -- so the reads that
// have to be robust and self-explaining under fleet contention live here once.
// --------------------------------------------------------------------------

// dosAuditMaxAttempts bounds the retry in requireDosCommitAuditOK. Three attempts
// cover a transient capped-git read (#5519) without letting a genuinely stuck dos
// stall the package: an unreadable commit fails after three quoted attempts.
const dosAuditMaxAttempts = 3

// dosAuditEmptyReason is the exact reason `dos commit-audit` emits when it decided
// the commit touched nothing -- dos/commit_audit.py's
// "code-effect claim but the commit is EMPTY (touched no files)". Matched as a
// substring so a reworded suffix still classifies, but nothing else does.
const dosAuditEmptyReason = "the commit is EMPTY"

// dosAuditRow is the subset of `dos commit-audit --json`'s per-commit object these
// smokes judge. `reason` is carried so a failure quotes dos's own words.
type dosAuditRow struct {
	SHA     string `json:"sha"`
	Verdict string `json:"verdict"`
	Witness string `json:"witness"`
	Reason  string `json:"reason"`
}

// dosAuditOutcome is what ONE `dos commit-audit` invocation proved, and therefore
// whether retrying it could ever be honest.
type dosAuditOutcome int

const (
	// dosAuditWitnessed: OK / diff-witnessed on a clean exit -- the property holds.
	dosAuditWitnessed dosAuditOutcome = iota
	// dosAuditStableFail: dos read the commit and DISAGREED about its diff. A real
	// regression looks exactly like this, so it must never be retried away.
	dosAuditStableFail
	// dosAuditUnreadable: dos produced no judgeable verdict at all -- no parseable
	// row (the contract_error arm prints to stderr and emits no JSON), or the #5519
	// "the commit is EMPTY" misreport over a commit git says touched files. Retryable.
	dosAuditUnreadable
)

// classifyDosAudit decides which of the three outcomes one invocation produced, given
// its exit error and its stdout. Kept PURE (no subprocess, no *testing.T) so the
// "a stable disagreement is never retried" rule is itself unit-testable --
// see handoff_chain_audit_classify_test.go.
//
// localNonEmpty is the caller's own git witness that the commit touched files; when
// git says the commit is non-empty, dos's "EMPTY" can only be a failed read. When the
// caller has no local witness, an EMPTY verdict is taken at face value as a stable
// disagreement -- the classifier never invents a reason to retry.
func classifyDosAudit(runErr error, stdout []byte, localNonEmpty bool) (dosAuditOutcome, dosAuditRow, bool) {
	var rows []dosAuditRow
	if err := json.Unmarshal(stdout, &rows); err != nil || len(rows) != 1 {
		return dosAuditUnreadable, dosAuditRow{}, false
	}
	row := rows[0]
	if runErr == nil && row.Verdict == "OK" && row.Witness == "diff-witnessed" {
		return dosAuditWitnessed, row, true
	}
	if localNonEmpty && strings.Contains(row.Reason, dosAuditEmptyReason) {
		return dosAuditUnreadable, row, true
	}
	return dosAuditStableFail, row, true
}

// dosAuditAttempt records one `dos commit-audit` invocation verbatim so a final
// failure can quote EVERY attempt -- exit error, stdout and stderr kept apart --
// instead of leaving the next reader to re-run the smoke to find out what happened.
type dosAuditAttempt struct {
	n       int
	elapsed time.Duration
	err     error
	stdout  string
	stderr  string
}

func (a dosAuditAttempt) String() string {
	return fmt.Sprintf("  attempt %d  elapsed=%s  exit=%v\n    stdout: %s\n    stderr: %s",
		a.n, a.elapsed.Round(time.Millisecond), a.err,
		strings.TrimSpace(a.stdout), strings.TrimSpace(a.stderr))
}

// requireCommitTouches is the LOCAL witness that commit sha in repo really did touch
// files, run before dos is consulted at all. It is the same `git show --numstat` read
// dos performs internally, which buys two things: an "EMPTY" audit verdict can be
// attributed to dos's read rather than to this test's fixture setup, and git's object
// cache for the scratch repo is already warm when dos makes its own capped read.
// Returns the touched paths (slash-normalised).
func requireCommitTouches(t *testing.T, repo, sha, wantPath string) []string {
	t.Helper()
	cmd := exec.Command("git", "-C", repo, "show", "--numstat", "--format=", "--no-renames", sha)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("git show --numstat %s in %s: %v\nstderr: %s", sha, repo, err, strings.TrimSpace(stderr.String()))
	}
	var files []string
	for _, line := range strings.Split(stdout.String(), "\n") {
		parts := strings.Split(strings.TrimRight(line, "\r"), "\t")
		if len(parts) != 3 {
			continue
		}
		if p := strings.TrimSpace(parts[2]); p != "" {
			files = append(files, filepath.ToSlash(p))
		}
	}
	if len(files) == 0 {
		t.Fatalf("fixture commit %s touched NO files -- this smoke's own setup is broken, "+
			"not dos: git show --numstat printed %q", sha, stdout.String())
	}
	if wantPath != "" {
		found := false
		for _, f := range files {
			if f == wantPath {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("fixture commit %s touched %v, want it to include %q", sha, files, wantPath)
		}
	}
	return files
}

// requireDosCommitAuditOK shells to the REAL dos binary and requires the
// OK/diff-witnessed verdict for sha in repo.
//
// It is deliberately not a bare one-shot exec. `dos commit-audit` reads the commit
// through git subprocesses the kernel caps at 10s each (dos/vcs.py `_GIT_TIMEOUT_S`);
// on timeout that read returns None, and dos/commit_audit.py `read_commit` folds
// "could not read" into files=() / is_empty=True, which classifies as
// CLAIM_UNWITNESSED "the commit is EMPTY (touched no files)" and exits non-zero --
// #5519, seen once under fleet contention on a commit that demonstrably touched a
// file. That misreport is transient and is NOT a judgement about the diff, so:
//
//  1. requireCommitTouches proves LOCALLY that the commit is non-empty, so an EMPTY
//     verdict can only come from dos's read;
//  2. a read-shaped failure -- no parseable verdict row, or exactly the EMPTY
//     misreport above -- is retried a bounded dosAuditMaxAttempts times;
//  3. ANY other non-OK verdict fails immediately and is never retried: a real
//     classification change must still red this smoke on the first attempt;
//  4. the final failure quotes every attempt's exit error, stdout and stderr next to
//     git's own file list, so the reason is on screen without a re-run.
func requireDosCommitAuditOK(t *testing.T, dosPath, repo, sha, wantPath string) {
	t.Helper()
	local := requireCommitTouches(t, repo, sha, wantPath)

	attempts := make([]dosAuditAttempt, 0, dosAuditMaxAttempts)
	for n := 1; n <= dosAuditMaxAttempts; n++ {
		cmd := exec.Command(dosPath, "commit-audit", sha, "--workspace", repo, "--json")
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		started := time.Now()
		runErr := cmd.Run()
		attempt := dosAuditAttempt{
			n: n, elapsed: time.Since(started), err: runErr,
			stdout: stdout.String(), stderr: stderr.String(),
		}
		attempts = append(attempts, attempt)

		switch outcome, row, _ := classifyDosAudit(runErr, stdout.Bytes(), len(local) > 0); outcome {
		case dosAuditWitnessed:
			return // the property under test held
		case dosAuditStableFail:
			t.Fatalf("dos commit-audit did not witness %s: verdict=%q witness=%q reason=%q exit=%v\n"+
				"git says that commit touched %v, and this is a STABLE disagreement about the diff "+
				"(not the #5519 empty-read misreport), so it is not retried.\n%s",
				sha, row.Verdict, row.Witness, row.Reason, runErr, local, attempt)
		}
		// dosAuditUnreadable: no single parseable row, or dos reported EMPTY for a commit
		// git just said touched files. Both are #5519's capped-git-read symptom -- retry.
		// LOG it: a retry that passes silently is how a flake stays invisible, and the
		// -v CI run should carry the evidence that #5519 fired without reddening.
		t.Logf("#5519: dos commit-audit produced no judgeable verdict for %s; "+
			"git says it touched %v, so this is a failed READ, not a disagreement.\n%s",
			sha, local, attempt)
		if n < dosAuditMaxAttempts {
			time.Sleep(time.Duration(n) * time.Second)
		}
	}

	var quoted strings.Builder
	for _, a := range attempts {
		quoted.WriteString(a.String())
		quoted.WriteString("\n")
	}
	t.Fatalf("dos commit-audit never produced an OK/diff-witnessed verdict for %s in %d attempts, "+
		"yet git says the commit touched %v -- so dos could not READ the commit "+
		"(#5519: a timed-out capped git read is reported as %q). Every attempt:\n%s",
		sha, dosAuditMaxAttempts, local, dosAuditEmptyReason+" (touched no files)", quoted.String())
}

// requireWouldCloseWitness dry-runs tools/issue_resolve_witnessed.py over the
// closure-audit fixture at fixturePath and requires the single would_close /
// witness_ok result.
//
// stdout and stderr are captured SEPARATELY on purpose: the script's JSON report is
// on stdout, so merging a stray warning line from stderr into the same buffer turns
// any diagnostic the script prints into an unparseable-JSON failure that hides it.
// Every failure path quotes stderr next to stdout.
func requireWouldCloseWitness(t *testing.T, pythonPath, repo, fixturePath string) {
	t.Helper()
	scriptPath := issueResolveWitnessedScript(t)
	ghDir := t.TempDir()
	ghPath := filepath.Join(ghDir, "gh")
	ghScript := "#!/bin/sh\nif [ \"$1 $2\" = \"issue view\" ]; then echo '{\"body\":\"\",\"labels\":[]}'; fi\nexit 0\n"
	if err := os.WriteFile(ghPath, []byte(ghScript), 0o755); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(pythonPath, scriptPath,
		"--workspace", repo,
		"--audit-json", fixturePath,
		"--no-require-pushed",
		"--json",
	)
	var stdout, stderr bytes.Buffer
	cmd.Env = append(os.Environ(), "PATH="+ghDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("issue_resolve_witnessed.py: %v\nstdout: %s\nstderr: %s",
			err, strings.TrimSpace(stdout.String()), strings.TrimSpace(stderr.String()))
	}
	var report struct {
		Verdict string           `json:"verdict"`
		Results []map[string]any `json:"results"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("bad issue_resolve_witnessed.py json: %v\nstdout: %s\nstderr: %s",
			err, strings.TrimSpace(stdout.String()), strings.TrimSpace(stderr.String()))
	}
	if len(report.Results) != 1 {
		t.Fatalf("results = %d, want 1\nstdout: %s\nstderr: %s",
			len(report.Results), strings.TrimSpace(stdout.String()), strings.TrimSpace(stderr.String()))
	}
	if action, _ := report.Results[0]["action"].(string); action != "would_close" {
		t.Fatalf("action = %v, want would_close: %+v", report.Results[0]["action"], report.Results[0])
	}
	if ok, _ := report.Results[0]["witness_ok"].(bool); !ok {
		t.Fatalf("witness_ok = %v, want true: %+v", report.Results[0]["witness_ok"], report.Results[0])
	}
}

// issueResolveWitnessedScript resolves tools/issue_resolve_witnessed.py under the
// checkout these smokes run from, failing with a precondition message rather than a
// bare errno when it cannot.
func issueResolveWitnessedScript(t *testing.T) string {
	t.Helper()
	root, err := repoRootFromCwd()
	if err != nil {
		t.Fatalf("%v", err)
	}
	script := filepath.Join(root, "tools", "issue_resolve_witnessed.py")
	if _, statErr := os.Stat(script); statErr != nil {
		t.Fatalf("this smoke needs tools/issue_resolve_witnessed.py under the checkout root "+
			"(resolved %s): %v", root, statErr)
	}
	return script
}

// repoRootFromCwd locates the checkout root these smokes read
// tools/issue_resolve_witnessed.py out of.
//
// The module root is the repository root here, so the go.mod walk-up is the primary
// read: it needs no subprocess and, unlike git, it resolves an EXPORT of the tree
// (`git archive HEAD | tar -x`) to the export rather than to whatever repo happens to
// contain it. `git rev-parse --show-toplevel` stays as the fallback for a checkout
// reached without a go.mod above it. Neither answer is silent on failure: a plain
// `git rev-parse` exits 128 with an empty message when there is no .git, which told
// the reader nothing, so both git's stderr and the directory the walk started from
// are reported together.
func repoRootFromCwd() (string, error) {
	cwd, cwdErr := os.Getwd()
	if cwdErr == nil {
		if root, ok := moduleRootFrom(cwd); ok {
			return root, nil
		}
	}
	var stderr bytes.Buffer
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	cmd.Stderr = &stderr
	out, gitErr := cmd.Output()
	if gitErr == nil {
		if root := strings.TrimSpace(string(out)); root != "" {
			return root, nil
		}
	}
	return "", fmt.Errorf("cannot locate the checkout root: no go.mod walking up from %q (cwd err: %v), "+
		"and `git rev-parse --show-toplevel` failed: %v (stderr: %q) -- this test must run inside the fak "+
		"checkout or an export of it", cwd, cwdErr, gitErr, strings.TrimSpace(stderr.String()))
}

// moduleRootFrom walks up from start for the nearest directory holding a go.mod, and
// reports whether it found one. Pure apart from the stat, and separated from
// repoRootFromCwd so the case that matters -- an EXPORT of the tree with no .git in
// it -- is provable without a process-wide chdir; see
// handoff_chain_audit_classify_test.go.
func moduleRootFrom(start string) (string, bool) {
	for dir := start; ; {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}
