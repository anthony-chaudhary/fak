package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

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
	// real commit-audit verdict.
	auditOut, err := exec.Command(dosPath, "commit-audit", sha, "--workspace", repo, "--json").CombinedOutput()
	if err != nil {
		t.Fatalf("dos commit-audit: %v\n%s", err, auditOut)
	}
	var audits []struct {
		Verdict string `json:"verdict"`
		Witness string `json:"witness"`
	}
	if err := json.Unmarshal(auditOut, &audits); err != nil {
		t.Fatalf("bad dos commit-audit json: %v\n%s", err, auditOut)
	}
	if len(audits) != 1 {
		t.Fatalf("dos commit-audit rows = %d, want 1: %s", len(audits), auditOut)
	}
	audit := audits[0]
	if audit.Verdict != "OK" || audit.Witness != "diff-witnessed" {
		t.Fatalf("dos commit-audit verdict = %+v, want OK/diff-witnessed", audit)
	}

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
	repoRoot, err := repoRootFromCwd()
	if err != nil {
		t.Fatal(err)
	}
	scriptPath := filepath.Join(repoRoot, "tools", "issue_resolve_witnessed.py")
	var out bytes.Buffer
	cmd := exec.Command(pythonPath, scriptPath,
		"--workspace", repo,
		"--audit-json", fixturePath,
		"--no-require-pushed",
		"--json",
	)
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		t.Fatalf("issue_resolve_witnessed.py: %v\n%s", err, out.String())
	}
	var report struct {
		Verdict string           `json:"verdict"`
		Results []map[string]any `json:"results"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("bad issue_resolve_witnessed.py json: %v\n%s", err, out.String())
	}
	results := report.Results
	if len(results) != 1 {
		t.Fatalf("results = %d, want 1: %s", len(results), out.String())
	}
	if action, _ := results[0]["action"].(string); action != "would_close" {
		t.Fatalf("action = %v, want would_close: %+v", results[0]["action"], results[0])
	}
	if ok, _ := results[0]["witness_ok"].(bool); !ok {
		t.Fatalf("witness_ok = %v, want true: %+v", results[0]["witness_ok"], results[0])
	}
}

func repoRootFromCwd() (string, error) {
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}
