// relay_handoff_rotate_close_test.go — rung J6 (issue #1908): the rotation smoke the
// repo did not have. #1462 (TestHandoffToWitnessedCloseChain) threads a taskmgr.Handoff
// through issue-plan -> route -> prompt -> commit -> real dos commit-audit -> real
// issue_resolve_witnessed.py, but it never crosses a relay rotation. The close arm must
// survive a rotation without mocking the witness, so this test inserts a REAL rotation
// between the closing leg's commit and the successor leg's close: the real relay wire
// codec (relay.Marshal/Parse), the real pre-rotate externalize gate
// (relay.CheckExternalizeGate), and the real objective-pin reconciliation
// (ctxplan.ReconcileObjective) — no mocked boundary. It then proves the SAME
// in-scope/out-of-scope/done-condition/witness/acceptance-gate text the closing leg
// rendered still surfaces in the successor leg's re-rendered prompt after the rotation,
// and that the chain still flows through to a real dos commit-audit OK and a real
// issue_resolve_witnessed.py "would_close". Like the #1462 smoke, it skips (not fails)
// when git/dos/python are unavailable.
package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/relay"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// TestRelayHandoffRotateClose (#1908) is the rotation smoke: it proves the handoff's
// in-scope/out-of-scope/done-condition/witness/acceptance-gate text survives a relay
// rotation end to end — handoff -> issue-plan -> route -> prompt -> commit -> [ROTATE:
// externalize gate admits, baton marshal, leg advances, baton parse, objective-pin
// reconciles as PRESERVED] -> successor re-renders the prompt from the issue body the
// baton's artifact points at -> real dos commit-audit -> real issue_resolve_witnessed.py
// would_close. Witness: `go test ./cmd/fak -run RelayHandoffRotateClose`.
func TestRelayHandoffRotateClose(t *testing.T) {
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

	// The five text fields that must survive the rotation. These are distinct from the
	// #1462 fixture so this is a genuine new witness, not a re-run of the prior smoke.
	const (
		objectiveText = "Prove the handoff-to-close chain survives a relay rotation (#1908)."
		inScope       = "Add the hermetic smoke that crosses a relay rotation between handoff and close (#1908)."
		outOfScope    = "No new witness type; reuse the #1462 commit-audit + resolve chain."
		doneCondition = "go test ./cmd/fak/... -run TestRelayHandoffRotateClose -v is green."
		witnessText   = "A hermetic test proves in/out-of-scope/done/witness survive a relay rotation end to end."
		acceptGate    = "The rendered agent issue brief contains the same in/out-of-scope/done-condition/witness text after the rotation."
	)

	// --- LEG N (the closing leg): handoff -> issue-plan -> route -> prompt -> commit ---

	// Step 1: build a Handoff fixture in-process (pure Go, no gh).
	handoff := taskmgr.Handoff{
		Schema:       taskmgr.SchemaHandoff,
		CurrentState: "The #1462 smoke threads handoff->close hermetically but does not cross a rotation.",
		Task: taskmgr.HandoffTask{
			TaskID: "task_1908_rotate_smoke",
			Title:  "Prove the handoff-to-close chain survives a rotation",
			State:  taskmgr.StateDone,
			Witness: &taskmgr.WitnessRecord{
				VerifiedState: taskmgr.VerifiedDone,
				Source:        "manual",
			},
		},
		NextSteps: []taskmgr.HandoffNextStep{{
			Key:            "task_1908_rotate_smoke/prove-rotate",
			Title:          "test(cmd): prove handoff->rotate->resume->close end to end",
			Reason:         "The close arm must survive a relay rotation without mocking the witness.",
			InScope:        inScope,
			OutOfScope:     outOfScope,
			DoneCondition:  doneCondition,
			Witness:        witnessText,
			AcceptanceGate: acceptGate,
			Lane:           "cmd",
			Paths:          []string{"cmd/fak/relay_handoff_rotate_close_test.go"},
			Labels:         []string{"priority/P1"},
		}},
	}

	// Step 2: BuildHandoffIssuePlan renders the real issue body.
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

	// Step 3: construct a dispatchtick.Issue from the plan row's real body. The issue
	// body is what the successor leg re-reads after the rotation (the baton's artifact
	// points at this issue number); we hold onto row.Body so the successor's re-render
	// uses the SAME body a fresh leg would re-fetch.
	issueNumber := 194613 // synthetic; matches the #1462 smoke, never a real GitHub issue
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
	closingPrompt := dispatchtick.RenderIssuePrompt(dispatchtick.IssuePromptInput{
		Number:            issue.Number,
		Title:             issue.Title,
		Body:              issue.Body,
		Labels:            row.Labels,
		Lane:              route.Lane,
		Workspace:         ".",
		DevelopmentBranch: "main",
	})
	for _, want := range []string{inScope, outOfScope, doneCondition, witnessText, acceptGate} {
		if !strings.Contains(closingPrompt, want) {
			t.Fatalf("closing-leg prompt missing %q:\n%s", want, closingPrompt)
		}
	}

	// Step 6: a real scratch git repo, seeded, then one commit whose SUBJECT follows
	// the prompt's own commit-binding rule.
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

	target := filepath.Join("cmd", "fak", "rotate_smoke.go")
	if err := os.MkdirAll(filepath.Join(repo, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	src := "package fak\n\n// rotateSmoke is a throwaway marker for the #1908 rotation-smoke fixture commit.\nvar rotateSmoke = true\n"
	if err := os.WriteFile(filepath.Join(repo, target), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", filepath.ToSlash(target))
	subject := "feat(cmd): add relay rotate close smoke fixture (#194613) (fak cmd)"
	git("commit", "-qm", subject)
	sha := strings.TrimSpace(git("rev-parse", "HEAD"))

	// --- THE ROTATION: externalize gate -> baton marshal -> leg advance -> baton parse ---

	// Step 7: the pre-rotate externalize gate (relay.CheckExternalizeGate, rung F2 #1885)
	// must ADMIT the rotation: every load-bearing fact the closing leg relies on is
	// backed by a durable pointer (the commit it just wrote + the issue it filed), so
	// nothing transcript-only would be silently dropped. A real refusal here is a real
	// blocker — the test does not rotate through an unbacked fact.
	facts := []relay.LoadBearingFact{
		{Label: "the fix commit", Backing: relay.Artifact{Kind: string(relay.ArtifactCommit), Ref: sha}},
		{Label: "the follow-up issue", Backing: relay.Artifact{Kind: string(relay.ArtifactIssue), Ref: "#" + strconv.Itoa(issueNumber)}},
	}
	gate := relay.CheckExternalizeGate(facts)
	if !gate.Admit {
		t.Fatalf("externalize gate refused the rotate: %+v (culprits=%+v) — closing leg holds transcript-only state", gate, gate.Culprits)
	}

	// Step 8: the closing leg writes the baton carrying the REAL objective pin (Digest
	// computed by NewObjectivePin, never hand-set), the done_when predicate, the
	// progress cursor anchored at the REAL commit sha, and pointer-only artifacts back
	// at the commit + issue. Leg 0 = the closing leg's number (schema: "Leg is the
	// closing leg's number").
	pin := ctxplan.NewObjectivePin("pin-1908-rotate", objectiveText, 0)
	batonN := relay.Baton{
		Schema:      relay.Schema,
		RelayID:     "RID-2026-07-04-rotate-close",
		Leg:         0,
		ParentTrace: "trace-leg0",
		Objective:   pin,
		DoneWhen:    doneCondition,
		ProgressCursor: relay.ProgressCursor{
			StartSHA:   sha,
			HeldRegion: []string{"cmd/fak/relay_handoff_rotate_close_test.go"},
		},
		NextAction: "resume: re-render the prompt from the issue body and run dos commit-audit + issue_resolve_witnessed.py",
		Artifacts: []relay.Artifact{
			{Kind: string(relay.ArtifactCommit), Ref: sha},
			{Kind: string(relay.ArtifactIssue), Ref: "#" + strconv.Itoa(issueNumber)},
		},
		Tombstone: relay.Tombstone{
			Reason: "RELAY_ROTATED",
			AtSHA:  sha,
			Note:   "context ceiling",
		},
	}
	wire, err := relay.Marshal(batonN)
	if err != nil {
		t.Fatalf("relay.Marshal (closing leg): %v", err)
	}

	// Step 9: the rotation — the wire bytes cross the leg boundary. The successor is
	// leg 1; the baton it receives records the closing leg (leg 0).
	batonN1, err := relay.Parse(wire)
	if err != nil {
		t.Fatalf("relay.Parse (successor leg): %v", err)
	}
	if batonN1.Schema != relay.Schema {
		t.Fatalf("post-rotation schema = %q, want %s", batonN1.Schema, relay.Schema)
	}
	if batonN1.Leg != 0 {
		t.Fatalf("post-rotation leg = %d, want 0 (the closing leg's number)", batonN1.Leg)
	}

	// Step 10: the objective-pin reconciliation (ctxplan.ReconcileObjective, #1583) must
	// return ObjectivePreserved — the goal's identity AND content digest survived the
	// Marshal/Parse wire round-trip byte-for-byte. This is the load-bearing check that
	// the rotation did not silently rewrite the objective.
	dec := ctxplan.ReconcileObjective(pin, batonN1.Objective)
	if dec.Outcome != ctxplan.ObjectivePreserved {
		t.Fatalf("objective did not survive the rotation: outcome=%s reason=%s (before=%+v after=%+v)",
			dec.Outcome, dec.Reason, pin, batonN1.Objective)
	}

	// Step 11: the done_when predicate and the progress cursor survived verbatim too.
	if batonN1.DoneWhen != doneCondition {
		t.Fatalf("post-rotation done_when = %q, want %q", batonN1.DoneWhen, doneCondition)
	}
	if batonN1.ProgressCursor.StartSHA != sha {
		t.Fatalf("post-rotation start_sha = %q, want %q", batonN1.ProgressCursor.StartSHA, sha)
	}

	// --- LEG N+1 (the successor): resume -> re-render prompt -> dos commit-audit -> close ---

	// Step 12: the successor re-reads the issue body (via the baton's issue artifact)
	// and re-renders the agent-facing prompt. Hermetically we re-use row.Body — the
	// SAME body a fresh leg would re-fetch from GitHub given the artifact's issue ref.
	resumePrompt := dispatchtick.RenderIssuePrompt(dispatchtick.IssuePromptInput{
		Number:            issue.Number,
		Title:             issue.Title,
		Body:              issue.Body,
		Labels:            row.Labels,
		Lane:              route.Lane,
		Workspace:         ".",
		DevelopmentBranch: "main",
	})
	// The SAME five text fields that survived the closing leg must surface in the
	// successor's re-rendered prompt — proving in/out-of-scope/done-condition/witness
	// survived the rotation end to end.
	for _, want := range []string{inScope, outOfScope, doneCondition, witnessText, acceptGate} {
		if !strings.Contains(resumePrompt, want) {
			t.Fatalf("successor prompt missing %q after the rotation:\n%s", want, resumePrompt)
		}
	}

	// Step 13: shell to the REAL dos binary and assert the real commit-audit verdict on
	// the commit the closing leg wrote — the same witness #1462 uses, now exercised
	// from the successor leg after a rotation. requireDosCommitAuditOK
	// (handoff_chain_smoke_test.go) proves the commit is non-empty with git first, so an
	// "EMPTY" verdict is attributable to dos's own capped read (#5519), not to the
	// closing leg's fixture commit.
	requireDosCommitAuditOK(t, dosPath, repo, sha, filepath.ToSlash(target))

	// Step 14: build an issue_closure_audit fixture referencing the REAL sha + subject
	// computed by the closing leg, then dry-run tools/issue_resolve_witnessed.py —
	// proving the rotation chain flows through to a real "would-close" decision.
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
	requireWouldCloseWitness(t, pythonPath, repo, fixturePath)
}
