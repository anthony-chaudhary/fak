package main

// Tests for `fak steer pause` / `fak steer resume` (#5031): a pause ledgers an
// attributable hold bound to the unit's bound issue and the dispatch route
// fold then skips exactly that issue with the dispatcher's EXISTING
// BLOCKED_BY_HUMAN token; a resume releases it; the verb writes nothing but
// the ledger (pause is not a kill); paused units render as paused, with
// paused-since, in the prs view; and every refusal ledgers nothing.

import (
	"bytes"
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

// steerPauseRoutedPayload is a routed two-lane tick view: the gateway unit's
// bound issue #1146 plus an unrelated tools issue that must never be touched
// by a gateway pause.
func steerPauseRoutedPayload() dispatchtick.RouterPayload {
	return dispatchtick.RouterPayload{
		Schema: dispatchtick.RouterSchema,
		OK:     true,
		Issues: []dispatchtick.IssueRoute{
			{Number: 1146, Title: "gateway: treat same-tick ready as positive", Lane: "gateway", WorkUnit: "leaf", ExpectedSteps: 3},
			{Number: 2000, Title: "tools: unrelated work", Lane: "tools", WorkUnit: "leaf", ExpectedSteps: 2},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"gateway": {Issues: []int{1146}, Count: 1, StepBudget: 3, IssueSteps: map[int]int{1146: 3}, Priority: map[int]int{1146: 5}},
			"tools":   {Issues: []int{2000}, Count: 1, StepBudget: 2, IssueSteps: map[int]int{2000: 2}},
		},
		Counts: dispatchtick.RouterCounts{Open: 2, Routed: 2, RoutedStepBudget: 5},
	}
}

// The done condition's core: `fak steer pause` causes the bound issue to be
// skipped with BLOCKED_BY_HUMAN on the next dispatch tick — through the
// existing guard-escalations seam, no new reason token — and
// `fak steer resume` clears it. The captured post-hold payload IS the tick's
// routing view (dispatchRouteIssuesNative folds holdSteerPausedForRoute in
// before the prereq hold).
func TestSteerPauseHoldsBoundIssueWithBlockedByHumanAndResumeClears(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)

	var stdout, stderr bytes.Buffer
	if code := runSteer(&stdout, &stderr, []string{"pause", "gateway", "-m", "clearly the wrong shape", "--by", "op-jane", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("pause exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	var row steerpr.Pause
	if err := json.NewDecoder(strings.NewReader(stdout.String())).Decode(&row); err != nil {
		t.Fatalf("decode echoed row: %v\n%s", err, stdout.String())
	}
	if row.Schema != steerpr.PauseSchema || row.Action != steerpr.PauseActionPause || row.Leaf != "gateway" || row.By != "op-jane" || row.At == "" {
		t.Fatalf("echoed row = %#v, want an attributable fak.steerpr.pause.v1 pause", row)
	}
	if row.Issue != "#1146" {
		t.Fatalf("row bound issue = %q, want the unit's closure-grade #1146", row.Issue)
	}

	payload := steerPauseRoutedPayload()
	held := holdSteerPausedForRoute(root, payload)

	// Exactly ONE skip, for exactly the bound issue, with exactly the existing token.
	if len(held.SkippedHumanBlocked) != 1 {
		t.Fatalf("skipped = %#v, want exactly the one paused issue", held.SkippedHumanBlocked)
	}
	skip := held.SkippedHumanBlocked[0]
	// Pin the LITERAL wire value: the hold must ride the existing token, and
	// the token must still be the string every consumer already matches on.
	if skip.Number != 1146 || skip.Reason != "BLOCKED_BY_HUMAN" {
		t.Fatalf("skip = %#v, want #1146 under the existing BLOCKED_BY_HUMAN token", skip)
	}
	if skip.WorkUnit != "leaf" || skip.ExpectedSteps != 3 {
		t.Fatalf("skip = %#v, want the route's work-unit/step metadata carried over", skip)
	}
	// The hint names who paused, since when (paused-since), the reason, and the release verb.
	for _, want := range []string{"op-jane", row.At, "clearly the wrong shape", "fak steer resume gateway"} {
		if !strings.Contains(skip.NextAction, want) {
			t.Errorf("skip next-action missing %q: %s", want, skip.NextAction)
		}
	}
	// The paused issue left the routable set; the unrelated issue did not.
	if _, ok := held.Lanes["gateway"]; ok {
		t.Fatalf("gateway lane survived the hold: %#v", held.Lanes)
	}
	if grp, ok := held.Lanes["tools"]; !ok || len(grp.Issues) != 1 || grp.Issues[0] != 2000 {
		t.Fatalf("tools lane = %#v, want the unrelated issue untouched", held.Lanes["tools"])
	}
	if len(held.Issues) != 1 || held.Issues[0].Number != 2000 {
		t.Fatalf("candidates = %#v, want only the unrelated issue", held.Issues)
	}
	if held.Counts.Routed != 1 || held.Counts.RoutedStepBudget != 2 || held.Counts.SkippedHumanBlocked != 1 || held.Counts.SkippedByReason[reasonBlockedByHuman] != 1 {
		t.Fatalf("counts = %#v, want one routed / one human-blocked skip", held.Counts)
	}

	// resume clears the hold: the same routed payload passes through untouched.
	stdout.Reset()
	stderr.Reset()
	if code := runSteer(&stdout, &stderr, []string{"resume", "gateway", "--by", "op-jane"}); code != 0 {
		t.Fatalf("resume exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	var rrow steerpr.Pause
	if err := json.NewDecoder(strings.NewReader(stdout.String())).Decode(&rrow); err != nil {
		t.Fatalf("decode echoed resume row: %v\n%s", err, stdout.String())
	}
	if rrow.Action != steerpr.PauseActionResume || rrow.Leaf != "gateway" {
		t.Fatalf("resume row = %#v, want a resume for gateway", rrow)
	}
	released := holdSteerPausedForRoute(root, steerPauseRoutedPayload())
	if !reflect.DeepEqual(released, steerPauseRoutedPayload()) {
		t.Fatalf("after resume the payload still differs: %#v", released)
	}
}

// Pause is not a kill: the verb's ONLY write is the append-only pause ledger,
// and the route fold rewrites only the FUTURE routing view — an in-flight
// worker's run record is untouched and the fold never mutates its input, so a
// worker mid-flight on the paused intent can still finish and land cleanly.
func TestSteerPauseIsNotAKillOnlyTheLedgerAndTheFutureMove(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)

	// An in-flight worker's run record, standing in for live work on #1146.
	runsDir := filepath.Join(root, ".fak", "runs")
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	runRec := filepath.Join(runsDir, "resolve-1146-20260718-120000.json")
	if err := os.WriteFile(runRec, []byte(`{"issue":1146,"state":"in-flight"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	before := steerPauseSnapshotTree(t, root)

	var stdout, stderr bytes.Buffer
	if code := runSteer(&stdout, &stderr, []string{"pause", "gateway", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("pause exit = %d, want 0; stderr=%s", code, stderr.String())
	}

	after := steerPauseSnapshotTree(t, root)
	ledgerRel, err := filepath.Rel(root, steerpr.PauseLedgerPath(root))
	if err != nil {
		t.Fatal(err)
	}
	delete(after, filepath.ToSlash(ledgerRel))
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("pause wrote more than the ledger:\nbefore=%v\nafter=%v", before, after)
	}

	// The fold leaves its INPUT untouched (copy-on-write, like the known-bad
	// hold): downstream consumers of the pre-hold payload cannot see the hold
	// appear under them, and nothing reachable from the fold touches a run.
	payload := steerPauseRoutedPayload()
	_ = holdSteerPausedForRoute(root, payload)
	if !reflect.DeepEqual(payload, steerPauseRoutedPayload()) {
		t.Fatalf("the hold mutated its input payload: %#v", payload)
	}
	if buf, err := os.ReadFile(runRec); err != nil || string(buf) != `{"issue":1146,"state":"in-flight"}` {
		t.Fatalf("the in-flight run record changed under the pause: %s (%v)", buf, err)
	}
}

// Paused units render as paused, with paused-since, in `fak steer prs` (text
// and JSON), and the render clears on resume. The machine band and residual
// count never move: a pause is a hold, not a verdict.
func TestSteerPauseRendersPausedSinceInPRsUntilResume(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	withSteerRoot(t)

	var stdout, stderr bytes.Buffer
	if code := runSteer(&stdout, &stderr, []string{"pause", "gateway", "--by", "op-jane", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("pause exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	var row steerpr.Pause
	if err := json.NewDecoder(strings.NewReader(stdout.String())).Decode(&row); err != nil {
		t.Fatalf("decode echoed row: %v\n%s", err, stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := runSteerPRs(&stdout, &stderr, []string{"--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("prs exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"**PAUSED** by op-jane since " + row.At, "#1146", "BLOCKED_BY_HUMAN", "fak steer resume gateway"} {
		if !strings.Contains(out, want) {
			t.Errorf("prs render missing %q:\n%s", want, out)
		}
	}
	if !strings.Contains(out, "1 RESIDUAL") {
		t.Fatalf("a pause must not move the residual count:\n%s", out)
	}

	stdout.Reset()
	if code := runSteerPRs(&stdout, &stderr, []string{"--json", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("prs --json exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	var payload struct {
		Pauses map[string]steerpr.Pause `json:"pauses"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("decode json: %v\n%s", err, stdout.String())
	}
	if p, ok := payload.Pauses["gateway"]; !ok || p.By != "op-jane" || p.At != row.At || p.Issue != "#1146" {
		t.Fatalf("json pauses = %#v, want the gateway hold with paused-since", payload.Pauses)
	}

	stdout.Reset()
	if code := runSteer(&stdout, &stderr, []string{"resume", "gateway", "--by", "op-jane"}); code != 0 {
		t.Fatalf("resume exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	stdout.Reset()
	if code := runSteerPRs(&stdout, &stderr, []string{"--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("prs exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if strings.Contains(stdout.String(), "PAUSED") {
		t.Fatalf("resumed unit still renders paused:\n%s", stdout.String())
	}
}

// Refusals: an unknown unit, an unattributable pause, a unit that binds no
// issue (nothing for the dispatch loop to hold), a resume of an unpaused
// unit, and a double pause — the refusal paths ledger nothing (and the double
// pause ledgers exactly once).
func TestSteerPauseRefusalsLedgerNothing(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)

	var stdout, stderr bytes.Buffer
	if code := runSteerPause(&stdout, &stderr, []string{"no-such-leaf", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("unknown unit exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "no forming unit") {
		t.Fatalf("refusal should name the missing unit: %s", stderr.String())
	}

	// No --by and the faked git yields no config user.name.
	stderr.Reset()
	if code := runSteerPause(&stdout, &stderr, []string{"gateway", "--base", "baseref", "--head", "headref"}); code != 2 {
		t.Fatalf("unattributable exit = %d, want 2; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "attributable") {
		t.Fatalf("refusal should say attribution is required: %s", stderr.String())
	}

	stderr.Reset()
	if code := runSteerPause(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("missing unit exit = %d, want 2; stderr=%s", code, stderr.String())
	}

	stderr.Reset()
	if code := runSteerResume(&stdout, &stderr, []string{"gateway", "--by", "op"}); code != 1 {
		t.Fatalf("resume-unpaused exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "not paused") {
		t.Fatalf("refusal should say the unit is not paused: %s", stderr.String())
	}

	if rows := steerpr.LoadPauses(steerpr.PauseLedgerPath(root)); len(rows) != 0 {
		t.Fatalf("refusals wrote %d ledger row(s): %#v", len(rows), rows)
	}

	// A unit with NO closure-grade bound issue cannot be held: the hold lands
	// by issue number, so the honest answer is a refusal, not a no-op row.
	withSteerFakes(t, "\x1efff6666666666666666666666666666666666666\x1ffeat(gateway): unbound work (fak gateway)\x1f\x1f\ninternal/gateway/g.go\n", steerpr.VerdictUnwitnessed)
	stderr.Reset()
	if code := runSteerPause(&stdout, &stderr, []string{"gateway", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("unbound unit exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "binds no issue") {
		t.Fatalf("refusal should say the unit binds no issue: %s", stderr.String())
	}
	if rows := steerpr.LoadPauses(steerpr.PauseLedgerPath(root)); len(rows) != 0 {
		t.Fatalf("the unbound refusal wrote %d ledger row(s): %#v", len(rows), rows)
	}

	// Double pause: the second is refused, and the ledger holds exactly one row.
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	stdout.Reset()
	stderr.Reset()
	if code := runSteerPause(&stdout, &stderr, []string{"gateway", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("first pause exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	stderr.Reset()
	if code := runSteerPause(&stdout, &stderr, []string{"gateway", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("second pause exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "already paused") {
		t.Fatalf("refusal should say the unit is already paused: %s", stderr.String())
	}
	if rows := steerpr.LoadPauses(steerpr.PauseLedgerPath(root)); len(rows) != 1 {
		t.Fatalf("double pause left %d ledger row(s), want exactly 1: %#v", len(rows), rows)
	}
}

// steerPauseSnapshotTree maps every file under root (relative, slashed) to its
// contents — the "what did the verb touch" witness.
func steerPauseSnapshotTree(t *testing.T, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		buf, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		out[filepath.ToSlash(rel)] = string(buf)
		return nil
	})
	if err != nil {
		t.Fatalf("snapshot %s: %v", root, err)
	}
	return out
}
