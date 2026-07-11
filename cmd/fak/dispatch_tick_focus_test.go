package main

import (
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// plantFocusLedger writes objectives to root's trajctl ledger (docs/nightrun/trajctl.jsonl)
// so the focusscore fold the dispatch tick reads sees a real breadth signal.
func plantFocusLedger(t *testing.T, root string, objs []trajctl.Objective) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(trajctl.DefaultLedgerRel))
	for _, o := range objs {
		if err := trajctl.Append(path, trajctl.ObjectiveRecord(o)); err != nil {
			t.Fatalf("append trajctl objective %q: %v", o.ID, err)
		}
	}
}

func decodeTick(t *testing.T, out string) map[string]any {
	t.Helper()
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad tick json: %v\n%s", err, out)
	}
	return got
}

// threeActiveNoTwelve declares 3 active objectives (== DefaultWIPCap) none of which
// reference issue #12 -- so dispatching the docs lane's #12 OPENS a new objective at/over
// the cap.
func threeActiveNoTwelve() []trajctl.Objective {
	return []trajctl.Objective{
		{ID: "alpha", Statement: "resolve #100", Status: trajctl.StatusActive},
		{ID: "beta", Statement: "resolve #200", Status: trajctl.StatusActive},
		{ID: "gamma", Statement: "resolve #300", Status: trajctl.StatusActive},
	}
}

// TestDispatchTickFocusWarnsOverCapNewObjective: at/over the WIP cap, a NEW-objective
// candidate (docs #12) earns the FOCUS_WIP_SATURATED advisory but STILL dispatches
// (warn-first default).
func TestDispatchTickFocusWarnsOverCapNewObjective(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()
	initDispatchGit(t, root)
	plantFocusLedger(t, root, threeActiveNoTwelve())

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit=%d, want 0 (warn-first still dispatches); stderr=%s", code, errb)
	}
	got := decodeTick(t, out)
	if got["action"] != "would_spawn" || got["verdict"] != "WOULD_SPAWN" {
		t.Fatalf("action/verdict = %v/%v, want would_spawn/WOULD_SPAWN (warn-first)", got["action"], got["verdict"])
	}
	focus, ok := got["focus"].(map[string]any)
	if !ok {
		t.Fatalf("missing focus advisory block over cap: %#v", got["focus"])
	}
	if focus["token"] != dispatchtick.FocusWIPSaturated {
		t.Fatalf("focus token = %v, want %q", focus["token"], dispatchtick.FocusWIPSaturated)
	}
	if focus["held"] != false || focus["posture"] != dispatchtick.FocusPostureWarn {
		t.Fatalf("focus held/posture = %v/%v, want false/warn", focus["held"], focus["posture"])
	}
	if focus["active"] != float64(3) || focus["wip_cap"] != float64(3) {
		t.Fatalf("focus active/wip_cap = %v/%v, want 3/3", focus["active"], focus["wip_cap"])
	}
}

// TestDispatchTickFocusHoldRefusesNewObjective: with --focus-hold, the same over-cap
// new-objective candidate is REFUSED with the closed FOCUS_WIP_SATURATED verdict, and the
// hold is benign (exit 0), distinct from a spawn.
func TestDispatchTickFocusHoldRefusesNewObjective(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()
	initDispatchGit(t, root)
	plantFocusLedger(t, root, threeActiveNoTwelve())

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--focus-hold", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit=%d, want 0 (a focus hold is a benign throttle); stderr=%s", code, errb)
	}
	got := decodeTick(t, out)
	if got["action"] != "focus_hold" || got["verdict"] != dispatchtick.FocusWIPSaturated {
		t.Fatalf("action/verdict = %v/%v, want focus_hold/%s", got["action"], got["verdict"], dispatchtick.FocusWIPSaturated)
	}
	if got["ok"] != false {
		t.Fatalf("ok = %v, want false (held)", got["ok"])
	}
	focus, ok := got["focus"].(map[string]any)
	if !ok || focus["held"] != true {
		t.Fatalf("focus block held = %#v, want held=true", got["focus"])
	}
}

// TestDispatchTickFocusContinuationNeverBlocked: an issue an OPEN objective already
// references (#12) is a CONTINUATION -- never held even over cap and even under
// --focus-hold, and it carries no focus advisory.
func TestDispatchTickFocusContinuationNeverBlocked(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()
	initDispatchGit(t, root)
	plantFocusLedger(t, root, []trajctl.Objective{
		{ID: "alpha", Statement: "resolve #100", Status: trajctl.StatusActive},
		{ID: "beta", Statement: "resolve #200", Status: trajctl.StatusActive},
		{ID: "gamma", Statement: "resolve #12", Status: trajctl.StatusActive}, // references docs #12
	})

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--focus-hold", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit=%d, want 0 (continuation never held); stderr=%s", code, errb)
	}
	got := decodeTick(t, out)
	if got["action"] != "would_spawn" {
		t.Fatalf("action = %v, want would_spawn (a continuation is never held by focus)", got["action"])
	}
	if _, present := got["focus"]; present {
		t.Fatalf("continuation carried a focus advisory, want none: %#v", got["focus"])
	}
}

// TestDispatchTickFocusUnderCapByteIdentical: below the WIP cap there is NO focus advisory
// and the tick is byte-identical to a no-ledger tick (the DoD "active < WIPCap" case).
func TestDispatchTickFocusUnderCapByteIdentical(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()
	initDispatchGit(t, root)
	plantFocusLedger(t, root, []trajctl.Objective{
		{ID: "alpha", Statement: "resolve #100", Status: trajctl.StatusActive},
		{ID: "beta", Statement: "resolve #200", Status: trajctl.StatusActive},
	})

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit=%d, want 0; stderr=%s", code, errb)
	}
	got := decodeTick(t, out)
	if got["action"] != "would_spawn" {
		t.Fatalf("action = %v, want would_spawn", got["action"])
	}
	if _, present := got["focus"]; present {
		t.Fatalf("under cap carried a focus advisory, want none (byte-identical): %#v", got["focus"])
	}
}

func TestDispatchFetchScopedIssuesSignalsViewFallback(t *testing.T) {
	oldView, oldBack := dispatchFetchViewIssues, dispatchFetchBacklogIssues
	defer func() { dispatchFetchViewIssues = oldView; dispatchFetchBacklogIssues = oldBack }()
	dispatchFetchViewIssues = func(string, string, int) ([]dispatchtick.Issue, error) { return nil, fmt.Errorf("boom") }
	dispatchFetchBacklogIssues = func(string, int) ([]dispatchtick.Issue, error) {
		return []dispatchtick.Issue{{Number: 1, Title: "fallback"}}, nil
	}
	got, injected, reason, err := dispatchFetchScopedIssuesWithSignal(t.TempDir(), io.Discard, "current", 10)
	if err != nil || injected || len(got) != 1 || reason == "" {
		t.Fatalf("got=%v injected=%v reason=%q err=%v", got, injected, reason, err)
	}
}
