package main

// superloop_fleet_test.go — the acceptance for the `fak superloop fleet` manager
// (#4958).
//
// The manager's whole claim is that it adds NO rival walker and NO private fold: its
// read-only verbs fold the same superloop.Walk every intent uses, and walk/run delegate
// to the standing shells verbatim. A claim like that is worth exactly what a test of it
// is worth, so these assert the identity directly — `fleet walk` against `walk
// tend-fleet`, and `fleet next` against the head `fleet status` shows — rather than
// re-asserting the numbers by hand.

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// fleetJSON drives one `superloop fleet ...` verb through the REAL top-level router, so
// a manager that is not routed from runSuperloop fails here rather than passing on a
// direct call nobody can make.
func fleetJSON(t *testing.T, root string, argv ...string) (int, []byte, string) {
	t.Helper()
	var out, errb bytes.Buffer
	full := append([]string{"fleet"}, argv...)
	full = append(full, "--workspace", root, "--json")
	code := runSuperloop(&out, &errb, full)
	return code, out.Bytes(), errb.String()
}

// The verb is REACHABLE from the top-level router and refuses what it does not know.
// The routing half matters most: without the `case "fleet"` hunk in superloop.go the
// whole manager is dead code that its own unit tests would still call happily.
func TestSuperloopFleetVerbIsRoutedAndRefusesUnknownSubcommands(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runSuperloop(&out, &errb, []string{"fleet", "--help"}); code != 0 {
		t.Fatalf("fleet --help exit=%d, want 0: %s", code, errb.String())
	}
	for _, want := range []string{"fak superloop fleet status", "next", "walk", "run"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("fleet usage does not document %q:\n%s", want, out.String())
		}
	}

	out.Reset()
	errb.Reset()
	code := runSuperloop(&out, &errb, []string{"fleet", "teleport"})
	if code != 2 {
		t.Fatalf("unknown fleet subcommand exit=%d, want 2 (usage refusal): %s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "teleport") {
		t.Fatalf("refusal does not name the unknown subcommand: %s", errb.String())
	}
}

// `fleet walk` DELEGATES to the standing walk shell rather than folding its own report:
// the payload must be identical to `superloop walk tend-fleet`, byte for byte. This is
// the file's central claim — a private walker here would be a second source of truth
// about fleet health, free to drift from the one the drive gate reads.
func TestSuperloopFleetWalkDelegatesToTheStandingShell(t *testing.T) {
	root := t.TempDir()

	_, viaFleet, errb := fleetJSON(t, root, "walk")
	if len(viaFleet) == 0 {
		t.Fatalf("fleet walk emitted no report: %s", errb)
	}
	var direct bytes.Buffer
	var directErr bytes.Buffer
	runSuperloop(&direct, &directErr, []string{"walk", superloopFleetIntent, "--workspace", root, "--json"})

	if !bytes.Equal(viaFleet, direct.Bytes()) {
		t.Fatalf("fleet walk is not the standing walk:\n via fleet: %s\n direct:    %s", viaFleet, direct.String())
	}
	var rep superloop.WalkReport
	if err := json.Unmarshal(viaFleet, &rep); err != nil {
		t.Fatalf("walk json: %v\n%s", err, viaFleet)
	}
	if rep.Name != superloopFleetIntent {
		t.Fatalf("fleet walk walked %q, want the registered %q intent", rep.Name, superloopFleetIntent)
	}
}

// `fleet status` is a FOLD of that same walk, not a recount: every dimension it prints
// has to equal the walk's own. A status that could disagree with the walk is worse than
// no status — it is a second number an operator has to reconcile during an incident.
func TestSuperloopFleetStatusFoldsTheSameWalkNumbers(t *testing.T) {
	root := t.TempDir()

	code, raw, errb := fleetJSON(t, root, "status")
	if len(raw) == 0 {
		t.Fatalf("fleet status emitted nothing (exit %d): %s", code, errb)
	}
	var got superloopFleetReport
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("status json: %v\n%s", err, raw)
	}
	if got.Schema != superloopFleetStatusSchema || got.Intent != superloopFleetIntent {
		t.Fatalf("status payload = schema %q intent %q, want %q/%q", got.Schema, got.Intent, superloopFleetStatusSchema, superloopFleetIntent)
	}
	// Non-vacuity. Membership comes from the REGISTRY, not the workspace, so even an
	// empty temp root walks real members — a zero here would make every equality below
	// trivially true and this test worthless without ever failing.
	if got.Members == 0 || got.Walked+got.Unmeasured == 0 {
		t.Fatalf("the tend-fleet walk enumerated nothing, so the comparisons below are vacuous: %+v", got)
	}

	_, walkRaw, _ := fleetJSON(t, root, "walk")
	var walk superloop.WalkReport
	if err := json.Unmarshal(walkRaw, &walk); err != nil {
		t.Fatalf("walk json: %v\n%s", err, walkRaw)
	}
	for _, d := range []struct {
		name             string
		status, fromWalk int
	}{
		{"members", got.Members, walk.Members},
		{"walked", got.Walked, walk.Walked},
		{"unmeasured", got.Unmeasured, walk.Unmeasured},
		{"dark", got.Dark, walk.Dark},
		{"spinning", got.Spinning, walk.Spinning},
		{"orphaned", got.Orphaned, walk.Orphaned},
		{"total_debt", got.TotalDebt, walk.TotalDebt},
		{"floor", got.Floor, walk.Floor},
	} {
		if d.status != d.fromWalk {
			t.Errorf("status %s = %d but the walk says %d — the fold recounted instead of reading", d.name, d.status, d.fromWalk)
		}
	}
	if got.Satisfied != walk.Satisfied || got.Verdict != walk.Verdict {
		t.Errorf("status verdict/satisfied = %q/%t, walk says %q/%t", got.Verdict, got.Satisfied, walk.Verdict, walk.Satisfied)
	}
	if got.Rollup == nil {
		t.Error("fleet status report missing Rollup summary")
	} else if got.Rollup.LeafMembers != walk.Rollup.LeafMembers {
		t.Errorf("status rollup leaf members %d != walk rollup %d", got.Rollup.LeafMembers, walk.Rollup.LeafMembers)
	}
	// The exit code carries the same verdict the payload does, so a scripted caller
	// that never parses the JSON still reads the truth.
	if wantCode := map[bool]int{true: 0, false: 1}[got.Satisfied]; code != wantCode {
		t.Errorf("status exit=%d with satisfied=%t, want %d", code, got.Satisfied, wantCode)
	}
}

// `fleet next` reports the SAME worst-first selection `status` heads with — and the same
// one `fleet run` would enter, because both take it from superloop.Drive over the same
// walk. If these could diverge, the read-only preview would be advertising a member the
// drive does not pick, which is the one thing a "what would run next" verb must never do.
func TestSuperloopFleetNextAgreesWithTheStatusHead(t *testing.T) {
	root := t.TempDir()

	_, statusRaw, _ := fleetJSON(t, root, "status")
	var status superloopFleetReport
	if err := json.Unmarshal(statusRaw, &status); err != nil {
		t.Fatalf("status json: %v\n%s", err, statusRaw)
	}

	_, nextRaw, errb := fleetJSON(t, root, "next")
	if len(nextRaw) == 0 {
		t.Fatalf("fleet next emitted nothing: %s", errb)
	}
	var next struct {
		Schema    string                  `json:"schema"`
		Decision  superloop.DriveDecision `json:"decision"`
		FrontDoor superloop.FrontDoor     `json:"front_door"`
	}
	if err := json.Unmarshal(nextRaw, &next); err != nil {
		t.Fatalf("next json: %v\n%s", err, nextRaw)
	}
	if next.Schema != superloop.DriveSchema {
		t.Fatalf("next schema = %q, want the shared %q", next.Schema, superloop.DriveSchema)
	}
	// An unsatisfied walk of an empty root must produce a head; without one the
	// comparison below would never run and the test would pass by doing nothing.
	if status.Head == nil {
		t.Fatalf("unsatisfied=%t walk produced no worst-first head, so there is nothing to compare next against: %+v",
			!status.Satisfied, status)
	}
	if next.Decision.Member != status.Head.Member {
		t.Fatalf("next selects %+v but status heads %+v — the preview and the drive disagree",
			next.Decision.Member, status.Head.Member)
	}
	if next.Decision.Debt != status.Head.Debt {
		t.Fatalf("next debt %d vs status head debt %d", next.Decision.Debt, status.Head.Debt)
	}
	// The front-door classification rides along, so the operator is told HOW it would be
	// entered rather than only what.
	if next.Decision.Enter && next.FrontDoor.Kind == "" {
		t.Fatalf("an enterable selection carries no front-door classification: %+v", next)
	}
}
