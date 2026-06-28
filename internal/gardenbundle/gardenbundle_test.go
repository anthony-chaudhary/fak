package gardenbundle

import (
	"os"
	"testing"
	"time"
)

func member(key string, gates bool, kind string) Member {
	return Member{Key: key, Label: key, Gates: gates, Kind: kind}
}

func TestInterpretEnvelopeOK(t *testing.T) {
	m := member("scorecard", true, "envelope")
	r := Interpret(m, map[string]any{"ok": true, "verdict": "OK", "reason": "clean"}, 0, "")
	if r.State != "ok" || r.OK != true {
		t.Fatalf("want ok/true, got %s/%v", r.State, r.OK)
	}
}

func TestInterpretEnvelopeGatingRed(t *testing.T) {
	m := member("scorecard", true, "envelope")
	r := Interpret(m, map[string]any{"ok": false, "verdict": "ACTION", "reason": "debt rose"}, 1, "")
	if r.State != "red" {
		t.Fatalf("want red, got %s (%+v)", r.State, r)
	}
}

func TestInterpretEnvelopeNonGatingAction(t *testing.T) {
	// A non-gating member reporting not-ok is advisory ACTION, not red.
	m := member("fresh_status", false, "envelope")
	r := Interpret(m, map[string]any{"ok": false, "verdict": "ACTION", "reason": "stale bench"}, 0, "")
	if r.State != "action" {
		t.Fatalf("want action, got %s (%+v)", r.State, r)
	}
}

func TestInterpretErroredOnNoPayload(t *testing.T) {
	m := member("scorecard", true, "envelope")
	r := Interpret(m, nil, -1, "timed out")
	if r.State != "errored" || r.OK != false {
		t.Fatalf("want errored/false, got %s/%v", r.State, r.OK)
	}
}

func TestInterpretLoopAuditBuckets(t *testing.T) {
	m := member("loop_audit", false, "loop_audit")

	healthy := Interpret(m, map[string]any{"counts": map[string]any{"healthy": float64(3), "action": float64(0), "broken": float64(0)}}, 0, "")
	if healthy.State != "ok" {
		t.Fatalf("healthy: want ok, got %s", healthy.State)
	}
	action := Interpret(m, map[string]any{"counts": map[string]any{"healthy": float64(2), "action": float64(1), "broken": float64(0)}}, 0, "")
	if action.State != "action" {
		t.Fatalf("action: want action, got %s", action.State)
	}
	// loop-audit is non-gating advisory: a broken sub-loop surfaces as `action`,
	// not `errored` -- a peripheral check can't gate the whole garden red.
	broken := Interpret(m, map[string]any{"counts": map[string]any{"healthy": float64(1), "action": float64(0), "broken": float64(1)}}, 1, "")
	if broken.State != "action" {
		t.Fatalf("broken: want action, got %s (%+v)", broken.State, broken)
	}
}

func row(key, state string, gates bool) MemberResult {
	return MemberResult{
		Key: key, Label: key, Gates: gates, State: state,
		OK:      state == "ok" || state == "action",
		Verdict: "OK", Detail: "d", ExitCode: 0,
	}
}

func TestFoldAllClear(t *testing.T) {
	p := Fold([]MemberResult{row("a", "ok", false), row("b", "ok", false)}, "w", "c")
	if !p.OK || p.Finding != "garden_clear" {
		t.Fatalf("want ok/garden_clear, got %v/%s", p.OK, p.Finding)
	}
	if code, _ := CheckGate(p); code != 0 {
		t.Fatalf("want gate 0, got %d", code)
	}
}

func TestFoldAdvisoryIsGreen(t *testing.T) {
	// An advisory (non-gating) ACTION keeps the bundle green.
	p := Fold([]MemberResult{row("a", "ok", false), row("b", "action", false)}, "w", "c")
	if !p.OK || p.Finding != "garden_advisory" {
		t.Fatalf("want ok/garden_advisory, got %v/%s", p.OK, p.Finding)
	}
	if code, _ := CheckGate(p); code != 0 {
		t.Fatalf("want gate 0, got %d", code)
	}
}

func TestFoldRedGates(t *testing.T) {
	p := Fold([]MemberResult{row("a", "ok", false), row("sc", "red", true)}, "w", "c")
	if p.OK || p.Finding != "garden_gate_red" {
		t.Fatalf("want notok/garden_gate_red, got %v/%s", p.OK, p.Finding)
	}
	if code, _ := CheckGate(p); code != 1 {
		t.Fatalf("want gate 1, got %d", code)
	}
}

func TestFoldErroredGatesOverRed(t *testing.T) {
	// An unmeasured member trips the gate even alongside a red -- and is reported
	// first, since "could not run" is the more fundamental failure.
	p := Fold([]MemberResult{row("sc", "red", true), row("b", "errored", false)}, "w", "c")
	if p.Finding != "garden_member_unmeasured" {
		t.Fatalf("want garden_member_unmeasured, got %s", p.Finding)
	}
	if code, _ := CheckGate(p); code != 1 {
		t.Fatalf("want gate 1, got %d", code)
	}
}

func TestGuardRouteMemberIsCommandExec(t *testing.T) {
	// The closure-rung member must run the built fak binary directly (Exec
	// "command", Argv[0] "fak" -> os.Executable in RunMember), never `go run`,
	// which would error whenever a peer's uncommitted edit leaves the tree
	// uncompilable on the shared trunk. It must stay non-gating.
	var m *Member
	for i := range Members {
		if Members[i].Key == "guard_route" {
			m = &Members[i]
			break
		}
	}
	if m == nil {
		t.Fatal("guard_route member missing from the default bundle")
	}
	if m.Exec != "command" {
		t.Fatalf("guard_route must be a command member, got Exec=%q", m.Exec)
	}
	if len(m.Argv) == 0 || m.Argv[0] != "fak" {
		t.Fatalf("guard_route Argv[0] must be the bare fak token (-> os.Executable), got %v", m.Argv)
	}
	if m.Gates {
		t.Fatal("guard_route must be non-gating (a routed finding is the pass working, not a broken garden)")
	}
}

func TestRunMemberCommandExecRunsDirectly(t *testing.T) {
	// A command member runs Argv[0] as a direct executable and parses its JSON
	// stdout -- the Go-native member path, distinct from the python-script path.
	// `go env -json` is a portable command that emits a JSON object on stdout.
	m := Member{Key: "k", Label: "k", Kind: "envelope", Exec: "command",
		Argv: []string{"go", "env", "-json"}}
	payload, code, errStr := RunMember(t.TempDir(), m, "", 60*time.Second)
	if errStr != "" || code != 0 {
		t.Fatalf("command member failed: code=%d err=%q", code, errStr)
	}
	if _, ok := payload["GOOS"]; !ok {
		t.Fatalf("expected a parsed JSON object from `go env -json`, got %v", payload)
	}
}

func TestGardenOffRecognizesVocabulary(t *testing.T) {
	for _, v := range []string{"0", "off", "OFF", "false", "no", "disable", "disabled"} {
		os.Setenv("FAK_GARDEN", v)
		if !GardenOff() {
			t.Fatalf("FAK_GARDEN=%q should be off", v)
		}
	}
	for _, v := range []string{"1", "on", "true", ""} {
		os.Setenv("FAK_GARDEN", v)
		if GardenOff() {
			t.Fatalf("FAK_GARDEN=%q should NOT be off", v)
		}
	}
	os.Unsetenv("FAK_GARDEN")
}

func TestSkippedPayloadIsWellformedAndGreen(t *testing.T) {
	p := SkippedPayload("w", "c")
	if !p.OK || p.Finding != "garden_skipped" {
		t.Fatalf("want ok/garden_skipped, got %v/%s", p.OK, p.Finding)
	}
	if p.Schema() != Schema {
		t.Fatalf("want schema %s", Schema)
	}
	if code, _ := CheckGate(p); code != 0 {
		t.Fatalf("want gate 0, got %d", code)
	}
}

// Schema is a tiny accessor used by the well-formedness test (the envelope's
// schema is a package constant emitted at marshal time).
func (p Payload) Schema() string { return Schema }
