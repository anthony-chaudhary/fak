package sessionctl

// broadcast_test.go — the #2764 witness: a multi-session table test asserting a
// broadcast op lands on EXACTLY the selected set with a per-session
// applied/refused report, and that non-matching (and untagged) sessions are never
// touched. Plus the closed-refusal edge cases and the dos.toml registration gate
// for the BROADCAST_MALFORMED token (the same structural discipline as
// internal/assumecheck/dosreasons_test.go).

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// seedSession materializes trace in tbl (Running) and tags it for broadcast
// selection. Registered for cleanup so the process-global tag registry leaks no
// per-test state.
func seedSession(t *testing.T, tbl *session.Table, trace string, meta BroadcastMeta) {
	t.Helper()
	if _, ok := tbl.Transition(trace, session.Running, ""); !ok {
		t.Fatalf("seed %s: transition refused", trace)
	}
	if !meta.IsZero() {
		TagSession(trace, meta)
		t.Cleanup(func() { ClearSessionTag(trace) })
	}
}

// TestBroadcastLaneSelectsExactSet is the issue's core witness: one op quiesces
// every session on a lane — and ONLY that lane. Sessions on another lane and
// untagged sessions keep their run-state untouched.
func TestBroadcastLaneSelectsExactSet(t *testing.T) {
	tbl := &session.Table{}
	seedSession(t, tbl, "alpha-1", BroadcastMeta{Lane: "cmd", Wave: "w7"})
	seedSession(t, tbl, "alpha-2", BroadcastMeta{Lane: "cmd", Labels: []string{"dogfood"}})
	seedSession(t, tbl, "beta-1", BroadcastMeta{Lane: "gateway", Wave: "w7"})
	seedSession(t, tbl, "untagged-1", BroadcastMeta{})

	rep, ref := Broadcast(tbl, BroadcastSelector{Lane: "cmd"}, OpThrottle, "wave-slowdown")
	if ref != nil {
		t.Fatalf("broadcast refused: %v", ref)
	}
	if rep.Matched != 2 || rep.Applied != 2 || rep.Refused != 0 {
		t.Fatalf("want matched=2 applied=2 refused=0, got matched=%d applied=%d refused=%d", rep.Matched, rep.Applied, rep.Refused)
	}
	got := map[string]BroadcastResult{}
	for _, r := range rep.Results {
		got[r.Trace] = r
	}
	for _, trace := range []string{"alpha-1", "alpha-2"} {
		row, ok := got[trace]
		if !ok {
			t.Fatalf("no report row for matched session %s", trace)
		}
		if !row.Applied || row.Refusal != nil || row.Run != "throttled" {
			t.Errorf("%s: want applied throttled row, got applied=%v run=%q refusal=%v", trace, row.Applied, row.Run, row.Refusal)
		}
		if st := tbl.Get(trace); st.Run != session.Throttled || st.Reason != "wave-slowdown" {
			t.Errorf("%s: want table state Throttled/wave-slowdown, got %s/%q", trace, st.Run, st.Reason)
		}
	}
	// The over-match fence: the other lane and the untagged session are untouched.
	for _, trace := range []string{"beta-1", "untagged-1"} {
		if _, reported := got[trace]; reported {
			t.Errorf("%s: non-matching session appears in the report", trace)
		}
		if st := tbl.Get(trace); st.Run != session.Running {
			t.Errorf("%s: non-matching session was mutated to %s", trace, st.Run)
		}
	}
}

// TestBroadcastReportsPerSessionRefusal proves an accepted broadcast still
// refuses PER SESSION with the drive table's own closed token: a terminal session
// in the matched set reports CONTROL_SESSION_TERMINAL while its live siblings
// apply — a fanned op refuses exactly as it would alone.
func TestBroadcastReportsPerSessionRefusal(t *testing.T) {
	tbl := &session.Table{}
	seedSession(t, tbl, "live-1", BroadcastMeta{Lane: "cmd"})
	seedSession(t, tbl, "dead-1", BroadcastMeta{Lane: "cmd"})
	if _, ok := tbl.Transition("dead-1", session.Stopped, "DONE"); !ok {
		t.Fatal("arrange: could not stop dead-1")
	}

	rep, ref := Broadcast(tbl, BroadcastSelector{Lane: "cmd"}, OpPause, "quiesce")
	if ref != nil {
		t.Fatalf("broadcast refused: %v", ref)
	}
	if rep.Matched != 2 || rep.Applied != 1 || rep.Refused != 1 {
		t.Fatalf("want matched=2 applied=1 refused=1, got matched=%d applied=%d refused=%d", rep.Matched, rep.Applied, rep.Refused)
	}
	for _, r := range rep.Results {
		switch r.Trace {
		case "live-1":
			if !r.Applied || r.Refusal != nil {
				t.Errorf("live-1: want applied, got applied=%v refusal=%v", r.Applied, r.Refusal)
			}
		case "dead-1":
			if r.Applied || r.Refusal == nil || r.Refusal.Reason != session.ReasonControlSessionTerminal {
				t.Errorf("dead-1: want refused %s, got applied=%v refusal=%v", session.ReasonControlSessionTerminal, r.Applied, r.Refusal)
			}
		default:
			t.Errorf("unexpected report row %q", r.Trace)
		}
	}
}

// TestBroadcastWaveAndLabelSelectors exercises the other two selector axes and
// the AND semantics of a multi-axis selector.
func TestBroadcastWaveAndLabelSelectors(t *testing.T) {
	tbl := &session.Table{}
	seedSession(t, tbl, "w7-cmd", BroadcastMeta{Lane: "cmd", Wave: "w7"})
	seedSession(t, tbl, "w7-gw", BroadcastMeta{Lane: "gateway", Wave: "w7"})
	seedSession(t, tbl, "labeled", BroadcastMeta{Lane: "cmd", Labels: []string{"dogfood", "nightrun"}})

	rep, ref := Broadcast(tbl, BroadcastSelector{Wave: "w7"}, OpPause, "")
	if ref != nil {
		t.Fatalf("wave broadcast refused: %v", ref)
	}
	if rep.Matched != 2 {
		t.Fatalf("wave=w7: want 2 matched, got %d", rep.Matched)
	}

	rep, ref = Broadcast(tbl, BroadcastSelector{Label: "nightrun"}, OpResume, "")
	if ref != nil {
		t.Fatalf("label broadcast refused: %v", ref)
	}
	if rep.Matched != 1 || rep.Results[0].Trace != "labeled" {
		t.Fatalf("label=nightrun: want exactly [labeled], got %+v", rep.Results)
	}
	if st := tbl.Get("labeled"); st.Run != session.Running {
		t.Fatalf("labeled: resume did not land, run=%s", st.Run)
	}

	// AND semantics: lane=cmd AND wave=w7 excludes the other-lane wave member and
	// the waveless lane member.
	rep, ref = Broadcast(tbl, BroadcastSelector{Lane: "cmd", Wave: "w7"}, OpResume, "")
	if ref != nil {
		t.Fatalf("lane+wave broadcast refused: %v", ref)
	}
	if rep.Matched != 1 || rep.Results[0].Trace != "w7-cmd" {
		t.Fatalf("lane=cmd wave=w7: want exactly [w7-cmd], got %+v", rep.Results)
	}
}

// TestBroadcastRefusesMalformed pins the closed edge refusals: an empty selector
// (never match-all), an unknown op, a non-lifecycle op, and a nil table are each
// refused BROADCAST_MALFORMED before any session is touched.
func TestBroadcastRefusesMalformed(t *testing.T) {
	tbl := &session.Table{}
	seedSession(t, tbl, "bystander", BroadcastMeta{Lane: "cmd"})

	cases := []struct {
		name string
		tbl  *session.Table
		sel  BroadcastSelector
		op   ControlOp
	}{
		{"empty selector", tbl, BroadcastSelector{}, OpPause},
		{"unknown op", tbl, BroadcastSelector{Lane: "cmd"}, ControlOp("nonsense")},
		{"non-lifecycle op steer", tbl, BroadcastSelector{Lane: "cmd"}, OpSteer},
		{"non-lifecycle op budget", tbl, BroadcastSelector{Lane: "cmd"}, OpBudget},
		{"nil table", nil, BroadcastSelector{Lane: "cmd"}, OpPause},
	}
	for _, tc := range cases {
		rep, ref := Broadcast(tc.tbl, tc.sel, tc.op, "")
		if ref == nil {
			t.Errorf("%s: want BROADCAST_MALFORMED, got accepted report %+v", tc.name, rep)
			continue
		}
		if ref.Reason != BroadcastMalformed {
			t.Errorf("%s: want reason %s, got %s", tc.name, BroadcastMalformed, ref.Reason)
		}
	}
	if st := tbl.Get("bystander"); st.Run != session.Running {
		t.Fatalf("a refused broadcast touched a session: bystander run=%s", st.Run)
	}
}

// TestBroadcastableOpsAreLifecycleSubset pins the broadcastable set to the
// lifecycle subset of the closed vocabulary, in stable vocabulary order — the
// fence that a payload-carrying op can never silently become fan-able without a
// deliberate edit here.
func TestBroadcastableOpsAreLifecycleSubset(t *testing.T) {
	want := []ControlOp{OpPause, OpResume, OpCancel, OpTerminate, OpThrottle}
	got := BroadcastableOps()
	if len(got) != len(want) {
		t.Fatalf("want %v, got %v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("want %v, got %v", want, got)
		}
	}
	for _, op := range got {
		if _, known := Spec(op); !known {
			t.Errorf("broadcastable op %q is not in the closed vocabulary", op)
		}
	}
}

// TestTagRegistryLifecycle pins the tag store's edge behavior: zero tags and
// empty traces are never stored, clear is idempotent.
func TestTagRegistryLifecycle(t *testing.T) {
	TagSession("", BroadcastMeta{Lane: "cmd"})
	TagSession("zero-meta", BroadcastMeta{})
	if _, ok := SessionTag("zero-meta"); ok {
		t.Error("a zero meta was stored")
	}
	TagSession("tagged", BroadcastMeta{Lane: "cmd"})
	t.Cleanup(func() { ClearSessionTag("tagged") })
	if m, ok := SessionTag("tagged"); !ok || m.Lane != "cmd" {
		t.Errorf("want lane=cmd tag, got %+v ok=%v", m, ok)
	}
	ClearSessionTag("tagged")
	ClearSessionTag("tagged") // idempotent
	if _, ok := SessionTag("tagged"); ok {
		t.Error("cleared tag still resolves")
	}
}

// TestBroadcastMalformedRegisteredInDosToml is the structural registration gate:
// the BROADCAST_MALFORMED token must be declared in the workspace dos.toml
// [reasons] table (refusal = true, citing issue #2764), so dos_check_reason
// resolves it instead of classifying it UNCLASSIFIED drift — the same gate shape
// as internal/assumecheck/dosreasons_test.go.
func TestBroadcastMalformedRegisteredInDosToml(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed — cannot locate the test source path")
	}
	root := filepath.Join(filepath.Dir(thisFile), "..", "..")
	b, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read repo dos.toml: %v", err)
	}
	content := string(b)
	header := "[reasons." + string(BroadcastMalformed) + "]"
	i := strings.Index(content, header)
	if i < 0 {
		t.Fatalf("refusal reason %q has no %s table in dos.toml — dos_check_reason would return known=false", BroadcastMalformed, header)
	}
	block := content[i:]
	if j := strings.Index(block[len(header):], "\n["); j >= 0 {
		block = block[:len(header)+j]
	}
	if !strings.Contains(block, "refusal") || !strings.Contains(block, "true") {
		t.Errorf("%s is registered but not marked refusal = true", header)
	}
	if !strings.Contains(block, "issue #2764") {
		t.Errorf("%s registration does not cite issue #2764 — the gate provenance is unbound", header)
	}
}
