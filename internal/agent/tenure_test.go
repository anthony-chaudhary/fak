package agent

import (
	"testing"
)

// TestTenurePromoteThenDemote is the core acceptance test: a command that recurs past
// the threshold is PROMOTED to tenured with a rollup, and a tenured command that goes
// quiet past its TTL is DEMOTED back to young (its rollup dropped) on a Sweep.
func TestTenurePromoteThenDemote(t *testing.T) {
	const threshold = 3
	const ttl = int64(1000) // 1s quiet window
	tt := newTenureTable(threshold, ttl)

	// Invocations 1 and 2 are young — below threshold, no rollup.
	for i, now := range []int64{0, 100} {
		roll, tenured := tt.Record("/quality-score", now)
		if tenured {
			t.Fatalf("invocation %d: command tenured before threshold", i+1)
		}
		if !roll.IsZero() {
			t.Fatalf("invocation %d: non-zero rollup before promotion: %+v", i+1, roll)
		}
	}
	if got := tt.Recurrence("/quality-score"); got != 2 {
		t.Fatalf("recurrence after 2 invocations = %d, want 2", got)
	}

	// Invocation 3 crosses the threshold → PROMOTE.
	roll, tenured := tt.Record("/quality-score", 200)
	if !tenured {
		t.Fatalf("invocation 3 (== threshold) did not tenure the command")
	}
	if roll.IsZero() {
		t.Fatalf("promoted command has a zero rollup")
	}
	if roll.Command != "/quality-score" || roll.Recurrence != 3 {
		t.Fatalf("rollup = %+v, want command=/quality-score recurrence=3", roll)
	}
	if r, ok := tt.Rollup("/quality-score"); !ok || r.IsZero() {
		t.Fatalf("Rollup() after promotion = (%+v, %v), want a tenured rollup", r, ok)
	}

	// Sweep BEFORE the TTL elapses: still tenured (in its quiet window, not yet expired).
	if demoted := tt.Sweep(300); len(demoted) != 0 {
		t.Fatalf("sweep within TTL demoted %v, want none", demoted)
	}
	if _, ok := tt.Rollup("/quality-score"); !ok {
		t.Fatalf("command demoted within its TTL window")
	}

	// Sweep AFTER the TTL elapses with no further Record → DEMOTE back to young.
	// last Record was at t=200; ttl=1000, grace=0 → expires at t>=1200.
	demoted := tt.Sweep(1500)
	if len(demoted) != 1 || demoted[0] != "/quality-score" {
		t.Fatalf("cold sweep demoted %v, want [/quality-score]", demoted)
	}
	if r, ok := tt.Rollup("/quality-score"); ok || !r.IsZero() {
		t.Fatalf("demoted command still tenured: (%+v, %v)", r, ok)
	}
	// History is kept (NEVER deletes) — recurrence survives demotion.
	if got := tt.Recurrence("/quality-score"); got != 3 {
		t.Fatalf("recurrence after demotion = %d, want 3 (history kept, not deleted)", got)
	}
}

// TestTenureReviveOnHotKeepsTenured proves the Lifecycle revive-on-hot: a tenured
// command that keeps recurring keeps Touching its Lifecycle, so a Sweep after the TTL
// does NOT demote it — it only demotes commands that actually went quiet.
func TestTenureReviveOnHotKeepsTenured(t *testing.T) {
	tt := newTenureTable(2, 1000)
	tt.Record("/conflation-score", 0)
	if _, tenured := tt.Record("/conflation-score", 100); !tenured {
		t.Fatalf("command not tenured after crossing threshold 2")
	}
	// Keep recurring every 500ms (< ttl 1000) and sweep each time: must stay tenured.
	for _, now := range []int64{600, 1100, 1600, 2100} {
		tt.Record("/conflation-score", now)
		if demoted := tt.Sweep(now); len(demoted) != 0 {
			t.Fatalf("a still-recurring command was demoted at t=%d: %v", now, demoted)
		}
		if _, ok := tt.Rollup("/conflation-score"); !ok {
			t.Fatalf("a still-recurring command lost tenure at t=%d", now)
		}
	}
}

func TestTenureSweepDemotionsAreSorted(t *testing.T) {
	tt := newTenureTable(1, 1000)
	tt.Record("/zeta", 0)
	tt.Record("/alpha", 0)

	demoted := tt.Sweep(2000)
	want := []string{"/alpha", "/zeta"}
	if len(demoted) != len(want) {
		t.Fatalf("demoted = %v, want %v", demoted, want)
	}
	for i := range want {
		if demoted[i] != want[i] {
			t.Fatalf("demoted = %v, want stable sorted order %v", demoted, want)
		}
	}
}

// TestTenureRollupDistinctFromPerTurnContext proves the rollup is a DISTINCT, compact
// representation — not the raw per-turn derived context. The rollup's digest is the
// derived-once form keyed by command identity, independent of any turn's transcript.
func TestTenureRollupDistinctFromPerTurnContext(t *testing.T) {
	tt := newTenureTable(1, 1000) // threshold 1: tenures on first invocation
	roll, tenured := tt.Record("/repo-hygiene", 0)
	if !tenured {
		t.Fatalf("threshold-1 command did not tenure on first invocation")
	}
	// The rollup is the compact promoted form: command identity + a digest, NOT the
	// full per-turn context. It carries no raw transcript bytes.
	if roll.Command != "/repo-hygiene" {
		t.Fatalf("rollup command = %q, want /repo-hygiene", roll.Command)
	}
	if len(roll.Digest) == 0 {
		t.Fatalf("rollup digest empty; expected a compact derived-once form")
	}
	if string(roll.Digest) != "tenured:/repo-hygiene" {
		t.Fatalf("rollup digest = %q, want a command-keyed digest distinct from per-turn context", roll.Digest)
	}
	// Distinctness witness: the rollup is keyed by command, so two different commands
	// produce two different digests — it is not a single shared per-turn blob.
	other, _ := tt.Record("/industry-score", 0)
	if string(other.Digest) == string(roll.Digest) {
		t.Fatalf("two commands shared a digest; rollup is not command-distinct")
	}
}

// TestTenureDefaultOffNilSafe proves the default-off path: a nil *tenureTable (the
// SessionPlanner default) records nothing, tenures nothing, and never panics — the
// behavior-preserving guarantee.
func TestTenureDefaultOffNilSafe(t *testing.T) {
	var tt *tenureTable // nil — the default-off table
	if roll, tenured := tt.Record("/x", 0); tenured || !roll.IsZero() {
		t.Fatalf("nil table tenured a command: (%+v, %v)", roll, tenured)
	}
	if roll, ok := tt.Rollup("/x"); ok || !roll.IsZero() {
		t.Fatalf("nil table returned a rollup: (%+v, %v)", roll, ok)
	}
	if demoted := tt.Sweep(0); demoted != nil {
		t.Fatalf("nil table sweep returned %v, want nil", demoted)
	}
	if got := tt.Recurrence("/x"); got != 0 {
		t.Fatalf("nil table recurrence = %d, want 0", got)
	}
}

// TestSessionPlannerTenuringWiring proves the SessionPlanner wiring is opt-in and
// default-off: a fresh planner does not tenure; after EnableTenuring it does.
func TestSessionPlannerTenuringWiring(t *testing.T) {
	sp := NewSessionPlanner(0)
	// Default-off: RecordCommand is a no-op (tenure table is nil).
	if _, tenured := sp.RecordCommand("/quality-score", 0); tenured {
		t.Fatalf("default SessionPlanner tenured a command without EnableTenuring")
	}
	if _, ok := sp.CommandRollup("/quality-score"); ok {
		t.Fatalf("default SessionPlanner returned a rollup without EnableTenuring")
	}

	// Enable, then recur to the threshold: it tenures.
	sp.EnableTenuring(2, 1000)
	sp.RecordCommand("/quality-score", 0)
	if _, tenured := sp.RecordCommand("/quality-score", 100); !tenured {
		t.Fatalf("SessionPlanner did not tenure after EnableTenuring + threshold")
	}
	if _, ok := sp.CommandRollup("/quality-score"); !ok {
		t.Fatalf("SessionPlanner.CommandRollup did not return the tenured rollup")
	}
	// Cold sweep demotes it back to young.
	if demoted := sp.SweepTenure(2000); len(demoted) != 1 {
		t.Fatalf("SessionPlanner.SweepTenure cold demote = %v, want one command", demoted)
	}
}

// TestSessionPlannerPinsUnchangedByTenuring proves tenuring does NOT perturb the
// resident-span pin set — object lifetime is orthogonal to placement. The pins() output
// is identical whether or not tenuring is enabled and recording.
func TestSessionPlannerPinsUnchangedByTenuring(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "first"},
		{Role: RoleUser, Content: "last"},
	}

	base := NewSessionPlanner(0)
	base.ingest(msgs)
	want := base.pins()

	withTenure := NewSessionPlanner(0)
	withTenure.EnableTenuring(2, 1000)
	withTenure.ingest(msgs)
	withTenure.RecordCommand("/quality-score", 0)
	withTenure.RecordCommand("/quality-score", 100) // promote
	got := withTenure.pins()

	if len(got) != len(want) {
		t.Fatalf("pins differ in length: got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("pin %d differs: got %q want %q (tenuring perturbed the resident plan)", i, got[i], want[i])
		}
	}
}
