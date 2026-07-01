package session

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// descriptor_test.go — the pure-core unit tests for the durable descriptor index
// (issue #1197). They exercise the three moves over the in-memory store with an
// injected clock: register-on-start (+ idempotent re-register), update-on-transition,
// TTL-GC (stale reaped, fresh survives, a live lease is never reaped), and the
// restart re-attach (a persisted descriptor re-attaches a session into a fresh Table
// at its REAL drive state, not DefaultState's Running/unbounded default).

// fixedClock returns a base time tests advance by hand, so register/update/GC are
// asserted to an exact sequence with no real time.
func fixedClock() time.Time {
	return time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC)
}

func TestRegistryRegisterOnStart(t *testing.T) {
	store := NewMemStore()
	r := NewRegistry(store)
	now := fixedClock()

	st := State{
		TraceID:  "trace-a",
		Run:      Running,
		Budget:   Budget{TurnsLeft: 7, TokensLeft: 500},
		Priority: 3,
	}
	meta := DescriptorMeta{
		PID:      4242,
		Argv:     []string{"claude", "--continue"},
		StartSHA: "0123456789abcdef",
		CacheKey: "cache-key-1",
	}
	d, err := r.RegisterWithMeta("sess-1", "host-a", st, time.Minute, now, meta)
	if err != nil {
		t.Fatalf("Register: %v", err)
	}
	if d.ID != "sess-1" || d.Host != "host-a" || d.Trace != "trace-a" {
		t.Fatalf("descriptor identity wrong: %+v", d)
	}
	if d.Run != Running || d.PCBState != "RUNNING" || d.Priority != 3 || d.Budget.TurnsLeft != 7 {
		t.Fatalf("descriptor did not project drive state: %+v", d)
	}
	if d.PID != 4242 || d.StartSHA != "0123456789abcdef" || d.CacheKey != "cache-key-1" || len(d.Argv) != 2 || d.Argv[0] != "claude" {
		t.Fatalf("descriptor metadata not stamped: %+v", d)
	}
	if !d.CreatedAt.Equal(now) || !d.LastSeen.Equal(now) {
		t.Fatalf("timestamps not stamped at now: %+v", d)
	}

	// It is actually persisted: the store has exactly one row, addressable by id.
	got, ok, err := r.Get("sess-1")
	if err != nil || !ok {
		t.Fatalf("Get after Register: ok=%v err=%v", ok, err)
	}
	if got.Trace != "trace-a" {
		t.Fatalf("persisted descriptor wrong: %+v", got)
	}
	all, _ := store.List()
	if len(all) != 1 {
		t.Fatalf("expected exactly one persisted descriptor, got %d", len(all))
	}
}

func TestRegistryRegisterIsIdempotent(t *testing.T) {
	store := NewMemStore()
	r := NewRegistry(store)
	t0 := fixedClock()

	st := State{TraceID: "trace-a", Run: Running, Priority: 1}
	if _, err := r.Register("sess-1", "host-a", st, time.Minute, t0); err != nil {
		t.Fatalf("first Register: %v", err)
	}

	// A relaunch re-registers the SAME id, with a blank host, later. It must not
	// duplicate, must preserve the original CreatedAt, and must keep the known host.
	t1 := t0.Add(5 * time.Minute)
	st2 := State{TraceID: "trace-a2", Run: Throttled, Priority: 2}
	d, err := r.Register("sess-1", "", st2, time.Minute, t1)
	if err != nil {
		t.Fatalf("re-Register: %v", err)
	}
	all, _ := store.List()
	if len(all) != 1 {
		t.Fatalf("re-register duplicated the row: %d rows", len(all))
	}
	if !d.CreatedAt.Equal(t0) {
		t.Fatalf("re-register lost original CreatedAt: got %v want %v", d.CreatedAt, t0)
	}
	if !d.LastSeen.Equal(t1) {
		t.Fatalf("re-register did not advance LastSeen: %v", d.LastSeen)
	}
	if d.Host != "host-a" {
		t.Fatalf("blank-host relaunch erased the known host: %q", d.Host)
	}
	if d.Run != Throttled || d.Trace != "trace-a2" {
		t.Fatalf("re-register did not re-project drive state: %+v", d)
	}
}

func TestRegistryUpdateOnTransition(t *testing.T) {
	store := NewMemStore()
	r := NewRegistry(store)
	t0 := fixedClock()

	st := State{TraceID: "trace-a", Run: Running, Budget: Budget{TurnsLeft: 2, TokensLeft: Unbounded}, Rev: 1}
	if _, err := r.Register("sess-1", "host-a", st, time.Minute, t0); err != nil {
		t.Fatalf("Register: %v", err)
	}

	// Drive the live table through a real transition, then mirror it into the index.
	tbl := NewTable()
	tbl.Restore("trace-a", st)
	out, ok := tbl.Transition("trace-a", Paused, "operator hold")
	if !ok {
		t.Fatalf("Transition to Paused refused")
	}

	t1 := t0.Add(time.Minute)
	d, err := r.Update("sess-1", out, t1)
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
	if d.Run != Paused {
		t.Fatalf("descriptor did not track the live PCB transition: run=%v", d.Run)
	}
	if d.PCBState != "PAUSED" {
		t.Fatalf("descriptor pcb_state = %q, want PAUSED", d.PCBState)
	}
	if d.Reason != "operator hold" {
		t.Fatalf("descriptor did not carry the transition reason: %q", d.Reason)
	}
	if d.Rev != out.Rev {
		t.Fatalf("descriptor rev %d != live rev %d", d.Rev, out.Rev)
	}
	if !d.UpdatedAt.Equal(t1) || !d.LastSeen.Equal(t1) {
		t.Fatalf("Update did not advance timestamps: %+v", d)
	}
	// CreatedAt is preserved across the update (the row is the same session).
	if !d.CreatedAt.Equal(t0) {
		t.Fatalf("Update clobbered CreatedAt: got %v want %v", d.CreatedAt, t0)
	}
}

func TestRegistryUpdatePreservesMetadata(t *testing.T) {
	store := NewMemStore()
	r := NewRegistry(store)
	t0 := fixedClock()
	meta := DescriptorMeta{PID: 77, Argv: []string{"codex", "exec"}, StartSHA: "abc", CacheKey: "ck"}

	if _, err := r.RegisterWithMeta("sess-meta", "host-a", State{TraceID: "trace-a", Run: Running}, time.Minute, t0, meta); err != nil {
		t.Fatalf("RegisterWithMeta: %v", err)
	}
	d, err := r.Update("sess-meta", State{TraceID: "trace-a", Run: Draining, Reason: ReasonBudgetContext}, t0.Add(time.Second))
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
	if d.PID != 77 || d.StartSHA != "abc" || d.CacheKey != "ck" || len(d.Argv) != 2 || d.Argv[1] != "exec" {
		t.Fatalf("Update lost descriptor metadata: %+v", d)
	}
	if d.PCBState != "DRAINING" {
		t.Fatalf("Update pcb_state = %q, want DRAINING", d.PCBState)
	}
}

func TestRegistryTTLGarbageCollects(t *testing.T) {
	store := NewMemStore()
	r := NewRegistry(store)
	t0 := fixedClock()

	// Two sessions: "stale" with a 1-minute TTL, "live" with a 1-hour TTL.
	stale := State{TraceID: "trace-stale", Run: Running}
	live := State{TraceID: "trace-live", Run: Running}
	if _, err := r.Register("sess-stale", "h", stale, time.Minute, t0); err != nil {
		t.Fatalf("register stale: %v", err)
	}
	if _, err := r.Register("sess-live", "h", live, time.Hour, t0); err != nil {
		t.Fatalf("register live: %v", err)
	}

	// Advance 90s: the 1-minute session is past its TTL; the 1-hour one is not.
	t1 := t0.Add(90 * time.Second)
	got, err := r.List(t1)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 1 || got[0].ID != "sess-live" {
		t.Fatalf("TTL sweep did not reap the stale descriptor and keep the live one: %+v", got)
	}
	// The reap is durable — the stale row is gone from the store, not just filtered.
	if _, ok, _ := r.Get("sess-stale"); ok {
		t.Fatalf("stale descriptor was filtered but not GC'd from the store")
	}

	// A live lease re-stamped just before the deadline is NEVER reaped. Re-stamp the
	// live session at t1, then sweep one hour later: still within its fresh window.
	if _, err := r.Update("sess-live", live, t1); err != nil {
		t.Fatalf("re-stamp live: %v", err)
	}
	t2 := t1.Add(59 * time.Minute) // < the 1h TTL from t1
	got, err = r.List(t2)
	if err != nil {
		t.Fatalf("List after re-stamp: %v", err)
	}
	if len(got) != 1 || got[0].ID != "sess-live" {
		t.Fatalf("re-stamped live lease was wrongly reaped: %+v", got)
	}

	// Now let the live one go stale too and confirm it is reaped.
	t3 := t2.Add(2 * time.Hour)
	reaped, err := r.GC(t3)
	if err != nil {
		t.Fatalf("GC: %v", err)
	}
	if reaped != 1 {
		t.Fatalf("expected to reap the now-stale live session, reaped=%d", reaped)
	}
	if remaining, _ := store.List(); len(remaining) != 0 {
		t.Fatalf("index not empty after final GC: %d rows", len(remaining))
	}
}

func TestRegistryRestartReattachesAtPersistedState(t *testing.T) {
	// Process A: a session runs, is throttled with a cut budget and a priority bump,
	// and its descriptor is persisted to a (shared, durable) store.
	store := NewMemStore()
	r1 := NewRegistry(store)
	t0 := fixedClock()

	live := NewTable()
	live.Restore("trace-x", State{TraceID: "trace-x", Run: Running, Budget: Budget{TurnsLeft: 100, TokensLeft: 100}})
	live.SetPriority("trace-x", 9)
	live.SetBudget("trace-x", Budget{TurnsLeft: 4, TokensLeft: 40})
	st, _ := live.Transition("trace-x", Throttled, "slow lane")
	if _, err := r1.Register("sess-x", "host-a", st, time.Hour, t0); err != nil {
		t.Fatalf("register: %v", err)
	}

	// --- process restart: a brand-new Registry + Table over the SAME durable store,
	// and a fresh in-memory drive table that has never seen trace-x. Without the
	// descriptor it would re-attach at DefaultState (Running, unbounded). ---
	r2 := NewRegistry(store)
	t1 := t0.Add(time.Minute)
	descs, err := r2.List(t1)
	if err != nil {
		t.Fatalf("List on restart: %v", err)
	}
	if len(descs) != 1 {
		t.Fatalf("restart did not read back the descriptor: %d rows", len(descs))
	}
	d := descs[0]

	restarted := NewTable()
	// Prove the default would be wrong, so the test is not vacuous: an unseen trace
	// reads Running/unbounded, NOT the persisted Throttled/cut-budget state.
	def := restarted.Get(d.Trace)
	if def.Run == Throttled || def.Priority == 9 {
		t.Fatalf("precondition: fresh table already had the state; test would be vacuous")
	}

	// Re-attach from the descriptor.
	got := restarted.Restore(d.Trace, d.RestoredState())
	if got.Run != Throttled {
		t.Fatalf("restart re-attached at default run-state, not persisted: %v", got.Run)
	}
	if got.Priority != 9 {
		t.Fatalf("restart lost the persisted priority: %d", got.Priority)
	}
	if got.Budget.TurnsLeft != 4 || got.Budget.TokensLeft != 40 {
		t.Fatalf("restart lost the persisted budget: %+v", got.Budget)
	}
	if got.Reason != "slow lane" {
		t.Fatalf("restart lost the persisted reason: %q", got.Reason)
	}
	// Restore preserves Rev (it does not bump) — the descriptor's Rev round-trips.
	if got.Rev != d.Rev {
		t.Fatalf("restart did not preserve Rev: got %d want %d", got.Rev, d.Rev)
	}

	// And the re-attached live table now Decides from the REAL state: a Throttled
	// session with 4 turns left proceeds (not a defaulted unbounded Running one that
	// would also proceed but with no cut budget). Burn the budget down and confirm it
	// drains at the persisted allotment, proving the restart inherited the real cap.
	for i := 0; i < 4; i++ {
		if v := restarted.Decide(d.Trace); !v.Proceed {
			t.Fatalf("turn %d: persisted-budget session should still proceed, got stop %q", i, v.Reason)
		}
	}
	v := restarted.Decide(d.Trace)
	if v.Proceed {
		t.Fatalf("session did not drain at its persisted 4-turn budget; restart used defaults")
	}
	if v.Reason != ReasonBudgetTurns {
		t.Fatalf("drain reason wrong: %q", v.Reason)
	}
}

func TestRegistryRestoresCacheAffinityDecision(t *testing.T) {
	store := NewMemStore()
	r := NewRegistry(store)
	t0 := fixedClock()

	live := NewTable()
	live.SetBudget("trace-affinity", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 10})
	st := live.DebitUsage("trace-affinity", Usage{ContextTokens: 11})
	if st.CacheAffinity.IsZero() {
		t.Fatalf("setup produced no cache affinity decision: %+v", st)
	}
	if _, err := r.Register("sess-affinity", "host-a", st, time.Hour, t0); err != nil {
		t.Fatalf("register: %v", err)
	}

	d, ok, err := r.Get("sess-affinity")
	if err != nil || !ok {
		t.Fatalf("get descriptor: ok=%v err=%v", ok, err)
	}
	restored := d.RestoredState()
	if restored.CacheAffinity != st.CacheAffinity {
		t.Fatalf("restored cache affinity = %+v, want %+v", restored.CacheAffinity, st.CacheAffinity)
	}
}

// TestRegistryRestartReattachesObjectivePin is the #1589 process-restart witness for
// the managed-context objective pin (#1583): a session that pinned its standing
// objective, then crossed a hidden process restart (the same registry-restart shape
// TestRegistryRestartReattachesAtPersistedState proves for run-state/budget/priority),
// must re-attach carrying the SAME PinID and Digest — not a dropped/reset pin — so the
// migrated session can keep reconciling against the original objective identity.
func TestRegistryRestartReattachesObjectivePin(t *testing.T) {
	store := NewMemStore()
	r1 := NewRegistry(store)
	t0 := fixedClock()

	pin := ctxplan.NewObjectivePin("pin-migrate", "land the session-migration continuity fix", 1)
	live := NewTable()
	live.Restore("trace-obj", State{TraceID: "trace-obj", Run: Running, Budget: Budget{TurnsLeft: 10, TokensLeft: 10}, ObjectivePin: pin})
	if _, err := r1.Register("sess-obj", "host-a", live.Get("trace-obj"), time.Hour, t0); err != nil {
		t.Fatalf("register: %v", err)
	}

	// --- process restart: a brand-new Registry + Table over the SAME durable store. ---
	r2 := NewRegistry(store)
	t1 := t0.Add(time.Minute)
	descs, err := r2.List(t1)
	if err != nil {
		t.Fatalf("list on restart: %v", err)
	}
	if len(descs) != 1 {
		t.Fatalf("restart did not read back the descriptor: %d rows", len(descs))
	}
	d := descs[0]

	restarted := NewTable()
	// Precondition: an unseen trace has no pin, so the assertion below is not vacuous.
	if !restarted.Get(d.Trace).ObjectivePin.IsZero() {
		t.Fatalf("precondition: fresh table already carried the pin; test would be vacuous")
	}
	got := restarted.Restore(d.Trace, d.RestoredState())
	if got.ObjectivePin != pin {
		t.Fatalf("restart lost the pinned objective: got %+v want %+v", got.ObjectivePin, pin)
	}
	if got.ObjectivePin.PinID != "pin-migrate" || !got.ObjectivePin.Verify() {
		t.Fatalf("restart re-attached an invalid/mismatched pin: %+v", got.ObjectivePin)
	}
}

func TestMemStoreRejectsBlankID(t *testing.T) {
	s := NewMemStore()
	if err := s.Put(Descriptor{ID: ""}); err == nil {
		t.Fatalf("MemStore.Put accepted a blank id")
	}
	r := NewRegistry(s)
	if _, err := r.Register("", "h", State{TraceID: "t"}, time.Minute, fixedClock()); err == nil {
		t.Fatalf("Register accepted a blank id")
	}
}

func TestFileStorePersistsAcrossInstances(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session-registry.json")
	t0 := fixedClock()
	store1 := NewFileStore(path)
	r1 := NewRegistry(store1)
	if _, err := r1.Register("sess-file", "host-a", State{
		TraceID: "trace-file",
		Run:     Running,
		Budget:  Budget{TurnsLeft: 2, TokensLeft: 200},
		Rev:     3,
	}, time.Hour, t0); err != nil {
		t.Fatalf("Register file-backed descriptor: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("file store did not create registry file: %v", err)
	}

	store2 := NewFileStore(path)
	r2 := NewRegistry(store2)
	got, ok, err := r2.Get("sess-file")
	if err != nil || !ok {
		t.Fatalf("Get from second store: ok=%v err=%v", ok, err)
	}
	if got.Trace != "trace-file" || got.Budget.TokensLeft != 200 || got.Rev != 3 {
		t.Fatalf("descriptor did not persist across store instances: %+v", got)
	}

	if err := store2.Delete("sess-file"); err != nil {
		t.Fatalf("Delete file-backed descriptor: %v", err)
	}
	if _, ok, err := NewRegistry(NewFileStore(path)).Get("sess-file"); err != nil || ok {
		t.Fatalf("deleted descriptor reappeared: ok=%v err=%v", ok, err)
	}
}

func TestRegistryNilReceiverIsInert(t *testing.T) {
	var r *Registry
	now := fixedClock()
	// A nil registry projects without persisting and never panics — the no-op shell.
	d, err := r.Register("sess", "h", State{TraceID: "t", Run: Paused}, time.Minute, now)
	if err != nil {
		t.Fatalf("nil Register: %v", err)
	}
	if d.Run != Paused {
		t.Fatalf("nil Register did not project state: %+v", d)
	}
	if _, err := r.Update("sess", State{}, now); err != nil {
		t.Fatalf("nil Update: %v", err)
	}
	if got, err := r.List(now); err != nil || got != nil {
		t.Fatalf("nil List should be empty: got=%v err=%v", got, err)
	}
	if n, err := r.GC(now); err != nil || n != 0 {
		t.Fatalf("nil GC should reap nothing: n=%d err=%v", n, err)
	}
}
