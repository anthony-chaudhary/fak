package main

// session_control_test.go — exercises the cmd/fak closures that bind the gateway's
// /v1/fak/session control surface (#620) to a real internal/session.Table: the
// verb→table dispatch (applySessionControl), the optimistic-concurrency CAS path
// (if_rev), the terminal-refusal (ok=false), and the SessionState projection. The
// HTTP routing/validation is covered by internal/gateway/session_routes_test.go;
// this file proves the host wiring actually drives the table it owns.

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestApplySessionControlDispatchesEveryVerb proves each route verb lands on its
// matching Table write and the returned SessionState reflects the new drive.
func TestApplySessionControlDispatchesEveryVerb(t *testing.T) {
	tbl := session.NewTable()
	const trace = "drive-1"

	// run: throttle the session, carrying a reason.
	st, ok, err := applySessionControl(tbl, trace, "run", gateway.SessionControlRequest{
		Run: "throttled", Reason: "operator-slowdown",
	})
	if err != nil || !ok || st.Run != session.Throttled || st.Reason != "operator-slowdown" {
		t.Fatalf("run verb: st=%+v ok=%v err=%v", st, ok, err)
	}

	// budget: cut the turns allotment live.
	st, ok, err = applySessionControl(tbl, trace, "budget", gateway.SessionControlRequest{
		Budget: &gateway.SessionBudget{TurnsLeft: 4, TokensLeft: -1},
	})
	if err != nil || !ok || st.Budget.TurnsLeft != 4 || st.Budget.TokensLeft != -1 {
		t.Fatalf("budget verb: st=%+v ok=%v err=%v", st, ok, err)
	}

	// pace: tighten the per-turn cap.
	st, ok, err = applySessionControl(tbl, trace, "pace", gateway.SessionControlRequest{
		Pace: &gateway.SessionPace{MaxTokensPerTurn: 256, MinTurnGapMs: 100},
	})
	if err != nil || !ok || st.Pace.MaxTokensPerTurn != 256 || st.Pace.MinTurnGapMs != 100 {
		t.Fatalf("pace verb: st=%+v ok=%v err=%v", st, ok, err)
	}

	// priority: lower the rank so an urgent session passes.
	st, ok, err = applySessionControl(tbl, trace, "priority", gateway.SessionControlRequest{
		Priority: intPtr(3),
	})
	if err != nil || !ok || st.Priority != 3 {
		t.Fatalf("priority verb: st=%+v ok=%v err=%v", st, ok, err)
	}
	if st.Rev != 4 {
		t.Fatalf("expected Rev=4 after four writes, got %d", st.Rev)
	}

	// Unknown verb ⇒ error (the route maps this to 400).
	if _, _, err := applySessionControl(tbl, trace, "nope", gateway.SessionControlRequest{}); err == nil {
		t.Fatalf("unknown verb must return an error")
	}
	// Missing body field ⇒ error.
	if _, _, err := applySessionControl(tbl, trace, "budget", gateway.SessionControlRequest{}); err == nil {
		t.Fatalf("budget verb without a body must return an error")
	}
}

// TestApplySessionControlCAS proves if_rev is the optimistic-concurrency guard: a
// matching rev applies the write; a stale rev loses the race (ok=false).
func TestApplySessionControlCAS(t *testing.T) {
	tbl := session.NewTable()
	const trace = "cas-1"

	// Seed a budget at Rev 1.
	seed, _, _ := applySessionControl(tbl, trace, "budget", gateway.SessionControlRequest{
		Budget: &gateway.SessionBudget{TurnsLeft: 10},
	})
	if seed.Rev != 1 {
		t.Fatalf("seed Rev = %d, want 1", seed.Rev)
	}

	// A stale if_rev (0 is "no CAS"; use an obviously-wrong rev) loses the race.
	stale, ok, err := applySessionControl(tbl, trace, "budget", gateway.SessionControlRequest{
		Budget: &gateway.SessionBudget{TurnsLeft: 5}, IfRev: 999,
	})
	if err != nil || ok {
		t.Fatalf("stale CAS must refuse: st=%+v ok=%v err=%v", stale, ok, err)
	}

	// The matching if_rev applies and bumps the rev.
	good, ok, err := applySessionControl(tbl, trace, "budget", gateway.SessionControlRequest{
		Budget: &gateway.SessionBudget{TurnsLeft: 5}, IfRev: seed.Rev,
	})
	if err != nil || !ok || good.Budget.TurnsLeft != 5 || good.Rev != 2 {
		t.Fatalf("matching CAS must apply: st=%+v ok=%v err=%v", good, ok, err)
	}
}

// TestApplySessionControlTerminalRefused proves a stopped session rejects every
// control verb (ok=false) — you start a new session, you do not un-stop one.
func TestApplySessionControlTerminalRefused(t *testing.T) {
	tbl := session.NewTable()
	const trace = "term-1"

	if _, _, err := applySessionControl(tbl, trace, "run", gateway.SessionControlRequest{
		Run: "stopped", Reason: "operator-stop",
	}); err != nil {
		t.Fatalf("stop seed: %v", err)
	}
	// Every verb on the now-terminal session must refuse (ok=false, no error).
	for _, verb := range []string{"run", "budget", "pace", "priority"} {
		req := gateway.SessionControlRequest{
			Budget: &gateway.SessionBudget{TurnsLeft: 1}, Pace: &gateway.SessionPace{MaxTokensPerTurn: 1},
			Priority: intPtr(1), Run: "running",
		}
		if _, ok, err := applySessionControl(tbl, trace, verb, req); ok || err != nil {
			t.Fatalf("terminal session verb %q must refuse with ok=false,err=nil; got ok=%v err=%v", verb, ok, err)
		}
	}
}

// TestControlAndObserveRoundTrip proves the package-global closures wired into the
// gateway Config (observeSession/controlSession over serveSessions) are connected
// end to end: a control write is visible to the next observe read.
func TestControlAndObserveRoundTrip(t *testing.T) {
	const trace = "roundtrip-1"
	t.Cleanup(func() { serveSessions.Reset(trace) })

	if _, _, err := controlSession(context.Background(), trace, "run",
		gateway.SessionControlRequest{Run: "paused"}); err != nil {
		t.Fatalf("control pause: %v", err)
	}
	got := observeSession(context.Background(), trace)
	if got.Run != "paused" || got.TraceID != trace {
		t.Fatalf("observe after pause = %+v, want run=paused", got)
	}
	// An unseen trace reads its safe default (Running, unbounded), never a phantom.
	fresh := observeSession(context.Background(), "never-seen-"+trace)
	if fresh.Run != "running" || fresh.Budget.TurnsLeft != session.Unbounded {
		t.Fatalf("unseen trace = %+v, want running/unbounded default", fresh)
	}
}

// TestDecideAndDebitSessionHooks proves the served-request hot-path callbacks wired
// into gateway.Config use the same process-local session table as the operator
// control surface.
func TestDecideAndDebitSessionHooks(t *testing.T) {
	const trace = "serve-hook-1"
	t.Cleanup(func() { serveSessions.Reset(trace) })

	if _, _, err := controlSession(context.Background(), trace, "budget",
		gateway.SessionControlRequest{Budget: &gateway.SessionBudget{TurnsLeft: 1, TokensLeft: 10}}); err != nil {
		t.Fatalf("seed budget: %v", err)
	}
	v := decideSession(context.Background(), trace)
	if !v.Proceed || v.State.Budget.TurnsLeft != 0 {
		t.Fatalf("first decide = %+v, want proceed with turn debited to 0", v)
	}
	st := debitSession(context.Background(), trace, gateway.SessionUsage{CompletionTokens: 10})
	if st.Budget.TokensLeft != 0 {
		t.Fatalf("debit state = %+v, want token budget 0", st)
	}
	v = decideSession(context.Background(), trace)
	if v.Proceed || !v.Stop {
		t.Fatalf("post-budget decide = %+v, want stop after token exhaustion", v)
	}
}

func TestDecideAndDebitPersistDurableDescriptor(t *testing.T) {
	const trace = "durable-decide-debit"
	ctx := context.Background()
	now := time.Date(2026, 6, 29, 13, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "registry.json")
	installDurableSessionTest(t, path, &now, &fakeSessionLeasePublisher{}, session.DescriptorMeta{
		PID:      12345,
		Argv:     []string{"claude", "--continue"},
		StartSHA: "abc123",
		CacheKey: "cache-key",
	})

	serveSessions.Restore(trace, session.State{
		TraceID: trace,
		Run:     session.Running,
		Budget: session.Budget{
			TurnsLeft:         1,
			TokensLeft:        session.Unbounded,
			ContextTokensLeft: 20,
		},
	})
	if err := registerServeSessionDurability(ctx, trace); err != nil {
		t.Fatalf("register durable session: %v", err)
	}

	now = now.Add(time.Second)
	v := decideSession(ctx, trace)
	if !v.Proceed || v.State.Budget.TurnsLeft != 0 {
		t.Fatalf("decide = %+v, want proceed with debited turn budget", v)
	}
	decideDesc := readSessionDescriptor(t, path, trace)
	if decideDesc.Budget.TurnsLeft != 0 || decideDesc.PCBState != "RUNNING" {
		t.Fatalf("descriptor after Decide = %+v, want running with turns_left=0", decideDesc)
	}
	if !decideDesc.UpdatedAt.Equal(now) || decideDesc.PID != 12345 || decideDesc.CacheKey != "cache-key" || len(decideDesc.Argv) != 2 {
		t.Fatalf("descriptor after Decide lost metadata/timestamp: %+v", decideDesc)
	}

	now = now.Add(time.Second)
	st := debitSession(ctx, trace, gateway.SessionUsage{ContextTokens: 21})
	if st.Run != "draining" || st.Reason != session.ReasonBudgetContext || st.ContinuationID == "" {
		t.Fatalf("debit = %+v, want draining with continuation", st)
	}
	debitDesc := readSessionDescriptor(t, path, trace)
	if debitDesc.Run != session.Draining || debitDesc.PCBState != "DRAINING" || debitDesc.Reason != session.ReasonBudgetContext {
		t.Fatalf("descriptor after DebitUsage = %+v, want draining budget context", debitDesc)
	}
	if !debitDesc.UpdatedAt.Equal(now) {
		t.Fatalf("descriptor UpdatedAt after DebitUsage = %v, want %v", debitDesc.UpdatedAt, now)
	}
}

func TestDefaultSessionIDDerivedFromCacheKey(t *testing.T) {
	key := sessionCacheKey("host-a", `C:\work\fak`, "deadbeef", []string{"claude", "--continue"})
	id := defaultSessionIDFromMeta(session.DescriptorMeta{CacheKey: key})
	if id != "guard-"+key[:16] {
		t.Fatalf("derived session id = %q, want guard-%s", id, key[:16])
	}
	if again := defaultSessionIDFromMeta(session.DescriptorMeta{CacheKey: key}); again != id {
		t.Fatalf("derived session id not stable: %q then %q", id, again)
	}
}

func TestDebitSessionHookDebitsContextBudget(t *testing.T) {
	const trace = "serve-hook-context-1"
	t.Cleanup(func() { serveSessions.Reset(trace) })

	if _, _, err := controlSession(context.Background(), trace, "budget",
		gateway.SessionControlRequest{Budget: &gateway.SessionBudget{
			TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 20,
		}}); err != nil {
		t.Fatalf("seed context budget: %v", err)
	}
	st := debitSession(context.Background(), trace, gateway.SessionUsage{ContextTokens: 21})
	if st.Run != "draining" || st.Reason != session.ReasonBudgetContext || st.ContinuationID == "" {
		t.Fatalf("context debit state = %+v, want draining with continuation id", st)
	}
}

func TestResetServedSessionOnBudgetRecontinuesWithCarryover(t *testing.T) {
	const trace = "reset-hook-1"
	var child string
	t.Cleanup(func() {
		serveSessions.Reset(trace)
		if child != "" {
			serveSessions.Reset(child)
		}
	})

	if _, _, err := controlSession(context.Background(), trace, "budget",
		gateway.SessionControlRequest{Budget: &gateway.SessionBudget{
			TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 5,
		}}); err != nil {
		t.Fatalf("seed context budget: %v", err)
	}
	st := debitSession(context.Background(), trace, gateway.SessionUsage{ContextTokens: 6})
	child = st.ContinuationID
	if child == "" {
		t.Fatalf("context debit state = %+v, want continuation id", st)
	}

	hook := resetServedSessionOnBudget(50)
	if hook == nil {
		t.Fatalf("reset hook must be enabled with a positive fresh context budget")
	}
	nextTrace, seed, ok := hook(context.Background(), trace, []agent.Message{
		{Role: agent.RoleSystem, Content: "You are fak."},
		{Role: agent.RoleUser, Content: "Help me add reset."},
		{Role: agent.RoleAssistant, Content: "I will wire the served reset hook."},
		{Role: agent.RoleUser, Content: "I prefer concise answers."},
	})
	if !ok || nextTrace != child || len(seed) != 1 {
		t.Fatalf("reset hook = trace=%q seed=%+v ok=%v, want child trace with one carryover message", nextTrace, seed, ok)
	}
	if seed[0].Role != agent.RoleSystem || !strings.Contains(strings.ToLower(seed[0].Content), "continuation") {
		t.Fatalf("seed message = %+v, want system continuation recap", seed[0])
	}

	fresh := observeSession(context.Background(), child)
	if fresh.Run != "running" || fresh.ParentTrace != trace || fresh.Generation != 1 || fresh.Budget.ContextTokensLeft != 50 {
		t.Fatalf("fresh child state = %+v, want running child with parent/generation/context budget", fresh)
	}
	tx := fresh.ResetTransaction
	if tx.Schema != session.ResetTransactionSchema || tx.OldTrace != trace || tx.NewTrace != child {
		t.Fatalf("reset transaction lineage = %+v, want schema and %s -> %s", tx, trace, child)
	}
	if tx.SeedDigest == "" || len(tx.Contributors) == 0 || len(tx.OmittedSpans) == 0 {
		t.Fatalf("reset transaction missing replay proof fields: %+v", tx)
	}
	if tx.BudgetRearm.ContextTokensLeft != 50 || tx.BudgetRearm.ContextTokensCap != 50 {
		t.Fatalf("reset transaction budget rearm = %+v, want 50/50", tx.BudgetRearm)
	}
}

func TestSessionPauseResumeDurableWriteThrough(t *testing.T) {
	const trace = "durable-pause-resume"
	ctx := context.Background()
	now := time.Date(2026, 6, 29, 10, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "registry.json")
	leases := &fakeSessionLeasePublisher{}
	installDurableSessionTest(t, path, &now, leases)

	serveSessions.Restore(trace, session.State{
		TraceID:  trace,
		Run:      session.Running,
		Budget:   session.Budget{TurnsLeft: 3, TokensLeft: 99, ContextTokensLeft: 500, ContextTokensCap: 500},
		Priority: 7,
		Pace:     session.Pace{MaxTokensPerTurn: 256, MinTurnGapMs: 25},
		Rev:      4,
	})
	if err := registerServeSessionDurability(ctx, trace); err != nil {
		t.Fatalf("register durable session: %v", err)
	}

	now = now.Add(time.Second)
	paused, ok, err := controlSession(ctx, trace, "run", gateway.SessionControlRequest{
		Run: "paused", Reason: "operator-hold",
	})
	if err != nil || !ok {
		t.Fatalf("pause control: state=%+v ok=%v err=%v", paused, ok, err)
	}
	if paused.Run != "paused" || paused.Reason != "operator-hold" {
		t.Fatalf("paused state = %+v, want paused with reason", paused)
	}
	pausedDesc := readSessionDescriptor(t, path, trace)
	if pausedDesc.Run != session.Paused || pausedDesc.Reason != "operator-hold" {
		t.Fatalf("descriptor after pause = %+v, want paused with reason", pausedDesc)
	}
	if pausedDesc.Budget.TurnsLeft != 3 || pausedDesc.Budget.TokensLeft != 99 || pausedDesc.Priority != 7 {
		t.Fatalf("descriptor lost drive axes after pause: %+v", pausedDesc)
	}
	if last := leases.lastPublished(t); last.PCBState != "PAUSED" {
		t.Fatalf("side-ref publish after pause = %+v, want PAUSED", last)
	}

	// Simulate a fresh process table over the same registry: ls/status read the
	// persisted row, then resume restores it and flips it back to RUNNING.
	serveSessions = session.NewTable()
	row, ok := findGatewaySession(listSessions(ctx), trace)
	if !ok || row.Run != "paused" {
		t.Fatalf("list after restart = %+v ok=%v, want persisted paused row", row, ok)
	}
	if got := observeSession(ctx, trace); got.Run != "paused" {
		t.Fatalf("status after restart = %+v, want persisted paused", got)
	}

	now = now.Add(time.Second)
	resumed, ok, err := controlSession(ctx, trace, "run", gateway.SessionControlRequest{Run: "running"})
	if err != nil || !ok {
		t.Fatalf("resume control: state=%+v ok=%v err=%v", resumed, ok, err)
	}
	if resumed.Run != "running" || resumed.Reason != "" {
		t.Fatalf("resumed state = %+v, want running with cleared reason", resumed)
	}
	if resumed.Budget.TurnsLeft != 3 || resumed.Budget.TokensLeft != 99 || resumed.Priority != 7 || resumed.Pace.MaxTokensPerTurn != 256 {
		t.Fatalf("resume lost drive axes: %+v", resumed)
	}
	resumedDesc := readSessionDescriptor(t, path, trace)
	if resumedDesc.Run != session.Running || resumedDesc.Budget.ContextTokensLeft != 500 {
		t.Fatalf("descriptor after resume = %+v, want running with context budget", resumedDesc)
	}
	if last := leases.lastPublished(t); last.PCBState != "RUNNING" {
		t.Fatalf("side-ref publish after resume = %+v, want RUNNING", last)
	}
}

func TestSessionStopDurableWriteThroughReleasesSideRef(t *testing.T) {
	const trace = "durable-stop"
	ctx := context.Background()
	now := time.Date(2026, 6, 29, 11, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "registry.json")
	leases := &fakeSessionLeasePublisher{}
	installDurableSessionTest(t, path, &now, leases)

	serveSessions.Restore(trace, session.State{
		TraceID: trace,
		Run:     session.Running,
		Budget:  session.Budget{TurnsLeft: 2, TokensLeft: 20},
		Rev:     2,
	})
	if err := registerServeSessionDurability(ctx, trace); err != nil {
		t.Fatalf("register durable session: %v", err)
	}

	now = now.Add(time.Second)
	stopped, ok, err := controlSession(ctx, trace, "run", gateway.SessionControlRequest{
		Run: "stopped", Reason: session.ReasonStopped,
	})
	if err != nil || !ok {
		t.Fatalf("stop control: state=%+v ok=%v err=%v", stopped, ok, err)
	}
	if stopped.Run != "stopped" || stopped.Reason != session.ReasonStopped {
		t.Fatalf("stopped state = %+v, want stopped with reason", stopped)
	}
	desc := readSessionDescriptor(t, path, trace)
	if desc.Run != session.Stopped || desc.Reason != session.ReasonStopped {
		t.Fatalf("descriptor after stop = %+v, want stopped with reason", desc)
	}
	if len(leases.removed) != 1 || leases.removed[0] != trace {
		t.Fatalf("side-ref removes = %+v, want [%s]", leases.removed, trace)
	}

	restarted := session.NewTable()
	mirror := newSessionDurability(session.NewRegistry(session.NewFileStore(path)), leases, "test-host", time.Hour, func() time.Time { return now }, nil)
	if err := mirror.restore(restarted); err != nil {
		t.Fatalf("restore after stop: %v", err)
	}
	if got := restarted.Get(trace); got.Run != session.Stopped || got.Reason != session.ReasonStopped {
		t.Fatalf("restart restored = %+v, want stopped with reason", got)
	}
}

func TestSessionStopUnknownIDTypedError(t *testing.T) {
	ctx := context.Background()
	now := time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)
	installDurableSessionTest(t, filepath.Join(t.TempDir(), "registry.json"), &now, &fakeSessionLeasePublisher{})

	st, ok, err := controlSession(ctx, "missing-session", "run", gateway.SessionControlRequest{
		Run: "stopped", Reason: session.ReasonStopped,
	})
	if err == nil || ok {
		t.Fatalf("unknown stop must fail with typed error: state=%+v ok=%v err=%v", st, ok, err)
	}
	var unknown unknownSessionError
	if !errors.As(err, &unknown) || unknown.id != "missing-session" {
		t.Fatalf("unknown stop error = %T %v, want unknownSessionError for missing-session", err, err)
	}
}

type fakeSessionLeasePublisher struct {
	published []leaseref.SessionDescriptor
	removed   []string
}

func (f *fakeSessionLeasePublisher) PublishSession(_ context.Context, d leaseref.SessionDescriptor) (string, error) {
	f.published = append(f.published, d)
	return d.Ref(), nil
}

func (f *fakeSessionLeasePublisher) RemoveSession(_ context.Context, id string) error {
	f.removed = append(f.removed, id)
	return nil
}

func (f *fakeSessionLeasePublisher) lastPublished(t *testing.T) leaseref.SessionDescriptor {
	t.Helper()
	if len(f.published) == 0 {
		t.Fatalf("expected at least one side-ref publish")
	}
	return f.published[len(f.published)-1]
}

func installDurableSessionTest(t *testing.T, path string, now *time.Time, leases *fakeSessionLeasePublisher, meta ...session.DescriptorMeta) {
	t.Helper()
	oldSessions := serveSessions
	oldDurability := serveSessionDurability
	serveSessions = session.NewTable()
	serveSessionDurability = newSessionDurability(
		session.NewRegistry(session.NewFileStore(path)),
		leases,
		"test-host",
		time.Hour,
		func() time.Time { return *now },
		nil,
		meta...,
	)
	if err := serveSessionDurability.restore(serveSessions); err != nil {
		t.Fatalf("install durable session test: %v", err)
	}
	t.Cleanup(func() {
		serveSessions = oldSessions
		serveSessionDurability = oldDurability
	})
}

func readSessionDescriptor(t *testing.T, path, id string) session.Descriptor {
	t.Helper()
	reg := session.NewRegistry(session.NewFileStore(path))
	d, ok, err := reg.Get(id)
	if err != nil {
		t.Fatalf("read descriptor %s: %v", id, err)
	}
	if !ok {
		t.Fatalf("descriptor %s not found", id)
	}
	return d
}

func findGatewaySession(rows []gateway.SessionState, trace string) (gateway.SessionState, bool) {
	for _, row := range rows {
		if row.TraceID == trace {
			return row, true
		}
	}
	return gateway.SessionState{}, false
}

// intPtr is a small helper so the pointer-typed Priority field reads cleanly.
func intPtr(v int) *int { return &v }
