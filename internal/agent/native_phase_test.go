package agent

import (
	"context"
	"strconv"
	"testing"
	"time"
)

// native_phase_test.go — the #13120 witness suite for the observed native-phase producer. It
// exercises the REAL recorder the generate path calls (recordNativePhase) and reads it back
// through the NativePhaseReporter API; nothing here reimplements the vocabulary or the ledger.

// TestNativePhaseVocabularyClosedSet is the closure arm: the vocabulary is exactly the three
// minted tokens, String() round-trips each, and an unminted token is not a member. This is
// the property that keeps a phase label from drifting into free text.
func TestNativePhaseVocabularyClosedSet(t *testing.T) {
	minted := []NativePhase{NativePhasePrefill, NativePhaseDecode, NativePhaseTerminal}
	for _, p := range minted {
		if !nativePhaseKnown(p) {
			t.Errorf("minted token %q reported unknown", p)
		}
		if p.String() != string(p) {
			t.Errorf("String() = %q, want verbatim %q", p.String(), string(p))
		}
		if p.String() == "" {
			t.Errorf("minted token %v renders blank", p)
		}
	}

	// Distinct tokens must have distinct wire labels (a shared label would merge phases).
	seen := map[string]bool{}
	for _, p := range minted {
		if seen[p.String()] {
			t.Errorf("phase label %q is emitted by two tokens", p.String())
		}
		seen[p.String()] = true
	}

	for _, bogus := range []NativePhase{"", "prefill", "native_phase_free_text", "unknown"} {
		if nativePhaseKnown(bogus) {
			t.Errorf("unminted token %q admitted into the closed vocabulary", bogus)
		}
	}
}

// TestNativePhaseRecorderRejectsUnknownToken proves closure at the recorder, not just the
// predicate: an unknown token must leave NO observation behind (a typo cannot enter the log).
func TestNativePhaseRecorderRejectsUnknownToken(t *testing.T) {
	p := &InKernelPlanner{}
	p.recordNativePhase("trace-bad", NativePhase("native_phase_bogus"), time.Unix(1, 0).UTC(), time.Second, true)
	if obs, ok := p.NativePhaseObservation("trace-bad"); ok {
		t.Fatalf("recorder stored an unminted token: %+v", obs)
	}
}

// TestNativePhaseReadbackPerTraceID is the identity arm: an observation is readable only under
// its own trace id, latest-wins per id, and a second trace id does not alias it.
func TestNativePhaseReadbackPerTraceID(t *testing.T) {
	p := &InKernelPlanner{}
	at := time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)

	p.recordNativePhase("trace-1", NativePhasePrefill, at, 900*time.Millisecond, true)

	got, ok := p.NativePhaseObservation("trace-1")
	if !ok {
		t.Fatal("observation for trace-1 not readable")
	}
	if got.TraceID != "trace-1" || got.Phase != NativePhasePrefill {
		t.Errorf("trace/phase = %q/%q, want trace-1/%q", got.TraceID, got.Phase, NativePhasePrefill)
	}
	if !got.At.Equal(at) || got.Elapsed != 900*time.Millisecond || !got.Completed {
		t.Errorf("field round-trip lost data: %+v", got)
	}
	if _, ok := p.NativePhaseObservation("trace-2"); ok {
		t.Error("observation aliased across trace ids")
	}

	// A later observation for the same id replaces the earlier one.
	p.recordNativePhase("trace-1", NativePhaseDecode, at.Add(time.Second), 2*time.Second, false)
	if got, ok := p.NativePhaseObservation("trace-1"); !ok || got.Phase != NativePhaseDecode || got.Completed {
		t.Errorf("latest observation = %+v/%v, want decode incomplete", got, ok)
	}
}

// TestNativePhaseEmptyTraceIDIsRetained is the empty-bucket arm: a request that carried no
// trace id is attributed to its phase under the empty key rather than silently dropped, and
// the stored TraceID stays empty (never fabricated into a placeholder).
func TestNativePhaseEmptyTraceIDIsRetained(t *testing.T) {
	p := &InKernelPlanner{}
	at := time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)

	p.recordNativePhase("", NativePhaseTerminal, at, 0, true)

	obs, ok := p.NativePhaseObservation("")
	if !ok {
		t.Fatal("empty trace id was dropped instead of retained")
	}
	if obs.TraceID != "" {
		t.Errorf("TraceID = %q, want empty (never fabricated)", obs.TraceID)
	}
	if obs.Phase != NativePhaseTerminal {
		t.Errorf("Phase = %q, want %q", obs.Phase, NativePhaseTerminal)
	}
}

// TestNativePhaseLedgerEvictsOldestAtCap is the bounded-growth arm: distinct trace ids are
// capped at nativePhaseLogCap and evicted oldest-first, while repeats of a live id do not
// consume new slots.
func TestNativePhaseLedgerEvictsOldestAtCap(t *testing.T) {
	p := &InKernelPlanner{}
	const overflow = 7
	total := nativePhaseLogCap + overflow
	for i := 0; i < total; i++ {
		p.recordNativePhase(nativePhaseKey(i), NativePhaseDecode, time.Unix(int64(i), 0).UTC(), 0, true)
	}

	if n := len(p.nativePhaseLog); n != nativePhaseLogCap {
		t.Errorf("ledger holds %d ids, want cap %d", n, nativePhaseLogCap)
	}
	if n := len(p.nativePhaseSeq); n != nativePhaseLogCap {
		t.Errorf("eviction sequence holds %d ids, want cap %d", n, nativePhaseLogCap)
	}

	// The first `overflow` ids were evicted; the newest is retained.
	for i := 0; i < overflow; i++ {
		if _, ok := p.NativePhaseObservation(nativePhaseKey(i)); ok {
			t.Errorf("oldest id %s survived past the cap", nativePhaseKey(i))
		}
	}
	if _, ok := p.NativePhaseObservation(nativePhaseKey(total - 1)); !ok {
		t.Errorf("newest id %s evicted under the cap", nativePhaseKey(total-1))
	}

	// Repeating an id already in the ledger must not evict another id.
	before := len(p.nativePhaseSeq)
	p.recordNativePhase(nativePhaseKey(total-1), NativePhaseTerminal, time.Unix(9999, 0).UTC(), 0, true)
	if after := len(p.nativePhaseSeq); after != before {
		t.Errorf("re-record grew the sequence from %d to %d", before, after)
	}
}

// TestNativePhaseNilAndZeroPlannerSafety checks the object-safety arms: a nil receiver reports
// not-observed and its record is a no-op (no panic), while a bare zero-value planner records
// from its first call. It also pins the trace-id context helper both ways.
func TestNativePhaseNilAndZeroPlannerSafety(t *testing.T) {
	var nilPlanner *InKernelPlanner
	if _, ok := nilPlanner.NativePhaseObservation("trace"); ok {
		t.Error("nil planner reported an observation")
	}
	nilPlanner.recordNativePhase("trace", NativePhasePrefill, time.Now(), 0, true) // must not panic

	zero := &InKernelPlanner{}
	zero.recordNativePhase("trace", NativePhasePrefill, time.Now(), 0, true)
	if _, ok := zero.NativePhaseObservation("trace"); !ok {
		t.Error("zero planner failed to record")
	}
	if _, ok := zero.NativePhaseObservation("absent"); ok {
		t.Error("zero planner reported an unrecorded id")
	}

	if id := nativePhaseTraceID(context.Background()); id != "" {
		t.Errorf("trace id from a bare context = %q, want empty", id)
	}
	if id := nativePhaseTraceID(context.WithValue(context.Background(), "trace_id", "ctx-7")); id != "ctx-7" {
		t.Errorf("trace id from context = %q, want ctx-7", id)
	}
	// A wrong-typed value must not panic; it yields the empty id.
	if id := nativePhaseTraceID(context.WithValue(context.Background(), "trace_id", 42)); id != "" {
		t.Errorf("non-string trace id = %q, want empty", id)
	}
	if id := nativePhaseTraceID(nil); id != "" {
		t.Errorf("nil context trace id = %q, want empty", id)
	}
}

func nativePhaseKey(i int) string { return "np-" + strconv.Itoa(i) }
