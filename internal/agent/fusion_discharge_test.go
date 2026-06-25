package agent

import (
	"context"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ---------------------------------------------------------------------------
// FUSION — refcount-1 producer→consumer intermediate (#847).
// ---------------------------------------------------------------------------

// dispatchTranscript builds a small transcript: a user task, an assistant turn that
// emits one tool call (the producer), the tool result (the refcount-1 intermediate),
// and the next assistant reasoning step (the single consumer).
func dispatchTranscript() []Message {
	return []Message{
		{Role: RoleUser, Content: "what files are here?"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_1", Type: "function", Function: Func{Name: "ls"}}}},
		{Role: RoleTool, ToolCallID: "call_1", Content: "a.go\nb.go"},
		{Role: RoleAssistant, Content: "There are two files."},
	}
}

// TestFuseEligible_ProvenRefcount1 proves the eligibility predicate returns true on a
// proven refcount==1 producer→consumer pair: the result at index 2 is produced by the
// assistant at index 1 and consumed by exactly the next step at index 3.
func TestFuseEligible_ProvenRefcount1(t *testing.T) {
	msgs := dispatchTranscript()
	if !FuseEligible(msgs, 2) {
		t.Fatalf("expected the refcount-1 tool result at index 2 to be fusion-eligible")
	}
	cands := fusionCandidates(msgs)
	if len(cands) != 1 {
		t.Fatalf("expected exactly 1 fusion candidate, got %d: %+v", len(cands), cands)
	}
	got := cands[0]
	want := FusionCandidate{Producer: 1, Intermediate: 2, Consumer: 3, CallID: "call_1"}
	if got != want {
		t.Fatalf("candidate mismatch:\n got %+v\nwant %+v", got, want)
	}
}

// TestFuseEligible_UnprovenSkipped proves the predicate SKIPS (returns false) when the
// only-one-consumer property is unproven: (a) the result is the last message (no
// consumer witnessed), and (b) the result's call id is re-emitted by a later turn
// (refcount>1, cited again downstream).
func TestFuseEligible_UnprovenSkipped(t *testing.T) {
	// (a) result is the last message — no consumer step → not eligible.
	lastIsResult := []Message{
		{Role: RoleUser, Content: "go"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c", Function: Func{Name: "ls"}}}},
		{Role: RoleTool, ToolCallID: "c", Content: "out"},
	}
	if FuseEligible(lastIsResult, 2) {
		t.Fatalf("a dangling result with no consumer must NOT be eligible (refcount 0)")
	}

	// (b) the same call id is re-dispatched downstream → refcount>1 → not eligible.
	citedTwice := []Message{
		{Role: RoleUser, Content: "go"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c", Function: Func{Name: "ls"}}}},
		{Role: RoleTool, ToolCallID: "c", Content: "out"},
		{Role: RoleAssistant, Content: "thinking"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c", Function: Func{Name: "ls"}}}}, // re-emits call id c
	}
	if FuseEligible(citedTwice, 2) {
		t.Fatalf("a result whose call id is re-emitted downstream must NOT be eligible (refcount>1)")
	}
}

// TestFusion_PromoteFoldsRollbackByteIdentical proves the Promote/Rollback contract:
// a PROMOTED fold yields the fused transcript (the standalone result object is gone,
// its content carried on the consumer), while a ROLLED-BACK fold yields the original
// transcript byte-identical to no-fusion.
func TestFusion_PromoteFoldsRollbackByteIdentical(t *testing.T) {
	orig := dispatchTranscript()
	cands := fusionCandidates(orig)
	if len(cands) != 1 {
		t.Fatalf("setup: expected 1 candidate, got %d", len(cands))
	}
	fused := foldResultIntoConsumer(orig, cands[0])

	// The original must be UNTOUCHED by the fold (so rollback is byte-exact).
	if !reflect.DeepEqual(orig, dispatchTranscript()) {
		t.Fatalf("foldResultIntoConsumer mutated the original transcript")
	}
	// The fused transcript drops the standalone RoleTool object.
	for _, m := range fused {
		if m.Role == RoleTool {
			t.Fatalf("fused transcript still carries a standalone RoleTool object: %+v", m)
		}
	}
	if len(fused) != len(orig)-1 {
		t.Fatalf("fused transcript should be one message shorter; got %d want %d", len(fused), len(orig)-1)
	}
	// The consumer carries the result content (behavior-preserving for the model):
	// the intermediate's "a.go\nb.go" is prepended to the consumer's own reasoning.
	consumer := fused[len(fused)-1]
	wantConsumer := "a.go\nb.go\nThere are two files."
	if consumer.Role != RoleAssistant || consumer.Content != wantConsumer {
		t.Fatalf("consumer should carry folded content %q; got %+v", wantConsumer, consumer)
	}

	const txn = abi.TxnID(7)
	const epoch = uint64(3)

	// PROMOTE arm: the committed transcript is the fused one.
	sp := newFusionSink()
	sp.open(txn, epoch, orig, fused)
	if err := sp.Promote(context.Background(), txn, epoch); err != nil {
		t.Fatalf("Promote: %v", err)
	}
	got, isFused, ok := sp.resolved(txn, epoch)
	if !ok || !isFused {
		t.Fatalf("after Promote expected the fused transcript; ok=%v fused=%v", ok, isFused)
	}
	if !reflect.DeepEqual(got, fused) {
		t.Fatalf("Promote did not yield the fused transcript")
	}

	// ROLLBACK arm: the committed transcript is the ORIGINAL, byte-identical.
	sr := newFusionSink()
	sr.open(txn, epoch, orig, fused)
	if err := sr.Rollback(context.Background(), txn, epoch); err != nil {
		t.Fatalf("Rollback: %v", err)
	}
	gotR, isFusedR, okR := sr.resolved(txn, epoch)
	if !okR || isFusedR {
		t.Fatalf("after Rollback expected the original transcript; ok=%v fused=%v", okR, isFusedR)
	}
	if !reflect.DeepEqual(gotR, dispatchTranscript()) {
		t.Fatalf("Rollback transcript is NOT byte-identical to the original no-fusion transcript")
	}
}

// TestFusionSink_Idempotent proves Promote/Rollback are idempotent and that an
// unknown key is a no-op (mirrors spec.Sink).
func TestFusionSink_Idempotent(t *testing.T) {
	s := newFusionSink()
	// Unknown key: both resolutions are no-ops returning nil.
	if err := s.Promote(context.Background(), 1, 1); err != nil {
		t.Fatalf("Promote of unknown key should be nil, got %v", err)
	}
	if err := s.Rollback(context.Background(), 1, 1); err != nil {
		t.Fatalf("Rollback of unknown key should be nil, got %v", err)
	}
	// Open then double-Promote: second is a no-op, state stays fused.
	s.open(2, 2, dispatchTranscript(), foldResultIntoConsumer(dispatchTranscript(), fusionCandidates(dispatchTranscript())[0]))
	_ = s.Promote(context.Background(), 2, 2)
	_ = s.Promote(context.Background(), 2, 2)
	if _, fused, ok := s.resolved(2, 2); !ok || !fused {
		t.Fatalf("double Promote should leave the fold committed-fused")
	}
}

// ---------------------------------------------------------------------------
// DISCHARGE — sound goal-root collection (#847).
// ---------------------------------------------------------------------------

// recordingPinner is a CASPinner-implementing Resolver that records Pin/Unpin calls,
// so a discharge test can witness exactly which digests were freed.
type recordingPinner struct {
	pinned   map[string]int
	unpinned []string
}

func newRecordingPinner() *recordingPinner { return &recordingPinner{pinned: map[string]int{}} }

func (p *recordingPinner) Resolve(_ context.Context, _ abi.Ref) ([]byte, error) { return nil, nil }
func (p *recordingPinner) Put(_ context.Context, _ []byte) (abi.Ref, error)     { return abi.Ref{}, nil }
func (p *recordingPinner) Pin(d string)                                         { p.pinned[d]++ }
func (p *recordingPinner) Unpin(d string)                                       { p.unpinned = append(p.unpinned, d) }

// recordingBackend wraps a recordingPinner as a RegionBackend.
type recordingBackend struct{ r *recordingPinner }

func (b recordingBackend) Resolver() abi.Resolver    { return b.r }
func (b recordingBackend) Caps() []abi.Capability    { return nil }

func blobRef(digest string) abi.Ref {
	return abi.Ref{Kind: abi.RefBlob, Digest: digest, Len: 1}
}

// TestDischarge_NotWitnessedIsNoop proves discharge is a no-op when the stop is NOT
// witnessed — nothing is unpinned, even with a live CASPinner registered.
func TestDischarge_NotWitnessedIsNoop(t *testing.T) {
	abi.ResetForTest()
	defer abi.ResetForTest()
	p := newRecordingPinner()
	abi.RegisterRegionBackend(recordingBackend{p})

	goal := Root{ID: "goal-A", Spans: []abi.Ref{blobRef("d1"), blobRef("d2")}}

	// nil witness → not witnessed.
	if res := Discharge(goal, nil, nil); res.Discharged {
		t.Fatalf("nil witness must not discharge; got %+v", res)
	}
	// witness that returns false → not witnessed.
	never := StopWitnessFunc(func(string) bool { return false })
	res := Discharge(goal, nil, never)
	if res.Discharged {
		t.Fatalf("unwitnessed stop must not discharge; got %+v", res)
	}
	if len(p.unpinned) != 0 {
		t.Fatalf("no span should be unpinned without a witnessed stop; unpinned=%v", p.unpinned)
	}
}

// TestDischarge_WitnessedUnpinsOwnSpans proves a witnessed discharge unpins the goal's
// spans through the CASPinner seam.
func TestDischarge_WitnessedUnpinsOwnSpans(t *testing.T) {
	abi.ResetForTest()
	defer abi.ResetForTest()
	p := newRecordingPinner()
	abi.RegisterRegionBackend(recordingBackend{p})

	goal := Root{ID: "goal-A", Spans: []abi.Ref{blobRef("d1"), blobRef("d2")}}
	witnessed := StopWitnessFunc(func(id string) bool { return id == "goal-A" })

	res := Discharge(goal, nil, witnessed)
	if !res.Discharged {
		t.Fatalf("witnessed stop must discharge; got %+v", res)
	}
	wantUnpinned := []string{"d1", "d2"}
	if !reflect.DeepEqual(res.Unpinned, wantUnpinned) {
		t.Fatalf("unpinned mismatch: got %v want %v", res.Unpinned, wantUnpinned)
	}
	if !reflect.DeepEqual(p.unpinned, wantUnpinned) {
		t.Fatalf("CASPinner saw wrong unpins: got %v want %v", p.unpinned, wantUnpinned)
	}
}

// TestDischarge_NoOpWhenOtherRootHolds proves the soundness guard: a span the
// discharged goal shares with ANOTHER live root is RETAINED (not unpinned); only the
// spans held solely by the discharged goal are freed.
func TestDischarge_NoOpWhenOtherRootHolds(t *testing.T) {
	abi.ResetForTest()
	defer abi.ResetForTest()
	p := newRecordingPinner()
	abi.RegisterRegionBackend(recordingBackend{p})

	// goal-A holds d1 (solely) and d2 (shared with goal-B). goal-B is still live.
	goalA := Root{ID: "goal-A", Spans: []abi.Ref{blobRef("d1"), blobRef("d2")}}
	goalB := Root{ID: "goal-B", Spans: []abi.Ref{blobRef("d2"), blobRef("d3")}}
	witnessed := StopWitnessFunc(func(id string) bool { return id == "goal-A" })

	res := Discharge(goalA, []Root{goalB}, witnessed)
	if !res.Discharged {
		t.Fatalf("witnessed stop must discharge; got %+v", res)
	}
	// d1 is held only by goal-A → freed. d2 is held by live goal-B → retained.
	if !reflect.DeepEqual(res.Unpinned, []string{"d1"}) {
		t.Fatalf("only d1 (held solely by goal-A) should be unpinned; got %v", res.Unpinned)
	}
	if !reflect.DeepEqual(res.Retained, []string{"d2"}) {
		t.Fatalf("d2 (held by live goal-B) should be retained; got %v", res.Retained)
	}
	if !reflect.DeepEqual(p.unpinned, []string{"d1"}) {
		t.Fatalf("CASPinner must see ONLY d1 unpinned (d2 retained); got %v", p.unpinned)
	}
}

// TestDischarge_InlineRefsNotUnpinned proves inline refs (no backend digest) are never
// unpinned — they carry their own bytes and were never pinned.
func TestDischarge_InlineRefsNotUnpinned(t *testing.T) {
	abi.ResetForTest()
	defer abi.ResetForTest()
	p := newRecordingPinner()
	abi.RegisterRegionBackend(recordingBackend{p})

	goal := Root{ID: "g", Spans: []abi.Ref{
		{Kind: abi.RefInline, Inline: []byte("x")}, // inline → no digest → skip
		blobRef("d1"),
	}}
	witnessed := StopWitnessFunc(func(string) bool { return true })

	res := Discharge(goal, nil, witnessed)
	if !reflect.DeepEqual(res.Unpinned, []string{"d1"}) {
		t.Fatalf("only the backend-resident d1 should be unpinned; got %v", res.Unpinned)
	}
	if !reflect.DeepEqual(p.unpinned, []string{"d1"}) {
		t.Fatalf("CASPinner saw wrong unpins: got %v", p.unpinned)
	}
}
