package model

// v41_expert_fault_attribution_test.go — the #13294 DoD-item-1 witness: the
// V4.1 routed-expert tier fault cost is attributed PER FORWARD PHASE, so the
// physical strix3 serve log can say "prefill faulted N experts over T tokens =
// faults/token" and the 492s first token can be attributed.
//
// The phase split is the load-bearing property (not the totals): on the tier-only
// fixture every routed-expert read is a checkpoint fault, so a prefill-scoped
// forward must attribute ALL of its faults and tokens to the prefill ledger and
// NONE to the decode ledger, and a decode-scoped step must land in decode alone.
// Reuses the exact TestV41ExpertTierRead fixture (v41TierOnlyModel: tier-only
// routed experts at H=I=256, 1 layer) so this witness stays about the same read.

import (
	"math"
	"testing"
)

// v41TierFaultPerProjection is the exact byte cost of ONE faulted fixture
// projection: the fixture geometry is [256 out, 256 in] in Q4_K, i.e. 256 rows
// x (256/256 super-blocks) x 144 block bytes — and 256x256x4 f32 delegated to
// the dequant. All three projections (w1/w3/w2) share this geometry.
const (
	v41TierFaultPerProjectionFaulted = int64(v41TierWidth * (v41TierWidth / qkK) * q4kBlockBytes)
	v41TierFaultPerProjectionDequant = int64(v41TierWidth * v41TierWidth * 4)
)

// TestV41ExpertFaultAttributionPhaseSplit drives the real reduced forward with
// the phase set the way the session entry points set it (prefill, then decode)
// and asserts the two ledgers stay separated, the byte attribution matches the
// tier's geometry, the derived rates are consistent with their denominators,
// and the resident fraction is honestly 0 on an all-tier arm.
func TestV41ExpertFaultAttributionPhaseSplit(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("tier-only admission error = %v, want nil", err)
	}

	// Inert-by-default purity: before any phase was ever set, the snapshot is
	// the zero attribution.
	if got := m.V41ExpertFaultAttribution(); got != (V41ExpertFaultAttribution{}) {
		t.Fatalf("snapshot before any phase = %+v, want the zero attribution", got)
	}

	// ---- prefill phase, scoped exactly as prefillV41 scopes it. ----
	m.v41SetExpertFaultPhase(V41PhasePrefill)
	act, err := m.forwardV41([]int{1, 3}, nil)
	m.v41NoteExpertFaultToken(2)
	m.v41SetExpertFaultPhase(V41PhaseUnknown)
	if err != nil {
		t.Fatalf("tier-only routed-expert prefill error = %v, want nil", err)
	}
	if len(act.Logits) != 2 {
		t.Fatalf("prefill produced %d logit rows, want 2", len(act.Logits))
	}
	for r, row := range act.Logits {
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logits[%d][%d] = %v, want finite", r, i, v)
			}
		}
	}

	got := m.V41ExpertFaultAttribution()
	p := got.Prefill
	if p.Tokens != 2 {
		t.Fatalf("prefill Tokens = %d, want 2", p.Tokens)
	}
	if p.Faults <= 0 {
		t.Fatalf("prefill Faults = %d, want > 0 (every routed-expert read is a tier fault on this fixture)", p.Faults)
	}
	if p.ResidentHits != 0 {
		t.Fatalf("prefill ResidentHits = %d, want 0: the fixture has NO resident routed experts", p.ResidentHits)
	}
	if f := p.ResidentHitFraction; f != 0 {
		t.Fatalf("prefill ResidentHitFraction = %v, want 0 (all expert reads are tier faults)", f)
	}
	if p.DequantBytes != int64(p.Faults)*v41TierFaultPerProjectionDequant {
		t.Fatalf("prefill DequantBytes = %d for %d faults, want %d (per-fault f32 = %d)",
			p.DequantBytes, p.Faults, int64(p.Faults)*v41TierFaultPerProjectionDequant, v41TierFaultPerProjectionDequant)
	}
	if p.FaultedBytes != int64(p.Faults)*v41TierFaultPerProjectionFaulted {
		t.Fatalf("prefill FaultedBytes = %d for %d faults, want %d (per-fault stride = %d)",
			p.FaultedBytes, p.Faults, int64(p.Faults)*v41TierFaultPerProjectionFaulted, v41TierFaultPerProjectionFaulted)
	}
	if want := float64(p.Faults) / float64(p.Tokens); math.Abs(p.FaultsPerToken-want) > 1e-9 {
		t.Fatalf("prefill FaultsPerToken = %v, want %v", p.FaultsPerToken, want)
	}
	// Each routed token picks top-6 experts and serves each pick with exactly
	// 3 tier faults, so the 2-token prefill faults 2*6*3 projections.
	if want := 2 * V41RouterTopK * 3; p.Faults != want {
		t.Fatalf("prefill Faults = %d, want %d (2 tokens x top-%d experts x 3 projections)",
			p.Faults, want, V41RouterTopK)
	}

	// Phase separation is real before any decode ran.
	if d := got.Decode; d.Tokens != 0 || d.Faults != 0 || d.FaultedBytes != 0 || d.DequantBytes != 0 || d.ResidentHits != 0 {
		t.Fatalf("decode ledger after prefill-only run = %+v, want all zero", d)
	}

	// ---- decode phase, scoped exactly as stepV41 scopes it. ----
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	act2, err := m.forwardV41([]int{5}, nil)
	m.v41NoteExpertFaultToken(1)
	m.v41SetExpertFaultPhase(V41PhaseUnknown)
	if err != nil {
		t.Fatalf("tier-only routed-expert decode error = %v, want nil", err)
	}
	if len(act2.Logits) != 1 {
		t.Fatalf("decode produced %d logit rows, want 1", len(act2.Logits))
	}
	got = m.V41ExpertFaultAttribution()
	d := got.Decode
	if d.Tokens != 1 || d.Faults <= 0 {
		t.Fatalf("decode ledger Tokens=%d Faults=%d, want Tokens=1 with Faults>0: the decode step's faults must land in the decode ledger, not prefill", d.Tokens, d.Faults)
	}
	if d.FaultedBytes != int64(d.Faults)*v41TierFaultPerProjectionFaulted ||
		d.DequantBytes != int64(d.Faults)*v41TierFaultPerProjectionDequant {
		t.Fatalf("decode byte attribution inconsistent: FaultedBytes=%d DequantBytes=%d for %d faults",
			d.FaultedBytes, d.DequantBytes, d.Faults)
	}
	if want := float64(d.Faults); math.Abs(d.FaultsPerToken-want) > 1e-9 {
		t.Fatalf("decode FaultsPerToken = %v, want %v (1 token)", d.FaultsPerToken, want)
	}
	// The prefill ledger must be UNCHANGED by the decode step.
	if p2 := got.Prefill; p2 != p {
		t.Fatalf("prefill ledger changed by the decode step: %+v, want %+v", p2, p)
	}
}

// TestV41ExpertFaultAttributionSessionScoping pins the wiring the session
// entry points own: Session.Prefill scopes its tokens to the prefill ledger
// and Session.Step lands in the decode ledger — the split a bare
// forwardV41-against-ledger test cannot witness on its own.
func TestV41ExpertFaultAttributionSessionScoping(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("tier-only admission error = %v, want nil", err)
	}
	s := &Session{M: m}
	// The fixture's reduced vocab is 8, so the ids stay inside [0,8).
	s.Prefill([]int{1, 3})
	s.Prefill([]int{2, 5, 6})
	s.Step(4)

	got := m.V41ExpertFaultAttribution()
	if got.Decode.Tokens != 1 || got.Decode.Faults <= 0 {
		t.Fatalf("after Session.Step the decode ledger = %+v, want Tokens=1 with Faults>0", got.Decode)
	}
	if got.Prefill.Tokens != 5 {
		t.Fatalf("after Session.Prefill the prefill Tokens = %d, want 5 (2 + 3 ids)", got.Prefill.Tokens)
	}
	if got.Prefill.Faults <= 0 {
		t.Fatalf("after Session.Prefill the prefill Faults = %d, want > 0", got.Prefill.Faults)
	}
}

// TestV41ExpertFaultAttributionResidentHit pins the resident arm: a read the
// tier never touches counts a ResidentHit, not a fault, so
// ResidentHitFraction reads as the share of routed-expert reads served from
// residency. Built on the note seams directly because the tier-only fixture
// cannot produce a resident hit.
func TestV41ExpertFaultAttributionResidentHit(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	m.v41NoteExpertResidentHit()
	m.v41NoteExpertResidentHit()
	m.v41NoteExpertTierFault(7, 16)
	m.v41NoteExpertFaultToken(2)
	m.v41SetExpertFaultPhase(V41PhaseUnknown)

	got := m.V41ExpertFaultAttribution().Decode
	if got.ResidentHits != 2 || got.Faults != 1 {
		t.Fatalf("decode ResidentHits=%d Faults=%d, want 2/1", got.ResidentHits, got.Faults)
	}
	if want := 2.0 / 3.0; math.Abs(got.ResidentHitFraction-want) > 1e-9 {
		t.Fatalf("decode ResidentHitFraction = %v, want %v", got.ResidentHitFraction, want)
	}
	if got.FaultedBytes != 7 || got.DequantBytes != 16 {
		t.Fatalf("decode FaultedBytes=%d DequantBytes=%d, want 7/16", got.FaultedBytes, got.DequantBytes)
	}
}

// TestV41ExpertFaultAttributionInertDefault pins the purity contract: a model
// that never sets a phase records nothing, a nil model never panics, and an
// empty live phase reports zero rates (never NaN).
func TestV41ExpertFaultAttributionInertDefault(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	// Observations WITHOUT ever setting a phase must be dropped…
	m.v41NoteExpertTierFault(123, 456)
	m.v41NoteExpertResidentHit()
	m.v41NoteExpertFaultToken(9)
	// …including on a nil model, which must not panic.
	var nilModel *Model
	nilModel.v41NoteExpertTierFault(1, 1)
	nilModel.v41NoteExpertResidentHit()
	nilModel.v41NoteExpertFaultToken(1)
	nilModel.v41SetExpertFaultPhase(V41PhasePrefill)
	if got := nilModel.V41ExpertFaultAttribution(); got != (V41ExpertFaultAttribution{}) {
		t.Fatalf("nil model snapshot = %+v, want zero", got)
	}
	if got := m.V41ExpertFaultAttribution(); got != (V41ExpertFaultAttribution{}) {
		t.Fatalf("unscoped notes on a phase-less model = %+v, want zero (the notes must not fabricate a denominator)", got)
	}
	// A note under a CLEARED phase routes nowhere either.
	m.v41SetExpertFaultPhase(V41PhasePrefill)
	m.v41SetExpertFaultPhase(V41PhaseUnknown)
	m.v41NoteExpertTierFault(123, 456)
	if got := m.V41ExpertFaultAttribution(); got != (V41ExpertFaultAttribution{}) {
		t.Fatalf("notes after a cleared phase = %+v, want zero", got)
	}
	// A live but empty phase reports zeros, never NaN.
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	got := m.V41ExpertFaultAttribution().Decode
	m.v41SetExpertFaultPhase(V41PhaseUnknown)
	if math.IsNaN(got.FaultsPerToken) || math.IsNaN(got.ResidentHitFraction) {
		t.Fatalf("empty live phase reports NaN rates: %+v, want zeros", got)
	}
}

// TestV41ExpertFaultAttributionFaultedRawBytes pins the stride source the
// instrument reports and the fail-closed interplay: a detached-tier model
// refuses at admission (named refusal, no faults counted anywhere) and a
// zero-representation weight reports a zero stride.
func TestV41ExpertFaultAttributionFaultedRawBytes(t *testing.T) {
	detached := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	detached.expertCheckpoint = nil
	// TestV41ExpertTierReadFailsClosed's contract, re-asserted for this leaf:
	// the refusal happens at admission, so no phase is ever set and the
	// attribution stays zero — a refusal can never inflate the denominator.
	if err := detached.v41ForwardAdmitted(); err == nil {
		t.Fatal("admission admitted a tier-only model after the tier was detached; want a named refusal")
	}
	if got := detached.V41ExpertFaultAttribution(); got != (V41ExpertFaultAttribution{}) {
		t.Fatalf("detached-tier attribution = %+v, want zero", got)
	}
	if b := faultedRawBytes(expertWeight{}); b != 0 {
		t.Fatalf("faultedRawBytes(expertWeight{}) = %d, want 0", b)
	}
	if b := faultedRawBytes(expertWeight{q4: &q4kTensor{raw: []byte("0123")}}); b != 4 {
		t.Fatalf("faultedRawBytes(q4) = %d, want 4", b)
	}
	if b := faultedRawBytes(expertWeight{kq: &kQuantTensor{raw: []byte("01")}}); b != 2 {
		t.Fatalf("faultedRawBytes(kq) = %d, want 2", b)
	}
}
