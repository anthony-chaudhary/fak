package model

// weightdelta_test.go — pins the phase-1 WeightDelta/WeightEpoch invariants:
// bit-identical LoRA round trip, zero base mutation, monotonic generations, atomic
// all-or-nothing apply, revert restoration of weights AND digest identity, typed
// fail-closed refusals (errors.Is sentinels, never prose), idempotent re-apply, and
// the kv-compat declaration window the span-check consumes. All witnesses are
// [SW-VERIFIED]: synthetic in-memory model, CPU reference backend, no hardware.

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// weightDeltaTestTarget is the tiny synthetic q_proj the LoRA witnesses use
// ([Out=32, In=32] under loraTestCfg).
const weightDeltaTestTarget = "model.layers.0.self_attn.q_proj.weight"

func weightDeltaFixture(t *testing.T) (*Model, *WeightDeltaSource, []float32) {
	t.Helper()
	m := NewSynthetic(loraTestCfg())
	base := append([]float32(nil), m.tensor(weightDeltaTestTarget)...)
	wds := NewWeightDeltaSource(m, nil)
	return m, wds, base
}

func weightDeltaOf(adapters ...*LoRAAdapter) WeightDelta {
	return WeightDelta{Kind: WeightDeltaLoRAAdapterAdd, Adapters: adapters, KVCompat: KVCompatEpochBump}
}

func mustWeight(t *testing.T, wds *WeightDeltaSource, name string, want compute.Dtype) compute.Tensor {
	t.Helper()
	tn, err := wds.Weight(name, want)
	if err != nil {
		t.Fatalf("Weight(%s, %s): %v", name, want, err)
	}
	return tn
}

// hostF32Of extracts the host f32 bytes of a tensor served through the CPU reference
// (the compute.HostBuffer seam), failing the test on any non-host storage.
func hostF32Of(t *testing.T, tn compute.Tensor) []float32 {
	t.Helper()
	hb, ok := tn.Buf().(compute.HostBuffer)
	if !ok {
		t.Fatalf("served tensor has no host buffer (got %T)", tn.Buf())
	}
	return hb.F32()
}

func firstDiffIndex(a, b []float32) int {
	for i := range a {
		if i >= len(b) || a[i] != b[i] {
			return i
		}
	}
	return -1
}

// int8sEqual reports byte-wise equality of two int8 slices (Q8_0 codes are signed).
func int8sEqual(a, b []int8) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestWeightDeltaLoRARoundTripMatchesMergePath pins the phase-1 acceptance: serving
// a patched tensor through the delta layer is BIT-IDENTICAL to the existing LoRA
// merge path (a fresh copy of the base folded with LoRASet.MergeInto), the base blob
// is never mutated (zero base mutation), and un-patched names pass through the base
// funnel untouched.
func TestWeightDeltaLoRARoundTripMatchesMergePath(t *testing.T) {
	m, wds, base := weightDeltaFixture(t)
	a := newTestAdapter("math", weightDeltaTestTarget, 32, 32, 4, 8, 7)
	d := weightDeltaOf(a)
	d.BaseDigest = wds.Digest()

	rec, err := wds.Apply(d)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if rec.EpochFrom != 0 || rec.EpochTo != 1 || rec.NoOp || rec.Reverted {
		t.Fatalf("receipt = %+v, want an epoch 0->1 apply", rec)
	}
	if rec.BaseDigest == rec.ResultDigest {
		t.Fatalf("receipt digests equal: the delta changed nothing")
	}
	if rec.MovedBytes != 32*32*4 {
		t.Fatalf("MovedBytes = %d, want %d", rec.MovedBytes, 32*32*4)
	}

	got := hostF32Of(t, mustWeight(t, wds, weightDeltaTestTarget, compute.F32))

	// The existing LoRA path: a fresh copy of the base folded with the proven merge.
	ref := append([]float32(nil), base...)
	set := NewLoRASet()
	if err := set.Add(a); err != nil {
		t.Fatal(err)
	}
	if err := set.MergeInto(weightDeltaTestTarget, ref); err != nil {
		t.Fatal(err)
	}
	if diff := firstDiffIndex(got, ref); diff >= 0 {
		t.Fatalf("delta-layer round trip diverges from the existing LoRA merge path at element %d", diff)
	}

	// Zero base mutation: the resident blob is still byte-identical to before.
	if diff := firstDiffIndex(m.tensor(weightDeltaTestTarget), base); diff >= 0 {
		t.Fatalf("base blob was mutated by Apply at element %d (zero-base-mutation violated)", diff)
	}

	// Un-patched names pass through the base funnel untouched.
	other := "model.layers.0.self_attn.k_proj.weight"
	baseOther := append([]float32(nil), m.tensor(other)...)
	if diff := firstDiffIndex(hostF32Of(t, mustWeight(t, wds, other, compute.F32)), baseOther); diff >= 0 {
		t.Fatalf("un-patched name %s differs from the base funnel at element %d", other, diff)
	}
}

// TestWeightDeltaMultiAdapterComposition pins additive composition across deltas:
// two deltas on the same target sum exactly like the LoRASet multi-adapter path, and
// each transition keeps its own receipt.
func TestWeightDeltaMultiAdapterComposition(t *testing.T) {
	_, wds, base := weightDeltaFixture(t)
	a1 := newTestAdapter("math", weightDeltaTestTarget, 32, 32, 4, 8, 11)
	a2 := newTestAdapter("code", weightDeltaTestTarget, 32, 32, 4, 8, 13)

	d1 := weightDeltaOf(a1)
	d1.BaseDigest = wds.Digest()
	if _, err := wds.Apply(d1); err != nil {
		t.Fatalf("Apply D1: %v", err)
	}
	d2 := weightDeltaOf(a2)
	d2.BaseDigest = wds.Digest()
	if _, err := wds.Apply(d2); err != nil {
		t.Fatalf("Apply D2: %v", err)
	}

	got := hostF32Of(t, mustWeight(t, wds, weightDeltaTestTarget, compute.F32))

	// The existing path for the composed state: one set, both adapters, one merge.
	ref := append([]float32(nil), base...)
	set := NewLoRASet()
	if err := set.Add(a1); err != nil {
		t.Fatal(err)
	}
	if err := set.Add(a2); err != nil {
		t.Fatal(err)
	}
	if err := set.MergeInto(weightDeltaTestTarget, ref); err != nil {
		t.Fatal(err)
	}
	if diff := firstDiffIndex(got, ref); diff >= 0 {
		t.Fatalf("composed delta state diverges from the LoRASet multi-adapter merge at element %d", diff)
	}
}

// TestWeightDeltaEpochMonotonicAcrossApplyAndRevert pins invariant 1: generation
// values are never reused or rewound — every committed apply AND every revert
// advances them, and no value ever repeats.
func TestWeightDeltaEpochMonotonicAcrossApplyAndRevert(t *testing.T) {
	_, wds, _ := weightDeltaFixture(t)
	seen := []uint64{wds.Epoch()} // 0

	step := func(label string, fn func() error) {
		t.Helper()
		if err := fn(); err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		e := wds.Epoch()
		if e <= seen[len(seen)-1] {
			t.Fatalf("%s: epoch %d does not advance past %d (monotonicity violated)", label, e, seen[len(seen)-1])
		}
		seen = append(seen, e)
	}

	a1 := newTestAdapter("math", weightDeltaTestTarget, 32, 32, 4, 8, 21)
	a2 := newTestAdapter("code", weightDeltaTestTarget, 32, 32, 4, 8, 23)

	revert := func() error { _, err := wds.Revert(); return err }
	step("apply D1", func() error { d := weightDeltaOf(a1); d.BaseDigest = wds.Digest(); _, err := wds.Apply(d); return err })
	step("apply D2", func() error { d := weightDeltaOf(a2); d.BaseDigest = wds.Digest(); _, err := wds.Apply(d); return err })
	step("revert D2", revert)
	step("revert D1", revert)
	step("re-apply D1", func() error { d := weightDeltaOf(a1); d.BaseDigest = wds.Digest(); _, err := wds.Apply(d); return err })

	if len(seen) != 6 {
		t.Fatalf("observed %d generations, want 6", len(seen))
	}
	for i := range seen {
		for j := i + 1; j < len(seen); j++ {
			if seen[i] == seen[j] {
				t.Fatalf("epoch %d reused at positions %d and %d", seen[i], i, j)
			}
		}
	}
}

// TestWeightDeltaRevertRestoresWeightsAndDigest pins "revert restores prior weights
// + epoch": the content comes back bit-identically AND its digest identity is
// restored (spans stamped under it become servable again), while the generation
// counter keeps advancing. A two-level revert lands on the intermediate state.
func TestWeightDeltaRevertRestoresWeightsAndDigest(t *testing.T) {
	m, wds, base := weightDeltaFixture(t)
	preDigest := wds.Digest()
	preEpoch := wds.Epoch()

	a := newTestAdapter("math", weightDeltaTestTarget, 32, 32, 4, 8, 31)
	d := weightDeltaOf(a)
	d.BaseDigest = preDigest
	if _, err := wds.Apply(d); err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if wds.Digest() == preDigest {
		t.Fatalf("digest unchanged after apply")
	}

	rrec, err := wds.Revert()
	if err != nil {
		t.Fatalf("Revert: %v", err)
	}
	if !rrec.Reverted || rrec.ResultDigest != preDigest {
		t.Fatalf("revert receipt = %+v, want ResultDigest restored to %s", rrec, preDigest)
	}
	if got := wds.Digest(); got != preDigest {
		t.Fatalf("Digest after revert = %s, want the pre-apply digest %s", got, preDigest)
	}
	if diff := firstDiffIndex(hostF32Of(t, mustWeight(t, wds, weightDeltaTestTarget, compute.F32)), base); diff >= 0 {
		t.Fatalf("weights after revert differ from the pre-apply base at element %d", diff)
	}
	if wds.Epoch() <= preEpoch {
		t.Fatalf("epoch %d did not advance past pre-apply %d (monotonicity)", wds.Epoch(), preEpoch)
	}
	// Revert with an empty active chain is a typed refusal.
	if _, err := wds.Revert(); !errors.Is(err, ErrWeightDeltaRevertEmpty) {
		t.Fatalf("Revert on empty chain: err=%v, want ErrWeightDeltaRevertEmpty", err)
	}

	// Two-level revert lands on the intermediate state, not the base.
	kTarget := "model.layers.0.self_attn.k_proj.weight"
	b := newTestAdapter("kv", kTarget, 16, 32, 4, 8, 33)
	dA := weightDeltaOf(a)
	dA.BaseDigest = wds.Digest()
	recA, err := wds.Apply(dA)
	if err != nil {
		t.Fatalf("Apply D-A: %v", err)
	}
	dB := weightDeltaOf(b)
	dB.BaseDigest = wds.Digest()
	if _, err := wds.Apply(dB); err != nil {
		t.Fatalf("Apply D-B: %v", err)
	}
	if _, err := wds.Revert(); err != nil {
		t.Fatalf("Revert D-B: %v", err)
	}
	if got := wds.Digest(); got != recA.ResultDigest {
		t.Fatalf("Digest after two-level revert = %s, want the intermediate digest %s", got, recA.ResultDigest)
	}
	qGot := hostF32Of(t, mustWeight(t, wds, weightDeltaTestTarget, compute.F32))
	ref := append([]float32(nil), base...)
	set := NewLoRASet()
	if err := set.Add(a); err != nil {
		t.Fatal(err)
	}
	if err := set.MergeInto(weightDeltaTestTarget, ref); err != nil {
		t.Fatal(err)
	}
	if diff := firstDiffIndex(qGot, ref); diff >= 0 {
		t.Fatalf("q_proj after two-level revert diverges from the D-A-only state at element %d", diff)
	}
	kBase := append([]float32(nil), m.tensor(kTarget)...)
	if diff := firstDiffIndex(hostF32Of(t, mustWeight(t, wds, kTarget, compute.F32)), kBase); diff >= 0 {
		t.Fatalf("k_proj after two-level revert is not back to the base at element %d", diff)
	}
}

// TestWeightDeltaAtomicAllOrNothing pins invariant 2 (atomic apply): a delta whose
// second adapter fails geometry validation leaves the live state EXACTLY as it was —
// weights, digest, epoch — and a CAS mismatch is refused before any materialization;
// failed attempts leave no residue behind.
func TestWeightDeltaAtomicAllOrNothing(t *testing.T) {
	_, wds, base := weightDeltaFixture(t)
	beforeDigest := wds.Digest()
	beforeEpoch := wds.Epoch()

	good := newTestAdapter("math", weightDeltaTestTarget, 32, 32, 4, 8, 41)
	bad := &LoRAAdapter{
		Adapter: "broken", Target: weightDeltaTestTarget,
		In: 31, Out: 32, Rank: 4, Alpha: 8,
		A: lcgFill(4*31, 3, 0.1), B: lcgFill(32*4, 5, 0.1),
	}
	d := weightDeltaOf(good, bad)
	d.BaseDigest = beforeDigest
	if _, err := wds.Apply(d); err == nil {
		t.Fatalf("Apply with a bad adapter succeeded, want a refusal")
	}
	if wds.Digest() != beforeDigest || wds.Epoch() != beforeEpoch {
		t.Fatalf("failed Apply moved the state (digest=%q epoch=%d, want %q/%d)", wds.Digest(), wds.Epoch(), beforeDigest, beforeEpoch)
	}
	if diff := firstDiffIndex(hostF32Of(t, mustWeight(t, wds, weightDeltaTestTarget, compute.F32)), base); diff >= 0 {
		t.Fatalf("failed Apply left partial patched bytes behind at element %d", diff)
	}

	// CAS mismatch is refused before any materialization, state intact.
	d2 := weightDeltaOf(good)
	d2.BaseDigest = "stale-digest"
	if _, err := wds.Apply(d2); !errors.Is(err, ErrWeightDeltaBaseMismatch) {
		t.Fatalf("CAS mismatch: err=%v, want ErrWeightDeltaBaseMismatch", err)
	}
	if wds.Digest() != beforeDigest || wds.Epoch() != beforeEpoch {
		t.Fatalf("CAS refusal moved the state")
	}

	// The failed attempts left no residue: a clean apply now succeeds.
	d3 := weightDeltaOf(good)
	d3.BaseDigest = wds.Digest()
	if _, err := wds.Apply(d3); err != nil {
		t.Fatalf("clean apply after refusals: %v", err)
	}
}

// TestWeightDeltaUnknownKindFailsClosed pins invariant 3 (typed, errors.Is-assertable
// refusal): an unknown delta kind, an empty delta, and a ghost target are each
// refused with their sentinel, and the state stays put.
func TestWeightDeltaUnknownKindFailsClosed(t *testing.T) {
	_, wds, _ := weightDeltaFixture(t)

	unknown := WeightDelta{Kind: WeightDeltaKind(99), BaseDigest: wds.Digest(),
		Adapters: []*LoRAAdapter{newTestAdapter("x", weightDeltaTestTarget, 32, 32, 4, 8, 51)}}
	if _, err := wds.Apply(unknown); !errors.Is(err, ErrUnknownWeightDeltaKind) {
		t.Fatalf("unknown kind: err=%v, want ErrUnknownWeightDeltaKind", err)
	}
	if _, err := wds.Apply(WeightDelta{Kind: WeightDeltaLoRAAdapterAdd, BaseDigest: wds.Digest()}); !errors.Is(err, ErrEmptyWeightDelta) {
		t.Fatalf("empty delta: err=%v, want ErrEmptyWeightDelta", err)
	}
	ghost := newTestAdapter("ghost", "model.layers.9.self_attn.q_proj.weight", 32, 32, 4, 8, 53)
	dg := weightDeltaOf(ghost)
	dg.BaseDigest = wds.Digest()
	if _, err := wds.Apply(dg); !errors.Is(err, ErrWeightDeltaTargetNotResident) {
		t.Fatalf("ghost target: err=%v, want ErrWeightDeltaTargetNotResident", err)
	}
	if wds.Epoch() != 0 {
		t.Fatalf("refusals advanced the epoch to %d", wds.Epoch())
	}
}

// TestWeightDeltaReapplyIsIdempotent pins invariant 2 (CAS identity): re-applying a
// delta that is already active is a no-op — no generation bump, no chain growth,
// receipt marked NoOp — and a subsequent revert still sees exactly one transition.
func TestWeightDeltaReapplyIsIdempotent(t *testing.T) {
	_, wds, _ := weightDeltaFixture(t)
	a := newTestAdapter("math", weightDeltaTestTarget, 32, 32, 4, 8, 61)
	d := weightDeltaOf(a)
	d.BaseDigest = wds.Digest()
	rec1, err := wds.Apply(d)
	if err != nil {
		t.Fatalf("first Apply: %v", err)
	}
	rec2, err := wds.Apply(d) // same BaseDigest, same factors
	if err != nil {
		t.Fatalf("re-apply: %v", err)
	}
	if !rec2.NoOp {
		t.Fatalf("re-apply receipt = %+v, want NoOp=true", rec2)
	}
	if rec2.EpochTo != rec1.EpochTo || wds.Epoch() != rec1.EpochTo {
		t.Fatalf("re-apply bumped the generation (%d -> %d)", rec1.EpochTo, wds.Epoch())
	}
	if rec2.ResultDigest != rec1.ResultDigest {
		t.Fatalf("re-apply changed the state digest")
	}
	// Exactly one committed transition: one revert works, the next is refused.
	if _, err := wds.Revert(); err != nil {
		t.Fatalf("revert after idempotent re-apply: %v", err)
	}
	if _, err := wds.Revert(); !errors.Is(err, ErrWeightDeltaRevertEmpty) {
		t.Fatalf("second revert: err=%v, want ErrWeightDeltaRevertEmpty (no chain growth from the no-op)", err)
	}
}

// TestWeightDeltaDeclarationsSince pins the span-check window: declarations are
// recorded per committed transition, oldest first; no-ops record nothing; and a
// revert records an epoch_bump (a revert IS a weight change).
func TestWeightDeltaDeclarationsSince(t *testing.T) {
	_, wds, _ := weightDeltaFixture(t)
	a1 := newTestAdapter("math", weightDeltaTestTarget, 32, 32, 4, 8, 71)
	a2 := newTestAdapter("code", weightDeltaTestTarget, 32, 32, 4, 8, 73)

	d1 := weightDeltaOf(a1)
	d1.BaseDigest = wds.Digest()
	d1.KVCompat = KVCompatEpochBump
	if _, err := wds.Apply(d1); err != nil {
		t.Fatal(err)
	}
	d2 := weightDeltaOf(a2)
	d2.BaseDigest = wds.Digest()
	d2.KVCompat = KVCompatCompatible
	if _, err := wds.Apply(d2); err != nil {
		t.Fatal(err)
	}

	check := func(since uint64, want []WeightDeltaKVCompat) {
		t.Helper()
		got := wds.DeclarationsSince(since)
		if len(got) != len(want) {
			t.Fatalf("DeclarationsSince(%d) = %v, want %v", since, got, want)
			return
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("DeclarationsSince(%d)[%d] = %v, want %v", since, i, got[i], want[i])
			}
		}
	}
	check(0, []WeightDeltaKVCompat{KVCompatEpochBump, KVCompatCompatible})
	check(1, []WeightDeltaKVCompat{KVCompatCompatible})
	check(2, nil)

	if _, err := wds.Revert(); err != nil {
		t.Fatal(err)
	}
	check(0, []WeightDeltaKVCompat{KVCompatEpochBump, KVCompatCompatible, KVCompatEpochBump})
}

// TestWeightDeltaQ8ServingConsistent pins the quantized side of the funnel: a
// patched name requested as Q8_0 quantizes the PATCHED f32 on demand (bit-identical
// codes and scales to quantizing the reference-merged copy), and a patched name
// requested from a quantized store the delta layer cannot patch is refused, not
// answered stale.
func TestWeightDeltaQ8ServingConsistent(t *testing.T) {
	_, wds, _ := weightDeltaFixture(t)
	a := newTestAdapter("math", weightDeltaTestTarget, 32, 32, 4, 8, 81)
	d := weightDeltaOf(a)
	d.BaseDigest = wds.Digest()
	if _, err := wds.Apply(d); err != nil {
		t.Fatal(err)
	}

	patchedF32 := hostF32Of(t, mustWeight(t, wds, weightDeltaTestTarget, compute.F32))
	want := quantizeQ8(append([]float32(nil), patchedF32...), 32, 32)

	tn, err := wds.Weight(weightDeltaTestTarget, compute.Q8_0)
	if err != nil {
		t.Fatalf("Weight(Q8_0, patched): %v", err)
	}
	hb, ok := tn.Buf().(compute.HostBuffer)
	if !ok {
		t.Fatalf("Q8_0 tensor has no host buffer: %T", tn.Buf())
	}
	if !int8sEqual(hb.I8(), want.q) {
		t.Fatalf("patched Q8_0 codes diverge from quantize(patched f32)")
	}
	scales := tn.Quant.Scale
	if len(scales) != len(want.d) {
		t.Fatalf("patched Q8_0 scale count = %d, want %d", len(scales), len(want.d))
	}
	for i := range scales {
		if scales[i] != want.d[i] {
			t.Fatalf("patched Q8_0 scale[%d] = %g, want %g", i, scales[i], want.d[i])
		}
	}

	// A patched name from a quantized store the delta layer cannot patch refuses.
	if _, err := wds.Weight(weightDeltaTestTarget, compute.Q4_K); !errors.Is(err, ErrPatchedWeightQuantStoreRefused) {
		t.Fatalf("patched name from Q4_K store: err=%v, want ErrPatchedWeightQuantStoreRefused", err)
	}
	// An UNPATCHED name from that store passes through to the base funnel's own
	// refusal (its contract, not this layer's).
	if _, err := wds.Weight("model.layers.0.self_attn.k_proj.weight", compute.Q4_K); err == nil {
		t.Fatalf("unpatched name from Q4_K store: expected the base funnel's refusal")
	}
}

// TestWeightEpochStampThreadsThroughKVBackendSeam pins the stack threading: the
// session's abi.KVBackend adapter exposes the live weight-state stamp, it moves with
// every committed transition, and a model without a delta layer reports generation 0
// with the base-inventory digest.
func TestWeightEpochStampThreadsThroughKVBackendSeam(t *testing.T) {
	m, wds, _ := weightDeltaFixture(t)
	m.SetWeightDeltaSource(wds)
	s := m.NewSession()
	kv, ok := KVBackend(s)
	if !ok {
		t.Fatal("KVBackend(session): ok=false")
	}
	ep, ok := kv.(interface{ WeightEpoch() WeightEpochStamp })
	if !ok {
		t.Fatalf("kvBackend does not expose the WeightEpoch seam (%T)", kv)
	}
	base := ep.WeightEpoch()
	if base.Generation != 0 || base.Digest != wds.Digest() {
		t.Fatalf("initial stamp = %+v, want generation 0 / the base digest", base)
	}

	a := newTestAdapter("math", weightDeltaTestTarget, 32, 32, 4, 8, 91)
	d := weightDeltaOf(a)
	d.BaseDigest = base.Digest
	if _, err := wds.Apply(d); err != nil {
		t.Fatal(err)
	}
	cur := ep.WeightEpoch()
	if cur.Generation != 1 || cur.Digest != wds.Digest() || cur.Digest == base.Digest {
		t.Fatalf("stamp after apply = %+v, want generation 1 / a new digest", cur)
	}

	// A model without a delta layer reports generation 0 + the base-state digest —
	// the same value an installed layer starts at, so pre-install spans match.
	m2 := NewSynthetic(loraTestCfg())
	st := m2.WeightEpoch()
	if st.Generation != 0 || st.Digest != baseStateDigest(baseInventoryFingerprint(m2)) {
		t.Fatalf("no-layer stamp = %+v, want generation 0 / the base-state digest", st)
	}
}
