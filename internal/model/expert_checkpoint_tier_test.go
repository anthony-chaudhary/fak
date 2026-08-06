package model

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"
)

// expert_checkpoint_tier_test.go — the R5 witnesses for #5616 (epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md).
//
// The rung's claim is a BYTE claim, so the tests are byte tests. A hit rate would not do: the number
// that decides whether a checkpoint whose expert bulk exceeds host RAM can be served at all is how
// many bytes one decode step reads against the E*stride a fully-resident load pays. So the central
// witness runs the same layer twice — once with every expert resident, once with the identical bytes
// laid out as fused checkpoint slabs and nothing resident — and asserts the outputs are bit-equal
// while the streamed arm reads exactly the activated k experts.
//
// The second load-bearing witness is the one that would catch the easy mistake: faulting eagerly.
// A tier that resolved a routed expert by reading it would still pay checkpoint IO on a ring HIT,
// which defeats R0 through R4 and makes this rung a regression dressed as a feature.

// expertCheckpointTestModel builds the SAME MoE as expertPrefetchModel, then MOVES the routed-expert
// projections out of the resident stores and into fused [E, H, H] checkpoint slabs — one per
// projection, experts contiguous, exactly the layout a GGUF `blk.L.ffn_*_exps.weight` tensor has.
// The bytes are the same bytes, which is the only reason the resident and streamed arms are
// comparable at all: any difference in the output is then attributable to residency, not to data.
//
// It returns the streamed model, its tier and one expert's byte stride (the unit both the ring budget
// and the byte-scaling assertions are stated in).
func expertCheckpointTestModel(t *testing.T, hidden, experts, topK int, hostBytes int64) (*Model, *ExpertCheckpointTier, int64) {
	t.Helper()
	resident := expertPrefetchModel(t, hidden, experts, topK)

	var blob []byte
	fused := make([]FusedExpertTensor, 0, 3)
	for _, proj := range []string{"gate", "up", "down"} {
		suffix := proj + "_proj.weight"
		offset := int64(len(blob))
		for e := 0; e < experts; e++ {
			blob = append(blob, resident.q4kw[expertName(0, e, suffix)].raw...)
		}
		fused = append(fused, FusedExpertTensor{
			Name:    "blk.0.ffn_" + proj + "_exps.weight",
			Layer:   0,
			Proj:    proj + "_proj",
			Quant:   ExpertCheckpointQ4K,
			Offset:  offset,
			Experts: experts,
			Rows:    hidden,
			Cols:    hidden,
		})
	}
	stride := int64(len(resident.q4kw[expertName(0, 0, "gate_proj.weight")].raw))

	// The router stays resident — it is one small dense weight every token reads, and the rung is
	// about the expert bulk. Nothing else is carried over, so a routed projection resolving at all
	// PROVES it came from the checkpoint.
	m := &Model{Cfg: resident.Cfg, q4kw: map[string]*q4kTensor{routerName(0): resident.q4kw[routerName(0)]}}
	tier := NewExpertCheckpointTier(hostBytes)
	if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), fused); err != nil {
		t.Fatalf("AddShard over a well-formed %d-byte checkpoint: %v", len(blob), err)
	}
	m.SetExpertCheckpoint(tier)
	return m, tier, stride
}

// TestExpertCheckpointReadsOnlyTheActivatedSetAndIsBitEqual is this rung's headline. Same layer, same
// input, same expert bytes; the resident arm holds all E experts in host RAM and the streamed arm
// holds none. The outputs must be bit-for-bit identical, and the streamed arm must have read exactly
// the k routed experts' three projections — k*3*stride, not the E*3*stride a fully-resident load
// materializes. On the GLM-5.2 shape this test miniaturizes (top-8 of 256) that ratio is the whole
// rung: 3% of the slab bytes instead of 100%.
func TestExpertCheckpointReadsOnlyTheActivatedSetAndIsBitEqual(t *testing.T) {
	const H, E, K = 256, 8, 4
	x := expertRingTestInput(H)

	resident := expertPrefetchModel(t, H, E, K)
	rs, _ := expertPrefetchSession(resident, 0) // no ring: the plain fully-resident path
	defer rs.Close()
	want := moeFFN{}.apply(resident, 0, x, sessionQ4KKernel{s: rs})

	m, tier, stride := expertCheckpointTestModel(t, H, E, K, 0)
	s, _ := expertPrefetchSession(m, stride*3*K) // a ring sized for the activated set
	defer s.Close()
	got := moeFFN{}.apply(m, 0, x, sessionQ4KKernel{s: s})

	if len(got) != len(want) {
		t.Fatalf("streamed layer output length %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("streaming the experts changed the layer output at %d: %v vs resident %v", i, got[i], want[i])
		}
	}

	st := tier.Stats()
	if !st.Enabled || st.Tensors != E*3 {
		t.Fatalf("tier indexes %d projections (enabled=%v), want %d", st.Tensors, st.Enabled, E*3)
	}
	if st.Failures != 0 {
		t.Fatalf("%d checkpoint reads failed (%s); the bit-equality above would be meaningless", st.Failures, st.LastError)
	}
	if st.Reads != 3*K {
		t.Fatalf("Reads=%d, want %d — the three projections of each of the %d routed experts, and nothing else",
			st.Reads, 3*K, K)
	}
	if st.BytesRead != int64(3*K)*stride {
		t.Fatalf("BytesRead=%d, want %d (k=%d experts x 3 projections x %d-byte stride)",
			st.BytesRead, int64(3*K)*stride, K, stride)
	}
	// The comparison that names the rung: what a fully-resident load of the same layer would cost.
	if slab := int64(3*E) * stride; st.BytesRead >= slab {
		t.Fatalf("BytesRead=%d did not beat the whole-slab cost %d; the tier read like the resident loader", st.BytesRead, slab)
	}
	// Streamed at the default budget means streamed: the expert bulk never accumulates in host RAM.
	if st.ResidentBytes != 0 || st.ResidentCount != 0 || st.PeakBytes != 0 {
		t.Fatalf("stream-through tier retained host bytes: %+v", st)
	}
}

// TestExpertCheckpointRingHitCostsNoCheckpointRead is the composition witness, and the one that
// catches the easy mistake. Resolving a routed expert must read NOTHING — the key, dtype and byte
// cost all come from the index — so a second decode step over the same activated set, served by the
// ring, must add zero reads and zero bytes. A tier that faulted at resolve time would pass every
// other test here and quietly make the bounded ring worthless.
func TestExpertCheckpointRingHitCostsNoCheckpointRead(t *testing.T) {
	const H, E, K = 256, 8, 4
	x := expertRingTestInput(H)

	m, tier, stride := expertCheckpointTestModel(t, H, E, K, 0)
	s, _ := expertPrefetchSession(m, stride*3*K)
	defer s.Close()

	first := moeFFN{}.apply(m, 0, x, sessionQ4KKernel{s: s})
	cold := tier.Stats()
	if cold.Reads == 0 {
		t.Fatal("the first step read nothing from the checkpoint; the forward never took the tier and this proves nothing")
	}

	second := moeFFN{}.apply(m, 0, x, sessionQ4KKernel{s: s})
	warm := tier.Stats()
	if warm.Reads != cold.Reads || warm.BytesRead != cold.BytesRead {
		t.Fatalf("a ring-resident activated set still faulted the checkpoint: reads %d->%d, bytes %d->%d",
			cold.Reads, warm.Reads, cold.BytesRead, warm.BytesRead)
	}
	if ring := s.ExpertRing(); ring.PageIns != 3*K {
		t.Fatalf("ring page-ins %d over two identical steps, want %d — one per projection of the activated set",
			ring.PageIns, 3*K)
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("the ring-served step disagreed with the faulted one at %d: %v vs %v", i, second[i], first[i])
		}
	}
}

// TestExpertCheckpointHostResidencyIsBounded is the rung's OTHER bound. R0 bounded device residency;
// nothing bounded host residency, and a tier that cached every expert it faulted would re-introduce
// exactly the unbounded host footprint this rung exists to remove. Faults are driven directly here
// rather than through a forward, because the property under test belongs to the tier: at ANY request
// order, retained bytes never exceed the declared budget, and the default budget retains nothing.
func TestExpertCheckpointHostResidencyIsBounded(t *testing.T) {
	const H, E, K = 256, 8, 4

	names := make([]string, 0, E*3)
	for e := 0; e < E; e++ {
		for _, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
			names = append(names, expertName(0, e, suffix))
		}
	}

	// Stream-through (the default): every expert is read, handed over and dropped.
	_, streamed, stride := expertCheckpointTestModel(t, H, E, K, 0)
	for _, name := range names {
		if _, err := streamed.fault(name); err != nil {
			t.Fatalf("fault %s: %v", name, err)
		}
	}
	st := streamed.Stats()
	if st.ResidentBytes != 0 || st.ResidentCount != 0 || st.PeakBytes != 0 {
		t.Fatalf("the stream-through default retained %d bytes in %d entries (peak %d); host residency must stay at zero",
			st.ResidentBytes, st.ResidentCount, st.PeakBytes)
	}
	if st.Reads != len(names) {
		t.Fatalf("Reads=%d over %d distinct experts, want one read each", st.Reads, len(names))
	}

	// A declared host budget makes the middle "pinned-host" rung real — bounded, never unbounded.
	// The request stream models what routing actually looks like: one recurring hot expert plus a
	// cold tail. A strict round-robin over all E experts would be LRU's pathological case and would
	// witness the bound while proving nothing about the cache above it.
	budget := stride * 4
	_, cached, _ := expertCheckpointTestModel(t, H, E, K, budget)
	hot := names[:3] // expert 0's three projections — the pick that recurs every step
	requests := 0
	for _, cold := range names[3:] {
		for _, name := range append(append([]string{}, hot...), cold) {
			requests++
			if _, err := cached.fault(name); err != nil {
				t.Fatalf("fault %s: %v", name, err)
			}
			if got := cached.Stats(); got.ResidentBytes > budget {
				t.Fatalf("after %s: resident %d exceeds budget %d", name, got.ResidentBytes, budget)
			}
		}
	}
	ct := cached.Stats()
	if ct.BudgetBytes != budget {
		t.Fatalf("BudgetBytes=%d, want the declared %d", ct.BudgetBytes, budget)
	}
	if ct.PeakBytes > budget {
		t.Fatalf("peak host residency %d exceeded budget %d", ct.PeakBytes, budget)
	}
	if ct.Evictions == 0 {
		t.Fatalf("no evictions over %d faults at a 4-projection budget; the bound was never exercised", requests)
	}
	if ct.Hits == 0 {
		t.Fatal("a retaining tier served no fault from its host cache; the recurring expert was re-read every step")
	}
	if ct.Reads+ct.Hits != requests {
		t.Fatalf("%d reads + %d hits != %d faults; the ledger is losing requests", ct.Reads, ct.Hits, requests)
	}
}

// errReaderAt fails every read, standing in for a checkpoint whose file went away mid-decode.
type errReaderAt struct{ err error }

func (r errReaderAt) ReadAt([]byte, int64) (int, error) { return 0, r.err }

// TestExpertCheckpointReadFailureIsCountedNotSwallowed guards the diagnosis. A fault that fails must
// surface: over a streamed checkpoint the weight is resident nowhere else, so silently reporting the
// tensor "absent" would send the caller down a missing-weight path with the real IO error buried. It
// must also be COUNTED, or an intermittently failing disk would be indistinguishable from a
// checkpoint that simply does not carry the tensor.
func TestExpertCheckpointReadFailureIsCountedNotSwallowed(t *testing.T) {
	const H, E = 256, 4
	boom := errors.New("checkpoint shard vanished")
	tier := NewExpertCheckpointTier(0)
	err := tier.AddShard(errReaderAt{err: boom}, int64(E*H*(H/qkK)*q4kBlockBytes), []FusedExpertTensor{{
		Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj",
		Quant: ExpertCheckpointQ4K, Offset: 0, Experts: E, Rows: H, Cols: H,
	}})
	if err != nil {
		t.Fatalf("AddShard performs no payload IO, so a failing reader must still index cleanly: %v", err)
	}

	name := expertName(0, 1, "gate_proj.weight")
	if _, faultErr := tier.fault(name); !errors.Is(faultErr, boom) {
		t.Fatalf("fault error = %v, want the underlying %v", faultErr, boom)
	}
	// The staging path has no soft fallback: it raises, matching the uniform missing-weight panic.
	ck, ok := tier.staging(name)
	if !ok {
		t.Fatal("staging declined a name the tier indexes")
	}
	func() {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("a failed checkpoint read was swallowed; the staging returned a weight it never read")
			}
			if msg := fmt.Sprint(r); !strings.Contains(msg, name) || !strings.Contains(msg, boom.Error()) {
				t.Fatalf("panic %q names neither the tensor nor the cause", msg)
			}
		}()
		ck.mk()
	}()

	st := tier.Stats()
	if st.Failures != 2 {
		t.Fatalf("Failures=%d after two failing reads, want 2", st.Failures)
	}
	if !strings.Contains(st.LastError, boom.Error()) {
		t.Fatalf("LastError=%q does not carry the cause", st.LastError)
	}
	if st.Reads != 0 || st.BytesRead != 0 {
		t.Fatalf("a failed read was counted as %d reads / %d bytes; the byte ledger must count bytes that landed",
			st.Reads, st.BytesRead)
	}
}

// TestExpertCheckpointRefusesMalformedLayoutAtLoad pins WHERE a bad layout is refused. The tier's
// stride math assumes experts are contiguous, equal-stride, unpadded segments; a description that
// breaks that assumption must fail at AddShard, before any decode, rather than reading misaligned
// bytes at token time and producing plausible garbage. A refusal must also leave the tier exactly as
// it was found — a half-indexed tier would answer some experts and not others.
func TestExpertCheckpointRefusesMalformedLayoutAtLoad(t *testing.T) {
	const H, E = 256, 4
	stride := int64(H * (H / qkK) * q4kBlockBytes)
	good := FusedExpertTensor{
		Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj",
		Quant: ExpertCheckpointQ4K, Offset: 0, Experts: E, Rows: H, Cols: H,
	}
	size := stride * int64(E)

	mutate := func(f func(*FusedExpertTensor)) []FusedExpertTensor {
		d := good
		f(&d)
		return []FusedExpertTensor{d}
	}
	cases := []struct {
		name  string
		fused []FusedExpertTensor
	}{
		{"slab runs past the shard", mutate(func(d *FusedExpertTensor) { d.Offset = stride })},
		{"reduction dim is not whole quant blocks", mutate(func(d *FusedExpertTensor) { d.Cols = H + 1 })},
		{"no experts", mutate(func(d *FusedExpertTensor) { d.Experts = 0 })},
		{"negative layer", mutate(func(d *FusedExpertTensor) { d.Layer = -1 })},
		{"no projection", mutate(func(d *FusedExpertTensor) { d.Proj = "" })},
		{"unstageable representation", mutate(func(d *FusedExpertTensor) { d.Quant = ExpertCheckpointQuant(99) })},
		{"two tensors claim one projection", []FusedExpertTensor{good, {
			Name: "blk.0.ffn_gate_exps.dup", Layer: 0, Proj: "gate_proj",
			Quant: ExpertCheckpointQ4K, Offset: 0, Experts: E, Rows: H, Cols: H,
		}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tier := NewExpertCheckpointTier(0)
			if err := tier.AddShard(bytes.NewReader(make([]byte, size)), size, tc.fused); err == nil {
				t.Fatal("AddShard accepted a layout the stride math cannot serve")
			}
			if st := tier.Stats(); st.Tensors != 0 {
				t.Fatalf("a refused shard left %d projections indexed; the refusal must be atomic", st.Tensors)
			}
		})
	}

	// The control: the same description, unmutated, is accepted and indexes every expert.
	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(bytes.NewReader(make([]byte, size)), size, []FusedExpertTensor{good}); err != nil {
		t.Fatalf("the well-formed control was refused: %v", err)
	}
	if st := tier.Stats(); st.Tensors != E {
		t.Fatalf("indexed %d projections, want %d", st.Tensors, E)
	}
}

// TestExpertCheckpointIsInertWithoutATier is the default-off gate. A model that never had a tier —
// every model shipping today — must resolve and report exactly as it did before: the nil tier is a
// valid "no tier" on every method, and the fully-resident forward is untouched.
func TestExpertCheckpointIsInertWithoutATier(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	if st := m.ExpertCheckpointStats(); st.Enabled || st.Tensors != 0 || st.Reads != 0 {
		t.Fatalf("a model with no checkpoint tier reported activity: %+v", st)
	}

	s, _ := expertPrefetchSession(m, 0)
	defer s.Close()
	// A resident expert resolves from the resident store, and an absent one declines rather than
	// reaching a tier that is not there.
	if w, ok := s.resolveExpertWeight(expertName(0, 0, "gate_proj.weight")); !ok || w.q4 == nil || w.ck != nil {
		t.Fatalf("resident expert resolved to %+v (ok=%v); it must come from the resident store", w, ok)
	}
	if _, ok := s.resolveExpertWeight(expertName(0, E+1, "gate_proj.weight")); ok {
		t.Fatal("an expert no store carries resolved anyway")
	}

	// And the streamed model's tier answers the same three-projection question the demand path asks.
	streamed, _, _ := expertCheckpointTestModel(t, H, E, K, 0)
	ss, _ := expertPrefetchSession(streamed, 0)
	defer ss.Close()
	w, ok := ss.resolveExpertWeight(expertName(0, 0, "gate_proj.weight"))
	if !ok || w.ck == nil {
		t.Fatalf("a checkpoint-only expert resolved to %+v (ok=%v); it must come from the tier", w, ok)
	}
	if w.halKey() != "q4k:"+expertName(0, 0, "gate_proj.weight") {
		t.Fatalf("checkpoint staging key %q differs from the resident rule; a faulted and a resident copy "+
			"of one projection would become two ring entries", w.halKey())
	}
	if streamed.ExpertCheckpointStats().Reads != 0 {
		t.Fatal("resolving a checkpoint expert read bytes; resolution must be index-only")
	}
}
