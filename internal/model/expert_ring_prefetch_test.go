package model

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_ring_prefetch_test.go — the R3 witnesses for #5614 (epic #5606).
//
// The plan's stated witness for this rung is "fraction of activated-expert page-in latency
// overlapped with compute", and that number is NOT produced here: compute.Backend exposes Upload as
// the only host->device path and no stream or event handle, so there is nothing to measure an
// overlap against on the cpu-ref backend these tests run on. What IS witnessed is everything the
// overlap would rest on — the set is issued BEFORE the first GEMM, it costs zero extra page-ins, it
// stays inside the budget, it never promotes a weight to permanent residency, and it leaves R2's
// histogram and R4's replay trace byte-for-byte unperturbed. See the file header of
// expert_ring_prefetch.go for the missing wiring and TestExpertRingPrefetchIsIssuedBeforeAnyGEMM for
// the ordering claim that an async Upload would turn into real overlap.

// expertOrderRecordingBackend records the ORDER of uploads and matmuls, not just their counts. The
// prefetch's whole claim is about ordering — the counts are identical either way — so a counting
// backend cannot witness it.
type expertOrderRecordingBackend struct {
	compute.Backend
	events []string // "upload" / "matmul", in the order the forward issued them
}

func (b *expertOrderRecordingBackend) Name() string                     { return "cuda-test-ordered" }
func (b *expertOrderRecordingBackend) SupportsRoutedExpertKQuant() bool { return true }
func (b *expertOrderRecordingBackend) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: true, UploadDtype: true}
}

func (b *expertOrderRecordingBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	// Only weight-class uploads matter here; the per-token activation upload is an F32 vector that
	// every path issues identically, so counting it would drown the signal.
	if as != compute.F32 {
		b.events = append(b.events, "upload")
	}
	return b.Backend.Upload(t, as)
}

func (b *expertOrderRecordingBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.events = append(b.events, "matmul")
	return b.Backend.MatMul(w, x)
}

// expertPrefetchModel is a routed-expert MoE with a working Q4_K router, so a test can drive the
// REAL moeFFN.apply seam rather than calling the prefetch directly. Every expert projection is the
// same shape, which is what makes "the prefix of the set that fits" a statement about expert COUNT.
func expertPrefetchModel(t *testing.T, hidden, experts, topK int) *Model {
	t.Helper()
	cfg := expertHALTestConfig(hidden)
	cfg.NumExperts = experts
	cfg.NumExpertsPerTok = topK
	m := &Model{Cfg: cfg, q4kw: map[string]*q4kTensor{}}
	m.q4kw[routerName(0)] = &q4kTensor{out: experts, in: hidden, nblk: hidden / qkK, raw: buildRawQ4K(t, experts, hidden, 7)}
	for e := 0; e < experts; e++ {
		for i, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
			name := expertName(0, e, suffix)
			m.q4kw[name] = &q4kTensor{out: hidden, in: hidden, nblk: hidden / qkK, raw: buildRawQ4K(t, hidden, hidden, 101+e*3+i)}
		}
	}
	return m
}

// expertPrefetchSession pairs that model with an order-recording device session at a ring budget.
func expertPrefetchSession(m *Model, ringBytes int64) (*Session, *expertOrderRecordingBackend) {
	be := &expertOrderRecordingBackend{Backend: compute.Default()}
	return &Session{
		M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{},
		ExpertRingBytes: ringBytes,
	}, be
}

// TestExpertRingPrefetchIsIssuedBeforeAnyGEMM is the load-bearing ordering witness: on a layer whose
// activated set fits, EVERY weight upload is issued before the FIRST expert GEMM. That is the
// property an async Upload would turn into real overlap, and it is exactly what the pre-R3 path
// could not do — it interleaved upload, GEMM, upload, GEMM, so nothing could ever be in flight.
func TestExpertRingPrefetchIsIssuedBeforeAnyGEMM(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)

	s, be := expertPrefetchSession(m, perWeight*3*K) // the whole activated set fits
	defer s.Close()
	moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})

	firstMatMul, uploadsBefore, uploadsAfter := -1, 0, 0
	for i, ev := range be.events {
		switch {
		case ev == "matmul" && firstMatMul < 0:
			firstMatMul = i
		case ev == "upload" && firstMatMul < 0:
			uploadsBefore++
		case ev == "upload":
			uploadsAfter++
		}
	}
	if firstMatMul < 0 {
		t.Fatal("no GEMM ran; the forward never reached the expert path and this witness proves nothing")
	}
	if uploadsBefore != 3*K {
		t.Fatalf("%d weight uploads before the first GEMM, want all %d of the activated set's projections",
			uploadsBefore, 3*K)
	}
	if uploadsAfter != 0 {
		t.Fatalf("%d weight uploads were still issued AFTER a GEMM had started; the set was not fully prefetched", uploadsAfter)
	}

	st := s.ExpertRing()
	if st.ActivatedExperts != K || st.ActivatedCovered != K {
		t.Fatalf("coverage %d/%d, want the whole top-%d covered at a budget sized for it",
			st.ActivatedCovered, st.ActivatedExperts, K)
	}
	if st.Prefetched != 3*K {
		t.Fatalf("Prefetched=%d, want %d weights staged ahead of their GEMM", st.Prefetched, 3*K)
	}
}

// TestExpertRingPrefetchCostsNoExtraPageIns is the "prefetch is free" claim, and simultaneously the
// guard on ringStaging agreeing with hal.go: a prefetched weight must be the SAME ring resident the
// demand path then finds. If the keys, dtypes or sizes ever drift apart, the demand staging misses
// and page-ins double — which is precisely what this asserts cannot happen.
func TestExpertRingPrefetchCostsNoExtraPageIns(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 3 * K
	x := expertRingTestInput(H)

	// Same layer, same input, same budget — the only difference is whether the set was declared.
	with, _ := expertPrefetchSession(m, budget)
	defer with.Close()
	deltaWith := moeFFN{}.apply(m, 0, x, sessionQ4KKernel{s: with})

	without, _ := expertPrefetchSession(m, budget)
	defer without.Close()
	without.ExpertPrefetch = ExpertPrefetchOnDemand
	deltaWithout := moeFFN{}.apply(m, 0, x, sessionQ4KKernel{s: without})

	a, b := with.ExpertRing(), without.ExpertRing()
	if a.PageIns != b.PageIns {
		t.Fatalf("prefetch changed page-ins: %d with vs %d without — a prefetch must move uploads earlier, never add them",
			a.PageIns, b.PageIns)
	}
	if a.Evictions != 0 || b.Evictions != 0 {
		t.Fatalf("a budget sized for the whole set evicted (%d with / %d without)", a.Evictions, b.Evictions)
	}
	if a.Hits <= b.Hits {
		t.Fatalf("prefetched weights were not hits on the demand path: %d vs %d hits", a.Hits, b.Hits)
	}
	// Load-bearing: a prefetch that changes the arithmetic is a bug, not an optimization.
	if len(deltaWith) != len(deltaWithout) {
		t.Fatalf("delta length %d vs %d", len(deltaWith), len(deltaWithout))
	}
	for i := range deltaWith {
		if deltaWith[i] != deltaWithout[i] {
			t.Fatalf("prefetch changed the layer output at %d: %v vs %v", i, deltaWith[i], deltaWithout[i])
		}
	}
}

// TestExpertRingPrefetchStopsAtTheBudget is the anti-thrash rule. A prefetcher that runs past its
// budget evicts what it just fetched — paying the upload and throwing it away. Only the prefix that
// fits may be staged, the bound must hold, and the coverage meter must report the shortfall rather
// than quietly claiming the whole set.
func TestExpertRingPrefetchStopsAtTheBudget(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 3 * 2 // room for two of the four activated experts

	s, _ := expertPrefetchSession(m, budget)
	defer s.Close()
	moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})

	st := s.ExpertRing()
	if st.ActivatedExperts != K {
		t.Fatalf("ActivatedExperts=%d, want the whole top-%d the router asked for — the meter must count what was WANTED", st.ActivatedExperts, K)
	}
	if st.ActivatedCovered != 2 {
		t.Fatalf("ActivatedCovered=%d, want the 2 experts the budget can hold", st.ActivatedCovered)
	}
	if st.Prefetched != 3*2 {
		t.Fatalf("Prefetched=%d, want exactly the %d projections of the fitting prefix", st.Prefetched, 3*2)
	}
	if st.PeakBytes > st.BudgetBytes {
		t.Fatalf("peak resident %d exceeds budget %d", st.PeakBytes, st.BudgetBytes)
	}
}

// TestExpertRingPrefetchLeavesTheEvidenceStreamsAlone protects R2 and R4 from R3. A prefetch is a
// HINT: it must not appear in the usage histogram the pin-set learns from, nor in the ordered trace
// the victim-policy gate replays, nor earn heat in the live ranking. A policy trained on its own
// prefetcher's guesses is self-confirming.
func TestExpertRingPrefetchLeavesTheEvidenceStreamsAlone(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 3 * K
	x := expertRingTestInput(H)

	run := func(prefetch bool) (ExpertAccessTrace, map[int]int, int) {
		s, _ := expertPrefetchSession(m, budget)
		defer s.Close()
		s.ExpertRingEvict = ExpertRingEvictValueAware
		if !prefetch {
			s.ExpertPrefetch = ExpertPrefetchOnDemand
		}
		moeFFN{}.apply(m, 0, x, sessionQ4KKernel{s: s})
		heat := map[int]int{}
		for id, h := range s.expertRing.heat {
			if layer, expert, ok := routedExpertIdentity(ringKeyTensorName(string(id))); ok && layer == 0 {
				heat[expert] += h
			}
		}
		return s.ExpertRingTrace(), heat, s.expertRing.accesses
	}

	withTrace, withHeat, withAccesses := run(true)
	withoutTrace, withoutHeat, withoutAccesses := run(false)

	if len(withTrace.Events) != len(withoutTrace.Events) {
		t.Fatalf("prefetch changed the replay trace length: %d vs %d — R4's gauge would score a stream the workload never produced",
			len(withTrace.Events), len(withoutTrace.Events))
	}
	for i := range withTrace.Events {
		if withTrace.Events[i] != withoutTrace.Events[i] {
			t.Fatalf("trace event %d differs: %+v vs %+v", i, withTrace.Events[i], withoutTrace.Events[i])
		}
	}
	// The policy-ACCESS count is the load-bearing comparison. Summed heat is not: the decay pass
	// (heat >>= 1 every expertRingDecayEveryAccesses) can halve a doubled count straight back to the
	// original total, so a prefetch that wrongly earned heat can hide inside an equal sum. accesses is
	// the raw driver of that cadence and cannot be cancelled.
	if withoutAccesses == 0 {
		t.Fatal("the on-demand arm recorded no policy accesses at all; this comparison is vacuous")
	}
	if withAccesses != withoutAccesses {
		t.Fatalf("prefetch changed the ring's policy-access count: %d with vs %d without — a hint must not be ranked on, "+
			"or the victim policy protects what was speculated instead of what was read",
			withAccesses, withoutAccesses)
	}
	if len(withHeat) != len(withoutHeat) {
		t.Fatalf("prefetch changed which experts carry heat: %v vs %v", withHeat, withoutHeat)
	}
	for e, h := range withoutHeat {
		if withHeat[e] != h {
			t.Fatalf("expert %d earned heat %d with prefetch vs %d without; a hint must not be ranked on", e, withHeat[e], h)
		}
	}
}

// TestExpertRingPrefetchIsInertWithoutARing is the default-off gate: a session that declared no ring
// budget must take the pre-R3 path byte-for-byte, allocating and counting nothing.
func TestExpertRingPrefetchIsInertWithoutARing(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	x := expertRingTestInput(H)

	s, be := expertPrefetchSession(m, 0) // no ring
	defer s.Close()
	got := moeFFN{}.apply(m, 0, x, sessionQ4KKernel{s: s})

	if s.expertRing != nil {
		t.Fatal("a session with no declared budget built a ring; the prefetch must not be what turns the feature on")
	}
	if st := s.ExpertRing(); st.Enabled || st.Prefetched != 0 || st.ActivatedExperts != 0 {
		t.Fatalf("inert session reported ring activity: %+v", st)
	}
	// The unbounded halW path interleaves upload and GEMM exactly as it always did.
	if len(be.events) == 0 || be.events[0] != "upload" {
		t.Fatalf("unexpected event stream head %v", be.events)
	}
	// And the ring+prefetch path computes the identical layer output, so nothing on this rung is
	// paid for with arithmetic.
	ringed, _ := expertPrefetchSession(m, expertRingWeightBytes(t, m)*3*K)
	defer ringed.Close()
	want := moeFFN{}.apply(m, 0, x, sessionQ4KKernel{s: ringed})
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("the prefetched ring changed the layer output at %d: %v vs %v", i, want[i], got[i])
		}
	}
}

// ringKeyTensorName strips the dtype prefix a ring key carries ("q4k:", "kquant-raw:") back to the
// canonical tensor name, so a test can ask which expert a resident belongs to.
func ringKeyTensorName(key string) string {
	if i := strings.IndexByte(key, ':'); i >= 0 {
		return key[i+1:]
	}
	return key
}

// ---- oss-port-strata-layer-major-prefill leaf 1 (prefill warm hook) ----
// Closed contract for the packed per-layer expert warm hook, kept in this file to
// honor the ticket scope (allowed test file: expert_ring_prefetch_test.go).

// batch_prefill_test.go — closed contract for oss-port-strata-layer-major-prefill leaf 1
// (prefill-warm-hook): the packed rectangular prefill walk carries a session-flag-gated
// per-layer expert warm hook at the entry of BOTH packed layer loops (F32 and Q8 lanes).
// The flag's zero value keeps the walk byte-for-byte unchanged (rejection probe); armed,
// the hook fires exactly once per layer index, in layer order 0..NumLayers-1, counted
// through the fake hook seam — one pass per layer for the whole panel, never once per
// prompt/chunk. Software witness only ([SW-VERIFIED]); no throughput or hardware claim.

// layerMajorTestConfig is a tiny dense PreNorm geometry that routes to BOTH packed
// rectangular lanes (the MoE axis is excluded by batchPreNormFastPathOK, which is the
// point: the hook pins the layer-major discipline these lanes owe their expert reads).
func layerMajorTestConfig() Config {
	return Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
}

func layerMajorPrompts(B, P, V int) [][]int {
	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		p := make([]int, P)
		for t := range p {
			p[t] = (b*41 + t*17 + 5) % V
		}
		prompts[b] = p
	}
	return prompts
}

// assertBitIdenticalLogits pins []float32 rows byte-for-byte (math.Float32bits), the
// same bar TestPrefillEachRectangularMatchesSerial holds the packed lane to the serial
// per-token walk.
func assertBitIdenticalLogits(t *testing.T, got, want [][]float32, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d logit rows, want %d", label, len(got), len(want))
	}
	for b := range got {
		if len(got[b]) != len(want[b]) {
			t.Fatalf("%s row %d: width %d, want %d", label, b, len(got[b]), len(want[b]))
		}
		for i := range got[b] {
			if math.Float32bits(got[b][i]) != math.Float32bits(want[b][i]) {
				t.Fatalf("%s row %d logit[%d]: %v != %v", label, b, i, got[b][i], want[b][i])
			}
		}
	}
}

// TestLayerMajorPrefillExpertWarm pins the leaf-1 closed contract on both packed lanes:
//   - flag off (zero value): zero warm hook fires even with a hook installed, and the
//     F32 lane's logits stay bit-identical to the independent serial per-token walk —
//     the disabled arm retains the current walk visibly (rejection probe);
//   - flag on: exactly one warm hook fire per layer index per prefill call, in layer
//     order 0..NumLayers-1, counted through the fake hook seam;
//   - one-pass-per-layer: a two-prompt panel warms layer L's expert set once, not once
//     per prompt/chunk (total fires == NumLayers, each layer exactly once across the
//     panel's sessions).
func TestLayerMajorPrefillExpertWarm(t *testing.T) {
	cfg := layerMajorTestConfig()
	const B, P = 2, 4
	V := cfg.VocabSize
	prompts := layerMajorPrompts(B, P, V)

	// Premise: this geometry routes to both packed rectangular lanes, so the test is
	// not vacuous.
	if !batchRectFastPathOK(cfg, false) {
		t.Fatal("premise broken: config no longer routes to prefillEachPackedF32")
	}
	if !batchRectFastPathOK(cfg, true) {
		t.Fatal("premise broken: config no longer routes to prefillEachPackedQ")
	}
	if got, ok := rectangularPrefillLen(prompts); !ok || got != P {
		t.Fatalf("premise broken: prompts not rectangular P=%d ok=%v", got, ok)
	}

	t.Run("f32", func(t *testing.T) {
		m := NewSynthetic(cfg)

		// Independent baseline: the serial per-token walk ("today's output").
		refLogits := make([][]float32, B)
		for b := 0; b < B; b++ {
			refLogits[b] = m.NewSession().Prefill(prompts[b])
		}

		// Flag OFF with a hook installed: the flag, not the hook, gates the fire.
		var offFires []int
		offHook := func(l int) { offFires = append(offFires, l) }
		bsOff := m.NewBatchSession(B)
		for _, s := range bsOff.Seqs {
			s.SetPrefillExpertWarm(false, offHook)
		}
		gotOff := bsOff.PrefillEach(prompts)
		if len(offFires) != 0 {
			t.Fatalf("flag off: %d warm hook fires, want 0 (fires=%v)", len(offFires), offFires)
		}
		assertBitIdenticalLogits(t, gotOff, refLogits, "flag-off f32 lane")

		// Flag ON: one fire per layer index, in layer order, once per panel.
		fires := make([][]int, B) // per-session counters: no session may double-warm
		bsOn := m.NewBatchSession(B)
		for b, s := range bsOn.Seqs {
			b := b
			s.SetPrefillExpertWarm(true, func(l int) { fires[b] = append(fires[b], l) })
		}
		gotOn := bsOn.PrefillEach(prompts)

		total := 0
		for b := range fires {
			total += len(fires[b])
		}
		if total != cfg.NumLayers {
			t.Fatalf("flag on: %d total warm hook fires for a %d-prompt panel, want %d (one pass per layer, not per prompt; fires=%v)",
				total, B, cfg.NumLayers, fires)
		}
		seen := map[int]int{}
		for b := range fires {
			for _, l := range fires[b] {
				seen[l]++
			}
		}
		for l := 0; l < cfg.NumLayers; l++ {
			if seen[l] != 1 {
				t.Fatalf("layer %d: %d warm hook fires across the panel, want exactly 1 (fires=%v)", l, seen[l], fires)
			}
		}
		// Layer order 0..NumLayers-1: the panel's combined fire sequence is ascending
		// and contiguous.
		combined := make([]int, 0, cfg.NumLayers)
		for l := 0; l < cfg.NumLayers; l++ {
			for b := range fires {
				for _, fl := range fires[b] {
					if fl == l {
						combined = append(combined, fl)
					}
				}
			}
		}
		for i, l := range combined {
			if l != i {
				t.Fatalf("warm hook fire %d is layer %d, want %d (layer order broken; fires=%v)", i, l, i, fires)
			}
		}
		// Arming changes nothing numerically.
		assertBitIdenticalLogits(t, gotOn, gotOff, "flag-on f32 lane")
	})

	t.Run("q8", func(t *testing.T) {
		m := NewSynthetic(cfg)
		m.Quantize()

		// Pure zero-value default: no arm installed at all.
		bsDefault := m.NewBatchSession(B)
		bsDefault.SetQuant(true)
		gotDefault := bsDefault.PrefillEach(prompts)
		if len(bsDefault.Seqs) != B {
			t.Fatalf("q8 default arm returned %d sessions, want %d", len(bsDefault.Seqs), B)
		}

		// Flag ON through the quantized twin.
		var onFires []int
		bsOn := m.NewBatchSession(B)
		bsOn.SetQuant(true)
		for _, s := range bsOn.Seqs {
			s.SetPrefillExpertWarm(true, func(l int) { onFires = append(onFires, l) })
		}
		gotOn := bsOn.PrefillEach(prompts)

		if len(onFires) != cfg.NumLayers {
			t.Fatalf("q8 flag on: %d warm hook fires, want %d (one pass per layer; fires=%v)",
				len(onFires), cfg.NumLayers, onFires)
		}
		for i, l := range onFires {
			if l != i {
				t.Fatalf("q8 warm hook fire %d is layer %d, want %d (layer order broken; fires=%v)", i, l, i, onFires)
			}
		}
		// The gate preserves the quantized lane byte-for-byte: the flag-off run IS
		// today's walk (the arm adds no arithmetic when un-armed), so flag-on must
		// equal it bit-for-bit.
		assertBitIdenticalLogits(t, gotOn, gotDefault, "q8 lane")
	})
}
