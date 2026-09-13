package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_readahead_test.go ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the cross-layer gate-prefetch witnesses (#5614 lineage, epic #5606,
// expert_readahead.go). The feature predicts layer L+1's top-k routed experts by applying layer
// L+1's ROUTER GATE (gate path only, no expert GEMM) to layer L's hidden state, stages them into
// the routed-expert ring as HINTS during layer L's compute, and records precision/recall against
// layer L+1's realized activation. Default OFF; a mispredict must NEVER change outputs.
//
// The tests drive the predictor/stager/recorder directly (the moe.go call sites are owned by a
// separate lane and are not wired here), which is exactly the seam a caller would use:
//
//	predicted := predictNextLayerExperts(m, L, xn, mat)
//	s.prefetchNextLayerGateExperts(L, predicted)   // stage L+1's set as hints during L
//	// ... layer L+1 computes, its router names the realized set ...
//	s.recordNextLayerActual(L+1, actual)

// expertCrossLayerModel is a TWO-layer routed-expert MoE with a working Q4_K router on each
// layer, so the real router->top-k->expert dataflow runs on layer 0 and the cross-layer predictor
// can apply layer 1's gate to layer 0's hidden state. Both layers share the SAME router weights
// and the SAME per-layer expert template (same seeds), which makes the prediction deterministic:
// gate_{L+1}(x_L) selects the same top-k that layer L+1's own route then realizes, so Hits>0 is a
// construction rather than a hope. Every expert projection is [H,H] so a ring budget is stated in
// whole experts via expertRingWeightBytes.
func expertCrossLayerModel(t *testing.T, hidden, experts, topK int) *Model {
	t.Helper()
	cfg := expertHALTestConfig(hidden)
	cfg.NumLayers = 2
	cfg.NumExperts = experts
	cfg.NumExpertsPerTok = topK
	m := &Model{Cfg: cfg, q4kw: map[string]*q4kTensor{}}
	for l := 0; l < 2; l++ {
		// Identical router bytes on both layers => identical gate geometry.
		m.q4kw[routerName(l)] = &q4kTensor{out: experts, in: hidden, nblk: hidden / qkK, raw: buildRawQ4K(t, experts, hidden, 7)}
		for e := 0; e < experts; e++ {
			for i, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
				name := expertName(l, e, suffix)
				m.q4kw[name] = &q4kTensor{out: hidden, in: hidden, nblk: hidden / qkK, raw: buildRawQ4K(t, hidden, hidden, 101+e*3+i)}
			}
		}
	}
	return m
}

// expertCrossLayerSession pairs that model with a device-capable Q4_K session at a ring budget,
// using the order-recording backend so the ordering witness can read upload/GEMM interleaving.
func expertCrossLayerSession(m *Model, ringBytes int64) (*Session, *expertOrderRecordingBackend) {
	be := &expertOrderRecordingBackend{Backend: compute.Default()}
	return &Session{
		M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{},
		ExpertRingBytes: ringBytes,
	}, be
}

// expertIDs extracts the expert ids from a pick list, for diagnostics.
func expertIDs(picks []routePick) []int {
	out := make([]int, len(picks))
	for i, p := range picks {
		out[i] = p.expert
	}
	return out
}

// TestCrossLayerGatePrefetchRecordsPrecisionRecall builds a two-layer MoE whose layers share
// their router geometry, enables the prefetch, drives layer 0's forward plus the
// predictor/stager/recorder for layer 1, and asserts the precision/recall meter is well-defined
// and populated. The fixture is constructed so the prediction matches the realization (identical
// routers), so the test requires Hits>0 rather than merely accepting a well-defined zero.
func TestCrossLayerGatePrefetchRecordsPrecisionRecall(t *testing.T) {
	const H, E, K = 256, 8, 2
	m := expertCrossLayerModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)

	s, _ := expertCrossLayerSession(m, perWeight*3*E) // holds the whole activated set
	defer s.Close()

	SetCrossLayerGatePrefetch(true)
	defer SetCrossLayerGatePrefetch(false)

	x := expertRingTestInput(H)
	mat := sessionQ4KKernel{s: s}

	// Layer 0 compute, then predict layer 1's experts from layer 0's hidden state, stage them
	// as hints, and record layer 1's realized activation (its own router's picks).
	moeFFN{}.apply(m, 0, x, mat)
	predicted := predictNextLayerExperts(m, 0, x, mat)
	if len(predicted) == 0 {
		t.Fatal("prediction came back empty with the gate ON and a 2-layer model; the witness is vacuous")
	}
	s.prefetchNextLayerGateExperts(0, predicted)
	actual := route(m, 1, x, mat)
	s.recordNextLayerActual(1, actual)

	st := CrossLayerPrefetchStatsFor(s)
	if st.Predicted <= 0 {
		t.Fatalf("Predicted=%d after %d predicted picks; the meter never counted a prediction", st.Predicted, len(predicted))
	}
	if st.Actual <= 0 {
		t.Fatalf("Actual=%d after %d realized picks; the meter never counted a realization", st.Actual, len(actual))
	}
	if p := st.Precision(); p < 0 || p > 1 {
		t.Fatalf("Precision()=%v out of [0,1]", p)
	}
	if r := st.Recall(); r < 0 || r > 1 {
		t.Fatalf("Recall()=%v out of [0,1]", r)
	}
	// Match-quality construction: both layers carry the identical router, so the prediction and
	// the realization overlap. Exact equality is NOT guaranteed because layer 1's router consumes
	// layer 0's OUTPUT residual, not layer 0's input that the predictor saw; but the shared gate
	// geometry makes an overlap (Hits>0) reliable. If it proves flaky on this fixture, the ledger's
	// well-definedness (Predicted==Actual==K, ratios in [0,1]) is still the contracted invariant.
	if st.Hits <= 0 {
		t.Fatalf("Hits=%d: identical layer routers should overlap the same top-k; predicted=%v actual=%v",
			st.Hits, expertIDs(predicted), expertIDs(actual))
	}
	if st.Predicted != len(predicted) || st.Actual != len(actual) {
		t.Fatalf("ledger totals predicted/actual=%d/%d, want %d/%d (expert-slot accounting)",
			st.Predicted, st.Actual, len(predicted), len(actual))
	}
	t.Logf("cross-layer gate prefetch: predicted=%d actual=%d hits=%d precision=%.2f recall=%.2f",
		st.Predicted, st.Actual, st.Hits, st.Precision(), st.Recall())
}

// TestCrossLayerGatePrefetchMispredictDoesNotChangeOutputs is the load-bearing equivalence
// witness: at a ring budget sized to hold the whole activated set, a layer's output with the
// cross-layer prefetch OFF is BIT-IDENTICAL to the output with it ON. A mispredict (or a correct
// prediction) is a HINT; if it ever moved logits, the feature would be unsafe.
func TestCrossLayerGatePrefetchMispredictDoesNotChangeOutputs(t *testing.T) {
	const H, E, K = 256, 8, 2
	m := expertCrossLayerModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 3 * E
	x := expertRingTestInput(H)

	// OFF arm: default gate is false; run layers 0 and 1 plainly and capture layer 1's output.
	SetCrossLayerGatePrefetch(false)
	off, _ := expertCrossLayerSession(m, budget)
	defer off.Close()
	matOff := sessionQ4KKernel{s: off}
	moeFFN{}.apply(m, 0, x, matOff)
	offOut := moeFFN{}.apply(m, 1, x, matOff)

	// ON arm: same forwards, with the cross-layer predictor/stager/recorder engaged on layer 1.
	on, _ := expertCrossLayerSession(m, budget)
	defer on.Close()
	matOn := sessionQ4KKernel{s: on}
	SetCrossLayerGatePrefetch(true)
	defer SetCrossLayerGatePrefetch(false)
	moeFFN{}.apply(m, 0, x, matOn)
	predicted := predictNextLayerExperts(m, 0, x, matOn)
	on.prefetchNextLayerGateExperts(0, predicted)
	onOut := moeFFN{}.apply(m, 1, x, matOn)
	on.recordNextLayerActual(1, route(m, 1, x, matOn))

	// Compare the layer-1 outputs element-by-element as raw float32 bits.
	if len(offOut) != len(onOut) {
		t.Fatalf("layer-1 output length %d with prefetch ON vs %d OFF", len(onOut), len(offOut))
	}
	for i := range offOut {
		if math.Float32bits(offOut[i]) != math.Float32bits(onOut[i]) {
			t.Fatalf("cross-layer prefetch changed the layer output at %d: off=%v (%#x) on=%v (%#x) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â a mispredict must never alter arithmetic",
				i, offOut[i], math.Float32bits(offOut[i]), onOut[i], math.Float32bits(onOut[i]))
		}
	}
}

// TestCrossLayerGatePrefetchDefaultOffIsInert pins the package-level opt-in default: with the
// gate never turned on, the predictor refuses (nil), the stager stages nothing, the ledger stays
// the zero value, and the ring ledger matches a plain run byte-for-byte.
func TestCrossLayerGatePrefetchDefaultOffIsInert(t *testing.T) {
	const H, E, K = 256, 8, 2
	m := expertCrossLayerModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 3 * E
	x := expertRingTestInput(H)

	SetCrossLayerGatePrefetch(false)
	s, _ := expertCrossLayerSession(m, budget)
	defer s.Close()
	mat := sessionQ4KKernel{s: s}
	moeFFN{}.apply(m, 0, x, mat)

	// The predictor must refuse outright with the gate off ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â no picks means no staging offered.
	if got := predictNextLayerExperts(m, 0, x, mat); got != nil {
		t.Fatalf("predictNextLayerExperts returned %v with the gate OFF, want nil", expertIDs(got))
	}
	// Even if a caller hands the stager a record, the default-off path stages nothing and the
	// ledger stays zero: prefetchNextLayerGateExperts is inert for an empty prediction.
	s.prefetchNextLayerGateExperts(0, nil)
	if st := CrossLayerPrefetchStatsFor(s); st != (CrossLayerPrefetchStats{}) {
		t.Fatalf("cross-layer stats=%+v with the gate OFF, want the zero value", st)
	}

	// A plain run (no cross-layer calls at all) must leave an identical ring ledger.
	plain, _ := expertCrossLayerSession(m, budget)
	defer plain.Close()
	matPlain := sessionQ4KKernel{s: plain}
	moeFFN{}.apply(m, 0, x, matPlain)
	moeFFN{}.apply(m, 1, x, matPlain)
	moeFFN{}.apply(m, 1, x, mat)

	if a, b := s.ExpertRing(), plain.ExpertRing(); a.Prefetched != b.Prefetched || a.PageIns != b.PageIns {
		t.Fatalf("gate OFF changed the ring ledger: prefetched=%d/%d pageIns=%d/%d",
			a.Prefetched, b.Prefetched, a.PageIns, b.PageIns)
	}
}

// TestCrossLayerGatePrefetchStagesBeforeNextLayerGEMM reuses the ordering-recording backend: with
// the cross-layer gate ON, layer 1's predicted expert weights are staged (uploaded) by
// prefetchNextLayerGateExperts BEFORE layer 1's first expert GEMM runs. That is the property an
// async upload would turn into real overlap.
func TestCrossLayerGatePrefetchStagesBeforeNextLayerGEMM(t *testing.T) {
	const H, E, K = 256, 8, 2
	m := expertCrossLayerModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)

	s, be := expertCrossLayerSession(m, perWeight*3*E)
	defer s.Close()

	SetCrossLayerGatePrefetch(true)
	defer SetCrossLayerGatePrefetch(false)

	x := expertRingTestInput(H)
	mat := sessionQ4KKernel{s: s}

	// Layer 0 computes THROUGH THE LIVE SEAM: with the gate ON, moeFFN.apply runs its router and
	// then crossLayerGatePrefetch predicts layer 1's set and stages it as hints. We capture the
	// prediction the seam made so the residency assertion below is about the real pre-GEMM state.
	moeFFN{}.apply(m, 0, x, mat)
	if len(be.events) == 0 {
		t.Fatal("layer 0 recorded no upload/GEMM events; the ordering witness is vacuous")
	}
	predicted := predictNextLayerExperts(m, 0, x, mat)
	if len(predicted) == 0 {
		t.Fatal("predictor returned no picks for layer 1; the ordering witness is vacuous")
	}

	// The ordering claim: the PREDICTED experts' weights are resident in the ring BEFORE layer 1
	// runs a single GEMM. (Layer 1's own R3 same-layer prefetch also uploads at its entry, so
	// counting every upload before the first GEMM would conflate the two rungs; the cross-layer
	// claim is about the predicted set specifically.)
	for _, pk := range predicted {
		for _, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
			if !s.expertRing.isResident("q4k:" + expertName(1, pk.expert, suffix)) {
				t.Fatalf("predicted layer-1 expert %d's %s was not resident after layer 0's cross-layer prefetch and before layer 1's GEMM",
					pk.expert, suffix)
			}
		}
	}

	// Layer 1 runs; because its predicted weights are already resident, the demand path's first
	// touch of a predicted expert must be a HIT, not a cold page-in.
	hitsBefore := s.ExpertRing().Hits
	moeFFN{}.apply(m, 1, x, mat)
	if st := s.ExpertRing(); st.Hits <= hitsBefore {
		t.Fatalf("layer 1 recorded no new ring hits after a cross-layer prefetch: hits %d -> %d; the hints were not resident when demanded",
			hitsBefore, st.Hits)
	}
}

// fusedExpertFixture builds a single fused [E, out, in] f32 expert tensor whose bytes are
// filled with their own file offset, so expert e's stride-byte sub-range is trivially
// verifiable, and returns the on-disk path plus the geometry readExpertSlice needs.
func fusedExpertFixture(t *testing.T, experts, out, in int) (path string, base, stride int64, data []byte) {
	t.Helper()
	stride = int64(out * in * 4) // f32
	total := int64(experts) * stride
	data = make([]byte, total)
	for i := range data {
		data[i] = byte(i)
	}
	name := "model.layers.0.mlp.experts.gate_up_proj"
	buf := tinySafetensorsBytes(t, map[string]tinySTTensor{
		name: {dtype: "F32", shape: []int{experts, out, in}, data: data},
	})
	path = filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write fused expert safetensors: %v", err)
	}
	// The fused tensor is the only tensor, so its data starts at dataBase (DataOffsets[0]==0).
	// Recover dataBase from a header parse of the raw bytes so base is derived, not assumed.
	hdr, dataBase, err := parseSafetensorsHeader(buf)
	if err != nil {
		t.Fatalf("parse header: %v", err)
	}
	var e stEntry
	if err := json.Unmarshal(hdr[name], &e); err != nil {
		t.Fatalf("decode entry: %v", err)
	}
	base = int64(dataBase) + int64(e.DataOffsets[0])
	return path, base, stride, data
}

// TestReadExpertSliceAvoidsReadAmplification is the #4359 compatibility witness: on the
// demand-paged (ReadAt) path a top-k route reads only the picked experts' k*stride bytes ÃŽâ€œÃƒâ€¡ÃƒÂ¶
// byte-identical to the whole-tensor read then slice ÃŽâ€œÃƒâ€¡ÃƒÂ¶ instead of the whole E*stride layer.
func TestReadExpertSliceAvoidsReadAmplification(t *testing.T) {
	const experts, out, in = 8, 2, 1
	path, base, stride, data := fusedExpertFixture(t, experts, out, in)
	buf, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	rr := &v4ExpertSourceReaderAt{data: buf}
	sf, err := newSafetensorsFile(rr, int64(len(buf)), nil)
	if err != nil {
		t.Fatalf("newSafetensorsFile: %v", err)
	}
	rr.dataBase = sf.dataBase
	if sf.data != nil {
		t.Fatalf("expected the ReadAt path (sf.data must be nil)")
	}

	// Baseline: reading the whole fused block moves the entire E*stride layer in one ReadAt.
	whole, err := sf.readExpertSlice(base, 0, 1, int64(experts)*stride)
	if err != nil {
		t.Fatalf("whole-block read: %v", err)
	}
	wholeBytes := rr.tensorBytes
	if int64(len(whole)) != int64(experts)*stride || wholeBytes != int64(experts)*stride {
		t.Fatalf("whole-block read moved %d bytes (len %d), want %d", wholeBytes, len(whole), int64(experts)*stride)
	}

	// A top-k route: read only the picked experts, each exactly stride bytes, byte-identical
	// to the corresponding slice of the whole block.
	rr.tensorReads, rr.tensorBytes = 0, 0
	picks := []int{1, 4, 6}
	for _, e := range picks {
		got, err := sf.readExpertSlice(base, e, experts, stride)
		if err != nil {
			t.Fatalf("readExpertSlice(e=%d): %v", e, err)
		}
		want := data[int64(e)*stride : int64(e+1)*stride]
		if string(got) != string(want) {
			t.Fatalf("expert %d bytes=%v, want %v", e, got, want)
		}
	}
	pickedBytes := rr.tensorBytes
	if want := int64(len(picks)) * stride; pickedBytes != want || rr.tensorReads != len(picks) {
		t.Fatalf("top-k route moved %d bytes over %d reads, want %d over %d", pickedBytes, rr.tensorReads, want, len(picks))
	}
	if pickedBytes >= wholeBytes {
		t.Fatalf("read amplification NOT avoided: top-k moved %d bytes, whole block %d", pickedBytes, wholeBytes)
	}
	t.Logf("read-amplification avoidance: top-%d/%d route moved %d/%d bytes (%.0f%% of the layer)",
		len(picks), experts, pickedBytes, wholeBytes, 100*float64(pickedBytes)/float64(wholeBytes))
}

// TestExpertSliceBoundsAndWillneed pins the bounds rejections and that the WILLNEED readahead
// is a safe no-op on the ReadAt path but actually fires on the mmap path where available.
func TestExpertSliceBoundsAndWillneed(t *testing.T) {
	const experts, out, in = 8, 2, 1
	path, base, stride, _ := fusedExpertFixture(t, experts, out, in)
	buf, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	sf, err := newSafetensorsFile(&v4ExpertSourceReaderAt{data: buf}, int64(len(buf)), nil)
	if err != nil {
		t.Fatalf("newSafetensorsFile: %v", err)
	}

	// Out-of-range expert index and an overrunning stride are both refused.
	if _, err := sf.readExpertSlice(base, experts, experts, stride); err == nil {
		t.Fatal("readExpertSlice accepted an out-of-range expert index")
	}
	if _, err := sf.readExpertSlice(base, 0, 1, int64(experts)*stride+4); err == nil {
		t.Fatal("readExpertSlice accepted a slice that overruns the data region")
	}

	// ReadAt path: no mapped region to advise, so the hint is a no-op returning false.
	if sf.willneedExpertSlice(base, 0, experts, stride) {
		t.Fatal("willneedExpertSlice fired on the ReadAt path (sf.data is nil)")
	}

	// mmap path (unix hosts): the hint actually fires; the slice read is byte-identical and
	// zero-copy. On platforms without mmap (e.g. native Windows) openSafetensorsFileMmap
	// returns errMmapUnsupported and this leg is skipped ÃŽâ€œÃƒâ€¡ÃƒÂ¶ the ReadAt legs above still gate.
	msf, err := openSafetensorsFileMmap(path)
	if err != nil {
		t.Logf("mmap unsupported (%v); skipping the WILLNEED/zero-copy leg", err)
		return
	}
	defer msf.Close()
	if msf.data == nil {
		t.Fatal("mmap open returned no mapped bytes")
	}
	if !msf.willneedExpertSlice(base, 1, experts, stride) {
		t.Fatal("willneedExpertSlice did not fire on the mmap path")
	}
	got, err := msf.readExpertSlice(base, 1, experts, stride)
	if err != nil {
		t.Fatalf("mmap readExpertSlice: %v", err)
	}
	if int64(len(got)) != stride {
		t.Fatalf("mmap slice len=%d, want %d", len(got), stride)
	}
}
