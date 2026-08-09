package ggufload

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_activated_fit_test.go — witnesses for the activated-working-set device admission (#5612).
//
// The claim under test is the one the plan doc makes: a routed-MoE checkpoint whose FULL expert
// band exceeds the device is still servable when its ACTIVATED slice fits, and the refusal line
// moves down to the floor (dense base + one MoE layer's K experts) rather than staying at the band.
// So the load-bearing witnesses are (a) a budget that refuses under the old admission and is
// admitted under this one, and (b) a budget under the floor that is still refused.

// glmMoeLayersGGUF writes a minimal glm_moe_dsa GGUF carrying L MoE layers, each with the three
// batched routed-expert tensors (gate/up/down, dims [E,I,H], type F32), plus one non-expert
// token_embd tensor of nonElems F32 elements. Payloads are zero-filled: every arithmetic under test
// is header-only (dims + type), so the bytes need not be valid data — only the payload LENGTH is
// read. Unlike glmQuantArmGGUF (one layer, one projection) this fixture exists to make MoELayers
// and the per-layer floor distinguishable from the whole-model band.
func glmMoeLayersGGUF(t *testing.T, L, E, I, H, K, nonElems int) []byte {
	t.Helper()
	const align = 32
	type tensor struct {
		name string
		dims []uint64
	}
	var tensors []tensor
	for l := 0; l < L; l++ {
		blk := "blk." + itoaForTest(l) + "."
		for _, suffix := range []string{glmGGUFExpertsGate, glmGGUFExpertsUp, glmGGUFExpertsDown} {
			tensors = append(tensors, tensor{name: blk + suffix, dims: []uint64{uint64(H), uint64(I), uint64(E)}})
		}
	}
	tensors = append(tensors, tensor{name: "token_embd.weight", dims: []uint64{uint64(nonElems)}})

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 11)
	writeKVString(&b, "general.architecture", "glm_moe_dsa")
	writeKVUint32(&b, "general.alignment", align)
	writeKVUint32(&b, "glm_moe_dsa.embedding_length", uint32(H))
	writeKVUint32(&b, "glm_moe_dsa.block_count", uint32(L))
	writeKVUint32(&b, "glm_moe_dsa.attention.head_count", 2)
	writeKVUint32(&b, "glm_moe_dsa.attention.head_count_kv", 1)
	writeKVUint32(&b, "glm_moe_dsa.feed_forward_length", 8)
	writeKVUint32(&b, "glm_moe_dsa.expert_count", uint32(E))
	writeKVUint32(&b, "glm_moe_dsa.expert_used_count", uint32(K))
	writeKVUint32(&b, "glm_moe_dsa.expert_feed_forward_length", uint32(I))
	writeKVFloat32(&b, "glm_moe_dsa.attention.layer_norm_rms_epsilon", 1e-5)

	var offset uint64
	for _, tn := range tensors {
		writeTensorInfoForTest(&b, tn.name, tn.dims, TensorF32, offset)
		n := mustPayloadBytes(t, tn.name, tn.dims, TensorF32)
		offset = (offset + n + align - 1) / align * align
	}
	padToAlignment(&b, align)
	start := b.Len()
	padToLen(&b, start+int(offset))
	return b.Bytes()
}

// openMoELayersFixture writes the L-layer fixture to a temp file and opens it.
func openMoELayersFixture(t *testing.T, L, E, I, H, K, nonElems int) *WeightSource {
	t.Helper()
	p := filepath.Join(t.TempDir(), "glm.gguf")
	if err := os.WriteFile(p, glmMoeLayersGGUF(t, L, E, I, H, K, nonElems), 0o644); err != nil {
		t.Fatal(err)
	}
	ws, err := OpenWeights(p)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	t.Cleanup(func() { ws.Close() })
	return ws
}

// TestRoutedExpertActiveSetCountsMoELayers pins the denominator the per-layer floor divides by. It
// must be the count of blocks that actually CARRY routed experts, which on a hybrid checkpoint with
// a dense prefix is not block_count — here every block is MoE, and three projections per block must
// still count as ONE layer.
func TestRoutedExpertActiveSetCountsMoELayers(t *testing.T) {
	const L, E, I, H, K = 4, 4, 3, 2, 2
	ws := openMoELayersFixture(t, L, E, I, H, K, 8)
	as, ok, err := ws.RoutedExpertActiveSet()
	if err != nil || !ok {
		t.Fatalf("RoutedExpertActiveSet ok=%v err=%v", ok, err)
	}
	if as.MoELayers != L {
		t.Fatalf("MoELayers=%d, want %d (three projections per block are one layer)", as.MoELayers, L)
	}
	wantBand := int64(L * 3 * E * I * H * 4) // L layers × 3 projections × [E,I,H] F32
	if as.RoutedResident != wantBand {
		t.Fatalf("RoutedResident=%d, want %d", as.RoutedResident, wantBand)
	}
}

// TestActivatedExpertFitLevels pins the three demand levels and the ring the placement sizes,
// against byte counts derived independently from the fixture's shape.
func TestActivatedExpertFitLevels(t *testing.T) {
	const L, E, I, H, K, nonElems = 4, 4, 3, 2, 2, 8
	ws := openMoELayersFixture(t, L, E, I, H, K, nonElems)

	band := int64(L * 3 * E * I * H * 4)
	base := int64(nonElems * 4)
	perExpertLayer := band / int64(E) / int64(L) // divides evenly for this fixture
	wantFloor := perExpertLayer * K
	wantToken := wantFloor * L

	// A budget sized to hold the base plus exactly two layers' activated slice: above the floor,
	// below a whole token, and well below the band.
	budget := base + 2*wantFloor
	f, ok, err := ws.ActivatedExpertFitFor(budget)
	if err != nil || !ok {
		t.Fatalf("ActivatedExpertFitFor ok=%v err=%v", ok, err)
	}
	if f.MoELayers != L || f.NumExperts != E || f.ExpertsUsed != K {
		t.Fatalf("shape = layers %d experts %d used %d, want %d/%d/%d", f.MoELayers, f.NumExperts, f.ExpertsUsed, L, E, K)
	}
	if f.DeviceBaseBytes != base {
		t.Fatalf("DeviceBaseBytes=%d, want %d", f.DeviceBaseBytes, base)
	}
	if f.RoutedBandBytes != band {
		t.Fatalf("RoutedBandBytes=%d, want %d", f.RoutedBandBytes, band)
	}
	if f.ActivatedLayerBytes != wantFloor {
		t.Fatalf("ActivatedLayerBytes=%d, want %d (K experts of ONE layer)", f.ActivatedLayerBytes, wantFloor)
	}
	if f.ActivatedTokenBytes != wantToken {
		t.Fatalf("ActivatedTokenBytes=%d, want %d (K experts on every layer)", f.ActivatedTokenBytes, wantToken)
	}
	// The levels are monotone and the budget lands between the first two.
	if !f.MinFits {
		t.Fatalf("MinFits=false at a budget holding the base plus two layers' activation: %+v", f)
	}
	if f.TokenFits || f.BandFits {
		t.Fatalf("TokenFits=%v BandFits=%v at a budget below a whole token", f.TokenFits, f.BandFits)
	}
	// The ring takes exactly what the base leaves, and the rest of the band is host-scoped.
	if want := budget - base; f.RingBytes != want {
		t.Fatalf("RingBytes=%d, want %d (the budget after the dense base)", f.RingBytes, want)
	}
	if f.HostBandBytes != band-f.RingBytes {
		t.Fatalf("HostBandBytes=%d, want %d", f.HostBandBytes, band-f.RingBytes)
	}
	if f.RingBytes+f.HostBandBytes != band {
		t.Fatalf("ring %d + host %d != band %d — routed bytes were lost or double-counted", f.RingBytes, f.HostBandBytes, band)
	}

	// A budget above the whole band saturates: the ring holds everything, nothing spills.
	full, ok, err := ws.ActivatedExpertFitFor(base + band + 1<<20)
	if err != nil || !ok {
		t.Fatalf("ActivatedExpertFitFor(generous) ok=%v err=%v", ok, err)
	}
	if !full.MinFits || !full.TokenFits || !full.BandFits {
		t.Fatalf("a budget above the band did not report all three levels: %+v", full)
	}
	if full.RingBytes != band || full.HostBandBytes != 0 {
		t.Fatalf("generous budget: ring=%d host=%d, want ring=%d host=0", full.RingBytes, full.HostBandBytes, band)
	}
}

// TestActivatedExpertFitAdmitsWhatTheBandAdmissionRefuses is the claim of #5612's preflight half: a
// checkpoint the all-resident admission refuses is ADMITTED on its activated working set, at the
// same device budget and the same headroom, because the ring never has to hold the whole band.
func TestActivatedExpertFitAdmitsWhatTheBandAdmissionRefuses(t *testing.T) {
	const L, E, I, H, K, nonElems = 4, 8, 4, 4, 2, 8
	ws := openMoELayersFixture(t, L, E, I, H, K, nonElems)

	f, ok, err := ws.ActivatedExpertFitFor(0)
	if err != nil || !ok {
		t.Fatalf("ActivatedExpertFitFor ok=%v err=%v", ok, err)
	}
	// A device that holds the base plus one whole token's activation, but NOT the routed band.
	budget := f.DeviceBaseBytes + f.ActivatedTokenBytes
	if budget >= f.DeviceBaseBytes+f.RoutedBandBytes {
		t.Fatalf("fixture is not discriminating: token budget %d already covers base+band %d", budget, f.DeviceBaseBytes+f.RoutedBandBytes)
	}
	be := capBackend{total: budget, free: budget, known: true}

	// Today's admission — the whole checkpoint resident — refuses at this budget.
	if err := ws.FitOnDevice(be, 0); err == nil {
		t.Fatal("FitOnDevice admitted a checkpoint whose band exceeds the budget; the fixture proves nothing")
	}
	// The activated-working-set admission accepts it.
	if err := ws.FitActivatedExpertsOnDevice(be, 0); err != nil {
		t.Fatalf("FitActivatedExpertsOnDevice refused a checkpoint whose activated set fits: %v", err)
	}

	// ...and it still refuses below the floor: no ring budget can assemble one layer's K experts.
	tight := f.DeviceBaseBytes + f.ActivatedLayerBytes - 1
	small := capBackend{total: tight, free: tight, known: true}
	err = ws.FitActivatedExpertsOnDevice(small, 0)
	if err == nil {
		t.Fatal("a budget below the activated floor was admitted; decode could not assemble one expert group")
	}
	var fitErr *compute.FitError
	if !errors.As(err, &fitErr) {
		t.Fatalf("want a typed *compute.FitError so the refusal carries the demand classes, got %T: %v", err, err)
	}
}

// TestFitActivatedExpertsFailsOpenAndFallsBack pins the two contracts this check inherits from its
// neighbours: a backend that cannot report memory is never refused, and a checkpoint with no routed
// band falls back to the ordinary resident admission rather than silently admitting everything.
func TestFitActivatedExpertsFailsOpenAndFallsBack(t *testing.T) {
	ws := openMoELayersFixture(t, 2, 4, 3, 2, 2, 8)
	if err := ws.FitActivatedExpertsOnDevice(nil, 0); err != nil {
		t.Fatalf("a nil backend must fail open, got %v", err)
	}
	if err := ws.FitActivatedExpertsOnDevice(capBackend{total: 1, free: 1}, 0); err != nil {
		t.Fatalf("a backend with no capacity probe must fail open, got %v", err)
	}

	// A checkpoint with no routed-expert band has no activated working set: the fallback must be
	// the ordinary resident fit, which at a one-byte budget refuses.
	dense, err := NewWeightSource(&File{
		Metadata: synthGLMMeta(0), // expert_count 0 -> no batched routed-expert band
		Tensors:  []TensorInfo{{Name: "token_embd.weight", Dims: []uint64{1024}, Type: TensorF32}},
	}, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	if _, ok, err := dense.ActivatedExpertFitFor(1 << 30); err != nil || ok {
		t.Fatalf("dense ActivatedExpertFitFor ok=%v err=%v, want ok=false", ok, err)
	}
	if err := dense.FitActivatedExpertsOnDevice(capBackend{total: 1, free: 1, known: true}, 0); err == nil {
		t.Fatal("a dense checkpoint at a 1-byte budget was admitted; the fallback to FitOnDevice did not happen")
	}
}

// TestActivatedExpertFitUnknownKChargesTheWholeBand: expert_used_count is the one scalar the
// active-set derivation waits on. Without it the activated slice is UNKNOWN, and an unknown slice
// must not be admitted as a small one — the floor collapses back to the whole band.
func TestActivatedExpertFitUnknownK(t *testing.T) {
	f, ok, err := activatedExpertFit(RoutedExpertActiveSet{
		NumExperts: 8, ExpertsUsed: 0, MoELayers: 4, RoutedResident: 4096, NonExpertResident: 512,
	}, 1<<20)
	if err != nil || !ok {
		t.Fatalf("activatedExpertFit ok=%v err=%v", ok, err)
	}
	if f.ActivatedLayerBytes != 4096 || f.ActivatedTokenBytes != 4096 {
		t.Fatalf("unknown K: floor=%d token=%d, want both 4096 (the whole band)", f.ActivatedLayerBytes, f.ActivatedTokenBytes)
	}

	// A K larger than E is a nonsense header; clamping keeps the floor a real byte count (the whole
	// layer) instead of an over-reservation that refuses a servable checkpoint.
	f, ok, err = activatedExpertFit(RoutedExpertActiveSet{
		NumExperts: 4, ExpertsUsed: 9, MoELayers: 2, RoutedResident: 800, NonExpertResident: 0,
	}, 1<<20)
	if err != nil || !ok {
		t.Fatalf("activatedExpertFit(K>E) ok=%v err=%v", ok, err)
	}
	if f.ExpertsUsed != 4 {
		t.Fatalf("ExpertsUsed=%d, want 4 (clamped to E)", f.ExpertsUsed)
	}
	if f.ActivatedLayerBytes != 400 {
		t.Fatalf("ActivatedLayerBytes=%d, want 400 (one layer's whole expert set)", f.ActivatedLayerBytes)
	}

	// Rounding is UP: an uneven band must over-reserve rather than under-count into an OOM.
	f, _, err = activatedExpertFit(RoutedExpertActiveSet{
		NumExperts: 3, ExpertsUsed: 1, MoELayers: 2, RoutedResident: 100, NonExpertResident: 0,
	}, 1<<20)
	if err != nil {
		t.Fatalf("activatedExpertFit(uneven): %v", err)
	}
	if f.ActivatedLayerBytes != 17 { // ceil(100/6) = 17, not 16
		t.Fatalf("ActivatedLayerBytes=%d, want 17 (ceil, not floor)", f.ActivatedLayerBytes)
	}
}
