package ggufload

// The direct-resident native-q4_0 load witnesses (#5497). A checkpoint distributed as native
// q4_0 is published BECAUSE it is the small artifact; the default lean-Q8 load expanded every
// weight band to f32 and re-quantized it to Q8_0, so the model went resident at roughly double
// its own on-disk size AND held the f32 expansion next to the Q8 result at the load's peak.
//
// Two things have to be shown, and the second is the one a merely-plausible fix would miss:
//
//  1. STEADY STATE — the resident bytes of a q4_0 band track the block layout the file
//     actually uses (18 B per 32 weights = 0.5625 B/weight), not the Q8_0 density
//     (34 B per 32 weights + f32 scales = 1.125 B/weight) the round trip produced.
//  2. PEAK TRANSIENT — a direct path that still spikes through f32 has not fixed the fit
//     problem. runtime.MemStats.TotalAlloc is CUMULATIVE and never decreases, so the bytes
//     allocated across the whole load bound the peak live footprint FROM ABOVE. If an f32
//     (4 B/weight) or Q8 (1.125 B/weight) copy of the band had ever been materialized — even
//     for an instant, even collected before anything could observe the live heap — it would
//     still be counted here. A total below the Q8 density is therefore proof that no such
//     copy was ever built, which a live-heap sample taken after the fact could not establish.
//
// Everything below is measured from the loader's own accounting over synthetic headers. No
// model is run and no hardware figure is claimed.

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// q40 block geometry, restated locally so a change to the loader constants has to be
// deliberate: 32 weights per block, 18 bytes per block.
const (
	q40TestBlockWeights = 32
	q40TestBlockBytes   = 18
)

// writeQ40Fixture writes a one-layer GGUF whose named tensors are all [dim,dim] Q4_0 and
// returns its path. Payload bytes are a fixed non-zero pattern so a routing check cannot pass
// on an all-zero read, and every scale field stays whatever the pattern makes it — these tests
// assert BYTES and ROUTING, never decoded values.
func writeQ40Fixture(t *testing.T, dim, layers int, names []string) string {
	t.Helper()
	if dim%q40TestBlockWeights != 0 {
		t.Fatalf("fixture dim %d must be a multiple of %d", dim, q40TestBlockWeights)
	}
	tensorBytes := dim * (dim / q40TestBlockWeights) * q40TestBlockBytes

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(names)), 8)
	writeKVString(&b, "general.architecture", "llama")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "llama.embedding_length", uint32(dim))
	writeKVUint32(&b, "llama.block_count", uint32(layers))
	writeKVUint32(&b, "llama.attention.head_count", 1)
	writeKVUint32(&b, "llama.attention.key_length", uint32(dim))
	writeKVUint32(&b, "llama.feed_forward_length", uint32(dim))
	writeKVFloat32(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)

	for i, name := range names {
		writeTensorInfoForTest(&b, name, []uint64{uint64(dim), uint64(dim)}, TensorQ4_0, uint64(i*tensorBytes))
	}
	padToAlignment(&b, 32)
	payload := make([]byte, tensorBytes)
	for i := range payload {
		payload[i] = byte(i*31 + 7)
	}
	for range names {
		b.Write(payload)
	}

	path := filepath.Join(t.TempDir(), "q40.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestLoadModelQ40RoutesNativeBlocksResident is the routing + steady-state byte-accounting
// witness. One load carries both arms of the comparison, because the SAME source type splits
// on eligibility: the identity-normalized weights are held raw at the file's own density,
// while the normalize-sensitive q_proj still takes the proven dequant->normalize->Q8 route and
// lands at the Q8 density. That contrast inside a single model is the ticket's claim measured
// rather than asserted.
func TestLoadModelQ40RoutesNativeBlocksResident(t *testing.T) {
	const dim = 256
	path := writeQ40Fixture(t, dim, 1, []string{
		"blk.0.attn_v.weight",   // identity-normalized -> resident raw
		"blk.0.ffn_down.weight", // identity-normalized -> resident raw
		"output.weight",         // lm_head, identity-normalized -> resident raw
		"blk.0.attn_q.weight",   // rotary, normalize-sensitive -> dequant -> Q8
	})

	m, err := LoadModelQ4K(path)
	if err != nil {
		t.Fatalf("LoadModelQ4K: %v", err)
	}

	for _, name := range []string{
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
		"lm_head.weight",
	} {
		if !m.HasKQuant(name) {
			t.Errorf("expected %s resident (native q4_0 blocks held raw)", name)
		}
		if m.HasQ8(name) {
			t.Errorf("%s must NOT be in q8w: routing it through the round trip is the whole defect", name)
		}
	}
	qp := "model.layers.0.self_attn.q_proj.weight"
	if !m.HasQ8(qp) {
		t.Errorf("expected %s in q8w (rotary -> normalize-sensitive, stays on the proven path)", qp)
	}
	if m.HasKQuant(qp) {
		t.Errorf("%s must NOT be held raw: its bytes are re-laid-out by normalization", qp)
	}

	// Read back one resident tensor's bytes and require them to be the FILE's bytes. A residency
	// flag alone would also be set by a path that stored a re-encoded payload of the right size.
	raw, ok := m.KQuantRaw("model.layers.0.mlp.down_proj.weight")
	if !ok {
		t.Fatal("KQuantRaw missing for the resident down_proj")
	}
	wantBytes := dim * (dim / q40TestBlockWeights) * q40TestBlockBytes
	if len(raw) != wantBytes {
		t.Fatalf("resident payload = %d B, want %d B (the on-disk q4_0 payload)", len(raw), wantBytes)
	}
	for i := range raw {
		if raw[i] != byte(i*31+7) {
			t.Fatalf("resident byte %d = %#x, want %#x: the resident bytes are not the file's bytes",
				i, raw[i], byte(i*31+7))
		}
	}

	// Steady-state density, as an exact rational so there is no tolerance to hide in:
	// KQuantBytes/KQuantParams must be exactly 18/32.
	r := m.ResidentReport()
	if r.KQuantTensors != 3 || r.Q8Tensors != 1 {
		t.Fatalf("resident split: raw=%d q8=%d, want 3/1", r.KQuantTensors, r.Q8Tensors)
	}
	if r.KQuantParams == 0 || r.KQuantBytes*int64(q40TestBlockWeights) != r.KQuantParams*int64(q40TestBlockBytes) {
		t.Fatalf("resident density %d B / %d params is not the native q4_0 %d/%d",
			r.KQuantBytes, r.KQuantParams, q40TestBlockBytes, q40TestBlockWeights)
	}
	residentBPW := float64(r.KQuantBytes) / float64(r.KQuantParams)
	roundTripBPW := float64(r.Q8Bytes) / float64(r.Q8Params)
	t.Logf("q4_0 band resident %.4f B/weight; the same source type on the round trip %.4f B/weight",
		residentBPW, roundTripBPW)
	if residentBPW > 0.625 {
		t.Errorf("resident %.4f B/weight exceeds the 0.625 target", residentBPW)
	}
	if roundTripBPW <= residentBPW {
		t.Fatalf("round-trip arm %.4f B/weight is not above the resident arm %.4f: the comparison is vacuous",
			roundTripBPW, residentBPW)
	}
	// The resident band must also cost strictly less than the file-sized Q8 alternative it
	// replaces; this is the "bigger than its own file" inversion, stated as an inequality.
	if residentBPW >= roundTripBPW/1.5 {
		t.Errorf("resident %.4f B/weight is not decisively below the round trip's %.4f",
			residentBPW, roundTripBPW)
	}

	// Q4_0 is admitted through the SHARED dense raw-quant arm, not a branch of its own, so it
	// inherits that arm's safety gate: a backend with no dense raw-quant kernel turns dense
	// residency off, and these weights must then take the proven route rather than go resident
	// and be unreachable at decode. Held raw behind a backend that cannot read them, the saving
	// would be paid for with a serve that cannot answer.
	t.Run("DenseResidencyGateHonored", func(t *testing.T) {
		ws, err := OpenWeights(path)
		if err != nil {
			t.Fatal(err)
		}
		defer ws.Close()
		gated, err := ws.QuantModelQ4KProfileOptions(nil, WithDenseKQuantResident(false))
		if err != nil {
			t.Fatalf("load with dense residency disabled: %v", err)
		}
		for _, name := range []string{
			"model.layers.0.self_attn.v_proj.weight",
			"model.layers.0.mlp.down_proj.weight",
			"lm_head.weight",
		} {
			if gated.HasKQuant(name) {
				t.Errorf("%s stayed raw-resident though dense residency is disabled", name)
			}
			if !gated.HasQ8(name) {
				t.Errorf("%s did not fall back to the dequant->Q8 route", name)
			}
		}
	})
}

// TestQ40BatchedExpertsLandResidentRaw covers the SECOND store this change has to reach. A
// quantized MoE keeps most of its parameters in BATCHED routed-expert blobs, and those take a
// different loader arm from the dense weights above: the blob is sliced into per-expert byte
// ranges and never dequantized at all. Covering only the dense arm would leave the bulk of such
// a checkpoint on the round trip while every dense assertion above still passed — the exact
// half-application a format with more than one store invites. This drives the real split helper
// and the real collector apply step, not a reimplementation of either.
func TestQ40BatchedExpertsLandResidentRaw(t *testing.T) {
	blockWeights, blockBytes, ok := residentExpertBlockGeometry(TensorQ4_0)
	if !ok {
		t.Fatal("residentExpertBlockGeometry(TensorQ4_0) ok = false: batched q4_0 experts would take the f32 split")
	}
	if blockWeights != q40TestBlockWeights || blockBytes != q40TestBlockBytes {
		t.Fatalf("geometry = %d weights / %d bytes, want %d/%d",
			blockWeights, blockBytes, q40TestBlockWeights, q40TestBlockBytes)
	}

	const (
		experts = 3
		out     = 2
		in      = 64
	)
	perBytes := (out * in / blockWeights) * blockBytes
	raw := make([]byte, experts*perBytes)
	for i := range raw {
		raw[i] = byte(i*17 + 3) // position-dependent, so a wrong stride cannot pass
	}
	split, aligned, err := splitGLMMoeDsaExpertsRawQuant(0, "down_proj", []int{experts, out, in}, raw, blockWeights, blockBytes)
	if err != nil || !aligned {
		t.Fatalf("raw expert split: aligned=%v err=%v", aligned, err)
	}
	if len(split) != experts {
		t.Fatalf("split produced %d experts, want %d", len(split), experts)
	}

	cfg := model.Config{ModelType: "qwen35", HiddenSize: in, IntermediateSize: in}
	builder := model.NewQuantBuilder(cfg, false)
	kvbHalf := map[int]glmKVBHalf{}
	for _, ex := range split {
		w := tensorWork{pending: []pendingTensor{{
			resident: true, residentType: TensorQ4_0, name: ex.Name, shape: ex.Shape, raw: ex.Raw,
		}}}
		if err := applyQ4KTensorWork(w, nil, cfg, builder, kvbHalf, false); err != nil {
			t.Fatalf("apply resident q4_0 expert %s: %v", ex.Name, err)
		}
	}
	// Build() refuses a model whose Q8 store is empty, so one ordinary weight rides along.
	f32Work := tensorWork{pending: []pendingTensor{{
		name: "model.layers.0.self_attn.o_proj.weight", shape: []int{in, in}, f32: make([]float32, in*in),
	}}}
	if err := applyQ4KTensorWork(f32Work, nil, cfg, builder, kvbHalf, false); err != nil {
		t.Fatalf("apply f32: %v", err)
	}
	m, err := builder.Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	for x, ex := range split {
		got, ok := m.KQuantRaw(ex.Name)
		if !ok {
			t.Fatalf("expert %s is not resident: the batched arm half-applied the format", ex.Name)
		}
		if want := raw[x*perBytes : (x+1)*perBytes]; !bytes.Equal(got, want) {
			t.Fatalf("expert %s resident bytes are not its own slice of the blob", ex.Name)
		}
	}
	r := m.ResidentReport()
	if r.KQuantTensors != experts {
		t.Fatalf("resident expert count = %d, want %d", r.KQuantTensors, experts)
	}
	if r.KQuantParams == 0 || r.KQuantBytes*int64(q40TestBlockWeights) != r.KQuantParams*int64(q40TestBlockBytes) {
		t.Fatalf("expert band density %d B / %d params is not the native q4_0 %d/%d",
			r.KQuantBytes, r.KQuantParams, q40TestBlockBytes, q40TestBlockWeights)
	}
}

// TestLoadModelQ40PeakAllocStaysUnderQ8Density is the PEAK witness. TotalAlloc is cumulative,
// so the bytes allocated across the whole load bound the peak live footprint from above: a
// path that expanded a band to f32 and let it be collected again before returning is still
// caught here, which is exactly the failure mode a steady-state check would wave through.
//
// Why cumulative allocation is a SHARP instrument on this particular path: the direct loader
// runs per-tensor work on a worker pool, and a pool cannot share one scratch arena, so each
// tensor that takes a dequant gets a FRESH f32 buffer (gguf_parload.go's contract — workers
// are pure, "dequantF32 allocates fresh"). A direct path that quietly expanded the eligible
// band to f32 would therefore add ~4 B per band weight to the total, tensor after tensor, and
// could not hide behind buffer reuse.
//
// The fixture is 41 eligible weights and one deliberately non-eligible q_proj, because the
// builder refuses a model with nothing on the Q8 path at all. That q_proj really does pay the
// round trip, so its cost is inside the number below — the measurement is not carved to
// exclude it. The claim is therefore about the WHOLE load: bringing this q4_0 checkpoint up
// allocated less in total, start to finish, than one Q8 copy of it would merely occupy.
//
// The lean-Q8 arm runs the identical fixture as a comparison. Note what it does NOT show: its
// total is well under f32 width, because that path reuses a single dequant arena across every
// tensor, so its f32 transient is one tensor at a time rather than the whole band. Its total is
// dominated by the Q8 band it leaves resident — which is the point. The round trip's peak is
// that whole Q8 band PLUS a live f32 arena, both above what the file itself needs.
func TestLoadModelQ40PeakAllocStaysUnderQ8Density(t *testing.T) {
	const (
		dim    = 512
		layers = 8
	)
	var names []string
	for l := 0; l < layers; l++ {
		p := "blk." + string(rune('0'+l)) + "."
		names = append(names,
			p+"attn_v.weight",      // -> self_attn.v_proj, eligible
			p+"attn_output.weight", // -> self_attn.o_proj, eligible
			p+"ffn_gate.weight",    // -> mlp.gate_proj,    eligible
			p+"ffn_up.weight",      // -> mlp.up_proj,      eligible
			p+"ffn_down.weight",    // -> mlp.down_proj,    eligible
		)
	}
	names = append(names,
		"output.weight",       // -> lm_head, eligible
		"blk.0.attn_q.weight", // rotary: stays on the dequant -> normalize -> Q8 route
	)
	const eligible = layers*5 + 1
	path := writeQ40Fixture(t, dim, layers, names)
	params := int64(len(names)) * int64(dim) * int64(dim)

	// Densities in bytes per weight, derived from the block layouts in this package rather
	// than quoted: Q8_0 resident is 32 int8 codes + an f32 scale per 32-weight block; f32 is
	// 4 B/weight flat.
	const q8DensityBPW = (32.0 + 4.0) / 32.0 // 1.125
	const f32DensityBPW = 4.0

	allocPerWeight := func(load func() error) float64 {
		t.Helper()
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		if err := load(); err != nil {
			t.Fatalf("load: %v", err)
		}
		runtime.ReadMemStats(&after)
		return float64(after.TotalAlloc-before.TotalAlloc) / float64(params)
	}

	direct := allocPerWeight(func() error {
		m, err := LoadModelQ4K(path)
		if err != nil {
			return err
		}
		r := m.ResidentReport()
		if r.KQuantTensors != eligible || r.Q8Tensors != len(names)-eligible {
			t.Fatalf("resident split raw=%d q8=%d, want %d/%d; the peak measurement would not be about the direct path",
				r.KQuantTensors, r.Q8Tensors, eligible, len(names)-eligible)
		}
		runtime.KeepAlive(m)
		return nil
	})

	roundTrip := allocPerWeight(func() error {
		ws, err := OpenWeights(path)
		if err != nil {
			return err
		}
		defer ws.Close()
		m, err := ws.QuantModel()
		if err != nil {
			return err
		}
		runtime.KeepAlive(m)
		return nil
	})

	// The floor a band-wide f32 expansion could not go below, whatever else the load did.
	bandF32Floor := f32DensityBPW * float64(eligible) / float64(len(names))

	t.Logf("total allocated across the load: direct %.4f B/weight, lean-Q8 round trip %.4f B/weight "+
		"(Q8 resident density %.4f, f32 %.1f, band-f32 floor %.4f)",
		direct, roundTrip, q8DensityBPW, f32DensityBPW, bandF32Floor)

	// The load-wide total for the direct path stays under the Q8 density, so no Q8 copy of the
	// band was ever allocated at any instant — not merely absent at the end.
	if direct >= q8DensityBPW {
		t.Errorf("direct load allocated %.4f B/weight in total, which is not below the Q8 density %.4f: "+
			"the band was materialized at Q8 width or wider somewhere in the load", direct, q8DensityBPW)
	}
	// And the sharper one: expanding the eligible band to f32 costs at least this much on a path
	// whose workers each allocate fresh, so staying below it rules out an f32 spike specifically.
	if direct >= bandF32Floor {
		t.Errorf("direct load allocated %.4f B/weight, at or above the %.4f a band-wide f32 expansion "+
			"would cost: it still spikes through f32", direct, bandF32Floor)
	}
	// Non-vacuity, two ways. The round trip must exceed the Q8 density (it leaves the whole band
	// resident at Q8 and reads the file besides), and it must exceed the direct arm decisively —
	// a meter stuck at a small constant would fail both.
	if roundTrip <= q8DensityBPW {
		t.Fatalf("lean-Q8 arm allocated only %.4f B/weight; expected above the Q8 density %.4f it leaves "+
			"resident, so the measurement is not reading the load's allocations", roundTrip, q8DensityBPW)
	}
	if direct >= roundTrip/1.5 {
		t.Errorf("direct %.4f B/weight is not decisively below the round trip's %.4f", direct, roundTrip)
	}
}
