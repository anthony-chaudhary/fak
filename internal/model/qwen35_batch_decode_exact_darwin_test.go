//go:build darwin && arm64 && cgo

package model

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

const exactQwen38FixturePhysicalCeiling = int64(2) << 30

type exactQwen38WeightShape struct{ out, in int }

type exactQwen38Fixture struct {
	m            *Model
	q8           map[exactQwen38WeightShape]*q8Tensor
	q4           map[exactQwen38WeightShape]*q4kTensor
	q8Owners     []*metalgemm.Q8Weight
	q4Owners     []*metalgemm.Q4KWeight
	baseQ8       int
	baseGDN      int
	plannedBytes int64
	closed       bool
}

// TestExactQwen38HybridQ4KStepBatchActiveMatchesSerial is the bounded, development-only
// exact-topology witness for #10137. It uses the production batch runner and genuine native
// quant/state owners, but deliberately does not load a GGUF or make a throughput claim.
func TestExactQwen38HybridQ4KStepBatchActiveMatchesSerial(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	cfg := exactQwen38MixedBatchConfig()
	planned := exactQwen38FixturePhysicalBytes(cfg)
	if planned > exactQwen38FixturePhysicalCeiling {
		t.Fatalf("refusing exact fixture before allocation: planned physical bytes=%d ceiling=%d", planned, exactQwen38FixturePhysicalCeiling)
	}
	f := newExactQwen38Fixture(t, cfg, planned)
	t.Cleanup(func() { f.close(t) })

	for _, active := range [][]bool{
		{true, true},
		{true, true, true, true},
		{true, true, true, true, true, true, true, true},
		{true, false, true, false, true, false, true, false},
	} {
		active := active
		t.Run(qwen35BatchCaseName(active), func(t *testing.T) {
			f.runBatchCase(t, active)
		})
	}

	f.close(t)
	t.Logf("exact_qwen38_fixture physical_bytes=%d ceiling=%d q8_shape_owners=%d q4_shape_owners=%d",
		f.plannedBytes, exactQwen38FixturePhysicalCeiling, len(f.q8Owners), len(f.q4Owners))
}

func exactQwen38MixedBatchConfig() Config {
	types := make([]string, 64)
	for i := range types {
		if i%4 == 3 {
			types[i] = "full_attention"
		} else {
			types[i] = "linear_attention"
		}
	}
	return Config{
		HiddenSize: 5120, NumLayers: 64, NumHeads: 24, NumKVHeads: 4, HeadDim: 256,
		IntermediateSize: 17408, VocabSize: 256, RMSNormEps: 1e-5, RopeTheta: 10_000_000,
		TieWordEmbeddings: true, EOSTokenID: -1, ModelType: "qwen3_5_text", LayerTypes: types,
		QKNorm: true, QKNormEps: 1e-6, NormGain1p: true, PartialRotaryFactor: .25,
		LinearConvKernelDim: 4, LinearKeyHeadDim: 128, LinearNumKeyHeads: 16,
		LinearValueHeadDim: 128, LinearNumValueHeads: 48, AttnOutputGate: true,
		FullAttentionInterval: 4,
	}
}

func exactQwen38FixturePhysicalBytes(cfg Config) int64 {
	page := int64(os.Getpagesize())
	roundPage := func(n int64) int64 { return (n + page - 1) / page * page }
	q8Bytes := func(s exactQwen38WeightShape) int64 {
		return roundPage(int64(s.out*s.in)) + roundPage(int64(s.out*(s.in/qBlk)*4))
	}
	q4Bytes := func(s exactQwen38WeightShape) int64 {
		return roundPage(int64(s.out * (s.in / qkK) * q4kBlockBytes))
	}
	_, nV, kHd, vHd, _, valDim, convDim := cfg.linearAttnDims()
	q8Shapes := []exactQwen38WeightShape{
		{convDim, cfg.HiddenSize}, {valDim, cfg.HiddenSize}, {nV, cfg.HiddenSize},
		{cfg.HiddenSize, valDim}, {2 * cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize},
		{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize},
	}
	q4Shapes := []exactQwen38WeightShape{
		{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize}, {cfg.HiddenSize, cfg.NumHeads * cfg.HeadDim},
		{cfg.IntermediateSize, cfg.HiddenSize}, {cfg.HiddenSize, cfg.IntermediateSize},
		{cfg.VocabSize, cfg.HiddenSize},
	}
	var weights int64
	for _, s := range q8Shapes {
		weights += q8Bytes(s)
	}
	for _, s := range q4Shapes {
		weights += q4Bytes(s)
	}
	// At the comparison peak B8 owns 8*48 states and one independent serial oracle owns
	// another 48. The runner's persistent state is two private Metal buffers per owner.
	perOwner := int64(((cfg.LinearConvKernelDim-1)*convDim + nV*kHd*vHd) * 4)
	gdnPeak := int64((8+1)*48) * perOwner
	// Host-only tensors: embeddings, layer norms, full-attention q/k norms, and GDN constants.
	manifest := int64(cfg.VocabSize*cfg.HiddenSize+cfg.HiddenSize) * 4
	for l := 0; l < cfg.NumLayers; l++ {
		manifest += int64(2*cfg.HiddenSize) * 4
		if cfg.isLinearAttnLayer(l) {
			manifest += int64(cfg.LinearConvKernelDim*convDim+2*nV+vHd) * 4
		} else {
			manifest += int64(cfg.NumHeads*cfg.HeadDim+cfg.NumKVHeads*cfg.HeadDim) * 4
		}
	}
	// Full-attention KV for nine-token prefixes plus one step, three forms (K/Kraw/V),
	// and a conservative 128 MiB allowance for the largest B8 transient panels/readbacks.
	kvPeak := int64(8*16*10*3*cfg.NumKVHeads*cfg.HeadDim) * 4
	return weights + gdnPeak + manifest + kvPeak + 128<<20
}

func newExactQwen38Fixture(t *testing.T, cfg Config, planned int64) *exactQwen38Fixture {
	t.Helper()
	f := &exactQwen38Fixture{
		q8: make(map[exactQwen38WeightShape]*q8Tensor), q4: make(map[exactQwen38WeightShape]*q4kTensor),
		baseQ8: metalgemm.LiveQ8Weights(), baseGDN: metalgemm.GDNLiveBufferCount(), plannedBytes: planned,
	}
	t.Cleanup(func() { f.close(t) })

	man, raw := exactQwen38FixtureManifest(cfg)
	f.m = &Model{Cfg: cfg, manifest: man, raw: raw, q8w: make(map[string]*q8Tensor), q4kw: make(map[string]*q4kTensor)}

	q8For := func(out, in int) (*q8Tensor, *metalgemm.Q8Weight) {
		shape := exactQwen38WeightShape{out, in}
		if qt := f.q8[shape]; qt != nil {
			for i, prior := range f.q8Owners {
				if prior.Out == out && prior.In == in {
					return qt, f.q8Owners[i]
				}
			}
		}
		qt := newQ8Tensor(out, in, in/qBlk)
		for row := 0; row < out; row++ {
			block := row % qt.nblk
			qt.q[row*in+block*qBlk+(row%qBlk)] = int8(1 + row%7)
			qt.d[row*qt.nblk+block] = .0025 + float32(row%5)*.00025
		}
		w := metalgemm.AliasQ8(qt.q, qt.d, out, in)
		if w == nil || !w.NoCopy() {
			t.Fatalf("Q8 shape %dx%d did not produce a real no-copy native owner", out, in)
		}
		f.q8[shape], f.q8Owners = qt, append(f.q8Owners, w)
		return qt, w
	}
	q4For := func(out, in int) (*q4kTensor, *metalgemm.Q4KWeight) {
		shape := exactQwen38WeightShape{out, in}
		if qt := f.q4[shape]; qt != nil {
			for i, prior := range f.q4Owners {
				if prior.Out == out && prior.In == in {
					return qt, f.q4Owners[i]
				}
			}
		}
		nblk := in / qkK
		raw := makePageAlignedResidentBytes(out * nblk * q4kBlockBytes)
		rng := rand.New(rand.NewSource(int64(out)*1_000_003 + int64(in)))
		blk := make([]byte, q4kBlockBytes)
		for i := 0; i < out*nblk; i++ {
			randQ4KBlockBounded(rng, blk, 2, 6)
			copy(raw[i*q4kBlockBytes:], blk)
		}
		qt := &q4kTensor{out: out, in: in, nblk: nblk, raw: raw}
		w := metalgemm.UploadQ4K(qt.raw, out, in)
		if w == nil || !w.NoCopy() {
			t.Fatalf("Q4_K shape %dx%d did not produce a real no-copy native owner", out, in)
		}
		f.q4[shape], f.q4Owners = qt, append(f.q4Owners, w)
		return qt, w
	}

	q8Table := make(map[string]*metalgemm.Q8Weight, 272)
	q4Table := make(map[string]*metalgemm.Q4KWeight, 224)
	_, nV, _, _, _, valDim, convDim := cfg.linearAttnDims()
	for l := 0; l < cfg.NumLayers; l++ {
		p := func(s string) string { return layerName(l, s) }
		if cfg.isLinearAttnLayer(l) {
			for _, spec := range []struct {
				name    string
				out, in int
			}{{p("linear_attn.in_proj_qkv.weight"), convDim, cfg.HiddenSize}, {p("linear_attn.in_proj_z.weight"), valDim, cfg.HiddenSize}, {p("linear_attn.in_proj_b.weight"), nV, cfg.HiddenSize}, {p("linear_attn.in_proj_a.weight"), nV, cfg.HiddenSize}, {p("linear_attn.out_proj.weight"), cfg.HiddenSize, valDim}} {
				qt, w := q8For(spec.out, spec.in)
				f.m.q8w[spec.name], q8Table[spec.name] = qt, w
			}
		} else {
			for _, spec := range []struct {
				name    string
				out, in int
			}{{p("self_attn.q_proj.weight"), 2 * cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize}, {p("self_attn.k_proj.weight"), cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize}} {
				qt, w := q8For(spec.out, spec.in)
				f.m.q8w[spec.name], q8Table[spec.name] = qt, w
			}
			for _, spec := range []struct {
				name    string
				out, in int
			}{{p("self_attn.v_proj.weight"), cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize}, {p("self_attn.o_proj.weight"), cfg.HiddenSize, cfg.NumHeads * cfg.HeadDim}} {
				qt, w := q4For(spec.out, spec.in)
				f.m.q4kw[spec.name], q4Table[spec.name] = qt, w
			}
		}
		for _, spec := range []struct {
			name    string
			out, in int
		}{{p("mlp.gate_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize}, {p("mlp.up_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize}, {p("mlp.down_proj.weight"), cfg.HiddenSize, cfg.IntermediateSize}} {
			qt, w := q4For(spec.out, spec.in)
			f.m.q4kw[spec.name], q4Table[spec.name] = qt, w
		}
	}
	headQT, headW := q4For(cfg.VocabSize, cfg.HiddenSize)
	f.m.q4kw["model.embed_tokens.weight"], f.m.q4khead = headQT, headQT
	q4Table["model.embed_tokens.weight"] = headW

	runtimeNames, err := qwen38MetalQ8RuntimeNames(cfg)
	if err != nil {
		t.Fatalf("derive exact Qwen3.8 runtime inventory: %v", err)
	}
	if len(q8Table) != len(runtimeNames) {
		t.Fatalf("immutable Q8 inventory names=%d, want canonical runtime inventory=%d", len(q8Table), len(runtimeNames))
	}
	for _, name := range runtimeNames {
		if q8Table[name] == nil {
			t.Fatalf("canonical Q8 runtime projection %q has no native owner", name)
		}
	}
	if len(f.q8Owners) != len(f.q8) || len(f.q4Owners) != len(f.q4) {
		t.Fatalf("immutable owner inventory q8_owners=%d q8_shapes=%d q4_owners=%d q4_shapes=%d", len(f.q8Owners), len(f.q8), len(f.q4Owners), len(f.q4))
	}
	assertExactQwen38ImmutableOwnerIDs(t, "Q8", len(f.q8Owners), func(i int) int { return f.q8Owners[i].ID() })
	assertExactQwen38ImmutableOwnerIDs(t, "Q4_K", len(f.q4Owners), func(i int) int { return f.q4Owners[i].ID() })
	metalQ4KMu.Lock()
	metalQ8KW[f.m] = q8Table
	metalQ8Exact[f.m] = &metalQ8ExactState{handles: append([]*metalgemm.Q8Weight(nil), f.q8Owners...)}
	metalQ4KW[f.m] = q4Table
	metalQ4KMu.Unlock()
	if got := metalgemm.LiveQ8Weights(); got != f.baseQ8+len(f.q8Owners) {
		t.Fatalf("Q8 native owners=%d over baseline, want %d", got-f.baseQ8, len(f.q8Owners))
	}
	return f
}

func assertExactQwen38ImmutableOwnerIDs(t *testing.T, kind string, count int, id func(int) int) {
	t.Helper()
	seen := make(map[int]struct{}, count)
	for i := 0; i < count; i++ {
		ownerID := id(i)
		if ownerID < 0 {
			t.Fatalf("%s shape owner %d has invalid native registry ID %d", kind, i, ownerID)
		}
		if _, duplicate := seen[ownerID]; duplicate {
			t.Fatalf("%s shape owner %d duplicates native registry ID %d", kind, i, ownerID)
		}
		seen[ownerID] = struct{}{}
	}
}

func exactQwen38FixtureManifest(cfg Config) (map[string]tensorMeta, []byte) {
	tensors := []synthTensor{{name: "model.embed_tokens.weight", shape: []int{cfg.VocabSize, cfg.HiddenSize}}}
	_, nV, _, vHd, _, _, convDim := cfg.linearAttnDims()
	for l := 0; l < cfg.NumLayers; l++ {
		p := func(s string) string { return layerName(l, s) }
		tensors = append(tensors,
			synthTensor{name: p("input_layernorm.weight"), shape: []int{cfg.HiddenSize}},
			synthTensor{name: p("post_attention_layernorm.weight"), shape: []int{cfg.HiddenSize}},
		)
		if cfg.isLinearAttnLayer(l) {
			tensors = append(tensors,
				synthTensor{name: p("linear_attn.conv1d.weight"), shape: []int{cfg.LinearConvKernelDim * convDim}},
				synthTensor{name: p("linear_attn.A_log"), shape: []int{nV}},
				synthTensor{name: p("linear_attn.dt_bias"), shape: []int{nV}},
				synthTensor{name: p("linear_attn.norm.weight"), shape: []int{vHd}},
			)
		} else {
			tensors = append(tensors,
				synthTensor{name: p("self_attn.q_norm.weight"), shape: []int{cfg.NumHeads * cfg.HeadDim}},
				synthTensor{name: p("self_attn.k_norm.weight"), shape: []int{cfg.NumKVHeads * cfg.HeadDim}},
			)
		}
	}
	tensors = append(tensors, synthTensor{name: "model.norm.weight", shape: []int{cfg.HiddenSize}})
	return synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		if name == "model.embed_tokens.weight" {
			return next() * .05
		}
		if name == "model.norm.weight" || len(name) >= len("norm.weight") && name[len(name)-len("norm.weight"):] == "norm.weight" || len(name) >= len("layernorm.weight") && name[len(name)-len("layernorm.weight"):] == "layernorm.weight" {
			return 0
		}
		return next() * .002
	})
}

func (f *exactQwen38Fixture) runBatchCase(t *testing.T, active []bool) {
	t.Helper()
	base := metalgemm.GDNLiveBufferCount()
	if base != f.baseGDN {
		t.Fatalf("GDN baseline drift before B%d: got %d want %d", len(active), base, f.baseGDN)
	}
	batch := make([]*Session, len(active))
	defer func() {
		for _, s := range batch {
			if s != nil {
				s.Close()
			}
		}
	}()
	for lane := range batch {
		batch[lane] = exactQwen38PreparedSession(t, f.m, lane)
	}
	wantBuffers := base + 2*48*len(batch)
	if got := metalgemm.GDNLiveBufferCount(); got != wantBuffers {
		t.Fatalf("B%d GDN live buffers=%d want %d (%d distinct owners)", len(batch), got, wantBuffers, 48*len(batch))
	}
	assertExactQwen38DistinctMutableOwners(t, batch)

	inactiveBefore := make(map[int][32]byte)
	for lane, enabled := range active {
		if !enabled {
			inactiveBefore[lane] = exactQwen38SessionDigest(t, batch[lane])
		}
	}
	ids := make([]int, len(batch))
	for lane := range ids {
		ids[lane] = (41 + 17*lane) % f.m.Cfg.VocabSize
	}
	bs := &BatchSession{M: f.m, Seqs: batch}
	got := bs.StepBatchActive(ids, active)
	if bs.LastStepMACs() == 0 || bs.LastStepSharedPanels() == 0 {
		t.Fatalf("B%d native receipt macs=%d shared_panels=%d", len(batch), bs.LastStepMACs(), bs.LastStepSharedPanels())
	}

	for lane, enabled := range active {
		if !enabled {
			if got[lane] != nil {
				t.Fatalf("inactive lane %d produced logits", lane)
			}
			continue
		}
		serial := exactQwen38PreparedSession(t, f.m, lane)
		defer serial.Close()
		want := append([]float32(nil), serial.Step(ids[lane])...)
		assertExactQwen38Finite(t, "serial logits", want)
		assertExactQwen38Finite(t, "batch logits", got[lane])
		assertCosineAtLeast(t, "exact batch logits", want, got[lane], Qwen35GDNParityCosineMin)
		assertMaxAbsAtMost(t, "exact batch logits", want, got[lane], 2e-3)
		if argmax(want) != argmax(got[lane]) {
			t.Fatalf("lane %d accepted token=%d want %d", lane, argmax(got[lane]), argmax(want))
		}
		compareExactQwen38SessionState(t, lane, serial, batch[lane])
		serial.Close()
		if live := metalgemm.GDNLiveBufferCount(); live != wantBuffers {
			t.Fatalf("serial lane %d release live buffers=%d want batch baseline %d", lane, live, wantBuffers)
		}
	}
	for lane, before := range inactiveBefore {
		if after := exactQwen38SessionDigest(t, batch[lane]); after != before {
			t.Fatalf("inactive lane %d state mutated", lane)
		}
	}
	for _, s := range batch {
		s.Close()
	}
	if got := metalgemm.GDNLiveBufferCount(); got != base {
		t.Fatalf("B%d cleanup live buffers=%d want baseline %d", len(batch), got, base)
	}
	t.Logf("B%d active=%d gdn_owners=%d shared_panels=%d macs=%d", len(batch), countExactActive(active), 48*len(batch), bs.LastStepSharedPanels(), bs.LastStepMACs())
}

func exactQwen38PreparedSession(t *testing.T, m *Model, lane int) *Session {
	t.Helper()
	cfg := m.Cfg
	cache := &KVCache{cfg: cfg, K: make([][]float32, cfg.NumLayers), Kraw: make([][]float32, cfg.NumLayers), V: make([][]float32, cfg.NumLayers)}
	prefix := lane + 2
	for pos := 0; pos < prefix; pos++ {
		cache.appendPosition(pos, (lane*31+pos*7+3)%cfg.VocabSize)
	}
	kvw := cfg.NumKVHeads * cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			continue
		}
		cache.K[l] = exactQwen38StateValues(prefix*kvw, lane, l, 1)
		cache.Kraw[l] = exactQwen38StateValues(prefix*kvw, lane, l, 2)
		cache.V[l] = exactQwen38StateValues(prefix*kvw, lane, l, 3)
	}
	s := &Session{M: m, Cache: cache}
	s.initMixedQKV()
	s.Q4K, s.MetalQ4K = true, true
	backend := newQwen35MetalGDNSequenceBackend()
	accepted, err := s.initQwen35GDNPreprojectedSequence(backend)
	if err != nil || !accepted {
		t.Fatalf("lane %d init GDN owners accepted=%v err=%v", lane, accepted, err)
	}
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	conv := make([]float32, (cfg.LinearConvKernelDim-1)*convDim)
	recurrent := make([]float32, nV*kHd*vHd)
	for i := 0; i < 64 && i < len(conv); i++ {
		conv[(i*1543+lane)%len(conv)] = float32(lane+1) * .0001 * float32(i+1)
	}
	for i := 0; i < 256 && i < len(recurrent); i++ {
		recurrent[(i*3079+lane)%len(recurrent)] = float32(lane+1) * .00001 * float32(i+1)
	}
	snaps := make([]qwen35GDNLayerSnapshot, 0, 48)
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			snaps = append(snaps, qwen35GDNLayerSnapshot{layer: l, conv: conv, recurrent: recurrent})
		}
	}
	if ok, err := s.promoteQwen35MetalGDNDecode(snaps); err != nil || !ok {
		s.Close()
		t.Fatalf("lane %d promote GDN owners ok=%v err=%v", lane, ok, err)
	}
	if path, ok := s.Qwen35GDNDecodePath(); !ok || path != Qwen35MetalGDNDecodeForwardPath {
		s.Close()
		t.Fatalf("lane %d native decode path=%q ok=%v", lane, path, ok)
	}
	return s
}

func exactQwen38StateValues(n, lane, layer, band int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(((i*13+lane*17+layer*19+band*23)%101)-50) * 1e-4
	}
	return x
}

func assertExactQwen38DistinctMutableOwners(t *testing.T, sessions []*Session) {
	t.Helper()
	seen := make(map[Qwen35GDNAuxHandle]struct{}, len(sessions)*48*2)
	for lane, s := range sessions {
		for layer, state := range s.qwen35HAL.sequenceLayers {
			if !s.M.Cfg.isLinearAttnLayer(layer) {
				continue
			}
			for _, handle := range []Qwen35GDNAuxHandle{state.Convolution, state.Recurrent} {
				if _, duplicate := seen[handle]; duplicate {
					t.Fatalf("lane %d layer %d aliased mutable handle %d", lane, layer, handle)
				}
				seen[handle] = struct{}{}
			}
		}
	}
	if want := len(sessions) * 48 * 2; len(seen) != want {
		t.Fatalf("distinct mutable handles=%d want %d", len(seen), want)
	}
}

func compareExactQwen38SessionState(t *testing.T, lane int, serial, batched *Session) {
	t.Helper()
	if serial.Cache.Len() != batched.Cache.Len() {
		t.Fatalf("lane %d position=%d want %d", lane, batched.Cache.Len(), serial.Cache.Len())
	}
	if !slices.Equal(serial.Cache.pos, batched.Cache.pos) ||
		!slices.Equal(serial.Cache.lineage.ids, batched.Cache.lineage.ids) ||
		serial.Cache.lineage.fault != batched.Cache.lineage.fault {
		t.Fatalf("lane %d position/token lineage diverged", lane)
	}
	for l := 0; l < serial.M.Cfg.NumLayers; l++ {
		if serial.M.Cfg.isLinearAttnLayer(l) {
			ss := serial.qwen35HAL.sequenceBackend.(qwen35GDNSequenceSnapshotter)
			bs := batched.qwen35HAL.sequenceBackend.(qwen35GDNSequenceSnapshotter)
			sc, sr, err := ss.SnapshotQwen35GDNAuxState(serial.qwen35HAL.sequenceLayers[l])
			if err != nil {
				t.Fatal(err)
			}
			bc, br, err := bs.SnapshotQwen35GDNAuxState(batched.qwen35HAL.sequenceLayers[l])
			if err != nil {
				t.Fatal(err)
			}
			assertExactQwen38Finite(t, fmt.Sprintf("lane %d layer %d conv", lane, l), bc)
			assertExactQwen38Finite(t, fmt.Sprintf("lane %d layer %d recurrent", lane, l), br)
			assertCosineAtLeast(t, "exact conv state", sc, bc, Qwen35GDNParityCosineMin)
			assertMaxAbsAtMost(t, "exact conv state", sc, bc, 5e-4)
			assertCosineAtLeast(t, "exact recurrent state", sr, br, Qwen35GDNParityCosineMin)
			assertMaxAbsAtMost(t, "exact recurrent state", sr, br, 5e-4)
			continue
		}
		for _, pair := range []struct {
			name      string
			want, got []float32
		}{{"K", serial.Cache.K[l], batched.Cache.K[l]}, {"Kraw", serial.Cache.Kraw[l], batched.Cache.Kraw[l]}, {"V", serial.Cache.V[l], batched.Cache.V[l]}} {
			assertExactQwen38Finite(t, fmt.Sprintf("lane %d layer %d %s", lane, l, pair.name), pair.got)
			assertCosineAtLeast(t, pair.name, pair.want, pair.got, .99999)
			assertMaxAbsAtMost(t, pair.name, pair.want, pair.got, 5e-4)
		}
	}
}

func exactQwen38SessionDigest(t *testing.T, s *Session) [32]byte {
	t.Helper()
	h := sha256.New()
	var bits [4]byte
	writeUint32 := func(v uint32) {
		binary.LittleEndian.PutUint32(bits[:], v)
		_, _ = h.Write(bits[:])
	}
	write := func(x []float32) {
		for _, v := range x {
			writeUint32(math.Float32bits(v))
		}
	}
	writeUint32(uint32(len(s.Cache.pos)))
	for _, pos := range s.Cache.pos {
		writeUint32(uint32(pos))
	}
	writeUint32(uint32(len(s.Cache.lineage.ids)))
	for _, id := range s.Cache.lineage.ids {
		writeUint32(id)
	}
	writeUint32(uint32(len(s.Cache.lineage.fault)))
	_, _ = h.Write([]byte(s.Cache.lineage.fault))
	for l := 0; l < s.M.Cfg.NumLayers; l++ {
		write(s.Cache.K[l])
		write(s.Cache.Kraw[l])
		write(s.Cache.V[l])
		if !s.M.Cfg.isLinearAttnLayer(l) {
			continue
		}
		snap := s.qwen35HAL.sequenceBackend.(qwen35GDNSequenceSnapshotter)
		conv, recurrent, err := snap.SnapshotQwen35GDNAuxState(s.qwen35HAL.sequenceLayers[l])
		if err != nil {
			t.Fatal(err)
		}
		write(conv)
		write(recurrent)
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

func assertExactQwen38Finite(t *testing.T, name string, x []float32) {
	t.Helper()
	if len(x) == 0 {
		t.Fatalf("%s is empty", name)
	}
	for i, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s[%d]=%v is not finite", name, i, v)
		}
	}
}

func countExactActive(active []bool) int {
	n := 0
	for _, ok := range active {
		if ok {
			n++
		}
	}
	return n
}

func (f *exactQwen38Fixture) close(t *testing.T) {
	t.Helper()
	if f == nil || f.closed {
		return
	}
	f.closed = true
	metalQ4KMu.Lock()
	delete(metalQ8KW, f.m)
	delete(metalQ8Exact, f.m)
	delete(metalQ8Budget, f.m)
	delete(metalQ4KW, f.m)
	metalQ4KMu.Unlock()
	for i := len(f.q4Owners) - 1; i >= 0; i-- {
		f.q4Owners[i].Release()
	}
	for i := len(f.q8Owners) - 1; i >= 0; i-- {
		f.q8Owners[i].Release()
	}
	for i, w := range f.q4Owners {
		if w.ID() != -1 {
			t.Errorf("Q4_K owner %d survived exact-once release", i)
		}
	}
	for i, w := range f.q8Owners {
		if w.ID() != -1 {
			t.Errorf("Q8 owner %d survived exact-once release", i)
		}
	}
	if got := metalgemm.LiveQ8Weights(); got != f.baseQ8 {
		t.Errorf("Q8 owner cleanup live=%d want baseline %d", got, f.baseQ8)
	}
	if got := metalgemm.GDNLiveBufferCount(); got != f.baseGDN {
		t.Errorf("GDN owner cleanup live buffers=%d want baseline %d", got, f.baseGDN)
	}
	if len(f.q8Owners) != len(f.q8) || len(f.q4Owners) != len(f.q4) {
		t.Errorf("immutable owner/shape mismatch after cleanup q8_owners=%d q8_shapes=%d q4_owners=%d q4_shapes=%d", len(f.q8Owners), len(f.q8), len(f.q4Owners), len(f.q4))
	}
}
