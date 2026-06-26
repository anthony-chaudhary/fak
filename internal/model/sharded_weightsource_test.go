package model

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// sharded_weightsource_test.go — the issue-#40 acceptance gate on a SYNTHETIC sharded
// fixture (two tiny shards + an index.json weight_map), with no real 7B download. It
// proves the two claims the issue names:
//
//	(a) a weight read through compute.WeightSource returns the RIGHT tensor from the RIGHT
//	    shard (the weight_map routing is correct end-to-end), and
//	(b) quant-on-load materializes each big matmul weight DIRECTLY to Q8_0 — the loader
//	    never holds the full f32 blob, so the peak resident footprint is the quantized
//	    size, not the f32 size.
//
// The fixture is built in-test as real little-endian safetensors files, so the loader's
// header parse, shard routing, bf16/f32 decode, and quant-on-load all run for real.

// stTensor is one tensor to place into a synthetic safetensors shard.
type stTensor struct {
	name  string
	shape []int
	data  []float32 // row-major, len == prod(shape)
}

// writeSafetensorsShard writes tensors as one little-endian F32 safetensors file:
// [8-byte header length][JSON header][packed F32 data], matching the format
// openSafetensorsFile parses. Returns the on-disk byte size.
func writeSafetensorsShard(t *testing.T, path string, tensors []stTensor) int64 {
	t.Helper()
	hdr := map[string]any{}
	var data []byte
	off := 0
	for _, ts := range tensors {
		n := 1
		for _, d := range ts.shape {
			n *= d
		}
		if n != len(ts.data) {
			t.Fatalf("tensor %s: shape wants %d elems, got %d", ts.name, n, len(ts.data))
		}
		nbytes := n * 4
		hdr[ts.name] = map[string]any{
			"dtype":        "F32",
			"shape":        ts.shape,
			"data_offsets": []int{off, off + nbytes},
		}
		buf := make([]byte, nbytes)
		for i, v := range ts.data {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		data = append(data, buf...)
		off += nbytes
	}
	hb, err := json.Marshal(hdr)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	out := make([]byte, 8+len(hb)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(hb)))
	copy(out[8:], hb)
	copy(out[8+len(hb):], data)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatalf("write shard %s: %v", path, err)
	}
	return int64(len(out))
}

// rampF32 fills n elements with a deterministic, non-degenerate ramp so the Q8_0 scale
// per 32-block is non-zero (a degenerate all-equal block would make d==0 and hide rounding).
func rampF32(n, salt int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*7+salt)%251)/251.0*2 - 1 // in (-1,1), varies within each 32-block
	}
	return out
}

// buildSyntheticShardedFixture writes a 1-layer synthetic checkpoint across TWO shards
// with a model.safetensors.index.json weight_map, and returns the dir + the f32 data of
// one quantized weight (q_proj, in shard 1) and one small weight (norm, in shard 2) so the
// test can check shard routing. in-dim is a multiple of 32 for the Q8_0 precondition.
func buildSyntheticShardedFixture(t *testing.T) (dir string, cfg Config, qProjF32 []float32, normF32 []float32) {
	t.Helper()
	dir = t.TempDir()

	const H = 64 // hidden, multiple of 32 (Q8_0 reduction dim)
	const I = 96 // intermediate, multiple of 32
	const V = 32 // vocab, small
	cfg = Config{
		HiddenSize: H, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: I, VocabSize: V, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true,
	}
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim

	// Shard 1 holds the attention projections (the big quant weights live here).
	qProjF32 = rampF32(nH*hd*H, 1)
	shard1 := []stTensor{
		{"model.layers.0.self_attn.q_proj.weight", []int{nH * hd, H}, qProjF32},
		{"model.layers.0.self_attn.k_proj.weight", []int{nKV * hd, H}, rampF32(nKV*hd*H, 2)},
		{"model.layers.0.self_attn.v_proj.weight", []int{nKV * hd, H}, rampF32(nKV*hd*H, 3)},
		{"model.layers.0.self_attn.o_proj.weight", []int{H, nH * hd}, rampF32(H*nH*hd, 4)},
	}
	// Shard 2 holds the MLP weights, embedding (tied head), and the small f32 norms.
	normF32 = ones(H)
	shard2 := []stTensor{
		{"model.layers.0.mlp.gate_proj.weight", []int{I, H}, rampF32(I*H, 5)},
		{"model.layers.0.mlp.up_proj.weight", []int{I, H}, rampF32(I*H, 6)},
		{"model.layers.0.mlp.down_proj.weight", []int{H, I}, rampF32(H*I, 7)},
		{"model.embed_tokens.weight", []int{V, H}, rampF32(V*H, 8)},
		{"model.layers.0.input_layernorm.weight", []int{H}, ones(H)},
		{"model.layers.0.post_attention_layernorm.weight", []int{H}, ones(H)},
		{"model.norm.weight", []int{H}, normF32},
	}

	writeSafetensorsShard(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), shard1)
	writeSafetensorsShard(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), shard2)

	weightMap := map[string]string{}
	for _, ts := range shard1 {
		weightMap[ts.name] = "model-00001-of-00002.safetensors"
	}
	for _, ts := range shard2 {
		weightMap[ts.name] = "model-00002-of-00002.safetensors"
	}
	index := map[string]any{
		"metadata":   map[string]any{"total_size": 0},
		"weight_map": weightMap,
	}
	ib, err := json.Marshal(index)
	if err != nil {
		t.Fatalf("marshal index: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	return dir, cfg, qProjF32, normF32
}

func ones(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

// TestShardedQuantOnLoadWeightSource is the issue-#40 acceptance test: load a synthetic
// two-shard checkpoint with a weight_map quant-on-load, then read weights back through the
// compute.WeightSource seam and assert (a) shard routing is correct and (b) the quant-on-
// load footprint never holds the full f32 blob.
func TestShardedQuantOnLoadWeightSource(t *testing.T) {
	dir, cfg, qProjF32, normF32 := buildSyntheticShardedFixture(t)

	m, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir: %v", err)
	}

	ws := m.WeightSource(nil) // nil -> the cpu-ref reference backend (Default)
	be := compute.Default()
	if be == nil {
		t.Fatal("no compute backend registered")
	}

	// (a) shard routing: q_proj came from shard 1 — read it back through WeightSource as
	// Q8_0 and confirm it dequantizes to the SAME f32 a direct Quantize of the source data
	// gives. Bit-identity to the regular quantizeQ8 proves the right bytes from the right
	// shard reached q8w (a swapped/missing shard would produce different codes).
	qName := "model.layers.0.self_attn.q_proj.weight"
	wt, err := ws.Weight(qName, compute.Q8_0)
	if err != nil {
		t.Fatalf("WeightSource.Weight(%s, Q8_0): %v", qName, err)
	}
	if wt.Dtype != compute.Q8_0 {
		t.Fatalf("%s: want dtype Q8_0, got %s", qName, wt.Dtype)
	}
	if got := wt.Shape; len(got) != 2 || got[0] != cfg.NumHeads*cfg.HeadDim || got[1] != cfg.HiddenSize {
		t.Fatalf("%s: shape %v, want [%d %d]", qName, got, cfg.NumHeads*cfg.HeadDim, cfg.HiddenSize)
	}
	// The codes the WeightSource hands out must equal a fresh quantizeQ8 of the SOURCE f32
	// (same data, from the right shard).
	wantQ := quantizeQ8(qProjF32, cfg.NumHeads*cfg.HeadDim, cfg.HiddenSize)
	gotCodes := wt.Buf().(interface{ I8() []int8 }).I8()
	if len(gotCodes) != len(wantQ.q) {
		t.Fatalf("%s: code len %d != %d", qName, len(gotCodes), len(wantQ.q))
	}
	for i := range wantQ.q {
		if gotCodes[i] != wantQ.q[i] {
			t.Fatalf("%s code[%d]: WeightSource %d != source-quantized %d (wrong shard?)", qName, i, gotCodes[i], wantQ.q[i])
		}
	}

	// (a, cont.) the small f32 norm came from shard 2 — read it back as F32 through the
	// seam and confirm it matches the source bytes exactly (lossless, and right shard).
	nName := "model.norm.weight"
	nt, err := ws.Weight(nName, compute.F32)
	if err != nil {
		t.Fatalf("WeightSource.Weight(%s, F32): %v", nName, err)
	}
	if nt.Dtype != compute.F32 {
		t.Fatalf("%s: want dtype F32, got %s", nName, nt.Dtype)
	}
	gotF32, ok := be.Host(nt)
	if !ok {
		t.Fatalf("%s: F32 weight is not host-addressable", nName)
	}
	if len(gotF32) != len(normF32) {
		t.Fatalf("%s: f32 len %d != %d", nName, len(gotF32), len(normF32))
	}
	for i := range normF32 {
		if math.Float32bits(gotF32[i]) != math.Float32bits(normF32[i]) {
			t.Fatalf("%s[%d]: WeightSource %v != source %v (wrong shard?)", nName, i, gotF32[i], normF32[i])
		}
	}

	// (b) the memory win: the big matmul weights are Q8_0-only — their f32 was never
	// retained. Requesting one as F32 must fail (it cannot be re-inflated), and the weight
	// must NOT be present in the f32 manifest.
	for _, big := range []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
	} {
		if m.has(big) {
			t.Errorf("%s retained in f32 manifest — quant-on-load did not drop the f32", big)
		}
		if _, err := ws.Weight(big, compute.F32); err == nil {
			t.Errorf("%s: WeightSource served F32 for a Q8_0-only weight (re-inflated the blob)", big)
		}
	}

	// (b, cont.) footprint: the RESIDENT bytes are the quantized size, not the f32 size. We
	// total the source f32 bytes of the quantized weights, the actual Q8_0 bytes the loader
	// kept, and the f32 blob it kept (small tensors only), and assert the resident total is
	// far under the all-f32 total — the peak-footprint claim, measured on resident state.
	var f32SizeOfQuantWeights int
	for _, ts := range []struct {
		name string
		n    int
	}{
		{"model.layers.0.self_attn.q_proj.weight", cfg.NumHeads * cfg.HeadDim * cfg.HiddenSize},
		{"model.layers.0.self_attn.k_proj.weight", cfg.NumKVHeads * cfg.HeadDim * cfg.HiddenSize},
		{"model.layers.0.self_attn.v_proj.weight", cfg.NumKVHeads * cfg.HeadDim * cfg.HiddenSize},
		{"model.layers.0.self_attn.o_proj.weight", cfg.HiddenSize * cfg.NumHeads * cfg.HeadDim},
		{"model.layers.0.mlp.gate_proj.weight", cfg.IntermediateSize * cfg.HiddenSize},
		{"model.layers.0.mlp.up_proj.weight", cfg.IntermediateSize * cfg.HiddenSize},
		{"model.layers.0.mlp.down_proj.weight", cfg.HiddenSize * cfg.IntermediateSize},
	} {
		f32SizeOfQuantWeights += ts.n * 4
	}
	q8Resident := residentQ8Bytes(m)
	f32Resident := len(m.raw) // small f32 tensors the Q8 path reads directly
	resident := q8Resident + f32Resident

	// If the loader had concatenated the f32 blob first (the thing the issue forbids), the
	// quantized weights' f32 would still be resident: resident >= f32SizeOfQuantWeights.
	// Quant-on-load means the Q8_0 of those weights is ~1.06 B/param, < 4 B/param f32, so:
	if q8Resident >= f32SizeOfQuantWeights {
		t.Errorf("resident Q8 bytes %d >= source f32 bytes %d — no quant-on-load win", q8Resident, f32SizeOfQuantWeights)
	}
	// And the f32 the loader DID keep must not include any quantized weight (it is the
	// small-tensor remainder only): resident f32 < the quantized weights' f32 size.
	if f32Resident >= f32SizeOfQuantWeights {
		t.Errorf("resident f32 blob %d >= quantized-weights f32 %d — the big f32 was not dropped", f32Resident, f32SizeOfQuantWeights)
	}

	t.Logf("sharded quant-on-load: q8_resident=%dB f32_resident=%dB total=%dB vs all-f32-of-quant-weights=%dB (%.2fx leaner on the matmul weights)",
		q8Resident, f32Resident, resident, f32SizeOfQuantWeights, float64(f32SizeOfQuantWeights)/float64(q8Resident))
}

func TestQwen25ProductionShardedQuantLoadContract(t *testing.T) {
	tests := []struct {
		name      string
		cfg       Config
		groupSize int
	}{
		{
			name: "7b-contract",
			cfg: Config{
				HiddenSize: 224, NumLayers: 1, NumHeads: 7, NumKVHeads: 1, HeadDim: 32,
				IntermediateSize: 320, VocabSize: 64, RMSNormEps: 1e-6, RopeTheta: 1000000,
				TieWordEmbeddings: false, AttentionBias: true, HiddenAct: "silu",
				ModelType: "qwen2", MaxPositionEmbeddings: 32768,
			},
			groupSize: 7, // real Qwen2.5-7B is 28 query heads / 4 KV heads.
		},
		{
			name: "32b-contract",
			cfg: Config{
				HiddenSize: 160, NumLayers: 1, NumHeads: 5, NumKVHeads: 1, HeadDim: 32,
				IntermediateSize: 256, VocabSize: 64, RMSNormEps: 1e-6, RopeTheta: 1000000,
				TieWordEmbeddings: false, AttentionBias: true, HiddenAct: "silu",
				ModelType: "qwen2", MaxPositionEmbeddings: 32768,
			},
			groupSize: 5, // real Qwen2.5-32B is 40 query heads / 8 KV heads.
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			dir := buildQwen25ProductionShardFixture(t, tt.cfg)
			m, err := LoadSafetensorsQuantDir(dir, tt.cfg)
			if err != nil {
				t.Fatalf("LoadSafetensorsQuantDir: %v", err)
			}

			if m.Cfg.ModelType != "qwen2" || !m.Cfg.AttentionBias || m.Cfg.activationName() != "silu" {
				t.Fatalf("qwen2 axes not preserved: model_type=%q attention_bias=%v act=%q",
					m.Cfg.ModelType, m.Cfg.AttentionBias, m.Cfg.activationName())
			}
			if got := m.Cfg.GroupSize(); got != tt.groupSize {
				t.Fatalf("GQA group size = %d, want %d", got, tt.groupSize)
			}

			H, I, V := tt.cfg.HiddenSize, tt.cfg.IntermediateSize, tt.cfg.VocabSize
			qRows := tt.cfg.NumHeads * tt.cfg.HeadDim
			kvRows := tt.cfg.NumKVHeads * tt.cfg.HeadDim
			assertQ8Shape(t, m, layerName(0, "self_attn.q_proj.weight"), qRows, H)
			assertQ8Shape(t, m, layerName(0, "self_attn.k_proj.weight"), kvRows, H)
			assertQ8Shape(t, m, layerName(0, "self_attn.v_proj.weight"), kvRows, H)
			assertQ8Shape(t, m, layerName(0, "self_attn.o_proj.weight"), H, qRows)
			assertQ8Shape(t, m, layerName(0, "mlp.gate_proj.weight"), I, H)
			assertQ8Shape(t, m, layerName(0, "mlp.up_proj.weight"), I, H)
			assertQ8Shape(t, m, layerName(0, "mlp.down_proj.weight"), H, I)
			assertQ8Shape(t, m, "lm_head.weight", V, H)

			assertTensorShape(t, m, "model.embed_tokens.weight", []int{V, H})
			assertTensorShape(t, m, "model.norm.weight", []int{H})
			assertTensorShape(t, m, layerName(0, "input_layernorm.weight"), []int{H})
			assertTensorShape(t, m, layerName(0, "post_attention_layernorm.weight"), []int{H})
			assertTensorShape(t, m, layerName(0, "self_attn.q_proj.bias"), []int{qRows})
			assertTensorShape(t, m, layerName(0, "self_attn.k_proj.bias"), []int{kvRows})
			assertTensorShape(t, m, layerName(0, "self_attn.v_proj.bias"), []int{kvRows})

			assertQuantPrefillFinite(t, m, tt.cfg)
		})
	}
}

func buildQwen25ProductionShardFixture(t *testing.T, cfg Config) string {
	t.Helper()
	dir := t.TempDir()
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	qRows := cfg.NumHeads * cfg.HeadDim
	kvRows := cfg.NumKVHeads * cfg.HeadDim

	shard1 := []stTensor{
		{layerName(0, "self_attn.q_proj.weight"), []int{qRows, H}, rampF32(qRows*H, 11)},
		{layerName(0, "self_attn.k_proj.weight"), []int{kvRows, H}, rampF32(kvRows*H, 12)},
		{layerName(0, "self_attn.v_proj.weight"), []int{kvRows, H}, rampF32(kvRows*H, 13)},
		{layerName(0, "self_attn.o_proj.weight"), []int{H, qRows}, rampF32(H*qRows, 14)},
		{layerName(0, "self_attn.q_proj.bias"), []int{qRows}, rampF32(qRows, 15)},
		{layerName(0, "self_attn.k_proj.bias"), []int{kvRows}, rampF32(kvRows, 16)},
		{layerName(0, "self_attn.v_proj.bias"), []int{kvRows}, rampF32(kvRows, 17)},
		{layerName(0, "input_layernorm.weight"), []int{H}, ones(H)},
		{layerName(0, "post_attention_layernorm.weight"), []int{H}, ones(H)},
	}
	shard2 := []stTensor{
		{layerName(0, "mlp.gate_proj.weight"), []int{I, H}, rampF32(I*H, 18)},
		{layerName(0, "mlp.up_proj.weight"), []int{I, H}, rampF32(I*H, 19)},
		{layerName(0, "mlp.down_proj.weight"), []int{H, I}, rampF32(H*I, 20)},
		{"model.embed_tokens.weight", []int{V, H}, rampF32(V*H, 21)},
		{"model.norm.weight", []int{H}, ones(H)},
		{"lm_head.weight", []int{V, H}, rampF32(V*H, 22)},
	}

	shard1Name := "model-00001-of-00002.safetensors"
	shard2Name := "model-00002-of-00002.safetensors"
	writeSafetensorsShard(t, filepath.Join(dir, shard1Name), shard1)
	writeSafetensorsShard(t, filepath.Join(dir, shard2Name), shard2)

	weightMap := map[string]string{}
	for _, ts := range shard1 {
		weightMap[ts.name] = shard1Name
	}
	for _, ts := range shard2 {
		weightMap[ts.name] = shard2Name
	}
	index := map[string]any{
		"metadata":   map[string]any{"total_size": 0},
		"weight_map": weightMap,
	}
	ib, err := json.Marshal(index)
	if err != nil {
		t.Fatalf("marshal index: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	return dir
}

// residentQ8Bytes totals the bytes the Q8_0 store actually holds resident: int8 codes (1
// B/code) + per-block f32 scales (4 B each). This is the quant-on-load footprint of the big
// weights — what peak RSS reflects, not the f32 the loader never built.
func residentQ8Bytes(m *Model) int {
	total := 0
	for _, qt := range m.q8w {
		total += len(qt.q)*1 + len(qt.d)*4
	}
	return total
}

// TestWeightSourceErrorsAreCleanNotPanics confirms the seam reports an absent weight (and a
// dtype it cannot serve) as an error, never a panic — so a backend probing for an optional
// tensor gets a value it can branch on.
func TestWeightSourceErrorsAreCleanNotPanics(t *testing.T) {
	dir, cfg, _, _ := buildSyntheticShardedFixture(t)
	m, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir: %v", err)
	}
	ws := m.WeightSource(nil)

	if _, err := ws.Weight("does.not.exist", compute.Q8_0); err == nil {
		t.Error("absent weight (Q8_0) returned no error")
	}
	if _, err := ws.Weight("does.not.exist", compute.F32); err == nil {
		t.Error("absent weight (F32) returned no error")
	}
	if _, err := ws.Weight("model.norm.weight", compute.FP8); err == nil {
		t.Error("unsupported dtype FP8 returned no error")
	}
	// A present small f32 weight requested as Q8_0 quantizes on demand only if 2-D; norm is
	// 1-D, so it must error rather than silently mis-quantize.
	if _, err := ws.Weight("model.norm.weight", compute.Q8_0); err == nil {
		t.Error("1-D weight requested as Q8_0 returned no error")
	}
}
