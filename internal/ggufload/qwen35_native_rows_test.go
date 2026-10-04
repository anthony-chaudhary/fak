package ggufload

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// qwen35_native_rows_test.go — witnesses for the fak#13567 native-row route: qwen35 row-permuted
// projections keep native Q4_K/Q6_K bytes (rows reordered raw) instead of a Q8 requantization.

// nativeRowsTestCfg is a hybrid geometry with nV != nK (ratio 2), so every linear transform is
// a real, non-identity permutation.
func nativeRowsTestCfg(gated bool) model.Config {
	return model.Config{
		ModelType:           "qwen35",
		HiddenSize:          256,
		NumLayers:           4,
		NumHeads:            2,
		NumKVHeads:          1,
		HeadDim:             128,
		LayerTypes:          []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearNumKeyHeads:   2,
		LinearNumValueHeads: 4,
		LinearKeyHeadDim:    64,
		LinearValueHeadDim:  64,
		LinearConvKernelDim: 4,
		AttnOutputGate:      gated,
	}
}

// randKQuantRaw returns rows*in/256 random super-blocks of type t whose decoded weights are
// finite, small and ZERO-MEAN (so a projection's output depends on its input rather than on a
// shared bias, which keeps the forward-parity witness sensitive to a wrong row order). Q4_K/Q5_K
// blocks use d == dmin with every sub-block scale/min pair chosen so d*sc*mean(q) == dmin*m;
// Q6_K quants and int8 scales are already centred.
func randKQuantRaw(t *testing.T, typ TensorType, rows, in int, seed int64) []byte {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	var blk int
	var scales [12]byte
	switch typ {
	case TensorQ4_K: // q in [0,15]: sc=2, m=15
		blk = blockQ4KBytes
		scales = [12]byte{2, 2, 2, 2, 15, 15, 15, 15, 0xF2, 0xF2, 0xF2, 0xF2}
	case TensorQ5_K: // q in [0,31]: sc=2, m=31 (6-bit min split across the packed layout)
		blk = blockQ5KBytes
		scales = [12]byte{2, 2, 2, 2, 0x5F, 0x5F, 0x5F, 0x5F, 0xF2, 0xF2, 0xF2, 0xF2}
	case TensorQ6_K:
		blk = blockQ6KBytes
	default:
		t.Fatalf("randKQuantRaw: unsupported type %s", typ)
	}
	raw := make([]byte, rows*(in/qkK)*blk)
	rng.Read(raw)
	for off := 0; off < len(raw); off += blk {
		if typ == TensorQ6_K {
			binary.LittleEndian.PutUint16(raw[off+blk-2:], 0x0400|uint16(rng.Intn(0x100)))
			continue
		}
		d := 0x1800 | uint16(rng.Intn(0x400))
		binary.LittleEndian.PutUint16(raw[off:], d)
		binary.LittleEndian.PutUint16(raw[off+2:], d)
		copy(raw[off+4:off+16], scales[:])
	}
	return raw
}

func dequantForNativeRowsTest(t *testing.T, typ TensorType, raw []byte, rows, in int) []float32 {
	t.Helper()
	out, err := dequantF32(TensorInfo{Name: "probe", Dims: []uint64{uint64(in), uint64(rows)}, Type: typ}, raw)
	if err != nil {
		t.Fatalf("dequant %s: %v", typ, err)
	}
	return out
}

// TestQwen35NativeRowsByteParity is the byte-level witness: for every admitted tensor kind,
// reordering the raw rows and then dequantizing is bit-identical to dequantizing and then running
// the production float normalizer (normalizeCanonicalTensorData), with nV != nK.
func TestQwen35NativeRowsByteParity(t *testing.T) {
	cases := []struct {
		canon string
		rows  int
		gated bool
	}{
		{"model.layers.0.self_attn.qkv_proj.weight", 2*2*64 + 4*64, true},       // -> linear_attn.in_proj_qkv
		{"model.layers.0.linear_attn.in_proj_qkv.weight", 2*2*64 + 4*64, true},  // already-resolved spelling
		{"model.layers.1.self_attn.q_gate_proj.weight", 4 * 64, true},           // -> linear_attn.in_proj_z
		{"model.layers.1.linear_attn.in_proj_a.weight", 4, true},                // span-1 value rows
		{"model.layers.2.linear_attn.in_proj_b.weight", 4, true},                // span-1 value rows
		{"model.layers.3.self_attn.q_proj.weight", 2 * 2 * 128, true},           // gated rotary unpermute
		{"model.layers.3.self_attn.q_proj.weight", 2 * 128, false},              // plain rotary unpermute
		{"model.layers.3.self_attn.k_proj.weight", 1 * 128, true},               // rotary unpermute
		{"model.layers.0.linear_attn.in_proj_qkv.weight", 2*2*64 + 4*64, false}, // ungated cfg, same GDN
	}
	for i, tc := range cases {
		for _, typ := range []TensorType{TensorQ4_K, TensorQ6_K} {
			t.Run(tc.canon+"/"+typ.String()+"/"+strconv.Itoa(i), func(t *testing.T) {
				cfg := nativeRowsTestCfg(tc.gated)
				shape := []int{tc.rows, cfg.HiddenSize}
				if !qwen35NativeRowResident(cfg, tc.canon, typ, shape, q4kLoadOptions{residentDenseKQuant: true}) {
					t.Fatalf("predicate rejected admitted tensor %s", tc.canon)
				}
				raw := randKQuantRaw(t, typ, tc.rows, cfg.HiddenSize, int64(1000+i))
				src, err := qwen35CanonicalRowSource(tc.canon, tc.rows, cfg)
				if err != nil {
					t.Fatalf("row source: %v", err)
				}
				identity := true
				for d, s := range src {
					identity = identity && d == s
				}
				if identity {
					t.Fatalf("%s row map is identity under nV != nK; the parity witness would be vacuous", tc.canon)
				}
				reordered, err := normalizeQwen35NativeRows(tc.canon, shape, raw, cfg)
				if err != nil {
					t.Fatalf("normalize rows: %v", err)
				}
				got := dequantForNativeRowsTest(t, typ, reordered, tc.rows, cfg.HiddenSize)
				want, err := normalizeCanonicalTensorData(tc.canon, dequantForNativeRowsTest(t, typ, raw, tc.rows, cfg.HiddenSize), cfg)
				if err != nil {
					t.Fatalf("float normalize: %v", err)
				}
				if len(got) != len(want) {
					t.Fatalf("len %d, want %d", len(got), len(want))
				}
				for j := range want {
					if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
						t.Fatalf("element %d = %v, want %v (raw row reorder != dequant+normalize)", j, got[j], want[j])
					}
				}
			})
		}
	}
}

// TestQwen35NativeRowsPredicateBoundaries pins what the route must NOT admit.
func TestQwen35NativeRowsPredicateBoundaries(t *testing.T) {
	cfg := nativeRowsTestCfg(true)
	on := q4kLoadOptions{residentDenseKQuant: true}
	qkv := "model.layers.0.self_attn.qkv_proj.weight"
	qkvShape := []int{512, 256}
	if qwen35NativeRowResident(cfg, "model.layers.0.linear_attn.out_proj.weight", TensorQ4_K, []int{256, 256}, on) {
		t.Error("out_proj (input-column transform) admitted")
	}
	if qwen35NativeRowResident(cfg, qkv, TensorQ5_K, qkvShape, on) {
		t.Error("Q5_K admitted (no Metal Q5_K GEMV; must stay on Q8)")
	}
	if qwen35NativeRowResident(cfg, qkv, TensorQ6_K, qkvShape, q4kLoadOptions{}) {
		t.Error("Q6_K admitted while dense Q6_K residency is off")
	}
	if !qwen35NativeRowResident(cfg, qkv, TensorQ6_K, qkvShape, q4kLoadOptions{residentDenseQ6K: true}) {
		t.Error("Q6_K rejected while selective dense Q6_K residency is on")
	}
	if qwen35NativeRowResident(cfg, qkv, TensorQ4_K, qkvShape, q4kLoadOptions{residentDenseKQuant: true, nativeRowsOff: true}) {
		t.Error("WithNativeLinearAttnRows(false) did not disable the route")
	}
	if qwen35NativeRowResident(cfg, "model.layers.3.mlp.down_proj.weight", TensorQ4_K, []int{256, 256}, on) {
		t.Error("identity tensor admitted (it belongs to ResidentQ4KEligible)")
	}
	if qwen35NativeRowResident(cfg, "mtp.layers.0.self_attn.q_proj.weight", TensorQ4_K, []int{512, 256}, on) {
		t.Error("MTP q_proj admitted (owned by the MTP materializer)")
	}
	dense := cfg
	dense.LayerTypes = nil
	if qwen35NativeRowResident(dense, "model.layers.3.self_attn.q_proj.weight", TensorQ4_K, []int{512, 256}, on) {
		t.Error("non-hybrid config admitted")
	}
	// The exact 64-layer Qwen3.8 runtime requires these weights as Q8.
	exact := cfg
	exact.NumLayers = 64
	exact.LayerTypes = make([]string, 64)
	for l := range exact.LayerTypes {
		exact.LayerTypes[l] = "linear_attention"
		if (l+1)%4 == 0 {
			exact.LayerTypes[l] = "full_attention"
		}
	}
	if !model.Qwen38ExactMetalQ8Runtime(exact) {
		t.Fatal("test geometry is not the exact Qwen3.8 runtime")
	}
	if qwen35NativeRowResident(exact, qkv, TensorQ4_K, qkvShape, on) {
		t.Error("exact Qwen3.8 runtime admitted (its Metal decode requires Q8)")
	}
	// nV == nK: the linear transforms are identity, the rotary q/k unpermute is not.
	same := cfg
	same.LinearNumValueHeads = same.LinearNumKeyHeads
	src, err := qwen35CanonicalRowSource(qkv, 2*2*64+2*64, same)
	if err != nil {
		t.Fatal(err)
	}
	for d, s := range src {
		if d != s {
			t.Fatalf("nV == nK qkv row map moved row %d from %d", d, s)
		}
	}
	raw := randKQuantRaw(t, TensorQ4_K, len(src), 256, 3)
	if out, err := permuteRawRows(qkv, raw, src); err != nil || &out[0] != &raw[0] {
		t.Fatalf("identity map must return raw unchanged without a copy (err=%v)", err)
	}
}

// nativeRowsFixtureTensor is one tensor of the in-memory qwen35 checkpoint.
type nativeRowsFixtureTensor struct {
	name string
	dims []uint64 // GGUF order: {in, out}
	typ  TensorType
}

// qwen35NativeRowsFixture builds a 4-layer (3 GDN + 1 full-attention) Qwen3.5 hybrid checkpoint
// with nV=4 != nK=2 whose transformed projections carry every routing class: Q4_K (native),
// Q6_K in_proj_qkv (native kqw), Q5_K in_proj_qkv (stays Q8), and the out_proj column transform
// (stays Q8).
func qwen35NativeRowsFixture(t *testing.T) *WeightSource {
	t.Helper()
	const (
		h, ffn, vocab     = 256, 256, 32
		heads, kvHeads    = 2, 1
		headDim           = 128
		linHD, nK, nV     = 64, 2, 4
		keyDim, valDim    = nK * linHD, nV * linHD
		convDim, convKern = 2*keyDim + valDim, 4
	)
	var tensors []nativeRowsFixtureTensor
	add := func(name string, typ TensorType, dims ...uint64) {
		tensors = append(tensors, nativeRowsFixtureTensor{name, dims, typ})
	}
	add("token_embd.weight", TensorF32, h, vocab)
	add("output_norm.weight", TensorF32, h)
	add("output.weight", TensorQ4_K, h, vocab)
	qkvType := []TensorType{TensorQ4_K, TensorQ6_K, TensorQ5_K}
	for l := 0; l < 3; l++ {
		p := "blk." + strconv.Itoa(l) + "."
		add(p+"attn_norm.weight", TensorF32, h)
		add(p+"post_attention_norm.weight", TensorF32, h)
		add(p+"attn_qkv.weight", qkvType[l], h, convDim)
		add(p+"attn_gate.weight", TensorQ4_K, h, valDim)
		add(p+"ssm_alpha.weight", TensorQ4_K, h, nV)
		add(p+"ssm_beta.weight", TensorQ4_K, h, nV)
		add(p+"ssm_out.weight", TensorQ4_K, valDim, h)
		add(p+"ssm_a", TensorF32, nV)
		add(p+"ssm_dt.bias", TensorF32, nV)
		add(p+"ssm_conv1d.weight", TensorF32, convKern, convDim)
		add(p+"ssm_norm.weight", TensorF32, linHD)
		add(p+"ffn_gate.weight", TensorQ4_K, h, ffn)
		add(p+"ffn_up.weight", TensorQ4_K, h, ffn)
		add(p+"ffn_down.weight", TensorQ6_K, ffn, h)
	}
	p := "blk.3."
	add(p+"attn_norm.weight", TensorF32, h)
	add(p+"post_attention_norm.weight", TensorF32, h)
	add(p+"attn_q.weight", TensorQ4_K, h, 2*heads*headDim)
	add(p+"attn_k.weight", TensorQ4_K, h, kvHeads*headDim)
	add(p+"attn_v.weight", TensorQ6_K, h, kvHeads*headDim)
	add(p+"attn_output.weight", TensorQ4_K, heads*headDim, h)
	add(p+"attn_q_norm.weight", TensorF32, headDim)
	add(p+"attn_k_norm.weight", TensorF32, headDim)
	add(p+"ffn_gate.weight", TensorQ4_K, h, ffn)
	add(p+"ffn_up.weight", TensorQ4_K, h, ffn)
	add(p+"ffn_down.weight", TensorQ4_K, ffn, h)

	meta := synthQwen35Meta(4, 0)
	set := func(k string, v uint64) { meta["qwen35."+k] = Value{Type: TypeUint64, Value: v} }
	set("embedding_length", h)
	set("feed_forward_length", ffn)
	set("attention.head_count", heads)
	set("attention.head_count_kv", kvHeads)
	set("attention.key_length", headDim)
	set("attention.value_length", headDim)
	set("rope.dimension_count", 64)
	set("ssm.conv_kernel", convKern)
	set("ssm.state_size", linHD)
	set("ssm.group_count", nK)
	set("ssm.time_step_rank", nV)
	set("ssm.inner_size", valDim)
	toks := make([]Value, vocab)
	for i := range toks {
		toks[i] = Value{Type: TypeString, Value: "t" + strconv.Itoa(i)}
	}
	meta["tokenizer.ggml.tokens"] = Value{Type: TypeArray, Value: toks}

	var payload bytes.Buffer
	infos := make([]TensorInfo, 0, len(tensors))
	for i, tt := range tensors {
		info := TensorInfo{Name: tt.name, Dims: tt.dims, Type: tt.typ, FileOffset: int64(payload.Len())}
		n, err := tensorPayloadBytes(info)
		if err != nil {
			t.Fatalf("tensorPayloadBytes(%s): %v", tt.name, err)
		}
		var raw []byte
		if tt.typ == TensorF32 {
			elems := int(n / 4)
			vals := tinyQwen35Data(tt.name, elems)
			switch {
			case strings.HasSuffix(tt.name, ".ssm_a"):
				for j := range vals {
					vals[j] = -float32(math.Exp(float64(0.25 * float32(j))))
				}
			case strings.HasSuffix(tt.name, "norm.weight"):
				for j := range vals {
					vals[j] = 1 + vals[j] // qwen35 GGUF norms store 1+gain
				}
			case tt.name == "token_embd.weight":
				for j := range vals {
					vals[j] *= 20
				}
			}
			raw = make([]byte, n)
			for j, v := range vals {
				binary.LittleEndian.PutUint32(raw[4*j:], math.Float32bits(v))
			}
		} else {
			raw = randKQuantRaw(t, tt.typ, int(tt.dims[1]), int(tt.dims[0]), int64(7000+i))
		}
		if int64(len(raw)) != int64(n) {
			t.Fatalf("%s payload %d, want %d", tt.name, len(raw), n)
		}
		infos = append(infos, info)
		payload.Write(raw)
	}
	ws, err := NewWeightSource(&File{Metadata: meta, Tensors: infos}, bytes.NewReader(payload.Bytes()), int64(payload.Len()))
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// TestQwen35NativeRowsLoaderEstimatorAndForward is the end-to-end witness on a GGUF fixture:
// (1) the loader routes every class to the documented store; (2) the resident bytes equal the
// estimator exactly, with the route on AND off, and the route saves bytes; (3) the stored
// native rows decode to dequant+normalize bit-exactly; (4) prefill and decode logits served
// from native rows match the Q8-served baseline within quant tolerance and argmax.
func TestQwen35NativeRowsLoaderEstimatorAndForward(t *testing.T) {
	load := func(opts ...Q4KLoadOption) (*model.Model, int64) {
		ws := qwen35NativeRowsFixture(t)
		plan, err := ws.EstimateQ4KLoadMemoryPlan(opts...)
		if err != nil {
			t.Fatalf("estimate: %v", err)
		}
		m, err := ws.QuantModelQ4KProfileOptions(nil, opts...)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		t.Cleanup(func() { _ = m.CloseWeights() })
		if got := m.ResidentReport().TotalResidentBytes; got != plan.Total() {
			t.Fatalf("estimate %d != loaded resident %d (opts=%d)", plan.Total(), got, len(opts))
		}
		return m, plan.Total()
	}
	native, nativeBytes := load()
	baseline, baselineBytes := load(WithNativeLinearAttnRows(false))
	if nativeBytes >= baselineBytes {
		t.Fatalf("native-row resident %d not below Q8 baseline %d", nativeBytes, baselineBytes)
	}

	lp := func(l int, s string) string { return "model.layers." + strconv.Itoa(l) + "." + s }
	type route struct {
		name  string
		store string
	}
	routes := []route{
		{lp(0, "linear_attn.in_proj_qkv.weight"), "q4k"},
		{lp(1, "linear_attn.in_proj_qkv.weight"), "kq"},
		{lp(2, "linear_attn.in_proj_qkv.weight"), "q8"}, // Q5_K stays on Q8
		{lp(0, "linear_attn.in_proj_z.weight"), "q4k"},
		{lp(1, "linear_attn.in_proj_a.weight"), "q4k"},
		{lp(2, "linear_attn.in_proj_b.weight"), "q4k"},
		{lp(0, "linear_attn.out_proj.weight"), "q8"}, // input-column transform
		{lp(3, "self_attn.q_proj.weight"), "q4k"},
		{lp(3, "self_attn.k_proj.weight"), "q4k"},
	}
	storeOf := func(m *model.Model, name string) string {
		switch {
		case m.HasQ4K(name):
			return "q4k"
		case m.HasKQuant(name):
			return "kq"
		case m.HasQ8(name):
			return "q8"
		}
		return "none"
	}
	for _, r := range routes {
		if got := storeOf(native, r.name); got != r.store {
			t.Errorf("native %s store = %s, want %s", r.name, got, r.store)
		}
		if got := storeOf(baseline, r.name); got != "q8" {
			t.Errorf("baseline %s store = %s, want q8", r.name, got)
		}
	}

	// (3) stored native bytes decode to the production float normalization of the GGUF tensor.
	ws := qwen35NativeRowsFixture(t)
	cfg := native.Cfg
	for _, c := range []struct{ gguf, canon string }{
		{"blk.0.attn_qkv.weight", lp(0, "linear_attn.in_proj_qkv.weight")},
		{"blk.1.attn_qkv.weight", lp(1, "linear_attn.in_proj_qkv.weight")},
		{"blk.0.attn_gate.weight", lp(0, "linear_attn.in_proj_z.weight")},
		{"blk.3.attn_q.weight", lp(3, "self_attn.q_proj.weight")},
	} {
		raw, info, err := ws.TensorBytes(c.gguf)
		if err != nil {
			t.Fatal(err)
		}
		rows, in := int(info.Dims[1]), int(info.Dims[0])
		srcCanon, _ := CanonicalTensorNameArch(c.gguf, cfg.ModelType)
		want, err := normalizeCanonicalTensorData(srcCanon, dequantForNativeRowsTest(t, info.Type, raw, rows, in), cfg)
		if err != nil {
			t.Fatal(err)
		}
		stored, ok := native.Q4KRaw(c.canon)
		if !ok {
			stored, ok = native.KQuantRaw(c.canon)
		}
		if !ok {
			t.Fatalf("%s not resident", c.canon)
		}
		got := dequantForNativeRowsTest(t, info.Type, stored[:len(raw)], rows, in)
		for j := range want {
			if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
				t.Fatalf("%s element %d = %v, want %v", c.canon, j, got[j], want[j])
			}
		}
	}

	// (4) forward parity: native rows vs the same weights requantized to Q8.
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 1, 2, 4, 6, 8, 10, 12}
	follow := []int{9, 13, 14, 15}
	run := func(m *model.Model) [][]float32 {
		s := m.NewSession()
		s.Q4K = true
		// Copy each result: a session may reuse its logits buffer across calls.
		out := [][]float32{append([]float32(nil), s.Prefill(prompt)...)}
		for _, id := range follow {
			out = append(out, append([]float32(nil), s.Step(id)...))
		}
		return out
	}
	got, want := run(native), run(baseline)
	for i := range want {
		if len(got[i]) != cfg.VocabSize || len(want[i]) != cfg.VocabSize {
			t.Fatalf("step %d logits len %d/%d, want %d", i, len(got[i]), len(want[i]), cfg.VocabSize)
		}
		var dot, ng, nw float64
		for j := range want[i] {
			g, w := float64(got[i][j]), float64(want[i][j])
			if math.IsNaN(g) || math.IsInf(g, 0) {
				t.Fatalf("step %d native logit %d not finite: %v", i, j, g)
			}
			dot, ng, nw = dot+g*w, ng+g*g, nw+w*w
		}
		cos := dot / math.Sqrt(ng*nw)
		t.Logf("step %d native-vs-Q8 logits cosine %.7f argmax %d/%d", i, cos, argmaxF32(got[i]), argmaxF32(want[i]))
		if cos < 0.9999 {
			t.Errorf("step %d native-vs-Q8 logits cosine %.6f < 0.9999", i, cos)
		}
		if a, b := argmaxF32(got[i]), argmaxF32(want[i]); a != b {
			t.Errorf("step %d argmax native=%d Q8=%d", i, a, b)
		}
	}
}
