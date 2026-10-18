package model

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func dflash2TinyGeometry() dflash2Geometry {
	return dflash2Geometry{32, 64, 4, 2, 8, 97, 4, 4, 4, 9, 0, 1e-6, 10000, []int{0, 2, 3}}
}

// This is a real BF16 safetensors fixture for all81 weights, opened by the same
// validated native owner as the stock artifact. It does not replace forward math.
func dflash2TinyWeights(t *testing.T, g dflash2Geometry) *dflash2Weights {
	return dflash2TinyWeightsWithValue(t, g, nil)
}

func dflash2TinyWeightsWithValue(t *testing.T, g dflash2Geometry, override func(string, int) float32) *dflash2Weights {
	t.Helper()
	shapes := map[string][]int{
		"fc.weight":          {g.hidden, len(g.taps) * g.hidden},
		"hidden_norm.weight": {g.hidden}, "norm.weight": {g.hidden},
		"candidate_selector.hidden_projection.weight": {g.rank, g.hidden},
		"candidate_selector.predecessor_codebook":     {g.vocab, g.rank},
		"candidate_selector.successor_codebook":       {g.vocab, g.rank},
	}
	layer := map[string][]int{
		"input_layernorm.weight": {g.hidden}, "post_attention_layernorm.weight": {g.hidden},
		"self_attn.q_proj.weight": {g.heads * g.headDim, g.hidden},
		"self_attn.k_proj.weight": {g.kvHeads * g.headDim, g.hidden},
		"self_attn.v_proj.weight": {g.kvHeads * g.headDim, g.hidden},
		"self_attn.o_proj.weight": {g.hidden, g.heads * g.headDim},
		"self_attn.q_norm.weight": {g.headDim}, "self_attn.k_norm.weight": {g.headDim},
		"mlp.gate_proj.weight": {g.intermediate, g.hidden}, "mlp.up_proj.weight": {g.intermediate, g.hidden},
		"mlp.down_proj.weight":                    {g.hidden, g.intermediate},
		"attention_conv.kernel_projection.weight": {4 * (g.hidden / g.group), g.hidden},
		"mlp_conv.kernel_projection.weight":       {4 * (g.hidden / g.group), g.hidden},
		"attention_conv.base_kernel":              {2, 2, g.hidden}, "mlp_conv.base_kernel": {2, 2, g.hidden},
	}
	for i := 0; i < 5; i++ {
		for name, shape := range layer {
			shapes[fmt.Sprintf("layers.%d.%s", i, name)] = shape
		}
	}
	if len(shapes) != 81 { //boundarylint:ignore CHANGE_DETECTOR_TEST pinned five-layer artifact requires 6 global plus 5*15 layer tensors
		t.Fatal("fixture must contain all81 tensors")
	}
	stockShapes := qwen38DFlash2WeightContract().shapes
	if len(shapes) != len(stockShapes) {
		t.Fatalf("fixture tensor count=%d, stock contract=%d", len(shapes), len(stockShapes))
	}
	for name := range stockShapes {
		if _, ok := shapes[name]; !ok {
			t.Fatalf("fixture is missing stock tensor %q", name)
		}
	}
	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	sort.Strings(names)
	header := make(map[string]any, len(shapes))
	var payload []byte
	for index, name := range names {
		count := 1
		for _, dim := range shapes[name] {
			count *= dim
		}
		start := len(payload)
		for j := 0; j < count; j++ {
			value := float32((j*3+index*5)%11-5) / 128
			if strings.Contains(name, "norm.weight") {
				value = 1
			}
			if strings.HasSuffix(name, "base_kernel") {
				value = 1
				if (j/g.hidden)%2 == 1 {
					value = 0.125
				}
			}
			if override != nil {
				value = override(name, j)
			}
			var bits [2]byte
			binary.LittleEndian.PutUint16(bits[:], uint16(math.Float32bits(value)>>16))
			payload = append(payload, bits[:]...)
		}
		header[name] = map[string]any{"dtype": "BF16", "shape": shapes[name], "data_offsets": []int{start, len(payload)}}
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	file := make([]byte, 8)
	binary.LittleEndian.PutUint64(file, uint64(len(encoded)))
	file = append(file, encoded...)
	file = append(file, payload...)
	digest := sha256.Sum256(file)
	w, err := newDFlash2Weights(context.Background(), bytes.NewReader(file), int64(len(file)), nil, dflash2WeightContract{fmt.Sprintf("%x", digest), int64(len(file)), shapes})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := w.Close(); err != nil {
			t.Errorf("weights Close: %v", err)
		}
	})
	return w
}

func dflash2ReferenceFixture(t *testing.T) (*dflash2Reference, *Session, []float32) {
	t.Helper()
	m := qwen35TargetFeatureTestModel(t)
	target, plain := m.NewSession(), m.NewSession()
	t.Cleanup(target.Close)
	t.Cleanup(plain.Close)
	target.captureTargetHidden, plain.captureTargetHidden = true, true
	g := dflash2TinyGeometry()
	f, err := newDFlash2Reference(dflash2TinyWeights(t, g), target, g, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Errorf("reference Close: %v", err)
		}
	})
	// Teacher-force the known prompt; generation begins at its real logits.
	prompt := []int{3, 7, 11}
	tx, err := beginQwen35TargetFeatureTransaction(target, make([]float32, g.vocab), g.taps)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Abort() })
	if _, err := tx.Verify(context.Background(), prompt); err != nil {
		t.Fatal(err)
	}
	before, rows, err := tx.Commit(context.Background(), len(prompt))
	if err != nil {
		t.Fatal(err)
	}
	if err := f.AppendCommitted(context.Background(), rows); err != nil {
		t.Fatal(err)
	}
	want := plain.Prefill(prompt)
	assertFloat32BitsEqual(t, "seed boundary", want, before)
	normalizeSnapshotForTest(t, target)
	normalizeSnapshotForTest(t, plain)
	return f, plain, before
}

type dflash2ReadObserver struct {
	reader  io.ReaderAt
	observe func(int64) error
}

func (r dflash2ReadObserver) ReadAt(p []byte, offset int64) (int, error) {
	if err := r.observe(offset); err != nil {
		return 0, err
	}
	return r.reader.ReadAt(p, offset)
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2ReferenceFiveLayersExactGreedyContinuation(t *testing.T) {
	f, plain, before := dflash2ReferenceFixture(t)
	ctx := context.Background()
	seen := make(map[int64]bool)
	f.w.r = dflash2ReadObserver{f.w.r, func(offset int64) error { seen[offset] = true; return nil }}
	want := append([]float32(nil), before...)
	for round := 0; round < 12; round++ {
		budget := 1 + round%8
		tokens, after, err := f.GreedyRound(ctx, before, budget)
		if err != nil || len(tokens) < 1 || len(tokens) > budget {
			t.Fatalf("round%d tokens=%v err=%v", round, tokens, err)
		}
		for _, token := range tokens {
			if token != argmaxF32(want) {
				t.Fatalf("round%d token differs from plain greedy", round)
			}
			want = plain.Step(token)
		}
		assertFloat32BitsEqual(t, "round boundary", want, after)
		assertQwen35MTPTargetStateEqual(t, f.target, plain)
		if f.nextPosition != f.target.Cache.Len() || len(f.prefixTokens) != f.nextPosition {
			t.Fatal("context publication boundary differs")
		}
		for layer, cache := range f.cache {
			rows := len(cache.keys) / (f.g.kvHeads * f.g.headDim)
			if rows != min(f.nextPosition, f.g.window-1) || len(cache.keys) != len(cache.values) {
				t.Fatalf("layer%d ring rows=%d", layer, rows)
			}
		}
		before = after
	}
	for layer := 0; layer < 5; layer++ {
		name := fmt.Sprintf("layers.%d.self_attn.q_proj.weight", layer)
		if !seen[f.w.tensors[name].offset] {
			t.Fatalf("layer%d was not executed", layer)
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2ReferenceRoundFailureRestoresBothOwners(t *testing.T) {
	for _, mode := range []string{"cancel", "read-error", "panic"} {
		t.Run(mode, func(t *testing.T) {
			f, plain, before := dflash2ReferenceFixture(t)
			oldCache, oldPrefix, oldPosition := f.cache, append([]uint32(nil), f.prefixTokens...), f.nextPosition
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			original := f.w.r
			trigger := f.w.tensors["layers.4.self_attn.k_proj.weight"].offset
			fired := false
			f.w.r = dflash2ReadObserver{original, func(offset int64) error {
				// Fail late in draft-context commit, after the real target committed.
				if !fired && offset == trigger && f.target.Cache.Len() > oldPosition {
					fired = true
					switch mode {
					case "cancel":
						cancel()
					case "read-error":
						return io.ErrUnexpectedEOF
					case "panic":
						panic("injected final-layer context read")
					}
				}
				return nil
			}}
			tokens, logits, err := f.GreedyRound(ctx, before, 8)
			if !fired || err == nil || tokens != nil || logits != nil {
				t.Fatalf("failed round published: fired=%v err=%v", fired, err)
			}
			if mode == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel identity lost: %v", err)
			}
			if f.nextPosition != oldPosition || !reflect.DeepEqual(oldPrefix, f.prefixTokens) || !reflect.DeepEqual(oldCache, f.cache) {
				t.Fatal("failed round changed draft owner")
			}
			assertQwen35MTPTargetStateEqual(t, f.target, plain)
			if f.target.qwen35TargetFeatureTxn != nil || f.target.qwen35TargetLayerCapture != nil {
				t.Fatal("failed round retained target transaction")
			}
			f.w.r = original
			tokens, after, err := f.GreedyRound(context.Background(), before, 8)
			if err != nil {
				t.Fatalf("reusable after failure: %v", err)
			}
			want := before
			for _, token := range tokens {
				if token != argmaxF32(want) {
					t.Fatal("reused token differs")
				}
				want = plain.Step(token)
			}
			assertFloat32BitsEqual(t, "reused boundary", want, after)
			assertQwen35MTPTargetStateEqual(t, f.target, plain)
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2ReferenceProposalOwnershipLineageAndEOS(t *testing.T) {
	f, plain, before := dflash2ReferenceFixture(t)
	oldCache, oldPosition := f.cache, f.nextPosition
	first, err := f.Propose(context.Background(), argmaxF32(before), 7)
	if err != nil {
		t.Fatal(err)
	}
	second, err := f.Propose(context.Background(), argmaxF32(before), 7)
	if err != nil || !reflect.DeepEqual(first, second) {
		t.Fatal("proposal repeat changed")
	}
	if !reflect.DeepEqual(oldCache, f.cache) || f.nextPosition != oldPosition {
		t.Fatal("proposal mutated context")
	}
	assertQwen35MTPTargetStateEqual(t, f.target, plain)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if out, err := f.Propose(ctx, argmaxF32(before), 7); !errors.Is(err, context.Canceled) || out != nil {
		t.Fatal("canceled proposal exposed output")
	}
	old := f.target.Cache.lineage.ids[0]
	f.target.Cache.lineage.ids[0] = (old + 1) % uint32(f.g.vocab)
	if _, err := f.Propose(context.Background(), argmaxF32(before), 1); err == nil {
		t.Fatal("same-length different prefix accepted")
	}
	f.target.Cache.lineage.ids[0] = old
	f.target.M.Cfg.EOSTokenID = argmaxF32(before)
	tokens, after, err := f.GreedyRound(context.Background(), before, 8)
	if err != nil || len(tokens) != 1 || tokens[0] != f.target.M.Cfg.EOSTokenID {
		t.Fatalf("EOS budget: %v %v", tokens, err)
	}
	want := plain.Step(tokens[0])
	assertFloat32BitsEqual(t, "EOS boundary", want, after)
	assertQwen35MTPTargetStateEqual(t, f.target, plain)
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Propose(context.Background(), 0, 1); err == nil {
		t.Fatal("closed reference reused")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDFlash2ReferenceConvolutionAndAbsoluteRoPE(t *testing.T) {
	// Two channels share each dynamic coefficient; pre/post sides differ.
	x := []float32{1, 2, 3, 4, 5, 6}
	dynamic := []float32{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
	base := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	for side := 0; side < 2; side++ {
		got := dflash2Convolve(x, dynamic, base, 2, 2, side)
		want := make([]float32, len(x))
		for row := 0; row < 3; row++ {
			for channel := 0; channel < 2; channel++ {
				for tap := 0; tap < 2; tap++ {
					if row >= tap {
						want[row*2+channel] += (base[(side*2+tap)*2+channel] + dynamic[row*4+side*2+tap]) * x[(row-tap)*2+channel]
					}
				}
			}
		}
		assertFloat32BitsEqual(t, "grouped conv", want, got)
	}
	rotated := []float32{1, 2, 3, 4}
	dflash2RoPE(rotated, 1, 4, 5, 10000)
	for j := 0; j < 2; j++ {
		angle := float64(5) / math.Pow(10000, float64(j)/2)
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		u, v := float32(j+1), float32(j+3)
		if rotated[j] != u*c-v*s || rotated[j+2] != u*s+v*c {
			t.Fatal("RoPE split-half absolute position differs")
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2ReferenceRefusesIncompleteContextTail(t *testing.T) {
	f, _, before := dflash2ReferenceFixture(t)
	for _, first := range []int{1, f.target.Cache.Len()} {
		w := dflash2TinyWeights(t, f.g)
		if other, err := newDFlash2Reference(w, f.target, f.g, first); err == nil {
			_ = other.Close()
			t.Fatalf("underseeded constructor first=%d succeeded", first)
		}
	}
	for layer := range f.cache {
		old := f.cache[layer]
		f.cache[layer].keys = old.keys[:len(old.keys)-f.g.kvHeads*f.g.headDim]
		if _, err := f.Propose(context.Background(), argmaxF32(before), 1); err == nil {
			t.Fatalf("layer%d truncated keys accepted", layer)
		}
		if _, _, err := f.GreedyRound(context.Background(), before, 1); err == nil {
			t.Fatal("anchor-only round accepted incomplete context")
		}
		f.cache[layer] = old
		f.cache[layer].values = nil
		if _, err := f.Propose(context.Background(), argmaxF32(before), 1); err == nil {
			t.Fatalf("layer%d missing values accepted", layer)
		}
		f.cache[layer] = old
	}
	// Even a legal late starting position cannot propose before its complete tail
	// has been projected. Target lineage alone is insufficient.
	for i := 0; i < 9; i++ {
		f.target.Step(13 + i)
	}
	first := f.target.Cache.Len() - (f.g.window - 1)
	late, err := newDFlash2Reference(dflash2TinyWeights(t, f.g), f.target, f.g, first)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := late.Close(); err != nil {
			t.Error(err)
		}
	})
	if _, err := late.Propose(context.Background(), 7, 1); err == nil {
		t.Fatal("unseeded legal tail proposed")
	}
	if other, err := newDFlash2Reference(dflash2TinyWeights(t, f.g), f.target, f.g, first+1); err == nil {
		_ = other.Close()
		t.Fatal("short tail constructor accepted")
	}
}

func dflash2OracleValue(g dflash2Geometry, name string, j int) float32 {
	if strings.Contains(name, "norm.weight") {
		return 1
	}
	if strings.HasSuffix(name, "base_kernel") {
		if (j/g.hidden)%2 == 0 {
			return 1
		}
		return 0
	}
	columns := 0
	switch {
	case strings.HasSuffix(name, "self_attn.v_proj.weight"):
		columns = g.hidden
	case strings.HasSuffix(name, "self_attn.o_proj.weight"):
		columns = g.heads * g.headDim
	case strings.HasSuffix(name, "mlp.gate_proj.weight"), strings.HasSuffix(name, "mlp.up_proj.weight"):
		columns = g.hidden
	case strings.HasSuffix(name, "mlp.down_proj.weight"):
		columns = g.intermediate
	}
	if columns > 0 && j/columns == j%columns {
		return 1
	}
	return 0 // Q/K and dynamic convolutions zero: attention has a closed-form mean.
}

func dflash2OracleReference(t *testing.T) *dflash2Reference {
	t.Helper()
	m := qwen35TargetFeatureTestModel(t)
	target := m.NewSession()
	t.Cleanup(target.Close)
	g := dflash2TinyGeometry()
	w := dflash2TinyWeightsWithValue(t, g, func(name string, j int) float32 { return dflash2OracleValue(g, name, j) })
	f, err := newDFlash2Reference(w, target, g, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	return f
}

// A separate scalar closed form, deliberately not using linear/norm/attention/
// convolution helpers from the implementation. Five residual+SwiGLU stages are
// visible even though the oracle's Q/K and dynamic kernels are zero.
func dflash2FiveLayerClosedForm(x []float32, g dflash2Geometry, layers int) []float32 {
	normalize := func(in []float32) []float32 {
		out := make([]float32, len(in))
		for start := 0; start < len(in); start += g.hidden {
			var squared float64
			for _, v := range in[start : start+g.hidden] {
				squared += float64(v) * float64(v)
			}
			inverse := 1 / math.Sqrt(squared/float64(g.hidden)+g.eps)
			for i, v := range in[start : start+g.hidden] {
				out[start+i] = float32(float64(v) * inverse)
			}
		}
		return out
	}
	x = append([]float32(nil), x...)
	rows := len(x) / g.hidden
	for layer := 0; layer < layers; layer++ {
		n := normalize(x)
		for channel := 0; channel < g.hidden; channel++ {
			kvChannel := (channel/g.headDim)/(g.heads/g.kvHeads)*g.headDim + channel%g.headDim
			var mean float64
			for row := 0; row < rows; row++ {
				mean += float64(n[row*g.hidden+kvChannel]) / float64(rows)
			}
			for row := 0; row < rows; row++ {
				x[row*g.hidden+channel] += float32(mean)
			}
		}
		n = normalize(x)
		for i, value := range n {
			z := float64(value)
			x[i] += float32(z * z / (1 + math.Exp(-z)))
		}
	}
	return normalize(x)[g.hidden:]
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2ReferenceIndependentFiveLayerAndAttentionOracle(t *testing.T) {
	f := dflash2OracleReference(t)
	const anchor, proposals = 7, 2
	input := make([]float32, 0, (proposals+1)*f.g.hidden)
	for row := 0; row <= proposals; row++ {
		token := f.g.mask
		if row == 0 {
			token = anchor
		}
		embedding, err := f.target.TokenEmbedding(token)
		if err != nil {
			t.Fatal(err)
		}
		input = append(input, embedding...)
	}
	got, err := f.hidden(context.Background(), anchor, proposals)
	if err != nil {
		t.Fatal(err)
	}
	want := dflash2FiveLayerClosedForm(input, f.g, 5)
	four := dflash2FiveLayerClosedForm(input, f.g, 4)
	var discriminates float64
	for i, value := range got {
		if math.Abs(float64(value-want[i])) > 2e-4 {
			t.Fatalf("five-layer hidden[%d]=%g, independent oracle%g", i, value, want[i])
		}
		discriminates = math.Max(discriminates, math.Abs(float64(want[i]-four[i])))
	}
	if discriminates < 1e-3 {
		t.Fatal("oracle does not distinguish five layers from four")
	}
	// Eight committed context values, all three proposal positions visible.
	// Later queries drop the oldest context keys under the sliding window.
	stride := f.g.kvHeads * f.g.headDim
	f.cache[0] = dflash2ContextKV{make([]float32, 8*stride), make([]float32, 8*stride)}
	for row := 0; row < 8; row++ {
		for c := 0; c < stride; c++ {
			f.cache[0].values[row*stride+c] = float32(row + 1 + 100*(c/f.g.headDim))
		}
	}
	f.nextPosition = 8
	block := make([]float32, 3*f.g.hidden)
	for row := 0; row < 3; row++ {
		for c := 0; c < f.g.hidden; c++ {
			block[row*f.g.hidden+c] = float32((row+1)*10 + 100*(c/f.g.headDim))
		}
	}
	attention, err := f.attention(context.Background(), 0, block)
	if err != nil {
		t.Fatal(err)
	}
	means := []float64{96.0 / 11, 95.0 / 10, 93.0 / 9}
	for i, value := range attention {
		kvHead := ((i % f.g.hidden) / f.g.headDim) / (f.g.heads / f.g.kvHeads)
		if math.Abs(float64(value)-(means[i/f.g.hidden]+100*float64(kvHead))) > 4e-5 {
			t.Fatalf("sliding noncausal attention[%d]=%g", i, value)
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2ReferenceIndependentMarkovSelectorOracle(t *testing.T) {
	m := qwen35TargetFeatureTestModel(t)
	target := m.NewSession()
	t.Cleanup(target.Close)
	g := dflash2TinyGeometry()
	hidden := make([]float32, 2*g.hidden)
	hidden[0], hidden[g.hidden] = 1, 1
	unary := target.headResident(hidden[:g.hidden])
	order := make([]int, g.vocab)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		if unary[order[i]] == unary[order[j]] {
			return order[i] < order[j]
		}
		return unary[order[i]] > unary[order[j]]
	})
	best, chosen := order[0], order[1]
	anchor := (chosen + 1) % g.vocab
	if unary[best]-unary[chosen] >= 64 {
		t.Fatal("fixture Markov edge cannot flip unary winner")
	}
	w := dflash2TinyWeightsWithValue(t, g, func(name string, j int) float32 {
		switch name {
		case "candidate_selector.hidden_projection.weight":
			if j == 0 {
				return 1
			}
		case "candidate_selector.predecessor_codebook":
			if j == anchor*g.rank {
				return 1
			}
			if j == chosen*g.rank {
				return -1
			}
		case "candidate_selector.successor_codebook":
			if j == chosen*g.rank {
				return 64
			}
		}
		return 0
	})
	f, err := newDFlash2Reference(w, target, g, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	got, err := f.selectCandidates(context.Background(), anchor, hidden)
	if err != nil || !reflect.DeepEqual(got, []int{chosen, best}) {
		t.Fatalf("Markov greedy path=%v err=%v, want[%d %d]", got, err, chosen, best)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestDFlash2ReferenceFullEightTokenAcceptance(t *testing.T) {
	m := qwen35TargetFeatureTestModel(t)
	// Independent zero untied head makes exact greedy token0 without zeroing
	// embeddings, attention, FFN or recurrent state. Set it before any owner opens.
	m.Cfg.TieWordEmbeddings = false
	bytes := m.Cfg.HiddenSize * m.Cfg.VocabSize * 4
	offset := len(m.raw)
	m.raw = append(m.raw, make([]byte, bytes)...)
	m.manifest["lm_head.weight"] = tensorMeta{Dtype: "F32", Shape: []int{m.Cfg.VocabSize, m.Cfg.HiddenSize}, Offset: offset, Nbytes: bytes}
	target, plain := m.NewSession(), m.NewSession()
	t.Cleanup(target.Close)
	t.Cleanup(plain.Close)
	target.captureTargetHidden, plain.captureTargetHidden = true, true
	g := dflash2TinyGeometry()
	w := dflash2TinyWeightsWithValue(t, g, func(name string, j int) float32 {
		if strings.HasPrefix(name, "candidate_selector.") {
			return 0
		}
		return dflash2OracleValue(g, name, j)
	})
	f, err := newDFlash2Reference(w, target, g, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	prompt := []int{3, 7, 11}
	tx, err := beginQwen35TargetFeatureTransaction(target, make([]float32, g.vocab), g.taps)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Abort() })
	if _, err := tx.Verify(context.Background(), prompt); err != nil {
		t.Fatal(err)
	}
	before, rows, err := tx.Commit(context.Background(), len(prompt))
	if err != nil {
		t.Fatal(err)
	}
	if err := f.AppendCommitted(context.Background(), rows); err != nil {
		t.Fatal(err)
	}
	want := plain.Prefill(prompt)
	normalizeSnapshotForTest(t, target)
	normalizeSnapshotForTest(t, plain)
	const requestedTokens = 8
	tokens, after, err := f.GreedyRound(context.Background(), before, requestedTokens)
	if err != nil || len(tokens) != requestedTokens {
		t.Fatalf("full acceptance tokens=%v want=%d err=%v", tokens, requestedTokens, err)
	}
	for _, token := range tokens {
		if token != 0 || token != argmaxF32(want) {
			t.Fatal("full-accept token differs from target")
		}
		want = plain.Step(token)
	}
	assertFloat32BitsEqual(t, "full-accept boundary", want, after)
	assertQwen35MTPTargetStateEqual(t, target, plain)
	if f.nextPosition != len(prompt)+requestedTokens {
		t.Fatal("full acceptance did not commit all rows")
	}
	if len(f.cache[0].keys) != (g.window-1)*g.kvHeads*g.headDim {
		t.Fatal("full acceptance did not retain complete context tail")
	}
	if _, err := f.Propose(context.Background(), 0, 1); err != nil {
		t.Fatalf("full-accept context not reusable: %v", err)
	}
}
