package model

import (
	"encoding/binary"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

func TestQwen38MTPQ4KForwardMatchesDequantizedReference(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	q4, ref := qwen38MTPQ4KTestModels(t)
	q4Forward, err := q4.NewQwen35MTPForward()
	if err != nil {
		t.Fatalf("construct Q4_K MTP forward: %v", err)
	}
	t.Cleanup(q4Forward.Close)
	// The cross-platform parity witness compares the resident CPU Q4_K path to
	// its dequantized oracle. The Darwin-only test separately proves Metal ran.
	q4Forward.draft.MetalQ4K = false

	refForward, err := ref.NewQwen35MTPForward()
	if err != nil {
		t.Fatalf("construct F32 oracle MTP forward: %v", err)
	}
	t.Cleanup(refForward.Close)

	prior, embedding := qwen38MTPInputs(q4.Cfg.HiddenSize, 0)
	got, err := q4Forward.Forward(0, prior, embedding)
	if err != nil {
		t.Fatalf("execute Q4_K MTP forward: %v", err)
	}
	want, err := refForward.Forward(0, prior, embedding)
	if err != nil {
		t.Fatalf("execute dequantized oracle MTP forward: %v", err)
	}
	if cos := cosine(got, want); cos < 0.99999 {
		t.Fatalf("Q4_K MTP forward cosine=%.8f, want >= 0.99999", cos)
	}
	if argmaxF32(got) != argmaxF32(want) {
		t.Fatalf("Q4_K MTP argmax=%d, oracle=%d", argmaxF32(got), argmaxF32(want))
	}
	for _, name := range qwen38MTPMatrixTensors {
		if _, ok := q4.manifest[name]; ok {
			t.Fatalf("%s retained a persistent F32 matrix beside Q4_K execution", name)
		}
		if q4.q4kw[name] == nil {
			t.Fatalf("%s missing from resident Q4_K store", name)
		}
	}
}

func TestQwen38MTPQ4KSpeculativeAcceptanceAndMechanismReceipt(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	q4, ref := qwen38MTPQ4KTestModels(t)
	q4Forward, err := q4.NewQwen35MTPForward()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(q4Forward.Close)
	q4Forward.draft.MetalQ4K = false
	refForward, err := ref.NewQwen35MTPForward()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(refForward.Close)

	const depth = 3
	draft := func([]int) []int {
		out := make([]int, depth)
		for pos := range out {
			prior, embedding := qwen38MTPInputs(q4.Cfg.HiddenSize, pos)
			logits, err := q4Forward.Forward(pos, prior, embedding)
			if err != nil {
				t.Fatalf("Q4_K draft position %d: %v", pos, err)
			}
			out[pos] = argmaxF32(logits)
		}
		return out
	}
	verify := func(_ []int, proposed []int) []int {
		rows := make([]int, 0, len(proposed)+1)
		for pos := range proposed {
			prior, embedding := qwen38MTPInputs(ref.Cfg.HiddenSize, pos)
			logits, err := refForward.Forward(pos, prior, embedding)
			if err != nil {
				t.Fatalf("F32 verifier position %d: %v", pos, err)
			}
			rows = append(rows, argmaxF32(logits))
		}
		return append(rows, 0)
	}
	run, err := polymodel.SpecDecode([]int{1}, draft, verify, polymodel.SpecDecodeConfig{
		MaxNewTokens: depth,
		MaxDraft:     depth,
	})
	if err != nil {
		t.Fatalf("Q4_K MTP speculative acceptance: %v", err)
	}
	if run.Rounds != 1 || run.DraftedTokens != depth || run.AcceptedDrafts != depth || run.EvictKV != 0 {
		t.Fatalf("speculative accounting=%+v, want one fully accepted depth-%d round", run, depth)
	}

	receipt := EvaluateQwen38MTPEligibility(Qwen38MTPEligibilityInput{
		Qwen38MTPArtifact: true,
		MTPBackendReady:   true,
		Backend:           Qwen38MTPBackendMetal,
		Model:             q4,
		Greedy:            true,
		Depth:             depth,
		FreshSession:      true,
		MemoryHeadroomOK:  true,
		OperatorEnabled:   true,
	})
	if !receipt.Eligible ||
		receipt.Engine != Qwen38EngineMTP ||
		receipt.Backend != Qwen38MTPBackendMetal ||
		receipt.MTPTensorFormat != Qwen38MTPFormatQ4K ||
		receipt.RequestedDepth != depth ||
		!receipt.TargetEquivalent {
		t.Fatalf("Q4_K mechanism receipt=%+v", receipt)
	}
}

func qwen38MTPQ4KTestModels(t *testing.T) (q4, ref *Model) {
	t.Helper()
	cfg := qwen35MTPTestConfig()
	cfg.Name = "Qwen3.8 Q4_K MTP mechanism fixture"
	cfg.NumLayers = 1
	cfg.LayerTypes = []string{"full_attention"}
	cfg.HiddenSize = qkK
	cfg.NumHeads = 4
	cfg.NumKVHeads = 2
	cfg.HeadDim = 64
	cfg.IntermediateSize = qkK
	cfg.VocabSize = 8
	cfg.RMSNormEps = 1e-5
	cfg.RopeTheta = 10000
	cfg.AttnOutputGate = true
	cfg.QKNorm = true

	shapes, err := qwen35MTPExpectedShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	q4 = &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q4kw: map[string]*q4kTensor{}}
	ref = &Model{Cfg: cfg, manifest: map[string]tensorMeta{}}

	for i, name := range qwen38MTPNormTensors {
		data := make([]float32, shapes[name][0])
		for j := range data {
			data[j] = 0.75 + float32((i+j)%5)/16
		}
		appendQwen38MTPF32Tensor(q4, name, shapes[name], data)
		appendQwen38MTPF32Tensor(ref, name, shapes[name], data)
	}

	rng := rand.New(rand.NewSource(9985))
	for _, name := range qwen38MTPMatrixTensors {
		shape := shapes[name]
		raw := make([]byte, shape[0]*(shape[1]/qkK)*q4kBlockBytes)
		for block := 0; block < len(raw)/q4kBlockBytes; block++ {
			randQ4KBlockBounded(rng, raw[block*q4kBlockBytes:(block+1)*q4kBlockBytes], 2, 5)
		}
		q4.q4kw[name] = quantizeQ4KFromRaw(append([]byte(nil), raw...), shape[0], shape[1])
		dequant := make([]float32, shape[0]*shape[1])
		rowBytes := (shape[1] / qkK) * q4kBlockBytes
		for row := 0; row < shape[0]; row++ {
			dequantQ4KRef(dequant[row*shape[1]:(row+1)*shape[1]], raw[row*rowBytes:(row+1)*rowBytes])
		}
		appendQwen38MTPF32Tensor(ref, name, shape, dequant)
	}

	head := make([]float32, cfg.VocabSize*cfg.HiddenSize)
	for token := 0; token < cfg.VocabSize; token++ {
		for j := 0; j < cfg.HiddenSize; j++ {
			head[token*cfg.HiddenSize+j] = float32(((token+1)*(j%13+1))%17-8) / 32
		}
	}
	appendQwen38MTPF32Tensor(q4, "lm_head.weight", []int{cfg.VocabSize, cfg.HiddenSize}, head)
	appendQwen38MTPF32Tensor(ref, "lm_head.weight", []int{cfg.VocabSize, cfg.HiddenSize}, head)
	return q4, ref
}

func appendQwen38MTPF32Tensor(m *Model, name string, shape []int, data []float32) {
	offset := len(m.raw)
	for _, value := range data {
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(value))
		m.raw = append(m.raw, buf[:]...)
	}
	m.manifest[name] = tensorMeta{
		Dtype:  "F32",
		Shape:  append([]int(nil), shape...),
		Offset: offset,
		Nbytes: 4 * len(data),
	}
}

func qwen38MTPInputs(hidden, pos int) ([]float32, []float32) {
	prior := make([]float32, hidden)
	embedding := make([]float32, hidden)
	for i := 0; i < hidden; i++ {
		prior[i] = float32((i+3*pos)%19-9) / 16
		embedding[i] = float32((2*i+5*pos)%23-11) / 16
	}
	return prior, embedding
}

func TestQwen38MTPQ4KFixtureActuallyDiffersFromF32Storage(t *testing.T) {
	q4, ref := qwen38MTPQ4KTestModels(t)
	q4Layout, err := q4.Qwen38MTPTensorLayout()
	if err != nil {
		t.Fatal(err)
	}
	refLayout, err := ref.Qwen38MTPTensorLayout()
	if err != nil {
		t.Fatal(err)
	}
	if q4Layout.Format != Qwen38MTPFormatQ4K || refLayout.Format != Qwen38MTPFormatF32 {
		t.Fatalf("layouts q4=%+v ref=%+v", q4Layout, refLayout)
	}
	if reflect.DeepEqual(q4Layout.TensorTypes, refLayout.TensorTypes) {
		t.Fatal("Q4_K and F32 fixtures reported the same actual retained tensor types")
	}
}
