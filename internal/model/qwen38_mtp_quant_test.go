package model

import (
	"encoding/binary"
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

func TestQwen38MTPMixedQ4KMForwardMatchesDequantizedReference(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	q4, ref := qwen38MTPQ4KTestModels(t)
	q4Forward, err := q4.NewQwen35MTPForward()
	if err != nil {
		t.Fatalf("construct mixed Q4_K_M MTP forward: %v", err)
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
		t.Fatalf("execute mixed Q4_K_M MTP forward: %v", err)
	}
	want, err := refForward.Forward(0, prior, embedding)
	if err != nil {
		t.Fatalf("execute dequantized oracle MTP forward: %v", err)
	}
	if cos := cosine(got, want); cos < 0.99999 {
		t.Fatalf("mixed Q4_K_M MTP forward cosine=%.8f, want >= 0.99999", cos)
	}
	if argmaxF32(got) != argmaxF32(want) {
		t.Fatalf("mixed Q4_K_M MTP argmax=%d, oracle=%d", argmaxF32(got), argmaxF32(want))
	}
	for name, format := range qwen38MTPQ4KMArtifactMatrixTypes {
		if _, ok := q4.manifest[name]; ok {
			t.Fatalf("%s retained a persistent F32 matrix beside %s execution", name, format)
		}
		switch format {
		case Qwen38MTPFormatQ8:
			if q4.q8w[name] == nil {
				t.Fatalf("%s missing from resident Q8_0 store", name)
			}
		case Qwen38MTPFormatQ4K:
			if q4.q4kw[name] == nil {
				t.Fatalf("%s missing from resident Q4_K store", name)
			}
		case Qwen38MTPFormatQ6K:
			if q4.kqw[name] == nil {
				t.Fatalf("%s missing from resident Q6_K store", name)
			}
		}
	}
	if q4.kqw["lm_head.weight"] == nil {
		t.Fatal("canonical output.weight missing from resident Q6_K head store")
	}
}

func TestQwen38MTPMixedQ4KMSpeculativeAcceptanceAndMechanismReceipt(t *testing.T) {
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

// qwen38MTPQ4KTestModels retains its shared helper name for eligibility tests,
// but its quantized side deliberately mirrors the exact mixed Q4_K_M artifact.
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
	q4 = &Model{
		Cfg: cfg, manifest: map[string]tensorMeta{},
		q4kw: map[string]*q4kTensor{}, q8w: map[string]*q8Tensor{}, kqw: map[string]*kQuantTensor{},
	}
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
		appendQwen38MTPMixedQuantTensor(t, q4, ref, name, shapes[name], qwen38MTPQ4KMArtifactMatrixTypes[name], rng)
	}
	appendQwen38MTPMixedQuantTensor(t, q4, ref, "lm_head.weight", []int{cfg.VocabSize, cfg.HiddenSize}, Qwen38MTPFormatQ6K, rng)
	return q4, ref
}

func appendQwen38MTPMixedQuantTensor(t *testing.T, mixed, ref *Model, name string, shape []int, format Qwen38MTPTensorFormat, rng *rand.Rand) {
	t.Helper()
	out, in := shape[0], shape[1]
	dequant := make([]float32, out*in)
	switch format {
	case Qwen38MTPFormatQ8:
		values := make([]float32, out*in)
		for i := range values {
			values[i] = float32(rng.Intn(33)-16) / 32
		}
		qt := quantizeQ8(values, out, in)
		mixed.q8w[name] = qt
		for row := 0; row < out; row++ {
			for col := 0; col < in; col++ {
				block := col / qBlk
				dequant[row*in+col] = float32(qt.q[row*in+col]) * qt.d[row*qt.nblk+block]
			}
		}
	case Qwen38MTPFormatQ4K:
		raw := make([]byte, out*(in/qkK)*q4kBlockBytes)
		for block := 0; block < len(raw)/q4kBlockBytes; block++ {
			randQ4KBlockBounded(rng, raw[block*q4kBlockBytes:(block+1)*q4kBlockBytes], 2, 5)
		}
		mixed.q4kw[name] = quantizeQ4KFromRaw(append([]byte(nil), raw...), out, in)
		rowBytes := (in / qkK) * q4kBlockBytes
		for row := 0; row < out; row++ {
			dequantQ4KRef(dequant[row*in:(row+1)*in], raw[row*rowBytes:(row+1)*rowBytes])
		}
	case Qwen38MTPFormatQ6K:
		nblk := in / kindQ6K.blockWeights()
		raw := make([]byte, out*nblk*kindQ6K.blockBytes())
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
		pinResidentQuantScales(raw, out, nblk, kindQ6K)
		qt := quantizeKQuantFromRaw(raw, out, in, kindQ6K)
		mixed.kqw[name] = qt
		buf := make([]float32, kindQ6K.blockWeights())
		for row := 0; row < out; row++ {
			for block := 0; block < nblk; block++ {
				base := (row*nblk + block) * kindQ6K.blockBytes()
				kQuantDequantSuperBlock(buf, raw[base:base+kindQ6K.blockBytes()], kindQ6K)
				copy(dequant[row*in+block*len(buf):], buf)
			}
		}
	default:
		t.Fatalf("unsupported mixed fixture format %q for %s", format, name)
	}
	appendQwen38MTPF32Tensor(ref, name, shape, dequant)
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

func TestQwen38MTPQ4KMFixtureReportsExactMixedInventory(t *testing.T) {
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
	for name, want := range qwen38MTPQ4KMArtifactMatrixTypes {
		if got := q4Layout.TensorTypes[name]; got != string(want) {
			t.Fatalf("mixed inventory %s=%q, want %q; layout=%+v", name, got, want, q4Layout)
		}
	}
	if got := q4Layout.TensorTypes["lm_head.weight"]; got != string(Qwen38MTPFormatQ6K) {
		t.Fatalf("mixed inventory canonical output.weight=%q, want Q6_K; layout=%+v", got, q4Layout)
	}
	for _, name := range qwen38MTPNormTensors {
		if got := q4Layout.TensorTypes[name]; got != string(Qwen38MTPFormatF32) {
			t.Fatalf("mixed inventory norm %s=%q, want F32", name, got)
		}
	}
}

func TestQwen35MTPForwardConcurrentCloseReleasesWeightLifetimeOnce(t *testing.T) {
	m, _ := qwen38MTPQ4KTestModels(t)
	closer := &countWeightCloser{}
	m.SetWeightCloser(closer)
	forward, err := m.NewQwen35MTPForward()
	if err != nil {
		t.Fatal(err)
	}
	if err := m.CloseWeights(); err == nil {
		t.Fatal("CloseWeights admitted while MTP forward held the checkpoint")
	}

	const callers = 32
	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			forward.Close()
		}()
	}
	wg.Wait()
	if got := closer.n.Load(); got != 1 {
		t.Fatalf("checkpoint closes=%d, want exactly one after %d concurrent Forward.Close calls", got, callers)
	}
	if err := m.CloseWeights(); err != nil {
		t.Fatalf("completed CloseWeights: %v", err)
	}
}
