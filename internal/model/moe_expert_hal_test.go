package model

import (
	"encoding/binary"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type expertHALRecordingBackend struct {
	compute.Backend
	matmuls  int
	swiglu   int
	uploads  map[compute.Dtype]int
	capacity int64
}

func (b *expertHALRecordingBackend) Name() string                     { return "cuda-test" }
func (b *expertHALRecordingBackend) SupportsRoutedExpertKQuant() bool { return true }
func (b *expertHALRecordingBackend) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: true, UploadDtype: true, CapacityProbe: b.capacity > 0}
}
func (b *expertHALRecordingBackend) DeviceMemory() (total, free int64, known bool) {
	if b.capacity <= 0 {
		return 0, compute.FreeUnknown, false
	}
	return b.capacity, b.capacity, true
}
func (b *expertHALRecordingBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	b.uploads[as]++
	return b.Backend.Upload(t, as)
}
func (b *expertHALRecordingBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	return b.Backend.MatMul(w, x)
}
func (b *expertHALRecordingBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.swiglu++
	return b.Backend.SwiGLU(g, u)
}

func expertHALTestConfig(hidden int) Config {
	return Config{
		HiddenSize:          hidden,
		NumLayers:           1,
		NumHeads:            1,
		NumKVHeads:          1,
		HeadDim:             2,
		IntermediateSize:    hidden,
		MoEIntermediateSize: hidden,
		VocabSize:           4,
		RMSNormEps:          1e-5,
		RopeTheta:           10000,
		NumExperts:          1,
		NumExpertsPerTok:    1,
		NormTopKProb:        true,
		EOSTokenID:          -1,
	}
}

func TestExpertSwiGLUUsesResidentHALQ4K(t *testing.T) {
	const H = 256
	cfg := expertHALTestConfig(H)
	m := NewSyntheticMoE(cfg)
	gn := expertName(0, 0, "gate_proj.weight")
	un := expertName(0, 0, "up_proj.weight")
	dn := expertName(0, 0, "down_proj.weight")
	m.q4kw = map[string]*q4kTensor{}
	for i, name := range []string{gn, un, dn} {
		raw := buildRawQ4K(t, H, H, 31+i)
		m.q4kw[name] = &q4kTensor{out: H, in: H, raw: raw, nblk: 1}
	}
	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%17)-8) / 17
	}

	got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	if be.matmuls != 3 || be.swiglu != 1 {
		t.Fatalf("device expert ops matmul=%d swiglu=%d, want 3/1", be.matmuls, be.swiglu)
	}
	if be.uploads[compute.Q4_K] != 3 {
		t.Fatalf("Q4_K uploads=%d, want one resident upload per projection", be.uploads[compute.Q4_K])
	}
	if len(got) != H {
		t.Fatalf("output len=%d want %d", len(got), H)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d]=%v", i, v)
		}
	}

	// A second token reuses all three resident weights; only the activation upload repeats.
	expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	if be.uploads[compute.Q4_K] != 3 {
		t.Fatalf("resident weights reuploaded: %d", be.uploads[compute.Q4_K])
	}
}

func TestExpertSwiGLUStagesQ6KOnceRaw(t *testing.T) {
	const H = 256
	cfg := expertHALTestConfig(H)
	m := NewSyntheticMoE(cfg)
	gn := expertName(0, 0, "gate_proj.weight")
	un := expertName(0, 0, "up_proj.weight")
	dn := expertName(0, 0, "down_proj.weight")
	m.q4kw = map[string]*q4kTensor{}
	for i, name := range []string{gn, un} {
		m.q4kw[name] = &q4kTensor{out: H, in: H, raw: buildRawQ4K(t, H, H, 71+i), nblk: 1}
	}
	// Zero Q6_K blocks are valid and dequantize to a zero down projection.
	m.kqw = map[string]*kQuantTensor{dn: {out: H, in: H, nblk: 1, kind: kindQ6K, raw: make([]byte, H*kindQ6K.blockBytes())}}
	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
	x := make([]float32, H)

	expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	if be.uploads[compute.Q6_K] != 1 {
		t.Fatalf("Q6_K raw staging uploads=%d want 1", be.uploads[compute.Q6_K])
	}
	found := false
	for key := range s.halW {
		if strings.HasPrefix(key, "kquant-raw:") {
			found = true
		}
	}
	if !found {
		t.Fatal("missing explicit raw k-quant resident representation")
	}
}

func expertHALQ5KTensor(out, in int, seed uint64) *kQuantTensor {
	nblk := in / qkK
	bb := kindQ5K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, seed)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*bb:]
			binary.LittleEndian.PutUint16(blk[0:], 0x1000) // d = 2^-11, keeps three-projection fixture finite
			binary.LittleEndian.PutUint16(blk[2:], 0)
		}
	}
	return quantizeKQuantFromRaw(raw, out, in, kindQ5K)
}

func expertHALQ6KTensor(out, in int, seed int64) *kQuantTensor {
	nblk := in / qkK
	raw := make([]byte, out*nblk*q6kBlockBytes)
	rng := rand.New(rand.NewSource(seed))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := (o*nblk+b)*q6kBlockBytes + q6kBlockBytes - 1
			raw[base] = 0x2C | (raw[base] & 0x03)
		}
	}
	return &kQuantTensor{out: out, in: in, nblk: nblk, kind: kindQ6K, raw: raw}
}

func TestExpertSwiGLUHALParity(t *testing.T) {
	// Compare the backend against the exact f32 Q4_K oracle. On arm64 the
	// production resident path defaults to activation-quantized int8 SDOT,
	// which has its own quality contract and is not a 1e-5 f32 reference.
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	const H = 256
	for _, tc := range []struct {
		name string
		put  func(*Model, string, int)
	}{
		{name: "Q4_K", put: func(m *Model, name string, seed int) {
			m.q4kw[name] = &q4kTensor{out: H, in: H, raw: buildRawQ4K(t, H, H, seed), nblk: 1}
		}},
		{name: "Q5_K", put: func(m *Model, name string, seed int) {
			m.kqw[name] = expertHALQ5KTensor(H, H, uint64(seed))
		}},
		{name: "Q6_K", put: func(m *Model, name string, seed int) {
			m.kqw[name] = expertHALQ6KTensor(H, H, int64(seed))
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := NewSyntheticMoE(expertHALTestConfig(H))
			m.q4kw = map[string]*q4kTensor{}
			m.kqw = map[string]*kQuantTensor{}
			for i, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
				tc.put(m, expertName(0, 0, suffix), 101+i)
			}
			x := make([]float32, H)
			for i := range x {
				x[i] = float32((i%19)-9) / 64
			}
			refSession := &Session{M: m, Backend: compute.Default(), Q4K: true, halW: map[string]compute.Tensor{}}
			want := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: refSession})

			be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
			devSession := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
			got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: devSession})
			if be.matmuls != 3 || be.swiglu != 1 {
				t.Fatalf("device route matmul=%d swiglu=%d, want 3/1", be.matmuls, be.swiglu)
			}
			var maxAbs, scale float64
			for i := range want {
				d := math.Abs(float64(got[i] - want[i]))
				if d > maxAbs {
					maxAbs = d
				}
				if a := math.Abs(float64(want[i])); a > scale {
					scale = a
				}
			}
			// The recording backend executes the same f32 math as the scalar reference.
			// CUDA retains Q5_K/Q6_K bytes raw and has its own native-kernel parity gate.
			tol := 1e-5 * math.Max(1, scale)
			if maxAbs > tol {
				t.Fatalf("%s expert parity maxAbs=%g > tol=%g", tc.name, maxAbs, tol)
			}
			t.Logf("%s routed expert parity maxAbs=%g tol=%g", tc.name, maxAbs, tol)
		})
	}
}

func TestExpertSwiGLUHALKeepsKQuantAtRawSize(t *testing.T) {
	const H = 256
	m := NewSyntheticMoE(expertHALTestConfig(H))
	m.q4kw = map[string]*q4kTensor{}
	m.kqw = map[string]*kQuantTensor{}
	for i, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
		m.kqw[expertName(0, 0, suffix)] = expertHALQ6KTensor(H, H, int64(301+i))
	}
	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}, capacity: 1}
	s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
	got := expertSwiGLU(m, 0, 0, make([]float32, H), sessionQ4KKernel{s: s})
	if len(got) != H || be.matmuls != 3 || be.uploads[compute.Q6_K] != 3 {
		t.Fatalf("raw path output=%d matmuls=%d q6_uploads=%d", len(got), be.matmuls, be.uploads[compute.Q6_K])
	}
	if be.uploads[compute.F16] != 0 {
		t.Fatalf("expanded F16 uploads=%d want 0", be.uploads[compute.F16])
	}
}

func TestExpertParallelRankLocalKQuantPrefersHALAndStagesOnlyOwned(t *testing.T) {
	const H = 256
	cfg := expertHALTestConfig(H)
	cfg.NumExperts = 2
	m := NewSyntheticMoE(cfg)

	// Model a rank-0 sharded load: only expert 0 is resident. Gate/up are raw Q4_K
	// and down is Q6_K, the same combination accepted by hostBatchedGLMExperts.
	m.q4kw = map[string]*q4kTensor{}
	m.kqw = map[string]*kQuantTensor{}
	for i, proj := range []string{"gate_proj.weight", "up_proj.weight"} {
		name := expertName(0, 0, proj)
		m.q4kw[name] = &q4kTensor{out: H, in: H, raw: buildRawQ4K(t, H, H, 211+i), nblk: 1}
	}
	down := expertName(0, 0, "down_proj.weight")
	m.kqw[down] = &kQuantTensor{out: H, in: H, nblk: 1, kind: kindQ6K, raw: make([]byte, H*kindQ6K.blockBytes())}

	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	sess := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
	plan, err := ExpertParallelPlan(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%13)-6) / 13
	}
	picks := []routePick{{expert: 0, weight: 0.75}, {expert: 1, weight: 0.25}}

	got, err := m.expertParallelRankPartial(0, x, sessionQ4KKernel{s: sess}, picks, plan, 0)
	if err != nil {
		t.Fatalf("rank partial: %v", err)
	}
	if len(got) != H {
		t.Fatalf("output len=%d want %d", len(got), H)
	}
	if be.matmuls != 3 || be.swiglu != 1 {
		t.Fatalf("rank-local device ops matmul=%d swiglu=%d, want 3/1", be.matmuls, be.swiglu)
	}
	if be.uploads[compute.Q4_K] != 2 || be.uploads[compute.Q6_K] != 1 {
		t.Fatalf("rank-local resident uploads q4=%d q6=%d, want 2/1", be.uploads[compute.Q4_K], be.uploads[compute.Q6_K])
	}
	for key := range sess.halW {
		if strings.Contains(key, "experts.1.") {
			t.Fatalf("non-owned expert staged: %s", key)
		}
	}
}
