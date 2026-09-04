package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// denseKQuantRecordingBackend wraps a compute.Backend, advertises Caps().UploadDtype = true,
// and records upload dtypes requested through Upload.
type denseKQuantRecordingBackend struct {
	compute.Backend
	uploads map[compute.Dtype]int
}

func newDenseKQuantRecordingBackend(be compute.Backend) *denseKQuantRecordingBackend {
	if be == nil {
		be = compute.Default()
	}
	return &denseKQuantRecordingBackend{
		Backend: be,
		uploads: make(map[compute.Dtype]int),
	}
}

func (b *denseKQuantRecordingBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = true
	return c
}

func (b *denseKQuantRecordingBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	b.uploads[as]++
	return b.Backend.Upload(t, as)
}

func TestUseHALKQuantWeights(t *testing.T) {
	// 1. Reference backend (UploadDtype = false): must report false.
	if compute.Default().Caps().UploadDtype {
		t.Skip("reference backend unexpectedly reports UploadDtype; skipping non-UploadDtype check")
	}
	m := &Model{kqw: map[string]*kQuantTensor{}}
	refSession := &Session{M: m, Backend: compute.Default()}
	if refSession.useHALKQuantWeights() {
		t.Fatalf("useHALKQuantWeights true on reference backend without UploadDtype")
	}

	// 2. UploadDtype backend, but m is nil: must report false.
	recBe := newDenseKQuantRecordingBackend(compute.Default())
	nilModelSession := &Session{M: nil, Backend: recBe}
	if nilModelSession.useHALKQuantWeights() {
		t.Fatalf("useHALKQuantWeights true with nil Model")
	}

	// 3. UploadDtype backend, but m.kqw is nil: must report false.
	nilKqwSession := &Session{M: &Model{}, Backend: recBe}
	if nilKqwSession.useHALKQuantWeights() {
		t.Fatalf("useHALKQuantWeights true with nil kqw")
	}

	// 4. UploadDtype backend and m.kqw populated: must report true.
	activeSession := &Session{M: m, Backend: recBe}
	if !activeSession.useHALKQuantWeights() {
		t.Fatalf("useHALKQuantWeights false on UploadDtype backend with initialized kqw")
	}
}

func TestDenseKQuantHALStaging(t *testing.T) {
	const out, in = 8, 256
	downName := "model.layers.0.mlp.down_proj.weight"
	oName := "model.layers.0.self_attn.o_proj.weight"

	q5Tensor := expertHALQ5KTensor(out, in, 1001)
	q6Tensor := expertHALQ6KTensor(out, in, 1002)

	m := &Model{
		kqw: map[string]*kQuantTensor{
			downName: q5Tensor,
			oName:    q6Tensor,
		},
	}

	be := newDenseKQuantRecordingBackend(compute.Default())
	s := &Session{
		M:       m,
		Backend: be,
		halW:    make(map[string]compute.Tensor),
	}

	// 1. Stage Q5_K dense weight through matWeightHAL.
	wQ5 := s.matWeightHAL(downName)
	if wQ5.Dtype != compute.Q5_K {
		t.Fatalf("matWeightHAL(%q) dtype = %v, want Q5_K", downName, wQ5.Dtype)
	}
	if be.uploads[compute.Q5_K] != 1 {
		t.Fatalf("uploads[Q5_K] = %d, want 1", be.uploads[compute.Q5_K])
	}
	if _, ok := s.halW["kquant-raw:"+downName]; !ok {
		t.Errorf("halW missing key kquant-raw:%s", downName)
	}

	// Second (warm) call must not re-upload.
	_ = s.matWeightHAL(downName)
	if be.uploads[compute.Q5_K] != 1 {
		t.Fatalf("warm matWeightHAL re-uploaded: uploads[Q5_K] = %d, want 1", be.uploads[compute.Q5_K])
	}

	// 2. Stage Q6_K dense weight through matWeightHAL.
	wQ6 := s.matWeightHAL(oName)
	if wQ6.Dtype != compute.Q6_K {
		t.Fatalf("matWeightHAL(%q) dtype = %v, want Q6_K", oName, wQ6.Dtype)
	}
	if be.uploads[compute.Q6_K] != 1 {
		t.Fatalf("uploads[Q6_K] = %d, want 1", be.uploads[compute.Q6_K])
	}
	if _, ok := s.halW["kquant-raw:"+oName]; !ok {
		t.Errorf("halW missing key kquant-raw:%s", oName)
	}

	// Second (warm) call must not re-upload.
	_ = s.matWeightHAL(oName)
	if be.uploads[compute.Q6_K] != 1 {
		t.Fatalf("warm matWeightHAL re-uploaded: uploads[Q6_K] = %d, want 1", be.uploads[compute.Q6_K])
	}
}

func TestDenseKQuantLMHeadHAL(t *testing.T) {
	const out, in = 8, 256

	t.Run("untied_q6k_head", func(t *testing.T) {
		headName := "lm_head.weight"
		q6Head := expertHALQ6KTensor(out, in, 2001)
		m := &Model{
			kqw: map[string]*kQuantTensor{
				headName: q6Head,
			},
		}
		be := newDenseKQuantRecordingBackend(compute.Default())
		s := &Session{M: m, Backend: be, halW: make(map[string]compute.Tensor)}

		w := s.lmHeadMatHAL()
		if w.Dtype != compute.Q6_K {
			t.Fatalf("lmHeadMatHAL dtype = %v, want Q6_K", w.Dtype)
		}
		if be.uploads[compute.Q6_K] != 1 {
			t.Fatalf("uploads[Q6_K] = %d, want 1", be.uploads[compute.Q6_K])
		}
		if _, ok := s.halW["kquant-raw:"+headName]; !ok {
			t.Errorf("halW missing key kquant-raw:%s", headName)
		}
	})

	t.Run("untied_q5k_head", func(t *testing.T) {
		headName := "lm_head.weight"
		q5Head := expertHALQ5KTensor(out, in, 2002)
		m := &Model{
			kqw: map[string]*kQuantTensor{
				headName: q5Head,
			},
		}
		be := newDenseKQuantRecordingBackend(compute.Default())
		s := &Session{M: m, Backend: be, halW: make(map[string]compute.Tensor)}

		w := s.lmHeadMatHAL()
		if w.Dtype != compute.Q5_K {
			t.Fatalf("lmHeadMatHAL dtype = %v, want Q5_K", w.Dtype)
		}
		if be.uploads[compute.Q5_K] != 1 {
			t.Fatalf("uploads[Q5_K] = %d, want 1", be.uploads[compute.Q5_K])
		}
		if _, ok := s.halW["kquant-raw:"+headName]; !ok {
			t.Errorf("halW missing key kquant-raw:%s", headName)
		}
	})

	t.Run("tied_embed_tokens_head", func(t *testing.T) {
		tiedName := "model.embed_tokens.weight"
		q6Tied := expertHALQ6KTensor(out, in, 2003)
		m := &Model{
			kqw: map[string]*kQuantTensor{
				tiedName: q6Tied,
			},
		}
		be := newDenseKQuantRecordingBackend(compute.Default())
		s := &Session{M: m, Backend: be, halW: make(map[string]compute.Tensor)}

		w := s.lmHeadMatHAL()
		if w.Dtype != compute.Q6_K {
			t.Fatalf("lmHeadMatHAL dtype = %v, want Q6_K", w.Dtype)
		}
		if be.uploads[compute.Q6_K] != 1 {
			t.Fatalf("uploads[Q6_K] = %d, want 1", be.uploads[compute.Q6_K])
		}
		if _, ok := s.halW["kquant-raw:"+tiedName]; !ok {
			t.Errorf("halW missing key kquant-raw:%s", tiedName)
		}
	})
}

func TestDenseKQuantHALParityWithCPUReference(t *testing.T) {
	const out, in = 16, 256

	for _, tc := range []struct {
		name string
		qt   *kQuantTensor
	}{
		{name: "Q5_K", qt: expertHALQ5KTensor(out, in, 3001)},
		{name: "Q6_K", qt: expertHALQ6KTensor(out, in, 3002)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			weightName := "model.layers.0.mlp.down_proj.weight"
			m := &Model{
				kqw: map[string]*kQuantTensor{
					weightName: tc.qt,
				},
			}
			be := newDenseKQuantRecordingBackend(compute.Default())
			s := &Session{M: m, Backend: be, halW: make(map[string]compute.Tensor)}

			w := s.matWeightHAL(weightName)

			x := make([]float32, in)
			for i := range x {
				x[i] = float32((i%13)-6) * 0.0625
			}

			want := kQuantMatRows(tc.qt, x)

			dx := compute.NewF32(compute.Default(), []int{in}, append([]float32(nil), x...))
			outTensor := be.MatMul(w, dx)
			got := be.Read(outTensor)

			if len(got) != len(want) {
				t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
			}
			for i := range want {
				diff := math.Abs(float64(got[i] - want[i]))
				tol := 1e-4 * math.Max(1, math.Abs(float64(want[i])))
				if diff > tol {
					t.Fatalf("[%d] drift: got %v, want %v (|diff|=%g > tol=%g)", i, got[i], want[i], diff, tol)
				}
			}
		})
	}
}

func TestDenseKQuantLMHeadHALParityWithCPUReference(t *testing.T) {
	const out, in = 16, 256

	for _, tc := range []struct {
		name string
		qt   *kQuantTensor
	}{
		{name: "Q5_K_head", qt: expertHALQ5KTensor(out, in, 3101)},
		{name: "Q6_K_head", qt: expertHALQ6KTensor(out, in, 3102)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			headName := "lm_head.weight"
			m := &Model{
				kqw: map[string]*kQuantTensor{
					headName: tc.qt,
				},
			}
			be := newDenseKQuantRecordingBackend(compute.Default())
			s := &Session{M: m, Backend: be, halW: make(map[string]compute.Tensor)}

			w := s.lmHeadMatHAL()

			x := make([]float32, in)
			for i := range x {
				x[i] = float32((i%11)-5) * 0.0625
			}

			want := kQuantMatRows(tc.qt, x)

			dx := compute.NewF32(compute.Default(), []int{in}, append([]float32(nil), x...))
			outTensor := be.MatMul(w, dx)
			got := be.Read(outTensor)

			if len(got) != len(want) {
				t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
			}
			for i := range want {
				diff := math.Abs(float64(got[i] - want[i]))
				tol := 1e-4 * math.Max(1, math.Abs(float64(want[i])))
				if diff > tol {
					t.Fatalf("[%d] drift: got %v, want %v (|diff|=%g > tol=%g)", i, got[i], want[i], diff, tol)
				}
			}
		})
	}
}

func TestDenseKQuantHALMixture(t *testing.T) {
	// Verifies a q4_k_m-like mixture where gate/up are Q4_K in q4kw, and down is Q6_K in kqw.
	const out, in = 8, 256
	gateName := "model.layers.0.mlp.gate_proj.weight"
	downName := "model.layers.0.mlp.down_proj.weight"
	headName := "lm_head.weight"

	q4GateRaw := buildRawQ4K(t, out, in, 4001)
	q6Down := expertHALQ6KTensor(out, in, 4002)
	q6Head := expertHALQ6KTensor(out, in, 4003)

	m := &Model{
		q4kw: map[string]*q4kTensor{
			gateName: {out: out, in: in, nblk: in / qkK, raw: q4GateRaw},
		},
		kqw: map[string]*kQuantTensor{
			downName: q6Down,
			headName: q6Head,
		},
	}

	be := newDenseKQuantRecordingBackend(compute.Default())
	s := &Session{
		M:       m,
		Backend: be,
		Q4K:     true,
		halW:    make(map[string]compute.Tensor),
	}

	// gate routes to Q4_K
	wGate := s.matWeightHAL(gateName)
	if wGate.Dtype != compute.Q4_K {
		t.Fatalf("gate_proj dtype = %v, want Q4_K", wGate.Dtype)
	}

	// down routes to Q6_K from kqw
	wDown := s.matWeightHAL(downName)
	if wDown.Dtype != compute.Q6_K {
		t.Fatalf("down_proj dtype = %v, want Q6_K", wDown.Dtype)
	}

	// lm_head routes to Q6_K from kqw
	wHead := s.lmHeadMatHAL()
	if wHead.Dtype != compute.Q6_K {
		t.Fatalf("lmHeadMatHAL dtype = %v, want Q6_K", wHead.Dtype)
	}

	if be.uploads[compute.Q4_K] != 1 {
		t.Errorf("uploads[Q4_K] = %d, want 1", be.uploads[compute.Q4_K])
	}
	if be.uploads[compute.Q6_K] != 2 {
		t.Errorf("uploads[Q6_K] = %d, want 2 (down + head)", be.uploads[compute.Q6_K])
	}
}

func TestDenseKQuantCUDAParityIfAvailable(t *testing.T) {
	cudaBe, ok := compute.Lookup("cuda")
	if !ok {
		t.Skip("cuda backend not available on this host")
	}

	const out, in = 16, 256
	q6Tensor := expertHALQ6KTensor(out, in, 5001)
	weightName := "model.layers.0.mlp.down_proj.weight"

	m := &Model{
		kqw: map[string]*kQuantTensor{
			weightName: q6Tensor,
		},
	}

	s := &Session{
		M:       m,
		Backend: cudaBe,
		halW:    make(map[string]compute.Tensor),
	}

	w := s.matWeightHAL(weightName)
	if w.Dtype != compute.Q6_K {
		t.Fatalf("CUDA matWeightHAL dtype = %v, want Q6_K", w.Dtype)
	}

	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i%11)-5) * 0.125
	}

	want := kQuantMatRows(q6Tensor, x)

	hostX := compute.NewF32(compute.Default(), []int{in}, append([]float32(nil), x...))
	dx := cudaBe.Upload(hostX, compute.F32)
	defer cudaBe.Free(dx)

	outDev := cudaBe.MatMul(w, dx)
	defer cudaBe.Free(outDev)

	got := cudaBe.Read(outDev)
	if len(got) != len(want) {
		t.Fatalf("CUDA output length = %d, want %d", len(got), len(want))
	}

	var maxAbs float64
	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > maxAbs {
			maxAbs = diff
		}
	}
	if maxAbs > 1e-4 {
		t.Fatalf("CUDA Q6_K GEMV drift maxAbs=%g > 1e-4", maxAbs)
	}
}
