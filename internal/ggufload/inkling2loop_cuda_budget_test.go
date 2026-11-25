package ggufload

import (
	"errors"
	"math/rand"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2BudgetBackend is a CPU backend that advertises the Inkling2Device op
// set and a fixed free-memory reading, and counts weight uploads.
type inkling2BudgetBackend struct {
	compute.Backend
	compute.Inkling2Device
	free    int64
	uploads int
}

func (b *inkling2BudgetBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	b.uploads++
	return b.Backend.Upload(t, as)
}

func (b *inkling2BudgetBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.CapacityProbe = true
	return c
}

func (b *inkling2BudgetBackend) DeviceMemory() (total, free int64, known bool) {
	return b.free * 2, b.free, true
}

// inkling2BudgetModel is a 4-layer loop model with core [1,3) x3, so expert
// residency order is 1, 2, 0, 3.
func inkling2BudgetModel(t *testing.T) (m *Inkling2LoopModel, dense int64, layerBytes int64) {
	t.Helper()
	r := rand.New(rand.NewSource(11))
	quant := func(rows, cols int) *inkling2QuantMatrix {
		rb, _ := inkling2RowBytes(TensorQ8_0, cols)
		return &inkling2QuantMatrix{name: "q8", typ: TensorQ8_0, raw: randomInkling2Raw(t, r, TensorQ8_0, rows, cols), rows: rows, cols: cols, rowBytes: rb}
	}
	const experts, rows, cols = 2, 4, 128
	rb, _ := inkling2RowBytes(TensorNVFP4, cols)
	exps := func() *inkling2QuantExperts {
		return &inkling2QuantExperts{typ: TensorNVFP4, raw: randomInkling2NVFP4(r, experts*rows, cols), experts: experts, rows: rows, cols: cols, rowLen: rb}
	}
	w := &model.Inkling2LoopWeights{Output: quant(5, 64)}
	for i := 0; i < 4; i++ {
		w.Layers = append(w.Layers, model.Inkling2LoopLayerWeights{Wq: quant(3, 64), GateUp: exps(), Down: exps()})
	}
	m = &Inkling2LoopModel{HP: model.Inkling2LoopHParams{Layers: 4, CoreStart: 1, CoreEnd: 3, Passes: 3}, Weights: w}
	dense = inkling2MatrixDeviceBytes(w.Output)
	for i := range w.Layers {
		dense += inkling2MatrixDeviceBytes(w.Layers[i].Wq)
	}
	layerBytes = inkling2ExpertsDeviceBytes(w.Layers[0].GateUp) + inkling2ExpertsDeviceBytes(w.Layers[0].Down)
	return m, dense, layerBytes
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2BindDeviceRespectsBudget(t *testing.T) {
	m, dense, layer := inkling2BudgetModel(t)
	budget := dense + 2*layer + layer/2
	t.Setenv(inkling2VRAMBudgetEnv, strconv.FormatInt(budget, 10))
	be := &inkling2BudgetBackend{Backend: compute.Default()}
	if err := bindInkling2Device(m, be); err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	p, ok := Inkling2DevicePlacementOf(m)
	if !ok {
		t.Fatal("no placement recorded")
	}
	if p.BudgetBytes != budget || p.DenseBytes != dense || p.ResidentLayers != 2 || p.ColdLayers != 2 ||
		p.ResidentBytes != 2*layer || p.ColdBytes != 2*layer || p.DenseBytes+p.ResidentBytes > budget {
		t.Fatalf("placement %+v (dense %d, layer %d, budget %d)", p, dense, layer, budget)
	}
	// 5 dense Q8_0 matrices + 2 resident layers x 2 stacks.
	if be.uploads != 5+4 {
		t.Fatalf("uploads = %d, want 9", be.uploads)
	}

	m2, _, _ := inkling2BudgetModel(t)
	t.Setenv(inkling2VRAMBudgetEnv, strconv.FormatInt(dense-1, 10))
	err := bindInkling2Device(m2, &inkling2BudgetBackend{Backend: compute.Default()})
	var over *Inkling2DenseOverBudgetError
	if !errors.As(err, &over) || over.DenseBytes != dense {
		t.Fatalf("dense over budget: err = %v", err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2BindDeviceColdExpertsStayHost(t *testing.T) {
	m, dense, layer := inkling2BudgetModel(t)
	t.Setenv(inkling2VRAMBudgetEnv, strconv.FormatInt(dense+2*layer, 10))
	cold := []model.Inkling2LoopExperts{m.Weights.Layers[0].GateUp, m.Weights.Layers[3].Down}
	if err := bindInkling2Device(m, &inkling2BudgetBackend{Backend: compute.Default()}); err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, i := range []int{1, 2} {
		lw := m.Weights.Layers[i]
		if _, ok := lw.GateUp.(*inkling2DeviceExperts); !ok {
			t.Fatalf("core layer %d gate_up stayed %T", i, lw.GateUp)
		}
		if _, ok := lw.Down.(*inkling2DeviceExperts); !ok {
			t.Fatalf("core layer %d down stayed %T", i, lw.Down)
		}
	}
	if m.Weights.Layers[0].GateUp != cold[0] || m.Weights.Layers[3].Down != cold[1] {
		t.Fatalf("cold experts moved: %T %T", m.Weights.Layers[0].GateUp, m.Weights.Layers[3].Down)
	}
	// A batch mixing a resident and a cold expert matches host math for both.
	x := make([]float32, 128)
	for i := range x {
		x[i] = float32(i%7) - 3
	}
	hot := m.Weights.Layers[1].GateUp
	gotHot, gotCold := make([]float32, 4), make([]float32, 4)
	m.Weights.Batcher.MatVecBatch([]model.Inkling2LoopMatVecJob{
		{Experts: hot, Expert: 1, Dst: gotHot, X: x},
		{Experts: cold[0], Expert: 1, Dst: gotCold, X: x},
	})
	wantCold := make([]float32, 4)
	cold[0].ExpertMatVec(1, wantCold, x)
	for i := range wantCold {
		if gotCold[i] != wantCold[i] {
			t.Fatalf("cold row %d: batch %v, host %v", i, gotCold[i], wantCold[i])
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2BindDeviceResidentIneligibleWhenCold(t *testing.T) {
	t.Setenv("FAK_INKLING2_RESIDENT", "1")
	m, dense, layer := inkling2BudgetModel(t)
	be := &inkling2BudgetBackend{Backend: compute.Default()}
	t.Setenv(inkling2VRAMBudgetEnv, strconv.FormatInt(dense+3*layer, 10))
	if err := bindInkling2Device(m, be); err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, ok := m.Weights.Layers[0].GateUp.(*inkling2DeviceExperts); !ok {
		t.Fatalf("layer 0 should be resident, got %T", m.Weights.Layers[0].GateUp)
	}
	if inkling2ResidentEligible(m, be) {
		t.Fatal("resident fast path eligible with layer 3 experts on the host")
	}

	full, _, _ := inkling2BudgetModel(t)
	t.Setenv(inkling2VRAMBudgetEnv, "")
	be.free = 1 << 30
	if err := bindInkling2Device(full, be); err != nil {
		t.Fatal(err)
	}
	defer full.Close()
	if p, _ := Inkling2DevicePlacementOf(full); !p.FullyResident() || p.BudgetBytes != (1<<30)-(1<<30)*15/100 {
		t.Fatalf("free-memory budget placement %+v", p)
	}
	if !inkling2ResidentEligible(full, be) {
		t.Fatal("fully resident model not eligible")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2DefaultBudgetKeepsMiniFullyResident(t *testing.T) {
	// Inkling-2-Mini UD-Q3_K_XL as deployed on a 16 GiB RTX 5080: a 13.63 GB
	// file whose expert tensors total 12.20 GB over 24 layers (core [6,18) x3).
	// Non-expert bytes are charged in full although the token embedding stays
	// host, so the plan here is stricter than the real bind.
	const gibB = int64(1) << 30
	const fileBytes, expertBytes = int64(13_630_000_000), int64(12_200_000_000)
	hp := model.Inkling2LoopHParams{Layers: 24, CoreStart: 6, CoreEnd: 18, Passes: 3}
	layerBytes := make([]int64, 24)
	for i := range layerBytes {
		layerBytes[i] = expertBytes / 24
	}
	noEnv := func(string) (string, bool) { return "", false }
	// 15.3 GiB is the card free after the CUDA context; 14.0 GiB leaves room
	// for a peer process holding VRAM.
	for _, free := range []int64{15*gibB + 3*gibB/10, 14 * gibB} {
		be := &inkling2BudgetBackend{Backend: compute.Default(), free: free}
		budget, err := inkling2DeviceBudget(be, noEnv)
		if err != nil {
			t.Fatal(err)
		}
		if budget != free-inkling2VRAMHeadroomCap {
			t.Fatalf("free %d: budget = %d, want free minus the 1 GiB headroom cap", free, budget)
		}
		_, p, err := inkling2PlanExpertResidency(inkling2ExpertResidencyOrder(hp, 24), layerBytes, fileBytes-expertBytes, budget)
		if err != nil {
			t.Fatal(err)
		}
		if !p.FullyResident() || p.ResidentLayers != 24 || p.ColdBytes != 0 {
			t.Fatalf("Mini UD-Q3_K_XL with %.1f GiB free: %s", inkling2GiB(free), p)
		}
	}
	be := &inkling2BudgetBackend{Backend: compute.Default(), free: 15*gibB + 3*gibB/10}
	budget, _ := inkling2DeviceBudget(be, noEnv)

	// Small-sized expert bytes on the same card spill; with no loop core the
	// leading layers win.
	small := make([]int64, 48)
	for i := range small {
		small[i] = 1 << 30
	}
	res, p, err := inkling2PlanExpertResidency(inkling2ExpertResidencyOrder(model.Inkling2LoopHParams{Layers: 48, Passes: 1}, 48), small, 2*gibB, budget)
	if err != nil {
		t.Fatal(err)
	}
	if p.FullyResident() || !res[0] || res[47] || p.ResidentLayers+p.ColdLayers != 48 {
		t.Fatalf("Small under 16 GiB: %s", p)
	}

	// No free-memory probe and no override keeps the unbounded upload-everything behaviour.
	if b, _ := inkling2DeviceBudget(compute.Default(), func(string) (string, bool) { return "", false }); b != -1 {
		t.Fatalf("unprobed budget = %d, want -1", b)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2BindDeviceSizesMatchUploadForm(t *testing.T) {
	// Q8_0 uploads as int8 codes plus one f32 scale per 32 values.
	if got := inkling2DeviceBytes(TensorQ8_0, 4*34, 4, 32); got != 4*32+4*4 {
		t.Fatalf("Q8_0 device bytes = %d, want 144", got)
	}
	r := rand.New(rand.NewSource(3))
	const rows, cols = 4, 128
	raw := randomInkling2NVFP4(r, rows, cols)
	packed, scales, err := repackInkling2NVFP4(raw, rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := inkling2DeviceBytes(TensorNVFP4, len(raw), rows, cols), int64(len(packed)+len(scales)); got != want {
		t.Fatalf("NVFP4 device bytes = %d, repacked = %d", got, want)
	}
}
