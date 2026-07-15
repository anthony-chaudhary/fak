package model

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestV4LiveExpertLoadSkipsRoutedPayloadsAndConstructsLazily(t *testing.T) {
	dir := t.TempDir()
	cfg := pinnedV4RuntimeConfig()
	cfgBytes := []byte(`{"model_type":"deepseek_v4","num_hidden_layers":61,"hidden_size":7168,"n_routed_experts":384,"num_experts_per_tok":6,"moe_intermediate_size":3072,"n_shared_experts":1,"expert_dtype":"fp4","norm_topk_prob":true,"routed_scaling_factor":2.5,"scoring_func":"sqrtsoftplus","topk_method":"noaux_tc","swiglu_limit":10}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	name := "model.layers.3.mlp.experts.0.gate_proj.weight"
	norm := "lm_head.weight"
	shard := "model-00001-of-00001.safetensors"
	if err := os.WriteFile(filepath.Join(dir, shard), tinySafetensorsBytes(t, map[string]tinySTTensor{
		name: {dtype: "F32", shape: []int{1}, data: f32TestBytes([]float32{9})},
		norm: {dtype: "F32", shape: []int{1, 32}, data: f32TestBytes(make([]float32, 32))},
	}), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(`{"weight_map":{"`+name+`":"`+shard+`","`+norm+`":"`+shard+`"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir: %v", err)
	}
	if _, ok := m.manifest[name]; ok {
		t.Fatalf("routed payload became resident")
	}
	if m.sourceDir != dir {
		t.Fatalf("sourceDir=%q want %q", m.sourceDir, dir)
	}
	fixtureDir, _ := writeV4RuntimeFixture(t)
	m.sourceDir = fixtureDir
	be := compute.Default()
	t.Setenv("FAK_V4_EXPERT_RING_BYTES", "4096")
	s := m.NewSession()
	s.Backend = be
	v, err := s.ensureV4LiveExpert()
	if err != nil {
		t.Fatalf("ensureV4LiveExpert: %v", err)
	}
	if got := v.Stats(); got.RingBudget != 4096 || got.SourceReads != 0 || got.PageIns != 0 {
		t.Fatalf("constructor touched payload or wrong cap: %+v", got)
	}
	s.Close()
}

func TestV4LiveExpertFailsClosed(t *testing.T) {
	cfg := pinnedV4RuntimeConfig()
	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{{Name: "x", Shape: []int{1}, Data: []float32{1}}})
	if err != nil {
		t.Fatal(err)
	}
	s := m.NewSession()
	be := compute.Default()
	s.Backend = be
	if _, err := s.ensureV4LiveExpert(); !errors.Is(err, ErrV4LiveExpert) {
		t.Fatalf("missing indexed source err=%v", err)
	}
	bad := cfg
	bad.NumExpertsPerTok = 5
	if _, err := NewFromF32Tensors(bad, []NamedTensorF32{{Name: "x", Shape: []int{1}, Data: []float32{1}}}); !errors.Is(err, ErrV4ConfigAdmission) {
		t.Fatalf("bad config err=%v", err)
	}
	t.Setenv("FAK_V4_EXPERT_RING_BYTES", "0")
	if _, _, err := v4RuntimeLimits(); !errors.Is(err, ErrV4LiveExpert) {
		t.Fatalf("bad cap err=%v", err)
	}
}

type residentOracleV4LiveExpert struct {
	layer    int
	tokenID  int
	x        []float32
	logits   []float32
	bias     []float32
	weights  map[string][]float32
	out      []float32
	picks    []routePick
	forwards int
	closed   bool
}

func (r *residentOracleV4LiveExpert) forward(layer, tokenID int, x, logits, correctionBias []float32) ([]float32, error) {
	r.layer = layer
	r.tokenID = tokenID
	r.x = append([]float32(nil), x...)
	r.logits = append([]float32(nil), logits...)
	r.bias = append([]float32(nil), correctionBias...)
	r.forwards++
	picks, err := v4ScoredRoute(logits, correctionBias, 6, float32(2.5))
	if err != nil {
		return nil, err
	}
	r.picks = append([]routePick(nil), picks...)
	oracle32 := runtimeResidentOracle(layer, picks, x[:32], r.weights, 10)
	r.out = make([]float32, len(x))
	copy(r.out, oracle32)
	return append([]float32(nil), r.out...), nil
}

func (r *residentOracleV4LiveExpert) Close() error {
	r.closed = true
	return nil
}

func (r *residentOracleV4LiveExpert) Stats() v4ExpertRuntimeStats { return v4ExpertRuntimeStats{} }

func TestV4LiveExpertHALForwardCapturedFixture(t *testing.T) {
	cfg := pinnedV4RuntimeConfig()
	const layer = 3
	prefix := layerName(layer, "mlp.")

	norm := make([]float32, cfg.HiddenSize)
	residual := make([]float32, cfg.HiddenSize)
	for i := range norm {
		norm[i] = 1
		residual[i] = 1
	}
	gate := make([]float32, cfg.NumExperts*cfg.HiddenSize)
	for expert := 0; expert < cfg.NumExperts; expert++ {
		if expert < 12 {
			gate[expert*cfg.HiddenSize] = float32(12 - expert)
		} else {
			gate[expert*cfg.HiddenSize] = -float32(expert)
		}
	}
	bias := make([]float32, cfg.NumExperts)
	for i := range bias {
		bias[i] = float32(i) / 1024
	}

	dir := t.TempDir()
	cfgBytes := []byte(`{"model_type":"deepseek_v4","num_hidden_layers":61,"hidden_size":7168,"n_routed_experts":384,"num_experts_per_tok":6,"moe_intermediate_size":3072,"n_shared_experts":1,"expert_dtype":"fp4","norm_topk_prob":true,"routed_scaling_factor":2.5,"scoring_func":"sqrtsoftplus","topk_method":"noaux_tc","swiglu_limit":10}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	gateName := prefix + "gate.weight"
	biasName := prefix + "gate.e_score_correction_bias"
	lmHeadName := "lm_head.weight"
	shard := "model-00001-of-00001.safetensors"
	if err := os.WriteFile(filepath.Join(dir, shard), tinySafetensorsBytes(t, map[string]tinySTTensor{
		gateName:   {dtype: "F32", shape: []int{cfg.NumExperts, cfg.HiddenSize}, data: f32TestBytes(gate)},
		biasName:   {dtype: "F32", shape: []int{cfg.NumExperts}, data: f32TestBytes(bias)},
		lmHeadName: {dtype: "F32", shape: []int{1, 32}, data: f32TestBytes(make([]float32, 32))},
	}), 0o644); err != nil {
		t.Fatal(err)
	}
	index := `{"weight_map":{"` + gateName + `":"` + shard + `","` + biasName + `":"` + shard + `","` + lmHeadName + `":"` + shard + `"}}`
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(index), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir: %v", err)
	}
	if m.sourceDir != dir {
		t.Fatalf("sourceDir=%q want %q", m.sourceDir, dir)
	}
	be := compute.Default()
	gateMeta, ok := m.q8w[gateName]
	if !ok {
		t.Fatalf("production quantized load did not retain router gate %s", gateName)
	}
	gateResident := make([]float32, cfg.NumExperts*cfg.HiddenSize)
	for row := 0; row < gateMeta.out; row++ {
		for col := 0; col < gateMeta.in; col++ {
			gateResident[row*gateMeta.in+col] = float32(gateMeta.q[row*gateMeta.in+col]) * gateMeta.d[row*gateMeta.nblk+col/qBlk]
		}
	}
	off := len(m.raw)
	appendQuantF32Tensor(m, &m.raw, &off, gateName, []int{cfg.NumExperts, cfg.HiddenSize}, gateResident)
	s := m.NewSession()
	s.Backend = be
	_, oracleWeights := writeV4RuntimeFixture(t)
	fake := &residentOracleV4LiveExpert{weights: oracleWeights}
	s.v4Expert = fake

	residualTensor := s.uploadHostF32([]int{cfg.HiddenSize}, residual, compute.MemoryActivation, "v4-test-residual")
	normTensor := s.uploadHostF32([]int{cfg.HiddenSize}, norm, compute.MemoryActivation, "v4-test-norm")
	defer be.Free(residualTensor)
	defer be.Free(normTensor)

	const tokenID = 4242
	if err := s.applyV4ExpertHAL(layer, tokenID, residualTensor, normTensor, float32(cfg.RMSNormEps)); err != nil {
		t.Fatalf("applyV4ExpertHAL: %v", err)
	}
	if fake.forwards != 1 || fake.layer != layer || fake.tokenID != tokenID {
		t.Fatalf("captured call forwards=%d layer=%d token=%d", fake.forwards, fake.layer, fake.tokenID)
	}
	if len(fake.x) != cfg.HiddenSize || len(fake.logits) != cfg.NumExperts || len(fake.bias) != cfg.NumExperts {
		t.Fatalf("captured widths x=%d logits=%d bias=%d", len(fake.x), len(fake.logits), len(fake.bias))
	}
	wantNormalized := float32(1 / math.Sqrt(1+float64(cfg.RMSNormEps)))
	for _, i := range []int{0, 1, cfg.HiddenSize - 1} {
		if diff := float32(math.Abs(float64(fake.x[i] - wantNormalized))); diff > 1e-5 {
			t.Fatalf("normalized[%d]=%g want %g", i, fake.x[i], wantNormalized)
		}
	}
	for _, i := range []int{0, 17, cfg.NumExperts - 1} {
		gateValue := -float32(i)
		if i < 12 {
			gateValue = float32(12 - i)
		}
		want := gateValue * wantNormalized
		if diff := float32(math.Abs(float64(fake.logits[i] - want))); diff > 1e-3 {
			t.Fatalf("logits[%d]=%g want %g", i, fake.logits[i], want)
		}
		if fake.bias[i] != bias[i] {
			t.Fatalf("bias[%d]=%g want %g", i, fake.bias[i], bias[i])
		}
	}
	if len(fake.picks) != cfg.NumExpertsPerTok || len(fake.out) != cfg.HiddenSize {
		t.Fatalf("oracle picks=%d output=%d", len(fake.picks), len(fake.out))
	}
	gotResidual := be.Read(residualTensor)
	for _, i := range []int{0, 6, 7, cfg.HiddenSize - 1} {
		want := float32(1) + fake.out[i]
		if gotResidual[i] != want {
			t.Fatalf("residual[%d]=%g want resident-oracle %g", i, gotResidual[i], want)
		}
	}

	s.Close()
	if !fake.closed {
		t.Fatal("session close did not close live V4 expert runtime")
	}
}
