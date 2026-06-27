package model

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// loraTestCfg is the tiny synthetic geometry the merge witnesses run on. q_proj is
// [Out=32, In=32] (NumHeads*HeadDim == HiddenSize), a clean LoRA target.
func loraTestCfg() Config {
	return Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 97, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
}

// lcgFill fills a slice with deterministic pseudo-random values in [-scale,scale].
func lcgFill(n int, seed uint64, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		seed = seed*6364136223846793005 + 1442695040888963407
		u := float32(seed>>40) / float32(1<<24)
		out[i] = (u*2 - 1) * scale
	}
	return out
}

func loraMaxAbsDiff(a, b []float32) float32 {
	var m float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

func newTestAdapter(bundle, target string, out, in, rank int, alpha float64, seed uint64) *LoRAAdapter {
	return &LoRAAdapter{
		Adapter: bundle, Target: target,
		In: in, Out: out, Rank: rank, Alpha: alpha,
		A: lcgFill(rank*in, seed, 0.1),
		B: lcgFill(out*rank, seed^0xabcdef, 0.1),
	}
}

// TestLoRAAdapterDeltaMatchesMerge proves the two ways to apply an adapter agree:
// the dynamic decode-time term (Delta) equals the difference a merged weight makes
// to the base matvec. This is the "apply at decode time" == "weight merging"
// equivalence both acceptance behaviors rest on.
func TestLoRAAdapterDeltaMatchesMerge(t *testing.T) {
	const out, in, rank = 48, 40, 6
	w := lcgFill(out*in, 1, 0.1)
	x := lcgFill(in, 2, 1.0)
	a := newTestAdapter("b", "model.layers.0.self_attn.q_proj.weight", out, in, rank, 12, 7)

	dynamic := matRows(w, x, out, in)
	delta := a.Delta(x)
	for o := range dynamic {
		dynamic[o] += delta[o]
	}

	wm := append([]float32(nil), w...)
	if err := a.MergeInto(wm); err != nil {
		t.Fatalf("MergeInto: %v", err)
	}
	merged := matRows(wm, x, out, in)

	if d := loraMaxAbsDiff(dynamic, merged); d > 1e-4 {
		t.Fatalf("dynamic apply vs merged weight diverge: max|Δ|=%g (>1e-4)", d)
	}
	// Sanity: the adapter is non-trivial (it actually moves the output).
	base := matRows(w, x, out, in)
	if loraMaxAbsDiff(base, merged) < 1e-6 {
		t.Fatalf("adapter is a no-op; witness would be vacuous")
	}
}

// TestLoRASetComposeAndSwitch covers "support 2+ adapters simultaneously" and
// "dynamic adapter switching": two bundles on the same target sum, and toggling a
// bundle changes the delta to exactly the remaining contribution.
func TestLoRASetComposeAndSwitch(t *testing.T) {
	const out, in, rank = 32, 32, 4
	target := "model.layers.0.self_attn.q_proj.weight"
	x := lcgFill(in, 3, 1.0)
	a1 := newTestAdapter("math", target, out, in, rank, 8, 11)
	a2 := newTestAdapter("code", target, out, in, rank, 8, 13)

	set := NewLoRASet()
	if err := set.Add(a1); err != nil {
		t.Fatal(err)
	}
	if err := set.Add(a2); err != nil {
		t.Fatal(err)
	}

	d1 := a1.Delta(x)
	d2 := a2.Delta(x)
	want := make([]float32, out)
	for o := range want {
		want[o] = d1[o] + d2[o]
	}
	both := set.Delta(target, x)
	if d := loraMaxAbsDiff(both, want); d > 1e-5 {
		t.Fatalf("composed delta != sum of adapters: max|Δ|=%g", d)
	}

	// Switch off "code": the set delta must equal just the "math" contribution.
	set.Deactivate("code")
	if set.IsActive("code") {
		t.Fatal("code should be inactive after Deactivate")
	}
	only := set.Delta(target, x)
	if d := loraMaxAbsDiff(only, d1); d > 1e-6 {
		t.Fatalf("after switching off code, delta != math alone: max|Δ|=%g", d)
	}

	// Switch both off: no contribution at all (cheap base path).
	set.Deactivate("math")
	if set.Delta(target, x) != nil {
		t.Fatal("delta should be nil when no bundle is active")
	}
	if len(set.Targets()) != 0 {
		t.Fatalf("no active targets expected, got %v", set.Targets())
	}

	// Switch one back on (dynamic switching restores without reloading).
	set.Activate("code")
	if d := loraMaxAbsDiff(set.Delta(target, x), d2); d > 1e-6 {
		t.Fatalf("reactivated code delta wrong: max|Δ|=%g", d)
	}
}

// TestLoRAOverheadWithinBudget pins the "within 5% overhead" acceptance: the
// dynamic-apply overhead of a typical rank-16 adapter is well under 5%, the merge
// path is 0, and the metric actually discriminates (a fat rank exceeds the budget).
func TestLoRAOverheadWithinBudget(t *testing.T) {
	typical := &LoRAAdapter{In: 2048, Out: 2048, Rank: 16}
	if of := typical.OverheadFraction(); of >= 0.05 {
		t.Fatalf("rank-16 dynamic overhead %.4f should be < 5%%", of)
	}
	// The metric is not vacuously small: a rank-128 adapter blows the budget, so a
	// caller can honestly choose merge (0%) over dynamic when rank is large.
	fat := &LoRAAdapter{In: 2048, Out: 2048, Rank: 128}
	if of := fat.OverheadFraction(); of < 0.05 {
		t.Fatalf("rank-128 dynamic overhead %.4f should exceed 5%% (metric must discriminate)", of)
	}
}

// TestMergeLoRAChangesForward witnesses end-to-end on a synthetic CPU model that a
// merged adapter actually reaches decode (the forward logits move), while a nil or
// empty set leaves the forward byte-identical (Llama-invariance of the no-adapter
// path — the property that makes shipping the seam safe).
func TestMergeLoRAChangesForward(t *testing.T) {
	cfg := loraTestCfg()
	ids := []int{1, 5, 9, 2}
	target := "model.layers.0.self_attn.q_proj.weight"

	base := NewSynthetic(cfg).Forward(ids)

	// nil and empty sets are exact no-ops.
	m := NewSynthetic(cfg)
	if err := m.MergeLoRA(nil); err != nil {
		t.Fatalf("MergeLoRA(nil): %v", err)
	}
	if err := m.MergeLoRA(NewLoRASet()); err != nil {
		t.Fatalf("MergeLoRA(empty): %v", err)
	}
	noop := m.Forward(ids)
	for p := 0; p < base.Seq; p++ {
		assertFloat32BitsEqual(t, "nil/empty merge invariance pos "+itoa(p), noop.Logits[p], base.Logits[p])
	}

	// A real adapter on q_proj moves the logits.
	set := NewLoRASet()
	if err := set.Add(newTestAdapter("math", target, 32, 32, 4, 8, 21)); err != nil {
		t.Fatal(err)
	}
	merged := NewSynthetic(cfg)
	if err := merged.MergeLoRA(set); err != nil {
		t.Fatalf("MergeLoRA: %v", err)
	}
	got := merged.Forward(ids)
	moved := false
	for p := 0; p < base.Seq; p++ {
		if loraMaxAbsDiff(got.Logits[p], base.Logits[p]) > 1e-5 {
			moved = true
		}
	}
	if !moved {
		t.Fatal("merged adapter did not change the forward logits — it never reached decode")
	}

	// Merging the same adapter into a hand-merged copy of the weight reproduces the
	// same forward (the model-level merge == the weight-level merge).
	ref := NewSynthetic(cfg)
	if err := newTestAdapter("math", target, 32, 32, 4, 8, 21).MergeInto(ref.tensor(target)); err != nil {
		t.Fatal(err)
	}
	refOut := ref.Forward(ids)
	for p := 0; p < base.Seq; p++ {
		assertFloat32BitsEqual(t, "model-merge == weight-merge pos "+itoa(p), got.Logits[p], refOut.Logits[p])
	}
}

// TestMergeLoRAMissingTargetErrors confirms an adapter for an absent base weight is
// reported, not silently dropped.
func TestMergeLoRAMissingTargetErrors(t *testing.T) {
	set := NewLoRASet()
	if err := set.Add(newTestAdapter("x", "model.layers.0.nonexistent.weight", 8, 8, 2, 4, 1)); err != nil {
		t.Fatal(err)
	}
	if err := NewSynthetic(loraTestCfg()).MergeLoRA(set); err == nil {
		t.Fatal("expected an error merging an adapter whose target weight is absent")
	}
}

// TestLoadLoRAAdapterFromDisk witnesses "load single LoRA adapter" against a real
// PEFT-shaped artifact (adapter_config.json + adapter_model.safetensors) written to
// a temp dir, then proves the loaded factors reproduce the same delta as the
// in-memory adapter they encode.
func TestLoadLoRAAdapterFromDisk(t *testing.T) {
	dir := t.TempDir()
	const out, in, rank = 32, 32, 4
	alpha := 8.0
	target := "model.layers.0.self_attn.q_proj.weight"
	aMat := lcgFill(rank*in, 31, 0.1)
	bMat := lcgFill(out*rank, 37, 0.1)

	// adapter_config.json
	cfg := LoRAConfig{R: rank, Alpha: alpha, TargetModules: []string{"q_proj"}, PeftType: "LORA"}
	cfgBytes, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"), cfgBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	// adapter_model.safetensors with the PEFT-named factor tensors.
	writeSafetensors(t, filepath.Join(dir, "adapter_model.safetensors"), []NamedTensorF32{
		{Name: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", Shape: []int{rank, in}, Data: aMat},
		{Name: "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", Shape: []int{out, rank}, Data: bMat},
	})

	set, err := LoadLoRAAdapter(dir)
	if err != nil {
		t.Fatalf("LoadLoRAAdapter: %v", err)
	}
	if got := set.Bundles(); len(got) != 1 || got[0] != filepath.Base(dir) {
		t.Fatalf("bundles = %v, want [%s]", got, filepath.Base(dir))
	}
	if got := set.Targets(); len(got) != 1 || got[0] != target {
		t.Fatalf("targets = %v, want [%s]", got, target)
	}

	x := lcgFill(in, 41, 1.0)
	want := (&LoRAAdapter{In: in, Out: out, Rank: rank, Alpha: alpha, A: aMat, B: bMat}).Delta(x)
	got := set.Delta(target, x)
	if d := loraMaxAbsDiff(got, want); d > 1e-6 {
		t.Fatalf("loaded adapter delta != reference: max|Δ|=%g", d)
	}
}

// writeSafetensors writes a minimal f32 safetensors file (8-byte LE header length,
// JSON header, then the raw f32 data) matching the format the model loader reads.
func writeSafetensors(t *testing.T, path string, tensors []NamedTensorF32) {
	t.Helper()
	type entry struct {
		Dtype       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	hdr := map[string]entry{}
	var data []byte
	for _, tn := range tensors {
		start := len(data)
		buf := make([]byte, len(tn.Data)*4)
		for i, v := range tn.Data {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		data = append(data, buf...)
		hdr[tn.Name] = entry{Dtype: "F32", Shape: tn.Shape, DataOffsets: []int{start, len(data)}}
	}
	hdrBytes, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	var out []byte
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(hdrBytes)))
	out = append(out, lenBuf[:]...)
	out = append(out, hdrBytes...)
	out = append(out, data...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}
