package model

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func readTapF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(b)%4 != 0 {
		t.Fatalf("%s: %d bytes not a multiple of 4", path, len(b))
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func tapFileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// TestHiddenTapDumpsPerLayerAndPerOp is the host-independent witness for the FAK_HIDDEN_TAP
// decode-time hidden dump (token3-drift-investigation §5 step 3): it drives the tiny qwen35 hybrid
// fixture through one real DECODE forward at a known absolute position with the tap armed, then
// asserts the on-disk artifacts match the format the token3-divergence-probe consumes:
//   - one layer_NN.f32 (width=hidden) per decoder layer,
//   - meta.json naming each layer's kind (linear_attention vs full_attention),
//   - the five GDN per-op taps (convOut/qk_norm/recurrent/gated_norm/out) inside every
//     linear_attention layer and NONE inside the full_attention layer, and
//   - that the dumped last-layer residual, run through finalNorm, reproduces the forward's own
//     returned hidden — i.e. the tap captured the real computation, not a stale or empty buffer.
func TestHiddenTapDumpsPerLayerAndPerOp(t *testing.T) {
	cfg := qwen35HybridTestCfg() // 4 layers: linear,linear,linear,full; hidden=32
	m := NewSynthetic(cfg)

	prompt := []int{3, 7, 11, 5, 17, 19, 23}
	s := m.NewSession()
	s.Prefill(prompt) // positions 0..len-1

	dir := t.TempDir()
	const decodePos = 7 // first decode step (== len(prompt)); the forward we tap
	s.tap = &hiddenTap{dir: dir, pos: decodePos, ops: true, promptIDs: prompt}

	hidden := s.tokenHidden(29, decodePos) // one decode forward at the armed position
	if s.tap.err != nil {
		t.Fatalf("tap reported a write error: %v", s.tap.err)
	}

	// ---- meta.json ----
	mb, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		t.Fatalf("read meta.json: %v", err)
	}
	var md hiddenTapMeta
	if err := json.Unmarshal(mb, &md); err != nil {
		t.Fatalf("parse meta.json: %v", err)
	}
	if md.Hidden != cfg.HiddenSize {
		t.Fatalf("meta hidden = %d, want %d", md.Hidden, cfg.HiddenSize)
	}
	if md.DecodeStep != decodePos {
		t.Fatalf("meta decode_step = %d, want %d", md.DecodeStep, decodePos)
	}
	if len(md.Layers) != cfg.NumLayers {
		t.Fatalf("meta layers = %d, want %d", len(md.Layers), cfg.NumLayers)
	}
	wantKind := []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"}
	for l, lm := range md.Layers {
		if lm.Index != l || lm.Kind != wantKind[l] {
			t.Fatalf("meta layer %d = {%d,%s}, want {%d,%s}", l, lm.Index, lm.Kind, l, wantKind[l])
		}
	}

	// ---- per-layer residual files (width == hidden) ----
	for l := 0; l < cfg.NumLayers; l++ {
		f := filepath.Join(dir, "layer_0"+string(rune('0'+l))+".f32")
		v := readTapF32(t, f)
		if len(v) != cfg.HiddenSize {
			t.Fatalf("layer %d dump width = %d, want %d", l, len(v), cfg.HiddenSize)
		}
	}

	// ---- the dump is the REAL residual: finalNorm(last layer) == the forward's returned hidden ----
	last := readTapF32(t, filepath.Join(dir, "layer_03.f32"))
	got := m.finalNorm(last)
	if d := maxAbsDelta(got, hidden); d > 1e-4 {
		t.Fatalf("finalNorm(dumped last residual) differs from returned hidden, max|delta|=%g", d)
	}

	// ---- per-op GDN taps: present (correct widths) in the 3 linear layers, absent in the full-attn layer ----
	keyDim := cfg.LinearNumKeyHeads * cfg.LinearKeyHeadDim     // 16
	valDim := cfg.LinearNumValueHeads * cfg.LinearValueHeadDim // 32
	convDim := 2*keyDim + valDim                               // 64
	wantOpWidth := map[string]int{
		"convOut":    convDim,
		"qk_norm":    keyDim,
		"recurrent":  valDim,
		"gated_norm": valDim,
		"out":        cfg.HiddenSize,
	}
	for _, l := range []int{0, 1, 2} { // linear_attention layers
		for op, w := range wantOpWidth {
			f := filepath.Join(dir, "layer_0"+string(rune('0'+l))+"_op_"+op+".f32")
			v := readTapF32(t, f)
			if len(v) != w {
				t.Fatalf("layer %d op %s width = %d, want %d", l, op, len(v), w)
			}
		}
	}
	for op := range wantOpWidth { // full_attention layer 3 has no GDN ops
		f := filepath.Join(dir, "layer_03_op_"+op+".f32")
		if tapFileExists(f) {
			t.Fatalf("full_attention layer 3 unexpectedly wrote a GDN op tap: %s", f)
		}
	}
}

// TestHiddenTapNonTargetPositionDumpsNothing pins the cost/safety contract: a tap armed for a
// position the forward never reaches writes NOTHING (no meta.json, no layer files), so prefill and
// non-target decode steps are untouched.
func TestHiddenTapNonTargetPositionDumpsNothing(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	s := m.NewSession()
	s.Prefill([]int{3, 7, 11})

	dir := t.TempDir()
	s.tap = &hiddenTap{dir: dir, pos: 999, ops: true} // never reached
	_ = s.tokenHidden(5, 3)

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected no tap output for a non-target position, got %d files", len(entries))
	}
}

// TestHiddenTapFromEnv pins the env contract (FAK_HIDDEN_TAP / _POS / _OPS).
func TestHiddenTapFromEnv(t *testing.T) {
	if hiddenTapFromEnv() != nil {
		// In case the harness has it set; make the unset-case assertion explicit instead.
		t.Skip("FAK_HIDDEN_TAP already set in the environment; skipping unset-case check")
	}
	dir := t.TempDir()
	t.Setenv("FAK_HIDDEN_TAP", dir)
	t.Setenv("FAK_HIDDEN_TAP_POS", "13")
	tp := hiddenTapFromEnv()
	if tp == nil {
		t.Fatal("hiddenTapFromEnv() = nil with FAK_HIDDEN_TAP set")
	}
	if tp.dir != dir || tp.pos != 13 || !tp.ops {
		t.Fatalf("env tap = {dir:%q pos:%d ops:%v}, want {%q 13 true}", tp.dir, tp.pos, tp.ops, dir)
	}
	t.Setenv("FAK_HIDDEN_TAP_OPS", "0")
	if tp := hiddenTapFromEnv(); tp == nil || tp.ops {
		t.Fatalf("FAK_HIDDEN_TAP_OPS=0 should disable per-op taps, got %+v", tp)
	}
}
