// Command token3-divergence-probe is the host-independent comparison + first-divergence finder
// for the Qwen3.6-27B token-3 correctness drift (token3-drift-investigation-2026-06-28.md §3c/§3d,
// §5 step 1). It turns "drifts at token 3" into "first diverges at layer L, op O".
//
// It consumes two directories of per-layer hidden-state dumps — one from fak (the FAK_HIDDEN_TAP
// follow-on) and one from llama.cpp b9707 (eval-callback / graph dump) — captured on the SAME
// fixed prompt at the SAME decode step, computes per-layer cosine + max|Δ|, and reports the first
// layer whose cosine drops below an ANOMALY threshold (and, if per-op taps are present for that
// layer, the first op that does). It emits the gradeable §3d witness JSON.
//
// Why an ANOMALY threshold and not a 1-ULP floor: the sibling experiment
// gdn-divergence-sensitivity shows that reduction-order / f16-state rounding compounds to only
// ~1e-7..1e-5 relative divergence over the 48-layer GDN stack — ~10^3..10^5x too small to flip
// the observed 1.75-logit near-tie. So the early agreeing layers carry a real (nonzero)
// quant-dequant noise floor, and the diverging layer is where the cosine drops ANOMALOUSLY below
// that floor — a genuine op mismatch, not accumulated rounding. The finder therefore measures the
// baseline floor from the agreeing layers and flags the first layer that falls a gap below it
// (-auto), in addition to the fixed -threshold mode.
//
// This program is PURE STDLIB (no internal/model dependency): the comparison logic is unit-tested
// on any box (probe_test.go injects a known divergence and asserts the finder returns it). It does
// NOT itself run fak or llama.cpp or load the 27B artifact — producing the two dumps is the
// Mac/artifact-gated step (investigation §5 steps 3-5). This is the comparator that consumes them.
//
// Dump format (one directory per engine):
//
//	<dir>/meta.json          {"hidden":H,"decode_step":S,"prompt_ids":[...],
//	                          "layers":[{"index":0,"kind":"linear_attention"},...]}
//	<dir>/layer_<NN>.f32     little-endian float32 x H — residual-stream hidden after layer NN
//	<dir>/layer_<NN>_op_<name>.f32   (optional) per-op tap inside layer NN (convOut|qk_norm|
//	                          recurrent|gated_norm|out), little-endian float32 x op-width
//
// Run:
//
//	go run ./experiments/qwen36/token3-divergence-probe -fak <fakdir> -llama <llamadir> -out probe.json
//	go run ./experiments/qwen36/token3-divergence-probe -fak <fakdir> -llama <llamadir> -auto
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// ---- dump format ----

type layerMeta struct {
	Index int    `json:"index"`
	Kind  string `json:"kind"`
}

type meta struct {
	Hidden     int         `json:"hidden"`
	DecodeStep int         `json:"decode_step"`
	PromptIDs  []int       `json:"prompt_ids"`
	Layers     []layerMeta `json:"layers"`
	// LlamaBuild / Quant are optional provenance carried through to the witness.
	LlamaBuild string `json:"llamacpp_build,omitempty"`
	Quant      string `json:"quant,omitempty"`
}

func loadMeta(dir string) (meta, error) {
	var m meta
	b, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		return m, err
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return m, fmt.Errorf("%s/meta.json: %w", dir, err)
	}
	return m, nil
}

func loadF32(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("%s: length %d not a multiple of 4", path, len(b))
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out, nil
}

// ---- comparison primitives (mirror oracle_test.go cosine/argmax) ----

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		if na == 0 && nb == 0 {
			return 1
		}
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func maxAbsDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

// ---- per-layer stats + first-divergence finder ----

type LayerStat struct {
	Layer  int     `json:"layer"`
	Kind   string  `json:"kind"`
	Cosine float64 `json:"cosine"`
	MaxAbs float64 `json:"max_abs"`
}

// perLayerStats loads layer_<NN>.f32 from both dirs and computes cosine/max|Δ| per layer. It
// requires the two metas to agree on hidden width and the per-layer kinds (a mismatch is a setup
// error, not a model divergence).
func perLayerStats(fakDir, llamaDir string, fm, lm meta) ([]LayerStat, error) {
	if fm.Hidden != lm.Hidden {
		return nil, fmt.Errorf("hidden mismatch: fak %d vs llama %d", fm.Hidden, lm.Hidden)
	}
	if len(fm.Layers) != len(lm.Layers) {
		return nil, fmt.Errorf("layer-count mismatch: fak %d vs llama %d", len(fm.Layers), len(lm.Layers))
	}
	stats := make([]LayerStat, 0, len(fm.Layers))
	for i := range fm.Layers {
		if fm.Layers[i].Index != lm.Layers[i].Index || fm.Layers[i].Kind != lm.Layers[i].Kind {
			return nil, fmt.Errorf("layer %d metadata mismatch: fak {%d,%s} vs llama {%d,%s}",
				i, fm.Layers[i].Index, fm.Layers[i].Kind, lm.Layers[i].Index, lm.Layers[i].Kind)
		}
		idx := fm.Layers[i].Index
		name := fmt.Sprintf("layer_%02d.f32", idx)
		fa, err := loadF32(filepath.Join(fakDir, name))
		if err != nil {
			return nil, err
		}
		la, err := loadF32(filepath.Join(llamaDir, name))
		if err != nil {
			return nil, err
		}
		if len(fa) != fm.Hidden || len(la) != fm.Hidden {
			return nil, fmt.Errorf("layer %d width: fak %d, llama %d, want hidden %d", idx, len(fa), len(la), fm.Hidden)
		}
		stats = append(stats, LayerStat{Layer: idx, Kind: fm.Layers[i].Kind, Cosine: cosine(fa, la), MaxAbs: maxAbsDiff(fa, la)})
	}
	return stats, nil
}

// baselineFloor is the lowest cosine among the first `n` (assumed-agreeing) layers — the measured
// quant-dequant noise floor the anomaly threshold sits just below.
func baselineFloor(stats []LayerStat, n int) float64 {
	if n > len(stats) {
		n = len(stats)
	}
	if n == 0 {
		return 1
	}
	lo := stats[0].Cosine
	for i := 1; i < n; i++ {
		if stats[i].Cosine < lo {
			lo = stats[i].Cosine
		}
	}
	return lo
}

// firstBelow returns the index (into stats) of the first layer whose cosine < threshold, or -1.
func firstBelow(stats []LayerStat, threshold float64) int {
	for i := range stats {
		if stats[i].Cosine < threshold {
			return i
		}
	}
	return -1
}

// ---- per-op localization within the first-diverging layer ----

var opOrder = []string{"convOut", "qk_norm", "recurrent", "gated_norm", "out"}

type OpStat struct {
	Op     string  `json:"op"`
	Cosine float64 `json:"cosine"`
	MaxAbs float64 `json:"max_abs"`
}

// perOpStats loads any layer_<NN>_op_<name>.f32 taps present in BOTH dirs for the given layer and
// returns them in the canonical GDN op order. Missing op files are skipped (per-op taps are an
// optional finer instrumentation).
func perOpStats(fakDir, llamaDir string, layer int) ([]OpStat, error) {
	var out []OpStat
	for _, op := range opOrder {
		name := fmt.Sprintf("layer_%02d_op_%s.f32", layer, op)
		fp := filepath.Join(fakDir, name)
		lp := filepath.Join(llamaDir, name)
		if !fileExists(fp) || !fileExists(lp) {
			continue
		}
		fa, err := loadF32(fp)
		if err != nil {
			return nil, err
		}
		la, err := loadF32(lp)
		if err != nil {
			return nil, err
		}
		if len(fa) != len(la) {
			return nil, fmt.Errorf("op %s layer %d width mismatch: fak %d vs llama %d", op, layer, len(fa), len(la))
		}
		out = append(out, OpStat{Op: op, Cosine: cosine(fa, la), MaxAbs: maxAbsDiff(fa, la)})
	}
	return out, nil
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// ---- the witness ----

type witness struct {
	Schema            string      `json:"schema"`
	LlamaBuild        string      `json:"llamacpp_build,omitempty"`
	Quant             string      `json:"quant,omitempty"`
	DecodeStep        int         `json:"decode_step"`
	PromptIDs         []int       `json:"prompt_ids,omitempty"`
	Threshold         float64     `json:"threshold"`
	ThresholdMode     string      `json:"threshold_mode"`
	BaselineCosFloor  float64     `json:"baseline_cosine_floor"`
	PerLayer          []LayerStat `json:"per_layer"`
	FirstDivergeLayer *int        `json:"first_divergence_layer"`
	FirstDivergeKind  string      `json:"first_divergence_kind,omitempty"`
	PerOpInFirstLayer []OpStat    `json:"per_op_in_first_layer,omitempty"`
	FirstDivergeOp    string      `json:"first_divergence_op,omitempty"`
}

// runProbe is the host-independent core: load both dumps, compute stats, locate the first
// anomalous divergence (layer then op), and build the witness. `threshold` is used directly unless
// `auto` is set, in which case the effective threshold is baselineFloor(first `baselineN`) - gap.
func runProbe(fakDir, llamaDir string, threshold float64, auto bool, baselineN int, gap float64) (witness, error) {
	fm, err := loadMeta(fakDir)
	if err != nil {
		return witness{}, err
	}
	lm, err := loadMeta(llamaDir)
	if err != nil {
		return witness{}, err
	}
	stats, err := perLayerStats(fakDir, llamaDir, fm, lm)
	if err != nil {
		return witness{}, err
	}
	floor := baselineFloor(stats, baselineN)
	mode := "fixed"
	eff := threshold
	if auto {
		mode = "auto(baseline-gap)"
		eff = floor - gap
		if eff > threshold {
			eff = threshold // never weaker than the hard floor
		}
	}
	w := witness{
		Schema:           "qwen36-token3-divergence/v1",
		LlamaBuild:       lm.LlamaBuild,
		Quant:            lm.Quant,
		DecodeStep:       fm.DecodeStep,
		PromptIDs:        fm.PromptIDs,
		Threshold:        eff,
		ThresholdMode:    mode,
		BaselineCosFloor: floor,
		PerLayer:         stats,
	}
	idx := firstBelow(stats, eff)
	if idx < 0 {
		return w, nil // parity: first_divergence_layer == null
	}
	l := stats[idx].Layer
	w.FirstDivergeLayer = &l
	w.FirstDivergeKind = stats[idx].Kind
	ops, err := perOpStats(fakDir, llamaDir, l)
	if err != nil {
		return w, err
	}
	w.PerOpInFirstLayer = ops
	for _, o := range ops {
		if o.Cosine < eff {
			w.FirstDivergeOp = o.Op
			break
		}
	}
	return w, nil
}

func main() {
	fakDir := flag.String("fak", "", "directory of fak per-layer dumps (layer_NN.f32 + meta.json)")
	llamaDir := flag.String("llama", "", "directory of llama.cpp per-layer dumps")
	threshold := flag.Float64("threshold", 0.9999, "cosine threshold below which a layer is 'diverged'")
	auto := flag.Bool("auto", false, "set the threshold to (baseline floor over the first -baseline-layers) - -gap")
	baselineN := flag.Int("baseline-layers", 4, "number of leading (assumed-agreeing) layers used to measure the noise floor")
	gap := flag.Float64("gap", 1e-3, "anomaly gap below the baseline floor (-auto mode)")
	out := flag.String("out", "", "write the witness JSON here (default: stdout)")
	flag.Parse()

	if *fakDir == "" || *llamaDir == "" {
		fmt.Fprintln(os.Stderr, "usage: -fak <dir> -llama <dir> [-out probe.json] [-auto]")
		os.Exit(2)
	}
	w, err := runProbe(*fakDir, *llamaDir, *threshold, *auto, *baselineN, *gap)
	if err != nil {
		fmt.Fprintf(os.Stderr, "probe: %v\n", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	if *out != "" {
		f, err := os.Create(*out)
		if err != nil {
			fmt.Fprintf(os.Stderr, "probe: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		enc = json.NewEncoder(f)
	}
	enc.SetIndent("", "  ")
	if err := enc.Encode(w); err != nil {
		fmt.Fprintf(os.Stderr, "probe: %v\n", err)
		os.Exit(1)
	}

	// Human one-liner to stderr so a -out run still narrates.
	if w.FirstDivergeLayer == nil {
		fmt.Fprintf(os.Stderr, "PARITY: no layer below cosine %.6g (baseline floor %.6g)\n", w.Threshold, w.BaselineCosFloor)
	} else {
		op := w.FirstDivergeOp
		if op == "" {
			op = "(no per-op taps)"
		}
		fmt.Fprintf(os.Stderr, "DIVERGES at layer %d (%s), op %s — cosine below %.6g (baseline floor %.6g)\n",
			*w.FirstDivergeLayer, w.FirstDivergeKind, op, w.Threshold, w.BaselineCosFloor)
	}
}
