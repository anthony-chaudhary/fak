package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// writeF32 writes a little-endian float32 dump (the on-disk tap format).
func writeF32(t *testing.T, path string, v []float32) {
	t.Helper()
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func writeMeta(t *testing.T, dir string, m meta) {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "meta.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// makeStack writes a synthetic per-layer dump pair into fakDir/llamaDir: identical hidden states
// up to a small quant-dequant "noise floor" everywhere, plus a planted divergence at layer
// divLayer (cosine driven well below the floor). Returns the metas written.
func makeStack(t *testing.T, fakDir, llamaDir string, hidden, layers, divLayer int, noise, divMag float64) {
	t.Helper()
	r := rand.New(rand.NewSource(1))
	var lmeta []layerMeta
	for l := 0; l < layers; l++ {
		kind := "linear_attention"
		if (l+1)%4 == 0 {
			kind = "full_attention"
		}
		lmeta = append(lmeta, layerMeta{Index: l, Kind: kind})

		base := make([]float32, hidden)
		for i := range base {
			base[i] = float32(r.NormFloat64())
		}
		fak := make([]float32, hidden)
		lla := make([]float32, hidden)
		for i := range base {
			// both engines carry the same hidden plus an independent tiny noise term (the
			// realistic quant-dequant floor that keeps early-layer cosine just under 1).
			fak[i] = base[i] + float32(r.NormFloat64()*noise)
			lla[i] = base[i] + float32(r.NormFloat64()*noise)
		}
		if l == divLayer {
			// plant an anomalous divergence: rotate fak's hidden by a large perturbation.
			for i := range fak {
				fak[i] += float32(r.NormFloat64() * divMag)
			}
		}
		name := layerFile(l)
		writeF32(t, filepath.Join(fakDir, name), fak)
		writeF32(t, filepath.Join(llamaDir, name), lla)
	}
	m := meta{Hidden: hidden, DecodeStep: 3, PromptIDs: []int{248068, 198}, Layers: lmeta, LlamaBuild: "b9707", Quant: "q4_k_m"}
	writeMeta(t, fakDir, m)
	writeMeta(t, llamaDir, m)
}

func layerFile(l int) string {
	return "layer_" + twoDigit(l) + ".f32"
}

func twoDigit(l int) string {
	if l < 10 {
		return "0" + string(rune('0'+l))
	}
	return string(rune('0'+l/10)) + string(rune('0'+l%10))
}

func dirs(t *testing.T) (string, string) {
	t.Helper()
	root := t.TempDir()
	fak := filepath.Join(root, "fak")
	lla := filepath.Join(root, "llama")
	if err := os.MkdirAll(fak, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(lla, 0o755); err != nil {
		t.Fatal(err)
	}
	return fak, lla
}

// TestFinderLocatesPlantedLayer is the load-bearing host-independent check (§5 step 1): inject a
// known divergence at layer 7 and assert the finder names it, with the agreeing layers staying
// above the noise floor.
func TestFinderLocatesPlantedLayer(t *testing.T) {
	fak, lla := dirs(t)
	const hidden, layers, plant = 256, 16, 7
	makeStack(t, fak, lla, hidden, layers, plant, 1e-5, 0.3)

	w, err := runProbe(fak, lla, 0.9999, false, 4, 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	if w.FirstDivergeLayer == nil {
		t.Fatalf("expected a divergence at layer %d, got parity (per_layer cosines: %v)", plant, cosList(w))
	}
	if *w.FirstDivergeLayer != plant {
		t.Fatalf("first_divergence_layer = %d, want %d", *w.FirstDivergeLayer, plant)
	}
	// Layers before the plant must be above-threshold (the agreeing floor).
	for _, s := range w.PerLayer {
		if s.Layer < plant && s.Cosine < w.Threshold {
			t.Fatalf("pre-plant layer %d cosine %.8f below threshold %.8f — noise floor too high", s.Layer, s.Cosine, w.Threshold)
		}
	}
}

// TestFinderAutoThreshold checks the anomaly (-auto) mode sets the threshold just under the
// measured baseline floor and still pins the planted layer.
func TestFinderAutoThreshold(t *testing.T) {
	fak, lla := dirs(t)
	const hidden, layers, plant = 256, 16, 9
	makeStack(t, fak, lla, hidden, layers, plant, 1e-5, 0.3)

	w, err := runProbe(fak, lla, 0.9999, true, 4, 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	if w.ThresholdMode != "auto(baseline-gap)" {
		t.Fatalf("threshold mode = %q", w.ThresholdMode)
	}
	if w.FirstDivergeLayer == nil || *w.FirstDivergeLayer != plant {
		t.Fatalf("auto mode: first_divergence_layer = %v, want %d", w.FirstDivergeLayer, plant)
	}
	if w.BaselineCosFloor < 0.9999 {
		t.Fatalf("baseline floor %.8f too low — the agreeing layers should sit near 1", w.BaselineCosFloor)
	}
}

// TestFinderParityWhenNoAnomaly: when both engines agree to within the noise floor at EVERY layer,
// the finder reports parity (first_divergence_layer == null) — the falsifiable closure condition.
func TestFinderParityWhenNoAnomaly(t *testing.T) {
	fak, lla := dirs(t)
	const hidden, layers = 256, 16
	makeStack(t, fak, lla, hidden, layers, -1 /* no plant */, 1e-5, 0)

	w, err := runProbe(fak, lla, 0.9999, false, 4, 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	if w.FirstDivergeLayer != nil {
		t.Fatalf("expected parity, but finder flagged layer %d (cosines: %v)", *w.FirstDivergeLayer, cosList(w))
	}
}

// TestFinderLocatesPlantedOp: with per-op taps in the first-diverging layer, the finder names the
// first op below threshold (the GDN "recurrent" scan, per hypothesis H1's locus).
func TestFinderLocatesPlantedOp(t *testing.T) {
	fak, lla := dirs(t)
	const hidden, layers, plant = 256, 16, 5
	makeStack(t, fak, lla, hidden, layers, plant, 1e-5, 0.3)

	// per-op taps for layer `plant`: convOut + qk_norm agree, recurrent diverges (then gated_norm
	// inherits it). Widths are op-specific; values are arbitrary but cosine-meaningful.
	r := rand.New(rand.NewSource(2))
	planted := map[string]bool{"recurrent": true, "gated_norm": true}
	for _, op := range opOrder {
		w := 64
		a := make([]float32, w)
		b := make([]float32, w)
		for i := range a {
			v := float32(r.NormFloat64())
			a[i] = v
			b[i] = v
			if planted[op] {
				a[i] = v + float32(r.NormFloat64()*0.3)
			} else {
				a[i] = v + float32(r.NormFloat64()*1e-5)
				b[i] = v + float32(r.NormFloat64()*1e-5)
			}
		}
		name := "layer_" + twoDigit(plant) + "_op_" + op + ".f32"
		writeF32(t, filepath.Join(fak, name), a)
		writeF32(t, filepath.Join(lla, name), b)
	}

	w, err := runProbe(fak, lla, 0.9999, false, 4, 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	if w.FirstDivergeLayer == nil || *w.FirstDivergeLayer != plant {
		t.Fatalf("first_divergence_layer = %v, want %d", w.FirstDivergeLayer, plant)
	}
	if w.FirstDivergeOp != "recurrent" {
		t.Fatalf("first_divergence_op = %q, want recurrent (per-op stats: %+v)", w.FirstDivergeOp, w.PerOpInFirstLayer)
	}
}

func TestCosinePrimitive(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	if c := cosine(a, a); math.Abs(c-1) > 1e-12 {
		t.Fatalf("cosine(a,a) = %.12f, want 1", c)
	}
	b := []float32{-1, -2, -3, -4}
	if c := cosine(a, b); math.Abs(c+1) > 1e-12 {
		t.Fatalf("cosine(a,-a) = %.12f, want -1", c)
	}
	if d := maxAbsDiff(a, b); math.Abs(d-8) > 1e-9 {
		t.Fatalf("maxAbsDiff = %.6f, want 8", d)
	}
}

func cosList(w witness) []float64 {
	out := make([]float64, len(w.PerLayer))
	for i, s := range w.PerLayer {
		out[i] = s.Cosine
	}
	return out
}
