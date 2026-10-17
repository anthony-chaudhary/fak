//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math/rand"
	"os"
	"slices"
	"testing"
)

// prefillShapeBenchEnv runs TestQwen35PrefillShapeGPUTime. Unset, the test skips: GPU time is only
// meaningful when deliberately captured on an otherwise idle GPU.
const prefillShapeBenchEnv = "FAK_PREFILL_SHAPE_BENCH"

// qwen35PrefillShape is one Qwen3.5-4B Q4_K_M device-prefill projection as FAK keeps it resident
// on Metal, with its count across the 32 layers (24 GDN + 8 full attention).
type qwen35PrefillShape struct {
	name, format string
	out, in      int
	count        float64
}

var qwen35PrefillShapes = []qwen35PrefillShape{
	{"gdn.qkv", "q5_k", 8192, 2560, 24},
	{"ffn.gate_up", "q4_k", 9216, 2560, 64},
	{"ffn.down", "q4_k", 2560, 9216, 16},
	{"ffn.down", "q6_k", 2560, 9216, 16},
	{"gdn.z", "q4_k", 4096, 2560, 24},
	{"gdn.out", "q8_0", 2560, 4096, 24},
	{"gdn.ba", "q8_0", 32, 2560, 48},
	{"attn.q", "q4_k", 8192, 2560, 8},
	{"attn.kv", "q4_k", 1024, 2560, 11},
	{"attn.v", "q6_k", 1024, 2560, 5},
	{"attn.o", "q4_k", 2560, 4096, 8},
}

// TestQwen35PrefillShapeGPUTime times each Qwen3.5-4B projection inside one P=128 projection graph
// (the device prefill panel), encoding it 8 times per command buffer and taking the median
// per-encode GPU window over interleaved repeats, then reports the per-panel projection GPU time
// the shapes add up to. Measurement only: nothing is gated.
//
// fak-test:runtime integration est=3s lane=opt-in
func TestQwen35PrefillShapeGPUTime(t *testing.T) {
	if os.Getenv(prefillShapeBenchEnv) == "" {
		t.Skipf("%s unset; per-shape GPU time is only captured on request", prefillShapeBenchEnv)
	}
	if !Available() {
		t.Skip("Metal unavailable")
	}
	const P, encodes, repeats = 128, 8, 7
	rng := rand.New(rand.NewSource(0x5eed))
	type arm struct {
		shape  qwen35PrefillShape
		encode func(g *ProjectionGraph, in *GraphResult) (*GraphResult, error)
		x      []float32
		ms     []float64
	}
	var arms []*arm
	for i, s := range qwen35PrefillShapes {
		a := &arm{shape: s, x: make([]float32, P*s.in)}
		for j := range a.x {
			a.x[j] = float32(rng.NormFloat64())
		}
		switch s.format {
		case "q4_k":
			w := UploadQ4K(q4kTestRaw(s.out, s.in, uint64(0x400+i)), s.out, s.in)
			if w == nil {
				t.Fatalf("%s: UploadQ4K failed", s.name)
			}
			defer w.Release()
			a.encode = func(g *ProjectionGraph, in *GraphResult) (*GraphResult, error) { return g.EncodeQ4KFrom(w, in) }
		case "q5_k":
			w := UploadQ5KGoOwned(randomQ5KRows(rng, s.out, s.in), s.out, s.in)
			if w == nil {
				t.Fatalf("%s: UploadQ5KGoOwned failed", s.name)
			}
			defer w.Release()
			a.encode = func(g *ProjectionGraph, in *GraphResult) (*GraphResult, error) { return g.EncodeQ5KFrom(w, in) }
		case "q6_k":
			w := UploadQ6K(q6kTestRaw(s.out, s.in, uint64(0x600+i)), s.out, s.in)
			if w == nil {
				t.Fatalf("%s: UploadQ6K failed", s.name)
			}
			defer w.Release()
			a.encode = func(g *ProjectionGraph, in *GraphResult) (*GraphResult, error) { return g.EncodeQ6KFrom(w, in) }
		case "q8_0":
			codes, wd, _, _ := q8MulMMFixture(q8MulMMCase{s.out, s.in, 1}, int64(0x800+i))
			w := UploadQ8(codes, wd, s.out, s.in)
			if w == nil {
				t.Fatalf("%s: UploadQ8 failed", s.name)
			}
			defer w.Release()
			var q *QuantizedGraphResult
			a.encode = func(g *ProjectionGraph, in *GraphResult) (*GraphResult, error) {
				if q == nil || q.graph != g {
					var err error
					if q, err = g.QuantizeQ8(in); err != nil {
						return nil, err
					}
				}
				return g.EncodeQ8From(w, q)
			}
		}
		arms = append(arms, a)
	}
	run := func(a *arm) float64 {
		g, err := BeginProjectionGraph(a.x, nil, nil, P, a.shape.in)
		if err != nil {
			t.Fatal(err)
		}
		defer g.Free()
		g.SetQ4KGEMMMode(Q4KGEMMModeForPrompt(P))
		g.SetQ6KGEMMMode(Q6KGEMMModeForPrompt(P))
		in, err := g.Input(a.shape.in)
		if err != nil {
			t.Fatal(err)
		}
		var outs []*GraphResult
		for k := 0; k < encodes; k++ {
			y, err := a.encode(g, in)
			if err != nil {
				t.Fatalf("%s/%s: %v", a.shape.name, a.shape.format, err)
			}
			outs = append(outs, y)
		}
		_, receipt, err := g.FinishRead(outs[0])
		if err != nil {
			t.Fatal(err)
		}
		if !receipt.TimingAvailable {
			t.Fatal("graph GPU timing unavailable")
		}
		return receipt.GPUMilliseconds / encodes
	}
	for _, a := range arms {
		run(a)
	}
	for r := 0; r < repeats; r++ {
		for i := range arms {
			a := arms[(i+r)%len(arms)]
			a.ms = append(a.ms, run(a))
		}
	}
	var panel float64
	for _, a := range arms {
		slices.Sort(a.ms)
		med := a.ms[len(a.ms)/2]
		tflops := 2 * float64(P) * float64(a.shape.out) * float64(a.shape.in) / (med * 1e-3) / 1e12
		panel += med * a.shape.count
		t.Logf("%-12s %-5s %5dx%-5d  %7.3f ms  %5.2f TFLOP/s  x%-3.0f = %7.2f ms/panel", a.shape.name, a.shape.format,
			a.shape.out, a.shape.in, med, tflops, a.shape.count, med*a.shape.count)
	}
	t.Logf("projection GPU per P=%d panel: %.1f ms", P, panel)
}
