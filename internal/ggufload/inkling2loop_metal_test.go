//go:build darwin && arm64 && cgo

package ggufload

import (
	"encoding/binary"
	"math"
	"os"
	"syscall"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestInkling2MetalOracleGreedyMatchesLlamaCpp(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_GGUF to the mainline Inkling-2-Mini UD-Q3_K_XL GGUF")
	}
	m, err := OpenInkling2LoopMetal(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, c := range inkling2OracleCases {
		t.Run(c.name, func(t *testing.T) {
			runInkling2Oracle(t, c, func(prompt []int, n int) ([]int, error) {
				out, stats, err := Inkling2LoopGenerate(m, nil, Inkling2LoopRequest{Prompt: prompt, MaxTokens: n})
				if err == nil {
					t.Logf("packed Metal: %d passes, %d prompt tokens in %s, %d decode steps in %s (%.2f tok/s), ids %v",
						stats.Passes, len(prompt), stats.Prefill, stats.DecodeSteps, stats.Decode, stats.DecodeTokensPerSecond(), out)
				}
				return out, err
			})
		})
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInkling2MetalExpertMLP(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal device required")
	}
	const hidden, width, experts = 5, 7, 3
	data, err := syscall.Mmap(-1, 0, os.Getpagesize(), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(data)
	coeff := func(i int) float32 { return float32((i*7+3)%19-9) / 8 }
	const downAt = experts * 2 * width * hidden
	for i := 0; i < downAt+experts*width*hidden; i++ {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(coeff(i)))
	}
	store, err := metalgemm.NewPackedGGUFStore(data)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	gu, err := store.Matrix(0, uint32(TensorF32), 2*width, hidden, experts)
	if err != nil {
		t.Fatal(err)
	}
	down, err := store.Matrix(downAt*4, uint32(TensorF32), hidden, width, experts)
	if err != nil {
		t.Fatal(err)
	}
	x := []float32{0.5, -0.25, 0.75, -1, 0.125}
	job := metalgemm.PackedExpertMLP{GateUp: gu, Down: down, Selected: []int{2, 0}, Weights: []float32{0.7, 0.3},
		GateUpScale: []float32{0.25, 1, -1.5}, DownScale: []float32{1.5, 1, -0.5}, Beta: 1.6, LinearBeta: 2.8}
	for _, lb := range []float32{2.8, 0} {
		job.LinearBeta = lb
		got, want := make([]float32, hidden), make([]float64, hidden)
		if err := store.ExpertMLP(job, x, got); err != nil {
			t.Fatal(err)
		}
		for slot, e := range job.Selected {
			act := make([]float64, width)
			for r := range act {
				var g, u float64
				for j, v := range x {
					base := (e*2*width+r)*hidden + j
					g += float64(coeff(base)) * float64(v)
					u += float64(coeff(base+width*hidden)) * float64(v)
				}
				g *= float64(job.GateUpScale[e])
				u *= float64(job.GateUpScale[e])
				if lb > 0 {
					u = float64(lb) * math.Tanh(u/float64(lb))
				}
				beta := float64(job.Beta)
				act[r] = beta * math.Tanh(g/beta) / (1 + math.Exp(-g)) * u
			}
			for r := range want {
				var v float64
				for j, a := range act {
					v += float64(coeff(downAt+(e*hidden+r)*width+j)) * a
				}
				want[r] += float64(job.Weights[slot]) * float64(job.DownScale[e]) * v
			}
		}
		for r, v := range got {
			if !(math.Abs(float64(v)-want[r]) < 1e-5) {
				t.Fatalf("linear beta %v row %d got %g, expected %.8g", lb, r, v, want[r])
			}
		}
	}
	const routerAt = downAt + experts*width*hidden
	for i := range experts * hidden {
		binary.LittleEndian.PutUint32(data[(routerAt+i)*4:], math.Float32bits(coeff(routerAt+i)))
	}
	router, err := store.Matrix(routerAt*4, uint32(TensorF32), experts, hidden, 1)
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, experts)
	if err := store.Run([]metalgemm.PackedGGUFJob{{Matrix: router, X: x, Dst: logits}}); err != nil {
		t.Fatal(err)
	}
	sigmoid := func(e int) float32 { return float32(1 / (1 + math.Exp(-float64(logits[e])))) }
	norm := sigmoid(2) + sigmoid(0)
	job.Weights = []float32{1.75 * sigmoid(2) / norm, 1.75 * sigmoid(0) / norm}
	want, got := make([]float32, hidden), make([]float32, hidden)
	if err := store.ExpertMLP(job, x, want); err != nil {
		t.Fatal(err)
	}
	ffn := metalgemm.PackedRoutedFFN{Router: router, GateUp: gu, Down: down, Slots: 2,
		Bias: []float32{0, -10, 10}, GateUpScale: job.GateUpScale, DownScale: job.DownScale,
		WeightScale: 1.75, Beta: job.Beta, LinearBeta: job.LinearBeta}
	ran, err := store.RoutedFFN(ffn, x, got)
	if err != nil || !ran {
		t.Fatalf("routed FFN ran %v, err %v", ran, err)
	}
	for i := range want {
		if !(math.Abs(float64(want[i]-got[i])) < 1e-5) {
			t.Fatalf("routed FFN row %d got %g want %g", i, got[i], want[i])
		}
	}

}

// fak-test:runtime integration est=1s lane=default
func TestInkling2MetalHostVisibleFP32Projections(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal device required")
	}
	const rows, cols, experts = 5, 9, 3
	data, err := syscall.Mmap(-1, 0, os.Getpagesize(), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(data)
	const offset = 28
	weight := func(e, r, c int) float32 { return float32((e*7+r*3+c*2)%13-6) / 16 }
	for e := range experts {
		for r := range rows {
			for c := range cols {
				binary.LittleEndian.PutUint32(data[offset+4*((e*rows+r)*cols+c):], math.Float32bits(weight(e, r, c)))
			}
		}
	}
	store, err := metalgemm.NewPackedGGUFStore(data)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	matrix, err := store.Matrix(offset, uint32(TensorF32), rows, cols, experts)
	if err != nil {
		t.Fatal(err)
	}
	for _, count := range []int{1, 3, 19, 16 * experts} {
		jobs := make([]metalgemm.PackedGGUFJob, count)
		for i := range jobs {
			jobs[i] = metalgemm.PackedGGUFJob{Matrix: matrix, Expert: (2*i + 1) % experts, X: make([]float32, cols), Dst: make([]float32, rows)}
			for c := range cols {
				jobs[i].X[c] = float32((i+4*c)%9-4) / 8
			}
		}
		if err := store.Run(jobs); err != nil {
			t.Fatal(err)
		}
		for i, j := range jobs {
			for r, got := range j.Dst {
				var want float32
				for c, x := range j.X {
					want += weight(j.Expert, r, c) * x
				}
				if got != want {
					t.Fatalf("%d jobs, job %d row %d: got %g, want %g", count, i, r, got, want)
				}
			}
		}
	}
}
