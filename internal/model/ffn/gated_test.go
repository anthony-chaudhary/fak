package ffn

import (
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestGatedValidatesBeforeMutation(t *testing.T) {
	tests := []struct {
		name     string
		gate     []float32
		up       []float32
		activate Activation
		down     DownProject
		wantText string
	}{
		{name: "empty", gate: []float32{}, up: []float32{}, activate: func(x float32) float32 { return x }, down: func(x []float32) ([]float32, error) { return x, nil }, wantText: "nonempty"},
		{name: "length mismatch", gate: []float32{1, 2}, up: []float32{3}, activate: func(x float32) float32 { return x }, down: func(x []float32) ([]float32, error) { return x, nil }, wantText: "length"},
		{name: "nil activation", gate: []float32{1}, up: []float32{2}, down: func(x []float32) ([]float32, error) { return x, nil }, wantText: "activation"},
		{name: "nil down projection", gate: []float32{1}, up: []float32{2}, activate: func(x float32) float32 { return x }, wantText: "down"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gateBefore := append([]float32(nil), tt.gate...)
			upBefore := append([]float32(nil), tt.up...)
			activationCalls, downCalls := 0, 0
			activate := tt.activate
			if activate != nil {
				activate = func(x float32) float32 {
					activationCalls++
					return tt.activate(x)
				}
			}
			down := tt.down
			if down != nil {
				down = func(x []float32) ([]float32, error) {
					downCalls++
					return tt.down(x)
				}
			}
			_, err := Gated(tt.gate, tt.up, activate, down)
			if err == nil || !strings.Contains(strings.ToLower(err.Error()), tt.wantText) {
				t.Fatalf("Gated error = %v, want meaningful text containing %q", err, tt.wantText)
			}
			if activationCalls != 0 || downCalls != 0 {
				t.Fatalf("callbacks ran before validation: activation=%d down=%d", activationCalls, downCalls)
			}
			if !slices.Equal(tt.gate, gateBefore) || !slices.Equal(tt.up, upBefore) {
				t.Fatalf("validation mutated inputs: gate=%v up=%v", tt.gate, tt.up)
			}
		})
	}
}

func TestGatedAppliesInOrderAndProjectsOnce(t *testing.T) {
	gate := []float32{3, -2, 0.5}
	up := []float32{4, 5, -6}
	upBefore := append([]float32(nil), up...)
	var seen []float32
	downCalls := 0
	got, err := Gated(gate, up, func(x float32) float32 {
		seen = append(seen, x)
		return x + 1
	}, func(x []float32) ([]float32, error) {
		downCalls++
		if !reflect.DeepEqual(x, []float32{16, -5, -9}) {
			t.Fatalf("down input = %v", x)
		}
		return []float32{x[0] + x[1], x[2]}, nil
	})
	if err != nil {
		t.Fatalf("Gated: %v", err)
	}
	if !reflect.DeepEqual(seen, []float32{3, -2, 0.5}) {
		t.Fatalf("activation order = %v", seen)
	}
	if downCalls != 1 {
		t.Fatalf("down calls = %d, want 1", downCalls)
	}
	if !reflect.DeepEqual(gate, []float32{16, -5, -9}) {
		t.Fatalf("caller-owned gate = %v", gate)
	}
	if !reflect.DeepEqual(up, upBefore) {
		t.Fatalf("up mutated: %v", up)
	}
	if !reflect.DeepEqual(got, []float32{11, -9}) {
		t.Fatalf("result = %v", got)
	}
}

func TestGatedReturnsProjectionErrorUnchanged(t *testing.T) {
	wantErr := errors.New("projection failed")
	_, gotErr := Gated([]float32{2}, []float32{3}, func(x float32) float32 { return x }, func([]float32) ([]float32, error) {
		return nil, wantErr
	})
	if gotErr != wantErr {
		t.Fatalf("projection error = %v, want identical sentinel %v", gotErr, wantErr)
	}
}

func TestGatedSuccessAllocations(t *testing.T) {
	seed := []float32{1, 2, 3, 4}
	gate := make([]float32, len(seed))
	up := []float32{5, 6, 7, 8}
	activate := Activation(func(x float32) float32 { return x * x })
	down := DownProject(func(x []float32) ([]float32, error) { return x, nil })
	allocs := testing.AllocsPerRun(1000, func() {
		copy(gate, seed)
		if _, err := Gated(gate, up, activate, down); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("successful Gated allocations = %v, want 0", allocs)
	}
}

var gatedBenchmarkSink []float32

func BenchmarkGatedMatchedInline(b *testing.B) {
	const width = 4096
	seed := make([]float32, width)
	up := make([]float32, width)
	gate := make([]float32, width)
	output := make([]float32, 64)
	for i := range seed {
		seed[i] = float32(i%17-8) / 16
		up[i] = float32(i%13-6) / 12
	}
	activate := Activation(func(x float32) float32 { return x / (1 + x*x) })
	down := DownProject(func(x []float32) ([]float32, error) {
		for i := range output {
			output[i] = x[i] + x[len(x)-1-i]
		}
		return output, nil
	})
	for _, bench := range []struct {
		name string
		run  func() []float32
	}{
		{name: "Gated", run: func() []float32 {
			copy(gate, seed)
			out, err := Gated(gate, up, activate, down)
			if err != nil {
				b.Fatal(err)
			}
			return out
		}},
		{name: "Inline", run: func() []float32 {
			copy(gate, seed)
			for i := range gate {
				gate[i] = activate(gate[i]) * up[i]
			}
			out, err := down(gate)
			if err != nil {
				b.Fatal(err)
			}
			return out
		}},
	} {
		b.Run(bench.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(width * 4)
			for i := 0; i < b.N; i++ {
				gatedBenchmarkSink = bench.run()
			}
		})
	}
}
