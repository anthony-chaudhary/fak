package fleetcompare

import (
	"reflect"
	"testing"
)

func fixtureCols() map[string][]float64 {
	return map[string][]float64{
		"agents":            []float64{50, 50, 50, 20},
		"turns":             []float64{30, 10, 20, 30},
		"shared_saved_mean": []float64{90, 40, 65, 30},
		"cross_uplift_mean": []float64{25, 10, 18, 8},
	}
}

func TestSliceFixedIsolatedIsSharedMinusCross(t *testing.T) {
	got := SliceFixed(fixtureCols(), "agents", 50)
	if !reflect.DeepEqual(got.Xs, []float64{10, 20, 30}) {
		t.Fatalf("xs = %+v", got.Xs)
	}
	if !reflect.DeepEqual(got.Shared, []float64{40, 65, 90}) {
		t.Fatalf("shared = %+v", got.Shared)
	}
	if !reflect.DeepEqual(got.Cross, []float64{10, 18, 25}) {
		t.Fatalf("cross = %+v", got.Cross)
	}
	if !reflect.DeepEqual(got.Isolated, []float64{30, 47, 65}) {
		t.Fatalf("isolated = %+v", got.Isolated)
	}
}

func TestSliceFixedTurnsSweepsAgents(t *testing.T) {
	got := SliceFixed(fixtureCols(), "turns", 30)
	if !reflect.DeepEqual(got.Xs, []float64{20, 50}) {
		t.Fatalf("xs = %+v", got.Xs)
	}
	if !reflect.DeepEqual(got.Shared, []float64{30, 90}) {
		t.Fatalf("shared = %+v", got.Shared)
	}
	if !reflect.DeepEqual(got.Isolated, []float64{22, 65}) {
		t.Fatalf("isolated = %+v", got.Isolated)
	}
}

func BenchmarkSliceFixed(b *testing.B) {
	cols := fixtureCols()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SliceFixed(cols, "agents", 50)
	}
}

func BenchmarkSliceFixedLarge(b *testing.B) {
	n := 1000
	cols := map[string][]float64{
		"agents":            make([]float64, n),
		"turns":             make([]float64, n),
		"shared_saved_mean": make([]float64, n),
		"cross_uplift_mean": make([]float64, n),
	}
	for i := 0; i < n; i++ {
		cols["agents"][i] = float64(i % 10)
		cols["turns"][i] = float64(i)
		cols["shared_saved_mean"][i] = float64(i * 2)
		cols["cross_uplift_mean"][i] = float64(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SliceFixed(cols, "agents", 5)
	}
}
