// fak-test:runtime fast est=1s lane=default
package perfledger

import (
	"math"
	"testing"
)

func TestSketch(t *testing.T) {
	t.Run("bounded centroids and sub-percent accuracy across compression cycles", func(t *testing.T) {
		defaultTD, err := NewTDigest(0)
		if err != nil || defaultTD.compression != DefaultTDigestCompression {
			t.Fatalf("NewTDigest(0) = (%+v, %v), want default compression %v", defaultTD, err, DefaultTDigestCompression)
		}
		const compression = 100.0
		td, err := NewTDigest(compression)
		if err != nil {
			t.Fatalf("NewTDigest: %v", err)
		}
		const n = 5000
		// Deterministic permutation via LCG stride coprime to n (stride=7919).
		sum := 0.0
		sumSq := 0.0
		for k := 0; k < n; k++ {
			v := float64((k * 7919) % n)
			if err := td.Add(v); err != nil {
				t.Fatalf("Add(%v): %v", v, err)
			}
			sum += v
			sumSq += v * v
		}
		if td.Count() != n {
			t.Fatalf("Count = %d, want %d", td.Count(), n)
		}
		if td.Min() != 0 || td.Max() != float64(n-1) {
			t.Fatalf("Min/Max = [%v, %v], want [0, %v]", td.Min(), td.Max(), float64(n-1))
		}
		if math.Abs(td.Sum()-sum) > 1e-3 {
			t.Fatalf("Sum = %v, want %v", td.Sum(), sum)
		}
		wantMean := sum / float64(n)
		wantStd := math.Sqrt(sumSq/float64(n) - wantMean*wantMean)
		if math.Abs(td.Mean()-wantMean) > 1e-6 {
			t.Fatalf("Mean = %v, want %v", td.Mean(), wantMean)
		}
		if math.Abs(td.StdDev()-wantStd) > 1e-4 {
			t.Fatalf("StdDev = %v, want %v", td.StdDev(), wantStd)
		}

		centroids := td.CentroidCount()
		if centroids == 0 || centroids > int(compression)+16 {
			t.Fatalf("CentroidCount = %d, expected bounded within compression %v", centroids, compression)
		}

		band, err := td.PercentileBand()
		if err != nil {
			t.Fatalf("PercentileBand: %v", err)
		}
		checks := []struct {
			name string
			got  float64
			want float64
			tol  float64
		}{
			{"p1", band.P1, 49.99, 15},
			{"p50", band.P50, 2499.5, 40},
			{"p90", band.P90, 4499.1, 40},
			{"p95", band.P95, 4749.05, 30},
			{"p99", band.P99, 4949.01, 20},
		}
		for _, c := range checks {
			if math.Abs(c.got-c.want) > c.tol {
				t.Errorf("%s = %.2f, want %.2f (tol %.2f)", c.name, c.got, c.want, c.tol)
			}
		}
	})

	t.Run("merge combines Welford std, bounds, and quantiles accurately", func(t *testing.T) {
		a, _ := NewTDigest(100)
		b, _ := NewTDigest(100)
		full, _ := NewTDigest(100)

		for i := 1; i <= 1500; i++ {
			v := float64(i) * 0.5
			_ = a.Add(v)
			_ = full.Add(v)
		}
		for i := 1501; i <= 4500; i++ {
			v := float64(i) * 0.5
			_ = b.Add(v)
			_ = full.Add(v)
		}
		if err := a.Merge(b); err != nil {
			t.Fatalf("Merge: %v", err)
		}
		if a.Count() != full.Count() || a.Min() != full.Min() || a.Max() != full.Max() {
			t.Fatalf("merged scalars mismatch: got count=%d min=%v max=%v, want %d %v %v",
				a.Count(), a.Min(), a.Max(), full.Count(), full.Min(), full.Max())
		}
		if math.Abs(a.Sum()-full.Sum()) > 1e-6 || math.Abs(a.StdDev()-full.StdDev()) > 1e-6 {
			t.Fatalf("merged sum/std mismatch: got (%v, %v), want (%v, %v)", a.Sum(), a.StdDev(), full.Sum(), full.StdDev())
		}
		qMerged, _ := a.Quantile(0.95)
		qFull, _ := full.Quantile(0.95)
		if math.Abs(qMerged-qFull) > 25 {
			t.Fatalf("merged p95 = %v vs full %v", qMerged, qFull)
		}
	})

	t.Run("list metric aggregator respects replay capability flag and rejects non-finite", func(t *testing.T) {
		noReplay, err := NewListMetricAggregator(100, false)
		if err != nil {
			t.Fatal(err)
		}
		if err := noReplay.ObserveList([]float64{10, 20, 30}); err != nil {
			t.Fatal(err)
		}
		if err := noReplay.ObserveList([]float64{40, 50}); err != nil {
			t.Fatal(err)
		}
		if got := noReplay.ReplayedValues(); got != nil {
			t.Fatalf("expected nil ReplayedValues when flag disabled, got %v", got)
		}
		if err := noReplay.ObserveList([]float64{1, math.NaN()}); err == nil {
			t.Fatal("expected error on NaN list sample")
		}
		sum, err := noReplay.Summary()
		if err != nil {
			t.Fatal(err)
		}
		if sum.Count != 5 || sum.Sum != 150 || sum.Min != 10 || sum.Max != 50 {
			t.Fatalf("summary = %+v", sum)
		}

		withReplay, err := NewListMetricAggregator(100, true)
		if err != nil {
			t.Fatal(err)
		}
		_ = withReplay.ObserveList([]float64{1, 2, 3})
		if got := withReplay.ReplayedValues(); len(got) != 3 {
			t.Fatalf("ReplayedValues len = %d, want 3", len(got))
		}
	})
}
