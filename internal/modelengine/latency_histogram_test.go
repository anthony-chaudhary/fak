package modelengine

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"
)

func TestLatencyHistogramGeoBuckets(t *testing.T) {
	bounds := LatencyHistogramDefaultBoundaries()
	// Geometric bounds 1.2^0..1.2^k below 6e7, plus the appended 60s cap.
	wantBounds := int(math.Ceil(math.Log(60e6)/math.Log(1.2))) + 1
	if len(bounds) != wantBounds {
		t.Fatalf("len(bounds) = %d, want %d", len(bounds), wantBounds)
	}
	if bounds[0] != 1 || bounds[len(bounds)-1] != 60e6 {
		t.Fatalf("bounds span [%v, %v], want [1, 6e7]", bounds[0], bounds[len(bounds)-1])
	}
	for i := 1; i < len(bounds)-1; i++ {
		if r := bounds[i] / bounds[i-1]; math.Abs(r-1.2) > 1e-9 {
			t.Fatalf("ratio at %d = %v, want 1.2", i, r)
		}
	}

	if got := NewLatencyHistogram("x", "", nil, []float64{math.NaN(), 5, math.Inf(1), 2, 5}).boundaries; fmt.Sprint(got) != "[2 5]" {
		t.Fatalf("normalized bounds = %v, want [2 5]", got)
	}
	if got := NewLatencyHistogram("x", "", nil, []float64{math.Inf(-1)}).boundaries; len(got) != len(bounds) {
		t.Fatalf("all-non-finite bounds should fall back to defaults, got %d", len(got))
	}

	h := NewLatencyHistogram("fak_test_latency_us", "test latency", map[string]string{"model": `a"b`}, nil)
	for _, tc := range []struct {
		us   uint64
		want int
	}{{0, 0}, {1, 0}, {2, 4}, {60e6, 99}, {60e6 + 1, 100}} {
		if got := h.bucketIndex(tc.us); got != tc.want {
			t.Fatalf("bucketIndex(%d) = %d, want %d", tc.us, got, tc.want)
		}
	}

	const workers, per = 8, 1000
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := 0; i < per; i++ {
				h.Observe(3 * time.Microsecond)
			}
		}()
	}
	wg.Wait()
	h.Observe(-time.Second)
	h.Observe(2 * time.Minute)

	snap := h.Collect()
	if snap.Count != workers*per+2 {
		t.Fatalf("count = %d, want %d", snap.Count, workers*per+2)
	}
	if want := uint64(3*workers*per + 120e6); snap.SumUS != want {
		t.Fatalf("sum = %d, want %d", snap.SumUS, want)
	}
	if snap.Cumulative[0] != 1 || snap.Cumulative[len(snap.Cumulative)-1] != workers*per+1 {
		t.Fatalf("cumulative edges = %d..%d", snap.Cumulative[0], snap.Cumulative[len(snap.Cumulative)-1])
	}

	var out strings.Builder
	h.AppendPrometheusText(&out)
	text := out.String()
	for _, want := range []string{
		"# HELP fak_test_latency_us test latency\n# TYPE fak_test_latency_us histogram\n",
		`fak_test_latency_us_bucket{model="a\"b",le="1"} 1` + "\n",
		`fak_test_latency_us_bucket{model="a\"b",le="1.2"} 1` + "\n",
		`fak_test_latency_us_bucket{model="a\"b",le="60000000"} 8001` + "\n",
		`fak_test_latency_us_bucket{model="a\"b",le="+Inf"} 8002` + "\n",
		`fak_test_latency_us_sum{model="a\"b"} 120024000` + "\n",
		`fak_test_latency_us_count{model="a\"b"} 8002` + "\n",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("export missing %q in:\n%s", want, text)
		}
	}
	if got := latencyHistogramFormatLe(bounds[2]); got != fmt.Sprintf("%.17g", bounds[2]) {
		t.Fatalf("le format = %q", got)
	}
	if got := latencyHistogramEscapeLabel("a\\b\n\"\xff"); got != `a\\b\n\"`+"\xff" {
		t.Fatalf("escape = %q", got)
	}
	if sz := unsafe.Sizeof(latencyHistogramShard{}); unsafe.Sizeof(uintptr(0)) == 8 && sz != 64 {
		t.Fatalf("shard size = %d, want 64", sz)
	}
	if n := strings.Count(text, "_bucket{"); n != 101 {
		t.Fatalf("bucket lines = %d, want 101", n)
	}
}
