package stripeload

import (
	"bytes"
	"errors"
	"io"
	"math"
	"math/rand"
	"sync/atomic"
	"testing"
	"time"
)

// countingReaderAt tracks ReadAt calls across mirrors.
type countingReaderAt struct {
	inner io.ReaderAt
	calls atomic.Int64
}

func (c *countingReaderAt) ReadAt(p []byte, off int64) (int, error) {
	c.calls.Add(1)
	return c.inner.ReadAt(p, off)
}

// blob returns deterministic pseudo-random bytes.
func blob(t *testing.T, rng *rand.Rand, size int) []byte {
	t.Helper()
	b := make([]byte, size)
	if _, err := rng.Read(b); err != nil {
		t.Fatalf("rng.Read: %v", err)
	}
	return b
}

// TestBitIdentity verifies striped reads match single reader bytes, counts, and EOF behavior.
func TestBitIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	data := blob(t, rng, 1<<20) // 1 MiB
	single := bytes.NewReader(data)
	s, err := New([]Source{
		{R: bytes.NewReader(data), BWWeight: 1},
		{R: bytes.NewReader(data), BWWeight: 2.5},
		{R: bytes.NewReader(data), BWWeight: 0.75},
	}, WithMinChunk(512))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for i := 0; i < 300; i++ {
		off := rng.Int63n(int64(len(data)) + 4096)
		n := rng.Intn(96 * 1024)
		want := make([]byte, n)
		got := make([]byte, n)
		wantN, wantErr := single.ReadAt(want, off)
		gotN, gotErr := s.ReadAt(got, off)
		if gotN != wantN {
			t.Fatalf("case %d (off=%d n=%d): got n=%d want n=%d", i, off, n, gotN, wantN)
		}
		if (wantErr == nil) != (gotErr == nil) || (wantErr != nil && !errors.Is(gotErr, wantErr)) {
			t.Fatalf("case %d (off=%d n=%d): got err=%v want err=%v", i, off, n, gotErr, wantErr)
		}
		if !bytes.Equal(got[:gotN], want[:wantN]) {
			t.Fatalf("case %d (off=%d n=%d): striped bytes differ from single-reader bytes", i, off, n)
		}
	}
}

// TestSingleSourcePassthrough verifies single source reads degrade to direct byte reads.
func TestSingleSourcePassthrough(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	data := blob(t, rng, 256*1024)
	s, err := New([]Source{{R: bytes.NewReader(data), BWWeight: 1}}, WithMinChunk(1))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := make([]byte, 200*1024)
	n, err := s.ReadAt(got, 1000)
	if err != nil || n != len(got) {
		t.Fatalf("ReadAt: n=%d err=%v", n, err)
	}
	if !bytes.Equal(got, data[1000:1000+len(got)]) {
		t.Fatalf("passthrough bytes differ")
	}
}

// TestSmallReadSingleSource verifies reads below MinChunk touch only the fastest mirror.
func TestSmallReadSingleSource(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	data := blob(t, rng, 64*1024)
	mirrors := []*countingReaderAt{
		{inner: bytes.NewReader(data)},
		{inner: bytes.NewReader(data)},
		{inner: bytes.NewReader(data)},
	}
	s, err := New([]Source{
		{R: mirrors[0], BWWeight: 1},
		{R: mirrors[1], BWWeight: 3}, // fastest
		{R: mirrors[2], BWWeight: 2},
	}, WithMinChunk(4096))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := make([]byte, 1024) // < MinChunk
	if n, err := s.ReadAt(got, 8); err != nil || n != len(got) {
		t.Fatalf("ReadAt: n=%d err=%v", n, err)
	}
	touched := 0
	var total int64
	for _, m := range mirrors {
		c := m.calls.Load()
		total += c
		if c > 0 {
			touched++
		}
	}
	if touched != 1 || total != 1 {
		t.Fatalf("small read touched %d mirrors with %d total calls; want exactly 1 call to 1 mirror", touched, total)
	}
	if mirrors[1].calls.Load() != 1 {
		t.Fatalf("small read did not go to the fastest source")
	}
}

// TestBandwidthWeighting verifies sub-range planning matches relative bandwidth weights.
func TestBandwidthWeighting(t *testing.T) {
	const n = int64(3 << 20)
	ranges := carveRanges(0, n, []float64{1, 2}, 1<<10)
	if len(ranges) != 2 {
		t.Fatalf("got %d ranges, want 2", len(ranges))
	}
	slow, fast := ranges[0].n, ranges[1].n
	if d := slow - n/3; d < -2 || d > 2 {
		t.Fatalf("1x source got %d bytes, want ~%d", slow, n/3)
	}
	if d := fast - 2*n/3; d < -2 || d > 2 {
		t.Fatalf("2x source got %d bytes, want ~%d", fast, 2*n/3)
	}
	ratio := float64(fast) / float64(slow)
	if ratio < 1.99 || ratio > 2.01 {
		t.Fatalf("fast/slow byte ratio = %v, want ~2", ratio)
	}
}

// TestCarveRanges verifies gap-free coverage, degradation, and rounding allocation.
func TestCarveRanges(t *testing.T) {
	cases := []struct {
		name       string
		off, n     int64
		weights    []float64
		minChunk   int64
		wantRanges int   // 0 = don't check the count
		wantSrc    []int // nil = don't check assignment
	}{
		{name: "even split", off: 0, n: 4 << 20, weights: []float64{1, 1}, minChunk: 1 << 10, wantRanges: 2},
		{name: "uneven with remainder", off: 7, n: 1_000_003, weights: []float64{1, 2, 4}, minChunk: 16, wantRanges: 3},
		{name: "below minChunk goes to fastest", off: 100, n: 512, weights: []float64{1, 5, 2}, minChunk: 1 << 20, wantRanges: 1, wantSrc: []int{1}},
		{name: "single source", off: 0, n: 10 << 20, weights: []float64{3}, minChunk: 1, wantRanges: 1, wantSrc: []int{0}},
		{name: "tiny weight rounds to zero and is skipped", off: 0, n: 1 << 20, weights: []float64{1e-9, 1, 1}, minChunk: 16, wantRanges: 2},
		{name: "prime length prime weights", off: 13, n: 104_729, weights: []float64{0.3, 0.7, 1.1, 0.9}, minChunk: 8},
		{name: "n equals minChunk", off: 0, n: 4096, weights: []float64{2, 1}, minChunk: 4096, wantRanges: 1, wantSrc: []int{0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ranges := carveRanges(tc.off, tc.n, tc.weights, tc.minChunk)
			if tc.wantRanges != 0 && len(ranges) != tc.wantRanges {
				t.Fatalf("got %d ranges, want %d: %+v", len(ranges), tc.wantRanges, ranges)
			}
			if len(ranges) > len(tc.weights) {
				t.Fatalf("emitted %d ranges for %d sources", len(ranges), len(tc.weights))
			}
			cur := tc.off
			var total int64
			for i, r := range ranges {
				if r.n <= 0 {
					t.Fatalf("range %d has non-positive length: %+v", i, r)
				}
				if r.off != cur {
					t.Fatalf("range %d starts at %d, want contiguous %d", i, r.off, cur)
				}
				if r.src < 0 || r.src >= len(tc.weights) {
					t.Fatalf("range %d has out-of-range source %d", i, r.src)
				}
				cur += r.n
				total += r.n
			}
			if total != tc.n {
				t.Fatalf("coverage %d bytes, want exactly %d", total, tc.n)
			}
			for i, want := range tc.wantSrc {
				if ranges[i].src != want {
					t.Fatalf("range %d assigned to source %d, want %d", i, ranges[i].src, want)
				}
			}
		})
	}
	if got := carveRanges(0, 0, []float64{1, 2}, 16); got != nil {
		t.Fatalf("zero-length plan should be nil, got %+v", got)
	}
}

// errReaderAt always fails reads.
type errReaderAt struct{ err error }

func (e *errReaderAt) ReadAt(p []byte, off int64) (int, error) { return 0, e.err }

// TestErrorPropagation verifies failing mirror errors are returned to the caller.
func TestErrorPropagation(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	data := blob(t, rng, 1<<20)
	boom := errors.New("disk on fire")
	s, err := New([]Source{
		{R: bytes.NewReader(data), BWWeight: 1},
		{R: &errReaderAt{err: boom}, BWWeight: 1},
		{R: bytes.NewReader(data), BWWeight: 1},
	}, WithMinChunk(64))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := make([]byte, 512*1024)
	if _, err := s.ReadAt(got, 0); !errors.Is(err, boom) {
		t.Fatalf("ReadAt err = %v, want wrapped %v", err, boom)
	}
}

// gateReaderAt tracks peak in-flight reads across mirrors.
type gateReaderAt struct {
	inner    io.ReaderAt
	inflight *atomic.Int64
	peak     *atomic.Int64
	reached2 chan struct{}
	closer   *atomic.Bool
}

func (g *gateReaderAt) ReadAt(p []byte, off int64) (int, error) {
	cur := g.inflight.Add(1)
	defer g.inflight.Add(-1)
	for {
		old := g.peak.Load()
		if cur <= old || g.peak.CompareAndSwap(old, cur) {
			break
		}
	}
	if cur >= 2 && g.closer.CompareAndSwap(false, true) {
		close(g.reached2)
	}
	select {
	case <-g.reached2:
	case <-time.After(5 * time.Second):
		return 0, errors.New("never saw 2 concurrent reads: fan-out is serialized")
	}
	return g.inner.ReadAt(p, off)
}

// TestBoundedConcurrency verifies peak in-flight reads never exceed the concurrency cap.
func TestBoundedConcurrency(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	data := blob(t, rng, 1<<20)
	var inflight, peak atomic.Int64
	var closer atomic.Bool
	reached2 := make(chan struct{})
	mk := func() Source {
		return Source{
			R:        &gateReaderAt{inner: bytes.NewReader(data), inflight: &inflight, peak: &peak, reached2: reached2, closer: &closer},
			BWWeight: 1,
		}
	}
	s, err := New([]Source{mk(), mk(), mk(), mk()}, WithMinChunk(64), WithMaxConcurrency(2))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := make([]byte, 1<<20)
	n, err := s.ReadAt(got, 0)
	if err != nil || n != len(got) {
		t.Fatalf("ReadAt: n=%d err=%v", n, err)
	}
	if !bytes.Equal(got, data) {
		t.Fatalf("gated read returned wrong bytes")
	}
	if p := peak.Load(); p > 2 {
		t.Fatalf("peak in-flight reads = %d, want <= 2", p)
	}
	if p := peak.Load(); p != 2 {
		t.Fatalf("peak in-flight reads = %d, want exactly 2 (cap should be saturated)", p)
	}
}

// TestNewValidation verifies construction rejects empty or invalid inputs.
func TestNewValidation(t *testing.T) {
	good := Source{R: bytes.NewReader([]byte("x")), BWWeight: 1}
	if _, err := New(nil); err == nil {
		t.Fatalf("New(nil sources) should error")
	}
	if _, err := New([]Source{{R: nil, BWWeight: 1}}); err == nil {
		t.Fatalf("New(nil reader) should error")
	}
	for _, w := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := New([]Source{{R: good.R, BWWeight: w}}); err == nil {
			t.Fatalf("New(weight=%v) should error", w)
		}
	}
	if _, err := New([]Source{good}, WithMinChunk(0)); err == nil {
		t.Fatalf("WithMinChunk(0) should error")
	}
	if _, err := New([]Source{good}, WithMaxConcurrency(0)); err == nil {
		t.Fatalf("WithMaxConcurrency(0) should error")
	}
}
