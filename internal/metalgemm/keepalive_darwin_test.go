//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

func keepAliveTestWeight(tb testing.TB) (*Q4KWeight, []float32) {
	tb.Helper()
	const in, out = 256, 32
	rng := rand.New(rand.NewSource(731))
	w := UploadQ4K(makeTestQ4KMatrix(rng, out, in), out, in)
	if w == nil {
		tb.Fatal("UploadQ4K failed")
	}
	tb.Cleanup(w.Release)
	x := makeTestActivationsRange(rng, in, -1, 1)
	return w, x
}

func keepAliveAssertParity(tb testing.TB, got, want []float32) {
	tb.Helper()
	for i := range want {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			tb.Fatalf("keepalive changed GEMV output[%d]: got %g want %g", i, got[i], want[i])
		}
	}
}

// fak-test:runtime medium est=2s lane=optin
func TestKeepAliveMetalOutputAndDrain(t *testing.T) {
	if testing.Short() || os.Getenv("FAK_METAL_KEEPALIVE_TEST") != "1" {
		t.Skip("physical Metal keepalive smoke requires FAK_METAL_KEEPALIVE_TEST=1 without -short")
	}
	lease, err := gpulease.Acquire(gpulease.Options{NoWait: true})
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if !Available() {
		t.Skip("Metal unavailable")
	}
	w, x := keepAliveTestWeight(t)
	want, got := make([]float32, w.Out), make([]float32, w.Out)
	t.Setenv("FAK_METAL_KEEPALIVE", "off")
	before := KeepAliveState()
	stopOff := BeginKeepAlive()
	w.GEMV(x, want)
	stopOff()
	if state := KeepAliveState(); state.SubmittedBuffers != before.SubmittedBuffers || state.Holders != 0 || state.InFlight != 0 {
		t.Fatalf("off policy submitted GPU work: before=%+v after=%+v", before, state)
	}
	t.Setenv("FAK_METAL_KEEPALIVE", "on")
	stop := BeginKeepAlive()
	defer stop()
	// BeginKeepAlive submits the initial native buffers before returning.
	if state := KeepAliveState(); state.Holders != 1 || state.InFlight > 2 || state.Failed || state.SubmittedBuffers <= before.SubmittedBuffers {
		t.Fatalf("bounded active keepalive not witnessed at start: %+v", state)
	}
	for range 4 {
		w.GEMV(x, got)
		keepAliveAssertParity(t, got, want)
		state := KeepAliveState()
		if state.Holders != 1 || state.InFlight > 2 || state.Failed || state.SubmittedBuffers <= before.SubmittedBuffers {
			t.Fatalf("bounded active keepalive not witnessed: %+v", state)
		}
	}
	stop()
	if !pollGraphUntil(time.Second, func() bool {
		state := KeepAliveState()
		return state.Holders == 0 && state.InFlight == 0
	}) {
		t.Fatalf("finite keepalive buffers did not drain: %+v", KeepAliveState())
	}
	w.GEMV(x, got)
	keepAliveAssertParity(t, got, want)
}

// BenchmarkKeepAliveIdleGap measures host GEMV latency after an actual recorded host
// sleep; gpu-us/op is only emitted when native command-buffer timestamps exist.
// The gap is excluded from ns/op. macOS may round 200us sleeps up: gap-us/op records
// that observed delay rather than claiming the requested duration was achieved.
// fak-test:runtime integration est=5s lane=optin
func BenchmarkKeepAliveIdleGap(b *testing.B) {
	lease, err := gpulease.Acquire(gpulease.Options{NoWait: true})
	if err != nil {
		b.Fatal(err)
	}
	defer lease.Release()
	if !Available() {
		b.Fatal("physical Metal device required")
	}
	w, x := keepAliveTestWeight(b)
	want := make([]float32, w.Out)
	w.GEMV(x, want)
	for _, gap := range []time.Duration{200 * time.Microsecond, 5 * time.Millisecond} {
		for _, policy := range []string{"off", "on"} {
			b.Run(fmt.Sprintf("gap=%s/%s", gap, policy), func(b *testing.B) {
				b.Setenv("FAK_METAL_KEEPALIVE", policy)
				stop := BeginKeepAlive()
				defer stop()
				if policy == "on" && KeepAliveState().Holders == 0 {
					b.Fatal("keepalive failed to start")
				}
				got := make([]float32, w.Out)
				var gpuMS, waitMS float64
				var gpuCount int
				var actualGap time.Duration
				b.ResetTimer()
				for range b.N {
					b.StopTimer()
					gapStart := time.Now()
					time.Sleep(gap)
					actualGap += time.Since(gapStart)
					observation := NewExecutionObservation(ExecutionQ4KGEMV)
					b.StartTimer()
					if err := w.GEMVWithEventsErr(x, got, observation); err != nil {
						b.Fatal(err)
					}
					b.StopTimer()
					keepAliveAssertParity(b, got, want)
					snapshot, err := observation.Snapshot()
					if err != nil {
						b.Fatal(err)
					}
					for _, event := range snapshot.Events {
						waitMS += event.WaitMilliseconds
						if event.TimingAvailable {
							gpuMS += event.GPUMilliseconds
							gpuCount++
						}
					}
				}
				b.ReportMetric(float64(actualGap.Microseconds())/float64(b.N), "gap-us/op")
				b.ReportMetric(waitMS*1000/float64(b.N), "wait-us/op")
				if gpuCount > 0 {
					b.ReportMetric(gpuMS*1000/float64(gpuCount), "gpu-us/op")
				}
			})
		}
	}
}
