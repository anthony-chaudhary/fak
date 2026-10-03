//go:build darwin && arm64 && cgo

package compute

import (
	"math"
	"runtime"
	"testing"
	"time"
)

// A public HostBuffer with an observable lifetime. Tests hold the finalization
// signal, never the owner or its slice after the upload helper returns.
type metalLifetimeHost struct {
	values    []float32
	finalized chan struct{}
}

func (*metalLifetimeHost) Ready() bool      { return true }
func (h *metalLifetimeHost) F32() []float32 { return h.values }
func (*metalLifetimeHost) I8() []int8       { return nil }

func uploadEphemeralMetalWeight(be Backend, sequence int) (Tensor, <-chan struct{}) {
	h := &metalLifetimeHost{values: make([]float32, 16), finalized: make(chan struct{})}
	for i := range h.values {
		h.values[i] = float32(sequence*32+i) + 0.25
	}
	runtime.SetFinalizer(h, func(owner *metalLifetimeHost) { close(owner.finalized) })
	signal := h.finalized
	host := Tensor{Dtype: F32, Shape: []int{4, 4}, buf: h, be: Default()}
	resident := be.Upload(host, F32)
	runtime.KeepAlive(h)
	return resident, signal
}

// fak-test:runtime slow est=5s
// Cached immutable identities must keep their ephemeral input owners alive.
// Exact readback also checks that a fresh input is not mistaken for an older
// upload after GC; retaining every input in the test would mask that defect.
func TestMetalImmutableUploadRetainsEphemeralHostAcrossGC(t *testing.T) {
	be := metalOrSkip(t)
	for sequence := 1; sequence <= 16; sequence++ {
		resident, finalized := uploadEphemeralMetalWeight(be, sequence)
		runtime.GC()
		runtime.GC()
		runtime.Gosched()
		select {
		case <-finalized:
			be.Free(resident)
			t.Fatalf("upload %d: immutable input owner collected while cached device identity remains live", sequence)
		default:
		}
		got := be.Read(resident)
		if len(got) != 16 {
			be.Free(resident)
			t.Fatalf("upload %d: read length=%d", sequence, len(got))
		}
		for i, value := range got {
			want := float32(sequence*32+i) + 0.25
			if math.Float32bits(value) != math.Float32bits(want) {
				be.Free(resident)
				t.Fatalf("upload %d element %d: device=%g want=%g after ephemeral GC", sequence, i, value, want)
			}
		}
		be.Free(resident)
		runtime.KeepAlive(resident)
	}
}

// fak-test:runtime slow est=5s
// Free evicts the cache identity and releases host ownership even when a caller
// retains the now-freed descriptor. This bounds the lifetime of the GC pin.
func TestMetalImmutableUploadFreeReleasesHostOwner(t *testing.T) {
	be := metalOrSkip(t)
	resident, finalized := uploadEphemeralMetalWeight(be, 73)
	be.Free(resident)
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		runtime.Gosched()
		select {
		case <-finalized:
			runtime.KeepAlive(resident)
			return
		default:
		}
		time.Sleep(time.Millisecond)
	}
	runtime.KeepAlive(resident)
	t.Fatal("freed upload still pins its host owner after cache eviction")
}
