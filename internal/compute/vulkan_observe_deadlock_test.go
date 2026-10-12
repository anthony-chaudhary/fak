//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:runtime fast est=3s — two small device GEMVs on the registered Vulkan device.

const vulkanObservedMatMulDeadline = 20 * time.Second

// TestVulkanMatMulWithKernelObserverDoesNotSelfDeadlock replays the `fak serve`
// warmup shape on a real device: stepobs.Attach installs a computetrace observer,
// so Emitting() is true and MatMul probes the performance-query surface, which
// takes vulkanMu itself. The op runs in a goroutine under a deadline so a
// regression fails instead of hanging the test binary. The observer also asserts
// it runs with vulkanMu released, so a /metrics reader can never wait inside it.
func TestVulkanMatMulWithKernelObserverDoesNotSelfDeadlock(t *testing.T) {
	v := vk(t)
	var events, heldDuringObserver atomic.Int64
	computetrace.SetObserver(func(e computetrace.Event) {
		if e.Backend != v.Name() || e.Operation != "matmul" {
			return
		}
		events.Add(1)
		if !vulkanMu.TryLock() {
			heldDuringObserver.Add(1)
			return
		}
		vulkanMu.Unlock()
	})
	t.Cleanup(func() { computetrace.SetObserver(nil) })
	if !computetrace.Emitting() {
		t.Fatal("computetrace.Emitting() = false with an observer attached; the probe path is not exercised")
	}

	c := cpu()
	var s lcg = 17
	out, in := 32, 64
	w, x := randVec(&s, out*in), randVec(&s, in)
	weights := map[string]Tensor{"F32": v.Upload(NewF32(c, []int{out, in}, w), F32)}
	if v.haveQ8 {
		weights["Q8_0"] = v.Upload(QuantizeQ8(c, []int{out, in}, w, 32), Q8_0)
	}
	dx := v.Upload(NewF32(c, []int{in}, x), F32)

	for name, dw := range weights {
		before := events.Load()
		done := make(chan []float32, 1)
		go func() { done <- v.Read(v.MatMul(dw, dx)) }()
		select {
		case got := <-done:
			if len(got) != out {
				t.Fatalf("%s MatMul returned %d rows; want %d", name, len(got), out)
			}
		case <-time.After(vulkanObservedMatMulDeadline):
			// The wedged goroutine keeps vulkanMu held, so every later device test
			// would block on it; abort the binary instead of hanging it.
			panic(fmt.Sprintf("%s MatMul with a kernel observer attached did not return within %s: vulkanMu self-deadlock", name, vulkanObservedMatMulDeadline))
		}
		if delta := events.Load() - before; delta != 1 {
			t.Fatalf("%s MatMul emitted %d kernel events; want exactly 1", name, delta)
		}
	}
	if n := heldDuringObserver.Load(); n != 0 {
		t.Fatalf("kernel observer ran %d time(s) while vulkanMu was held", n)
	}
}
