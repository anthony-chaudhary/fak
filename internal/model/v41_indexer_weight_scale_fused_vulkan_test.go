//go:build vulkan && (windows || linux) && cgo

package model

import (
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestV41IndexWeightScaleFusedVulkan observes the actual bound model callback,
// without the preceding projection publication. On this pinned path, one
// dispatch distinguishes fused scale/publication from the two-dispatch scalar
// fallback. The optional module must be enabled before backend initialization;
// setting an environment variable after initialization cannot create it.
// Run serially with the reviewed shader bundle and an independently identified
// device. This narrow seam witness does not qualify full-model inference.
// fak-test:runtime integration est=5s lane=optin
func TestV41IndexWeightScaleFusedVulkan(t *testing.T) {
	if os.Getenv("FAK_V41_INDEX_SCALE_BF16") != "1" {
		t.Skip("explicit fused index scale opt-in required before initialization")
	}
	if os.Getenv(v41DeferredBF16PublishEnv) == "0" {
		t.Fatal("fused model witness requires deferred publication admission")
	}
	be := v41SeamVulkanBackend(t)
	fused, ok := be.(compute.V41IndexScaleBF16Backend)
	if !ok {
		t.Fatal("Vulkan backend lacks the fused index scale capability")
	}
	const dim, heads = 128, 64
	m := v41IndexerTestFixture(t)
	m.Cfg.IndexNHeads = heads
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		if closeErr := m.CloseWeights(); closeErr != nil {
			t.Errorf("constructor cleanup: %v", closeErr)
		}
		t.Fatal(err)
	}
	defer func() {
		s.Close()
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}()

	// All inputs are already finite widened BF16. Include signed zero and a
	// subnormal, while retaining nonzero normal values as an anti-vacuity check.
	words := []uint32{0x80000000, 0x00010000, 0x3fc00000, 0xc0400000}
	values := make([]float32, heads)
	for i := range values {
		values[i] = math.Float32frombits(words[i%len(words)])
	}
	if !v41BF16Widened(values) {
		t.Fatal("fixture must enter the scale seam after projection publication")
	}
	want, err := v41IndexWeightsBF16(0, values, dim, heads)
	if err != nil || len(want) != heads || want[2] == 0 {
		t.Fatalf("nonempty host oracle: result=%v err=%v", want, err)
	}
	scale := v41IndexWeightScale(dim, heads)
	var got []float32
	var dispatches uint64
	var hostRounds int64
	func() {
		scope := s.beginBackendExecution()
		complete := false
		defer scope.finishCompleted(&complete)
		if scope == nil || !scope.live || scope.owner != backendExecutionOwner(be) || s.backendExecutionScope != scope {
			t.Fatal("fused model witness lacks the actual live execution owner")
		}
		fn := s.v41IndexWeightScaleFunc()
		if fn == nil {
			t.Fatal("ordinary scalar/publication prerequisites did not bind the scale seam")
		}
		if _, admitted := v41DeferredBF16Publisher(be, heads); !admitted {
			t.Fatal("backend refused the deferred read required by fused selection")
		}
		// Probe the real backend with owned device storage outside the measured
		// window. The callback will independently admit its own uploaded input.
		func() {
			input := uploadHostF32Class(be, []int{heads}, values, compute.MemoryActivation, "fused index scale witness admission")
			defer be.Free(input)
			if !fused.SupportsV41IndexScaleBF16(input, scale) {
				t.Fatal("requested fused module or exact input geometry unavailable")
			}
		}()
		dispatches = v41SeamDispatches(t, be, func() {
			hostRounds = v41BF16PublishHostRounds(t, func() {
				got, err = fn(0, values, scale)
			})
		})
		if err != nil || hostRounds != 0 || dispatches != 1 {
			t.Fatalf("selected fused model seam: dispatches=%d host_rounds=%d err=%v", dispatches, hostRounds, err)
		}
		if v41IndexWeightScaleCalls.Load() != 1 || v41BF16ResidentCalls.Load() != 1 || v41BF16DeferredCalls.Load() != 1 || v41BF16PublishCalls.Load() != 0 {
			t.Fatalf("scale/resident/deferred/ordinary calls=%d/%d/%d/%d", v41IndexWeightScaleCalls.Load(), v41BF16ResidentCalls.Load(), v41BF16DeferredCalls.Load(), v41BF16PublishCalls.Load())
		}
		complete = true
	}()
	if len(got) != len(want) {
		t.Fatalf("result width=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("lane=%d device=%08x host=%08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
		if math.Float32bits(values[i]) != words[i%len(words)] {
			t.Fatalf("input lane %d changed", i)
		}
	}
	if s.BackendSessionClosed() || s.backendExecutionScope != nil {
		t.Fatal("successful fused seam closed the session or retained its scope")
	}
	t.Logf("V41_FUSED_INDEX_SCALE_MODEL dispatches=%d host_rounds=%d bit_exact=%d", dispatches, hostRounds, len(got))
}
