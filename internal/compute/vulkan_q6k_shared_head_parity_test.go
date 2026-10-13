//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"os"
	"testing"
	"time"
)

// This same-backend witness compares the actual P=1 decode layout with each
// P=1..4 panel, including a real rows=2 dispatch and distinct BF16 input rows.
// At the full Flash shape this exercises the scalar-composition capability;
// reduced shapes still compare the separate generic panel implementation.
// The normal case uses a reduced vocabulary. Explicit FULL=1 selects synthetic
// packed weights at the pinned Flash [129280,5120] shape: 542,976,000 raw bytes
// plus a matching device allocation, never an expanded F32 weight matrix.
// Full-shape execution is estimated at 30s and bounded by a five-minute child.
// Neither case substitutes for a checkpoint-byte or end-to-end model witness.
// fak-test:runtime integration est=3s lane=optin
func TestVulkanQ6KSharedHeadPanelBitIdenticalToScalar(t *testing.T) {
	full := os.Getenv("FAK_VULKAN_Q6K_SHARED_HEAD_FULL") == "1"
	timeout := time.Minute
	if full {
		timeout = 5 * time.Minute
	}
	if runVulkanStartupEnvFixture(t, timeout, nil) {
		return
	}
	v := vk(t)
	if !v.SupportsQ6KMatMul() {
		t.Fatal("registered Vulkan backend lacks the packed Q6_K pipeline")
	}
	const in, maxRows = 5120, 4
	out := 129
	if full {
		out = 129280
	}
	composed := v.V41SharedHeadAvailable(Q6_K, out, in, maxRows)
	if full && !composed {
		t.Fatal("full-shape witness requires the admitted AMD UMA scalar-head route")
	}
	raw := q6KSharedHeadFixture(out, in)
	w := v.Upload(NewQ6K(Default(), []int{out, in}, raw), Q6_K)
	defer v.Free(w)
	x := q6KSharedHeadInput(maxRows, in)
	want := q6KSharedHeadScalar(t, v, w, x, out, in, maxRows)
	for rows := 1; rows <= maxRows; rows++ {
		func() {
			input := v.UploadClass(NewF32(Default(), []int{rows, in}, x[:rows*in]), F32, MemoryActivation, "Q6_K shared-head parity panel")
			defer v.Free(input)
			v.VulkanDebugResetDispatchProfile()
			output := v.BatchedMatMul(w, input, rows)
			defer v.Free(output)
			got := v.Read(output)
			q6KSharedHeadCompare(t, got, want[:rows*out], rows, out)
			profile := v.VulkanDebugDispatchProfileSnapshot()
			dispatches, copies := uint64(1), uint64(0)
			if composed {
				dispatches = uint64(rows)
				if rows > 1 {
					copies = uint64(2 * rows)
				}
			}
			if profile.ComputeDispatches != dispatches || profile.OtherComputeDispatches != dispatches || profile.OtherMatmulDispatches != dispatches || profile.D2DCopies != copies {
				t.Fatalf("rows=%d composed=%v require %d Q6_K dispatches and %d D2D copies, got %+v", rows, composed, dispatches, copies, profile)
			}
			if composed && (profile.BatchSubmits != 1 || profile.OneShotComputeSubmits != 0 || profile.OneShotD2DSubmits != 0) {
				t.Fatalf("rows=%d scalar composition must use one checked batch, got %+v", rows, profile)
			}
			t.Logf("Q6_K same-backend parity: synthetic [%d,%d], rows=%d, distinct BF16 inputs, exact F32 logits, native_dispatches=%d scalar_composition=%v", out, in, rows, dispatches, composed)
		}()
	}
}
