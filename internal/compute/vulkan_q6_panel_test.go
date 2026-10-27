//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime integration est=2s lane=optin
// UNRUN: independent native scalar-panel oracle and real ordinary consumer.
func TestVulkanQ6KPanelMatchesScalar(t *testing.T) {
	if os.Getenv("FAK_VULKAN_Q6K_PANEL") != "1" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("requires Q6 panel candidate and dispatch profiling at initialization")
	}
	v := vk(t)
	if !v.haveQ6KPanel {
		t.Fatal("requested optional Q6 panel is unavailable, not qualified")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
			t.Fatalf("required physical GPU not established: type %d", kind)
		}
	}
	for _, shape := range [][3]int{{1, 256, 16}, {17, 768, 17}, {31, 1024, 33}} {
		for _, batch := range []bool{false, true} {
			t.Run(fmt.Sprintf("out%d_in%d_P%d_batch%v", shape[0], shape[1], shape[2], batch), func(t *testing.T) {
				out, in, tokens := shape[0], shape[1], shape[2]
				raw := q6KStrixFixture(out, in)
				for block := 0; block < len(raw)/210; block++ {
					// Non-power-of-two half scales, both signs and alternating2B leads.
					d := [...]uint16{0x3555, 0xb955, 0x1403, 0x0001}[block%4]
					binary.LittleEndian.PutUint16(raw[block*210+208:], d)
				}
				w := v.Upload(NewQ6K(Default(), []int{out, in}, raw), Q6_K)
				defer v.Free(w)
				input := q6KStrixInput(tokens, in)
				for i := 0; i < in; i++ {
					input[i] = 0
				}
				x := v.Upload(NewF32(Default(), []int{tokens, in}, input), F32)
				defer v.Free(x)
				want := v.q6KPanelReferenceForTest(w, x, tokens)
				defer v.Free(want)
				var selected Tensor
				func() { vulkanMu.Lock(); defer vulkanMu.Unlock(); selected, _ = v.devTr([]int{tokens, out}, F32) }()
				defer v.Free(selected)
				if batch {
					v.BeginBatch()
				}
				before := v.VulkanDebugDispatchProfileSnapshot()
				used := func() bool {
					vulkanMu.Lock()
					defer vulkanMu.Unlock()
					v.validateQ6KMatMulInputs(w, x, tokens)
					return v.tryQ6KPanelLocked(w, x, selected, out, in, tokens)
				}()
				if batch {
					v.FlushBatch()
				}
				after := v.VulkanDebugDispatchProfileSnapshot()
				if !used || after.ComputeDispatches-before.ComputeDispatches != 1 {
					t.Fatal("candidate must select one native dispatch")
				}
				ordinary := v.BatchedMatMul(w, x, tokens)
				defer v.Free(ordinary)
				assertBits := func(name string, got Tensor) {
					t.Helper()
					a, b := v.Read(got), v.Read(want)
					if len(a) != len(b) {
						t.Fatalf("%s extent %d want%d", name, len(a), len(b))
					}
					for i := range a {
						if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
							t.Fatalf("%s[%d]=%08x want%08x", name, i, math.Float32bits(a[i]), math.Float32bits(b[i]))
						}
					}
				}
				assertBits("direct", selected)
				assertBits("ordinary BatchedMatMul", ordinary)
				// Native live extents remain authoritative even with forged shape metadata.
				short := v.Upload(NewF32(Default(), []int{in}, input[:in]), F32)
				defer v.Free(short)
				short.Shape = []int{tokens, in}
				before = v.VulkanDebugDispatchProfileSnapshot()
				var recovered any
				func() {
					vulkanMu.Lock()
					defer vulkanMu.Unlock()
					defer func() { recovered = recover() }()
					v.q6kMatMulLocked(w, short, selected, tokens)
				}()
				if recovered == nil || !strings.Contains(fmt.Sprint(recovered), "Q6_K panel failed closed") {
					t.Fatalf("selected short native input did not fail closed: %v", recovered)
				}
				after = v.VulkanDebugDispatchProfileSnapshot()
				if after.ComputeDispatches != before.ComputeDispatches {
					t.Fatal("invalid native extent recorded work")
				}
				assertBits("output unchanged after rejection", selected)
				// The private tile is not a decode/short-panel authority. Declines
				// record nothing; ordinary <16-token requests retain their old route.
				for _, p := range []int{1, 15} {
					refused := func() bool {
						vulkanMu.Lock()
						defer vulkanMu.Unlock()
						return v.tryQ6KPanelLocked(w, x, selected, out, in, p)
					}()
					if refused {
						t.Fatalf("private tile selected P=%d", p)
					}
				}
			})
		}
	}
}
