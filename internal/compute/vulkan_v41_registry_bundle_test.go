//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime integration est=1s lane=optin
// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1 and FAK_VULKAN_SPIRV naming
// a bundle built from the current trusted registry; a skip is no witness. It
// checks that a registry-built bundle binds every V4.1 device capability the
// model's device-only admission requires, including the indexer score whose
// production gate is its dispatch prerequisites (Ref #13668).
func TestVulkanV41RegistryBundleBindsDeviceOps(t *testing.T) {
	v := vk(t)
	dir := os.Getenv("FAK_VULKAN_SPIRV")
	if dir == "" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required registry-bundle witness: FAK_VULKAN_SPIRV is unset")
		}
		t.Skip("FAK_VULKAN_SPIRV unset; bundle census unknown")
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var census []string
	for _, entry := range entries {
		if name := entry.Name(); filepath.Ext(name) == ".spv" {
			census = append(census, strings.TrimSuffix(name, ".spv"))
		}
	}
	want := computebuild.CurrentVulkanShaderRegistryLatest()
	slices.Sort(census)
	slices.Sort(want)
	if !slices.Equal(census, want) {
		t.Fatalf("bundle census %d modules is not the current registry (%d)", len(census), len(want))
	}
	for _, c := range []struct {
		name string
		have bool
	}{
		{"V41RouterTopKAvailable", v.V41RouterTopKAvailable()},
		{"V41IndexSelectAvailable", v.V41IndexSelectAvailable()},
		{"SupportsV41MHC", v.SupportsV41MHC()},
		{"SupportsV41F32Finite(65)", v.SupportsV41F32Finite(65)},
		{"SupportsV41TailRoPE", v.SupportsV41TailRoPE()},
		{"V41TailRoPERowsAvailable", v.V41TailRoPERowsAvailable()},
		{"SupportsV41SharedAttention", v.SupportsV41SharedAttention()},
		{"V41EmbeddingGatherAvailable", v.V41EmbeddingGatherAvailable()},
		{"V41Q2KEmbeddingGatherAvailable", v.V41Q2KEmbeddingGatherAvailable()},
		{"SupportsV41ExactRMSNorm(512)", v.SupportsV41ExactRMSNorm(512, 1e-6)},
		{"SupportsV41ExactRMSNorm(128)", v.SupportsV41ExactRMSNorm(128, 1e-6)},
		{"V41RoutedSwiGLUAvailable", v.V41RoutedSwiGLUAvailable()},
		{"SupportsV41EngramGate", v.SupportsV41EngramGate()},
		{"SupportsV41CompressorPool", v.SupportsV41CompressorPool()},
		{"V41TailRoPEExactAvailable", v.V41TailRoPEExactAvailable()},
		{"V41ScaleExactAvailable", v.V41ScaleExactAvailable()},
		{"SupportsV41EngramRowDecode", v.SupportsV41EngramRowDecode()},
		{"V41RoutedIDAvailable", v.V41RoutedIDAvailable()},
		{"V41SharedAttentionFastAvailable", v.V41SharedAttentionFastAvailable()},
	} {
		if !c.have {
			t.Errorf("registry-built bundle leaves %s false", c.name)
		}
	}
	if !v.SupportsV41IndexerScore() {
		t.Errorf("registry-built bundle leaves SupportsV41IndexerScore false: %s", v.V41IndexerScoreUnavailableReason())
	}
}

// fak-test:runtime integration est=1s lane=optin
// Requires the exact current registry/native archive and a physical device.
// A skip is no arithmetic, transfer, lifetime or throughput qualification.
func TestVulkanV41F32Finite(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41F32Finite(193) {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required finite-status module/count capability missing")
		}
		t.Skip("optional finite-status module unavailable")
	}
	finiteBits := []uint32{0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0x3f800001}
	for _, n := range []int{1, 65, 193} {
		for _, batched := range []bool{false, true} {
			for _, site := range []uint32{V41F32FiniteRoutedSum, V41F32FiniteFinalNormRaw} {
				for _, poison := range []uint32{0, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc01234} {
					t.Run(fmt.Sprintf("n%d/batch%t/site%d/poison%x", n, batched, site, poison), func(t *testing.T) {
						host := make([]float32, n)
						for i := range host {
							host[i] = math.Float32frombits(finiteBits[i%len(finiteBits)])
						}
						bad := -1
						if poison != 0 {
							bad = n - 1
							host[bad] = math.Float32frombits(poison)
							if n > 65 {
								bad = 64
								host[bad] = math.Float32frombits(poison)
							}
						}
						input := v.Upload(NewF32(Default(), []int{n}, host), F32)
						defer v.Free(input)
						if batched {
							if !v.BeginV41StepBatch() {
								t.Fatal("finite step batch begin")
							}
							defer func() {
								if err := v.EndV41StepBatch(); err != nil {
									t.Errorf("finite step batch retirement: %v", err)
								}
							}()
						}
						before := v.VulkanDebugDispatchProfileSnapshot()
						status, err := v.V41F32FiniteDeferred(input, site)
						if err != nil {
							t.Fatal(err)
						}
						if status.Kind != V41DeferredF32Finite || status.Count != n || status.Site != site || V41DeferredRecords(status) != 1 {
							v.Free(status.Record)
							t.Fatal("finite status identity changed")
						}
						after := v.VulkanDebugDispatchProfileSnapshot()
						if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1" && after.ComputeDispatches-before.ComputeDispatches != 1 {
							v.Free(status.Record)
							t.Fatal("finite validation did not record exactly one device dispatch")
						}
						verdicts, rows, err := v.V41ReadDeferred([]V41DeferredStatus{status}, nil)
						if err != nil || len(verdicts) != 1 || len(rows) != 0 {
							t.Fatalf("finite status-only read: %v verdicts=%d rows=%d", err, len(verdicts), len(rows))
						}
						if bad < 0 {
							if verdicts[0] != nil {
								t.Fatalf("finite extreme or signed-zero rejected: %v", verdicts[0])
							}
						} else {
							var fault *V41F32FiniteArithmeticError
							if !errors.As(verdicts[0], &fault) || fault.Site != site || fault.Index != bad {
								t.Fatalf("lowest raw nonfinite index lost: %v", verdicts[0])
							}
						}
						// Test-only input read proves no mutation. Production primitive
						// reads only its 16-byte status and never downloads the input.
						got := v.Read(input)
						if len(got) != len(host) {
							t.Fatal("input extent changed")
						}
						for i := range host {
							if math.Float32bits(got[i]) != math.Float32bits(host[i]) {
								t.Fatalf("input mutation at %d", i)
							}
						}
					})
				}
			}
		}
	}
	// Raw F32 finiteness and BF16 publication deliberately differ at overflow.
	// The common reader preserves issued verdict order without aliasing records.
	input := v.Upload(NewF32(Default(), []int{1}, []float32{math.MaxFloat32}), F32)
	defer v.Free(input)
	raw, err := v.V41F32FiniteDeferred(input, V41F32FiniteRoutedSum)
	if err != nil {
		t.Fatal(err)
	}
	published, rounded, err := v.V41BF16PublishDeferred(input, V41BF16PublishRoutedDown)
	if err != nil {
		v.Free(raw.Record)
		t.Fatal(err)
	}
	defer v.Free(published)
	verdicts, rows, err := v.V41ReadDeferred([]V41DeferredStatus{raw, rounded}, nil)
	var rounding *V41BF16CheckedArithmeticError
	if err != nil || len(rows) != 0 || len(verdicts) != 2 || verdicts[0] != nil || !errors.As(verdicts[1], &rounding) {
		t.Fatalf("raw/rounded semantic distinction or order lost: %v %v", verdicts, err)
	}
}

// fak-test:runtime integration est=20ms lane=optin
func TestVulkanV41F32FiniteModuleABI(t *testing.T) {
	dir := os.Getenv("FAK_VULKAN_SPIRV")
	if dir == "" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required current-registry module directory missing")
		}
		t.Skip("current-registry module directory unset")
	}
	code, err := os.ReadFile(filepath.Join(dir, "v41_f32_finite.spv"))
	if err != nil {
		t.Fatal(err)
	}
	if !v41F32FiniteModuleABI(code) {
		t.Fatal("compiled finite module failed structural ABI")
	}
	if v41F32FiniteModuleABI(code[:len(code)-1]) || v41F32FiniteModuleABI(nil) {
		t.Fatal("truncated finite module admitted")
	}
	// Corruptions are parser inputs only; never build or submit these modules.
	for _, mutation := range []string{"magic", "local", "binding", "atomic"} {
		changed := append([]byte(nil), code...)
		found := mutation == "magic"
		if found {
			binary.LittleEndian.PutUint32(changed, 0)
		}
		for i := 20; !found && i < len(changed); {
			word := binary.LittleEndian.Uint32(changed[i:])
			n, op := int(word>>16), word&0xffff
			if n <= 0 || n*4 > len(changed)-i {
				t.Fatal("compiled module framing changed")
			}
			switch {
			case mutation == "local" && op == 16 && n == 6:
				binary.LittleEndian.PutUint32(changed[i+12:], 32)
				found = true
			case mutation == "binding" && op == 71 && n == 4 && binary.LittleEndian.Uint32(changed[i+8:]) == 33:
				binary.LittleEndian.PutUint32(changed[i+12:], 2)
				found = true
			case mutation == "atomic" && op == 237:
				binary.LittleEndian.PutUint32(changed[i:], uint32(n)<<16|236)
				found = true
			}
			i += n * 4
		}
		if !found || v41F32FiniteModuleABI(changed) {
			t.Fatalf("ABI corruption %q missed or was admitted", mutation)
		}
	}
}
