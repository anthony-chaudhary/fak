//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime integration est=2s lane=optin
// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1 and FAK_VULKAN_SPIRV naming
// the current V13 bundle; a skip is no witness. The ordinary archive must admit
// the strict scalar twin on a device reporting all three binary32 controls, the
// exact scalar proof must reject structural corruptions of the compiled bytes,
// and every product is bit-exact against one RNE rounding of the exact product,
// including subnormal, signed-zero, tie and overflow operands (Ref #13668).
func TestVulkanV41ScaleExact(t *testing.T) {
	v := vk(t)
	if !v.V41ScaleExactAvailable() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required device did not admit the strict scalar twin (float controls or module proof missing)")
		}
		t.Skip("strict scalar twin not admitted on this device")
	}
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatalf("required physical GPU not established: deviceType=%d tier=%s", kind, v.Tier())
		}
		t.Skipf("physical GPU not established: deviceType=%d", kind)
	}
	if dir := os.Getenv("FAK_VULKAN_SPIRV"); dir != "" {
		code, err := os.ReadFile(filepath.Join(dir, "v41_scale_exact.spv"))
		if err != nil {
			t.Fatal(err)
		}
		if !v41ScaleExactModuleABI(code) {
			t.Fatal("compiled v41_scale_exact.spv failed the exact scalar proof")
		}
		for _, other := range []string{"v41_tail_rope_exact.spv", "v41_bf16_checked.spv"} {
			raw, err := os.ReadFile(filepath.Join(dir, other))
			if err != nil {
				t.Fatal(err)
			}
			if v41ScaleExactModuleABI(raw) {
				t.Fatalf("exact scalar proof admitted %s", other)
			}
		}
		v41ScaleExactMutations(t, code)
	}

	scales := []float32{
		float32(math.Pow(128, -0.5) * math.Pow(64, -0.5)), // V4.1 Flash indexer: dim 128, 64 heads
		float32(math.Pow(128, -0.5)), float32(math.Pow(3, -0.5) * math.Pow(4, -0.5)),
		1, 0.5, 3, float32(math.Copysign(0, -1)), 0, math.Float32frombits(1), math.Float32frombits(0x007fffff),
		math.MaxFloat32,
	}
	words := []uint32{
		0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
		0x00010000, 0x80010000, 0x00018000, 0x3f800000, 0xbf800000, 0x3f808000, 0x3f818000, 0x3f848000,
		0x3eaaaaab, 0x41820000, 0xc0ee0000, 0x7f7f0000, 0xff7f0000, 0x7f7fffff, 0xff7fffff, 0x3fc00000,
	}
	rng := rand.New(rand.NewSource(13668))
	for len(words) < 4099 {
		bits := rng.Uint32()
		if bits&0x7f800000 == 0x7f800000 {
			continue
		}
		words = append(words, bits)
	}
	x := make([]float32, len(words))
	for i, w := range words {
		x[i] = math.Float32frombits(w)
	}
	in := v.UploadClass(NewF32(Default(), []int{len(x)}, x), F32, MemoryActivation, "V4.1 exact scale witness input")
	t.Cleanup(func() { v.Free(in) })
	lanes := 0
	for _, scale := range scales {
		out, err := v.V41ScaleExact(in, scale)
		if err != nil {
			t.Fatalf("scale %08x: %v", math.Float32bits(scale), err)
		}
		if out.Buf() == in.Buf() || len(out.Shape) != 1 || out.Shape[0] != len(x) {
			t.Fatal("result is not a fresh F32 [N] allocation")
		}
		got := append([]float32(nil), v.Read(out)...)
		v.Free(out)
		for i := range x {
			// The binary64 product of two binary32 values is exact; one
			// conversion is the single RNE rounding binary32 multiply defines.
			want := float32(float64(x[i]) * float64(scale))
			if host := x[i] * scale; math.Float32bits(host) != math.Float32bits(want) {
				t.Fatalf("oracle disagreement: host binary32 product %08x, single rounding %08x", math.Float32bits(host), math.Float32bits(want))
			}
			if math.Float32bits(got[i]) != math.Float32bits(want) {
				t.Fatalf("scale %08x lane %d input %08x bits=%08x want=%08x", math.Float32bits(scale), i, words[i], math.Float32bits(got[i]), math.Float32bits(want))
			}
		}
		lanes += len(x)
	}
	if again := v.Read(in); math.Float32bits(again[2]) != 0x00000001 {
		t.Fatal("input changed")
	}
	if _, err := v.V41ScaleExact(in, float32(math.NaN())); err == nil {
		t.Fatal("nonfinite scale dispatched")
	}
	t.Logf("V41_SCALE_EXACT scales=%d lanes=%d bit_exact=%d", len(scales), lanes, lanes)
}

// v41ScaleExactMutations proves the exact scalar proof bites on the actual
// compiled twin: every single structural corruption below is refused.
func v41ScaleExactMutations(t *testing.T, code []byte) {
	t.Helper()
	words := make([]uint32, len(code)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(code[4*i:])
	}
	for name, edit := range map[string]func([]uint32) bool{
		"flush denorms": func(w []uint32) bool {
			if w[0]&0xffff == 16 && len(w) == 4 && w[2] == 4459 {
				w[2] = 4460
				return true
			}
			return false
		},
		"round toward zero": func(w []uint32) bool {
			if w[0]&0xffff == 16 && len(w) == 4 && w[2] == 4462 {
				w[2] = 4463
				return true
			}
			return false
		},
		"missing signed-zero mode": func(w []uint32) bool {
			if w[0]&0xffff == 16 && len(w) == 4 && w[2] == 4461 {
				w[2] = 4459
				return true
			}
			return false
		},
		"missing no-contraction": func(w []uint32) bool {
			if w[0]&0xffff == 71 && len(w) == 3 && w[2] == 42 {
				w[2] = 0
				return true
			}
			return false
		},
		"add instead of multiply": func(w []uint32) bool {
			if w[0]&0xffff == 133 && len(w) == 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST a SPIR-V instruction's fixed word count
				w[0] = w[0]&0xffff0000 | 129
				return true
			}
			return false
		},
		"local width": func(w []uint32) bool {
			if w[0]&0xffff == 16 && len(w) == 6 && w[2] == 17 { //boundarylint:ignore CHANGE_DETECTOR_TEST a SPIR-V instruction's fixed word count
				w[3] = 32
				return true
			}
			return false
		},
		"push scale offset": func(w []uint32) bool {
			if w[0]&0xffff == 72 && len(w) == 5 && w[2] == 1 && w[3] == 35 && w[4] == 4 { //boundarylint:ignore CHANGE_DETECTOR_TEST a SPIR-V instruction's fixed word count
				w[4] = 8
				return true
			}
			return false
		},
	} {
		changed := append([]uint32(nil), words...)
		found := false
		for i := 5; i < len(changed); {
			n := int(changed[i] >> 16)
			if n <= 0 || n > len(changed)-i {
				t.Fatal("invalid compiler module framing")
			}
			if edit(changed[i : i+n]) {
				found = true
				break
			}
			i += n
		}
		if !found {
			t.Fatalf("mutation %q did not find its ABI field", name)
		}
		raw := make([]byte, len(changed)*4)
		for i, word := range changed {
			binary.LittleEndian.PutUint32(raw[4*i:], word)
		}
		if v41ScaleExactModuleABI(raw) {
			t.Fatalf("mutation %q was admitted", name)
		}
	}
}
