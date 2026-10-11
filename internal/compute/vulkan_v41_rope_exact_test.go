//go:build !v41_tail_rope_exact_witness && vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime integration est=1s lane=optin
// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1 and FAK_VULKAN_SPIRV naming
// the current V12 bundle; a skip is no witness. The ordinary archive must admit
// the strict twin on a device reporting all three binary32 controls, and the
// rows panel it then runs is bit-exact against separately rounded products,
// including subnormal, signed-zero and tie operands (#13816).
func TestVulkanV41TailRoPEExactRows(t *testing.T) {
	v := vk(t)
	if !v.V41TailRoPEExactAvailable() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required device did not admit the strict tail-RoPE twin (float controls or module proof missing)")
		}
		t.Skip("strict tail-RoPE twin not admitted on this device")
	}
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatalf("required physical GPU not established: deviceType=%d tier=%s", kind, v.Tier())
		}
		t.Skipf("physical GPU not established: deviceType=%d", kind)
	}
	if dir := os.Getenv("FAK_VULKAN_SPIRV"); dir != "" {
		for _, c := range []struct {
			module string
			strict bool
		}{{"v41_tail_rope_exact.spv", true}, {"v41_tail_rope_qk.spv", false}} {
			code, err := os.ReadFile(filepath.Join(dir, c.module))
			if err != nil {
				t.Fatal(err)
			}
			if v41TailRoPEExactModuleABI(code) != c.strict {
				t.Fatalf("%s strict ABI admission=%t, want %t", c.module, !c.strict, c.strict)
			}
		}
	}
	bits := func(u uint32) float32 { return math.Float32frombits(u) }
	// Each row is [prefix..., a, b] with rotaryDim 2. Angles 0 and 2 cover the
	// identity table (subnormal/signed-zero preservation) and a cancelling sum.
	operands := [][2]uint32{
		{0x80000000, 0}, {0, 0x80000000}, {0x80000000, 0x80000000},
		{0x00000001, 0}, {0x80000001, 0}, {0x00010000, 0}, {0x80010000, 0},
		{0x00400000, 0x80400000}, {0x007fffff, 0x00000003},
		{0x3f808000, 0}, {0xbf818000, 0}, {0x41820000, 0xc0ee0000},
		{0x00800000, 0x80800000}, {0x7f7fffff, 0x00000001},
	}
	const width = 4
	for _, angle := range []float64{0, 2, 0.37} {
		x := make([]float32, 0, len(operands)*width)
		for i, op := range operands {
			x = append(x, bits(0x80000000), float32(i)+0.5, bits(op[0]), bits(op[1]))
		}
		table := []float32{float32(math.Sin(angle)), float32(math.Cos(angle))}
		rows := len(operands)
		dx := v.UploadClass(NewF32(Default(), []int{rows, width}, x), F32, MemoryActivation, "V4.1 exact rows witness input")
		t.Cleanup(func() { v.Free(dx) })
		dt := v.UploadClass(NewF32(Default(), []int{1, 2}, table), F32, MemoryActivation, "V4.1 exact rows witness table")
		t.Cleanup(func() { v.Free(dt) })
		out, err := v.V41TailRoPERows(dx, dt, rows, width, 2)
		if err != nil {
			t.Fatal(err)
		}
		got := append([]float32(nil), v.Read(out)...)
		v.Free(out)
		want := make([]float32, len(x))
		copy(want, x)
		for r := 0; r < rows; r++ {
			i := r*width + width - 2
			a, b := float64(x[i]), float64(x[i+1])
			s, c := float64(table[0]), float64(table[1])
			ac, bs, bc, as := float32(a*c), float32(b*s), float32(b*c), float32(a*s)
			want[i], want[i+1] = float32(float64(ac)-float64(bs)), float32(float64(bc)+float64(as))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("angle %g lane %d bits=%08x want=%08x", angle, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
		t.Logf("V41_TAIL_ROPE_EXACT_ROWS angle=%g rows=%d lanes=%d bit_exact=%d", angle, rows, len(want), len(want))
	}
}
