//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime integration est=5s lane=optin
// Physical qualification requires FAK_V41_INDEX_SCALE_BF16=1 before init.
// Skipped execution is not a numerical witness.
func TestVulkanV41IndexScaleBF16(t *testing.T) {
	v := vk(t)
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("physical GPU required")
		}
		t.Skip("physical GPU absent")
	}
	finite := make([]float32, 0, 65536)
	for i := uint32(0); i < 65536; i++ {
		bits := i << 16
		if bits&0x7f800000 != 0x7f800000 {
			finite = append(finite, math.Float32frombits(bits))
		}
	}
	cases := [][]float32{finite}
	for _, bits := range []uint32{1, 0x80000001, 0x007fffff, 0x00800000, 0x3f807fff, 0x3f808000, 0x3f808001, 0x7f7f7fff, 0x7f7f8000, 0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001} {
		cases = append(cases, []float32{0, math.Float32frombits(bits), math.Float32frombits(bits)})
	}
	for ci, values := range cases {
		func() {
			input := v.UploadClass(NewF32(Default(), []int{len(values)}, values), F32, MemoryActivation, "index scale BF16 qualification")
			defer v.Free(input)
			if !v.SupportsV41IndexScaleBF16(input, 1) {
				if os.Getenv("FAK_V41_INDEX_SCALE_BF16") == "1" {
					t.Fatal("requested strict fused module unavailable")
				}
				t.Skip("optional module not enabled")
			}
			for _, scale := range []float32{0, math.Float32frombits(0x80000000), 1, -1, 0.5, math.SmallestNonzeroFloat32, math.MaxFloat32, float32(1 / math.Sqrt(128*64))} {
				func() {
					first := -1
					for i, value := range values {
						_, bad := v41IndexScaleBF16Oracle(value, scale)
						if bad {
							first = i
							break
						}
					}
					out, st, e := v.V41IndexScaleBF16Deferred(input, scale)
					if e != nil {
						t.Fatal(e)
					}
					defer v.Free(out)
					if st.Kind != V41DeferredBF16Publish || st.Count != len(values) || st.Site != V41BF16PublishIndexerScaledWeight {
						v.Free(st.Record)
						t.Fatal("wrong fixed status")
					}
					var outputs []Tensor
					if first < 0 {
						outputs = []Tensor{out}
					}
					verdicts, rows, e := v.V41ReadDeferred([]V41DeferredStatus{st}, outputs)
					if e != nil || len(verdicts) != 1 {
						t.Fatalf("case %d read: %v", ci, e)
					}
					if first >= 0 {
						var fault *V41BF16CheckedArithmeticError
						if !errors.As(verdicts[0], &fault) || fault.Site != V41BF16PublishIndexerScaledWeight || fault.Index != first {
							t.Fatalf("case %d fault %v at %d", ci, verdicts, first)
						}
						return
					}
					if verdicts[0] != nil || len(rows) != 1 || len(rows[0]) != len(values) {
						t.Fatal("result shape/verdict")
					}
					for i, value := range values {
						want, _ := v41IndexScaleBF16Oracle(value, scale)
						if math.Float32bits(rows[0][i]) != want {
							t.Fatalf("case %d lane %d scale %08x", ci, i, math.Float32bits(scale))
						}
					}
				}()
			}
		}()
	}
}

// fak-test:runtime fast est=20ms lane=optin
// Exercises actual native byte attestation without requiring a GPU. The witness
// must launch away from a sibling spirv directory and leave FAK_VULKAN_SPIRV unset.
func TestVulkanV41IndexScaleBF16ModuleABI(t *testing.T) {
	module := os.Getenv("FAK_V41_INDEX_SCALE_BF16_WITNESS_MODULE")
	if module == "" {
		t.Skip("explicit qualified module witness path absent")
	}
	if os.Getenv("FAK_VULKAN_SPIRV") != "" {
		t.Fatal("ABI witness must not request device initialization")
	}
	if _, registered := Lookup("vulkan"); registered {
		t.Fatal("ABI witness unexpectedly registered a device")
	}
	expected := os.Getenv("FAK_V41_INDEX_SCALE_BF16_WITNESS_MODULE_SHA256")
	if !filepath.IsAbs(module) || filepath.Base(module) != "v41_index_scale_bf16.spv" || len(expected) != hex.EncodedLen(sha256.Size) {
		t.Fatal("explicit module path and reviewed SHA256 required")
	}
	code, e := os.ReadFile(module)
	if e != nil {
		t.Fatal(e)
	}
	digest := sha256.Sum256(code)
	if hex.EncodeToString(digest[:]) != expected {
		t.Fatal("module differs from receipt")
	}
	if !v41IndexScaleBF16ModuleABI(code) {
		t.Fatal("qualified module refused")
	}
	for i := range code {
		bad := append([]byte(nil), code...)
		bad[i] ^= 1
		if v41IndexScaleBF16ModuleABI(bad) {
			t.Fatalf("byte mutation %d admitted", i)
		}
	}
	for n := 0; n < len(code); n++ {
		if v41IndexScaleBF16ModuleABI(code[:n]) {
			t.Fatalf("truncation %d admitted", n)
		}
	}
	bad := append(append([]byte(nil), code...), 0x3d, 0, 1, 0)
	if v41IndexScaleBF16ModuleABI(bad) {
		t.Fatal("malformed one-word OpLoad append admitted")
	}
	found := false
	for i := 20; i < len(code); {
		word := binary.LittleEndian.Uint32(code[i:])
		n := int(word>>16) * 4
		if n == 0 || n > len(code)-i {
			t.Fatal("invalid qualified framing")
		}
		if word&65535 == 133 {
			if n != 20 {
				t.Fatal("invalid FMul framing")
			}
			bad := append([]byte(nil), code...)
			copy(bad[i+16:i+20], bad[i+12:i+16])
			if v41IndexScaleBF16ModuleABI(bad) {
				t.Fatal("input-square substitution admitted")
			}
			found = true
		}
		i += n
	}
	if !found {
		t.Fatal("qualified product missing")
	}
}
