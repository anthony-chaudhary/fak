//go:build v41_bf16_boundary_witness && vulkan && (windows || linux) && cgo

package compute

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

// Exercise the actual native ABI parser against the reviewed compiler's real
// module and structural corruptions of those bytes. No device is needed, and
// no parser-only result establishes numerical or physical qualification.
// fak-test:runtime fast est=20ms lane=optin
func TestVulkanV41BF16BoundaryModuleABI(t *testing.T) {
	// Startup must leave FAK_VULKAN_SPIRV unset and keep this binary away
	// from a sibling spirv directory. An assertion after init cannot undo a
	// device initialization attempt; the execution receipt checks placement.
	if os.Getenv("FAK_VULKAN_SPIRV") != "" {
		t.Fatal("ABI-only witness must not request Vulkan initialization")
	}
	if _, registered := Lookup("vulkan"); registered {
		t.Fatal("ABI-only process unexpectedly registered a Vulkan device")
	}
	module := os.Getenv("FAK_V41_BF16_BOUNDARY_WITNESS_MODULE")
	expected := os.Getenv("FAK_V41_BF16_BOUNDARY_WITNESS_MODULE_SHA256")
	if !filepath.IsAbs(module) || filepath.Base(module) != "v41_bf16_boundary.spv" || len(expected) != 64 {
		t.Fatal("explicit compiled BF16 module path and reviewed SHA256 required")
	}
	code, err := os.ReadFile(module)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(code)
	if hex.EncodeToString(digest[:]) != expected {
		t.Fatal("ABI module bytes differ from execution receipt")
	}
	if !v41BF16BoundaryModuleABI(code) {
		t.Fatal("actual compiled BF16 shader failed native ABI admission")
	}
	words := make([]uint32, len(code)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(code[4*i:])
	}
	mutate := func(t *testing.T, edit func([]uint32) bool) {
		t.Helper()
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
			t.Fatal("mutation did not find its intended ABI field")
		}
		raw := make([]byte, len(changed)*4)
		for i, word := range changed {
			binary.LittleEndian.PutUint32(raw[4*i:], word)
		}
		if v41BF16BoundaryModuleABI(raw) {
			t.Fatal("incompatible module was admitted")
		}
	}
	for name, edit := range map[string]func([]uint32) bool{
		"local width": func(w []uint32) bool {
			if w[0]&0xffff == 16 && len(w) == 6 && w[2] == 17 {
				w[3] = 32
				return true
			}
			return false
		},
		"duplicate binding": func(w []uint32) bool {
			if w[0]&0xffff == 71 && len(w) == 4 && w[2] == 33 && w[3] == 1 {
				w[3] = 0
				return true
			}
			return false
		},
		"signed storage": func(w []uint32) bool {
			if w[0]&0xffff == 21 && len(w) == 4 && w[3] == 0 {
				w[3] = 1
				return true
			}
			return false
		},
		"writable input": func(w []uint32) bool {
			if w[0]&0xffff == 72 && len(w) == 4 && w[3] == 24 {
				w[3] = 25
				return true
			}
			return false
		},
		"wrong array stride": func(w []uint32) bool {
			if w[0]&0xffff == 71 && len(w) == 4 && w[2] == 6 {
				w[3] = 8
				return true
			}
			return false
		},
		"zero instruction size": func(w []uint32) bool { w[0] &= 0xffff; return true },
	} {
		t.Run(name, func(t *testing.T) { mutate(t, edit) })
	}
	for _, invalid := range [][]byte{nil, code[:19], code[:len(code)-1]} {
		if v41BF16BoundaryModuleABI(invalid) {
			t.Fatal("truncated module admitted")
		}
	}
}
