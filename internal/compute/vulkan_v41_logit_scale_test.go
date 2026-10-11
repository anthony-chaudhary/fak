//go:build v41_logit_scale_witness && vulkan && (windows || linux) && cgo

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
func TestVulkanV41LogitScaleModuleABI(t *testing.T) {
	// Startup must leave FAK_VULKAN_SPIRV unset and keep this binary away
	// from a sibling spirv directory. An assertion after init cannot undo a
	// device initialization attempt; the execution receipt checks placement.
	if os.Getenv("FAK_VULKAN_SPIRV") != "" {
		t.Fatal("ABI-only witness must not request Vulkan initialization")
	}
	if _, registered := Lookup("vulkan"); registered {
		t.Fatal("ABI-only process unexpectedly registered a Vulkan device")
	}
	module := os.Getenv("FAK_V41_LOGIT_SCALE_WITNESS_MODULE")
	expected := os.Getenv("FAK_V41_LOGIT_SCALE_WITNESS_MODULE_SHA256")
	if !filepath.IsAbs(module) || filepath.Base(module) != "v41_logit_scale.spv" || len(expected) != 64 {
		t.Fatal("explicit compiled logit-scale module path and reviewed SHA256 required")
	}
	code, err := os.ReadFile(module)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(code)
	if hex.EncodeToString(digest[:]) != expected {
		t.Fatal("ABI module bytes differ from execution receipt")
	}
	if !v41LogitScaleModuleABI(code) {
		t.Fatal("actual compiled logit-scale shader failed native ABI admission")
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
		if v41LogitScaleModuleABI(raw) {
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
		"non-binary32 storage": func(w []uint32) bool {
			if w[0]&0xffff == 22 && len(w) == 3 && w[2] == 32 {
				w[2] = 64
				return true
			}
			return false
		},
		"signed count": func(w []uint32) bool {
			if w[0]&0xffff == 21 && len(w) == 4 && w[3] == 0 {
				w[3] = 1
				return true
			}
			return false
		},
		"push scale offset": func(w []uint32) bool {
			if w[0]&0xffff == 72 && len(w) == 5 && w[2] == 1 && w[3] == 35 && w[4] == 4 {
				w[4] = 8
				return true
			}
			return false
		},
		"missing no-contraction": func(w []uint32) bool {
			if w[0]&0xffff == 71 && len(w) == 3 && w[2] == 42 {
				w[2] = 0 // RelaxedPrecision overrides the mode contract too.
				return true
			}
			return false
		},
		"add instead of multiply": func(w []uint32) bool {
			if w[0]&0xffff == 133 && len(w) == 5 {
				w[0] = w[0]&0xffff0000 | 129
				return true
			}
			return false
		},
		"flush denorms": func(w []uint32) bool {
			if w[0]&0xffff == 16 && len(w) == 4 && w[2] == 4459 {
				w[2] = 4460
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
		"round toward zero": func(w []uint32) bool {
			if w[0]&0xffff == 16 && len(w) == 4 && w[2] == 4462 {
				w[2] = 4463
				return true
			}
			return false
		},
		"wrong mode capability": func(w []uint32) bool {
			if w[0]&0xffff == 17 && len(w) == 2 && w[1] == 4464 {
				w[1] = 4465
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
	t.Run("duplicate multiply", func(t *testing.T) {
		var multiply []byte
		for i := 5; i < len(words); {
			n := int(words[i] >> 16)
			if n <= 0 || n > len(words)-i {
				t.Fatal("invalid compiler module framing")
			}
			if words[i]&0xffff == 133 {
				multiply = code[4*i : 4*(i+n)]
				break
			}
			i += n
		}
		if len(multiply) == 0 {
			t.Fatal("compiled module has no multiply")
		}
		extra := append(append([]byte(nil), code...), multiply...)
		if v41LogitScaleModuleABI(extra) {
			t.Fatal("multiple floating operations admitted")
		}
	})
	for _, invalid := range [][]byte{nil, code[:19], code[:len(code)-1]} {
		if v41LogitScaleModuleABI(invalid) {
			t.Fatal("truncated module admitted")
		}
	}
}
