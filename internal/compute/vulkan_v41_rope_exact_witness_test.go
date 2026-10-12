//go:build v41_tail_rope_exact_witness && vulkan && (windows || linux) && cgo

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
func TestVulkanV41TailRoPEExactModuleABI(t *testing.T) {
	// Startup must leave FAK_VULKAN_SPIRV unset and keep this binary away
	// from a sibling spirv directory. An assertion after init cannot undo a
	// device initialization attempt; the execution receipt checks placement.
	if os.Getenv("FAK_VULKAN_SPIRV") != "" {
		t.Fatal("ABI-only witness must not request Vulkan initialization")
	}
	if _, registered := Lookup("vulkan"); registered {
		t.Fatal("ABI-only process unexpectedly registered a Vulkan device")
	}
	module := os.Getenv("FAK_V41_TAIL_ROPE_EXACT_WITNESS_MODULE")
	expected := os.Getenv("FAK_V41_TAIL_ROPE_EXACT_WITNESS_MODULE_SHA256")
	if !filepath.IsAbs(module) || filepath.Base(module) != "v41_tail_rope_qk.spv" || len(expected) != hex.EncodedLen(sha256.Size) {
		t.Fatal("explicit compiled strict tail-RoPE module path and reviewed SHA256 required")
	}
	code, err := os.ReadFile(module)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(code)
	if hex.EncodeToString(digest[:]) != expected {
		t.Fatal("ABI module bytes differ from execution receipt")
	}
	if !v41TailRoPEExactModuleABI(code) {
		t.Fatal("actual compiled strict tail-RoPE shader failed native ABI admission")
	}
	rawModule := os.Getenv("FAK_V41_TAIL_ROPE_EXACT_WITNESS_RAW_MODULE")
	rawExpected := os.Getenv("FAK_V41_TAIL_ROPE_EXACT_WITNESS_RAW_MODULE_SHA256")
	if !filepath.IsAbs(rawModule) || rawModule == module || filepath.Base(rawModule) != "v41_tail_rope_qk.spv" || len(rawExpected) != hex.EncodedLen(sha256.Size) {
		t.Fatal("separately compiled ordinary tail module and reviewed SHA256 required")
	}
	rawCode, err := os.ReadFile(rawModule)
	if err != nil {
		t.Fatal(err)
	}
	rawDigest := sha256.Sum256(rawCode)
	if hex.EncodeToString(rawDigest[:]) != rawExpected || v41TailRoPEExactModuleABI(rawCode) {
		t.Fatal("ordinary module failed provenance or was admitted as exact")
	}
	words := make([]uint32, len(code)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(code[4*i:])
	}
	remove := func(t *testing.T, drop func([]uint32) bool) {
		t.Helper()
		kept := append([]uint32(nil), words[:5]...)
		removed := 0
		for i := 5; i < len(words); {
			n := int(words[i] >> 16)
			if n <= 0 || n > len(words)-i {
				t.Fatal("invalid compiler module framing")
			}
			if drop(words[i : i+n]) {
				removed++
			} else {
				kept = append(kept, words[i:i+n]...)
			}
			i += n
		}
		if removed == 0 {
			t.Fatal("removal did not find its intended proof field")
		}
		raw := make([]byte, 4*len(kept))
		for i, word := range kept {
			binary.LittleEndian.PutUint32(raw[4*i:], word)
		}
		if v41TailRoPEExactModuleABI(raw) {
			t.Fatal("module missing a required proof field was admitted")
		}
	}
	t.Run("missing no-contraction only", func(t *testing.T) {
		remove(t, func(w []uint32) bool { return w[0]&0xffff == 71 && len(w) == 3 && w[2] == 42 })
	})
	for _, role := range []struct {
		name            string
		binding, access uint32
	}{
		{"q readonly", 0, 24}, {"paired readonly", 1, 24},
		{"q output writeonly", 2, 25}, {"paired output writeonly", 3, 25},
		{"table readonly", 4, 24},
	} {
		t.Run("missing "+role.name+" only", func(t *testing.T) {
			var variableID, pointerID, structID uint32
			for i := 5; i < len(words); i += int(words[i] >> 16) {
				w := words[i : i+int(words[i]>>16)]
				if w[0]&0xffff == 71 && len(w) == 4 && w[2] == 33 && w[3] == role.binding {
					variableID = w[1]
				}
			}
			for i := 5; i < len(words); i += int(words[i] >> 16) {
				w := words[i : i+int(words[i]>>16)]
				if w[0]&0xffff == 59 && len(w) == 4 && w[2] == variableID {
					pointerID = w[1]
				}
			}
			for i := 5; i < len(words); i += int(words[i] >> 16) {
				w := words[i : i+int(words[i]>>16)]
				if w[0]&0xffff == 32 && len(w) == 4 && w[1] == pointerID {
					structID = w[3]
				}
			}
			if variableID == 0 || pointerID == 0 || structID == 0 {
				t.Fatal("compiled module lacks intended role's variable/type chain")
			}
			remove(t, func(w []uint32) bool {
				return (w[0]&0xffff == 71 && len(w) == 3 && w[1] == variableID && w[2] == role.access) ||
					(w[0]&0xffff == 72 && len(w) == 4 && w[1] == structID && w[2] == 0 && w[3] == role.access)
			})
		})
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
		if v41TailRoPEExactModuleABI(raw) {
			t.Fatal("incompatible module was admitted")
		}
	}
	for name, edit := range map[string]func([]uint32) bool{
		"local width": func(w []uint32) bool {
			if w[0]&0xffff == 16 && len(w) == 6 && w[2] == 17 { //boundarylint:ignore CHANGE_DETECTOR_TEST SPIR-V OpExecutionMode LocalSize has six words: opcode, entry point, mode, x, y, z.
				w[3] = 128
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
		"signed row storage": func(w []uint32) bool {
			if w[0]&0xffff == 21 && len(w) == 4 && w[3] == 0 {
				w[3] = 1
				return true
			}
			return false
		},
		"push head width offset": func(w []uint32) bool {
			if w[0]&0xffff == 72 && len(w) == 5 && w[2] == 1 && w[3] == 35 && w[4] == 4 { //boundarylint:ignore CHANGE_DETECTOR_TEST SPIR-V OpMemberDecorate Offset has five words: opcode, target, member, decoration, offset.
				w[4] = 8
				return true
			}
			return false
		},
		"add instead of multiply": func(w []uint32) bool {
			if w[0]&0xffff == 133 && len(w) == 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST SPIR-V binary float arithmetic has five words: opcode, result type, result ID, left operand, right operand.
				w[0] = w[0]&0xffff0000 | 129
				return true
			}
			return false
		},
		"wrong product operand": func(w []uint32) bool {
			if w[0]&0xffff == 133 && len(w) == 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST SPIR-V binary float arithmetic has five words: opcode, result type, result ID, left operand, right operand.
				w[4] = w[3]
				return true
			}
			return false
		},
		"add instead of subtract": func(w []uint32) bool {
			if w[0]&0xffff == 131 && len(w) == 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST SPIR-V binary float arithmetic has five words: opcode, result type, result ID, left operand, right operand.
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
	t.Run("relaxed precision override only", func(t *testing.T) {
		float32Types := make(map[uint32]bool)
		for i := 5; i < len(words); i += int(words[i] >> 16) {
			w := words[i : i+int(words[i]>>16)]
			if w[0]&0xffff == 22 && len(w) == 3 && w[2] == 32 {
				float32Types[w[1]] = true
			}
		}
		var id uint32
		for i := 5; i < len(words); i += int(words[i] >> 16) {
			w := words[i : i+int(words[i]>>16)]
			op := w[0] & 0xffff
			if len(w) == 5 && (op == 129 || op == 131 || op == 133) && float32Types[w[1]] { //boundarylint:ignore CHANGE_DETECTOR_TEST SPIR-V binary float arithmetic has five words: opcode, result type, result ID, left operand, right operand.
				id = w[2]
				break
			}
		}
		if id == 0 {
			t.Fatal("compiled module has no intended F32 arithmetic result")
		}
		// Keep NoContraction and every other existing proof unchanged, so
		// only the prohibited RelaxedPrecision decoration can reject this.
		extra := append([]byte(nil), code...)
		for _, word := range []uint32{3<<16 | 71, id, 0} {
			extra = binary.LittleEndian.AppendUint32(extra, word)
		}
		if v41TailRoPEExactModuleABI(extra) {
			t.Fatal("RelaxedPrecision override was admitted")
		}
	})
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
		if v41TailRoPEExactModuleABI(extra) {
			t.Fatal("multiple floating operations admitted")
		}
	})
	// Effective access combines variable and member decorations. A matching
	// member qualifier must never mask a contradictory variable qualifier.
	for _, control := range []struct {
		name                string
		binding, decoration uint32
	}{
		{"unreadable input variable", 0, 25},
		{"unreadable table variable", 4, 25},
		{"unwritable output variable", 2, 24},
	} {
		t.Run(control.name, func(t *testing.T) {
			var id uint32
			for i := 5; i < len(words); {
				n := int(words[i] >> 16)
				if n <= 0 || n > len(words)-i {
					t.Fatal("invalid compiler module framing")
				}
				if words[i]&0xffff == 71 && n == 4 && words[i+2] == 33 && words[i+3] == control.binding {
					id = words[i+1]
					break
				}
				i += n
			}
			if id == 0 {
				t.Fatal("compiled module has no intended buffer binding")
			}
			extra := append([]byte(nil), code...)
			for _, word := range []uint32{3<<16 | 71, id, control.decoration} {
				extra = binary.LittleEndian.AppendUint32(extra, word)
			}
			if v41TailRoPEExactModuleABI(extra) {
				t.Fatal("contradictory variable access was admitted")
			}
		})
	}
	for _, invalid := range [][]byte{nil, code[:19], code[:len(code)-1]} {
		if v41TailRoPEExactModuleABI(invalid) {
			t.Fatal("truncated module admitted")
		}
	}
}
