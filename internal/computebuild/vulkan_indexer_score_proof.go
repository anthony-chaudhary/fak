package computebuild

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"path/filepath"
)

const vulkanIndexerScoreStructureID = "fak.v41-indexer-score-f32-structure.v1"
const vulkanNativeArchivePath = "internal/compute/libfakvulkan.a"

// VulkanIndexerScoreContract describes independently inspected SPIR-V structure.
// It proves declarations and decorations, not control/data flow, source-to-module
// correspondence, ordered arithmetic semantics, or physical device support.
// The latter claims need a witnessed build and separate native qualification.
type VulkanIndexerScoreContract struct {
	PolicyID                 string    `json:"policy_id"`
	ModuleSHA256             string    `json:"module_sha256"`
	EntryPoint               string    `json:"entry_point"`
	LocalSize                [3]uint32 `json:"local_size"`
	FloatWidth               uint32    `json:"float_width"`
	DenormPreserve           bool      `json:"denorm_preserve"`
	SignedZeroInfNanPreserve bool      `json:"signed_zero_inf_nan_preserve"`
	RoundingModeRTE          bool      `json:"rounding_mode_rte"`
	NoContractionFAddCount   uint32    `json:"no_contraction_fadd_count"`
	NoContractionFMulCount   uint32    `json:"no_contraction_fmul_count"`
}

func expectedVulkanIndexerScoreContract(digest string) VulkanIndexerScoreContract {
	return VulkanIndexerScoreContract{
		PolicyID: vulkanIndexerScoreStructureID, ModuleSHA256: digest,
		EntryPoint: "main", LocalSize: [3]uint32{1, 1, 1}, FloatWidth: 32,
		DenormPreserve: true, SignedZeroInfNanPreserve: true, RoundingModeRTE: true,
		NoContractionFAddCount: 2, NoContractionFMulCount: 2,
	}
}

func validateVulkanV5Extras(archive *BuildArtifact, score *VulkanIndexerScoreContract) error {
	if archive == nil || archive.Path != vulkanNativeArchivePath || archive.SizeBytes <= 0 || archive.Signed || !validLowerSHA256(archive.SHA256) {
		return fmt.Errorf("Vulkan V5 native archive identity is incomplete")
	}
	if score == nil || !validLowerSHA256(score.ModuleSHA256) || *score != expectedVulkanIndexerScoreContract(score.ModuleSHA256) {
		return fmt.Errorf("Vulkan V5 indexer score structural contract is incomplete")
	}
	return nil
}

func observeVulkanNativeArchiveV5(root string) (BuildArtifact, error) {
	digest, size, err := stableRegularFileSHA256(filepath.Join(root, filepath.FromSlash(vulkanNativeArchivePath)), "Vulkan native archive")
	if err != nil || size <= 0 {
		return BuildArtifact{}, fmt.Errorf("Vulkan V5 native archive unavailable or empty: %v", err)
	}
	return BuildArtifact{Path: vulkanNativeArchivePath, SizeBytes: size, SHA256: digest}, nil
}

// inspectVulkanIndexerScoreSPIRV is deliberately a narrow structural check, not
// a replacement for spirv-val or a semantic-equivalence witness. Numeric values
// follow Khronos SPIR-V and SPV_KHR_float_controls. The byte slice is the same
// stable file snapshot hashed into the exact62 bundle, avoiding a second read.
func inspectVulkanIndexerScoreSPIRV(data []byte) (VulkanIndexerScoreContract, error) {
	fail := func(why string) (VulkanIndexerScoreContract, error) {
		return VulkanIndexerScoreContract{}, fmt.Errorf("Vulkan V5 indexer score SPIR-V: %s", why)
	}
	if len(data) < 20 || len(data) > 1024*1024 || len(data)%4 != 0 {
		return fail("invalid module length")
	}
	words := make([]uint32, len(data)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(data[4*i:])
	}
	if words[0] != 0x07230203 || words[1] < 0x00010000 || words[1] > 0x00010500 || words[1]&0xff != 0 || words[3] == 0 || words[4] != 0 {
		return fail("invalid or unsupported module header")
	}
	validID := func(id uint32) bool { return id != 0 && id < words[3] }
	caps := make(map[uint32]bool)
	floatTypes := make(map[uint32]bool)
	vectorElements := make(map[uint32]uint32)
	decorated := make(map[uint32]bool)
	results := make(map[uint32]uint32)
	type mode struct{ entry, width uint32 }
	modes := make(map[uint32]mode)
	var entry, localEntry, entries, locals, adds, muls, extensions uint32
	for pos := 5; pos < len(words); {
		count, op := int(words[pos]>>16), words[pos]&0xffff
		if count == 0 || count > len(words)-pos {
			return fail("malformed instruction length")
		}
		a := words[pos : pos+count]
		switch op {
		case 10: // OpExtension: the shader explicitly declares float controls.
			if count < 2 {
				return fail("malformed extension")
			}
			literal := make([]byte, 4*(count-1))
			for i, word := range a[1:] {
				binary.LittleEndian.PutUint32(literal[4*i:], word)
			}
			end := bytes.IndexByte(literal, 0)
			if end < 0 || string(literal[:end]) != "SPV_KHR_float_controls" {
				return fail("unsupported extension")
			}
			for _, b := range literal[end:] {
				if b != 0 {
					return fail("malformed extension padding")
				}
			}
			extensions++
		case 15: // OpEntryPoint: one GLCompute main, interface IDs may follow.
			if count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0 {
				return fail("expected one compute main entry point")
			}
			entry, entries = a[2], entries+1
		case 16: // OpExecutionMode: fail closed on unrecognized modes.
			if count < 3 || !validID(a[1]) {
				return fail("malformed execution mode")
			}
			switch a[2] {
			case 17: // LocalSize
				if count != 6 || a[3] != 1 || a[4] != 1 || a[5] != 1 {
					return fail("expected local size 1,1,1")
				}
				localEntry, locals = a[1], locals+1
			case 4459, 4461, 4462: // DenormPreserve, SignedZeroInfNanPreserve, RTE
				if count != 4 || a[3] != 32 {
					return fail("required execution mode must apply to binary32")
				}
				if _, exists := modes[a[2]]; exists {
					return fail("duplicate float-control execution mode")
				}
				modes[a[2]] = mode{a[1], a[3]}
			default:
				return fail("unsupported execution mode")
			}
		case 17: // OpCapability: this scalar kernel needs no alternate math profile.
			if count != 2 || caps[a[1]] || (a[1] != 1 && a[1] != 4464 && a[1] != 4466 && a[1] != 4467) {
				return fail("duplicate or unsupported capability")
			}
			caps[a[1]] = true
		case 22: // OpTypeFloat
			if count != 3 || !validID(a[1]) || a[2] != 32 || floatTypes[a[1]] {
				return fail("only scalar binary32 floating point is supported")
			}
			floatTypes[a[1]] = true
		case 23: // Integer gl_GlobalInvocationID vectors remain legal.
			if count != 4 || !validID(a[1]) || !validID(a[2]) {
				return fail("malformed vector type")
			}
			if _, exists := vectorElements[a[1]]; exists {
				return fail("duplicate vector type")
			}
			vectorElements[a[1]] = a[2]
		case 24: // OpTypeMatrix
			return fail("matrix floating point is unsupported")
		case 71: // OpDecorate
			if count < 3 || !validID(a[1]) {
				return fail("malformed decoration")
			}
			switch a[2] {
			case 0, 1, 39, 40: // RelaxedPrecision, SpecId, FPRoundingMode, FPFastMathMode
				return fail("floating-point override decoration")
			case 42: // NoContraction
				if count != 3 || decorated[a[1]] {
					return fail("duplicate or malformed NoContraction")
				}
				decorated[a[1]] = true
			}
		case 72: // OpMemberDecorate
			if count < 4 || !validID(a[1]) || a[3] == 0 || a[3] == 39 || a[3] == 40 || a[3] == 42 {
				return fail("unsupported member floating-point decoration")
			}
		case 48, 49, 50, 51, 52:
			return fail("specialization operations are unsupported")
		case 73, 74, 75, 331, 332, 5632, 5633:
			return fail("indirect execution modes or decorations are unsupported")
		case 129, 133: // OpFAdd, OpFMul
			if count != 5 || !validID(a[1]) || !validID(a[2]) || !validID(a[3]) || !validID(a[4]) {
				return fail("malformed scalar arithmetic")
			}
			if _, exists := results[a[2]]; exists {
				return fail("duplicate arithmetic result")
			}
			results[a[2]] = a[1]
			if op == 129 {
				adds++
			} else {
				muls++
			}
		case 12, 109, 110, 111, 112, 115, 116, 127, 131, 136, 140, 141, 142, 143, 144, 145, 146, 147, 148,
			207, 208, 209, 210, 211, 212, 213, 214, 215,
			265, 266, 269, 350, 352, 355, 358:
			return fail("extended, fused, or alternate floating-point arithmetic")
		}
		pos += count
	}
	if extensions != 1 || entries != 1 || locals != 1 || entry != localEntry || adds != 2 || muls != 2 || len(decorated) != 4 {
		return fail("float-controls extension, entry, local size, or exact four arithmetic sites missing")
	}
	for _, pair := range [][2]uint32{{4459, 4464}, {4461, 4466}, {4462, 4467}} {
		m, exists := modes[pair[0]]
		if !exists || m.entry != entry || m.width != 32 || !caps[pair[1]] || !caps[1] {
			return fail("missing main-entry binary32 mode or capability")
		}
	}
	for _, element := range vectorElements {
		if floatTypes[element] {
			return fail("floating-point vectors are unsupported")
		}
	}
	for id, resultType := range results {
		if !floatTypes[resultType] || !decorated[id] {
			return fail("arithmetic result lacks scalar binary32 type or NoContraction")
		}
	}
	digest := sha256.Sum256(data)
	return expectedVulkanIndexerScoreContract(hex.EncodeToString(digest[:])), nil
}
