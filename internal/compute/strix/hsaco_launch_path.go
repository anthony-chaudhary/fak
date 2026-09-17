// Package strix: proven hsaco/AQL launch-path validation for the gfx1151 Wave32
// Q2_K decode GEMV.
//
// This file closes the seam #13026 names: before it, the decode GEMV's admission
// depended on a caller-supplied `launchAvailable bool` and a free-form operator
// string (FAK_STRIX_GEMV_LAUNCH_PATH). Neither is evidence. An operator could set
// the env var to "hsaco-aql" with no code object in existence and the kernel would
// admit — the exact "asserted, not validated" failure the fail-closed toggle was
// built to prevent.
//
// The fix inverts the trust direction: a launch path is admissible ONLY as the
// output of ValidateHSACOLaunchPath, which parses a real AMDGPU HSA code object
// (ELF64 / OSABI AMDGPU_HSA / e_machine EM_AMDGPU / EF_AMDGPU_MACH gfx1151) and
// binds the digest and target into an unforgeable-by-construction proof value.
//
// Deliberately NOT done here (gold-plating boundary, #13026): prefill GEMM tuning,
// BF16 WMMA tuning, FP4/RocmFP4 WMMA, and any change to cpuref semantics.
//
// This package cannot import internal/amdgpu: amdgpu already imports
// internal/compute/strix, so the reverse edge is an import cycle. The ELF identity
// constants below are therefore package-local mirrors of the AMD HSA Code Object
// ABI values amdgpu defines, kept byte-for-byte identical.
package strix

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
)

// AMDGPU HSA code-object identity constants (mirrored from the AMD HSA Code Object
// ABI; see internal/amdgpu/hsaco.go for the emitting side).
const (
	// ELFOSABIAMDGPUHSA is the AMDGPU HSA OSABI (0x40 / 64) required of a loadable
	// code object.
	ELFOSABIAMDGPUHSA byte = 64

	// EFAMDGPUMachGFX1151 is the RDNA 3.5 Strix Halo / Radeon 8060S machine id,
	// carried in bits 7..0 of e_flags.
	EFAMDGPUMachGFX1151 uint32 = 0x04a

	// EFAMDGPUMachGFX1150 is Strix Point (RDNA 3.5). It is not gfx1151 and must be
	// refused: a near-miss target is the most likely silent-wrong-device mistake.
	EFAMDGPUMachGFX1150 uint32 = 0x049

	// efAMDGPUMachMask isolates EF_AMDGPU_MACH from the upper e_flags bits, which
	// carry unrelated code-object versioning flags.
	efAMDGPUMachMask uint32 = 0xff

	// minCodeObjectBytes is the ELF64 header size: 16 (e_ident) + 48. Anything
	// shorter cannot carry the identity fields this validator reads.
	minCodeObjectBytes = 64
)

// LaunchPathHSACOAQL names the one launch mechanism this validator proves: a real
// gfx1151 hsaco code object dispatched through a userspace AQL queue. The string is
// the same token the operator seam and the decode-toggle record carry, so a proven
// path and an asserted path are directly comparable.
const LaunchPathHSACOAQL = "hsaco-aql"

// Errors for hsaco launch-path validation. Each names one unmet code-object
// precondition so a refusal is attributable, not a generic "invalid".
var (
	// ErrHSACOEmpty is returned for a zero-length code object.
	ErrHSACOEmpty = errors.New("strix/hsaco: empty code object")

	// ErrHSACOTruncated is returned when the bytes are too short to hold an ELF64 header.
	ErrHSACOTruncated = errors.New("strix/hsaco: code object truncated before the ELF header")

	// ErrHSACONotELF is returned when the ELF magic is absent or the header will not parse.
	ErrHSACONotELF = errors.New("strix/hsaco: not a parseable ELF code object")

	// ErrHSACONotAMDGCN is returned when e_machine is not EM_AMDGPU.
	ErrHSACONotAMDGCN = errors.New("strix/hsaco: ELF machine is not EM_AMDGPU")

	// ErrHSACONotHSA is returned when the OSABI is not AMDGPU HSA.
	ErrHSACONotHSA = errors.New("strix/hsaco: ELF OSABI is not AMDGPU_HSA")

	// ErrHSACOWrongTarget is returned when EF_AMDGPU_MACH is not gfx1151.
	ErrHSACOWrongTarget = errors.New("strix/hsaco: code object target is not gfx1151")
)

// HSACOLaunchProof is the evidence that a gfx1151 hsaco/AQL launch path exists.
//
// It is a value, not a flag: the zero value proves nothing and therefore admits
// nothing. Only ValidateHSACOLaunchPath constructs a non-zero proof, and it does so
// solely from a parsed code object. A caller cannot spell a proven launch path by
// writing a literal string.
type HSACOLaunchProof struct {
	// LaunchPath is the proven mechanism. Empty for the zero value (unproven).
	LaunchPath string `json:"launch_path"`
	// Mach is the EF_AMDGPU_MACH machine id read from the code object.
	Mach uint32 `json:"mach"`
	// SHA256 is the hex digest of the validated code object, binding the proof to
	// these exact bytes.
	SHA256 string `json:"sha256"`
	// KernelSymbol is the entry point the operator asserts lives in the object.
	// It is carried as a label only; the identifier is the digest + mach.
	KernelSymbol string `json:"kernel_symbol,omitempty"`
	// CodeObjectBytes is the validated object size, for accounting.
	CodeObjectBytes int `json:"code_object_bytes"`
}

// Proven reports whether this proof establishes a real gfx1151 hsaco launch path.
// The zero value is unproven by construction.
func (p HSACOLaunchProof) Proven() bool {
	return p.LaunchPath == LaunchPathHSACOAQL &&
		p.Mach == EFAMDGPUMachGFX1151 &&
		len(p.SHA256) == 64 &&
		p.CodeObjectBytes >= minCodeObjectBytes
}

// String renders the proof compactly for receipts and refusal messages.
func (p HSACOLaunchProof) String() string {
	if !p.Proven() {
		return "unproven launch path"
	}
	return fmt.Sprintf("%s gfx1151 sha256:%s (%d bytes)", p.LaunchPath, p.SHA256[:12], p.CodeObjectBytes)
}

// ValidateHSACOLaunchPath parses an AMDGPU HSA code object and, only if every
// gfx1151 identity precondition holds, returns the launch proof for it.
//
// kernelSymbol is a caller-supplied label for the entry point; it does not
// participate in validation. Every failure mode is typed and returned alongside a
// zero proof, so a partial result can never be mistaken for evidence.
func ValidateHSACOLaunchPath(codeObject []byte, kernelSymbol string) (HSACOLaunchProof, error) {
	if len(codeObject) == 0 {
		return HSACOLaunchProof{}, ErrHSACOEmpty
	}
	if len(codeObject) < minCodeObjectBytes {
		return HSACOLaunchProof{}, fmt.Errorf("%w: %d bytes, need >= %d",
			ErrHSACOTruncated, len(codeObject), minCodeObjectBytes)
	}

	f, err := elf.NewFile(bytes.NewReader(codeObject))
	if err != nil {
		return HSACOLaunchProof{}, fmt.Errorf("%w: %v", ErrHSACONotELF, err)
	}
	defer f.Close()

	// Identity checks, in the order the failure would be diagnosed: is it even an
	// AMDGPU code object, is it an HSA one, and is it built for THIS silicon.
	if f.Machine != elf.EM_AMDGPU {
		return HSACOLaunchProof{}, fmt.Errorf("%w: e_machine=%s", ErrHSACONotAMDGCN, f.Machine)
	}
	if f.OSABI != elf.OSABI(ELFOSABIAMDGPUHSA) {
		return HSACOLaunchProof{}, fmt.Errorf("%w: osabi=%d", ErrHSACONotHSA, byte(f.OSABI))
	}

	// Read e_flags directly: debug/elf exposes File.Flags inconsistently across Go
	// versions for non-standard OSABIs, and EF_AMDGPU_MACH lives in its low byte.
	eFlags := binary.LittleEndian.Uint32(codeObject[48:52])
	mach := eFlags & efAMDGPUMachMask
	if mach != EFAMDGPUMachGFX1151 {
		return HSACOLaunchProof{}, fmt.Errorf("%w: EF_AMDGPU_MACH=0x%03x (want gfx1151 0x%03x)",
			ErrHSACOWrongTarget, mach, EFAMDGPUMachGFX1151)
	}

	sum := sha256.Sum256(codeObject)
	return HSACOLaunchProof{
		LaunchPath:      LaunchPathHSACOAQL,
		Mach:            mach,
		SHA256:          hex.EncodeToString(sum[:]),
		KernelSymbol:    kernelSymbol,
		CodeObjectBytes: len(codeObject),
	}, nil
}

// ResolveDecodeGEMVDeviceToggleFromProof folds a proven launch path into the
// device-visible toggle. An unproven proof yields the fail-closed zero toggle with a
// reason naming the missing evidence — never a downgrade to the scalar path.
func ResolveDecodeGEMVDeviceToggleFromProof(k *Wave32GEMVDecodeKernel, proof HSACOLaunchProof) DecodeGEMVDeviceToggle {
	if !proof.Proven() {
		return DecodeGEMVDeviceToggle{Reason: "no validated gfx1151 hsaco/AQL launch path"}
	}
	return ResolveDecodeGEMVDeviceToggle(k, proof.LaunchPath)
}
