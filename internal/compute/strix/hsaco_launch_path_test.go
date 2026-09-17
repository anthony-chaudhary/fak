package strix

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// buildGFX1151HSACO emits a minimal but structurally real AMDGPU HSA code object
// (ELF64, little-endian, ET_DYN, OSABI AMDGPU_HSA, e_machine EM_AMDGPU) with the
// gfx1151 EF_AMDGPU_MACH machine id in e_flags. It is the positive fixture for the
// hsaco launch-path validator: everything downstream of the ELF header is omitted
// because validation only depends on the code-object identity, not on sections.
func buildGFX1151HSACO(t *testing.T, mach uint32, osabi byte, class elf.Class, machine elf.Machine) []byte {
	t.Helper()

	buf := new(bytes.Buffer)
	buf.Write([]byte{0x7f, 'E', 'L', 'F'}) // e_ident[EI_MAG0..3]
	buf.WriteByte(byte(class))             // EI_CLASS
	buf.WriteByte(byte(elf.ELFDATA2LSB))   // EI_DATA
	buf.WriteByte(byte(elf.EV_CURRENT))    // EI_VERSION
	buf.WriteByte(osabi)                   // EI_OSABI (AMDGPU HSA = 0x40 / 64)
	buf.WriteByte(byte(elf.EV_CURRENT))    // EI_ABIVERSION
	for i := 0; i < 7; i++ {
		buf.WriteByte(0) // EI_PAD
	}

	write16 := func(v uint16) {
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], v)
		buf.Write(b[:])
	}
	write32 := func(v uint32) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], v)
		buf.Write(b[:])
	}
	write64 := func(v uint64) {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], v)
		buf.Write(b[:])
	}

	write16(uint16(elf.ET_DYN))     // e_type
	write16(uint16(machine))        // e_machine
	write32(uint32(elf.EV_CURRENT)) // e_version
	write64(0)                      // e_entry
	write64(0)                      // e_phoff
	write64(0)                      // e_shoff
	write32(mach)                   // e_flags carries EF_AMDGPU_MACH in bits 7..0
	write16(uint16(elf.ELFCLASS64)) // e_ehsize
	write16(0)                      // e_phentsize
	write16(0)                      // e_phnum
	write16(0)                      // e_shentsize
	write16(0)                      // e_shnum
	write16(0)                      // e_shstrndx

	return buf.Bytes()
}

// TestStrixHSACO_LaunchPathValidated is the reproduction/acceptance test for #13026:
// the Q2_K decode GEMV launch path must be PROVEN by validating a gfx1151 hsaco code
// object, not merely asserted by the operator. Before the fix, admission is a bare
// bool and any non-empty launch string admits, so the negative cases below fail.
func TestStrixHSACO_LaunchPathValidated(t *testing.T) {
	t.Run("a valid gfx1151 hsaco proves the launch path", func(t *testing.T) {
		hsaco := buildGFX1151HSACO(t, EFAMDGPUMachGFX1151, ELFOSABIAMDGPUHSA, elf.ELFCLASS64, elf.EM_AMDGPU)
		proof, err := ValidateHSACOLaunchPath(hsaco, "strix_q2k_decode_gemv")
		if err != nil {
			t.Fatalf("valid gfx1151 hsaco must prove the launch path: %v", err)
		}
		if proof.LaunchPath != LaunchPathHSACOAQL {
			t.Fatalf("launch path %q != %q", proof.LaunchPath, LaunchPathHSACOAQL)
		}
		if proof.Mach != EFAMDGPUMachGFX1151 {
			t.Fatalf("mach 0x%x != gfx1151 0x%x", proof.Mach, EFAMDGPUMachGFX1151)
		}
		if len(proof.SHA256) != 64 {
			t.Fatalf("code-object digest must be 64 hex chars, got %d (%q)", len(proof.SHA256), proof.SHA256)
		}
		if proof.KernelSymbol != "strix_q2k_decode_gemv" {
			t.Fatalf("kernel symbol %q not carried into the proof", proof.KernelSymbol)
		}
	})

	t.Run("a proven launch path admits the decode GEMV", func(t *testing.T) {
		hsaco := buildGFX1151HSACO(t, EFAMDGPUMachGFX1151, ELFOSABIAMDGPUHSA, elf.ELFCLASS64, elf.EM_AMDGPU)
		proof, err := ValidateHSACOLaunchPath(hsaco, "strix_q2k_decode_gemv")
		if err != nil {
			t.Fatalf("unexpected validation error: %v", err)
		}
		k := NewWave32GEMVDecodeKernelFromProof(DefaultWave32GEMVDecodeConfig(), proof)
		if !k.Available() {
			t.Fatalf("a validated hsaco launch path must admit the kernel: %s", k.Admission().Reason)
		}
		toggle := ResolveDecodeGEMVDeviceToggleFromProof(k, proof)
		if !toggle.Admitted {
			t.Fatalf("device toggle must admit on a validated proof: %s", toggle.Reason)
		}
		if toggle.LaunchPath != LaunchPathHSACOAQL {
			t.Fatalf("toggle launch path %q != %q", toggle.LaunchPath, LaunchPathHSACOAQL)
		}
	})

	t.Run("an empty proof never admits", func(t *testing.T) {
		k := NewWave32GEMVDecodeKernelFromProof(DefaultWave32GEMVDecodeConfig(), HSACOLaunchProof{})
		if k.Available() {
			t.Fatal("zero-value proof must fail closed")
		}
	})

	t.Run("a non-gfx1151 code object is refused", func(t *testing.T) {
		// gfx1150 (Strix Point) is the near-miss: same family, wrong target.
		hsaco := buildGFX1151HSACO(t, EFAMDGPUMachGFX1150, ELFOSABIAMDGPUHSA, elf.ELFCLASS64, elf.EM_AMDGPU)
		_, err := ValidateHSACOLaunchPath(hsaco, "strix_q2k_decode_gemv")
		if err == nil {
			t.Fatal("a gfx1150 code object must not prove a gfx1151 launch path")
		}
		if !errors.Is(err, ErrHSACOWrongTarget) {
			t.Fatalf("want ErrHSACOWrongTarget, got %v", err)
		}
	})

	t.Run("an empty code object is refused, not defaulted", func(t *testing.T) {
		_, err := ValidateHSACOLaunchPath(nil, "strix_q2k_decode_gemv")
		if err == nil {
			t.Fatal("an empty code object must not prove a launch path")
		}
	})

	t.Run("a non-AMDGPU ELF is refused", func(t *testing.T) {
		hsaco := buildGFX1151HSACO(t, EFAMDGPUMachGFX1151, ELFOSABIAMDGPUHSA, elf.ELFCLASS64, elf.EM_X86_64)
		if _, err := ValidateHSACOLaunchPath(hsaco, "k"); !errors.Is(err, ErrHSACONotAMDGCN) {
			t.Fatalf("want ErrHSACONotAMDGCN for a non-AMDGPU machine, got %v", err)
		}
	})

	t.Run("a non-HSA OSABI is refused", func(t *testing.T) {
		hsaco := buildGFX1151HSACO(t, EFAMDGPUMachGFX1151, 0 /* ELFOSABI_NONE */, elf.ELFCLASS64, elf.EM_AMDGPU)
		if _, err := ValidateHSACOLaunchPath(hsaco, "k"); !errors.Is(err, ErrHSACONotHSA) {
			t.Fatalf("want ErrHSACONotHSA for a non-HSA OSABI, got %v", err)
		}
	})

	t.Run("truncated bytes are refused", func(t *testing.T) {
		hsaco := buildGFX1151HSACO(t, EFAMDGPUMachGFX1151, ELFOSABIAMDGPUHSA, elf.ELFCLASS64, elf.EM_AMDGPU)
		if _, err := ValidateHSACOLaunchPath(hsaco[:24], "k"); err == nil {
			t.Fatal("a truncated ELF must not prove a launch path")
		}
	})
}

// TestStrixHSACO_FailClosedInvariantSurvives proves the wiring change does not
// weaken the #12997 fail-closed contract: an unproven kernel still refuses, and the
// refusal still names the toggle error rather than silently running the CPU path.
func TestStrixHSACO_FailClosedInvariantSurvives(t *testing.T) {
	k := DefaultWave32GEMVDecodeKernel()
	if k.Available() {
		t.Fatal("default kernel must stay unavailable")
	}
	raw := make([]byte, Q2KSuperBlockBytes)
	x := make([]float32, Q2KSuperBlockElems)
	if _, err := k.DispatchQ2KGEMV(raw, x, 1, Q2KSuperBlockElems); !errors.Is(err, ErrWave32GEMVUnavailable) {
		t.Fatalf("unproven kernel must refuse with ErrWave32GEMVUnavailable, got %v", err)
	}

	// The operator env seam stays fail-closed: 'none'/empty never admit.
	t.Setenv(EnvStrixGEMVLaunchPath, "none")
	if StrixGEMVLaunchPath() != "" {
		t.Fatal("'none' must normalize to the unproven empty launch path")
	}
}

// TestStrixHSACO_LaunchPathIsNotOperatorAsserted pins the security property that
// motivated #13026: a launch path string alone (no code object to validate) must
// never admit the kernel. This is the regression that keeps the fix honest.
func TestStrixHSACO_LaunchPathIsNotOperatorAsserted(t *testing.T) {
	// The strongest form: opt in AND assert the mechanism, but name no code object.
	// A bare launch-path string must still not admit, because nothing was validated.
	t.Setenv(EnvStrixWave32GEMVDecode, "1")
	t.Setenv(EnvStrixGEMVLaunchPath, LaunchPathHSACOAQL)
	t.Setenv(EnvStrixGEMVHSACOPath, "")

	toggle, kernel := ResolveStrixDecodeGEMVDevice()
	if toggle.Admitted || kernel.Available() {
		t.Fatalf("a bare launch-path string must not admit without a validated code object: %+v", toggle)
	}
	if toggle.LaunchPath != "" {
		t.Fatalf("refused toggle must carry no launch path, got %q", toggle.LaunchPath)
	}

	// And the refusal, surfaced through the device gate, is the fail-closed sentinel.
	if err := RequireDecodeGEMVDevice(toggle); !errors.Is(err, ErrWave32GEMVDecodeUnavailable) {
		t.Fatalf("want ErrWave32GEMVDecodeUnavailable, got %v", err)
	}
}

// TestStrixHSACO_OperatorPathEndToEnd drives the real operator entry point with a
// gfx1151 code object on disk: the object is read, validated, and admitted. This is
// the integration half of the witness — the validator alone is not proof the wiring
// works.
func TestStrixHSACO_OperatorPathEndToEnd(t *testing.T) {
	hsaco := buildGFX1151HSACO(t, EFAMDGPUMachGFX1151, ELFOSABIAMDGPUHSA, elf.ELFCLASS64, elf.EM_AMDGPU)

	path := filepath.Join(t.TempDir(), "strix_q2k_decode_gemv.hsaco")
	if err := os.WriteFile(path, hsaco, 0o644); err != nil {
		t.Fatalf("write code object: %v", err)
	}

	t.Setenv(EnvStrixWave32GEMVDecode, "1")
	t.Setenv(EnvStrixGEMVLaunchPath, LaunchPathHSACOAQL)
	t.Setenv(EnvStrixGEMVHSACOPath, path)

	toggle, kernel := ResolveStrixDecodeGEMVDevice()
	if !toggle.Admitted || !kernel.Available() {
		t.Fatalf("a validated on-disk gfx1151 code object must admit: %+v", toggle)
	}
	if toggle.LaunchPath != LaunchPathHSACOAQL {
		t.Fatalf("launch path %q != %q", toggle.LaunchPath, LaunchPathHSACOAQL)
	}
	if err := RequireDecodeGEMVDevice(toggle); err != nil {
		t.Fatalf("admitted toggle must pass the device gate: %v", err)
	}

	// The kernel now dispatches through the real path instead of refusing.
	raw := make([]byte, Q2KSuperBlockBytes)
	x := make([]float32, Q2KSuperBlockElems)
	if _, err := kernel.DispatchQ2KGEMV(raw, x, 1, Q2KSuperBlockElems); err != nil {
		t.Fatalf("admitted kernel must dispatch, got %v", err)
	}

	// A code object for the wrong target on disk is refused, not admitted.
	wrongPath := filepath.Join(t.TempDir(), "gfx1150.hsaco")
	wrong := buildGFX1151HSACO(t, EFAMDGPUMachGFX1150, ELFOSABIAMDGPUHSA, elf.ELFCLASS64, elf.EM_AMDGPU)
	if err := os.WriteFile(wrongPath, wrong, 0o644); err != nil {
		t.Fatalf("write wrong-target code object: %v", err)
	}
	t.Setenv(EnvStrixGEMVHSACOPath, wrongPath)
	toggle, kernel = ResolveStrixDecodeGEMVDevice()
	if toggle.Admitted || kernel.Available() {
		t.Fatal("a gfx1150 code object must not admit the gfx1151 kernel")
	}
}
