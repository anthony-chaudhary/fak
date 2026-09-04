package amdgpu

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"encoding/json"
	"strings"
	"testing"
	"unsafe"
)

// TestHSACOELFHeaders verifies that the generated HSACO ELF64 object contains
// valid ELF magic, 64-bit class, little-endian data, EM_AMDGPU machine (224),
// ELFOSABI_AMDGPU_HSA (64), and target machine flags (EF_AMDGPU_MACH_AMDGCN_GFX1151).
func TestHSACOELFHeaders(t *testing.T) {
	cfg := HSACOConfig{
		TargetArch: "gfx1151", // AMD Strix Halo / Radeon 8060S
		Version:    4,
		Kernels: []KernelConfig{
			{
				Name:          "gemm_wmma_f16",
				WavefrontSize: 32, // RDNA 3.5 Wave32
				SGPRCount:     32,
				VGPRCount:     64,
				Code:          []byte{0x00, 0x00, 0x81, 0xBF}, // s_endpgm
			},
		},
	}

	hsacoBytes, err := BuildHSACO(cfg)
	if err != nil {
		t.Fatalf("BuildHSACO failed: %v", err)
	}

	// Parse with standard library debug/elf
	f, err := elf.NewFile(bytes.NewReader(hsacoBytes))
	if err != nil {
		t.Fatalf("elf.NewFile failed to parse generated HSACO: %v", err)
	}
	defer f.Close()

	// 1. Validate ELF magic & identification
	if f.Class != elf.ELFCLASS64 {
		t.Errorf("ELF Class = %v, want ELFCLASS64", f.Class)
	}
	if f.Data != elf.ELFDATA2LSB {
		t.Errorf("ELF Data = %v, want ELFDATA2LSB", f.Data)
	}
	if f.Version != elf.EV_CURRENT {
		t.Errorf("ELF Version = %v, want EV_CURRENT", f.Version)
	}
	if f.OSABI != elf.OSABI(ELFOSABI_AMDGPU_HSA) {
		t.Errorf("ELF OSABI = %d, want ELFOSABI_AMDGPU_HSA (%d)", f.OSABI, ELFOSABI_AMDGPU_HSA)
	}
	if f.ABIVersion != ELFABIVERSION_AMDGPU_HSA_V4 {
		t.Errorf("ELF ABIVersion = %d, want %d", f.ABIVersion, ELFABIVERSION_AMDGPU_HSA_V4)
	}

	// 2. Validate Type, Machine, and Target Flags
	if f.Type != elf.ET_DYN {
		t.Errorf("ELF Type = %v, want ET_DYN", f.Type)
	}
	if f.Machine != elf.Machine(EM_AMDGPU) {
		t.Errorf("ELF Machine = %d, want EM_AMDGPU (%d)", f.Machine, EM_AMDGPU)
	}

	// In ELF64, e_flags is located at offset 48..52 of the ELF header
	eFlags := binary.LittleEndian.Uint32(hsacoBytes[48:52])
	if eFlags != EF_AMDGPU_MACH_AMDGCN_GFX1151 {
		t.Errorf("ELF Flags = 0x%08X, want EF_AMDGPU_MACH_AMDGCN_GFX1151 (0x%08X)", eFlags, EF_AMDGPU_MACH_AMDGCN_GFX1151)
	}

	// 3. Test Version 5 generation
	cfgV5 := cfg
	cfgV5.Version = 5
	hsacoV5Bytes, err := BuildHSACO(cfgV5)
	if err != nil {
		t.Fatalf("BuildHSACO (v5) failed: %v", err)
	}
	fV5, err := elf.NewFile(bytes.NewReader(hsacoV5Bytes))
	if err != nil {
		t.Fatalf("elf.NewFile (v5) failed: %v", err)
	}
	defer fV5.Close()

	if fV5.ABIVersion != ELFABIVERSION_AMDGPU_HSA_V5 {
		t.Errorf("ELF v5 ABIVersion = %d, want %d", fV5.ABIVersion, ELFABIVERSION_AMDGPU_HSA_V5)
	}
}

// TestHSACOSectionHeaders validates that the expected sections (.text, .rodata,
// .note.amdgpu, .symtab, .strtab, .shstrtab) are emitted with proper flags and symbols.
func TestHSACOSectionHeaders(t *testing.T) {
	cfg := HSACOConfig{
		TargetArch: "gfx1151",
		Version:    4,
		Kernels: []KernelConfig{
			{
				Name:                    "vec_add_kernel",
				Symbol:                  "vec_add_kernel.kd",
				GroupSegmentFixedSize:   2048,
				PrivateSegmentFixedSize: 128,
				KernargSegmentSize:      32,
				WavefrontSize:           32,
				SGPRCount:               32,
				VGPRCount:               32,
				Code:                    []byte{0x00, 0x00, 0x81, 0xBF}, // s_endpgm
			},
		},
	}

	hsacoBytes, err := BuildHSACO(cfg)
	if err != nil {
		t.Fatalf("BuildHSACO failed: %v", err)
	}

	f, err := elf.NewFile(bytes.NewReader(hsacoBytes))
	if err != nil {
		t.Fatalf("elf.NewFile failed: %v", err)
	}
	defer f.Close()

	// Verify required sections exist
	textSec := f.Section(".text")
	if textSec == nil {
		t.Fatal("missing .text section in HSACO")
	}
	if textSec.Flags&(elf.SHF_ALLOC|elf.SHF_EXECINSTR) != (elf.SHF_ALLOC | elf.SHF_EXECINSTR) {
		t.Errorf(".text section flags = 0x%X, want SHF_ALLOC|SHF_EXECINSTR", textSec.Flags)
	}

	rodataSec := f.Section(".rodata")
	if rodataSec == nil {
		t.Fatal("missing .rodata section in HSACO")
	}
	if rodataSec.Flags&elf.SHF_ALLOC != elf.SHF_ALLOC {
		t.Errorf(".rodata section flags = 0x%X, want SHF_ALLOC", rodataSec.Flags)
	}
	if rodataSec.Size < 64 {
		t.Errorf(".rodata section size = %d, expected at least 64 bytes for kernel descriptor", rodataSec.Size)
	}

	noteSec := f.Section(".note.amdgpu.v4")
	if noteSec == nil {
		t.Fatal("missing .note.amdgpu.v4 section in HSACO")
	}
	if noteSec.Flags&elf.SHF_ALLOC != elf.SHF_ALLOC {
		t.Errorf(".note.amdgpu.v4 section flags = 0x%X, want SHF_ALLOC", noteSec.Flags)
	}

	symtabSec := f.Section(".symtab")
	if symtabSec == nil {
		t.Fatal("missing .symtab section in HSACO")
	}

	strtabSec := f.Section(".strtab")
	if strtabSec == nil {
		t.Fatal("missing .strtab section in HSACO")
	}

	// Verify symbol table contains kernel entry and descriptor symbols
	syms, err := f.Symbols()
	if err != nil {
		t.Fatalf("f.Symbols failed: %v", err)
	}

	var foundFunc, foundDescriptor bool
	for _, sym := range syms {
		if sym.Name == "vec_add_kernel" {
			foundFunc = true
			if elf.ST_TYPE(sym.Info) != elf.STT_FUNC {
				t.Errorf("vec_add_kernel symbol type = %v, want STT_FUNC", elf.ST_TYPE(sym.Info))
			}
			if elf.ST_BIND(sym.Info) != elf.STB_GLOBAL {
				t.Errorf("vec_add_kernel symbol bind = %v, want STB_GLOBAL", elf.ST_BIND(sym.Info))
			}
		}
		if sym.Name == "vec_add_kernel.kd" {
			foundDescriptor = true
			if elf.ST_TYPE(sym.Info) != elf.STT_OBJECT {
				t.Errorf("vec_add_kernel.kd symbol type = %v, want STT_OBJECT", elf.ST_TYPE(sym.Info))
			}
			if sym.Size != 64 {
				t.Errorf("vec_add_kernel.kd symbol size = %d, want 64", sym.Size)
			}
		}
	}

	if !foundFunc {
		t.Error("expected global STT_FUNC symbol 'vec_add_kernel' not found")
	}
	if !foundDescriptor {
		t.Error("expected global STT_OBJECT symbol 'vec_add_kernel.kd' not found")
	}
}

// TestHSACOMetaDataAndGfx1151Registers validates the metadata JSON emitted into
// the .note.amdgpu note, checking kernel arguments, workgroup sizes, and RDNA 3.5 (gfx1151)
// register configuration (Wavefront size 32, WGP mode, granulated SGPR/VGPR).
func TestHSACOMetaDataAndGfx1151Registers(t *testing.T) {
	cfg := HSACOConfig{
		TargetArch: "gfx1151",
		Version:    4,
		Kernels: []KernelConfig{
			{
				Name:                    "attention_prefill_wave32",
				Symbol:                  "attention_prefill_wave32.kd",
				Language:                "HIP",
				LanguageVersion:         []int{2, 0},
				GroupSegmentFixedSize:   16384, // 16 KB LDS
				PrivateSegmentFixedSize: 512,
				KernargSegmentSize:      64,
				KernargSegmentAlign:     16,
				WavefrontSize:           32, // Wave32 for gfx1151
				SGPRCount:               48,
				VGPRCount:               96,
				MaxFlatWorkgroupSize:    512,
				Args: []KernelArgConfig{
					{
						Name:         "q_ptr",
						TypeName:     "half*",
						Size:         8,
						Offset:       0,
						ValueKind:    "global_buffer",
						AddressSpace: "global",
					},
					{
						Name:         "k_ptr",
						TypeName:     "half*",
						Size:         8,
						Offset:       8,
						ValueKind:    "global_buffer",
						AddressSpace: "global",
					},
					{
						Name:         "seq_len",
						TypeName:     "int",
						Size:         4,
						Offset:       16,
						ValueKind:    "by_value",
						AddressSpace: "",
					},
				},
				Code: []byte{0x00, 0x00, 0x81, 0xBF},
			},
		},
	}

	hsacoBytes, err := BuildHSACO(cfg)
	if err != nil {
		t.Fatalf("BuildHSACO failed: %v", err)
	}

	f, err := elf.NewFile(bytes.NewReader(hsacoBytes))
	if err != nil {
		t.Fatalf("elf.NewFile failed: %v", err)
	}
	defer f.Close()

	noteSec := f.Section(".note.amdgpu.v4")
	if noteSec == nil {
		t.Fatal(".note.amdgpu.v4 section not found")
	}

	noteBytes, err := noteSec.Data()
	if err != nil {
		t.Fatalf("reading note section data failed: %v", err)
	}

	// Parse Elf64_Nhdr: namesz (4), descsz (4), type (4)
	if len(noteBytes) < 12 {
		t.Fatalf("note data too short (%d bytes)", len(noteBytes))
	}
	nameSz := binary.LittleEndian.Uint32(noteBytes[0:4])
	descSz := binary.LittleEndian.Uint32(noteBytes[4:8])
	nType := binary.LittleEndian.Uint32(noteBytes[8:12])

	if nType != NT_AMDGPU_METADATA {
		t.Errorf("note type = %d, want NT_AMDGPU_METADATA (%d)", nType, NT_AMDGPU_METADATA)
	}

	namePadded := (nameSz + 3) &^ 3
	noteName := string(noteBytes[12 : 12+nameSz-1]) // strip null terminator
	if noteName != "AMDGPU" {
		t.Errorf("note name = %q, want 'AMDGPU'", noteName)
	}

	metaJSONBytes := noteBytes[12+namePadded : 12+namePadded+descSz]

	var meta AMDGPUMetadata
	if err := json.Unmarshal(metaJSONBytes, &meta); err != nil {
		t.Fatalf("unmarshalling metadata JSON failed: %v\nJSON:\n%s", err, string(metaJSONBytes))
	}

	// Verify target triple
	if meta.Target != "amdgcn-amd-amdhsa--gfx1151" {
		t.Errorf("metadata target = %q, want 'amdgcn-amd-amdhsa--gfx1151'", meta.Target)
	}
	if len(meta.Kernels) != 1 {
		t.Fatalf("metadata kernels count = %d, want 1", len(meta.Kernels))
	}

	km := meta.Kernels[0]
	if km.Name != "attention_prefill_wave32" {
		t.Errorf("kernel name = %q, want 'attention_prefill_wave32'", km.Name)
	}
	if km.WavefrontSize != 32 {
		t.Errorf("kernel wavefront_size = %d, want 32 (Strix Halo Wave32)", km.WavefrontSize)
	}
	if km.SGPRCount != 48 || km.VGPRCount != 96 {
		t.Errorf("kernel registers: SGPR=%d, VGPR=%d; want 48, 96", km.SGPRCount, km.VGPRCount)
	}
	if km.GroupSegmentFixedSize != 16384 {
		t.Errorf("kernel LDS = %d, want 16384", km.GroupSegmentFixedSize)
	}
	if len(km.Args) != 3 {
		t.Fatalf("kernel args count = %d, want 3", len(km.Args))
	}
	if km.Args[0].Name != "q_ptr" || km.Args[0].ValueKind != "global_buffer" {
		t.Errorf("arg 0 mismatch: %+v", km.Args[0])
	}
	if km.Args[2].Name != "seq_len" || km.Args[2].ValueKind != "by_value" {
		t.Errorf("arg 2 mismatch: %+v", km.Args[2])
	}

	// Test Register Allocation Helper Functions for gfx1151
	rsrc1 := ComputePgmRsrc1(96, 48, 32)
	// VGPR granulated: (96-1)/8 = 11 (0x0B) in bits 5..0
	// SGPR granulated: (48-1)/8 = 5 (0x05) in bits 9..6
	// WGP mode bit: 1 << 19 for Wave32
	if (rsrc1 & 0x3F) != 11 {
		t.Errorf("rsrc1 granulated VGPR = %d, want 11", rsrc1&0x3F)
	}
	if ((rsrc1 >> 6) & 0x0F) != 5 {
		t.Errorf("rsrc1 granulated SGPR = %d, want 5", (rsrc1>>6)&0x0F)
	}
	if (rsrc1 & (1 << 19)) == 0 {
		t.Error("rsrc1 WGP_MODE bit (1 << 19) not set for WavefrontSize 32 on gfx1151")
	}

	rsrc2 := ComputePgmRsrc2(2, 16384)
	// LDS granulated in 256-byte blocks: 16384 / 256 = 64
	ldsBlocks := (rsrc2 >> 15) & 0x1FF
	if ldsBlocks != 64 {
		t.Errorf("rsrc2 LDS blocks = %d, want 64", ldsBlocks)
	}
}

// TestHSACOMsgPackFormat verifies pure-Go MessagePack metadata serialization.
func TestHSACOMsgPackFormat(t *testing.T) {
	cfg := HSACOConfig{
		TargetArch: "gfx1151",
		Version:    4,
		Format:     "msgpack",
		Kernels: []KernelConfig{
			{
				Name:          "msgpack_kernel",
				WavefrontSize: 32,
				SGPRCount:     32,
				VGPRCount:     32,
				Args: []KernelArgConfig{
					{Name: "data", Size: 8, Offset: 0, ValueKind: "global_buffer"},
				},
				Code: []byte{0x00, 0x00, 0x81, 0xBF},
			},
		},
	}

	hsacoBytes, err := BuildHSACO(cfg)
	if err != nil {
		t.Fatalf("BuildHSACO with msgpack failed: %v", err)
	}

	f, err := elf.NewFile(bytes.NewReader(hsacoBytes))
	if err != nil {
		t.Fatalf("elf.NewFile failed: %v", err)
	}
	defer f.Close()

	noteSec := f.Section(".note.amdgpu.v4")
	if noteSec == nil {
		t.Fatal(".note.amdgpu.v4 section missing")
	}
	noteData, err := noteSec.Data()
	if err != nil {
		t.Fatalf("failed to read note data: %v", err)
	}

	// Payload starts after 12-byte header + 8-byte padded name "AMDGPU\0\0"
	msgPackPayload := noteData[20:]
	if len(msgPackPayload) == 0 {
		t.Fatal("empty MessagePack payload in note")
	}
	// A MessagePack map with 3 keys (amdhsa.version, amdhsa.target, amdhsa.kernels) starts with 0x83
	if msgPackPayload[0] != 0x83 {
		t.Errorf("MessagePack header byte = 0x%02X, want 0x83 (fixmap 3)", msgPackPayload[0])
	}
	// Verify target string is embedded in the binary
	if !strings.Contains(string(msgPackPayload), "gfx1151") {
		t.Error("MessagePack payload missing 'gfx1151' target string")
	}
}

// TestAMDGPUKernelCodeT verifies the 256-byte amdgpu_kernel_code_t descriptor structure.
func TestAMDGPUKernelCodeT(t *testing.T) {
	if sz := unsafe.Sizeof(AMDGPUKernelCodeT{}); sz != 256 {
		t.Fatalf("AMDGPUKernelCodeT struct size = %d, want 256", sz)
	}

	kd := AMDGPUKernelCodeT{
		KernelCodeVersionMajor:        1,
		KernelCodeVersionMinor:        2,
		MachineKind:                   1,
		Profile:                       0,
		KernelCodeProperties:          0x100,
		ComputePgmRsrc1:               0x002C014B,
		ComputePgmRsrc2:               0x00000382,
		KernelCodeEntryByteOffset:     256,
		WorkgroupGroupSegmentByteSize: 4096,
		KernargSegmentByteSize:        32,
		WavefrontSize:                 32,
	}

	b256 := kd.Marshal256()
	if len(b256) != 256 {
		t.Fatalf("Marshal256 returned %d bytes, want 256", len(b256))
	}
	bytesSlice := kd.Bytes()
	if len(bytesSlice) != 256 {
		t.Fatalf("Bytes returned %d bytes, want 256", len(bytesSlice))
	}

	// Verify decoded field positions
	if binary.LittleEndian.Uint32(bytesSlice[0:4]) != 1 {
		t.Errorf("major version = %d, want 1", binary.LittleEndian.Uint32(bytesSlice[0:4]))
	}
	if binary.LittleEndian.Uint32(bytesSlice[4:8]) != 2 {
		t.Errorf("minor version = %d, want 2", binary.LittleEndian.Uint32(bytesSlice[4:8]))
	}
	entryOffset := int64(binary.LittleEndian.Uint64(bytesSlice[24:32]))
	if entryOffset != 256 {
		t.Errorf("entry byte offset = %d, want 256", entryOffset)
	}
	if binary.LittleEndian.Uint32(bytesSlice[52:56]) != 4096 {
		t.Errorf("LDS size = %d, want 4096", binary.LittleEndian.Uint32(bytesSlice[52:56]))
	}
	if binary.LittleEndian.Uint64(bytesSlice[64:72]) != 32 {
		t.Errorf("kernarg size = %d, want 32", binary.LittleEndian.Uint64(bytesSlice[64:72]))
	}
	if binary.LittleEndian.Uint32(bytesSlice[76:80]) != 32 {
		t.Errorf("wavefront size = %d, want 32", binary.LittleEndian.Uint32(bytesSlice[76:80]))
	}
}

// TestHSACOMultipleKernels tests multi-kernel code object emission.
func TestHSACOMultipleKernels(t *testing.T) {
	cfg := HSACOConfig{
		TargetArch: "gfx1151",
		Kernels: []KernelConfig{
			{
				Name:          "kernel_alpha",
				WavefrontSize: 32,
				SGPRCount:     32,
				VGPRCount:     32,
				Code:          []byte{0x00, 0x00, 0x81, 0xBF},
			},
			{
				Name:          "kernel_beta",
				WavefrontSize: 32,
				SGPRCount:     64,
				VGPRCount:     128,
				Code:          []byte{0x00, 0x00, 0x81, 0xBF, 0x00, 0x00, 0x81, 0xBF},
			},
		},
	}

	hsacoBytes, err := BuildHSACO(cfg)
	if err != nil {
		t.Fatalf("BuildHSACO with multiple kernels failed: %v", err)
	}

	f, err := elf.NewFile(bytes.NewReader(hsacoBytes))
	if err != nil {
		t.Fatalf("elf.NewFile failed: %v", err)
	}
	defer f.Close()

	syms, err := f.Symbols()
	if err != nil {
		t.Fatalf("reading symbols failed: %v", err)
	}

	names := make(map[string]bool)
	for _, s := range syms {
		names[s.Name] = true
	}

	expected := []string{"kernel_alpha", "kernel_alpha.kd", "kernel_beta", "kernel_beta.kd"}
	for _, exp := range expected {
		if !names[exp] {
			t.Errorf("expected symbol %q not found in symbol table", exp)
		}
	}
}
