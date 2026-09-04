// Package amdgpu provides AMD GPU facts probing, hardware governor settings,
// Strix Halo APU operational serving profiles, direct AQL/PM4 packet dispatch,
// and native HSACO code-object emission.
package amdgpu

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unsafe"
)

// ELF and AMDGPU constants matching the AMD HSA Code Object ABI specification.
const (
	// ELF Identification indices and values.
	EI_MAG0       = 0
	EI_MAG1       = 1
	EI_MAG2       = 2
	EI_MAG3       = 3
	EI_CLASS      = 4
	EI_DATA       = 5
	EI_VERSION    = 6
	EI_OSABI      = 7
	EI_ABIVERSION = 8
	EI_NIDENT     = 16

	ELFMAG0 = 0x7f
	ELFMAG1 = 'E'
	ELFMAG2 = 'L'
	ELFMAG3 = 'F'

	ELFCLASS64  = 2 // 64-bit architecture
	ELFDATA2LSB = 1 // 2's complement, little endian
	EV_CURRENT  = 1 // Current ELF version

	// AMDGPU ELF OSABI and Machine constants.
	ELFOSABI_AMDGPU_HSA = 64  // AMDGPU HSA OSABI (0x40)
	EM_AMDGPU           = 224 // AMDGPU ELF machine architecture (0x00E0)

	// AMDGPU HSA Code Object ABI versions.
	ELFABIVERSION_AMDGPU_HSA_V4 = 1 // Code object v4
	ELFABIVERSION_AMDGPU_HSA_V5 = 2 // Code object v5

	// ELF file types.
	ET_DYN = 3 // Shared object / relocatable code object

	// AMDGPU Architecture Target Machine IDs (EF_AMDGPU_MACH).
	// Stored in bits 7..0 of e_flags in the ELF header.
	EF_AMDGPU_MACH_AMDGCN_GFX942  uint32 = 0x048 // MI300 (CDNA 3)
	EF_AMDGPU_MACH_AMDGCN_GFX1100 uint32 = 0x041 // Navi 31 (RDNA 3)
	EF_AMDGPU_MACH_AMDGCN_GFX1150 uint32 = 0x049 // Strix Point (RDNA 3.5)
	EF_AMDGPU_MACH_AMDGCN_GFX1151 uint32 = 0x04a // Strix Halo / Radeon 8060S (RDNA 3.5)
	EF_AMDGPU_MACH_AMDGCN_GFX1200 uint32 = 0x04e // Navi 41 (RDNA 4)
	EF_AMDGPU_MACH_AMDGCN_GFX1201 uint32 = 0x04f // Navi 44 (RDNA 4)

	// Section header types.
	SHT_NULL     = 0
	SHT_PROGBITS = 1
	SHT_SYMTAB   = 2
	SHT_STRTAB   = 3
	SHT_NOTE     = 7

	// Section header flags.
	SHF_WRITE     uint64 = 0x1
	SHF_ALLOC     uint64 = 0x2
	SHF_EXECINSTR uint64 = 0x4

	// Program header types and flags.
	PT_LOAD = 1
	PT_NOTE = 4

	PF_X uint32 = 0x1
	PF_W uint32 = 0x2
	PF_R uint32 = 0x4

	// Symbol bindings and types.
	STB_LOCAL  = 0
	STB_GLOBAL = 1

	STT_NOTYPE  = 0
	STT_OBJECT  = 1
	STT_FUNC    = 2
	STT_SECTION = 3

	// ELF Note types for AMDGPU.
	NT_AMDGPU_METADATA = 32 // Code object metadata note descriptor
)

// TargetArchFlags maps an AMD target architecture string (e.g. "gfx1151") to its ELF
// machine flags and AMDHSA target triple string.
func TargetArchFlags(target string) (uint32, string, error) {
	norm := strings.ToLower(strings.TrimSpace(target))
	switch norm {
	case "gfx1151", "strix-halo", "strix_halo", "8060s":
		return EF_AMDGPU_MACH_AMDGCN_GFX1151, "amdgcn-amd-amdhsa--gfx1151", nil
	case "gfx1150", "strix-point", "890m":
		return EF_AMDGPU_MACH_AMDGCN_GFX1150, "amdgcn-amd-amdhsa--gfx1150", nil
	case "gfx1100", "rx7900xtx", "navi31":
		return EF_AMDGPU_MACH_AMDGCN_GFX1100, "amdgcn-amd-amdhsa--gfx1100", nil
	case "gfx942", "mi300", "mi300x":
		return EF_AMDGPU_MACH_AMDGCN_GFX942, "amdgcn-amd-amdhsa--gfx942", nil
	case "gfx1200":
		return EF_AMDGPU_MACH_AMDGCN_GFX1200, "amdgcn-amd-amdhsa--gfx1200", nil
	case "gfx1201":
		return EF_AMDGPU_MACH_AMDGCN_GFX1201, "amdgcn-amd-amdhsa--gfx1201", nil
	case "":
		// Default to Strix Halo (gfx1151)
		return EF_AMDGPU_MACH_AMDGCN_GFX1151, "amdgcn-amd-amdhsa--gfx1151", nil
	default:
		return 0, "", fmt.Errorf("amdgpu: unsupported target architecture %q", target)
	}
}

// ComputePgmRsrc1 computes COMPUTE_PGM_RSRC1 register configuration for AMD GPUs.
// Handles granulated VGPR/SGPR allocations, floating-point IEEE mode, and Wave32/WGP flags.
func ComputePgmRsrc1(vgprs, sgprs, wavefrontSize uint32) uint32 {
	var vgprGran uint32
	if vgprs > 0 {
		vgprGran = (vgprs - 1) / 8
	}
	var sgprGran uint32
	if sgprs > 0 {
		sgprGran = (sgprs - 1) / 8
	}

	rsrc1 := (vgprGran & 0x3F) | ((sgprGran & 0x0F) << 6)
	// Default: IEEE float mode (round-to-nearest-even, denormals enabled)
	rsrc1 |= (0xC0 << 12) | (1 << 17) // FLOAT_MODE + IEEE_MODE

	// On RDNA 3.5 / gfx1151, WavefrontSize 32 runs in WGP mode (1 << 19)
	if wavefrontSize == 32 {
		rsrc1 |= (1 << 19) // WGP_MODE
	}
	return rsrc1
}

// ComputePgmRsrc2 computes COMPUTE_PGM_RSRC2 register configuration for AMD GPUs.
// Configures user SGPR parameters, threadgroup ID enables, and LDS byte allocation.
func ComputePgmRsrc2(userSGPRs, ldsBytes uint32) uint32 {
	rsrc2 := userSGPRs & 0x3F
	// Enable TGID_X, TGID_Y, TGID_Z workgroup dimension registers
	rsrc2 |= (1 << 7) | (1 << 8) | (1 << 9)
	// LDS size granulated in 256-byte blocks in bits 23:15
	if ldsBytes > 0 {
		ldsGran := (ldsBytes + 255) / 256
		rsrc2 |= (ldsGran & 0x1FF) << 15
	}
	return rsrc2
}

// AMDGPUKernelCodeT represents the 256-byte amdgpu_kernel_code_t descriptor
// defined by the AMD HSA ABI for GPU kernel execution.
type AMDGPUKernelCodeT struct {
	KernelCodeVersionMajor        uint32    // 0..3: Architecture major version
	KernelCodeVersionMinor        uint32    // 4..7: Architecture minor version
	MachineKind                   uint16    // 8..9: Machine kind / target architecture
	Profile                       uint16    // 10..11: Profile (BASE / FULL)
	KernelCodeProperties          uint32    // 12..15: Kernel properties bitfield
	ComputePgmRsrc1               uint32    // 16..19: COMPUTE_PGM_RSRC1 register bits
	ComputePgmRsrc2               uint32    // 20..23: COMPUTE_PGM_RSRC2 register bits
	KernelCodeEntryByteOffset     int64     // 24..31: Relative offset to kernel instructions
	Reserved1                     [20]byte  // 32..51: Reserved
	WorkgroupGroupSegmentByteSize uint32    // 52..55: LDS byte size
	GDSByteSize                   uint32    // 56..59: Global Data Share size
	KernargSegmentAlignment       uint32    // 60..63: Argument buffer alignment
	KernargSegmentByteSize        uint64    // 64..71: Kernel argument buffer size
	WorkitemPrivateSegmentByteSz  uint32    // 72..75: Per-thread scratch size
	WavefrontSize                 uint32    // 76..79: 32 for RDNA 3.5, 64 for CDNA
	Reserved2                     [176]byte // 80..255: Padding to exact 256 bytes
}

// Ensure struct size is statically 256 bytes.
var _ [256]byte = [unsafe.Sizeof(AMDGPUKernelCodeT{})]byte{}

// Marshal256 serializes AMDGPUKernelCodeT into an exact 256-byte little-endian array.
func (k *AMDGPUKernelCodeT) Marshal256() [256]byte {
	var buf [256]byte
	binary.LittleEndian.PutUint32(buf[0:4], k.KernelCodeVersionMajor)
	binary.LittleEndian.PutUint32(buf[4:8], k.KernelCodeVersionMinor)
	binary.LittleEndian.PutUint16(buf[8:10], k.MachineKind)
	binary.LittleEndian.PutUint16(buf[10:12], k.Profile)
	binary.LittleEndian.PutUint32(buf[12:16], k.KernelCodeProperties)
	binary.LittleEndian.PutUint32(buf[16:20], k.ComputePgmRsrc1)
	binary.LittleEndian.PutUint32(buf[20:24], k.ComputePgmRsrc2)
	binary.LittleEndian.PutUint64(buf[24:32], uint64(k.KernelCodeEntryByteOffset))
	copy(buf[32:52], k.Reserved1[:])
	binary.LittleEndian.PutUint32(buf[52:56], k.WorkgroupGroupSegmentByteSize)
	binary.LittleEndian.PutUint32(buf[56:60], k.GDSByteSize)
	binary.LittleEndian.PutUint32(buf[60:64], k.KernargSegmentAlignment)
	binary.LittleEndian.PutUint64(buf[64:72], k.KernargSegmentByteSize)
	binary.LittleEndian.PutUint32(buf[72:76], k.WorkitemPrivateSegmentByteSz)
	binary.LittleEndian.PutUint32(buf[76:80], k.WavefrontSize)
	copy(buf[80:256], k.Reserved2[:])
	return buf
}

// Bytes returns the serialized 256-byte slice.
func (k *AMDGPUKernelCodeT) Bytes() []byte {
	arr := k.Marshal256()
	out := make([]byte, 256)
	copy(out, arr[:])
	return out
}

// AMDGPUKernelDescriptor represents the 64-byte compact kernel descriptor
// used in AMDGPU Code Object v3, v4, and v5.
type AMDGPUKernelDescriptor struct {
	GroupSegmentFixedSize     uint32   // 0..3: LDS fixed allocation in bytes
	PrivateSegmentFixedSize   uint32   // 4..7: Scratch memory allocation per workitem
	KernargSize               uint32   // 8..11: Argument buffer size in bytes
	Reserved0                 uint32   // 12..15: Reserved
	KernelCodeEntryByteOffset int64    // 16..23: Relative offset to instruction stream
	Reserved1                 [20]byte // 24..43: Reserved
	ComputePgmRsrc3           uint32   // 44..47: COMPUTE_PGM_RSRC3
	ComputePgmRsrc1           uint32   // 48..51: COMPUTE_PGM_RSRC1
	ComputePgmRsrc2           uint32   // 52..55: COMPUTE_PGM_RSRC2
	KernelCodeProperties      uint16   // 56..57: Kernel properties
	Reserved2                 [6]byte  // 58..63: Padding to 64 bytes
}

// Ensure struct size is statically 64 bytes.
var _ [64]byte = [unsafe.Sizeof(AMDGPUKernelDescriptor{})]byte{}

// Marshal64 serializes AMDGPUKernelDescriptor into an exact 64-byte array.
func (kd *AMDGPUKernelDescriptor) Marshal64() [64]byte {
	var buf [64]byte
	binary.LittleEndian.PutUint32(buf[0:4], kd.GroupSegmentFixedSize)
	binary.LittleEndian.PutUint32(buf[4:8], kd.PrivateSegmentFixedSize)
	binary.LittleEndian.PutUint32(buf[8:12], kd.KernargSize)
	binary.LittleEndian.PutUint32(buf[12:16], kd.Reserved0)
	binary.LittleEndian.PutUint64(buf[16:24], uint64(kd.KernelCodeEntryByteOffset))
	copy(buf[24:44], kd.Reserved1[:])
	binary.LittleEndian.PutUint32(buf[44:48], kd.ComputePgmRsrc3)
	binary.LittleEndian.PutUint32(buf[48:52], kd.ComputePgmRsrc1)
	binary.LittleEndian.PutUint32(buf[52:56], kd.ComputePgmRsrc2)
	binary.LittleEndian.PutUint16(buf[56:58], kd.KernelCodeProperties)
	copy(buf[58:64], kd.Reserved2[:])
	return buf
}

// Bytes returns the serialized 64-byte slice.
func (kd *AMDGPUKernelDescriptor) Bytes() []byte {
	arr := kd.Marshal64()
	out := make([]byte, 64)
	copy(out, arr[:])
	return out
}

// KernelArgConfig specifies the layout of an argument accepted by the kernel.
type KernelArgConfig struct {
	Name         string `json:".name,omitempty"`
	TypeName     string `json:".type_name,omitempty"`
	Size         uint32 `json:".size"`
	Offset       uint32 `json:".offset"`
	ValueKind    string `json:".value_kind"`              // "global_buffer", "by_value", etc.
	AddressSpace string `json:".address_space,omitempty"` // "global", "local", "generic"
}

// KernelConfig describes an AMD GPU kernel to be assembled into the code object.
type KernelConfig struct {
	Name                    string            // Kernel function name (e.g. "gemm_wmma")
	Symbol                  string            // Symbol name; defaults to Name + ".kd"
	Language                string            // Source language (e.g. "OpenCL C", "HIP")
	LanguageVersion         []int             // Language version tuple (e.g. [2, 0])
	Args                    []KernelArgConfig // Kernel argument list
	GroupSegmentFixedSize   uint32            // Local Data Share (LDS) bytes
	PrivateSegmentFixedSize uint32            // Scratch bytes per workitem
	KernargSegmentSize      uint32            // Total argument segment bytes
	KernargSegmentAlign     uint32            // Argument alignment requirement (default 8)
	WavefrontSize           uint32            // 32 for RDNA 3.5 / gfx1151; 64 for CDNA
	SGPRCount               uint32            // Scalar registers required (e.g. 32)
	VGPRCount               uint32            // Vector registers required (e.g. 32)
	AGPRCount               uint32            // Accumulator registers (0 for RDNA)
	MaxFlatWorkgroupSize    uint32            // Max workitems per workgroup (e.g. 256)
	Code                    []byte            // GPU machine instructions (ISA)
}

// HSACOConfig holds the complete configuration for building a standalone HSACO ELF64 file.
type HSACOConfig struct {
	TargetArch string         // Architecture identifier (e.g. "gfx1151", "gfx942")
	Version    int            // Code object ABI version (4 or 5; default 4)
	Kernels    []KernelConfig // One or more kernel definitions
	Rodata     []byte         // Optional extra read-only data
	Format     string         // Metadata format: "json" (default) or "msgpack"
}

// AMDGPUMetadata represents the root structure of the .note.amdgpu code-object metadata.
type AMDGPUMetadata struct {
	Version []int                `json:"amdhsa.version"`
	Target  string               `json:"amdhsa.target"`
	Kernels []KernelMetadataJSON `json:"amdhsa.kernels"`
}

// KernelMetadataJSON encodes the metadata entry for a single kernel.
type KernelMetadataJSON struct {
	Name                    string          `json:".name"`
	Symbol                  string          `json:".symbol"`
	Language                string          `json:".language,omitempty"`
	LanguageVersion         []int           `json:".language_version,omitempty"`
	Args                    []KernelArgJSON `json:".args,omitempty"`
	GroupSegmentFixedSize   uint32          `json:".group_segment_fixed_size"`
	PrivateSegmentFixedSize uint32          `json:".private_segment_fixed_size"`
	KernargSegmentSize      uint32          `json:".kernarg_segment_size"`
	KernargSegmentAlign     uint32          `json:".kernarg_segment_align"`
	WavefrontSize           uint32          `json:".wavefront_size"`
	SGPRCount               uint32          `json:".sgpr_count"`
	VGPRCount               uint32          `json:".vgpr_count"`
	AGPRCount               uint32          `json:".agpr_count,omitempty"`
	MaxFlatWorkgroupSize    uint32          `json:".max_flat_workgroup_size"`
	WavefrontNumSGPR        uint32          `json:".wavefront_num_sgpr,omitempty"`
	WorkitemNumVGPR         uint32          `json:".workitem_num_vgpr,omitempty"`
}

// KernelArgJSON encodes the metadata entry for an individual kernel argument.
type KernelArgJSON struct {
	Name         string `json:".name,omitempty"`
	TypeName     string `json:".type_name,omitempty"`
	Size         uint32 `json:".size"`
	Offset       uint32 `json:".offset"`
	ValueKind    string `json:".value_kind"`
	AddressSpace string `json:".address_space,omitempty"`
}

// GenerateMetadataJSON generates formatted JSON bytes for the .note.amdgpu metadata.
func GenerateMetadataJSON(meta AMDGPUMetadata) ([]byte, error) {
	return json.MarshalIndent(meta, "", "  ")
}

// GenerateMetadataMsgPack generates pure-Go MessagePack bytes for the .note.amdgpu metadata.
func GenerateMetadataMsgPack(meta AMDGPUMetadata) ([]byte, error) {
	var buf bytes.Buffer
	m := map[string]any{
		"amdhsa.version": meta.Version,
		"amdhsa.target":  meta.Target,
	}
	var kernels []any
	for _, k := range meta.Kernels {
		km := map[string]any{
			".name":                       k.Name,
			".symbol":                     k.Symbol,
			".group_segment_fixed_size":   k.GroupSegmentFixedSize,
			".private_segment_fixed_size": k.PrivateSegmentFixedSize,
			".kernarg_segment_size":       k.KernargSegmentSize,
			".kernarg_segment_align":      k.KernargSegmentAlign,
			".wavefront_size":             k.WavefrontSize,
			".sgpr_count":                 k.SGPRCount,
			".vgpr_count":                 k.VGPRCount,
			".max_flat_workgroup_size":    k.MaxFlatWorkgroupSize,
		}
		if k.Language != "" {
			km[".language"] = k.Language
		}
		if len(k.LanguageVersion) > 0 {
			km[".language_version"] = k.LanguageVersion
		}
		if k.AGPRCount > 0 {
			km[".agpr_count"] = k.AGPRCount
		}
		if len(k.Args) > 0 {
			var args []any
			for _, a := range k.Args {
				am := map[string]any{
					".size":       a.Size,
					".offset":     a.Offset,
					".value_kind": a.ValueKind,
				}
				if a.Name != "" {
					am[".name"] = a.Name
				}
				if a.TypeName != "" {
					am[".type_name"] = a.TypeName
				}
				if a.AddressSpace != "" {
					am[".address_space"] = a.AddressSpace
				}
				args = append(args, am)
			}
			km[".args"] = args
		}
		kernels = append(kernels, km)
	}
	m["amdhsa.kernels"] = kernels

	if err := encodeMsgPackAny(&buf, m); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// encodeMsgPackAny is a lightweight pure-Go MessagePack serializer for primitive types, maps, and slices.
func encodeMsgPackAny(buf *bytes.Buffer, v any) error {
	if v == nil {
		buf.WriteByte(0xC0)
		return nil
	}
	switch val := v.(type) {
	case bool:
		if val {
			buf.WriteByte(0xC3)
		} else {
			buf.WriteByte(0xC2)
		}
	case int:
		encodeMsgPackInt(buf, int64(val))
	case int32:
		encodeMsgPackInt(buf, int64(val))
	case int64:
		encodeMsgPackInt(buf, val)
	case uint:
		encodeMsgPackUint(buf, uint64(val))
	case uint32:
		encodeMsgPackUint(buf, uint64(val))
	case uint64:
		encodeMsgPackUint(buf, val)
	case string:
		encodeMsgPackString(buf, val)
	case []int:
		encodeMsgPackArrayHeader(buf, len(val))
		for _, item := range val {
			encodeMsgPackInt(buf, int64(item))
		}
	case []any:
		encodeMsgPackArrayHeader(buf, len(val))
		for _, item := range val {
			if err := encodeMsgPackAny(buf, item); err != nil {
				return err
			}
		}
	case map[string]any:
		encodeMsgPackMapHeader(buf, len(val))
		for k, item := range val {
			encodeMsgPackString(buf, k)
			if err := encodeMsgPackAny(buf, item); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("amdgpu: unsupported MessagePack type %T", v)
	}
	return nil
}

func encodeMsgPackInt(buf *bytes.Buffer, val int64) {
	if val >= 0 && val <= 127 {
		buf.WriteByte(byte(val))
	} else if val >= -32 && val < 0 {
		buf.WriteByte(byte(val))
	} else if val >= -128 && val <= 127 {
		buf.WriteByte(0xD0)
		buf.WriteByte(byte(int8(val)))
	} else if val >= -32768 && val <= 32767 {
		buf.WriteByte(0xD1)
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], uint16(val))
		buf.Write(b[:])
	} else if val >= -2147483648 && val <= 2147483647 {
		buf.WriteByte(0xD2)
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], uint32(val))
		buf.Write(b[:])
	} else {
		buf.WriteByte(0xD3)
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], uint64(val))
		buf.Write(b[:])
	}
}

func encodeMsgPackUint(buf *bytes.Buffer, val uint64) {
	if val <= 127 {
		buf.WriteByte(byte(val))
	} else if val <= 0xFF {
		buf.WriteByte(0xCC)
		buf.WriteByte(byte(val))
	} else if val <= 0xFFFF {
		buf.WriteByte(0xCD)
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], uint16(val))
		buf.Write(b[:])
	} else if val <= 0xFFFFFFFF {
		buf.WriteByte(0xCE)
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], uint32(val))
		buf.Write(b[:])
	} else {
		buf.WriteByte(0xCF)
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], val)
		buf.Write(b[:])
	}
}

func encodeMsgPackString(buf *bytes.Buffer, s string) {
	length := len(s)
	if length <= 31 {
		buf.WriteByte(0xA0 | byte(length))
	} else if length <= 0xFF {
		buf.WriteByte(0xD9)
		buf.WriteByte(byte(length))
	} else if length <= 0xFFFF {
		buf.WriteByte(0xDA)
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], uint16(length))
		buf.Write(b[:])
	} else {
		buf.WriteByte(0xDB)
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], uint32(length))
		buf.Write(b[:])
	}
	buf.WriteString(s)
}

func encodeMsgPackArrayHeader(buf *bytes.Buffer, length int) {
	if length <= 15 {
		buf.WriteByte(0x90 | byte(length))
	} else if length <= 0xFFFF {
		buf.WriteByte(0xDC)
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], uint16(length))
		buf.Write(b[:])
	} else {
		buf.WriteByte(0xDD)
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], uint32(length))
		buf.Write(b[:])
	}
}

func encodeMsgPackMapHeader(buf *bytes.Buffer, length int) {
	if length <= 15 {
		buf.WriteByte(0x80 | byte(length))
	} else if length <= 0xFFFF {
		buf.WriteByte(0xDE)
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], uint16(length))
		buf.Write(b[:])
	} else {
		buf.WriteByte(0xDF)
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], uint32(length))
		buf.Write(b[:])
	}
}

// BuildAMDGPUELFNote constructs a standard 4-byte aligned ELF note structure with the
// "AMDGPU" name and NT_AMDGPU_METADATA type.
func BuildAMDGPUELFNote(payload []byte) []byte {
	const noteName = "AMDGPU\x00"
	nameBytes := []byte(noteName)              // length 7
	namePaddedLen := (len(nameBytes) + 3) &^ 3 // 8 bytes

	descLen := len(payload)
	descPaddedLen := (descLen + 3) &^ 3

	// Elf64_Nhdr: 3 * 4 bytes = 12 bytes
	totalLen := 12 + namePaddedLen + descPaddedLen
	buf := make([]byte, totalLen)

	binary.LittleEndian.PutUint32(buf[0:4], uint32(len(nameBytes)))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(descLen))
	binary.LittleEndian.PutUint32(buf[8:12], NT_AMDGPU_METADATA)

	copy(buf[12:12+len(nameBytes)], nameBytes)
	copy(buf[12+namePaddedLen:12+namePaddedLen+descLen], payload)

	return buf
}

// BuildHSACO generates a complete, valid AMDGPU HSACO ELF64LE shared code object.
// Pure-Go implementation without LLVM or external toolchain dependencies.
func BuildHSACO(cfg HSACOConfig) ([]byte, error) {
	if len(cfg.Kernels) == 0 {
		return nil, errors.New("amdgpu: at least one kernel is required to build HSACO")
	}

	eFlags, targetTriple, err := TargetArchFlags(cfg.TargetArch)
	if err != nil {
		return nil, err
	}

	version := cfg.Version
	if version == 0 {
		version = 4
	}
	if version != 4 && version != 5 {
		return nil, fmt.Errorf("amdgpu: unsupported HSACO version %d (must be 4 or 5)", version)
	}

	abiVersion := byte(ELFABIVERSION_AMDGPU_HSA_V4)
	versionTuple := []int{1, 1}
	noteSecName := ".note.amdgpu.v4"
	if version == 5 {
		abiVersion = byte(ELFABIVERSION_AMDGPU_HSA_V5)
		versionTuple = []int{1, 2}
		noteSecName = ".note.amdgpu.v5"
	}

	// 1. Prepare Kernel Code (.text) and Descriptors (.rodata)
	var textBuf bytes.Buffer
	var rodataBuf bytes.Buffer

	// If caller provided extra rodata, prepend it
	if len(cfg.Rodata) > 0 {
		rodataBuf.Write(cfg.Rodata)
		// Align to 64 bytes
		if rem := rodataBuf.Len() % 64; rem != 0 {
			rodataBuf.Write(make([]byte, 64-rem))
		}
	}

	type kernelPlacement struct {
		name         string
		symbol       string
		textOffset   uint64
		textSize     uint64
		rodataOffset uint64
	}
	placements := make([]kernelPlacement, len(cfg.Kernels))

	// Collect metadata entries
	metaKernels := make([]KernelMetadataJSON, len(cfg.Kernels))

	for i, k := range cfg.Kernels {
		kName := k.Name
		if kName == "" {
			kName = fmt.Sprintf("kernel_%d", i)
		}
		kSymbol := k.Symbol
		if kSymbol == "" {
			kSymbol = kName + ".kd"
		}

		// Align text to 256 bytes for instruction cache fetch
		if rem := textBuf.Len() % 256; rem != 0 {
			textBuf.Write(make([]byte, 256-rem))
		}
		textOffset := uint64(textBuf.Len())

		// Emit code instructions
		code := k.Code
		if len(code) == 0 {
			// Minimal valid AMD GCN/RDNA s_endpgm (0xBF810000)
			code = []byte{0x00, 0x00, 0x81, 0xBF}
		}
		textBuf.Write(code)
		textSize := uint64(len(code))

		// Align rodata to 64 bytes for kernel descriptor
		if rem := rodataBuf.Len() % 64; rem != 0 {
			rodataBuf.Write(make([]byte, 64-rem))
		}
		rodataOffset := uint64(rodataBuf.Len())

		// Wavefront size default: 32 for gfx1151 (RDNA 3.5)
		waveSize := k.WavefrontSize
		if waveSize == 0 {
			if strings.Contains(targetTriple, "gfx11") || strings.Contains(targetTriple, "gfx12") {
				waveSize = 32
			} else {
				waveSize = 64
			}
		}
		sgprs := k.SGPRCount
		if sgprs == 0 {
			sgprs = 32
		}
		vgprs := k.VGPRCount
		if vgprs == 0 {
			vgprs = 32
		}
		maxFlat := k.MaxFlatWorkgroupSize
		if maxFlat == 0 {
			maxFlat = 256
		}

		// Calculate relative entry offset from descriptor to text
		// In ELF layout: text comes first, then rodata.
		// Entry offset = textOffset - (rodataOffset + rodataBaseOffsetRelativeToText)
		// We'll compute the absolute relative offset once section file offsets are finalized.
		kd := AMDGPUKernelDescriptor{
			GroupSegmentFixedSize:   k.GroupSegmentFixedSize,
			PrivateSegmentFixedSize: k.PrivateSegmentFixedSize,
			KernargSize:             k.KernargSegmentSize,
			ComputePgmRsrc1:         ComputePgmRsrc1(vgprs, sgprs, waveSize),
			ComputePgmRsrc2:         ComputePgmRsrc2(2, k.GroupSegmentFixedSize),
			ComputePgmRsrc3:         0,
		}
		kdBytes := kd.Marshal64()
		rodataBuf.Write(kdBytes[:])

		placements[i] = kernelPlacement{
			name:         kName,
			symbol:       kSymbol,
			textOffset:   textOffset,
			textSize:     textSize,
			rodataOffset: rodataOffset,
		}

		// Construct metadata representation
		var argJSONs []KernelArgJSON
		for _, a := range k.Args {
			vk := a.ValueKind
			if vk == "" {
				vk = "global_buffer"
			}
			argJSONs = append(argJSONs, KernelArgJSON{
				Name:         a.Name,
				TypeName:     a.TypeName,
				Size:         a.Size,
				Offset:       a.Offset,
				ValueKind:    vk,
				AddressSpace: a.AddressSpace,
			})
		}

		align := k.KernargSegmentAlign
		if align == 0 {
			align = 8
		}

		metaKernels[i] = KernelMetadataJSON{
			Name:                    kName,
			Symbol:                  kSymbol,
			Language:                k.Language,
			LanguageVersion:         k.LanguageVersion,
			Args:                    argJSONs,
			GroupSegmentFixedSize:   k.GroupSegmentFixedSize,
			PrivateSegmentFixedSize: k.PrivateSegmentFixedSize,
			KernargSegmentSize:      k.KernargSegmentSize,
			KernargSegmentAlign:     align,
			WavefrontSize:           waveSize,
			SGPRCount:               sgprs,
			VGPRCount:               vgprs,
			AGPRCount:               k.AGPRCount,
			MaxFlatWorkgroupSize:    maxFlat,
			WavefrontNumSGPR:        sgprs,
			WorkitemNumVGPR:         vgprs,
		}
	}

	metadataObj := AMDGPUMetadata{
		Version: versionTuple,
		Target:  targetTriple,
		Kernels: metaKernels,
	}

	var metaBytes []byte
	if strings.ToLower(cfg.Format) == "msgpack" || strings.ToLower(cfg.Format) == "messagepack" {
		metaBytes, err = GenerateMetadataMsgPack(metadataObj)
	} else {
		metaBytes, err = GenerateMetadataJSON(metadataObj)
	}
	if err != nil {
		return nil, fmt.Errorf("amdgpu: failed to encode metadata: %w", err)
	}

	noteData := BuildAMDGPUELFNote(metaBytes)

	// 2. Build Symbol Table and String Tables
	var strTab bytes.Buffer
	strTab.WriteByte(0) // Index 0 is empty string

	addString := func(s string) uint32 {
		idx := uint32(strTab.Len())
		strTab.WriteString(s)
		strTab.WriteByte(0)
		return idx
	}

	type elf64Sym struct {
		stName  uint32
		stInfo  uint8
		stOther uint8
		stShndx uint16
		stValue uint64
		stSize  uint64
	}

	var syms []elf64Sym
	// Sym 0: STN_UNDEF
	syms = append(syms, elf64Sym{})

	// Section indices (1: .text, 2: .rodata, 3: note, 4: symtab, 5: strtab, 6: shstrtab)
	const (
		secIdxText     = 1
		secIdxRodata   = 2
		secIdxNote     = 3
		secIdxSymtab   = 4
		secIdxStrtab   = 5
		secIdxShstrtab = 6
		numSections    = 7
	)

	// Sym 1: .text section symbol
	syms = append(syms, elf64Sym{
		stName:  0,
		stInfo:  (STB_LOCAL << 4) | (STT_SECTION & 0x0F),
		stShndx: secIdxText,
		stValue: 0,
		stSize:  0,
	})

	// Add kernel symbols and kernel descriptor symbols
	for _, p := range placements {
		// Kernel code symbol (STT_FUNC)
		nameIdx := addString(p.name)
		syms = append(syms, elf64Sym{
			stName:  nameIdx,
			stInfo:  (STB_GLOBAL << 4) | (STT_FUNC & 0x0F),
			stShndx: secIdxText,
			stValue: p.textOffset,
			stSize:  p.textSize,
		})

		// Kernel descriptor symbol (STT_OBJECT)
		symIdx := addString(p.symbol)
		syms = append(syms, elf64Sym{
			stName:  symIdx,
			stInfo:  (STB_GLOBAL << 4) | (STT_OBJECT & 0x0F),
			stShndx: secIdxRodata,
			stValue: p.rodataOffset,
			stSize:  64,
		})
	}

	// Serialize Symbol Table
	var symTabBuf bytes.Buffer
	for _, s := range syms {
		var symBytes [24]byte
		binary.LittleEndian.PutUint32(symBytes[0:4], s.stName)
		symBytes[4] = s.stInfo
		symBytes[5] = s.stOther
		binary.LittleEndian.PutUint16(symBytes[6:8], s.stShndx)
		binary.LittleEndian.PutUint64(symBytes[8:16], s.stValue)
		binary.LittleEndian.PutUint64(symBytes[16:24], s.stSize)
		symTabBuf.Write(symBytes[:])
	}

	// Build Section Header String Table (.shstrtab)
	var shstrTab bytes.Buffer
	shstrTab.WriteByte(0)
	addShString := func(s string) uint32 {
		idx := uint32(shstrTab.Len())
		shstrTab.WriteString(s)
		shstrTab.WriteByte(0)
		return idx
	}

	nameIdxText := addShString(".text")
	nameIdxRodata := addShString(".rodata")
	nameIdxNote := addShString(noteSecName)
	nameIdxSymtab := addShString(".symtab")
	nameIdxStrtab := addShString(".strtab")
	nameIdxShstrtab := addShString(".shstrtab")

	// 3. Compute File Offsets and Segment Layout
	const (
		ehdrSize = 64
		phdrSize = 56
		numPhdrs = 3
		phdrsLen = numPhdrs * phdrSize // 168 bytes
	)

	// Offset after headers
	curOffset := uint64(ehdrSize + phdrsLen)

	// Align .text to 256 bytes
	curOffset = (curOffset + 255) &^ 255
	textOffset := curOffset
	textSize := uint64(textBuf.Len())
	curOffset += textSize

	// Align .rodata to 64 bytes
	curOffset = (curOffset + 63) &^ 63
	rodataOffset := curOffset
	rodataSize := uint64(rodataBuf.Len())

	// Fixup kernel descriptor entry offsets now that textOffset and rodataOffset are known
	rodataBytes := rodataBuf.Bytes()
	for _, p := range placements {
		// Relative offset from descriptor start to text entry point
		descFileOffset := rodataOffset + p.rodataOffset
		codeFileOffset := textOffset + p.textOffset
		relOffset := int64(codeFileOffset - descFileOffset)
		// Offset 16 in AMDGPUKernelDescriptor is int64 KernelCodeEntryByteOffset
		binary.LittleEndian.PutUint64(rodataBytes[p.rodataOffset+16:p.rodataOffset+24], uint64(relOffset))
	}
	curOffset += rodataSize

	// Align .note to 4 bytes
	curOffset = (curOffset + 3) &^ 3
	noteOffset := curOffset
	noteSize := uint64(len(noteData))
	curOffset += noteSize

	// Align .symtab to 8 bytes
	curOffset = (curOffset + 7) &^ 7
	symtabOffset := curOffset
	symtabSize := uint64(symTabBuf.Len())
	curOffset += symtabSize

	// .strtab
	strtabOffset := curOffset
	strtabSize := uint64(strTab.Len())
	curOffset += strtabSize

	// .shstrtab
	shstrtabOffset := curOffset
	shstrtabSize := uint64(shstrTab.Len())
	curOffset += shstrtabSize

	// Align Section Header Table to 8 bytes
	curOffset = (curOffset + 7) &^ 7
	shoff := curOffset
	const shdrSize = 64
	totalFileSize := shoff + uint64(numSections*shdrSize)

	// 4. Assemble ELF Header (Elf64_Ehdr)
	var elfHeader [ehdrSize]byte
	elfHeader[EI_MAG0] = ELFMAG0
	elfHeader[EI_MAG1] = ELFMAG1
	elfHeader[EI_MAG2] = ELFMAG2
	elfHeader[EI_MAG3] = ELFMAG3
	elfHeader[EI_CLASS] = ELFCLASS64
	elfHeader[EI_DATA] = ELFDATA2LSB
	elfHeader[EI_VERSION] = EV_CURRENT
	elfHeader[EI_OSABI] = ELFOSABI_AMDGPU_HSA
	elfHeader[EI_ABIVERSION] = abiVersion

	binary.LittleEndian.PutUint16(elfHeader[16:18], ET_DYN)         // e_type
	binary.LittleEndian.PutUint16(elfHeader[18:20], EM_AMDGPU)      // e_machine
	binary.LittleEndian.PutUint32(elfHeader[20:24], EV_CURRENT)     // e_version
	binary.LittleEndian.PutUint64(elfHeader[24:32], textOffset)     // e_entry
	binary.LittleEndian.PutUint64(elfHeader[32:40], ehdrSize)       // e_phoff
	binary.LittleEndian.PutUint64(elfHeader[40:48], shoff)          // e_shoff
	binary.LittleEndian.PutUint32(elfHeader[48:52], eFlags)         // e_flags
	binary.LittleEndian.PutUint16(elfHeader[52:54], ehdrSize)       // e_ehsize
	binary.LittleEndian.PutUint16(elfHeader[54:56], phdrSize)       // e_phentsize
	binary.LittleEndian.PutUint16(elfHeader[56:58], numPhdrs)       // e_phnum
	binary.LittleEndian.PutUint16(elfHeader[58:60], shdrSize)       // e_shentsize
	binary.LittleEndian.PutUint16(elfHeader[60:62], numSections)    // e_shnum
	binary.LittleEndian.PutUint16(elfHeader[62:64], secIdxShstrtab) // e_shstrndx

	// 5. Assemble Program Headers (Elf64_Phdr)
	var phdrs [phdrsLen]byte

	// Phdr 0: PT_LOAD for .text (RX)
	binary.LittleEndian.PutUint32(phdrs[0:4], PT_LOAD)
	binary.LittleEndian.PutUint32(phdrs[4:8], PF_R|PF_X)
	binary.LittleEndian.PutUint64(phdrs[8:16], textOffset)  // p_offset
	binary.LittleEndian.PutUint64(phdrs[16:24], textOffset) // p_vaddr
	binary.LittleEndian.PutUint64(phdrs[24:32], textOffset) // p_paddr
	binary.LittleEndian.PutUint64(phdrs[32:40], textSize)   // p_filesz
	binary.LittleEndian.PutUint64(phdrs[40:48], textSize)   // p_memsz
	binary.LittleEndian.PutUint64(phdrs[48:56], 256)        // p_align

	// Phdr 1: PT_LOAD for .rodata and .note (R)
	p1Size := (noteOffset + noteSize) - rodataOffset
	binary.LittleEndian.PutUint32(phdrs[56:60], PT_LOAD)
	binary.LittleEndian.PutUint32(phdrs[60:64], PF_R)
	binary.LittleEndian.PutUint64(phdrs[64:72], rodataOffset)
	binary.LittleEndian.PutUint64(phdrs[72:80], rodataOffset)
	binary.LittleEndian.PutUint64(phdrs[80:88], rodataOffset)
	binary.LittleEndian.PutUint64(phdrs[88:96], p1Size)
	binary.LittleEndian.PutUint64(phdrs[96:104], p1Size)
	binary.LittleEndian.PutUint64(phdrs[104:112], 64)

	// Phdr 2: PT_NOTE for .note (R)
	binary.LittleEndian.PutUint32(phdrs[112:116], PT_NOTE)
	binary.LittleEndian.PutUint32(phdrs[116:120], PF_R)
	binary.LittleEndian.PutUint64(phdrs[120:128], noteOffset)
	binary.LittleEndian.PutUint64(phdrs[128:136], noteOffset)
	binary.LittleEndian.PutUint64(phdrs[136:144], noteOffset)
	binary.LittleEndian.PutUint64(phdrs[144:152], noteSize)
	binary.LittleEndian.PutUint64(phdrs[152:160], noteSize)
	binary.LittleEndian.PutUint64(phdrs[160:168], 4)

	// 6. Assemble Section Headers (Elf64_Shdr)
	type elf64Shdr struct {
		name      uint32
		shType    uint32
		flags     uint64
		addr      uint64
		offset    uint64
		size      uint64
		link      uint32
		info      uint32
		addralign uint64
		entsize   uint64
	}

	shdrs := []elf64Shdr{
		// 0: SHT_NULL
		{},
		// 1: .text
		{
			name:      nameIdxText,
			shType:    SHT_PROGBITS,
			flags:     SHF_ALLOC | SHF_EXECINSTR,
			addr:      textOffset,
			offset:    textOffset,
			size:      textSize,
			addralign: 256,
		},
		// 2: .rodata
		{
			name:      nameIdxRodata,
			shType:    SHT_PROGBITS,
			flags:     SHF_ALLOC,
			addr:      rodataOffset,
			offset:    rodataOffset,
			size:      rodataSize,
			addralign: 64,
		},
		// 3: .note.amdgpu
		{
			name:      nameIdxNote,
			shType:    SHT_NOTE,
			flags:     SHF_ALLOC,
			addr:      noteOffset,
			offset:    noteOffset,
			size:      noteSize,
			addralign: 4,
		},
		// 4: .symtab
		{
			name:      nameIdxSymtab,
			shType:    SHT_SYMTAB,
			flags:     0,
			addr:      0,
			offset:    symtabOffset,
			size:      symtabSize,
			link:      secIdxStrtab, // Links to .strtab
			info:      2,            // First non-local symbol index
			addralign: 8,
			entsize:   24,
		},
		// 5: .strtab
		{
			name:      nameIdxStrtab,
			shType:    SHT_STRTAB,
			flags:     0,
			addr:      0,
			offset:    strtabOffset,
			size:      strtabSize,
			addralign: 1,
		},
		// 6: .shstrtab
		{
			name:      nameIdxShstrtab,
			shType:    SHT_STRTAB,
			flags:     0,
			addr:      0,
			offset:    shstrtabOffset,
			size:      shstrtabSize,
			addralign: 1,
		},
	}

	var shdrsBuf bytes.Buffer
	for _, sh := range shdrs {
		var b [64]byte
		binary.LittleEndian.PutUint32(b[0:4], sh.name)
		binary.LittleEndian.PutUint32(b[4:8], sh.shType)
		binary.LittleEndian.PutUint64(b[8:16], sh.flags)
		binary.LittleEndian.PutUint64(b[16:24], sh.addr)
		binary.LittleEndian.PutUint64(b[24:32], sh.offset)
		binary.LittleEndian.PutUint64(b[32:40], sh.size)
		binary.LittleEndian.PutUint32(b[40:44], sh.link)
		binary.LittleEndian.PutUint32(b[44:48], sh.info)
		binary.LittleEndian.PutUint64(b[48:56], sh.addralign)
		binary.LittleEndian.PutUint64(b[56:64], sh.entsize)
		shdrsBuf.Write(b[:])
	}

	// 7. Write complete ELF file buffer
	out := make([]byte, totalFileSize)

	// Write Ehdr & Phdrs
	copy(out[0:ehdrSize], elfHeader[:])
	copy(out[ehdrSize:ehdrSize+phdrsLen], phdrs[:])

	// Write sections
	copy(out[textOffset:textOffset+textSize], textBuf.Bytes())
	copy(out[rodataOffset:rodataOffset+rodataSize], rodataBytes)
	copy(out[noteOffset:noteOffset+noteSize], noteData)
	copy(out[symtabOffset:symtabOffset+symtabSize], symTabBuf.Bytes())
	copy(out[strtabOffset:strtabOffset+strtabSize], strTab.Bytes())
	copy(out[shstrtabOffset:shstrtabOffset+shstrtabSize], shstrTab.Bytes())

	// Write section headers
	copy(out[shoff:shoff+uint64(shdrsBuf.Len())], shdrsBuf.Bytes())

	return out, nil
}
