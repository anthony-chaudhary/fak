package compute

import (
	"strconv"
	"strings"
)

// CUDADeviceOp describes one source-declared CUDA weight operation and its
// generic full-build architecture floor. Format is an exact storage/scale pair,
// not permission to use every host-decodable tensor of that dtype.
type CUDADeviceOp struct {
	Operation string
	Format    string
	MinArch   string
}

var cudaDeviceOps = [...]CUDADeviceOp{
	{"matmul", "f32", "sm_80"},
	{"matmul", "f16", "sm_80"},
	{"matmul", "q8_0", "sm_80"},
	{"matmul", "q4_k", "sm_80"},
	{"matmul", "q5_k", "sm_80"},
	{"matmul", "q6_k", "sm_80"},
	{"matmul", "q2_0", "sm_80"},
	{"matmul", "q2_k", "sm_80"},
	{"matmul", "q3_k", "sm_80"},
	{"matmul", "iq2_xxs", "sm_80"},
	{"matmul", "iq2_xs", "sm_80"},
	{"matmul", "iq1_m", "sm_80"},
	{"matmul", "iq4_nl", "sm_80"},
	{"matmul", "nvfp4", "sm_80"},
	{"routed-expert-gemv", "F8_E4M3+F8_E8M0", "sm_80"},
	{"routed-expert-gemv", "I8+F8_E8M0", "sm_80"},
	{"routed-expert-resident", "F8_E4M3+F8_E8M0", "sm_80"},
	{"routed-expert-resident", "I8+F8_E8M0", "sm_80"},
}

// CUDADeviceOpTable returns a defensive copy of the current source declarations.
// MatMul NVFP4 denotes the ordinary W4A16 baseline (F32 inputs rounded to BF16),
// not the separate optional W4A4 interface. Expert I8+F8_E8M0 denotes packed
// MXFP4, not generic I8 MatMul or NVFP4. Resident rows describe expert upload.
// These rows do not inspect this binary, qualify a device, admit a model, or
// establish numerical parity, runtime availability, geometry or performance.
func CUDADeviceOpTable() []CUDADeviceOp {
	return append([]CUDADeviceOp(nil), cudaDeviceOps[:]...)
}

// CUDADeviceOpSupported checks a supplied declaration table against the generic
// full-build source coverage. Missing, unknown or duplicate operation/format
// rows fail closed. MinArch must be a known generic build target; architecture-
// specific floors such as sm_100a and sm_103a are never promoted to sm_120.
// A true result is only a source declaration, not artifact or device admission.
func CUDADeviceOpSupported(table []CUDADeviceOp, operation, format, arch string) bool {
	if !cudaDeviceOpDeclared(operation, format) {
		return false
	}
	dev, ok := cudaDeviceOpArch(arch)
	if !ok || ClassifyCUDAArch(dev.Major, dev.Minor) == CUDAUncovered {
		return false
	}
	found, supported := false, false
	for _, row := range table {
		if row.Operation != operation || row.Format != format {
			continue
		}
		if found {
			return false
		}
		found = true
		floor, valid := cudaDeviceOpArch(row.MinArch)
		knownFloor := false
		for _, known := range cudaEmbeddedArches {
			if valid && floor == known {
				knownFloor = true
				break
			}
		}
		supported = knownFloor && dev.atLeast(floor)
	}
	return found && supported
}

func cudaDeviceOpDeclared(operation, format string) bool {
	for _, row := range cudaDeviceOps {
		if row.Operation == operation && row.Format == format {
			return true
		}
	}
	return false
}

func cudaDeviceWeightDtypeSupported(dt Dtype) bool {
	// Preserve the backend's dtype-only predicate: do not add target admission.
	return cudaDeviceOpDeclared("matmul", dt.String())
}

func cudaDeviceOpArch(arch string) (cudaCC, bool) {
	if !strings.HasPrefix(arch, "sm_") {
		return cudaCC{}, false
	}
	text := strings.TrimPrefix(arch, "sm_")
	value, err := strconv.Atoi(text)
	if err != nil || value < 10 || strconv.Itoa(value) != text {
		return cudaCC{}, false
	}
	return cudaCC{Major: value / 10, Minor: value % 10}, true
}
