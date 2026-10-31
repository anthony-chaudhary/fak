package main

// model_preflight_device.go — the `--device cuda --arch sm_XY` leg of `fak model preflight`
// (ticket P-23 / #14005). It classifies every tensor of a checkpoint — GGUF or safetensors —
// into one closed class against the source-declared CUDA weight-op table at the target arch:
//
//	native            a device kernel exists for the source storage format;
//	expanded:<dtype>  the loader converts the tensor at load to a dtype the device runs;
//	host              the tensor stays on the host by design;
//	absent            the tensor fails closed at load or forward;
//	unknown           a role this header-only plan cannot expose (refused, never guessed).
//
// Each absent group becomes one UNSUPPORTED_DTYPE refusal with a cuda:<op>(<dtype>) axis, the
// same shape the host census uses with host_dequant(<dtype>) axes. The report states
// device.scope "weight-ops": READY is never read as "the forward runs on CUDA". The classifier
// takes the table as an argument so a test can pass a table with a row removed.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

// Closed device classes.
const (
	deviceClassNative    = "native"
	deviceClassHost      = "host"
	deviceClassAbsent    = "absent"
	deviceClassUnknown   = "unknown"
	deviceExpandedPrefix = "expanded:"
	deviceScopeWeightOps = "weight-ops"
)

// modelPreflightDeviceGap groups the tensors that fail closed under one (op, format) key.
type modelPreflightDeviceGap struct {
	Group   string   `json:"group"`
	Tensors []string `json:"tensors"`
}

// modelPreflightDevice is the additive `device` block of the preflight report.
type modelPreflightDevice struct {
	Device   string                    `json:"device"`
	Arch     string                    `json:"arch"`
	Coverage string                    `json:"arch_coverage"`
	Scope    string                    `json:"scope"`
	Classes  map[string]string         `json:"classes"`
	Gaps     []modelPreflightDeviceGap `json:"gaps,omitempty"`
}

// cudaMatmulNative reports whether the declared table carries a matmul kernel for the storage
// format at the target arch. The table is the source declaration under test; CUDADeviceOpTable
// is the live build's.
func cudaMatmulNative(table []compute.CUDADeviceOp, arch, format string) bool {
	return compute.CUDADeviceOpSupported(table, "matmul", strings.ToLower(format), arch)
}

// cudaExpertGemvNative reports whether the declared table carries the routed-expert GEMV kernel
// for the packed (weight, scale) pair at the target arch.
func cudaExpertGemvNative(table []compute.CUDADeviceOp, arch, pair string) bool {
	return compute.CUDADeviceOpSupported(table, "routed-expert-gemv", pair, arch)
}

// classifyGGUFTensor assigns one closed class to a GGUF tensor. retained is the loader's dense
// k-quant retention rule fed by serve's own load options (denseKQuantDecision.retained), so the
// preflight and the load agree on which packed formats stay packed.
func classifyGGUFTensor(info ggufload.TensorInfo, cfg fakmodel.Config, table []compute.CUDADeviceOp, arch string, retained func(ggufload.TensorType) bool) (class, group string) {
	t := info.Type
	switch t {
	case ggufload.TensorF32, ggufload.TensorF16, ggufload.TensorBF16:
		canon, ok := ggufload.CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			return deviceClassHost, "" // sidecar or unmapped: no device weight op runs
		}
		dims := make([]int, len(info.Dims))
		for i, d := range info.Dims {
			dims[i] = int(d)
		}
		if fakmodel.DecodedFloatGoesToQ8(cfg, canon, dims) && cudaMatmulNative(table, arch, "q8_0") {
			return deviceExpandedPrefix + "Q8_0", ""
		}
		return deviceClassHost, ""
	default:
		format := strings.ToLower(t.String())
		if cudaMatmulNative(table, arch, format) {
			return deviceClassNative, ""
		}
		if retained(t) {
			// The loader keeps the packed form and the device has no kernel for it: the load
			// fails closed rather than silently expanding.
			return deviceClassAbsent, "cuda:matmul(" + format + ")"
		}
		if ggufload.HostDequantSupported(t) && cudaMatmulNative(table, arch, "q8_0") {
			return deviceExpandedPrefix + "Q8_0", ""
		}
		return deviceClassAbsent, "cuda:matmul(" + format + ")"
	}
}

// classifySafetensorsTensor assigns one closed class to a safetensors tensor from its plan row.
func classifySafetensorsTensor(t fakmodel.SafetensorsPreflightTensor, table []compute.CUDADeviceOp, arch string) (class, group string) {
	switch t.Role {
	case fakmodel.SafetensorsRoleDroppedMTP,
		fakmodel.SafetensorsRoleDroppedHashTable,
		fakmodel.SafetensorsRoleDroppedSource,
		fakmodel.SafetensorsRoleSkippedBias:
		return deviceClassHost, "" // never materialized; no weight op runs
	case fakmodel.SafetensorsRoleV41Expert, fakmodel.SafetensorsRoleV41ExpertScale:
		if cudaExpertGemvNative(table, arch, "I8+F8_E8M0") {
			return deviceClassNative, ""
		}
		return deviceClassAbsent, "cuda:expert_gemv(I8+F8_E8M0)"
	case fakmodel.SafetensorsRoleV4DenseFP8, fakmodel.SafetensorsRoleV41DenseFP8,
		fakmodel.SafetensorsRoleCompressedHead:
		// These roles install a Q8_0 resident tensor at load.
		if cudaMatmulNative(table, arch, "q8_0") {
			return deviceExpandedPrefix + "Q8_0", ""
		}
		return deviceClassAbsent, "cuda:matmul(q8_0)"
	case fakmodel.SafetensorsRoleMXFP4, fakmodel.SafetensorsRoleFP8BlockScale:
		switch t.Representation {
		case fakmodel.SafetensorsRepQ8:
			if cudaMatmulNative(table, arch, "q8_0") {
				return deviceExpandedPrefix + "Q8_0", ""
			}
			return deviceClassAbsent, "cuda:matmul(q8_0)"
		case fakmodel.SafetensorsRepF32Host:
			return deviceClassHost, ""
		default:
			return deviceClassUnknown, "cuda:unknown(" + t.Role + ")"
		}
	default:
		switch t.Representation {
		case fakmodel.SafetensorsRepQ8:
			if cudaMatmulNative(table, arch, "q8_0") {
				return deviceExpandedPrefix + "Q8_0", ""
			}
			return deviceClassAbsent, "cuda:matmul(q8_0)"
		case fakmodel.SafetensorsRepF32Host:
			return deviceClassHost, ""
		case fakmodel.SafetensorsRepNone:
			return deviceClassHost, ""
		default:
			// An undecodable source dtype (no host decode path) fails closed at load.
			return deviceClassAbsent, "cuda:host_dequant(" + t.Dtype + ")"
		}
	}
}

// buildDeviceReport classifies every tensor of one checkpoint against the declared table at the
// target arch and returns the device block plus one UNSUPPORTED_DTYPE refusal per absent group.
// An uncovered arch refuses once (HARDWARE_ENVELOPE_INVALID) and classifies every tensor absent.
func buildDeviceReport(table []compute.CUDADeviceOp, arch string, retained func(ggufload.TensorType) bool,
	ggufTensors []ggufload.TensorInfo, stTensors []fakmodel.SafetensorsPreflightTensor, cfg fakmodel.Config) (*modelPreflightDevice, []newmodel.Refusal) {
	block := &modelPreflightDevice{
		Device:  "cuda",
		Arch:    arch,
		Scope:   deviceScopeWeightOps,
		Classes: map[string]string{},
	}
	major, minor, ok := compute.ParseCUDADeviceOpArch(arch)
	if !ok {
		block.Coverage = "invalid"
		return block, []newmodel.Refusal{newmodel.HeaderRefusal(newmodel.RefusalHardwareEnvelopeInvalid, "arch", fmt.Sprintf("unparseable arch %q, want a canonical sm_XY target such as sm_103", arch))}
	}
	cov := compute.ClassifyCUDAArch(major, minor)
	if cov == compute.CUDAUncovered {
		block.Coverage = "uncovered"
		names := make([]string, 0, len(ggufTensors)+len(stTensors))
		for _, t := range ggufTensors {
			block.Classes[t.Name] = deviceClassAbsent
			names = append(names, t.Name)
		}
		for _, t := range stTensors {
			block.Classes[t.Name] = deviceClassAbsent
			names = append(names, t.Name)
		}
		sort.Strings(names)
		block.Gaps = append(block.Gaps, modelPreflightDeviceGap{Group: "cuda:arch(" + arch + ")", Tensors: names})
		ref := newmodel.HeaderRefusal(newmodel.RefusalHardwareEnvelopeInvalid, "arch",
			fmt.Sprintf("%s is not covered by this generic full build's embedded arches", arch))
		return block, []newmodel.Refusal{ref}
	}
	block.Coverage = "covered"

	groups := map[string][]string{}
	add := func(name, class, group string) {
		block.Classes[name] = class
		if class == deviceClassAbsent || class == deviceClassUnknown {
			groups[group] = append(groups[group], name)
		}
	}
	for _, t := range ggufTensors {
		class, group := classifyGGUFTensor(t, cfg, table, arch, retained)
		add(t.Name, class, group)
	}
	for _, t := range stTensors {
		class, group := classifySafetensorsTensor(t, table, arch)
		add(t.Name, class, group)
	}

	var refusals []newmodel.Refusal
	keys := make([]string, 0, len(groups))
	for g := range groups {
		keys = append(keys, g)
	}
	sort.Strings(keys)
	for _, g := range keys {
		names := groups[g]
		sort.Strings(names)
		shown := names
		if len(shown) > 6 {
			shown = shown[:6]
		}
		block.Gaps = append(block.Gaps, modelPreflightDeviceGap{Group: g, Tensors: names})
		refusals = append(refusals, newmodel.HeaderRefusal(newmodel.RefusalUnsupportedDtype, g,
			fmt.Sprintf("%d tensors, e.g. %s", len(names), strings.Join(shown, ", "))))
	}
	return block, refusals
}

// cudaStaticKQuantDecision derives the static dense k-quant capability description for a CUDA
// target arch from the declared table: a packed format is retained exactly when the table
// declares a matmul kernel for it at that arch. Preflight feeds the result through the same
// denseKQuantOptions function serve uses, so both legs share one options computation.
func cudaStaticKQuantDecision(table []compute.CUDADeviceOp, arch string) denseKQuantDecision {
	supported := func(dt compute.Dtype) bool {
		return compute.CUDADeviceOpSupported(table, "matmul", dt.String(), arch)
	}
	d := denseKQuantDecision{IsCUDA: true}
	d.DenseQ2K = supported(compute.Q2_K)
	d.DenseQ3K = supported(compute.Q3_K)
	d.DenseQ5K = supported(compute.Q5_K)
	d.DenseQ6K = supported(compute.Q6_K)
	if types := ggufload.NativeIQResidentTypes(supported); len(types) > 0 {
		d.IQResident = types
	}
	return d
}
