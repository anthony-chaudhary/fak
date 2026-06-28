package compute

// openvino_arch.go — the always-compiled, hardware-independent half of the OpenVINO backend
// (issue #257 / C-006). It is the device-plugin taxonomy an OpenVINO runtime integration needs
// BEFORE any model runs: which Intel device classes fak targets through OpenVINO, the canonical
// OpenVINO plugin device token each maps to ("CPU" / "GPU" / "NPU"), the device-native inference
// precision, and which device-selection strings are VIRTUAL meta-plugins (AUTO / HETERO / MULTI /
// BATCH) that dispatch to a physical device rather than being a compile target themselves. None of
// this needs an Intel CPU/GPU/NPU to be correct, so it is pure Go with no build tag and is
// unit-witnessed on any host — the same split ROCM-C002-NOTES.md / TPU-C004-NOTES.md use (ship the
// exact, host-tractable part; defer the device run). The cgo OpenVINO backend itself (the
// `//go:build openvino` twin that exports the model to OpenVINO IR, registers an Approx backend
// named "openvino", and selects a device through OpenVINO's plugin API, mirroring cuda.go) lands on
// an Intel node where it can actually compile, run, and be witnessed; see OPENVINO-C006-NOTES.md
// for that hand-off.
//
// Why a taxonomy and not a free-form string: OpenVINO's whole value is "Device selection" (a named
// scope bullet of #257) — one IR runs on a CPU, an integrated or discrete Intel GPU, or the Intel
// NPU (the AI-Boost inference accelerator on Meteor/Lunar/Arrow Lake), and the device picks the
// native precision and the kind of part it is (a programmable CPU/GPU vs the NPU's whole-model
// ahead-of-time-compiled fixed graph). Picking the wrong plugin token, or treating a virtual
// meta-plugin as a physical target, silently mis-routes the model. This table is the single place
// that mapping lives, so an unsupported device fails closed instead of running on the wrong plugin.
//
// This is distinct from the Intel XPU / oneDNN-SYCL lens (#264, docs/explainers/hardware-
// portability.md): that lens reaches an Intel Arc discrete GPU through the oneAPI/SYCL runtime and
// hand-lowered oneDNN primitives; OpenVINO is Intel's higher-level INFERENCE runtime that ingests
// an IR and dispatches it across CPU/GPU/NPU plugins — a different integration whose load-bearing
// decision is the plugin/device selection, and whose unique reach is the NPU.

// OVDeviceClass is an OpenVINO physical device plugin fak can target. It is the load-bearing
// classification: it selects the plugin the IR is dispatched to and therefore the native inference
// precision and whether the part is a general programmable device or a fixed-graph accelerator.
type OVDeviceClass uint8

const (
	// OVUnknown is the zero value: a device string fak has no OpenVINO plugin for.
	OVUnknown OVDeviceClass = iota
	// OVClassCPU is the OpenVINO CPU plugin (Intel x86, oneDNN under the hood). It is the
	// "within 1.5× native CPU" baseline the #257 acceptance grades against; fp32-native, with
	// bf16/int8 acceleration on AMX/AVX-512 parts expressed via Dtype/QuantSpec.
	OVClassCPU
	// OVClassGPU is the OpenVINO GPU plugin (Intel integrated Gen graphics AND discrete Arc /
	// Data-Center GPU Max). Both are addressed through the one "GPU" plugin token, with the
	// device-instance ordinal ("GPU.0" integrated, "GPU.1" discrete) picking the instance.
	// fp16-native.
	OVClassGPU
	// OVClassNPU is the OpenVINO NPU plugin (the Intel AI-Boost NPU on Meteor/Lunar/Arrow Lake) —
	// the unique reach of this backend and the dedicated #257 "NPU support" acceptance bullet. The
	// NPU compiles the whole IR to a static device blob ahead of time and runs it as a fixed graph
	// (the Caps.GraphCompile shape), fp16-native with int8 the preferred deployment quantization.
	OVClassNPU
)

// String returns the short class label.
func (c OVDeviceClass) String() string {
	switch c {
	case OVClassCPU:
		return "CPU"
	case OVClassGPU:
		return "GPU"
	case OVClassNPU:
		return "NPU"
	default:
		return "unknown"
	}
}

// IsNPU reports whether the class is the Intel NPU plugin — the predicate the dedicated #257 "NPU
// support" acceptance bullet keys on.
func (c OVDeviceClass) IsNPU() bool { return c == OVClassNPU }

// FixedGraph reports whether the device runs a whole-model graph compiled ahead of time (the NPU),
// as opposed to a programmable device that dispatches primitives per op (CPU/GPU). This is the
// load-bearing property that decides whether the deferred backend stages a precompiled blob.
func (c OVDeviceClass) FixedGraph() bool { return c == OVClassNPU }

// PrimaryCap names the single optional Backend capability (a field of Caps) the deferred OpenVINO
// backend keys on for this device, so the taxonomy and the HAL contract stay in sync: the NPU's
// ahead-of-time whole-model compile is "GraphCompile"; a discrete GPU's separate VRAM is
// "DeviceMemory"; the CPU plugin is the programmable parity floor and advertises no special cap
// (""). It returns "" for an unknown class.
func (c OVDeviceClass) PrimaryCap() string {
	switch c {
	case OVClassNPU:
		return "GraphCompile"
	case OVClassGPU:
		return "DeviceMemory"
	case OVClassCPU:
		return ""
	default:
		return ""
	}
}

// OVArch is one supported OpenVINO device target: its canonical plugin device token, its class, the
// device-native inference precision, and a representative product.
type OVArch struct {
	Device          string        // canonical OpenVINO plugin device token, e.g. "CPU","GPU","NPU"
	Class           OVDeviceClass // physical plugin
	NativePrecision Dtype         // device-native inference tier: F32 on the CPU plugin, F16 on GPU/NPU
	Examples        string        // representative product(s)
	aliases         []string      // device-reported spellings (already normalized) that resolve here
}

// ovArches is the supported-target table, declared once in the OpenVINO plugin order the runtime
// itself reports: CPU, then GPU, then NPU. The aliases are the lowercased spellings a device probe
// (core.available_devices) actually reports; the instance ordinal ("GPU.0"/"GPU.1") is stripped by
// normalizeOVDevice before lookup, so it need not be enumerated here.
var ovArches = []OVArch{
	{Device: "CPU", Class: OVClassCPU, NativePrecision: F32, Examples: "Intel Xeon / Core (x86, oneDNN)", aliases: []string{"cpu"}},
	{Device: "GPU", Class: OVClassGPU, NativePrecision: F16, Examples: "Intel UHD/Iris iGPU, Arc A-series, Data-Center GPU Max", aliases: []string{"gpu", "igpu", "dgpu"}},
	{Device: "NPU", Class: OVClassNPU, NativePrecision: F16, Examples: "Intel AI Boost NPU (Meteor/Lunar/Arrow Lake)", aliases: []string{"npu", "vpu"}},
}

// ovByKey indexes the table by the canonical device token AND every alias (all normalized) for O(1)
// lookup of a device-reported string.
var ovByKey = func() map[string]OVArch {
	m := make(map[string]OVArch, len(ovArches)*2)
	for _, a := range ovArches {
		m[normalizeOVDevice(a.Device)] = a
		for _, al := range a.aliases {
			m[normalizeOVDevice(al)] = a
		}
	}
	return m
}()

// ovVirtualDevices is the set of OpenVINO VIRTUAL meta-plugins — they implement "Device selection"
// by dispatching the IR to one or more physical devices (AUTO picks the best available, HETERO
// splits one model across devices, MULTI runs on several in parallel, BATCH auto-batches), so they
// are recognized but are NOT a physical compile target. A selector may accept them; a backend
// resolves them to physical devices before placing the model.
var ovVirtualDevices = map[string]bool{
	"auto":   true,
	"hetero": true,
	"multi":  true,
	"batch":  true,
}

// normalizeOVDevice canonicalizes a device-reported OpenVINO string to a compact lowercase key.
// available_devices reports plugin tokens with case noise, whitespace, and — uniquely for OpenVINO
// — a device-instance ordinal ("GPU.0", "GPU.1") and, for a virtual plugin, a colon-delimited
// candidate list ("AUTO:GPU,CPU", "HETERO:GPU,CPU"). It lowercases, trims whitespace, drops the
// candidate list at the first ':' (virtual) and the ordinal at the first '.' (physical instance).
// It does not invent a device: an unrecognized key is returned for Lookup to reject (fail closed).
func normalizeOVDevice(s string) string {
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == ':' { // drop a virtual-plugin candidate list ("AUTO:GPU,CPU")
			break
		}
		if c == '.' { // drop a physical device-instance ordinal ("GPU.0" -> "gpu")
			break
		}
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' { // trim whitespace anywhere
			continue
		}
		if c >= 'A' && c <= 'Z' { // lowercase
			c += 'a' - 'A'
		}
		out = append(out, c)
	}
	return string(out)
}

// LookupOVDevice resolves a device-reported OpenVINO string (any case, optional ".N" instance
// ordinal) to its supported PHYSICAL device target, or (zero, false) if fak has no plugin for it.
// A virtual meta-plugin (AUTO/HETERO/MULTI/BATCH) is deliberately NOT a physical target and returns
// false here — test it with IsVirtualOVDevice. This is the fail-closed admission a build/runtime
// path uses so an unknown device is never silently dispatched through the wrong plugin.
func LookupOVDevice(name string) (OVArch, bool) {
	a, ok := ovByKey[normalizeOVDevice(name)]
	return a, ok
}

// IsVirtualOVDevice reports whether a device-reported string names an OpenVINO virtual meta-plugin
// (AUTO / HETERO / MULTI / BATCH, with or without a ":candidate,list" suffix) — a device-selection
// directive that delegates to physical devices rather than being a compile target itself.
func IsVirtualOVDevice(name string) bool {
	return ovVirtualDevices[normalizeOVDevice(name)]
}

// OVDeviceToken returns the canonical OpenVINO plugin device token ("CPU"/"GPU"/"NPU") for a
// device-reported string, or ("", false) if it is not a supported physical device. The deferred
// backend feeds the result straight to OpenVINO's core.compile_model(model, device); a `--backend
// openvino --device <token>` selector accepts the same token.
func OVDeviceToken(name string) (string, bool) {
	a, ok := LookupOVDevice(name)
	if !ok {
		return "", false
	}
	return a.Device, true
}

// KnownOVDevices returns the supported-target table in declared (plugin) order. A `fak` diagnostic
// or the OpenVINO integration enumerates it to print exactly which Intel devices this build of fak
// targets — the honest answer to "does fak run on my CPU / GPU / NPU through OpenVINO?".
func KnownOVDevices() []OVArch {
	out := make([]OVArch, len(ovArches))
	copy(out, ovArches)
	return out
}
