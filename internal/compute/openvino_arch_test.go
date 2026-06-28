package compute

import "testing"

// openvino_arch_test.go — host-tractable witnesses for the OpenVINO device-plugin taxonomy (#257).
// Every assertion here is hardware-independent: it checks the device->class mapping, the
// CPU/GPU/NPU split, the native-precision-per-device invariant, the instance-ordinal and virtual-
// plugin normalization a runtime integration keys on, and that the NPU acceptance predicate holds —
// none of which needs an Intel CPU/GPU/NPU. The device run (an IR actually compiled and executed on
// an Intel device through OpenVINO) is the deferred half — see OPENVINO-C006-NOTES.md.

func TestOVDeviceLookupKnown(t *testing.T) {
	cases := []struct {
		name      string
		class     OVDeviceClass
		precision Dtype
		isNPU     bool
	}{
		{"CPU", OVClassCPU, F32, false},
		{"GPU", OVClassGPU, F16, false},
		{"NPU", OVClassNPU, F16, true},
	}
	for _, c := range cases {
		a, ok := LookupOVDevice(c.name)
		if !ok {
			t.Fatalf("LookupOVDevice(%q): not found, want supported", c.name)
		}
		if a.Class != c.class {
			t.Errorf("%s: class = %v, want %v", c.name, a.Class, c.class)
		}
		if a.NativePrecision != c.precision {
			t.Errorf("%s: native precision = %v, want %v", c.name, a.NativePrecision, c.precision)
		}
		if a.Class.IsNPU() != c.isNPU {
			t.Errorf("%s: IsNPU() = %v, want %v", c.name, a.Class.IsNPU(), c.isNPU)
		}
	}
}

// TestOVDeviceClassInvariant pins the load-bearing split: every supported device is exactly one of
// the three OpenVINO physical plugins (CPU, GPU, NPU — never unknown), the native precision matches
// the device tier (CPU is fp32, GPU/NPU are fp16), and only the NPU is a fixed whole-model graph
// while its PrimaryCap stays consistent with the HAL contract. A new row added with a mistyped
// class, precision, or cap fails here.
func TestOVDeviceClassInvariant(t *testing.T) {
	for _, a := range KnownOVDevices() {
		switch a.Class {
		case OVClassCPU:
			if a.NativePrecision != F32 {
				t.Errorf("%s: CPU native precision = %v, want f32", a.Device, a.NativePrecision)
			}
			if a.Class.FixedGraph() {
				t.Errorf("%s: CPU is a programmable device, not a fixed graph", a.Device)
			}
			if a.Class.IsNPU() {
				t.Errorf("%s: CPU must not classify as NPU", a.Device)
			}
			if a.Class.PrimaryCap() != "" {
				t.Errorf("%s: CPU primary cap = %q, want \"\" (parity floor)", a.Device, a.Class.PrimaryCap())
			}
		case OVClassGPU:
			if a.NativePrecision != F16 {
				t.Errorf("%s: GPU native precision = %v, want f16", a.Device, a.NativePrecision)
			}
			if a.Class.FixedGraph() {
				t.Errorf("%s: GPU is a programmable device, not a fixed graph", a.Device)
			}
			if a.Class.PrimaryCap() != "DeviceMemory" {
				t.Errorf("%s: GPU primary cap = %q, want DeviceMemory", a.Device, a.Class.PrimaryCap())
			}
		case OVClassNPU:
			if a.NativePrecision != F16 {
				t.Errorf("%s: NPU native precision = %v, want f16", a.Device, a.NativePrecision)
			}
			if !a.Class.FixedGraph() {
				t.Errorf("%s: NPU must be an ahead-of-time whole-model graph", a.Device)
			}
			if !a.Class.IsNPU() {
				t.Errorf("%s: NPU must classify as NPU", a.Device)
			}
			if a.Class.PrimaryCap() != "GraphCompile" {
				t.Errorf("%s: NPU primary cap = %q, want GraphCompile", a.Device, a.Class.PrimaryCap())
			}
		default:
			t.Errorf("%s: unknown class %v", a.Device, a.Class)
		}
	}
}

// TestOVDeviceNormalization checks that the device selector accepts the noisy forms OpenVINO's
// available_devices actually reports (case, whitespace, the ".N" device-instance ordinal that
// distinguishes GPU.0 from GPU.1) and canonicalizes them to the bare plugin token — while still
// rejecting an unsupported device AND a virtual meta-plugin (which is not a physical compile
// target).
func TestOVDeviceNormalization(t *testing.T) {
	want := map[string]string{
		"CPU":    "CPU",
		"cpu":    "CPU",
		" CPU ":  "CPU",
		"GPU":    "GPU",
		"GPU.0":  "GPU", // integrated-GPU instance ordinal
		"GPU.1":  "GPU", // discrete-GPU instance ordinal
		"gpu.0":  "GPU",
		"iGPU":   "GPU",
		"NPU":    "NPU",
		"npu":    "NPU",
	}
	for in, exp := range want {
		got, ok := OVDeviceToken(in)
		if !ok || got != exp {
			t.Errorf("OVDeviceToken(%q) = (%q,%v), want (%q,true)", in, got, ok, exp)
		}
	}
	// Unsupported devices and bare product strings fail closed; virtual meta-plugins are not
	// physical targets and must not resolve through LookupOVDevice either.
	for _, in := range []string{"TPU", "gfx90a", "sm_90", "", "cuda", "AUTO", "HETERO", "MULTI", "BATCH", "AUTO:GPU,CPU"} {
		if got, ok := OVDeviceToken(in); ok {
			t.Errorf("OVDeviceToken(%q) = (%q,true), want unsupported", in, got)
		}
	}
}

// TestOVVirtualDevices pins OpenVINO's device-selection meta-plugins: AUTO/HETERO/MULTI/BATCH are
// recognized as VIRTUAL (with or without a ":candidate,list" suffix) and are never confused with a
// physical device — the distinction a backend keys on to know whether to resolve-then-place or
// place directly. A physical token is not virtual, and an unknown token is neither.
func TestOVVirtualDevices(t *testing.T) {
	for _, in := range []string{"AUTO", "auto", "HETERO", "MULTI", "BATCH", "AUTO:GPU,CPU", "HETERO:NPU,CPU"} {
		if !IsVirtualOVDevice(in) {
			t.Errorf("IsVirtualOVDevice(%q) = false, want true", in)
		}
		if _, ok := LookupOVDevice(in); ok {
			t.Errorf("LookupOVDevice(%q): a virtual meta-plugin must not resolve as a physical device", in)
		}
	}
	for _, in := range []string{"CPU", "GPU", "GPU.0", "NPU"} {
		if IsVirtualOVDevice(in) {
			t.Errorf("IsVirtualOVDevice(%q) = true, want false (physical)", in)
		}
	}
	for _, in := range []string{"TPU", "", "cuda"} {
		if IsVirtualOVDevice(in) {
			t.Errorf("IsVirtualOVDevice(%q) = true, want false (unknown)", in)
		}
	}
}

// TestOVPrecisionIsHALDtype ties the taxonomy back into the HAL: the native inference precision each
// device declares is a real compute.Dtype the seam already dispatches on (F32 / F16), not a
// free-form string — so the deferred backend's Upload(t, NativePrecision) narrows weights to a
// dtype the contract understands rather than inventing a parallel format vocabulary.
func TestOVPrecisionIsHALDtype(t *testing.T) {
	for _, a := range KnownOVDevices() {
		switch a.NativePrecision {
		case F32, F16:
			// ok — both are first-class HAL float dtypes (compute.go).
		default:
			t.Errorf("%s: native precision %v is not a HAL float tier", a.Device, a.NativePrecision)
		}
	}
}
