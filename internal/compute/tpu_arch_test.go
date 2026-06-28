package compute

import "testing"

// tpu_arch_test.go — host-tractable witnesses for the TPU / Neural-Engine device taxonomy
// (#261). Every assertion here is hardware-independent: it checks the accelerator->lane mapping,
// the XLA/CoreML split, the native-dtype-per-lane invariant, and the target normalization a
// compiler build keys on, none of which needs a TPU or a Neural Engine. The device run (an op-list
// actually lowered through XLA onto a TPU, or a CoreML MLProgram on the ANE) is the deferred half
// — see TPU-C004-NOTES.md.

func TestAccelArchLookupKnown(t *testing.T) {
	cases := []struct {
		name   string
		family AccelFamily
		lane   AccelLane
		dtype  Dtype
		isTPU  bool
	}{
		{"tpu-v2", TPUv2, LaneXLA, BF16, true},
		{"tpu-v3", TPUv3, LaneXLA, BF16, true},
		{"tpu-v4", TPUv4, LaneXLA, BF16, true},
		{"tpu-v5e", TPUv5e, LaneXLA, BF16, true},
		{"tpu-v5p", TPUv5p, LaneXLA, BF16, true},
		{"tpu-v6e", TPUv6e, LaneXLA, BF16, true},
		{"ane-a17", ANEA17, LaneCoreML, F16, false},
		{"ane-m4", ANEM4, LaneCoreML, F16, false},
	}
	for _, c := range cases {
		a, ok := LookupAccelArch(c.name)
		if !ok {
			t.Fatalf("LookupAccelArch(%q): not found, want supported", c.name)
		}
		if a.Family != c.family {
			t.Errorf("%s: family = %v, want %v", c.name, a.Family, c.family)
		}
		if a.Lane != c.lane {
			t.Errorf("%s: lane = %v, want %v", c.name, a.Lane, c.lane)
		}
		if a.NativeDtype != c.dtype {
			t.Errorf("%s: native dtype = %v, want %v", c.name, a.NativeDtype, c.dtype)
		}
		if a.Family.IsTPU() != c.isTPU {
			t.Errorf("%s: IsTPU() = %v, want %v", c.name, a.Family.IsTPU(), c.isTPU)
		}
		if a.Family.IsNeuralEngine() == c.isTPU {
			t.Errorf("%s: IsNeuralEngine()/IsTPU() not mutually exclusive", c.name)
		}
	}
}

// TestAccelLaneInvariant pins the load-bearing split: every supported target is exactly one of the
// two lanes (XLA or CoreML, never both, never neither), each lane carries its native compute tier
// (TPU/XLA -> bf16 MXU, ANE/CoreML -> fp16), and the lane's GraphCompiled predicate and PrimaryCap
// stay consistent with that classification. A new row added with a mistyped lane or dtype, or a
// lane whose cap drifts from the HAL contract, fails here.
func TestAccelLaneInvariant(t *testing.T) {
	for _, a := range KnownAccelArches() {
		switch a.Lane {
		case LaneXLA:
			if !a.Family.IsTPU() {
				t.Errorf("%s: XLA lane but family %v is not a TPU", a.Target, a.Family)
			}
			if a.NativeDtype != BF16 {
				t.Errorf("%s: XLA/TPU native dtype = %v, want bf16", a.Target, a.NativeDtype)
			}
			if !a.Lane.GraphCompiled() {
				t.Errorf("%s: XLA lane must be graph-compiled", a.Target)
			}
			if a.Lane.PrimaryCap() != "GraphCompile" {
				t.Errorf("%s: XLA primary cap = %q, want GraphCompile", a.Target, a.Lane.PrimaryCap())
			}
		case LaneCoreML:
			if !a.Family.IsNeuralEngine() {
				t.Errorf("%s: CoreML lane but family %v is not a Neural Engine", a.Target, a.Family)
			}
			if a.NativeDtype != F16 {
				t.Errorf("%s: CoreML/ANE native dtype = %v, want f16", a.Target, a.NativeDtype)
			}
			if a.Lane.GraphCompiled() {
				t.Errorf("%s: CoreML lane is a fixed op menu, not graph-compiled", a.Target)
			}
			if a.Lane.PrimaryCap() != "FusedFFN" {
				t.Errorf("%s: CoreML primary cap = %q, want FusedFFN", a.Target, a.Lane.PrimaryCap())
			}
		default:
			t.Errorf("%s: unknown lane %v", a.Target, a.Lane)
		}
	}
}

// TestAccelTargetNormalization checks that the target selector accepts the noisy forms a runtime
// actually reports (case, separators, vendor prefix, a parenthetical descriptor) and canonicalizes
// them to the bare fak target token — while still rejecting an unsupported accelerator.
func TestAccelTargetNormalization(t *testing.T) {
	want := map[string]string{
		"tpu-v5e":            "tpu-v5e",
		"TPU v5e":            "tpu-v5e",
		"TPUv5e":             "tpu-v5e",
		"  tpu_v5e ":         "tpu-v5e",
		"v5litepod":          "tpu-v5e",
		"Trillium":           "tpu-v6e",
		"Apple M3 Pro (ANE)": "ane-a17",
		"M4":                 "ane-m4",
	}
	for in, exp := range want {
		got, ok := AccelTarget(in)
		if !ok || got != exp {
			t.Errorf("AccelTarget(%q) = (%q,%v), want (%q,true)", in, got, ok, exp)
		}
	}
	for _, in := range []string{"gfx90a", "sm_90", "", "tpu", "tpu-v9", "wormhole", "gaudi3"} {
		if got, ok := AccelTarget(in); ok {
			t.Errorf("AccelTarget(%q) = (%q,true), want unsupported", in, got)
		}
	}
}

// TestAccelDtypeIsHALDtype ties the taxonomy back into the HAL: the native compute tier each lane
// declares is a real compute.Dtype the seam already dispatches on (BF16 / F16), not a free-form
// string — so the deferred backend's Upload(t, NativeDtype) narrows weights to a dtype the
// contract understands rather than inventing a parallel format vocabulary.
func TestAccelDtypeIsHALDtype(t *testing.T) {
	for _, a := range KnownAccelArches() {
		switch a.NativeDtype {
		case BF16, F16:
			// ok — both are first-class HAL dtypes (compute.go).
		default:
			t.Errorf("%s: native dtype %v is not a device-native HAL float tier", a.Target, a.NativeDtype)
		}
		if a.NativeDtype.Bytes() != 2 {
			t.Errorf("%s: native dtype %v width = %d bytes, want 2 (device-native half)", a.Target, a.NativeDtype, a.NativeDtype.Bytes())
		}
	}
}
