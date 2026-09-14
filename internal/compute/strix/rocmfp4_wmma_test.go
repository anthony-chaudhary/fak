package strix

import (
	"errors"
	"testing"
)

// TestROCmFP4OnTheFlyDequantWMMAEmitsFP4Opcode is the witness for issue #13003 Lane B:
// the on-the-fly in-register FP4 dequant WMMA path. It proves (a) the emitted GFX1151
// source carries a real v_wmma_f32_16x16x16_fp4 opcode with an in-register nibble-extract
// and a gfx1151 target, and (b) the roofline entry point ROCmFP4CoopMatMulInto fails closed
// with ErrNoFP4DeviceKernel when no device kernel is registered — never silently
// dequantizing on the CPU.
func TestROCmFP4OnTheFlyDequantWMMAEmitsFP4Opcode(t *testing.T) {
	t.Run("emits_real_fp4_wmma_with_in_register_dequant", func(t *testing.T) {
		src := EmitROCmFP4WMMASource()
		if src == "" {
			t.Fatal("EmitROCmFP4WMMASource() returned empty source")
		}
		if !contains(src, ROCmFP4WMMAOpcode) {
			t.Fatalf("emitted source does not contain literal opcode %q", ROCmFP4WMMAOpcode)
		}

		info := AnalyzeROCmFP4WMMASource(src)
		if !info.HasFP4WMMA || info.Opcode != ROCmFP4WMMAOpcode {
			t.Fatalf("analysis did not detect FP4 WMMA opcode: %s", info)
		}
		if !info.HasNibbleExtract {
			t.Fatalf("emitted source has no in-register nibble-extract instruction: %s", info)
		}
		if info.TargetArch != "gfx1151" {
			t.Fatalf("emitted source target = %q, want gfx1151: %s", info.TargetArch, info)
		}
		if info.WavefrontSize != 32 {
			t.Fatalf("emitted source wavefront size = %d, want 32: %s", info.WavefrontSize, info)
		}
		if !info.HasEndpgm {
			t.Fatalf("emitted source does not terminate with s_endpgm: %s", info)
		}
	})

	t.Run("telemetry_bandwidth_is_labelled_model_constant", func(t *testing.T) {
		tel := ComputeROCmFP4Telemetry(4, 64)
		if !tel.BandwidthIsModelConstant {
			t.Fatal("ComputeROCmFP4Telemetry(...).BandwidthIsModelConstant = false, want true")
		}
		if !contains(tel.BandwidthSource, "model_constant") {
			t.Fatalf("BandwidthSource = %q, want an explicit model_constant label", tel.BandwidthSource)
		}
	})

	t.Run("roofline_entry_fails_closed_without_device_kernel", func(t *testing.T) {
		// No device kernel is registered on this host.
		if ROCmFP4DeviceKernelAvailable() {
			t.Skip("an FP4 device kernel is registered on this host; fail-closed negative case is not reachable")
		}

		tensor, err := PackTensorROCmFP4(make([]float32, 32), 1, 32)
		if err != nil {
			t.Fatalf("PackTensorROCmFP4: %v", err)
		}
		vector := make([]float32, tensor.Cols)
		dst := make([]float32, tensor.Rows)

		err = ROCmFP4CoopMatMulInto(tensor, vector, dst)
		if err == nil {
			t.Fatal("ROCmFP4CoopMatMulInto returned nil with no device kernel; want ErrNoFP4DeviceKernel (fail-closed)")
		}
		if !errors.Is(err, ErrNoFP4DeviceKernel) {
			t.Fatalf("ROCmFP4CoopMatMulInto error = %v, want errors.Is(err, ErrNoFP4DeviceKernel)", err)
		}
	})
}

// contains is a tiny local helper so the witness does not depend on the strings package
// import being present for any particular build configuration.
func contains(haystack, needle string) bool {
	if needle == "" {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
