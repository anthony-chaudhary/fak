package strix

import (
	"errors"
	"math"
	"strings"
	"testing"
)

// dequantWMMAFP4TestKernelFunc adapts a plain function to the fp4DeviceKernel seam so
// a test can register a device-backed implementation without a struct.
type dequantWMMAFP4TestKernelFunc func(*ROCmFP4Tensor, []float32, []float32) error

func (f dequantWMMAFP4TestKernelFunc) CoopMatMulInto(t *ROCmFP4Tensor, v []float32, d []float32) error {
	return f(t, v, d)
}

// TestROCmFP4InRegisterDequantWMMAWitness is the witness for issue #13003's
// on-the-fly FP4 in-register dequant path. It proves the DequantWMMAKernel emits a
// real v_wmma_f32_16x16x16_fp4 opcode, that the embedded GFX1151 source actually
// carries the instruction, that E2M1 in-register decode matches UnpackFP4Block32Into,
// that the roofline entry point fails closed with a typed error when the device kernel
// is absent, and that the bandwidth telemetry is labelled a model constant.
//
// NOTE: the function is named ...Witness rather than the issue's suggested
// TestROCmFP4OnTheFlyDequantWMMAEmitsFP4Opcode because a peer lane already declares
// that exact symbol in rocmfp4_wmma_test.go within this same package; two identical
// test funcs cannot compile. This file carries the stronger, directly-reachable
// assertions the issue asked for.
func TestROCmFP4InRegisterDequantWMMAWitness(t *testing.T) {
	const fp4Opcode = "v_wmma_f32_16x16x16_fp4"

	// 1. An FP4-configured kernel reports the FP4 opcode via the typed accessors.
	t.Run("accessor_reports_fp4_opcode", func(t *testing.T) {
		cfg := DefaultDequantWMMAConfig()
		cfg.QuantType = QuantTypeFP4
		kernel, err := NewDequantWMMAKernel(cfg)
		if err != nil {
			t.Fatalf("NewDequantWMMAKernel: %v", err)
		}
		if !kernel.EmitsFP4WMMAOpcode() {
			t.Fatalf("EmitsFP4WMMAOpcode() = false, want true for QuantTypeFP4")
		}
		if got := kernel.FP4WMMAOpcode(); got != fp4Opcode {
			t.Fatalf("FP4WMMAOpcode() = %q, want %q", got, fp4Opcode)
		}
	})

	// 2. Inspection reflects emission, not merely a helper constant.
	t.Run("inspect_assembly_wmma_opcode", func(t *testing.T) {
		cfg := DefaultDequantWMMAConfig()
		cfg.QuantType = QuantTypeFP4
		kernel, err := NewDequantWMMAKernel(cfg)
		if err != nil {
			t.Fatalf("NewDequantWMMAKernel: %v", err)
		}
		if got := kernel.InspectAssembly().WMMAOpcode; got != fp4Opcode {
			t.Fatalf("InspectAssembly().WMMAOpcode = %q, want %q", got, fp4Opcode)
		}
	})

	// 3. The embedded source really contains the instruction (not only a comment
	// declaring it): at least one line carries the opcode as the first token.
	t.Run("embedded_source_contains_real_instruction", func(t *testing.T) {
		if !strings.Contains(GFX1151AssemblySource, fp4Opcode) {
			t.Fatalf("GFX1151AssemblySource does not contain %q", fp4Opcode)
		}
		foundInstruction := false
		for _, line := range strings.Split(GFX1151AssemblySource, "\n") {
			code := strings.TrimSpace(line)
			// Strip trailing line comments so a comment-only occurrence cannot witness.
			if idx := strings.Index(code, "//"); idx >= 0 {
				code = strings.TrimSpace(code[:idx])
			}
			if idx := strings.Index(code, ";"); idx >= 0 {
				code = strings.TrimSpace(code[:idx])
			}
			if strings.HasPrefix(code, fp4Opcode) {
				foundInstruction = true
				break
			}
		}
		if !foundInstruction {
			t.Fatalf("no non-comment assembly line begins with the instruction %q", fp4Opcode)
		}
	})

	// 4. Behavioral anchor: in-register FP4 dequant is reachable with NO full-tensor
	// FP32 expansion API, and matches UnpackFP4Block32Into byte-for-byte.
	t.Run("in_register_dequant_matches_unpack", func(t *testing.T) {
		cfg := DefaultDequantWMMAConfig()
		cfg.QuantType = QuantTypeFP4
		kernel, err := NewDequantWMMAKernel(cfg)
		if err != nil {
			t.Fatalf("NewDequantWMMAKernel: %v", err)
		}

		// One 32-element packed block: 16 bytes, 2 E2M1 nibbles per byte, unit scale.
		block := FP4Block32{
			Scale: FP32ToFP16(1.0),
			Flags: BlockFlagWave32Aligned,
		}
		for i := 0; i < 16; i++ {
			block.Data[i] = byte(i) | byte(15-i)<<4
		}

		packed := make([]byte, 16)
		copy(packed, block.Data[:])
		scales := []float32{1.0}

		got, err := kernel.DequantizeInRegister(packed, scales, QuantTypeFP4)
		if err != nil {
			t.Fatalf("DequantizeInRegister(FP4): %v", err)
		}
		if len(got) != 32 {
			t.Fatalf("in-register dequant len = %d, want 32", len(got))
		}

		want := make([]float32, 32)
		if err := UnpackFP4Block32Into(block, want); err != nil {
			t.Fatalf("UnpackFP4Block32Into: %v", err)
		}

		for i := 0; i < 32; i++ {
			if got[i] != want[i] {
				t.Errorf("index %d: in-register %v != unpack %v", i, got[i], want[i])
			}
		}

		// Anchor the decode against the E2M1 table itself at a known nibble.
		if buf0 := got[0]; buf0 != e2m1Table[0] {
			t.Errorf("index 0: got %v, want E2M1 dec(0)=%v", buf0, e2m1Table[0])
		}
	})

	// 5. Fail-closed: with no FP4 device kernel registered, the roofline entry point
	// refuses with the typed absence error rather than silently dequantizing on the
	// CPU. The guard is a process-wide nil-kernel registry, so the absence branch is
	// directly reachable: clear it, call, restore it.
	t.Run("fails_closed_without_device_kernel", func(t *testing.T) {
		if ErrFP4DeviceKernelAbsent == nil {
			t.Fatal("ErrFP4DeviceKernelAbsent is nil")
		}
		if !errors.Is(ErrFP4DeviceKernelAbsent, ErrFP4DeviceKernelAbsent) {
			t.Error("errors.Is(ErrFP4DeviceKernelAbsent, ErrFP4DeviceKernelAbsent) = false, want true")
		}
		// The legacy spelling must alias the registry sentinel so old callers match.
		if !errors.Is(ErrFP4DeviceKernelAbsent, ErrNoFP4DeviceKernel) {
			t.Error("errors.Is(ErrFP4DeviceKernelAbsent, ErrNoFP4DeviceKernel) = false, want true (legacy alias)")
		}

		const rows, cols = 1, 32
		src := make([]float32, rows*cols)
		for i := range src {
			src[i] = e2m1Table[i%16]
		}
		tensor, err := PackTensorROCmFP4(src, rows, cols)
		if err != nil {
			t.Fatalf("PackTensorROCmFP4: %v", err)
		}

		// Force the absence branch to be the real state we exercise.
		RegisterROCmFP4DeviceKernel(nil)
		t.Cleanup(func() { RegisterROCmFP4DeviceKernel(nil) })
		if ROCmFP4DeviceKernelAvailable() {
			t.Fatal("ROCmFP4DeviceKernelAvailable() = true after clearing; cannot witness absence")
		}

		vector := make([]float32, cols)
		dst := make([]float32, rows)
		gotErr := ROCmFP4CoopMatMulInto(tensor, vector, dst)
		if !errors.Is(gotErr, ErrFP4DeviceKernelAbsent) {
			t.Fatalf("ROCmFP4CoopMatMulInto err = %v, want errors.Is(..., ErrFP4DeviceKernelAbsent)", gotErr)
		}
	})

	// 5b. With a device kernel registered, the reference path runs and its values are
	// sane (no corruption), while a reference pack -> unpack round-trip also matches.
	t.Run("reference_path_values_sane_when_registered", func(t *testing.T) {
		const rows, cols = 1, 32
		src := make([]float32, rows*cols)
		for i := range src {
			src[i] = e2m1Table[i%16]
		}
		tensor, err := PackTensorROCmFP4(src, rows, cols)
		if err != nil {
			t.Fatalf("PackTensorROCmFP4: %v", err)
		}

		RegisterROCmFP4DeviceKernel(dequantWMMAFP4TestKernelFunc(func(*ROCmFP4Tensor, []float32, []float32) error { return nil }))
		t.Cleanup(func() { RegisterROCmFP4DeviceKernel(nil) })

		vector := make([]float32, cols)
		for i := range vector {
			vector[i] = 1.0
		}
		dst := make([]float32, rows)
		if err := ROCmFP4CoopMatMulInto(tensor, vector, dst); err != nil {
			t.Fatalf("ROCmFP4CoopMatMulInto(registered): %v", err)
		}
		// sum of e2m1Table over 32 lanes (each value twice) is the expected row total.
		var want float32
		for i := 0; i < 32; i++ {
			want += e2m1Table[i%16]
		}
		if math.Abs(float64(dst[0]-want)) > 1e-3 {
			t.Errorf("dst[0] = %v, want %v (reference accumulation)", dst[0], want)
		}

		got := make([]float32, 32)
		if err := UnpackFP4Block32Into(tensor.Blocks[0], got); err != nil {
			t.Fatalf("UnpackFP4Block32Into: %v", err)
		}
		for i := range got {
			if math.Abs(float64(got[i]-src[i])) > 1e-4 {
				t.Errorf("index %d: dequantized %v != source %v", i, got[i], src[i])
			}
		}
	})

	// 6. Roofline bandwidth is labelled a static model constant, not a measurement.
	t.Run("telemetry_bandwidth_is_model_constant", func(t *testing.T) {
		tel := ComputeROCmFP4Telemetry(4, 64)
		if !tel.BandwidthIsModelConstant {
			t.Fatal("ComputeROCmFP4Telemetry(...).BandwidthIsModelConstant = false, want true")
		}
	})
}
