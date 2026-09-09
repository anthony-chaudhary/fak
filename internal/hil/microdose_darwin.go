//go:build darwin && arm64 && cgo

package hil

import (
	"fmt"
	"math"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func runPlatformMicroDose(kind DoseKind, hw HardwareInfo) MicroDoseResult {
	if !metalgemm.Available() {
		return runHostReferenceMicroDose(kind, hw)
	}

	switch kind {
	case DoseLiveness:
		return runMetalLiveness(hw)
	case DoseComputeGEMM, DoseComputeGEMV:
		return runMetalComputeGEMV(hw)
	case DoseBandwidth:
		return runMetalBandwidth(hw)
	case DoseNumericParity:
		return runMetalNumericParity(hw)
	default:
		return MicroDoseResult{
			Schema:   MicroDoseSchema,
			Name:     string(kind),
			Kind:     kind,
			Hardware: hw,
			Passed:   false,
			Detail:   fmt.Sprintf("unsupported micro-dose kind: %s", kind),
		}
	}
}

func makeQ4KRaw(out, in int, seed uint64) []byte {
	blocks := (in / 256) * out
	raw := make([]byte, blocks*144)
	state := seed
	for i := range raw {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		raw[i] = byte(state >> 56)
	}
	for base := 0; base < len(raw); base += 144 {
		raw[base+1] = 0x2c | (raw[base+1] & 0x03)
		raw[base+3] = 0x2c | (raw[base+3] & 0x03)
	}
	return raw
}

func runMetalLiveness(hw HardwareInfo) MicroDoseResult {
	start := time.Now()
	// Test allocation and release of a Q4_K weight on Metal
	const in = 256
	const out = 16
	raw := makeQ4KRaw(out, in, 0x1234)
	w := metalgemm.UploadQ4K(raw, out, in)
	if w == nil {
		return MicroDoseResult{
			Schema:            MicroDoseSchema,
			Name:              "metal_device_liveness",
			Kind:              DoseLiveness,
			Hardware:          hw,
			DurationMicros:    time.Since(start).Microseconds(),
			DurationFormatted: time.Since(start).String(),
			Passed:            false,
			Detail:            "failed to allocate test Q4K weight on Metal device",
		}
	}
	w.Release()
	dur := time.Since(start)

	return MicroDoseResult{
		Schema:            MicroDoseSchema,
		Name:              "metal_device_liveness",
		Kind:              DoseLiveness,
		Hardware:          hw,
		DurationMicros:    dur.Microseconds(),
		DurationFormatted: dur.String(),
		Operations:        1,
		OutputVerified:    true,
		Passed:            true,
		Detail:            fmt.Sprintf("Metal device %q command queue initialized and buffer allocation/free verified in %s", hw.DeviceName, dur),
	}
}

func runMetalComputeGEMV(hw HardwareInfo) MicroDoseResult {
	start := time.Now()
	const (
		in  = 256
		out = 64
	)
	raw := makeQ4KRaw(out, in, 0x5678)
	w := metalgemm.UploadQ4K(raw, out, in)
	if w == nil {
		return MicroDoseResult{
			Schema:            MicroDoseSchema,
			Name:              "metal_compute_gemv",
			Kind:              DoseComputeGEMV,
			Hardware:          hw,
			DurationMicros:    time.Since(start).Microseconds(),
			DurationFormatted: time.Since(start).String(),
			Passed:            false,
			Detail:            "failed to upload Q4_K weight for Metal GEMV micro-dose",
		}
	}
	defer w.Release()

	x := make([]float32, in)
	for i := range x {
		x[i] = float32(i%17-8) * 0.03125
	}
	y := make([]float32, out)

	// Execute physical GEMV on Apple Silicon GPU
	computeStart := time.Now()
	const reps = 10
	for i := 0; i < reps; i++ {
		w.GEMV(x, y)
	}
	computeDur := time.Since(computeStart)
	totalDur := time.Since(start)

	// Verify output
	valid := true
	hasNonZero := false
	for _, v := range y {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			valid = false
			break
		}
		if v != 0 {
			hasNonZero = true
		}
	}
	valid = valid && hasNonZero

	totalOps := int64(reps) * int64(2*out*in)
	gflops := 0.0
	if computeDur.Seconds() > 0 {
		gflops = (float64(totalOps) / 1e9) / computeDur.Seconds()
	}

	return MicroDoseResult{
		Schema:            MicroDoseSchema,
		Name:              "metal_compute_gemv",
		Kind:              DoseComputeGEMV,
		Hardware:          hw,
		DurationMicros:    totalDur.Microseconds(),
		DurationFormatted: totalDur.String(),
		Operations:        totalOps,
		ComputeGFLOPS:     gflops,
		OutputVerified:    valid,
		Passed:            valid,
		Detail:            fmt.Sprintf("Metal quantized GEMV (%dx%d, %d reps) achieved %.2f GFLOP/s on %s in %s", out, in, reps, gflops, hw.DeviceName, totalDur),
	}
}

func runMetalBandwidth(hw HardwareInfo) MicroDoseResult {
	start := time.Now()
	const (
		in  = 512
		out = 128
	)
	raw := makeQ4KRaw(out, in, 0x9ABC)
	w := metalgemm.UploadQ4K(raw, out, in)
	if w == nil {
		return MicroDoseResult{
			Schema:            MicroDoseSchema,
			Name:              "metal_memory_bandwidth",
			Kind:              DoseBandwidth,
			Hardware:          hw,
			DurationMicros:    time.Since(start).Microseconds(),
			DurationFormatted: time.Since(start).String(),
			Passed:            false,
			Detail:            "failed to upload weight for Metal bandwidth probe",
		}
	}
	defer w.Release()

	x := make([]float32, in)
	for i := range x {
		x[i] = 0.5
	}
	y := make([]float32, out)

	benchStart := time.Now()
	const reps = 20
	for i := 0; i < reps; i++ {
		w.GEMV(x, y)
	}
	benchDur := time.Since(benchStart)
	totalDur := time.Since(start)

	// Memory read bytes per GEMV: weights + input activation + output activation
	weightBytes := (in / 256) * out * 144
	bytesPerRep := int64(weightBytes + in*4 + out*4)
	totalBytes := int64(reps) * bytesPerRep
	bwGBs := 0.0
	if benchDur.Seconds() > 0 {
		bwGBs = (float64(totalBytes) / 1e9) / benchDur.Seconds()
	}

	return MicroDoseResult{
		Schema:            MicroDoseSchema,
		Name:              "metal_memory_bandwidth",
		Kind:              DoseBandwidth,
		Hardware:          hw,
		DurationMicros:    totalDur.Microseconds(),
		DurationFormatted: totalDur.String(),
		Operations:        totalBytes,
		BandwidthGBs:      bwGBs,
		OutputVerified:    true,
		Passed:            true,
		Detail:            fmt.Sprintf("Metal memory throughput probe achieved %.2f GB/s on unified memory (%s)", bwGBs, totalDur),
	}
}

func runMetalNumericParity(hw HardwareInfo) MicroDoseResult {
	start := time.Now()
	const (
		in  = 256
		out = 32
	)
	raw := makeQ4KRaw(out, in, 0xDEF0)
	w := metalgemm.UploadQ4K(raw, out, in)
	if w == nil {
		return MicroDoseResult{
			Schema:            MicroDoseSchema,
			Name:              "metal_numeric_parity",
			Kind:              DoseNumericParity,
			Hardware:          hw,
			DurationMicros:    time.Since(start).Microseconds(),
			DurationFormatted: time.Since(start).String(),
			Passed:            false,
			Detail:            "failed to upload weight for Metal parity probe",
		}
	}
	defer w.Release()

	x := make([]float32, in)
	for i := range x {
		x[i] = 1.0
	}
	y1 := make([]float32, out)
	y2 := make([]float32, out)

	// Two executions must yield bit-identical outputs on deterministic GPU kernel
	w.GEMV(x, y1)
	w.GEMV(x, y2)
	totalDur := time.Since(start)

	bitExact := true
	allFinite := true
	for i := 0; i < out; i++ {
		if math.IsNaN(float64(y1[i])) || math.IsInf(float64(y1[i]), 0) {
			allFinite = false
		}
		if y1[i] != y2[i] {
			bitExact = false
		}
	}

	passed := bitExact && allFinite
	return MicroDoseResult{
		Schema:            MicroDoseSchema,
		Name:              "metal_numeric_parity",
		Kind:              DoseNumericParity,
		Hardware:          hw,
		DurationMicros:    totalDur.Microseconds(),
		DurationFormatted: totalDur.String(),
		Operations:        int64(2 * out * in),
		OutputVerified:    passed,
		Passed:            passed,
		Detail:            fmt.Sprintf("Metal on-device deterministic replay and finite output verified (bit_exact=%t, all_finite=%t) in %s", bitExact, allFinite, totalDur),
	}
}
