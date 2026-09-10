//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestVulkanQ8CooperativeBatchMatchesDecodeRows isolates the Q8 cooperative
// prefill kernel from model recurrence and fusion. Each case uploads one Q8
// matrix, executes its panel once through BatchedMatMul, then executes the same
// panel row-by-row through MatMul while retaining the exact same device weight.
// The checkpoint-scale case uploads about 100 MiB of packed weights.
func TestVulkanQ8CooperativeBatchMatchesDecodeRows(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		vulkanQ8RegressionUnavailable(t, "native Q8 support is unavailable")
	}
	if !v.q8CooperativeMatrixActive(32) {
		vulkanQ8RegressionUnavailable(t, "native Q8 cooperative matrix pipeline is unavailable")
	}
	if v.q8CooperativeMatrixActive(1) {
		t.Fatal("single-row reference unexpectedly selects the cooperative matrix path")
	}

	for _, shape := range []struct {
		name            string
		out, in, tokens int
		seed            uint64
	}{
		{name: "p2_output_tail", out: 33, in: 64, tokens: 2, seed: 0x20033},
		{name: "p3_output_tail", out: 65, in: 96, tokens: 3, seed: 0x30065},
		{name: "p32_modest_tail", out: 67, in: 512, tokens: 32, seed: 0x7108},
		{name: "p32_qwen35_down_projection", out: 5120, in: 17408, tokens: 32, seed: 0x351708},
	} {
		t.Run(shape.name, func(t *testing.T) {
			if !v.q8CooperativeMatrixActive(shape.tokens) {
				t.Fatalf("P=%d unexpectedly does not select the cooperative matrix path", shape.tokens)
			}
			codes, scales := q8BatchDecodeDiagnosticWeights(shape.out, shape.in, shape.seed)
			input := q8BatchDecodeDiagnosticInput(shape.tokens, shape.in, shape.seed^0x9e3779b97f4a7c15)

			c := cpu()
			hostWeight := NewQ8(c, []int{shape.out, shape.in}, codes, scales, 32)
			hostInput := NewF32(c, []int{shape.tokens, shape.in}, input)
			deviceWeight := v.Upload(hostWeight, Q8_0)
			deviceInput := v.Upload(hostInput, F32)
			defer v.Free(deviceWeight)
			defer v.Free(deviceInput)

			batchedTensor := v.BatchedMatMul(deviceWeight, deviceInput, shape.tokens)
			batched := v.Read(batchedTensor)
			v.Free(batchedTensor)

			decoded := make([]float32, shape.tokens*shape.out)
			for token := 0; token < shape.tokens; token++ {
				row := input[token*shape.in : (token+1)*shape.in]
				deviceRow := v.Upload(NewF32(c, []int{shape.in}, row), F32)
				decodedTensor := v.MatMul(deviceWeight, deviceRow)
				copy(decoded[token*shape.out:], v.Read(decodedTensor))
				v.Free(decodedTensor)
				v.Free(deviceRow)
			}

			metrics := q8BatchDecodeMetrics(batched, decoded, shape.out)
			encoded, err := json.Marshal(map[string]any{
				"schema":               "fak.vulkan.q8-batch-decode-regression.v1",
				"scope":                "same resident Q8 weights; cooperative BatchedMatMul versus direct scalar MatMul calls",
				"device":               v.Tier(),
				"shape":                [2]int{shape.out, shape.in},
				"tokens":               shape.tokens,
				"elements_compared":    len(batched),
				"l2_residual":          metrics.l2Residual,
				"reference_l2":         metrics.referenceL2,
				"normalized_residual":  metrics.normalizedResidual,
				"max_abs_delta":        metrics.maxAbsDelta,
				"bitwise_mismatches":   metrics.bitwiseMismatches,
				"all_finite":           metrics.allFinite,
				"per_row_argmax_exact": metrics.argmaxExact,
				"argmax_matching_rows": metrics.argmaxMatchingRows,
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Log(string(encoded))

			if len(batched) != shape.tokens*shape.out || len(decoded) != len(batched) {
				t.Fatalf("output lengths batched=%d decoded=%d want=%d", len(batched), len(decoded), shape.tokens*shape.out)
			}
			if !metrics.allFinite {
				t.Fatal("batch/decode comparison contains a non-finite output or residual")
			}
			if !metrics.argmaxExact {
				t.Fatalf("per-row argmax mismatch at token %d: batched=%d decoded=%d", metrics.argmaxMismatchRow, metrics.batchedArgmax, metrics.decodedArgmax)
			}
			// Both paths use the same Q8_0 weights and identically quantize each
			// 32-value activation block. Only FP32 accumulation mechanics may
			// differ, so this gate is intentionally much tighter than Approx
			// model-level parity tolerances.
			if metrics.normalizedResidual > 2e-6 || metrics.maxAbsDelta > 2e-5 {
				t.Fatalf("Q8 P%d/decode divergence: normalized residual=%g max abs delta=%g bitwise mismatches=%d", shape.tokens, metrics.normalizedResidual, metrics.maxAbsDelta, metrics.bitwiseMismatches)
			}
		})
	}
}

func vulkanQ8RegressionUnavailable(t *testing.T, reason string) {
	t.Helper()
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		t.Fatal(reason)
	}
	t.Skip(reason)
}

func q8BatchDecodeDiagnosticWeights(out, in int, seed uint64) ([]int8, []float32) {
	codes := make([]int8, out*in)
	scales := make([]float32, out*(in/32))
	row := make([]float32, in)
	state := seed
	for output := 0; output < out; output++ {
		for i := range row {
			row[i] = q8BatchDecodeDiagnosticFloat(&state, 0.125)
		}
		rowCodes, rowScales := quantizeVecQ8(row, 32)
		copy(codes[output*in:], rowCodes)
		copy(scales[output*(in/32):], rowScales)
	}
	return codes, scales
}

func q8BatchDecodeDiagnosticInput(tokens, in int, seed uint64) []float32 {
	input := make([]float32, tokens*in)
	state := seed
	for i := range input {
		input[i] = q8BatchDecodeDiagnosticFloat(&state, 0.75)
	}
	return input
}

func q8BatchDecodeDiagnosticFloat(state *uint64, amplitude float32) float32 {
	*state = *state*6364136223846793005 + 1442695040888963407
	unit := float32(int32(uint32(*state>>40))-1<<23) / float32(1<<23)
	return unit * amplitude
}

type q8BatchDecodeDiagnosticMetrics struct {
	l2Residual         float64
	referenceL2        float64
	normalizedResidual float64
	maxAbsDelta        float64
	bitwiseMismatches  int
	allFinite          bool
	argmaxExact        bool
	argmaxMismatchRow  int
	batchedArgmax      int
	decodedArgmax      int
	argmaxMatchingRows int
}

func q8BatchDecodeMetrics(batched, decoded []float32, out int) q8BatchDecodeDiagnosticMetrics {
	m := q8BatchDecodeDiagnosticMetrics{allFinite: len(batched) == len(decoded), argmaxExact: true, argmaxMismatchRow: -1}
	if len(batched) != len(decoded) || out <= 0 || len(batched)%out != 0 {
		m.allFinite = false
		m.argmaxExact = false
		return m
	}
	var squaredResidual, squaredReference float64
	for i, batchValue := range batched {
		decodeValue := decoded[i]
		if !isFiniteF32(batchValue) || !isFiniteF32(decodeValue) {
			m.allFinite = false
		}
		delta := float64(batchValue) - float64(decodeValue)
		squaredResidual += delta * delta
		squaredReference += float64(decodeValue) * float64(decodeValue)
		m.maxAbsDelta = math.Max(m.maxAbsDelta, math.Abs(delta))
		if math.Float32bits(batchValue) != math.Float32bits(decodeValue) {
			m.bitwiseMismatches++
		}
	}
	m.l2Residual = math.Sqrt(squaredResidual)
	m.referenceL2 = math.Sqrt(squaredReference)
	if m.referenceL2 > 0 {
		m.normalizedResidual = m.l2Residual / m.referenceL2
	} else if m.l2Residual != 0 {
		m.normalizedResidual = math.Inf(1)
	}
	if math.IsNaN(m.l2Residual) || math.IsInf(m.l2Residual, 0) || math.IsNaN(m.referenceL2) || math.IsInf(m.referenceL2, 0) || math.IsNaN(m.normalizedResidual) || math.IsInf(m.normalizedResidual, 0) {
		m.allFinite = false
	}
	for token := 0; token < len(batched)/out; token++ {
		lo, hi := token*out, (token+1)*out
		batchArgmax := argmaxF32(batched[lo:hi])
		decodeArgmax := argmaxF32(decoded[lo:hi])
		if batchArgmax == decodeArgmax {
			m.argmaxMatchingRows++
		} else {
			m.argmaxExact = false
			if m.argmaxMismatchRow < 0 {
				m.argmaxMismatchRow = token
				m.batchedArgmax = batchArgmax
				m.decodedArgmax = decodeArgmax
			}
		}
	}
	return m
}

func isFiniteF32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
