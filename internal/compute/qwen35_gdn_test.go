package compute

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestQwen35GDNTypedRefusalsCarryStableIdentity(t *testing.T) {
	if Qwen35GDNCUDAPath != "cuda/qwen35-gdn-ssm-decode-v1" {
		t.Fatalf("Qwen35GDNCUDAPath = %q", Qwen35GDNCUDAPath)
	}
	if Qwen35GDNParityCosineMin != 0.999 {
		t.Fatalf("Qwen35GDNParityCosineMin = %v, want 0.999", Qwen35GDNParityCosineMin)
	}

	var err error = &Qwen35GDNGeometryError{Operand: "geometry", Reason: "invalid head grouping"}
	var geometry *Qwen35GDNGeometryError
	if !errors.As(err, &geometry) || !strings.Contains(err.Error(), "invalid head grouping") {
		t.Fatalf("geometry refusal lost its type/detail: %T %v", err, err)
	}
	err = &Qwen35GDNKernelError{Stage: "recurrent-gated-norm", Code: 50001}
	var kernel *Qwen35GDNKernelError
	if !errors.As(err, &kernel) || !strings.Contains(err.Error(), "no CPU fallback") {
		t.Fatalf("kernel refusal lost its type/fail-closed detail: %T %v", err, err)
	}
	err = &Qwen35GDNAllocationError{Operand: "output", Bytes: 4096}
	var allocation *Qwen35GDNAllocationError
	if !errors.As(err, &allocation) || !strings.Contains(err.Error(), "managed-memory fallback refused") {
		t.Fatalf("allocation refusal lost its device-only detail: %T %v", err, err)
	}
	err = &Qwen35GDNInvalidStateError{Operand: "conv_state"}
	var invalid *Qwen35GDNInvalidStateError
	if !errors.As(err, &invalid) || !strings.Contains(err.Error(), "free it and reinitialize state") {
		t.Fatalf("invalid-state refusal lost its fail-closed detail: %T %v", err, err)
	}
}

func qwen35GDNGeometryTensors(hidden, nK, nV, kHd, vHd, kernel int) [12]Tensor {
	keyDim := nK * kHd
	valueDim := nV * vHd
	convDim := 2*keyDim + valueDim
	return [12]Tensor{
		{Shape: []int{hidden}},
		{Shape: []int{convDim, hidden}},
		{Shape: []int{valueDim, hidden}},
		{Shape: []int{nV, hidden}},
		{Shape: []int{nV, hidden}},
		{Shape: []int{convDim, 1, kernel}},
		{Shape: []int{nV}},
		{Shape: []int{nV}},
		{Shape: []int{vHd}},
		{Shape: []int{hidden, valueDim}},
		{Shape: []int{kernel - 1, convDim}},
		{Shape: []int{nV, kHd, vHd}},
	}
}

func validateQwen35GDNTensors(tensors [12]Tensor, hidden, nK, nV, kHd, vHd, kernel int, eps float32) error {
	_, _, _, _, err := validateQwen35GDNGeometry(
		tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
		tensors[5], tensors[6], tensors[7], tensors[8], tensors[9],
		tensors[10], tensors[11], nK, nV, kHd, vHd, kernel, eps,
	)
	return err
}

func TestQwen35GDNGeometryAcceptsCPUValidKernelOne(t *testing.T) {
	const hidden, nK, nV, kHd, vHd, kernel = 8, 1, 2, 2, 2, 1
	tensors := qwen35GDNGeometryTensors(hidden, nK, nV, kHd, vHd, kernel)
	if got := tensors[10].Shape; len(got) != 2 || got[0] != 0 {
		t.Fatalf("K=1 conv-state shape = %v, want [0 convDim]", got)
	}
	if err := validateQwen35GDNTensors(tensors, hidden, nK, nV, kHd, vHd, kernel, 1e-5); err != nil {
		t.Fatalf("CPU-valid K=1 geometry refused: %v", err)
	}
}

func TestQwen35GDNAllocationsPreserveDecodeAndSequenceLayouts(t *testing.T) {
	const hidden, keyDim, valueDim, valueHeads, convDim = 8, 2, 4, 2, 8
	tests := []struct {
		name                  string
		prefix                string
		separator             string
		scratchRows, outRows  int
		wantFirst, wantOutput []int
		wantFirstName, wantQ  string
	}{
		{"decode", "", "_", 0, 0, []int{convDim}, []int{hidden}, "mixed", "q_norm"},
		{"sequence-api", "", "_", 0, 3, []int{convDim}, []int{3, hidden}, "mixed", "q_norm"},
		{"resident-panel", "gdn-", "-", 3, 3, []int{3, convDim}, []int{3, hidden}, "gdn-mixed", "gdn-q-norm"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := qwen35GDNAllocations(tt.prefix, tt.separator, tt.scratchRows, tt.outRows, hidden, keyDim, valueDim, valueHeads, convDim)
			if len(got) != 9 {
				t.Fatalf("allocation count = %d, want 9", len(got))
			}
			if got[0].name != tt.wantFirstName || !reflect.DeepEqual(got[0].shape, tt.wantFirst) {
				t.Fatalf("first allocation = %#v, want name=%q shape=%v", got[0], tt.wantFirstName, tt.wantFirst)
			}
			if got[8].name != tt.prefix+"output" || !reflect.DeepEqual(got[8].shape, tt.wantOutput) {
				t.Fatalf("output allocation = %#v, want name=%q shape=%v", got[8], tt.prefix+"output", tt.wantOutput)
			}
			if got[5].name != tt.wantQ {
				t.Fatalf("q-norm allocation name = %q, want %q", got[5].name, tt.wantQ)
			}
		})
	}
}

func TestQwen35GDNGeometryChecksDerivedGridAndProductCapacity(t *testing.T) {
	t.Run("checked-conv-grid-boundary", func(t *testing.T) {
		got, ok := qwen35GDNCheckedAdd(qwen35GDNMaxCInt+255, qwen35GDNMaxCInt, 255)
		if !ok || got != qwen35GDNMaxCInt+255 || got/256 <= 0 {
			t.Fatalf("checked conv grid numerator = %d, ok=%v", got, ok)
		}
		if _, ok := qwen35GDNCheckedAdd(qwen35GDNMaxCInt, qwen35GDNMaxCInt, 255); ok {
			t.Fatal("int32-bounded convDim+255 unexpectedly succeeded")
		}
	})

	t.Run("fused-grid-sum", func(t *testing.T) {
		// Each scalar and individual derived dimension fits int32, but
		// convDim+valueDim+2*nV does not. Shapes are metadata only: no huge
		// allocation is performed by this deterministic preflight test.
		const hidden, nK, nV, kHd, vHd, kernel = 1, 1, 1 << 30, 1, 1, 1
		tensors := qwen35GDNGeometryTensors(hidden, nK, nV, kHd, vHd, kernel)
		err := validateQwen35GDNTensors(tensors, hidden, nK, nV, kHd, vHd, kernel, 1e-5)
		var geometry *Qwen35GDNGeometryError
		if !errors.As(err, &geometry) || !strings.Contains(err.Error(), "fused projection grid") {
			t.Fatalf("fused-grid overflow error = %T %v", err, err)
		}
	})

	t.Run("state-product", func(t *testing.T) {
		const hidden, nK, nV, kHd, vHd, kernel = 1, 1, 1 << 22, 1024, 1, 1
		tensors := qwen35GDNGeometryTensors(hidden, nK, nV, kHd, vHd, kernel)
		err := validateQwen35GDNTensors(tensors, hidden, nK, nV, kHd, vHd, kernel, 1e-5)
		var geometry *Qwen35GDNGeometryError
		if !errors.As(err, &geometry) || geometry.Operand != "recurrent_state" {
			t.Fatalf("state-product overflow error = %T %v", err, err)
		}
	})

	t.Run("non-finite-epsilon", func(t *testing.T) {
		const hidden, nK, nV, kHd, vHd, kernel = 8, 1, 2, 2, 2, 1
		tensors := qwen35GDNGeometryTensors(hidden, nK, nV, kHd, vHd, kernel)
		err := validateQwen35GDNTensors(tensors, hidden, nK, nV, kHd, vHd, kernel, float32(math.NaN()))
		var geometry *Qwen35GDNGeometryError
		if !errors.As(err, &geometry) || geometry.Operand != "rms_norm_epsilon" {
			t.Fatalf("NaN epsilon error = %T %v", err, err)
		}
	})
}
