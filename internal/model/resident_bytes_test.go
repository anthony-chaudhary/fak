package model

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// residencyBackend is a device backend whose raw i-quant kernel is either built (native)
// or missing, in which case Upload would widen the weight to Q8_0.
type residencyBackend struct {
	compute.Backend
	native bool
}

func (b residencyBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}

func (b residencyBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return compute.IsRawIQ(dt)
}

func (b residencyBackend) RequantizesWeightDtype(dt compute.Dtype) (compute.Dtype, bool) {
	if b.native || !compute.IsRawIQ(dt) {
		return dt, false
	}
	return compute.Q8_0, true
}

func residencyModel(t *testing.T) *Model {
	t.Helper()
	const out, in = 64, 512
	nblk := in / kindIQ3XXS.blockWeights()
	qt := &kQuantTensor{out: out, in: in, nblk: nblk, kind: kindIQ3XXS}
	m := &Model{kqw: map[string]*kQuantTensor{
		"model.layers.0.mlp.down_proj.weight":           qt,
		"model.layers.0.mlp.experts.0.down_proj.weight": {out: out, in: in, nblk: nblk, kind: kindIQ3XXS},
	}}
	file := 2 * int64(out) * int64(qt.rowBytes())
	m.SetFileWeightBytes(file)
	return m
}

// fak-test:runtime fast est=100ms
func TestResidentBytesRatioAlarmsOnRequantizedLoad(t *testing.T) {
	compute.ResetWeightRequantsForTest()
	t.Cleanup(compute.ResetWeightRequantsForTest)
	m := residencyModel(t)
	r := m.WeightResidency(residencyBackend{Backend: compute.Default()})

	if !r.Degraded || r.Ratio <= ResidentBytesRatioAlarm {
		t.Fatalf("requantized load not degraded: %+v", r)
	}
	if len(r.RequantizedTypes) != 1 || r.RequantizedTypes[0] != "IQ3_XXS->Q8_0" {
		t.Fatalf("requantized types = %v, want [IQ3_XXS->Q8_0]", r.RequantizedTypes)
	}
	if r.ResidentWeightBytes <= r.FileWeightBytes {
		t.Fatalf("resident %d not above file %d", r.ResidentWeightBytes, r.FileWeightBytes)
	}
	if !strings.HasPrefix(r.PerfClause(), " "+RequantizedWeightsDegradedToken+"=") {
		t.Fatalf("perf clause %q lacks %q", r.PerfClause(), RequantizedWeightsDegradedToken)
	}
	if err := r.Admit(true, false); !errors.Is(err, ErrRequantizedWeights) {
		t.Fatalf("appliance admit = %v, want ErrRequantizedWeights", err)
	}
	if err := r.Admit(true, true); err != nil {
		t.Fatalf("explicitly admitted appliance load refused: %v", err)
	}
	if err := r.Admit(false, false); err != nil {
		t.Fatalf("off-appliance load refused: %v", err)
	}
}

// fak-test:runtime fast est=100ms
func TestResidentBytesRatioQuietOnNativeKernelLoad(t *testing.T) {
	compute.ResetWeightRequantsForTest()
	t.Cleanup(compute.ResetWeightRequantsForTest)
	m := residencyModel(t)
	r := m.WeightResidency(residencyBackend{Backend: compute.Default(), native: true})

	if r.Degraded || r.Ratio != 1 || len(r.RequantizedTypes) != 0 {
		t.Fatalf("native-kernel load not quiet: %+v", r)
	}
	if r.PerfClause() != "" {
		t.Fatalf("native-kernel perf clause = %q, want empty", r.PerfClause())
	}
	if err := r.Admit(true, false); err != nil {
		t.Fatalf("native-kernel appliance load refused: %v", err)
	}
}

// fak-test:runtime fast est=100ms
func TestResidentBytesRatioCountsObservedRequants(t *testing.T) {
	compute.ResetWeightRequantsForTest()
	t.Cleanup(compute.ResetWeightRequantsForTest)
	m := residencyModel(t)
	be := residencyBackend{Backend: compute.Default(), native: true}
	compute.RecordWeightRequant(compute.IQ4_XS, compute.Q8_0, 1000, 2000)

	r := m.WeightResidency(be)
	if len(r.RequantizedTypes) != 1 || r.RequantizedTypes[0] != "IQ4_XS->Q8_0" {
		t.Fatalf("observed requant not on receipt: %+v", r)
	}
	if r.ResidentWeightBytes != r.FileWeightBytes+1000 {
		t.Fatalf("resident %d, want file %d + 1000", r.ResidentWeightBytes, r.FileWeightBytes)
	}
}
