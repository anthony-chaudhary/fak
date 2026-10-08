package model

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41DenseQ3PromiseBackend struct {
	*v41DenseTestBackend
	admitQ3 bool
}

func (b *v41DenseQ3PromiseBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	if dt == compute.Q3_K && !b.admitQ3 {
		return false
	}
	return b.v41DenseTestBackend.SupportsDeviceWeightDtype(dt)
}
func v41DenseQ3Fixture(t *testing.T) *Model {
	t.Helper()
	m := v41IncrementalExpertFixture(t, false, false)
	name := layerName(0, "ffn.shared_experts.w2.weight")
	m.kqw[name] = q3kFixtureTensor(m.Cfg.HiddenSize, m.Cfg.MoEIntermediateSize)
	delete(m.manifest, name)
	return m
}

// fak-test:runtime medium est=12s lane=default
func TestV41DenseProjectionQ3SharedDownRoutes(t *testing.T) {
	t.Parallel()
	descriptor, ok := LookupQuantDescriptor(kindQ3K)
	if !ok || !descriptor.SupportsHAL() || descriptor.Dtype() != compute.Q3_K {
		t.Fatal("existing Q3_K descriptor does not advertise its native HAL contract")
	}
	for _, admit := range []bool{true, false} {
		name := "promised"
		if !admit {
			name = "explicitly-declined"
		}
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			m, oracle := v41DenseQ3Fixture(t), v41DenseQ3Fixture(t)
			const leaf = "ffn.shared_experts.w2.weight"
			weightName := layerName(0, leaf)
			quant := m.kqw[weightName]
			packedBefore := append([]byte(nil), quant.raw...)
			hostTensor := descriptor.NewHostTensor(quant.out, quant.in, quant.raw)
			packed, ok := hostTensor.Buf().(compute.HostBuffer)
			if !ok || len(packed.I8()) != len(packedBefore) || len(packed.F32()) != 0 {
				t.Fatal("descriptor Q3_K factory expanded packed weights")
			}
			b := &v41DenseQ3PromiseBackend{v41DenseTestBackend: newV41DenseTestBackend(), admitQ3: admit}
			s := v41DenseTestSession(t, m, b)
			history := []int{1, 2, 3}
			for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
				phase := "prefill"
				if index == 1 {
					phase = "decode"
				}
				before := v41DenseTestPhase(t, m, phase)
				from := len(b.ops)
				var got []float32
				if index == 1 {
					got = s.Step(ids[0])
				} else {
					got = s.Prefill(ids)
				}
				if index > 0 {
					history = append(history, ids...)
				}
				v41DenseTestParity(t, got, lastLogits(oracle.Forward(history)))
				named, _, _ := v41DenseTestOps(s, b.v41DenseTestBackend, from)
				sharedRows := 0
				for _, op := range named[leaf] {
					sharedRows += op.rows
				}
				wantSharedRows := len(ids)
				if !admit {
					wantSharedRows = 0
				}
				if sharedRows != wantSharedRows {
					t.Errorf("actual shared Q3_K projection device rows=%d want %d", sharedRows, wantSharedRows)
				}
				delta := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), before)
				wantHostRows := 0
				wantDeviceRows := 7 * len(ids)
				if !admit {
					wantHostRows = len(ids)
					wantDeviceRows -= len(ids)
				}
				if delta["dense_projection_device_rows"] != float64(wantDeviceRows) || delta["dense_projection_host_rows"] != float64(wantHostRows) || delta["dense_projection_host_calls"] != float64(wantHostRows) {
					t.Errorf("Q3_K capability outcome ledger=%v want device_rows=%d host_rows=%d", delta, wantDeviceRows, wantHostRows)
				}
				calls := 0
				for _, ops := range named {
					calls += len(ops)
				}
				if delta["dense_projection_device_calls"] != float64(calls) {
					t.Error("Q3_K declined projection was charged as a selected device call")
				}
				var staged compute.Tensor
				stagedFound := false
				for key, tensor := range s.halW {
					if key == descriptor.KeyPrefix()+weightName {
						staged, stagedFound = tensor, true
					}
				}
				if admit {
					if !stagedFound || staged.Dtype != compute.Q3_K {
						t.Fatal("actual shared projection did not stage descriptor-selected packed Q3_K")
					}
					storage, ok := staged.Buf().(compute.HostBuffer)
					if !ok || len(storage.I8()) != len(packedBefore) || len(storage.F32()) != 0 {
						t.Fatal("selected Q3_K software backend expanded the staged weight")
					}
				} else if stagedFound {
					t.Error("explicitly unsupported shared Q3_K projection was staged")
				}
				if m.has(weightName) || len(quant.raw) != len(packedBefore) || !bytes.Equal(quant.raw, packedBefore) {
					t.Error("shared Q3_K projection changed packed weight storage")
				}
				raw, err := json.Marshal(delta)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("dense_q3 phase=%s tokens=%d backend_promised=%t packed_bytes=%d shared_device_rows=%d attribution=%s", phase, len(ids), admit, len(quant.raw), sharedRows, raw)
			}
		})
	}
}
