package model

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"testing"
)

type deviceBudgetReader struct{ calls atomic.Int64 }

func (r *deviceBudgetReader) ReadAt(p []byte, _ int64) (int, error) {
	r.calls.Add(1)
	clear(p)
	return len(p), nil
}

func deviceBudgetShard(t *testing.T, tier *ExpertCheckpointTier, r *deviceBudgetReader, layer, experts, rows, cols int, proj string, quant ExpertCheckpointQuant, bytesPerBlock int64) {
	t.Helper()
	stride := int64(rows) * int64(cols) / 256 * bytesPerBlock
	err := tier.AddShard(r, stride*int64(experts), []FusedExpertTensor{{
		Name: fmt.Sprintf("blk.%d.%s", layer, proj), Layer: layer, Proj: proj,
		Arch: "deepseek41", Quant: quant, Experts: experts, Rows: rows, Cols: cols,
	}})
	if err != nil {
		t.Fatalf("index shard layer=%d projection=%s: %v", layer, proj, err)
	}
}

func deviceBudgetTriplet(t *testing.T, tier *ExpertCheckpointTier, r *deviceBudgetReader, layer, experts, hidden, intermediate int) int64 {
	t.Helper()
	deviceBudgetShard(t, tier, r, layer, experts, intermediate, hidden, "gate_proj", ExpertCheckpointQ2K, 84)
	deviceBudgetShard(t, tier, r, layer, experts, intermediate, hidden, "up_proj", ExpertCheckpointQ2K, 84)
	deviceBudgetShard(t, tier, r, layer, experts, hidden, intermediate, "down_proj", ExpertCheckpointQ3K, 110)
	return int64(hidden) * int64(intermediate) / 256 * (84 + 84 + 110)
}

func deviceBudgetNoIO(t *testing.T, tier *ExpertCheckpointTier, r *deviceBudgetReader) {
	t.Helper()
	if calls := r.calls.Load(); calls != 0 {
		t.Fatalf("admission performed %d payload reads", calls)
	}
	st := tier.Stats()
	if st.Reads != 0 || st.BytesRead != 0 || st.BudgetBytes != 0 || st.ResidentBytes != 0 || st.ResidentCount != 0 {
		t.Fatalf("admission changed host accounting: reads=%d bytes=%d budget=%d resident=%d count=%d", st.Reads, st.BytesRead, st.BudgetBytes, st.ResidentBytes, st.ResidentCount)
	}
}

// fak-test:runtime fast est=1s
func TestExpertCheckpointDeviceBudgetMixedNativeMaximum(t *testing.T) {
	t.Parallel()
	tier, reader := NewExpertCheckpointTier(0), &deviceBudgetReader{}
	deviceBudgetTriplet(t, tier, reader, 0, 5, 256, 256)
	const hidden, intermediate = 5120, 2304
	want := deviceBudgetTriplet(t, tier, reader, 7, 1, hidden, intermediate)
	if want != 12810240 {
		t.Fatalf("independent mixed quant oracle = %d", want)
	}
	if !tier.Has("model.layers.7.ffn.experts.0.w1.weight") || !tier.Has("model.layers.7.ffn.experts.0.w2.weight") || !tier.Has("model.layers.7.ffn.experts.0.w3.weight") {
		t.Fatal("fixture did not index native DeepSeek expert names")
	}
	err := tier.SetDeviceRingBudget(want - 1)
	if !errors.Is(err, ErrExpertCheckpointDeviceBudget) {
		t.Fatalf("undersized ring class = %v", err)
	}
	var refusal *ExpertCheckpointDeviceBudgetError
	if !errors.As(err, &refusal) || refusal.Tensor != "model.layers.7.ffn.experts.0.w1.weight" || refusal.Bytes != want || refusal.Budget != want-1 {
		t.Fatalf("undersized ring operands: %v", err)
	}
	if tier.DeviceRingBudget() != 0 {
		t.Fatal("refused admission mutated device budget")
	}
	deviceBudgetNoIO(t, tier, reader)
	if err := tier.SetDeviceRingBudget(want); err != nil {
		t.Fatalf("exact largest complete expert admission: %v", err)
	}
	if tier.DeviceRingBudget() != want {
		t.Fatalf("device budget = %d, want %d", tier.DeviceRingBudget(), want)
	}
	deviceBudgetNoIO(t, tier, reader)
}

// fak-test:runtime fast est=1s
func TestExpertCheckpointDeviceBudgetLeavesHostStreamThrough(t *testing.T) {
	t.Parallel()
	tier, reader := NewExpertCheckpointTier(0), &deviceBudgetReader{}
	budget := deviceBudgetTriplet(t, tier, reader, 0, 1, 256, 256)
	if err := tier.SetDeviceRingBudget(budget); err != nil {
		t.Fatal(err)
	}
	deviceBudgetNoIO(t, tier, reader)
	for i := 0; i < 2; i++ {
		w, err := tier.fault("model.layers.0.ffn.experts.0.w1.weight")
		if err != nil || w.kq == nil || len(w.kq.raw) != 256*84 {
			t.Fatalf("stream-through fault %d: bytes present=%t error=%v", i, w.kq != nil, err)
		}
	}
	st := tier.Stats()
	if reader.calls.Load() != 2 || st.Reads != 2 || st.BytesRead != 2*256*84 || st.Hits != 0 {
		t.Fatalf("repeat stream-through reads=%d calls=%d bytes=%d hits=%d", st.Reads, reader.calls.Load(), st.BytesRead, st.Hits)
	}
	if st.BudgetBytes != 0 || st.ResidentBytes != 0 || st.ResidentCount != 0 || st.PeakBytes != 0 || tier.DeviceRingBudget() != budget {
		t.Fatal("device configuration changed the zero-host stream-through contract")
	}
}

// fak-test:runtime fast est=1s
func TestExpertCheckpointDeviceBudgetMalformedAndLegacyZero(t *testing.T) {
	t.Parallel()
	for _, scenario := range []string{"empty", "missing_down", "unknown_projection", "overflow", "negative"} {
		t.Run(scenario, func(t *testing.T) {
			t.Parallel()
			tier, reader := NewExpertCheckpointTier(0), &deviceBudgetReader{}
			switch scenario {
			case "missing_down":
				deviceBudgetShard(t, tier, reader, 0, 1, 256, 256, "gate_proj", ExpertCheckpointQ2K, 84)
				deviceBudgetShard(t, tier, reader, 0, 1, 256, 256, "up_proj", ExpertCheckpointQ2K, 84)
			case "unknown_projection":
				deviceBudgetTriplet(t, tier, reader, 0, 1, 256, 256)
				deviceBudgetShard(t, tier, reader, 0, 1, 256, 256, "other_proj", ExpertCheckpointQ2K, 84)
			case "overflow":
				rows := int(math.MaxInt64 / 256)
				for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
					deviceBudgetShard(t, tier, reader, 0, 1, rows, 256, proj, ExpertCheckpointQ6K, 210)
				}
			}
			if err := tier.SetDeviceRingBudget(0); err != nil {
				t.Fatalf("legacy zero should not inspect incomplete index: %v", err)
			}
			budget := int64(math.MaxInt64)
			if scenario == "negative" {
				budget = -1
			}
			if err := tier.SetDeviceRingBudget(budget); !errors.Is(err, ErrGGUFExpertMetadata) {
				t.Fatalf("malformed admission class = %v", err)
			}
			if tier.DeviceRingBudget() != 0 {
				t.Fatal("malformed admission mutated device budget")
			}
			deviceBudgetNoIO(t, tier, reader)
		})
	}
	var absent *ExpertCheckpointTier
	if absent.DeviceRingBudget() != 0 {
		t.Fatal("nil tier query is nonzero")
	}
	if err := absent.SetDeviceRingBudget(1); !errors.Is(err, ErrGGUFExpertMetadata) {
		t.Fatalf("nil tier admission class = %v", err)
	}
}

// fak-test:runtime fast est=1s
func TestExpertCheckpointDeviceBudgetSealsAdmission(t *testing.T) {
	t.Parallel()
	tier, reader := NewExpertCheckpointTier(0), &deviceBudgetReader{}
	budget := deviceBudgetTriplet(t, tier, reader, 0, 1, 256, 256)
	if err := tier.SetDeviceRingBudget(budget); err != nil {
		t.Fatal(err)
	}
	(&Model{}).SetExpertCheckpoint(tier)
	if err := tier.SetDeviceRingBudget(budget); err != nil {
		t.Fatalf("same budget is idempotent: %v", err)
	}
	for _, next := range []int64{0, budget - 1, budget + 1} {
		err := tier.SetDeviceRingBudget(next)
		var refusal *ExpertCheckpointDeviceBudgetError
		if !errors.Is(err, ErrExpertCheckpointDeviceBudget) || !errors.As(err, &refusal) || refusal.Bytes != budget || refusal.Budget != next {
			t.Fatalf("sealed budget change to %d class/operand: %v", next, err)
		}
		if tier.DeviceRingBudget() != budget {
			t.Fatal("sealed budget changed on refusal")
		}
	}
	before := tier.Stats().Tensors
	err := tier.AddShardData(reader, 256*84, nil, []FusedExpertTensor{{Name: "blk.1.gate", Layer: 1, Proj: "gate_proj", Arch: "deepseek41", Quant: ExpertCheckpointQ2K, Experts: 1, Rows: 256, Cols: 256}})
	if !errors.Is(err, ErrGGUFExpertMetadata) || tier.Stats().Tensors != before || tier.Has("model.layers.1.ffn.experts.0.w1.weight") {
		t.Fatalf("sealed index mutation class/state: %v", err)
	}
	deviceBudgetNoIO(t, tier, reader)
}
