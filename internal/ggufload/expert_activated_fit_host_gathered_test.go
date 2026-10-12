package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_activated_fit_host_gathered_test.go — the V4.1 packed Engram table is gathered row-wise
// on the host, so the activated-expert fit must not charge it as device dense base (it was ~60 of
// the 63.09 GiB DeviceBaseBytes on the Flash checkpoint). Its bytes move to HostGatheredResident,
// and the plan carries a bounded host page reserve for it instead.

func v41HostGatheredHeader(withEngram bool) *File {
	const hidden, experts = 256, 4
	f := &File{
		Metadata: ds41Meta("deepseek41"),
		Tensors: []TensorInfo{
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{hidden, hidden, experts}, Type: TensorQ2_K},
			{Name: "blk.0.ffn_up_exps.weight", Dims: []uint64{hidden, hidden, experts}, Type: TensorQ2_K},
			{Name: "blk.0.ffn_down_exps.weight", Dims: []uint64{hidden, hidden, experts}, Type: TensorQ3_K},
			{Name: "blk.0.attn_q_a.weight", Dims: []uint64{hidden, 64}, Type: TensorQ2_K},
			{Name: "blk.0.ffn_norm.weight", Dims: []uint64{hidden}, Type: TensorF32},
		},
	}
	if withEngram {
		f.Tensors = append(f.Tensors, TensorInfo{Name: "blk.1.engram_embd.weight", Dims: []uint64{256, 4096}, Type: TensorQ2_K})
	}
	return f
}

func v41HostGatheredSource(t *testing.T, withEngram bool) *WeightSource {
	t.Helper()
	ws, err := NewWeightSource(v41HostGatheredHeader(withEngram), nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// fak-test:runtime fast est=5ms lane=default
func TestRoutedExpertActiveSetMovesEngramTableToHostGathered(t *testing.T) {
	with, ok, err := v41HostGatheredSource(t, true).RoutedExpertActiveSet()
	if err != nil || !ok {
		t.Fatalf("active set with engram: ok=%v err=%v", ok, err)
	}
	without, ok, err := v41HostGatheredSource(t, false).RoutedExpertActiveSet()
	if err != nil || !ok {
		t.Fatalf("active set without engram: ok=%v err=%v", ok, err)
	}
	info, _ := v41HostGatheredSource(t, true).Tensor("blk.1.engram_embd.weight")
	engram, err := tensorPayloadBytes(info)
	if err != nil || engram == 0 {
		t.Fatalf("engram payload=%d err=%v", engram, err)
	}
	if with.NonExpertResident != without.NonExpertResident || with.NonExpertParams != without.NonExpertParams {
		t.Fatalf("engram table charged as non-expert resident: with=%d/%d without=%d/%d",
			with.NonExpertResident, with.NonExpertParams, without.NonExpertResident, without.NonExpertParams)
	}
	if with.HostGatheredResident != int64(engram) || without.HostGatheredResident != 0 {
		t.Fatalf("HostGatheredResident with=%d without=%d, want %d and 0", with.HostGatheredResident, without.HostGatheredResident, engram)
	}
	if with.RoutedResident != without.RoutedResident || with.ActiveBytesPerToken != without.ActiveBytesPerToken {
		t.Fatalf("engram table moved the routed band or the swept per-token bytes: %+v vs %+v", with, without)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestActivatedExpertFitChargesEngramAsHostPageReserve(t *testing.T) {
	const budget = int64(1 << 30)
	with, ok, err := v41HostGatheredSource(t, true).ActivatedExpertFitFor(budget)
	if err != nil || !ok {
		t.Fatalf("fit with engram: ok=%v err=%v", ok, err)
	}
	without, ok, err := v41HostGatheredSource(t, false).ActivatedExpertFitFor(budget)
	if err != nil || !ok {
		t.Fatalf("fit without engram: ok=%v err=%v", ok, err)
	}
	if with.DeviceBaseBytes != without.DeviceBaseBytes || with.RingBytes != without.RingBytes {
		t.Fatalf("engram table moved the device placement: base %d vs %d, ring %d vs %d",
			with.DeviceBaseBytes, without.DeviceBaseBytes, with.RingBytes, without.RingBytes)
	}
	if with.HostGatheredBytes <= 0 || with.HostGatheredPageReserve() != min(with.HostGatheredBytes, HostGatheredPageReserveCap) {
		t.Fatalf("host gathered=%d reserve=%d", with.HostGatheredBytes, with.HostGatheredPageReserve())
	}
	if _, ok := without.HostGatheredDemand(); ok {
		t.Fatal("a checkpoint without host-gathered tables grew a host page row")
	}
	var hostRow int64
	for _, d := range with.MemoryPlan() {
		if d.Detail == HostGatheredDetail {
			if d.DeviceScoped() || d.Class == compute.MemoryOffload {
				t.Fatalf("engram page reserve row is not a host page row: %+v", d)
			}
			hostRow += d.Bytes
		}
	}
	if hostRow != with.HostGatheredPageReserve() {
		t.Fatalf("plan host-gathered row=%d, want %d", hostRow, with.HostGatheredPageReserve())
	}
	if with.MemoryPlan().DeviceTotal() != without.MemoryPlan().DeviceTotal() {
		t.Fatal("engram page reserve leaked into the device total")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestHostGatheredPageReserveIsCapped(t *testing.T) {
	big := ActivatedExpertFit{HostGatheredBytes: 60 << 30}
	if got := big.HostGatheredPageReserve(); got != HostGatheredPageReserveCap {
		t.Fatalf("60 GiB table reserve=%d, want the cap %d", got, HostGatheredPageReserveCap)
	}
	small := ActivatedExpertFit{HostGatheredBytes: 1 << 20}
	if got := small.HostGatheredPageReserve(); got != 1<<20 {
		t.Fatalf("1 MiB table reserve=%d, want the table", got)
	}
}
