package model

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// qwen35ThreePeriodModel is the tiny Qwen3.5 hybrid extended to three
// [linear, linear, linear, full] periods so a three-device plan has a legal cut per band.
func qwen35ThreePeriodModel(t *testing.T) *Model {
	t.Helper()
	cfg := qwen35HybridTestCfg()
	cfg.NumLayers = 12
	cfg.LayerTypes = nil
	for p := 0; p < 3; p++ {
		cfg.LayerTypes = append(cfg.LayerTypes, "linear_attention", "linear_attention", "linear_attention", "full_attention")
	}
	return NewSynthetic(cfg)
}

// planThreeDevices plans the tiny model over three devices whose budgets force one
// period per device, the same shape as the laptop's 4070 + two eGPUs.
func planThreeDevices(t *testing.T, cfg Config) DeviceBandPlan {
	t.Helper()
	w := make([]int64, cfg.NumLayers)
	for l := range w {
		w[l] = 10 * mib
	}
	costs, err := HybridDeviceBandCosts(cfg, w, mib, mib, 0, 64, compute.KVPrecisionF32)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := PlanDeviceBands(cfg, costs, []DeviceBudget{
		{Device: 0, FreeBytes: 60 * mib}, {Device: 1, FreeBytes: 60 * mib}, {Device: 2, FreeBytes: 60 * mib},
	}, DeviceBandOptions{})
	if err != nil {
		t.Fatalf("PlanDeviceBands: %v", err)
	}
	for i, want := range [][2]int{{0, 4}, {4, 8}, {8, 12}} {
		if s := plan.Bands[i].Stage; s.Lo != want[0] || s.Hi != want[1] {
			t.Fatalf("band %d = [%d,%d), want [%d,%d)", i, s.Lo, s.Hi, want[0], want[1])
		}
	}
	return plan
}

func stagesFromPlan(m *Model, plan DeviceBandPlan) []PipelineStage {
	stages := make([]PipelineStage, len(plan.Bands))
	for i, b := range plan.Bands {
		stages[i] = PipelineStage{Spec: b.Stage, Model: m}
	}
	return stages
}

// fak-test:runtime fast est=50ms lane=default
func TestDeviceHopPipelineMatchesMonolithicBitExact(t *testing.T) {
	m := qwen35ThreePeriodModel(t)
	plan := planThreeDevices(t, m.Cfg)
	ids := []int{3, 1, 4, 1, 5, 9, 2, 6}
	full := m.Forward(ids).Logits

	peerOnly := func(src, dst int) (bool, error) { return src == 0 && dst == 1, nil }
	for name, hop := range map[string]compute.HostHop{
		"peer-then-bounce": {Probe: peerOnly},
		"forced-bounce":    {Probe: peerOnly, ForceBounce: true, StagingBytes: 52}, // 13 words: ragged chunks
	} {
		tr := NewDeviceHopTransport(hop, plan)
		logits, err := RunPipelineWith(ids, stagesFromPlan(m, plan), tr)
		if err != nil {
			t.Fatalf("%s: RunPipelineWith: %v", name, err)
		}
		if len(logits) != len(full) {
			t.Fatalf("%s: logits seq %d, monolithic %d", name, len(logits), len(full))
		}
		for p := range full {
			for i := range full[p] {
				if math.Float32bits(logits[p][i]) != math.Float32bits(full[p][i]) {
					t.Fatalf("%s: logit[%d][%d] = %v, monolithic %v (want bit-exact)", name, p, i, logits[p][i], full[p][i])
				}
			}
		}
		if len(tr.Receipts) != 2 {
			t.Fatalf("%s: %d crossings, want 2", name, len(tr.Receipts))
		}
		want := []compute.HandoffRoute{compute.HandoffPeer, compute.HandoffHostBounce}
		if hop.ForceBounce {
			want = []compute.HandoffRoute{compute.HandoffHostBounce, compute.HandoffHostBounce}
		}
		for i, r := range tr.Receipts {
			if r.Route != want[i] || r.Src != i || r.Dst != i+1 || r.Bytes != int64(len(ids)*m.Cfg.HiddenSize*4) {
				t.Fatalf("%s: receipt %d = route %s %d->%d %d B", name, i, r.Route, r.Src, r.Dst, r.Bytes)
			}
		}
	}
}

type failingHop struct{}

func (failingHop) HopF32([]float32, int, int) ([]float32, compute.DeviceHandoffReceipt, error) {
	return nil, compute.DeviceHandoffReceipt{}, compute.ErrHandoffFailed
}

// fak-test:runtime fast est=5ms lane=default
func TestDeviceHopTransportFailsClosed(t *testing.T) {
	m := qwen35ThreePeriodModel(t)
	plan := planThreeDevices(t, m.Cfg)
	_, err := RunPipelineWith([]int{1, 2}, stagesFromPlan(m, plan), NewDeviceHopTransport(failingHop{}, plan))
	if !errors.Is(err, compute.ErrHandoffFailed) {
		t.Fatalf("failing hop err = %v, want ErrHandoffFailed", err)
	}
	tr := NewDeviceHopTransport(compute.HostHop{}, plan)
	if _, _, err := tr.Send([][]float32{{1, 2}, {3}}, 4, 1); err == nil {
		t.Fatal("ragged rows accepted")
	}
	if _, _, err := tr.Send([][]float32{{1}}, 4, 3); err == nil {
		t.Fatal("stage past the plan accepted")
	}
}
