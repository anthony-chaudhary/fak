package agent

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=50ms lane=default
func TestCheckpointPageCacheTelemetryProducerConsumer(t *testing.T) {
	const stride = 84 // One Q2_K block: 256 weights, one expert projection.
	identity := model.ExpertWarmProfileIdentity{Checkpoint: "cpu-fixture", Quantization: "Q2_K", Layout: "ffn.experts"}
	fused := func(layer, experts int) []model.FusedExpertTensor {
		return []model.FusedExpertTensor{{Name: "gate", Arch: "deepseek41", Layer: layer, Proj: "gate_proj", Quant: model.ExpertCheckpointQ2K, Experts: experts, Rows: 1, Cols: 256}}
	}
	warm := func(t *testing.T, tier *model.ExpertCheckpointTier, experts int) *model.Model {
		t.Helper()
		m := &model.Model{}
		m.SetExpertCheckpoint(tier)
		profile := model.ExpertWarmProfile{Identity: identity}
		for expert := 0; expert < experts; expert++ {
			profile.Entries = append(profile.Entries, model.ExpertWarmDemandEntry{Layer: 0, Expert: expert, Count: float64(experts - expert)})
		}
		result := m.WarmExpertProfile(profile, identity, int64(experts*stride))
		if result.Failed != 0 || result.Retained != experts || result.Reads != experts {
			t.Fatalf("real bounded warm did not fault the selected experts: %+v", result)
		}
		return m
	}
	// Use only parent-existing APIs and JSON names: the parent fails because
	// fields never reach the consumer, not because a new symbol fails to compile.
	consume := func(t *testing.T, m *model.Model) map[string]any {
		t.Helper()
		report := (&model.Session{M: m}).MoEResidency(model.MoEResidencyOptions{})
		planner := &InKernelPlanner{}
		planner.foldMoEResidency(report, 0)
		planner.foldMoEResidency(report, 0) // Lifetime counters must REPLACE, not double.
		snapshot := planner.MoEResidencyStats().Checkpoint
		if snapshot == nil || snapshot.Scope != "model_lifetime" {
			t.Fatal("checkpoint lifetime snapshot did not reach the planner")
		}
		// debugMoEResidency forwards this exact Checkpoint value to gateway JSON.
		raw, err := json.Marshal(snapshot)
		if err != nil {
			t.Fatal(err)
		}
		var object map[string]any
		if err := json.Unmarshal(raw, &object); err != nil {
			t.Fatal(err)
		}
		pc, ok := object["page_cache"].(map[string]any)
		if !ok || pc["accounting"] != "modeled_source_io" {
			t.Fatal("page-cache ledger and honest source-I/O qualifier missing at actual residency consumer")
		}
		return pc
	}
	check := func(t *testing.T, got, want map[string]any) {
		t.Helper()
		for field, value := range want {
			if actual, present := got[field]; !present || actual != value {
				t.Fatalf("page_cache.%s = %v (present=%v), want %v", field, actual, present, value)
			}
		}
	}
	t.Run("mapped_cold_then_shared_page_warm", func(t *testing.T) {
		page := os.Getpagesize()
		if page < 2*stride {
			t.Fatal("fixture requires two expert projections in one actual page")
		}
		data := make([]byte, page)
		source := &checkpointPageCacheReadProbe{Reader: bytes.NewReader(data)}
		tier := model.NewExpertCheckpointTier(2 * stride)
		if err := tier.AddShardData(source, int64(len(data)), data, fused(0, 2)); err != nil {
			t.Fatal(err)
		}
		// An unmapped shard makes the source coverage explicitly partial.
		if err := tier.AddShard(bytes.NewReader(make([]byte, stride)), stride, fused(1, 1)); err != nil {
			t.Fatal(err)
		}
		m := warm(t, tier, 2)
		pc := consume(t, m)
		check(t, pc, map[string]any{
			"enabled": true, "experts_read": float64(1),
			"bytes_from_device": float64(page), "bytes_from_mapping": float64(stride),
			"warm_hits": float64(1), "fill_pins": float64(1), "advantage_bytes": float64(page - stride),
			"sources": float64(2), "mapped_sources": float64(1),
		})
		if source.calls != 1 || source.bytes != page {
			t.Fatalf("producer/consumer read accounting: calls=%d bytes=%d, want1/%d", source.calls, source.bytes, page)
		}
		t.Log("CPU fixture: all7 existing ledger fields reach residency JSON; source-read/mapping prediction matched, not physical SSD/GPU evidence")
	})
	t.Run("unmapped_source_is_explicitly_disabled", func(t *testing.T) {
		source := &checkpointPageCacheReadProbe{Reader: bytes.NewReader(make([]byte, stride))}
		tier := model.NewExpertCheckpointTier(stride)
		if err := tier.AddShard(source, stride, fused(0, 1)); err != nil {
			t.Fatal(err)
		}
		pc := consume(t, warm(t, tier, 1))
		check(t, pc, map[string]any{
			"enabled": false, "experts_read": float64(0),
			"bytes_from_device": float64(0), "bytes_from_mapping": float64(0),
			"warm_hits": float64(0), "fill_pins": float64(0), "advantage_bytes": float64(0),
			"sources": float64(1), "mapped_sources": float64(0),
		})
		if source.calls != 1 || source.bytes != stride {
			t.Fatal("unmapped source did not preserve ordinary ReadAt; reporting must not read payload")
		}
	})
}

type checkpointPageCacheReadProbe struct {
	*bytes.Reader
	calls int
	bytes int
}

func (r *checkpointPageCacheReadProbe) ReadAt(p []byte, off int64) (int, error) {
	r.calls++
	r.bytes += len(p)
	return r.Reader.ReadAt(p, off)
}
