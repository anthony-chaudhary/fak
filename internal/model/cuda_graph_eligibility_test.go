package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// eligibilityGraphBackend is a CPU-reference backend that advertises graph capture and
// records the KVConfig each NewKV receives. It never captures.
type eligibilityGraphBackend struct {
	compute.Backend
	kvConfigs []compute.KVConfig
}

func (b *eligibilityGraphBackend) GraphBegin() bool { return false }
func (b *eligibilityGraphBackend) GraphEndLaunch()  {}
func (b *eligibilityGraphBackend) GraphAbort()      {}
func (b *eligibilityGraphBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	b.kvConfigs = append(b.kvConfigs, cfg)
	return b.Backend.NewKV(cfg)
}

// fixedCapacityKV reports a GraphCaptureReady verdict over a CPU-reference store.
type fixedCapacityKV struct {
	compute.KVStore
	fixed bool
}

func (k fixedCapacityKV) GraphCaptureReady() bool { return k.fixed }

func cudaGraphDenseConfig() Config {
	return Config{NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8, HiddenSize: 32, IntermediateSize: 64, VocabSize: 64}
}

// fak-test:runtime fast est=1s
func TestCUDAGraphEligibilityPredicateDrivesKVConfig(t *testing.T) {
	for _, tc := range []struct {
		name     string
		cfg      Config
		graph    bool
		eligible bool
		reason   string
	}{
		{"dense on a capture backend", cudaGraphDenseConfig(), true, true, CUDAGraphEligible},
		{"dense without capture", cudaGraphDenseConfig(), false, false, CUDAGraphNoCaptureBackend},
		{"qwen hybrid without capture-legal GDN", qwen38ContextPlanConfig(), true, false, CUDAGraphQwen35GDNNotCaptured},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cpu := compute.Pick("cpu-ref")
			var be compute.Backend = cpu
			rec := &eligibilityGraphBackend{Backend: cpu}
			if tc.graph {
				be = rec
			}
			ok, reason := cudaGraphCaptureEligible(tc.cfg, be)
			if ok != tc.eligible || reason != tc.reason {
				t.Fatalf("eligible=%v reason=%q, want %v %q", ok, reason, tc.eligible, tc.reason)
			}
			if !tc.graph {
				return
			}
			kv := newHALKVStore(be, tc.cfg)
			defer kv.Free()
			if len(rec.kvConfigs) != 1 || rec.kvConfigs[0].GraphCapture != tc.eligible {
				t.Fatalf("NewKV configs = %+v, want one with GraphCapture=%v", rec.kvConfigs, tc.eligible)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestCUDAGraphEligibilitySessionReadiness(t *testing.T) {
	cpu := compute.Pick("cpu-ref")
	be := &eligibilityGraphBackend{Backend: cpu}
	store := cpu.NewKV(compute.KVConfig{NumLayers: 2, NumKVHeads: 2, HeadDim: 8})
	defer store.Free()
	for _, tc := range []struct {
		name   string
		cfg    Config
		hidden bool
		split  bool
		kv     compute.KVStore
		want   bool
	}{
		{"dense fixed KV", cudaGraphDenseConfig(), false, false, fixedCapacityKV{store, true}, true},
		{"dense store without readiness probe", cudaGraphDenseConfig(), false, false, store, true},
		{"growable KV", cudaGraphDenseConfig(), false, false, fixedCapacityKV{store, false}, false},
		{"captured target hidden", cudaGraphDenseConfig(), true, false, fixedCapacityKV{store, true}, false},
		{"split placement", cudaGraphDenseConfig(), false, true, fixedCapacityKV{store, true}, false},
		{"qwen hybrid", qwen38ContextPlanConfig(), false, false, fixedCapacityKV{store, true}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &Session{M: &Model{Cfg: tc.cfg}, Backend: be, halKV: tc.kv, captureTargetHidden: tc.hidden}
			if got := s.cudaGraphCaptureReady(tc.split); got != tc.want {
				t.Fatalf("cudaGraphCaptureReady = %v, want %v", got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestCUDAGraphKVSizingChargesQwen38CompactedRows(t *testing.T) {
	const positions = 32768
	be := &eligibilityGraphBackend{Backend: compute.Pick("cpu-ref")}
	sizing := CUDAGraphKVSizingFor(qwen38ContextPlanConfig(), be, true, positions, compute.FreeUnknown)
	if sizing.KVRows != 16 {
		t.Fatalf("Qwen3.8-27B KV rows = %d, want 16 compacted full-attention rows of 64 layers", sizing.KVRows)
	}
	if want := int64(6) << 30; sizing.Prealloc.Bytes != want {
		t.Fatalf("Qwen3.8-27B prealloc at %d positions = %d bytes, want %d (6 GiB)", positions, sizing.Prealloc.Bytes, want)
	}
	if sizing.Eligible || sizing.Prealloc.Preallocate || sizing.Prealloc.Reason != compute.GraphKVNotEligible {
		t.Fatalf("hybrid sizing = %+v, want no prealloc (%s) while the hybrid is not capture-eligible", sizing, compute.GraphKVNotEligible)
	}
	dense := CUDAGraphKVSizingFor(cudaGraphDenseConfig(), be, true, positions, compute.FreeUnknown)
	if !dense.Eligible || !dense.Prealloc.Preallocate || dense.KVRows != 2 {
		t.Fatalf("dense sizing = %+v, want eligible prealloc over 2 rows", dense)
	}
}
