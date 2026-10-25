package main

import (
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type serveCUDAGraphTestBackend struct{ compute.Backend }

func (serveCUDAGraphTestBackend) GraphBegin() bool { return false }
func (serveCUDAGraphTestBackend) GraphEndLaunch()  {}
func (serveCUDAGraphTestBackend) GraphAbort()      {}

// fak-test:runtime fast est=1s
func TestServeCUDAGraphKVCapacityEveryEnablePath(t *testing.T) {
	dense := model.Config{NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8, HiddenSize: 32, IntermediateSize: 64, VocabSize: 64}
	be := serveCUDAGraphTestBackend{compute.Pick("cpu-ref")}
	env := func(v string) func(string) string {
		return func(k string) string {
			if k == "FAK_CUDA_GRAPH" {
				return v
			}
			return ""
		}
	}
	for _, tc := range []struct {
		name     string
		flag     bool
		env      string
		resolved int
		want     int
		on       bool
		source   string
	}{
		{"flag", true, "", 32768, 32768, true, "--cuda-graph"},
		{"env var", false, "1", 32768, 32768, true, "FAK_CUDA_GRAPH=1"},
		{"flag below default", true, "", 512, compute.GraphKVDefaultPositions, true, "--cuda-graph"},
		{"env off", false, "0", 32768, 0, false, ""},
		{"neither", false, "", 32768, 0, false, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			capacity, msg, on := serveCUDAGraphKV(tc.flag, env(tc.env), be, &dense, tc.resolved)
			if on != tc.on || capacity != tc.want {
				t.Fatalf("capacity=%d on=%v, want %d %v", capacity, on, tc.want, tc.on)
			}
			if !on {
				return
			}
			if capacity <= compute.GraphKVDefaultPositions && tc.resolved > compute.GraphKVDefaultPositions {
				t.Fatalf("capacity %d would refuse a %d-position prompt", capacity, tc.resolved)
			}
			want := compute.GraphKVPreallocBytes(compute.KVConfig{NumLayers: 2, NumKVHeads: 2, HeadDim: 8}, capacity)
			if !strings.Contains(msg, tc.source) || !strings.Contains(msg, "prealloc="+strconv.FormatInt(want, 10)+" bytes per session") {
				t.Fatalf("startup message %q does not name source %q and prealloc %d bytes", msg, tc.source, want)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestServeCUDAGraphKVReportsIneligibleSessions(t *testing.T) {
	dense := model.Config{NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8, HiddenSize: 32, IntermediateSize: 64, VocabSize: 64}
	_, msg, on := serveCUDAGraphKV(true, func(string) string { return "" }, compute.Pick("cpu-ref"), &dense, 4096)
	if !on || !strings.Contains(msg, "ineligible ("+model.CUDAGraphNoCaptureBackend+")") {
		t.Fatalf("message %q (on=%v) does not report the ineligible backend", msg, on)
	}
	capacity, msg, on := serveCUDAGraphKV(false, func(string) string { return "1" }, compute.Pick("cpu-ref"), nil, 4096)
	if !on || capacity != 4096 || !strings.Contains(msg, "geometry not read") {
		t.Fatalf("capacity=%d message %q (on=%v), want 4096 with unknown geometry", capacity, msg, on)
	}
}
