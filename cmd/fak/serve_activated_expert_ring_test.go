package main

import (
	"bytes"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// fak#13668: a device-only (Halo) serve whose full device plan is FitTooBig must select the
// activated-expert device ring without --cpu-offload-experts; every other case keeps its refusal.

func serveRingFullPlanTooBig() error {
	return &compute.FitError{Verdict: compute.FitTooBig, Want: 1 << 40, Avail: 1 << 20, Scope: compute.MemoryScopeDevice}
}

func TestServeHaloGPUActivatedExpertsRingSelectedWhenFullPlanFitTooBig(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	be := serveCapBackend{total: 1 << 20, free: 1 << 20, known: true}
	fit := serveFitBudget{Base: 1 << 20}

	ring, ok, err := serveActivatedExpertRingPlacement(ws, be, true, serveRingFullPlanTooBig(), 0, fit)
	if err != nil || !ok {
		t.Fatalf("ring placement not selected: ok=%v err=%v", ok, err)
	}
	if ring.RingBytes < ring.Fit.ActivatedLayerBytes || ring.RingBytes > ring.Fit.RoutedBandBytes {
		t.Fatalf("ring %d outside [floor %d, band %d]", ring.RingBytes, ring.Fit.ActivatedLayerBytes, ring.Fit.RoutedBandBytes)
	}
	var ringRow int64
	for _, d := range ring.Plan {
		if d.Detail == serveActivatedRingDetail && d.DeviceScoped() {
			ringRow += d.Bytes
		}
		if d.Class == compute.MemoryOffload {
			t.Fatalf("ring placement charges a host expert offload row: %+v", d)
		}
	}
	if ringRow != ring.RingBytes {
		t.Fatalf("device ring row %d, want %d", ringRow, ring.RingBytes)
	}

	eff := ggufload.ApplyQ4KLoadOptions(serveActivatedExpertRingLoadOptions(ring))
	if !eff.StreamedExperts || eff.StreamedExpertBytes != 0 || eff.StreamedExpertDeviceRing != ring.RingBytes {
		t.Fatalf("load options %+v, want stream-through experts with device ring %d", eff, ring.RingBytes)
	}
}

func TestServeHaloGPUActivatedExpertsRingRefusesBelowFloor(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	be := serveCapBackend{total: 4096, free: 4096, known: true}
	_, ok, err := serveActivatedExpertRingPlacement(ws, be, true, serveRingFullPlanTooBig(), 0, serveFitBudget{Base: 4096})
	var fe *compute.FitError
	if ok || !errors.As(err, &fe) || fe.Verdict != compute.FitTooBig {
		t.Fatalf("below the activated floor: ok=%v err=%v, want typed FitTooBig", ok, err)
	}
}

func TestServeHaloGPUActivatedExpertsUnstageableRefusesByName(t *testing.T) {
	f := &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":       {Type: ggufload.TypeString, Value: "llama"},
			"llama.context_length":       {Type: ggufload.TypeUint64, Value: uint64(16)},
			"llama.embedding_length":     {Type: ggufload.TypeUint64, Value: uint64(32)},
			"llama.block_count":          {Type: ggufload.TypeUint64, Value: uint64(1)},
			"llama.attention.head_count": {Type: ggufload.TypeUint64, Value: uint64(4)},
		},
		Tensors: []ggufload.TensorInfo{{Name: "token_embd.weight", Dims: []uint64{256}, Type: ggufload.TensorF32}},
	}
	blob := make([]byte, 1024)
	ws, err := ggufload.NewWeightSource(f, bytes.NewReader(blob), int64(len(blob)))
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	be := serveCapBackend{total: 1 << 20, free: 1 << 20, known: true}
	_, ok, err := serveActivatedExpertRingPlacement(ws, be, true, serveRingFullPlanTooBig(), 0, serveFitBudget{Base: 1 << 20})
	var fe *compute.FitError
	if ok || !errors.Is(err, errServeActivatedExpertRingUnstageable) || !errors.As(err, &fe) {
		t.Fatalf("unstageable checkpoint: ok=%v err=%v, want FitTooBig naming the unstageable ring", ok, err)
	}
}

func TestServeHaloGPUActivatedExpertsNonHaloUnchanged(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	be := serveCapBackend{total: 1 << 20, free: 1 << 20, known: true}
	full := serveRingFullPlanTooBig()
	if _, ok, err := serveActivatedExpertRingPlacement(ws, be, false, full, 0, serveFitBudget{Base: 1 << 20}); ok || err != full {
		t.Fatalf("non-Halo serve: ok=%v err=%v, want the full-plan refusal unchanged", ok, err)
	}
	if _, ok, err := serveActivatedExpertRingPlacement(ws, be, true, nil, 0, serveFitBudget{Base: 1 << 20}); ok || err != nil {
		t.Fatalf("fitting full plan: ok=%v err=%v, want the resident arm", ok, err)
	}
}

func TestValidateServeHaloCPUOffloadStillRefusesHostExpertPlacement(t *testing.T) {
	for _, tc := range []struct {
		offload bool
		grade   string
	}{{offload: true}, {grade: "auto"}} {
		err := validateServeHaloCPUOffload(true, tc.offload, tc.grade, "")
		var halo *haloGPURequiredError
		if !errors.As(err, &halo) || halo.Reason != haloGPUReasonCPUExpertOffload {
			t.Fatalf("offload=%v grade=%q: err=%v, want cpu-expert-offload", tc.offload, tc.grade, err)
		}
	}
	if err := validateServeHaloCPUOffload(false, true, "", ""); err != nil {
		t.Fatalf("non-Halo host-expert placement must stay admitted: %v", err)
	}
}
