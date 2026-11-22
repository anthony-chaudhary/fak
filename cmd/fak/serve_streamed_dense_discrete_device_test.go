package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A discrete device keeps every uploaded dense copy in VRAM after the host transit frees, so the
// streamed --cpu-offload-experts plan must charge the eligible dense side on the device too (the
// aegis first-prefill ledger held 3.40 GB of device weights against a 0.40 GB plan). A device-less
// arm has no VRAM and keeps the plain fold.
// fak-test:justify why=regression when=changed:cmd/fak/serve_model_fit_streamed.go
// fak-test:runtime fast est=50ms lane=default
func TestServeStreamedDenseChargedOnDiscreteDevice(t *testing.T) {
	ws := serveStreamedV41AegisFixture(t)
	fit := serveFitBudget{Base: 24_137_716_203, Headroom: serveGGUFHostHeadroom}
	be := serveCapBackend{Backend: compute.Default(), total: 16 << 30, free: 16 << 30, known: true}

	_, discrete, selected, err := serveStreamedCPUOffloadCorePlan(ws, be, 1, 0, fit, false, false)
	if err != nil || !selected {
		t.Fatalf("discrete arm did not select the streamed plan: selected=%v err=%v", selected, err)
	}
	resident, rows := serveStreamedRowBytes(discrete, "gguf-device-dense-resident")
	if rows != 1 || resident <= 0 {
		t.Fatalf("discrete plan carries %d gguf-device-dense-resident rows (%d B), want one positive row; plan=%+v", rows, resident, discrete)
	}

	_, hostOnly, selected, err := serveStreamedCPUOffloadCorePlan(ws, nil, 1, 0, fit, false, false)
	if err != nil || !selected {
		t.Fatalf("device-less arm did not select the streamed plan: selected=%v err=%v", selected, err)
	}
	if _, rows := serveStreamedRowBytes(hostOnly, "gguf-device-dense-resident"); rows != 0 {
		t.Fatalf("device-less plan charged a device-resident dense row; plan=%+v", hostOnly)
	}
	if got, want := discrete.DeviceTotal()-hostOnly.DeviceTotal(), resident; got != want {
		t.Fatalf("device total grew by %d, want exactly the resident dense row %d", got, want)
	}
}
