package main

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serveCUDACapBackend is a discrete, capacity-probed CUDA device double: the shape the
// fit-with-offload default serves on (an RTX 5080 class card).
type serveCUDACapBackend struct{ serveCapBackend }

func (b serveCUDACapBackend) Name() string { return "cuda" }
func (b serveCUDACapBackend) Caps() compute.Caps {
	c := b.serveCapBackend.Caps()
	c.DeviceMemory = true
	return c
}

const serveAutoGPULayersTestContext = 16 // the synthetic header's context_length

func serveAutoGPULayersTestInputs() serveAutoGPULayersInputs {
	return serveAutoGPULayersInputs{
		Ranks:            1,
		RequestedContext: serveAutoGPULayersTestContext,
		Getenv:           func(string) string { return "" },
		HostAvail:        func() (int64, bool) { return 1 << 40, true },
		HostPeak:         func() (ggufload.HostLoadPeak, error) { return ggufload.HostLoadPeak{PeakAnonBytes: 1 << 30}, nil },
	}
}

// serveAutoGPULayersTestBackend sizes the device so the full placement refuses but a split fits.
func serveAutoGPULayersTestBackend(t *testing.T, ws *ggufload.WeightSource, num, den int64) serveCUDACapBackend {
	t.Helper()
	full, err := serveGGUFMemoryPlan(ws, false, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("serveGGUFMemoryPlan: %v", err)
	}
	if full.DeviceTotal() <= 0 {
		t.Fatalf("synthetic plan has no device weights: %+v", full)
	}
	return serveCUDACapBackend{serveCapBackend{Backend: compute.Default(), total: full.DeviceTotal() * num / den, free: compute.FreeUnknown, known: true, uploadDtype: true}}
}

// A model whose full device plan is FitTooBig must serve split: the default picks the largest
// contiguous band [0,N) the load's own device admission accepts, not a refusal.
func TestServeAutoGPULayersPicksLargestFittingBand(t *testing.T) {
	ws := serveSynthLayeredWeightSource(t)
	be := serveAutoGPULayersTestBackend(t, ws, 2, 3)
	if err := serveDevicePlanFitsAtLayers(ws, be, "", serveAutoGPULayersTestContext, 0); !serveFullPlanFitTooBig(err) {
		t.Fatalf("precondition: full device placement must be FitTooBig, got %v", err)
	}
	n, err := serveAutoGPULayers(ws, be, serveAutoGPULayersTestInputs())
	if err != nil {
		t.Fatalf("serveAutoGPULayers: %v", err)
	}
	if n < 1 || n >= 4 {
		t.Fatalf("auto --gpu-layers = %d, want a split in [1,4)", n)
	}
	if err := serveDevicePlanFitsAtLayers(ws, be, "", serveAutoGPULayersTestContext, n); err != nil {
		t.Fatalf("chosen --gpu-layers %d does not fit: %v", n, err)
	}
	if n+1 < 4 {
		if err := serveDevicePlanFitsAtLayers(ws, be, "", serveAutoGPULayersTestContext, n+1); err == nil {
			t.Fatalf("--gpu-layers %d also fits; %d is not the largest band", n+1, n)
		}
	}
	// A roomier device moves the band edge outward, never inward.
	roomy := serveAutoGPULayersTestBackend(t, ws, 9, 10)
	m, err := serveAutoGPULayers(ws, roomy, serveAutoGPULayersTestInputs())
	if err != nil || m < n {
		t.Fatalf("roomier device chose %d (%v), want >= %d", m, err, n)
	}
}

// The live seam writes the split into the serve flags (so the fit budget, the context sizer and
// the session all read it) and logs it; an explicit --gpu-layers is never moved.
func TestResolveAutoGPULayersWritesSplitUnlessPinned(t *testing.T) {
	ws := serveSynthLayeredWeightSource(t)
	be := serveAutoGPULayersTestBackend(t, ws, 2, 3)
	t.Setenv(ggufload.HostLoadPeakOverrideEnv, "off") // the synthetic header has no estimate route

	layers := 0
	offload := false
	sf := &serveFlags{nativeGPULayers: &layers, cpuOffloadExperts: &offload}
	rt := &serveRuntime{chatBackend: be}
	if err := rt.resolveAutoGPULayers(sf, ws, 1, serveAutoGPULayersTestContext, ""); err != nil {
		t.Fatalf("resolveAutoGPULayers: %v", err)
	}
	if layers < 1 || layers >= 4 {
		t.Fatalf("--gpu-layers after auto = %d, want a split in [1,4)", layers)
	}
	found := false
	for _, m := range rt.startupMessages {
		if m.Kind == "auto-gpu-layers" && strings.Contains(m.Text, "fit-with-offload chose --gpu-layers") {
			found = true
		}
	}
	if !found {
		t.Fatalf("no auto-gpu-layers startup message in %+v", rt.startupMessages)
	}

	for _, alias := range []string{"gpu-layers", "native-gpu-layers"} {
		pinned := 0
		psf := &serveFlags{nativeGPULayers: &pinned, cpuOffloadExperts: &offload, explicit: map[string]bool{alias: true}}
		if err := (&serveRuntime{chatBackend: be}).resolveAutoGPULayers(psf, ws, 1, serveAutoGPULayersTestContext, ""); err != nil || pinned != 0 {
			t.Fatalf("explicit --%s 0 moved to %d (%v)", alias, pinned, err)
		}
	}
}

// Every boot that cannot run a dense layer split, and every model that already fits, is untouched.
func TestServeAutoGPULayersInertWhenIneligibleOrFits(t *testing.T) {
	ws := serveSynthLayeredWeightSource(t)
	tight := serveAutoGPULayersTestBackend(t, ws, 2, 3)
	cases := []struct {
		name   string
		be     compute.Backend
		mutate func(*serveAutoGPULayersInputs)
	}{
		{"full plan fits", serveAutoGPULayersTestBackend(t, ws, 4, 1), nil},
		{"not cuda", tight.serveCapBackend, nil},
		{"no backend", nil, nil},
		{"device-only policy", tight, func(in *serveAutoGPULayersInputs) { in.RequireDevice = true }},
		{"expert-parallel ranks", tight, func(in *serveAutoGPULayersInputs) { in.Ranks = 2 }},
		{"cpu-offload-experts", tight, func(in *serveAutoGPULayersInputs) { in.CPUOffloadExperts = true }},
		{"metal", tight, func(in *serveAutoGPULayersInputs) { in.UseMetal = true }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := serveAutoGPULayersTestInputs()
			if tc.mutate != nil {
				tc.mutate(&in)
			}
			if n, err := serveAutoGPULayers(ws, tc.be, in); n != 0 || err != nil {
				t.Fatalf("serveAutoGPULayers = (%d, %v), want (0, nil)", n, err)
			}
		})
	}
}

// The split never admits a load the host cannot stage: a host peak over budget refuses typed,
// before the search, unless the operator disabled host-peak admission.
func TestServeAutoGPULayersHostPeakRefuses(t *testing.T) {
	ws := serveSynthLayeredWeightSource(t)
	be := serveAutoGPULayersTestBackend(t, ws, 2, 3)
	in := serveAutoGPULayersTestInputs()
	in.HostPeak = func() (ggufload.HostLoadPeak, error) {
		return ggufload.HostLoadPeak{PeakAnonBytes: 30 << 30}, nil
	}
	in.HostAvail = func() (int64, bool) { return 27 << 30, true }
	n, err := serveAutoGPULayers(ws, be, in)
	if !errors.Is(err, ggufload.ErrHostLoadPeakTooBig) || n != 0 {
		t.Fatalf("serveAutoGPULayers = (%d, %v), want HOST_LOAD_PEAK_TOO_BIG", n, err)
	}
	in.Getenv = func(k string) string {
		if k == ggufload.HostLoadPeakOverrideEnv {
			return "off"
		}
		return ""
	}
	if n, err := serveAutoGPULayers(ws, be, in); err != nil || n < 1 {
		t.Fatalf("host-peak admission off: got (%d, %v), want a split", n, err)
	}
}

// When even one device layer overflows, the refusal is typed and still carries the FitTooBig.
func TestServeAutoGPULayersNoBandFits(t *testing.T) {
	ws := serveSynthLayeredWeightSource(t)
	be := serveAutoGPULayersTestBackend(t, ws, 1, 20)
	n, err := serveAutoGPULayers(ws, be, serveAutoGPULayersTestInputs())
	if n != 0 || !errors.Is(err, errServeAutoGPULayersNoFit) || !serveFullPlanFitTooBig(err) {
		t.Fatalf("serveAutoGPULayers = (%d, %v), want AUTO_GPU_LAYERS_NO_FIT wrapping FitTooBig", n, err)
	}
}

// serveSynthLongContextWeightSource is serveSynthLayeredWeightSource with 16x larger layers and a
// declared 8192-token window, so a 4096-token KV is a real fraction of the weights.
func serveSynthLongContextWeightSource(t *testing.T) *ggufload.WeightSource {
	t.Helper()
	base := serveSynthLayeredWeightSource(t)
	md := map[string]ggufload.Value{}
	for k, v := range base.File.Metadata {
		md[k] = v
	}
	md["qwen2.context_length"] = ggufload.Value{Type: ggufload.TypeUint64, Value: uint64(8192)}
	tensors := make([]ggufload.TensorInfo, len(base.File.Tensors))
	copy(tensors, base.File.Tensors)
	for i := range tensors {
		if strings.HasPrefix(tensors[i].Name, "blk.") {
			tensors[i].Dims = []uint64{256 * 1024 * 16}
		}
	}
	ws, err := ggufload.NewWeightSource(&ggufload.File{Metadata: md, Tensors: tensors}, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// The split is chosen against the context it must serve: KV for the preferred window is charged
// against the SPLIT weights, so the chosen band leaves at least that window, not the 512 floor.
func TestServeAutoGPULayersLeavesPreferredContext(t *testing.T) {
	ws := serveSynthLongContextWeightSource(t)
	roomy := serveCUDACapBackend{serveCapBackend{Backend: compute.Default(), total: 1 << 40, free: compute.FreeUnknown, known: true, uploadDtype: true}}
	fullWeights, _, err := serveNativeContextSizingInputs(ws, roomy, false, false, 1, 0, "")
	if err != nil {
		t.Fatal(err)
	}
	// The weights alone exceed the budget after headroom, so full placement is FitTooBig.
	be := serveCUDACapBackend{serveCapBackend{Backend: compute.Default(), total: fullWeights.DeviceTotal(), free: compute.FreeUnknown, known: true, uploadDtype: true}}
	in := serveAutoGPULayersTestInputs()
	in.RequestedContext = 0
	n, err := serveAutoGPULayers(ws, be, in)
	if err != nil || n < 1 || n >= 4 {
		t.Fatalf("serveAutoGPULayers = (%d, %v), want a split in [1,4)", n, err)
	}
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatal(err)
	}
	csc := cfg.ContextSizeConfigWithPrecision(serveKVPrecision())
	weights, fit, err := serveNativeContextSizingInputs(ws, be, false, false, 1, n, "")
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := compute.AutoSizeContextPlan(csc, weights, fit.avail(), -1); got < serveAutoGPULayersContextFloor {
		t.Fatalf("--gpu-layers %d leaves a %d-token window, want >= %d", n, got, serveAutoGPULayersContextFloor)
	}
	if n+1 < 4 && serveSplitFitsContext(ws, be, cfg, "", serveAutoGPULayersContextFloor, n+1) == nil {
		t.Fatalf("--gpu-layers %d also leaves the window; %d is not the largest band", n+1, n)
	}
}
