package ggufload

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// readyWeightSource builds a header that BOTH parses an architecture (so File.Config succeeds)
// AND carries the same two synth tensors as synthWeightSource, so the lean EstimateLoadBytes is
// the known 1638400 B. It supplies the minimal required GGUF config keys File.Config reads
// (embedding_length divisible by head_count, block_count, feed_forward_length, the rms epsilon).
func readyWeightSource(t *testing.T) *WeightSource {
	t.Helper()
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                   {Type: TypeString, Value: "llama"},
			"llama.embedding_length":                 {Type: TypeUint32, Value: uint32(256)},
			"llama.block_count":                      {Type: TypeUint32, Value: uint32(2)},
			"llama.attention.head_count":             {Type: TypeUint32, Value: uint32(8)},
			"llama.feed_forward_length":              {Type: TypeUint32, Value: uint32(512)},
			"llama.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		},
		Tensors: []TensorInfo{
			{Name: "a", Dims: []uint64{256 * 1024}, Type: TensorF32},  // 262144 elems * 4 = 1 MiB
			{Name: "b", Dims: []uint64{256 * 4096}, Type: TensorQ4_K}, // 1048576 elems / 256 * 144 = 589824 B
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// archlessWeightSource is a header with tensors but no general.architecture, so File.Config()
// fails the arch gate — the REFUSE_BAD_ARCH case without a real malformed checkpoint.
func archlessWeightSource(t *testing.T) *WeightSource {
	t.Helper()
	f := &File{
		Tensors: []TensorInfo{
			{Name: "a", Dims: []uint64{256}, Type: TensorF32},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

func TestPreflightReadyFitOKOnKnownBigDevice(t *testing.T) {
	// synthWeightSource fails File.Config() (no arch) — so to exercise READY we need a header
	// that parses an arch. Build one with the minimum required GGUF config keys plus the two
	// synth tensors so EstimateLoadBytes is the known 1638400 B.
	ws := readyWeightSource(t)
	big := capBackend{total: 2 << 20, free: 2 << 20, known: true}
	pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Backend: big, Lean: true})
	if pf.Verdict != PreflightReady {
		t.Fatalf("verdict = %s, want READY (reason: %s)", pf.Verdict, pf.Reason)
	}
	if pf.FitState != FitOK {
		t.Fatalf("fit = %s, want FIT_OK", pf.FitState)
	}
	if pf.EstLoadBytes != 1638400 {
		t.Fatalf("est bytes = %d, want 1638400 (the synth lean footprint)", pf.EstLoadBytes)
	}
	if pf.Arch == "" {
		t.Fatalf("arch not populated on READY")
	}
	if pf.ETASecondsEst <= 0 {
		t.Fatalf("ETA must be positive on a non-empty model, got %v", pf.ETASecondsEst)
	}
	if pf.Refused() {
		t.Fatalf("READY must not report Refused()")
	}
}

func TestPreflightRefuseTooBigCarriesDeviceAvail(t *testing.T) {
	ws := readyWeightSource(t)
	small := capBackend{total: 1 << 20, free: 1 << 20, known: true}
	pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Backend: small, Lean: true})
	if pf.Verdict != PreflightRefuseTooBig {
		t.Fatalf("verdict = %s, want REFUSE_TOO_BIG", pf.Verdict)
	}
	if pf.FitState != FitTooBigState {
		t.Fatalf("fit = %s, want FIT_TOO_BIG", pf.FitState)
	}
	if pf.DeviceAvailBytes != 1<<20 {
		t.Fatalf("device avail = %d, want %d (from the FitError)", pf.DeviceAvailBytes, 1<<20)
	}
	if !pf.Refused() {
		t.Fatalf("REFUSE_TOO_BIG must report Refused()")
	}
	if !strings.Contains(pf.Reason, "FitTooBig") {
		t.Fatalf("reason should carry the typed FitError text, got %q", pf.Reason)
	}
}

func TestPreflightFitUnknownFailsOpen(t *testing.T) {
	ws := readyWeightSource(t)
	// Both a nil backend and a backend that cannot probe must yield READY + FIT_UNKNOWN — the
	// fail-open contract that keeps the portable floor loadable.
	for name, be := range map[string]compute.Backend{
		"nil":         nil,
		"unprobeable": capBackend{known: false},
	} {
		pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Backend: be, Lean: true})
		if pf.Verdict != PreflightReady {
			t.Fatalf("[%s] verdict = %s, want READY (fail-open)", name, pf.Verdict)
		}
		if pf.FitState != FitUnknown {
			t.Fatalf("[%s] fit = %s, want FIT_UNKNOWN", name, pf.FitState)
		}
		if pf.DeviceAvailBytes != 0 {
			t.Fatalf("[%s] unprobeable device must not report avail bytes, got %d", name, pf.DeviceAvailBytes)
		}
	}
}

func TestPreflightRefuseBadArch(t *testing.T) {
	ws := archlessWeightSource(t)
	pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Lean: true})
	if pf.Verdict != PreflightRefuseArch {
		t.Fatalf("verdict = %s, want REFUSE_BAD_ARCH", pf.Verdict)
	}
	if !pf.Refused() {
		t.Fatalf("REFUSE_BAD_ARCH must report Refused()")
	}
	if !strings.Contains(pf.Reason, "architecture") {
		t.Fatalf("reason should name the missing architecture, got %q", pf.Reason)
	}
}

func TestPreflightRefuseBadHeader(t *testing.T) {
	pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", OpenErr: errors.New("gguf: bad magic"), Source: nil})
	if pf.Verdict != PreflightRefuseHeader {
		t.Fatalf("verdict = %s, want REFUSE_BAD_HEADER", pf.Verdict)
	}
	if !pf.Refused() {
		t.Fatalf("REFUSE_BAD_HEADER must report Refused()")
	}
	if !strings.Contains(pf.Reason, "bad magic") {
		t.Fatalf("reason should carry the open error, got %q", pf.Reason)
	}
	// A nil source with no error is also a header refusal (defensive).
	pf2 := BuildModelPreflight(PreflightInput{Path: "x.gguf"})
	if pf2.Verdict != PreflightRefuseHeader {
		t.Fatalf("nil source / nil err verdict = %s, want REFUSE_BAD_HEADER", pf2.Verdict)
	}
}

func TestPreflightDeterministic(t *testing.T) {
	ws := readyWeightSource(t)
	in := PreflightInput{Path: "x.gguf", Source: ws, Backend: capBackend{total: 2 << 20, free: 2 << 20, known: true}, Lean: true}
	a := BuildModelPreflight(in)
	b := BuildModelPreflight(in)
	if a != b {
		t.Fatalf("preflight not deterministic:\n a=%+v\n b=%+v", a, b)
	}
}

func TestPreflightETAScalesWithBytes(t *testing.T) {
	// A higher assumed throughput must shorten the ETA for the same byte estimate.
	ws := readyWeightSource(t)
	slow := BuildModelPreflight(PreflightInput{Source: ws, Lean: true, AssumedGiBPerSec: 0.1})
	fast := BuildModelPreflight(PreflightInput{Source: ws, Lean: true, AssumedGiBPerSec: 10})
	if !(slow.ETASecondsEst > fast.ETASecondsEst) {
		t.Fatalf("slower assumed rate must give a longer ETA: slow=%v fast=%v", slow.ETASecondsEst, fast.ETASecondsEst)
	}
	if fast.ETASecondsEst <= 0 {
		t.Fatalf("ETA must stay positive, got %v", fast.ETASecondsEst)
	}
}

func TestPreflightF32PathEstimatesLargerThanLean(t *testing.T) {
	// The default (non-lean) GGUF path dequantizes to f32 resident, so its estimate must exceed
	// the lean raw-payload estimate for the same header — proving the regime selection is wired.
	ws := readyWeightSource(t)
	lean := BuildModelPreflight(PreflightInput{Source: ws, Lean: true})
	f32 := BuildModelPreflight(PreflightInput{Source: ws}) // default path
	if !(f32.EstLoadBytes > lean.EstLoadBytes) {
		t.Fatalf("f32 estimate (%d) must exceed lean estimate (%d)", f32.EstLoadBytes, lean.EstLoadBytes)
	}
}

func TestPreflightRenderMentionsVerdictAndEstimate(t *testing.T) {
	ws := readyWeightSource(t)
	pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Lean: true})
	out := pf.Render()
	for _, want := range []string{pf.Verdict, "load:", "GiB", "ETA", "estimate", "fit:"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render %q missing %q", out, want)
		}
	}
}
