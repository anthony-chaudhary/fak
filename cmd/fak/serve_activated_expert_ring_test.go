package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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

// fak-test:runtime fast est=10ms lane=default
func TestServeHaloGPUActivatedExpertsRingRefusesBelowFloor(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	be := serveCapBackend{total: 4096, free: 4096, known: true}
	ring, ok, err := serveActivatedExpertRingPlacement(ws, be, true, serveRingFullPlanTooBig(), 0, serveFitBudget{Base: 4096})
	var fe *compute.FitError
	if ok || !errors.As(err, &fe) || fe.Verdict != compute.FitTooBig {
		t.Fatalf("below the activated floor: ok=%v err=%v, want typed FitTooBig", ok, err)
	}
	if ring.Plan != nil || ring.RingBytes != 0 || ring.Fit != (ggufload.ActivatedExpertFit{}) {
		t.Fatalf("refusal returned an admitted placement: %+v", ring)
	}
	if fe.Scope != compute.MemoryScopeDevice || fe.Avail != 4096 || fe.Want <= fe.Avail {
		t.Fatalf("refusal lost its device-fit evidence: %+v", fe)
	}
	var baseBytes, ringBytes int64
	for _, d := range fe.Demands {
		switch d.Detail {
		case serveActivatedRingBaseDetail:
			baseBytes += d.Bytes
		case serveActivatedRingDetail:
			ringBytes += d.Bytes
		}
	}
	if baseBytes <= 0 || ringBytes <= 0 {
		t.Fatalf("refusal lacks the candidate placement rows: %+v", fe.Demands)
	}
	want := fmt.Sprintf("activated-expert device ring refused (dense base=%d bytes, ring=%d bytes):", baseBytes, ringBytes)
	if !strings.HasPrefix(err.Error(), want) {
		t.Fatalf("refusal %q does not expose the candidate byte split %q", err, want)
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

// fak#12640: a NON-Halo CUDA serve whose full device plan is FitTooBig must size its ordinary
// context against the activated-expert ring and load it through the same placement, without the
// session-wide device-only policy; every non-eligible case keeps its historical arm.

// nonHaloRingLoadBytes is the double's reported device budget for the LOAD subtest.
// Headroom-adjusted (~326 KiB) it admits the ring plan (dense base + one-token ring ~72 KiB plus
// the declared-context KV ~192 KiB and forward scratch ~17 KiB) while refusing the full plan
// (~142 KiB weights plus the same KV and scratch).
const nonHaloRingLoadBytes = 384 << 10

// nonHaloRingSizingBytes is the budget the CONTEXT-SIZING subtest sizes against. Headroom-
// adjusted (~136 KiB) it is under the full weight plan (~141 KiB), which is exactly when the
// seam switches the ordinary context charge to the ring; the ring plan (~72 KiB) still leaves
// room for a positive derived context.
const nonHaloRingSizingBytes = 160 << 10

// nonHaloCUDARingBackend stands in for a discrete CUDA GPU on a NON-Halo host: the embedded CPU
// reference executes every op, while the double re-reports the CUDA identity/capability surface
// the ring-eligibility predicate and the loader's arm gates read (Name, Tier, Caps, DeviceMemory,
// Q2_K packed support). It never claims the session-wide device-only policy.
type nonHaloCUDARingBackend struct {
	compute.Backend
	total, free int64
	tier        string
}

func newNonHaloCUDARingBackend(total, free int64) nonHaloCUDARingBackend {
	return nonHaloCUDARingBackend{Backend: compute.Default(), total: total, free: free, tier: "discrete:sm90"}
}

func (b nonHaloCUDARingBackend) Name() string { return "cuda" }
func (b nonHaloCUDARingBackend) Tier() string {
	if b.tier != "" {
		return b.tier
	}
	return "discrete:sm90"
}
func (b nonHaloCUDARingBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.CapacityProbe, c.UploadDtype = true, true, true
	return c
}
func (b nonHaloCUDARingBackend) DeviceMemory() (int64, int64, bool) { return b.total, b.free, true }
func (b nonHaloCUDARingBackend) SupportsQ2K() bool                  { return true }
func (b nonHaloCUDARingBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.Q2_K
}

// nonHaloRingMixedMoEGGUF builds the child-A intake shape (packed Q2_K gate/up slabs, raw Q3_K
// down slab, F32 router) plus the context axis ordinary context sizing reads. At E=2/H=I=256 the
// weights are ~142 KiB: against nonHaloRingDeviceBytes the full plan refuses while the dense base
// + one-token ring admits.
func nonHaloRingMixedMoEGGUF(t *testing.T, E, H, I int) []byte {
	t.Helper()
	const q2kBlock, q3kBlock = 84, 110 // 256-weight super-blocks: Q2_K 84 B, Q3_K 110 B
	perQ2 := H * I / 256 * q2kBlock
	perQ3 := I * H / 256 * q3kBlock
	slab := func(per int) []byte {
		out := make([]byte, E*per)
		for x := 0; x < E; x++ {
			for i := 0; i < per; i++ {
				out[x*per+i] = v41MixedExpertPatternByte(x, i)
			}
		}
		return out
	}
	router := make([]float32, E*H)
	for i := range router {
		router[i] = float32(i%97) * 0.001
	}
	tensors := []struct {
		name    string
		dims    []uint64
		typ     ggufload.TensorType
		payload []byte
	}{
		{name: "blk.0.ffn_down_exps.weight", dims: []uint64{uint64(I), uint64(H), uint64(E)}, typ: ggufload.TensorQ3_K, payload: slab(perQ3)},
		{name: "blk.0.ffn_gate_exps.weight", dims: []uint64{uint64(H), uint64(I), uint64(E)}, typ: ggufload.TensorQ2_K, payload: slab(perQ2)},
		{name: "blk.0.ffn_gate_inp.weight", dims: []uint64{uint64(H), uint64(E)}, typ: ggufload.TensorF32, payload: v41MixedF32Bytes(router)},
		{name: "blk.0.ffn_up_exps.weight", dims: []uint64{uint64(H), uint64(I), uint64(E)}, typ: ggufload.TensorQ2_K, payload: slab(perQ2)},
	}
	var b bytes.Buffer
	// 11 KVs: the child-A intake set plus qwen3moe.context_length for ordinary context sizing.
	writeMinimalHeaderForTest(&b, uint64(len(tensors)), 11)
	writeKVStringForTest(&b, "general.architecture", "qwen3moe")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "qwen3moe.embedding_length", uint32(H))
	writeKVUint32ForTest(&b, "qwen3moe.block_count", 1)
	writeKVUint32ForTest(&b, "qwen3moe.attention.head_count", 1)
	writeKVUint32ForTest(&b, "qwen3moe.feed_forward_length", uint32(I*2))
	writeKVFloat32ForTest(&b, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVUint32ForTest(&b, "qwen3moe.expert_count", uint32(E))
	writeKVUint32ForTest(&b, "qwen3moe.expert_used_count", 1)
	writeKVUint32ForTest(&b, "qwen3moe.expert_feed_forward_length", uint32(I))
	writeKVUint32ForTest(&b, "qwen3moe.context_length", 256)
	offsets := make([]uint64, len(tensors))
	off := 0
	for i, tt := range tensors {
		offsets[i] = uint64(off)
		off += len(tt.payload)
		off = (off + 31) / 32 * 32
	}
	for i, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, tt.dims, uint32(tt.typ), offsets[i])
	}
	padToAlignmentForTest(&b, 32)
	for _, tt := range tensors {
		b.Write(tt.payload)
		padToAlignmentForTest(&b, 32)
	}
	return b.Bytes()
}

// fak-test:runtime fast est=5s lane=default
func TestServeNonHaloCUDARingOrdinaryContextAndLoad(t *testing.T) {
	const E, H, I = 2, 256, 256
	path := filepath.Join(t.TempDir(), "mixed-moe.gguf")
	if err := os.WriteFile(path, nonHaloRingMixedMoEGGUF(t, E, H, I), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Run("predicate", func(t *testing.T) {
		cuda := newNonHaloCUDARingBackend(1<<20, 1<<20)
		if !serveExpertRingEligible(cuda, false) {
			t.Fatal("discrete CUDA backend must be ring-eligible without the device-only policy")
		}
		integrated := newNonHaloCUDARingBackend(1<<20, 1<<20)
		integrated.tier = "integrated:sm90"
		if serveExpertRingEligible(integrated, false) {
			t.Fatal("an integrated tier shares one pool with the OS; its ring keeps the historical refusal")
		}
		other := serveCapBackend{total: 1 << 20, free: 1 << 20, known: true}
		if serveExpertRingEligible(other, false) {
			t.Fatal("a non-CUDA backend must keep the historical refusal")
		}
		if serveExpertRingEligible(nil, false) || serveExpertRingEligible(nil, true) {
			t.Fatal("nil backend must never be eligible")
		}
		if !serveExpertRingEligible(other, true) {
			t.Fatal("a device-only serve stays eligible regardless of backend class")
		}
	})

	t.Run("context-sizing", func(t *testing.T) {
		be := newNonHaloCUDARingBackend(nonHaloRingSizingBytes, nonHaloRingSizingBytes)
		rt := &serveRuntime{chatBackend: be, requireDeviceExecution: false}
		if !rt.serveRingContextSizing(false, 1) {
			t.Fatal("single-rank non-offload CUDA serve must size its context against the ring")
		}
		if rt.serveRingContextSizing(true, 1) {
			t.Fatal("--cpu-offload-experts keeps its own streamed sizing")
		}
		if rt.serveRingContextSizing(false, 2) {
			t.Fatal("multi-rank expert parallelism keeps its own plan")
		}

		// The exact serve_stages.go seam: when the full weight plan overflows the budget, the
		// ordinary context charge becomes the ring plan on an eligible CUDA backend and stays the
		// full plan on a non-CUDA one.
		ws, err := ggufload.OpenWeights(path)
		if err != nil {
			t.Fatal(err)
		}
		defer ws.Close()
		fit := serveFitBudget{Base: nonHaloRingSizingBytes, Headroom: serveGGUFDeviceHeadroom}
		weights, _, err := serveNativeContextSizingInputs(ws, be, false, false, 1, 0, path)
		if err != nil {
			t.Fatalf("sizing inputs: %v", err)
		}
		if weights.DeviceTotal() <= fit.avail() {
			t.Fatalf("fixture drift: full plan %d must overflow avail %d to exercise the seam", weights.DeviceTotal(), fit.avail())
		}
		rsz, err := resolveServeExpertRingSize("", os.Getenv)
		if err != nil {
			t.Fatal(err)
		}
		ringWeights, ok := serveExpertRingContextWeights(ws, be, rt.serveRingContextSizing(false, 1), rsz, weights, fit)
		if !ok {
			t.Fatal("eligible non-Halo CUDA backend must size its ordinary context against the ring")
		}
		if ringWeights.DeviceTotal() >= weights.DeviceTotal() {
			t.Fatalf("ring charge %d not below the full plan %d", ringWeights.DeviceTotal(), weights.DeviceTotal())
		}
		capBe := serveCapBackend{total: nonHaloRingSizingBytes, free: nonHaloRingSizingBytes, known: true, uploadDtype: true}
		capRT := &serveRuntime{chatBackend: capBe, requireDeviceExecution: false}
		if capRT.serveRingContextSizing(false, 1) {
			t.Fatal("non-CUDA backend gate must stay closed")
		}
		if _, ok := serveExpertRingContextWeights(ws, capBe, capRT.serveRingContextSizing(false, 1), rsz, weights, fit); ok {
			t.Fatal("non-CUDA backend must keep the full weight plan")
		}

		// End-to-end: the ordinary entry point completes on a NON-Halo host without the
		// session-wide device-only policy and resolves a positive context.
		gguf, offload, ctxTokens, admit, gpuLayers := path, false, 0, 0, 0
		cudaGraph := false
		sf := &serveFlags{
			ggufPath:                   &gguf,
			cpuOffloadExperts:          &offload,
			nativeContextTokens:        &ctxTokens,
			nativeAdmissionTokenBudget: &admit,
			cudaGraph:                  &cudaGraph,
			nativeGPULayers:            &gpuLayers,
		}
		if err := rt.resolveNativeContext(sf, 1); err != nil {
			t.Fatalf("resolveNativeContext: %v", err)
		}
		if rt.nativeContext.ResolvedTokens <= 0 {
			t.Fatalf("eligible CUDA serve resolved no context tokens (ring plan should fit)")
		}
	})

	t.Run("load-selection", func(t *testing.T) {
		v41PackedTestEnv(t)
		be := newNonHaloCUDARingBackend(nonHaloRingLoadBytes, nonHaloRingLoadBytes)
		fit := serveFitBudget{Base: nonHaloRingLoadBytes, Headroom: serveGGUFDeviceHeadroom}
		// 64-token context budget: the ring plan (base + ring ~72 KiB + 192 KiB KV + scratch)
		// admits while the full plan (~141 KiB weights + the same KV and scratch) refuses.
		m, inKernelQ4K, profile, _ := loadServeInKernelModelPlaced(path, be, false, false, 64, nil, 1, &fit)
		if m == nil || !inKernelQ4K {
			t.Fatalf("ring load did not select resident packed storage (q4k=%t)", inKernelQ4K)
		}
		defer m.CloseWeights()
		residency := ""
		for _, msg := range profile.Messages {
			if msg.Kind == "serving-expert-residency" && msg.Level == "info" {
				residency = msg.Text
			}
		}
		if !containsAll(residency, "planned device expert-ring capacity=", "ring scope=model") {
			t.Fatalf("no admitted ring message: %q", residency)
		}
		if m.SharedExpertRingFor(be) == nil {
			t.Fatal("the declared device ring budget must register the model-scoped shared ring")
		}
		report := m.ResidentReport()
		wantRouterQ8 := (int64(E*H)*36 + 31) / 32
		if report.KQuantTensors != 0 || report.KQuantBytes != 0 {
			t.Fatalf("stream-through experts must leave no packed expert residency: tensors=%d bytes=%d", report.KQuantTensors, report.KQuantBytes)
		}
		if report.F32Bytes != 0 || report.Q8Bytes != wantRouterQ8 {
			t.Fatalf("dense side storage: f32=%d q8=%d want q8=%d (router gate native-Q8, no whole-slab FP32 decode)", report.F32Bytes, report.Q8Bytes, wantRouterQ8)
		}
		replicated, expert, ok := m.MoEResidentWeightBytes()
		if !ok || expert != 0 || replicated != wantRouterQ8 {
			t.Fatalf("selected consumers: replicated=%d expert=%d ok=%t, want replicated=%d expert=0", replicated, expert, ok, wantRouterQ8)
		}
		stats := m.ExpertCheckpointStats()
		if !stats.Enabled || stats.Tensors != 3*E {
			t.Fatalf("checkpoint tier must index the selected packed expert ranges: enabled=%t tensors=%d want %d", stats.Enabled, stats.Tensors, 3*E)
		}
	})

	t.Run("below-floor", func(t *testing.T) {
		ws, err := ggufload.OpenWeights(path)
		if err != nil {
			t.Fatal(err)
		}
		defer ws.Close()
		be := newNonHaloCUDARingBackend(4096, 4096)
		ring, ok, err := serveActivatedExpertRingPlacement(ws, be, false, serveRingFullPlanTooBig(), 0, serveFitBudget{Base: 4096})
		var fe *compute.FitError
		if ok || !errors.As(err, &fe) || fe.Verdict != compute.FitTooBig {
			t.Fatalf("below the activated floor: ok=%v err=%v, want typed FitTooBig", ok, err)
		}
		if ring.Plan != nil || ring.RingBytes != 0 {
			t.Fatalf("refusal returned an admitted placement: %+v", ring)
		}
	})

	t.Run("unstageable", func(t *testing.T) {
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
		be := newNonHaloCUDARingBackend(1<<20, 1<<20)
		_, ok, err := serveActivatedExpertRingPlacement(ws, be, false, serveRingFullPlanTooBig(), 0, serveFitBudget{Base: 1 << 20})
		var fe *compute.FitError
		if ok || !errors.Is(err, errServeActivatedExpertRingUnstageable) || !errors.As(err, &fe) {
			t.Fatalf("unstageable checkpoint: ok=%v err=%v, want FitTooBig naming the unstageable ring", ok, err)
		}
	})
}
