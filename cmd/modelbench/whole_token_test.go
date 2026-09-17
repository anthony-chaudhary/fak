package main

import (
	"encoding/json"
	"errors"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestWholeTokenWitnessUsesPromotedP1AndPreservesProfilerMode(t *testing.T) {
	// The promoted-P1 resident graph is a Metal-only instrument; the off-Metal
	// backend lane (Linux CPU / ROCm / Vulkan) is covered by
	// TestWholeTokenRunsWithoutMetal. On darwin this remains the physical oracle.
	if runtime.GOOS != "darwin" {
		t.Skipf("resident-Metal whole-token oracle requires darwin/arm64+cgo, this host is %s", runtime.GOOS)
	}
	if !metalgemm.Available() {
		t.Fatal("physical Metal device required")
	}
	cfg := model.Config{
		HiddenSize: 256, NumLayers: 4, NumHeads: 4, NumKVHeads: 2,
		HeadDim: 64, IntermediateSize: 256, VocabSize: 512,
		RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1,
		LayerTypes:          []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim: 3, LinearKeyHeadDim: 64, LinearNumKeyHeads: 2,
		LinearValueHeadDim: 64, LinearNumValueHeads: 4, AttnOutputGate: true,
		FullAttentionInterval: 4, NormGain1p: true, QKNorm: true,
	}
	m := model.NewSynthetic(cfg)
	var closeOnce sync.Once
	closeCalls := 0
	closeWeights := func() (err error) {
		closeOnce.Do(func() { closeCalls++; err = m.CloseWeights() })
		return err
	}
	t.Cleanup(func() {
		if err := closeWeights(); err != nil {
			t.Error(err)
		}
	})
	m.Quantize()
	newSession := func() *model.Session {
		s := m.NewSession()
		s.Q4K, s.Metal, s.MetalQ4K = true, true, true
		return s
	}
	// Existing profile behavior still opts into per-layer phase observation.
	profileSession := newSession()
	profiler, err := configureNativeProfileSession(profileSession, map[string]string{
		nativeProfileSequenceSelector:     nativeProfileSelectorOn,
		nativeProfileDecodeHandoffControl: model.Qwen35DecodeHandoffAuto.String(),
	})
	profileSession.Close()
	if err != nil || profiler == nil || profileSession.PhaseProfiler != profiler {
		t.Fatalf("existing profiler mode changed: profiler=%v err=%v", profiler, err)
	}
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*19 + 7) % cfg.VocabSize
	}
	report, err := runWholeToken(newSession(), prompt, 2, "whole-token", time.Now(), closeWeights)
	if err != nil {
		t.Fatal(err)
	}
	if closeCalls != 1 || report.Prefill.Tokens != 32 || len(report.Decode) != 2 || len(report.Tokens) != 2 || len(report.GreedyNext) != 2 || report.Handoff.BlockAcceptedCalls != 2 {
		t.Fatalf("runner omitted lifecycle/execution proof: %+v closes=%d", report, closeCalls)
	}
	var totalPhases float64
	for _, phase := range []string{"load_setup", "session_setup", "prefill", "finalize", "decode_with_head", "host_selection_and_receipt_bookkeeping", "cleanup"} {
		if report.Phases[phase] <= 0 {
			t.Fatalf("missing measured %s phase", phase)
		}
		totalPhases += report.Phases[phase]
	}
	if report.TotalSeconds < totalPhases || report.PeakRSSBytes == 0 {
		t.Fatalf("incomplete total/memory: %+v", report)
	}
	for _, receipt := range report.Decode {
		if receipt.Tokens != 1 || !receipt.CompletedWait || receipt.IntermediateReadbacks != 0 {
			t.Fatalf("missing physical P1: %+v", receipt)
		}
	}
	t.Logf("device=%s phases=%v P1=%+v", report.Device, report.Phases, report.Decode)
}

func TestWholeTokenSerializedReceiptRejectsStaleAndTamperedEvidence(t *testing.T) {
	op := wholeTokenOperation{WholeSequenceOperation: model.WholeSequenceOperation{CacheBefore: 32, CacheAfter: 33}}
	op.Before.Tokens = 32
	// A real serialized report carries the runtime's concrete capability path, which
	// the report also records at build time; readback must pin to that recorded token.
	const sequencePath = "metal/qwen35-gdn-preprojected-sequence-v1"
	op.After = model.WholeSequenceReceipt{Tokens: 1, Committed: true, CompletedWait: true, CommandBuffers: 1, TerminalWaits: 1, TerminalReadbacks: 1, Path: sequencePath, EvidenceState: model.WholeSequenceEvidenceExecuted}
	op.CountsAfter.BlockAcceptedCalls = 1
	report := wholeTokenReport{ExpectedRoute: "whole-token", SequencePath: sequencePath, Tokens: []int{7}, Operations: []wholeTokenOperation{op}}
	report.BindingSHA256, _ = wholeTokenBinding(report)
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	var decoded wholeTokenReport
	if err = json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if err = validateWholeTokenReport(decoded); err != nil {
		t.Fatal(err)
	}
	decoded.Tokens[0]++
	if validateWholeTokenReport(decoded) == nil {
		t.Fatal("accepted tampered serialized report")
	}
	decoded.Operations[0].Before = decoded.Operations[0].After
	decoded.BindingSHA256, _ = wholeTokenBinding(decoded)
	if validateWholeTokenReport(decoded) == nil {
		t.Fatal("accepted stale receipt with a recomputed binding")
	}
	// A fabricated capability path must not revalidate even with a recomputed
	// binding: readback pins the receipt to the token the report recorded.
	fabricated := report
	fabricated.Operations = append([]wholeTokenOperation(nil), op)
	fabricated.Operations[0].After.Path = "fabricated/other-backend-v1"
	fabricated.Operations[0].Before.Path = "fabricated/other-backend-v1"
	fabricated.BindingSHA256, _ = wholeTokenBinding(fabricated)
	if validateWholeTokenReport(fabricated) == nil {
		t.Fatal("accepted a receipt with a fabricated capability path and recomputed binding")
	}
	op.After = op.Before
	op.CountsAfter.ResidentAcceptedCalls = 0
	op.CountsAfter.BlockAcceptedCalls = 2
	if validateWholeTokenOperation(op, "per-layer", nil, sequencePath) == nil {
		t.Fatal("accepted two block calls for one Step")
	}
	op.CountsAfter.BlockAcceptedCalls = 1
	if err := validateWholeTokenOperation(op, "per-layer", nil, sequencePath); err != nil {
		t.Fatal(err)
	}
}

// deviceIdentitySeamBackend is a fake compute.Backend whose Name/Tier are fixture
// values, so the whole-token device-identity seam can be exercised with no Metal
// device and no registration side effects.
type deviceIdentitySeamBackend struct {
	compute.Backend
	name, tier string
}

func (b deviceIdentitySeamBackend) Name() string { return b.name }
func (b deviceIdentitySeamBackend) Tier() string { return b.tier }

func TestWholeTokenDeviceIdentitySeam(t *testing.T) {
	fallback := metalgemm.DeviceName()

	if got := wholeTokenDeviceIdentity(&model.Session{
		Backend: deviceIdentitySeamBackend{Backend: compute.Default(), name: "fixture-dev", tier: "fixture-tier"},
	}); got != "fixture-dev/fixture-tier" {
		t.Fatalf("backend name/tier identity = %q, want %q", got, "fixture-dev/fixture-tier")
	}

	if got := wholeTokenDeviceIdentity(&model.Session{
		Backend: deviceIdentitySeamBackend{Backend: compute.Default(), name: "fixture-dev", tier: ""},
	}); got != "fixture-dev" {
		t.Fatalf("backend name-only identity = %q, want %q", got, "fixture-dev")
	}

	if got := wholeTokenDeviceIdentity(&model.Session{}); got != fallback {
		t.Fatalf("nil backend identity = %q, want legacy fallback %q", got, fallback)
	}

	if got := wholeTokenDeviceIdentity(nil); got != fallback {
		t.Fatalf("nil session identity = %q, want legacy fallback %q", got, fallback)
	}
}

// wholeTokenBackendFixture is a non-Metal compute.Backend that advertises the
// whole-sequence prefill seam (the TICKET-02 route) and structurally implements
// the Qwen35 GDN contract, so the whole-token witness can run end-to-end on any
// host — Linux CPU included. It owns no GPU kernel: matmuls and attention run on
// the embedded cpu-ref backend, and its sequence/GDN bodies return finite,
// deterministic values. It is a routing fixture, not a model-quality witness.
type wholeTokenBackendFixture struct {
	compute.Backend
	m             *model.Model
	sequenceCalls int
	gdnCalls      int
}

func (b *wholeTokenBackendFixture) Name() string                    { return "linux-cpu-fixture" }
func (b *wholeTokenBackendFixture) Tier() string                    { return "fixture" }
func (b *wholeTokenBackendFixture) Class() compute.CorrectnessClass { return compute.Approx }
func (*wholeTokenBackendFixture) Qwen35GDNPath() string             { return model.Qwen35GDNCUDAPath }

// Qwen35SequencePrefill appends KV rows and returns finite resident products,
// mirroring the model package's recording fixture without any device.
func (b *wholeTokenBackendFixture) Qwen35SequencePrefill(req compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	b.sequenceCalls++
	for token := range req.TokenIDs {
		pos := req.StartPos + token
		for layer := range req.Layers {
			width := req.NumKVHeads * req.HeadDim
			z := compute.NewF32(b.Backend, []int{width}, make([]float32, width))
			req.KV.AppendKV(layer, z, z, z, pos)
		}
	}
	hidden := compute.NewF32(b.Backend, []int{req.Hidden}, make([]float32, req.Hidden))
	var logits compute.Tensor
	if req.NeedLogits {
		logits = compute.NewF32(b.Backend, []int{b.m.Cfg.VocabSize}, make([]float32, b.m.Cfg.VocabSize))
	}
	return compute.Qwen35SequencePrefillResult{LastHidden: hidden, Logits: logits, Tokens: len(req.TokenIDs)}, nil
}

// Qwen35GDNDecode honors the in-place state contract and returns a finite output
// of the correct width; the decode loop's finiteness is what the witness checks,
// not numeric quality.
func (b *wholeTokenBackendFixture) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState compute.Tensor, err error) {
	b.gdnCalls++
	width := len(b.Backend.Read(normalizedInput))
	if width == 0 {
		width = b.m.Cfg.HiddenSize
	}
	out := compute.NewF32(b.Backend, []int{width}, make([]float32, width))
	return out, convState, recurrentState, nil
}

// wholeTokenSequenceBackend is the same fixture with the whole-sequence prefill
// capability advertised. The two-type split is deliberate: a Go interface is
// satisfied statically, so "this backend provides no whole-sequence owner" must
// be a type that does not implement the path marker at all — exactly how a real
// non-sequence backend presents itself to the admission gate.
type wholeTokenSequenceBackend struct {
	wholeTokenBackendFixture
}

func (*wholeTokenSequenceBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

// wholeTokenFixtureCfg is a small Qwen3.5 GDN hybrid whose vocabulary is large
// enough for 32 distinct prompt IDs and whose full-attention layer exercises the
// standard HAL attention path.
func wholeTokenFixtureCfg() model.Config {
	return model.Config{
		HiddenSize: 32, NumLayers: 4, NumHeads: 4, NumKVHeads: 2,
		HeadDim: 8, IntermediateSize: 64, VocabSize: 97,
		RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1,
		LayerTypes:          []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim: 3, LinearKeyHeadDim: 8, LinearNumKeyHeads: 2,
		LinearValueHeadDim: 8, LinearNumValueHeads: 4, AttnOutputGate: true,
		FullAttentionInterval: 4, NormGain1p: true,
	}
}

func wholeTokenFixturePrompt(vocab int) []int {
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*19 + 7) % vocab
	}
	return prompt
}

// TestWholeTokenRunsWithoutMetal drives the whole-token witness end-to-end on a
// compute.Backend-selected session: no Metal device, no backend-nil legacy lane.
// It asserts a finite decode loop, an honest non-empty Device stamped from the
// selected backend, and a typed refusal when the selected backend advertises no
// whole-sequence owner.
func TestWholeTokenRunsWithoutMetal(t *testing.T) {
	cfg := wholeTokenFixtureCfg()
	m := model.NewSynthetic(cfg)
	var closeOnce sync.Once
	closeCalls := 0
	closeWeights := func() (err error) {
		closeOnce.Do(func() { closeCalls++; err = m.CloseWeights() })
		return err
	}
	t.Cleanup(func() {
		if err := closeWeights(); err != nil {
			t.Error(err)
		}
	})

	be := &wholeTokenSequenceBackend{wholeTokenBackendFixture: wholeTokenBackendFixture{Backend: compute.Default(), m: m}}
	session, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("backend-selected session refused on the whole-sequence lane: %v", err)
	}
	if path, admissible := session.WholeSequenceCapability(); !admissible || path != compute.Qwen35SequencePrefillPath {
		t.Fatalf("backend whole-sequence capability = %q admissible=%t, want %q", path, admissible, compute.Qwen35SequencePrefillPath)
	}
	session.Close()

	report, err := runWholeToken(m.NewBackendSession(be), wholeTokenFixturePrompt(cfg.VocabSize), 3, "whole-token", time.Now(), closeWeights)
	if err != nil {
		t.Fatalf("off-Metal whole-token witness failed: %v", err)
	}
	if (report.Device != "linux-cpu-fixture/fixture") && (report.Device != "linux-cpu-fixture") {
		t.Fatalf("device identity = %q, want the selected backend stamp", report.Device)
	}
	if report.SequencePath != compute.Qwen35SequencePrefillPath {
		t.Fatalf("sequence path = %q, want the backend whole-sequence route", report.SequencePath)
	}
	if len(report.Tokens) != 3 || len(report.GreedyNext) != 3 || len(report.Operations) != 3 {
		t.Fatalf("decode loop did not run %d finite steps: %+v", 3, report)
	}
	if report.Prefill.Tokens != 32 || !report.Prefill.CompletedWait || !report.Prefill.Available {
		t.Fatalf("missing backend P32 prefill evidence: %+v", report.Prefill)
	}
	if report.PeakRSSBytes == 0 {
		t.Fatalf("missing peak-RSS evidence: %+v", report)
	}
	if be.sequenceCalls != 1 {
		t.Fatalf("backend sequence prefill calls = %d, want exactly one whole-prompt call", be.sequenceCalls)
	}
	if be.gdnCalls == 0 {
		t.Fatal("decode loop did not exercise the linear-attention GDN route")
	}
	// Greedy selection above rejects non-finite logits per step, so a completed
	// loop with recorded tokens IS the finite-decode assertion.
	for _, id := range report.Tokens {
		if id < 0 || id >= cfg.VocabSize {
			t.Fatalf("forwarded token id %d outside vocabulary", id)
		}
	}
	t.Logf("device=%s path=%s phases=%v", report.Device, report.SequencePath, report.Phases)
}

// TestWholeTokenRefusesBackendWithoutWholeSequence is the Quarantined Fallback
// witness: a backend that advertises no whole-sequence owner is refused with a
// typed admission error before any prompt byte is consumed — never silently run
// through a per-layer host fallback.
func TestWholeTokenRefusesBackendWithoutWholeSequence(t *testing.T) {
	cfg := wholeTokenFixtureCfg()
	m := model.NewSynthetic(cfg)
	t.Cleanup(func() { _ = m.CloseWeights() })

	be := &wholeTokenBackendFixture{Backend: compute.Default(), m: m}
	session, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("session construction: %v", err)
	}
	if path, admissible := session.WholeSequenceCapability(); admissible || path != "" {
		t.Fatalf("capability = %q admissible=%t, want no whole-sequence owner", path, admissible)
	}
	session.Close()

	report, err := runWholeToken(m.NewBackendSession(be), wholeTokenFixturePrompt(cfg.VocabSize), 2, "whole-token", time.Now(), nil)
	if err == nil {
		t.Fatalf("accepted a backend with no whole-sequence owner: %+v", report)
	}
	if !strings.Contains(err.Error(), "whole-sequence-capable") {
		t.Fatalf("refusal is not typed/named: %v", err)
	}
	if be.sequenceCalls != 0 || be.gdnCalls != 0 {
		t.Fatalf("refused witness still executed: sequence=%d gdn=%d", be.sequenceCalls, be.gdnCalls)
	}
}

// TestWholeTokenDeepSeekV41Admission is the V4.1 admission witness: a DeepSeek
// V4.1 Flash session is admitted to the whole-sequence seam and refused by name
// with a TYPED, non-qualifying report — never a silent per-layer fallback and
// never routed through the Qwen35 backend adapter. The negative control proves
// the refusal is architecture-keyed (a Qwen35 hybrid config returns no V4.1
// reason), not a blanket refusal.
func TestWholeTokenDeepSeekV41Admission(t *testing.T) {
	cfg := wholeTokenFixtureCfg()
	m := model.NewSynthetic(cfg)
	t.Cleanup(func() { _ = m.CloseWeights() })
	// Model.Cfg is an exported value field, so the architecture key is settable
	// from package main without constructing the full official V4.1 wrapper.
	m.Cfg.ModelType = "deepseek_v41"

	if !m.Cfg.IsDeepSeekV41() {
		t.Fatal("fixture config is not recognized as DeepSeek V4.1")
	}
	// Not-mis-admitted-as-Qwen regression: even though this synthetic fixture
	// still carries linear_attention layers (so IsQwen35Hybrid stays true), the
	// V4.1 identity must win at the admission seam — the session exposes no
	// Qwen35 capability and carries the typed V4.1 refusal instead.

	session := m.NewSession()
	t.Cleanup(session.Close)
	if path, admissible := session.WholeSequenceCapability(); admissible || path != "" {
		t.Fatalf("V4.1 capability = %q admissible=%t, want refused", path, admissible)
	}
	reason := session.WholeSequenceUnsupportedReason()
	if reason == nil {
		t.Fatal("V4.1 session exposed no typed whole-sequence refusal")
	}
	if !errors.Is(reason, model.ErrV41WholeSequenceUnsupported) {
		t.Fatalf("refusal is not typed as ErrV41WholeSequenceUnsupported: %v", reason)
	}

	report, err := runWholeToken(session, wholeTokenFixturePrompt(cfg.VocabSize), 2, "whole-token", time.Now(), nil)
	if err == nil {
		t.Fatalf("V4.1 session was admitted to the whole-token witness: %+v", report)
	}
	if !errors.Is(err, model.ErrV41WholeSequenceUnsupported) {
		t.Fatalf("refusal is not the typed V4.1 refusal: %v", err)
	}
	if report.Error == "" {
		t.Fatal("non-qualifying V4.1 report did not record the refusal")
	}
	if report.SequencePath != "" {
		t.Fatalf("non-qualifying V4.1 report claimed a sequence path %q", report.SequencePath)
	}
	if len(report.Tokens) != 0 || len(report.Operations) != 0 {
		t.Fatalf("refused V4.1 witness still executed: tokens=%d operations=%d", len(report.Tokens), len(report.Operations))
	}

	// Negative control: the unchanged Qwen35 fixture is not hit by the V4.1
	// refusal, so the branch is keyed on architecture, not blanket.
	qm := model.NewSynthetic(wholeTokenFixtureCfg())
	t.Cleanup(func() { _ = qm.CloseWeights() })
	qs := qm.NewSession()
	t.Cleanup(qs.Close)
	if r := qs.WholeSequenceUnsupportedReason(); r != nil {
		t.Fatalf("Qwen35 fixture hit by the V4.1 refusal: %v", r)
	}
}

// TestValidateWholeTokenFlagsAdmitsBackendLane pins the flag gate: the legacy
// lane still demands -metal, a named backend must omit -metal, and the backend
// lane no longer requires -metal to select the witness.
func TestValidateWholeTokenFlagsAdmitsBackendLane(t *testing.T) {
	f := testCompleteBenchFlags()
	*wholeTokenOut = "witness.json"
	*wholeTokenPrompt = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"
	t.Cleanup(func() { *wholeTokenOut = "" })
	*f.gguf, *f.q4k, *f.decodeSteps = "model.gguf", true, 2

	*f.backendName, *f.metal = "legacy", false
	if err := validateWholeTokenFlags(f); err == nil {
		t.Fatal("legacy lane without -metal was admitted")
	}
	*f.backendName, *f.metal = "legacy", true
	if err := validateWholeTokenFlags(f); err != nil {
		t.Fatalf("legacy Metal lane refused: %v", err)
	}
	*f.backendName, *f.metal = "vulkan", false
	if err := validateWholeTokenFlags(f); err != nil {
		t.Fatalf("named backend lane refused: %v", err)
	}
	*f.backendName, *f.metal = "vulkan", true
	if err := validateWholeTokenFlags(f); err == nil {
		t.Fatal("-metal with a named -backend was admitted")
	}
}
