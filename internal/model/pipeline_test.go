package model

import (
	"math"
	"net"
	"strings"
	"sync"
	"testing"
)

// pipeline_test.go — the cross-worker handoff contract. forward_band_test.go
// proves a band runs in-process over a SHARED model; this proves the harder,
// real-deployment case: an N-stage pipeline where EACH stage is a SEPARATELY
// windowed-loaded model (only its band resident) and the hidden state crosses a
// serialize->bytes->deserialize boundary still yields bit-identical logits to one
// monolithic Forward. That is the cross-device-transport correctness gate
// (GLM-5.2-NATIVE-ENGINE-GAP first-land #2's remaining half), runnable on one box.

// TestPipelineSeparatelyLoadedStagesMatchMonolithic is the headline proof. On a
// 3-layer GLM-MoE-DSA checkpoint (indexers full,shared,full), it loads two stages
// STANDALONE via WithLayerWindow — stage A [0,2) keeps the full+shared DSA group
// intact, stage B [2,3) resumes at the full-indexer layer — runs them through
// RunPipeline (hidden state crosses the marshalled boundary), and asserts the
// logits are bit-identical to the monolithic full-window Forward. The residency
// checks prove each stage genuinely ran from its own narrowed checkpoint, so a
// bit-exact pass cannot be an artifact of accidentally running a full model.
func TestPipelineSeparatelyLoadedStagesMatchMonolithic(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)
	if cfg.NumLayers != 3 {
		t.Fatalf("fixture NumLayers = %d, test assumes 3", cfg.NumLayers)
	}

	mono, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir (monolithic): %v", err)
	}
	monoAct := mono.Forward([]int{3, 1, 4, 1, 5})

	stageA, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(0, 2))
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir (stage A [0,2)): %v", err)
	}
	stageB, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(2, 3))
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir (stage B [2,3)): %v", err)
	}

	// Each stage holds ONLY its band's resident matmul weights — proving the
	// pipeline ran across genuinely partitioned checkpoints, not one shared model.
	assertNoLayerTensors(t, "stage A", stageA, 2, 3)
	assertNoLayerTensors(t, "stage B", stageB, 0, 1)
	if !hasAnyLayerTensor(stageA, 0) || !hasAnyLayerTensor(stageA, 1) {
		t.Fatalf("stage A [0,2) is missing layer 0/1 weights; nothing to run")
	}
	if !hasAnyLayerTensor(stageB, 2) {
		t.Fatalf("stage B [2,3) is missing layer 2 weights; nothing to run")
	}

	logits, err := RunPipeline([]int{3, 1, 4, 1, 5}, []PipelineStage{
		{Spec: StageSpec{Lo: 0, Hi: 2, First: true}, Model: stageA},
		{Spec: StageSpec{Lo: 2, Hi: 3, Last: true}, Model: stageB},
	})
	if err != nil {
		t.Fatalf("RunPipeline: %v", err)
	}

	if len(logits) != len(monoAct.Logits) {
		t.Fatalf("pipeline logits seq = %d, monolithic = %d", len(logits), len(monoAct.Logits))
	}
	var maxAbs float32
	for t2 := range logits {
		if len(logits[t2]) != len(monoAct.Logits[t2]) {
			t.Fatalf("pipeline logits[%d] len = %d, monolithic = %d", t2, len(logits[t2]), len(monoAct.Logits[t2]))
		}
		for i := range logits[t2] {
			d := logits[t2][i] - monoAct.Logits[t2][i]
			if d < 0 {
				d = -d
			}
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	if maxAbs != 0 {
		t.Fatalf("separately-loaded pipeline logits differ from monolithic: max|delta|=%.3e (want bit-exact 0)", maxAbs)
	}
}

// TestDecodeBandGLMDsaMonolithicNoOp pins that routing tokenHiddenGLMDsa through
// decodeBandGLMDsa(0,N,first,last) did not move a bit: two fresh monolithic sessions
// must generate identical tokens (determinism), the cheapest guard that the refactor's
// single instruction stream is intact. (The whole GLM-DSA suite also re-covers this.)
func TestDecodeBandGLMDsaMonolithicNoOp(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)
	m, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	a := m.NewSession().Generate([]int{3, 1, 4, 1}, 6)
	b := m.NewSession().Generate([]int{3, 1, 4, 1}, 6)
	if len(a) != len(b) {
		t.Fatalf("nondeterministic len: %v vs %v", a, b)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("monolith nondeterministic at %d: %v vs %v", i, a, b)
		}
	}
}

// TestPipelineIncrementalMatchesMonolithicSession is the correctness gate for incremental
// decode: greedy decode where EACH stage keeps its band's KV resident and advances one
// position per call must produce token ids identical to monolithic Session.Generate, on
// the same fixture/split as the stateless-pipeline test, isolating the incremental path.
func TestPipelineIncrementalMatchesMonolithicSession(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)
	mono, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("mono load: %v", err)
	}
	stageA, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(0, 2))
	if err != nil {
		t.Fatalf("stage A load: %v", err)
	}
	stageB, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(2, 3))
	if err != nil {
		t.Fatalf("stage B load: %v", err)
	}
	prompt := []int{3, 1, 4, 1}
	const nGen = 6
	want := mono.NewSession().Generate(prompt, nGen)
	got, err := PipelineGenerateIncremental(prompt, nGen, []*PipelineStageDecoder{
		NewPipelineStageDecoder(StageSpec{Lo: 0, Hi: 2, First: true}, stageA),
		NewPipelineStageDecoder(StageSpec{Lo: 2, Hi: 3, Last: true}, stageB),
	})
	if err != nil {
		t.Fatalf("PipelineGenerateIncremental: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("incremental %d tokens, mono %d: got=%v want=%v", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("incremental diverged at token %d: got=%v want=%v", i, got, want)
		}
	}
}

// TestPipelineIncrementalLogitsBitExactPerStep asserts the FULL logits vector at every
// decode position is bit-identical (max|Δ|=0) between incremental pipeline decode and a
// monolithic Session stepped in lockstep — stronger than the token-id gate (a sub-argmax
// delta a near-tie hides would still trip this). A fresh Session.Step(id) decodes from
// pos 0 (Step passes Cache.Len()), so the mono side never calls Prefill and every
// position's logits are observable, matching the pipeline's per-position output exactly.
func TestPipelineIncrementalLogitsBitExactPerStep(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)
	mono, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("mono load: %v", err)
	}
	stageA, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(0, 2))
	if err != nil {
		t.Fatalf("stage A load: %v", err)
	}
	stageB, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(2, 3))
	if err != nil {
		t.Fatalf("stage B load: %v", err)
	}
	stages := []*PipelineStageDecoder{
		NewPipelineStageDecoder(StageSpec{Lo: 0, Hi: 2, First: true}, stageA),
		NewPipelineStageDecoder(StageSpec{Lo: 2, Hi: 3, Last: true}, stageB),
	}
	if _, err := validateDecoderPlan(stages); err != nil {
		t.Fatalf("validate: %v", err)
	}
	ResetStageDecoders(stages)

	ms := mono.NewSession()
	prompt := []int{3, 1, 4, 1}
	var monoLogits []float32
	bitEq := func(pos int, got, want []float32) {
		if len(got) != len(want) {
			t.Fatalf("pos %d logits len %d != mono %d", pos, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("pos %d logit[%d] bits %#x != mono %#x", pos, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	for pos, id := range prompt {
		monoLogits = ms.Step(id)
		pipeLogits, err := StepPipelineDecode(id, pos, stages)
		if err != nil {
			t.Fatalf("prefill StepPipelineDecode pos %d: %v", pos, err)
		}
		bitEq(pos, pipeLogits, monoLogits)
	}
	pos := len(prompt)
	for i := 0; i < 6; i++ {
		next := argmaxF32(monoLogits)
		if cfg.IsEOS(next) {
			break
		}
		monoLogits = ms.Step(next)
		pipeLogits, err := StepPipelineDecode(next, pos, stages)
		if err != nil {
			t.Fatalf("decode StepPipelineDecode %d: %v", i, err)
		}
		bitEq(pos, pipeLogits, monoLogits)
		pos++
	}
}

// TestPipelineIncrementalRejectsSharedBoundary proves a band beginning on the shared-
// indexer layer (1) is refused with an error (it has no top-k of its own), not a panic —
// the decodeBandGLMDsa lo-guard returns before any glmDsaAttentionStep call.
func TestPipelineIncrementalRejectsSharedBoundary(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)
	m, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	dec := NewPipelineStageDecoder(StageSpec{Lo: 1, Hi: 3, Last: true}, m)
	if _, err := dec.decode(0, make([]float32, cfg.HiddenSize), 0); err == nil {
		t.Fatal("expected decodeBandGLMDsa to reject a shared-indexer band start")
	}
}

// countingTransport wraps LocalTransport to prove a custom StageTransport is actually
// invoked at each boundary and that swapping the transport changes nothing the bit-exact
// gates depend on — the NCCL/RPC wire seam exercised with a real (non-default) impl.
type countingTransport struct {
	inner StageTransport
	sends int
}

func (c *countingTransport) Send(hidden [][]float32, nextLo, dstStage int) ([][]float32, int, error) {
	c.sends++
	return c.inner.Send(hidden, nextLo, dstStage)
}

// TestStageTransportSeamMatchesLocal proves the StageTransport seam: a custom transport
// (here a counting wrapper over LocalTransport) is invoked once per interior boundary on
// BOTH the re-forward (RunPipelineWith) and incremental (StepPipelineDecodeWith) paths,
// and produces results identical to the default LocalTransport. This is the plug point a
// real NCCL/RPC backend swaps into; the test pins that swapping it is behavior-preserving.
func TestStageTransportSeamMatchesLocal(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)
	mkStages := func() []PipelineStage {
		a, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(0, 2))
		if err != nil {
			t.Fatalf("stage A: %v", err)
		}
		b, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(2, 3))
		if err != nil {
			t.Fatalf("stage B: %v", err)
		}
		return []PipelineStage{
			{Spec: StageSpec{Lo: 0, Hi: 2, First: true}, Model: a},
			{Spec: StageSpec{Lo: 2, Hi: 3, Last: true}, Model: b},
		}
	}
	ids := []int{3, 1, 4, 1, 5}

	base, err := RunPipeline(ids, mkStages())
	if err != nil {
		t.Fatalf("RunPipeline (local): %v", err)
	}
	ct := &countingTransport{inner: LocalTransport{}}
	via, err := RunPipelineWith(ids, mkStages(), ct)
	if err != nil {
		t.Fatalf("RunPipelineWith (custom): %v", err)
	}
	if ct.sends != 1 {
		t.Fatalf("custom transport sends = %d, want 1 (one interior boundary in a 2-stage pipeline)", ct.sends)
	}
	if len(via) != len(base) {
		t.Fatalf("transport-routed logits seq = %d, local = %d", len(via), len(base))
	}
	var maxAbs float32
	for t2 := range via {
		for i := range via[t2] {
			d := via[t2][i] - base[t2][i]
			if d < 0 {
				d = -d
			}
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	if maxAbs != 0 {
		t.Fatalf("custom transport changed logits: max|delta|=%.3e (want 0)", maxAbs)
	}

	// Incremental path: the transport is invoked once per interior boundary per token.
	decStages := []*PipelineStageDecoder{
		NewPipelineStageDecoder(StageSpec{Lo: 0, Hi: 2, First: true}, mkStages()[0].Model),
		NewPipelineStageDecoder(StageSpec{Lo: 2, Hi: 3, Last: true}, mkStages()[1].Model),
	}
	if _, err := validateDecoderPlan(decStages); err != nil {
		t.Fatalf("validate: %v", err)
	}
	ResetStageDecoders(decStages)
	ict := &countingTransport{inner: LocalTransport{}}
	wantTok := mono(t, dir, cfg).NewSession().Generate(ids, 4)
	gotTok := make([]int, 0, 4)
	var logits []float32
	for pos, id := range ids {
		l, err := StepPipelineDecodeWith(id, pos, decStages, ict)
		if err != nil {
			t.Fatalf("prefill StepPipelineDecodeWith pos %d: %v", pos, err)
		}
		logits = l
	}
	pos := len(ids)
	for i := 0; i < 4; i++ {
		next := argmaxF32(logits)
		gotTok = append(gotTok, next)
		if cfg.IsEOS(next) {
			break
		}
		l, err := StepPipelineDecodeWith(next, pos, decStages, ict)
		if err != nil {
			t.Fatalf("decode StepPipelineDecodeWith %d: %v", i, err)
		}
		logits = l
		pos++
	}
	if ict.sends == 0 {
		t.Fatal("incremental custom transport was never invoked")
	}
	for i := range gotTok {
		if i < len(wantTok) && gotTok[i] != wantTok[i] {
			t.Fatalf("incremental via custom transport diverged at %d: got=%v want=%v", i, gotTok, wantTok)
		}
	}
}

// mono is a tiny helper that loads the full-window model for an oracle comparison.
func mono(t *testing.T, dir string, cfg Config) *Model {
	t.Helper()
	m, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("mono load: %v", err)
	}
	return m
}

// TestPipelineGenerateMatchesMonolithicSession proves the RUNNABLE generate path
// is correct, not just single-shot logits: greedy decode routed through two
// separately windowed-loaded GLM-DSA stages produces token ids identical to the
// monolithic Session.Generate from the full-window load. This is the gate behind
// `cmd/pipelinegen` — native pipelined generation == native monolithic generation.
func TestPipelineGenerateMatchesMonolithicSession(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)

	mono, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir (monolithic): %v", err)
	}
	stageA, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(0, 2))
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir (stage A): %v", err)
	}
	stageB, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(2, 3))
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir (stage B): %v", err)
	}

	prompt := []int{3, 1, 4, 1}
	const nGen = 6
	want := mono.NewSession().Generate(prompt, nGen)
	got, err := PipelineGenerate(prompt, nGen, []PipelineStage{
		{Spec: StageSpec{Lo: 0, Hi: 2, First: true}, Model: stageA},
		{Spec: StageSpec{Lo: 2, Hi: 3, Last: true}, Model: stageB},
	})
	if err != nil {
		t.Fatalf("PipelineGenerate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("pipeline generated %d tokens, monolithic %d: got=%v want=%v", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("pipeline generation diverged at token %d: got=%v want=%v", i, got, want)
		}
	}
}

// TestHiddenStateRoundTripBitExact proves the wire codec preserves every bit,
// including signed zero and extreme magnitudes — so the boundary serialization can
// never be the source of a logits delta. The truncated-buffer case proves it fails
// closed rather than reconstructing garbage.
func TestHiddenStateRoundTripBitExact(t *testing.T) {
	x := [][]float32{
		{0, float32(math.Copysign(0, -1)), 1, -1},
		{math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32, 3.14159},
	}
	buf, err := MarshalHidden(x, 7)
	if err != nil {
		t.Fatalf("MarshalHidden: %v", err)
	}
	got, nextLo, err := UnmarshalHidden(buf)
	if err != nil {
		t.Fatalf("UnmarshalHidden: %v", err)
	}
	if nextLo != 7 {
		t.Fatalf("round-trip nextLo = %d, want 7", nextLo)
	}
	if len(got) != len(x) {
		t.Fatalf("round-trip seq = %d, want %d", len(got), len(x))
	}
	for t2 := range x {
		if len(got[t2]) != len(x[t2]) {
			t.Fatalf("round-trip row %d len = %d, want %d", t2, len(got[t2]), len(x[t2]))
		}
		for i := range x[t2] {
			// Bit-exact: compare the raw IEEE-754 words so -0.0 vs +0.0 is caught.
			if math.Float32bits(got[t2][i]) != math.Float32bits(x[t2][i]) {
				t.Fatalf("round-trip [%d][%d] bits = %#x, want %#x", t2, i, math.Float32bits(got[t2][i]), math.Float32bits(x[t2][i]))
			}
		}
	}
	if _, _, err := UnmarshalHidden(buf[:len(buf)-1]); err == nil {
		t.Fatalf("UnmarshalHidden(truncated) returned nil error; want a fail-closed rejection")
	}
	if _, _, err := UnmarshalHidden([]byte{1, 2, 3}); err == nil {
		t.Fatalf("UnmarshalHidden(short header) returned nil error; want a rejection")
	}
}

// TestPartitionPlanRejectsBadPlans is the validator gate: every malformed tiling
// must fail closed, and valid tilings must pass. The GLM shared-indexer boundary
// case ties the validator to the design decision (no IndexShare group crosses the
// wire) using the existing 2-layer full,shared fixture's config.
func TestPartitionPlanRejectsBadPlans(t *testing.T) {
	dense := Config{NumLayers: 4, ModelType: "llama"}
	_, glmCfg := tinyGLMDsaSafetensorsFixture(t, "BF16", false, true, true, true) // 2 layers: full,shared

	// Positive controls.
	if _, err := NewPartitionPlan(dense, nil); err != nil {
		t.Fatalf("single-stage [0,N) plan rejected: %v", err)
	}
	if _, err := NewPartitionPlan(dense, []int{2}); err != nil {
		t.Fatalf("valid two-stage dense plan rejected: %v", err)
	}

	// A GLM cut on the shared-indexer layer (layer 1) must be rejected.
	if _, err := NewPartitionPlan(glmCfg, []int{1}); err == nil {
		t.Fatalf("GLM cut at shared-indexer layer 1 accepted; want rejection")
	} else if !strings.Contains(err.Error(), "shared-indexer") {
		t.Fatalf("GLM shared-indexer rejection reason = %q, want it to name shared-indexer", err.Error())
	}

	// Hand-built malformed plans Validate must reject.
	bad := []struct {
		name string
		plan PartitionPlan
	}{
		{"gap", PartitionPlan{NumLayers: 4, Stages: []StageSpec{
			{Lo: 0, Hi: 1, First: true}, {Lo: 2, Hi: 4, Last: true}}}},
		{"overlap", PartitionPlan{NumLayers: 4, Stages: []StageSpec{
			{Lo: 0, Hi: 3, First: true}, {Lo: 2, Hi: 4, Last: true}}}},
		{"empty-band", PartitionPlan{NumLayers: 4, Stages: []StageSpec{
			{Lo: 0, Hi: 2, First: true}, {Lo: 2, Hi: 2, Last: true}}}},
		{"hi-out-of-range", PartitionPlan{NumLayers: 4, Stages: []StageSpec{
			{Lo: 0, Hi: 5, First: true, Last: true}}}},
		{"incomplete-coverage", PartitionPlan{NumLayers: 4, Stages: []StageSpec{
			{Lo: 0, Hi: 3, First: true, Last: true}}}},
		{"no-first", PartitionPlan{NumLayers: 4, Stages: []StageSpec{
			{Lo: 0, Hi: 2}, {Lo: 2, Hi: 4, Last: true}}}},
		{"no-last", PartitionPlan{NumLayers: 4, Stages: []StageSpec{
			{Lo: 0, Hi: 2, First: true}, {Lo: 2, Hi: 4}}}},
		{"first-not-first", PartitionPlan{NumLayers: 4, Stages: []StageSpec{
			{Lo: 0, Hi: 2, First: true}, {Lo: 2, Hi: 4, First: true, Last: true}}}},
		{"empty", PartitionPlan{NumLayers: 4}},
	}
	for _, tc := range bad {
		if err := tc.plan.Validate(dense); err == nil {
			t.Errorf("plan %q validated; want a fail-closed rejection", tc.name)
		}
	}
}

// assertNoLayerTensors fails if the store holds any resident matmul weight for any
// of the given (out-of-band) layer indices.
func assertNoLayerTensors(t *testing.T, who string, m *Model, layers ...int) {
	t.Helper()
	for _, l := range layers {
		if hasAnyLayerTensor(m, l) {
			t.Fatalf("%s holds layer-%d resident weights; want none (out of band)", who, l)
		}
	}
}

// hasAnyLayerTensor reports whether the store has any resident matmul weight named
// for the given layer index, across the q8/q4/q4k quant maps the lean loader fills.
func hasAnyLayerTensor(m *Model, layer int) bool {
	prefix := layerPrefix(layer)
	for name := range m.q8w {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	for name := range m.q4w {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	for name := range m.q4kw {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// TestTCPTransportMatchesLocal proves the StageTransport seam works over a REAL
// cross-process wire: the same pipeline run through TCPTransport over a loopback
// socket (with an EchoFrames peer standing in for the next worker's receive) yields
// logits bit-identical (max|Δ|=0) to the in-process LocalTransport. This is the
// structural proof that "the NCCL/RPC wire is a swap underneath the contract" — not
// an aspiration but a working network backend, verifiable on one box.
func TestTCPTransportMatchesLocal(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)
	mkStages := func() []PipelineStage {
		a, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(0, 2))
		if err != nil {
			t.Fatalf("stage A: %v", err)
		}
		b, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(2, 3))
		if err != nil {
			t.Fatalf("stage B: %v", err)
		}
		return []PipelineStage{
			{Spec: StageSpec{Lo: 0, Hi: 2, First: true}, Model: a},
			{Spec: StageSpec{Lo: 2, Hi: 3, Last: true}, Model: b},
		}
	}
	ids := []int{3, 1, 4, 1, 5}

	base, err := RunPipeline(ids, mkStages())
	if err != nil {
		t.Fatalf("RunPipeline (local): %v", err)
	}

	// Loopback peer: accept one connection and echo frames (the identity "next worker").
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		peer, aerr := ln.Accept()
		if aerr != nil {
			return
		}
		defer peer.Close()
		_ = EchoFrames(peer)
	}()

	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	via, err := RunPipelineWith(ids, mkStages(), NewTCPTransport(conn))
	if err != nil {
		t.Fatalf("RunPipelineWith (TCP): %v", err)
	}
	conn.Close()
	wg.Wait()

	if len(via) != len(base) {
		t.Fatalf("TCP-routed logits seq = %d, local = %d", len(via), len(base))
	}
	var maxAbs float32
	for t2 := range via {
		if len(via[t2]) != len(base[t2]) {
			t.Fatalf("TCP logits[%d] len = %d, local = %d", t2, len(via[t2]), len(base[t2]))
		}
		for i := range via[t2] {
			d := via[t2][i] - base[t2][i]
			if d < 0 {
				d = -d
			}
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	if maxAbs != 0 {
		t.Fatalf("TCP transport changed logits: max|delta|=%.3e (want bit-exact 0)", maxAbs)
	}
}

// TestFrameRoundTripBitExact proves the wire framing (writeFrame/readFrame over a
// loopback pipe) preserves an arbitrary payload exactly, including the empty frame.
func TestFrameRoundTripBitExact(t *testing.T) {
	c1, c2 := net.Pipe()
	defer c1.Close()
	defer c2.Close()
	payloads := [][]byte{{}, {0}, {1, 2, 3, 4, 5}, make([]byte, 4096)}
	for i := range make([]byte, 4096) {
		payloads[3][i] = byte(i % 251)
	}
	done := make(chan error, 1)
	go func() {
		for _, p := range payloads {
			if err := writeFrame(c1, p); err != nil {
				done <- err
				return
			}
		}
		done <- nil
	}()
	for _, want := range payloads {
		got, err := readFrame(c2)
		if err != nil {
			t.Fatalf("readFrame: %v", err)
		}
		if len(got) != len(want) {
			t.Fatalf("frame len = %d, want %d", len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("frame byte %d = %d, want %d", i, got[i], want[i])
			}
		}
	}
	if err := <-done; err != nil {
		t.Fatalf("writeFrame: %v", err)
	}
}
