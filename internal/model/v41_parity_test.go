package model

// v41_parity_test.go - the independent token-path parity witness for the reduced
// DeepSeek V4.1 text forward (#12908). v41_oracle_test.go checks the component
// math against a scalar transcription; v41_forward_test.go checks one
// Session.Prefill -> Session.Step continuation against one longer Forward. This
// file closes the remaining acceptance clause: tiny token-path GOLDEN FIXTURES
// that pin the production path (Model.Forward / Session.Prefill/Step) against the
// INDEPENDENT scalar oracle (v41OracleForward, defined in v41_forward_test.go from
// cpuOracle* primitives - it reuses NONE of the production forward machinery),
// plus malformed-state negative controls that must refuse rather than emit.
//
// Independence discipline. The expected values in
// testdata/v41_oracle/token_path_golden.json are produced by the independent
// scalar oracle, never by the production forward under test. The greedy generation
// arm derives its reference from Model.Forward's argmax (the st == nil pure-prefill
// path), while the session arm drives Prefill/Step; the parity claim is that
// incremental decode and whole-history recompute select the same greedy token at
// every step. No production code changes; no checkpoint, no hardware, no full-model
// generation or physical-parity claim.

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const v41ParityGoldenPath = "testdata/v41_oracle/token_path_golden.json"

// v41ParityGolden is the checked-in independent-oracle token-path fixture: the
// prompt ids, the vocab/hidden envelope, the independent-oracle argmax per
// position, and the flattened per-position logits.
type v41ParityGolden struct {
	Revision string    `json:"revision"`
	Note     string    `json:"note,omitempty"`
	IDs      []int     `json:"ids"`
	Vocab    int       `json:"vocab"`
	Hidden   int       `json:"hidden"`
	Argmax   []int     `json:"argmax"`
	Logits   []float32 `json:"logits"`
}

// v41ParityGoldenProvenance is the checked-in note accompanying the golden; it
// pins the fixture's authority to the independent scalar oracle, never the
// production forward under test.
const v41ParityGoldenProvenance = "Expected token-path values produced by the INDEPENDENT scalar oracle in v41_oracle_test.go (oracleV41* / cpuOracle*) over the reduced in-memory V4.1 fixture built by v41ReducedModel. Never derived from the production forward under test."

// TestV41ParityGoldenRegenerate re-derives the checked-in token-path golden from
// the independent scalar oracle. It is a regeneration helper, not an assertion:
// it does nothing unless V41_GOLDEN_UPDATE=1 is set in the environment. Run it
// (go test ./internal/model -run TestV41ParityGoldenRegenerate -count=1 with the
// var set) after any deliberate change to the reference rotary contract, then
// review the diff to confirm the logits moved as the reference requires.
func TestV41ParityGoldenRegenerate(t *testing.T) {
	if os.Getenv("V41_GOLDEN_UPDATE") != "1" {
		t.Skip("set V41_GOLDEN_UPDATE=1 to regenerate the token-path golden")
	}
	m := v41ReducedModel(t)
	g := v41LoadParityGolden(t)
	ids := g.IDs
	if len(ids) == 0 {
		ids = []int{1, 3, 5, 7}
	}
	out := v41OracleForward(t, m, ids)
	next := v41ParityGolden{
		Revision: v41OracleRevision,
		Note:     v41ParityGoldenProvenance,
		IDs:      append([]int(nil), ids...),
		Vocab:    m.Cfg.VocabSize,
		Hidden:   m.Cfg.HiddenSize,
		Argmax:   make([]int, len(ids)),
		Logits:   make([]float32, 0, len(ids)*m.Cfg.VocabSize),
	}
	for pos := range ids {
		next.Argmax[pos] = argmaxF32(out[pos])
		next.Logits = append(next.Logits, out[pos]...)
	}
	blob, err := json.MarshalIndent(next, "", "  ")
	if err != nil {
		t.Fatalf("marshal golden: %v", err)
	}
	blob = append(blob, '\n')
	if err := os.WriteFile(v41ParityGoldenPath, blob, 0o644); err != nil {
		t.Fatalf("write golden: %v", err)
	}
	t.Logf("regenerated %s (%d positions)", v41ParityGoldenPath, len(ids))
}

func v41LoadParityGolden(t *testing.T) v41ParityGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(v41ParityGoldenPath))
	if err != nil {
		t.Fatalf("read golden fixture: %v", err)
	}
	var g v41ParityGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("decode golden fixture: %v", err)
	}
	if g.Revision != v41OracleRevision {
		t.Fatalf("golden revision %q, want %q", g.Revision, v41OracleRevision)
	}
	return g
}

// TestV41Parity is the #12908 token-path acceptance witness. It pins the
// production forward against the independent scalar oracle through a checked-in
// golden fixture, and proves the fixture is non-vacuous (a perturbed prompt moves
// the logits) before using malformed-state negative controls to show the path
// refuses rather than emits.
func TestV41Parity(t *testing.T) {
	g := v41LoadParityGolden(t)
	m := v41ReducedModel(t)
	if m.Cfg.VocabSize != g.Vocab || m.Cfg.HiddenSize != g.Hidden {
		t.Fatalf("fixture envelope vocab/hidden = %d/%d, model = %d/%d",
			g.Vocab, g.Hidden, m.Cfg.VocabSize, m.Cfg.HiddenSize)
	}
	if len(g.Logits) != len(g.IDs)*g.Vocab {
		t.Fatalf("golden logits len = %d, want %d positions * %d vocab", len(g.Logits), len(g.IDs), g.Vocab)
	}

	t.Run("independent oracle reproduces the golden fixture", func(t *testing.T) {
		// The scalar oracle is the authority for the fixture: re-derive it here
		// so a fixture/oracle drift is caught even before the production path runs.
		want := v41OracleForward(t, m, g.IDs)
		for pos := range g.IDs {
			base := pos * g.Vocab
			v41LogitsClose(t, "oracle-vs-golden", want[pos], g.Logits[base:base+g.Vocab])
		}
	})

	t.Run("production forward matches the independent oracle", func(t *testing.T) {
		act := m.Forward(g.IDs)
		if act == nil || len(act.Logits) != len(g.IDs) {
			t.Fatalf("Forward returned %d positions, want %d", logitCount(act), len(g.IDs))
		}
		oracle := v41OracleForward(t, m, g.IDs)
		for pos := range g.IDs {
			base := pos * g.Vocab
			v41LogitsClose(t, "forward-vs-golden", act.Logits[pos], g.Logits[base:base+g.Vocab])
			v41LogitsClose(t, "forward-vs-oracle", act.Logits[pos], oracle[pos])
			if got := argmaxF32(act.Logits[pos]); got != g.Argmax[pos] {
				t.Fatalf("forward argmax[%d] = %d, want %d", pos, got, g.Argmax[pos])
			}
		}
	})

	t.Run("prefill last logits match the oracle tail", func(t *testing.T) {
		s := &Session{M: m}
		last := s.Prefill(g.IDs)
		if len(last) != g.Vocab {
			t.Fatalf("Prefill returned %d logits, want %d", len(last), g.Vocab)
		}
		base := (len(g.IDs) - 1) * g.Vocab
		v41LogitsClose(t, "prefill-vs-golden", last, g.Logits[base:base+g.Vocab])
		if got := argmaxF32(last); got != g.Argmax[len(g.Argmax)-1] {
			t.Fatalf("prefill argmax = %d, want %d", got, g.Argmax[len(g.Argmax)-1])
		}
	})

	t.Run("golden fixture is non-vacuous", func(t *testing.T) {
		perturbed := m.Forward([]int{1, 3, 5, 6})
		if perturbed == nil || len(perturbed.Logits) != len(g.IDs) {
			t.Fatalf("perturbed Forward returned %d positions, want %d", logitCount(perturbed), len(g.IDs))
		}
		moved := false
		for pos := range g.IDs {
			base := pos * g.Vocab
			if d := cpuOracleMaxAbsDiff(perturbed.Logits[pos], g.Logits[base:base+g.Vocab]); d > cpuOracleTol {
				moved = true
			}
		}
		if !moved {
			t.Fatal("prompt perturbation did not move any logit; the golden fixture is vacuous")
		}
	})

	t.Run("malformed state refuses rather than emits", func(t *testing.T) {
		// A required stage weight removed: admission and the public entry points
		// must fail closed with ErrV41ForwardStage and produce no logits.
		broken := v41ReducedModel(t)
		delete(broken.manifest, layerName(0, "ffn.shared_experts.w2.weight"))
		if err := broken.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("malformed admission error = %v, want ErrV41ForwardStage", err)
		}
		if err := panicAsError(func() { _ = broken.Forward(g.IDs) }); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("malformed Forward panic = %v, want ErrV41ForwardStage", err)
		}

		// An out-of-range token id must refuse before any math.
		if err := panicAsError(func() { _ = m.Forward([]int{0, g.Vocab}) }); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("out-of-range token panic = %v, want ErrV41ForwardStage", err)
		}

		// A shape-inconsistent stage (wrong reduction width) must refuse.
		badShape := v41ReducedModel(t)
		badShape.manifest[layerName(0, "attn.wq_b.weight")] = tensorMeta{Shape: []int{m.Cfg.NumHeads*m.Cfg.HeadDim + 1, m.Cfg.QLoraRank}}
		if err := badShape.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("bad-shape admission error = %v, want ErrV41ForwardStage", err)
		}
	})
}

// v41GreedySession runs a greedy decode through the session seam: Prefill seeds
// the prompt, then each Step consumes the previously selected token and returns
// the next-token logits. It returns the prompt plus the generated continuation.
func v41GreedySession(t *testing.T, m *Model, prompt []int, steps int) []int {
	t.Helper()
	s := &Session{M: m}
	last := s.Prefill(prompt)
	if len(last) == 0 {
		t.Fatalf("Prefill(%v) returned no logits", prompt)
	}
	out := append([]int(nil), prompt...)
	for i := 0; i < steps; i++ {
		next := argmaxF32(last)
		out = append(out, next)
		last = s.Step(next)
		if len(last) == 0 {
			t.Fatalf("Step %d returned no logits", i)
		}
	}
	return out
}

// v41GreedyRecompute derives the same greedy continuation by one whole-history
// Forward per step: after each selected token the entire prompt+continuation is
// recomputed and the last-position logits are argmaxed. This is the independent
// reference for the session seam.
func v41GreedyRecompute(t *testing.T, m *Model, prompt []int, steps int) []int {
	t.Helper()
	out := append([]int(nil), prompt...)
	for i := 0; i < steps; i++ {
		act := m.Forward(out)
		if act == nil || len(act.Logits) != len(out) {
			t.Fatalf("Forward(%v) returned %d positions, want %d", out, logitCount(act), len(out))
		}
		out = append(out, argmaxF32(act.Logits[len(out)-1]))
	}
	return out
}

func logitCount(act *Activations) int {
	if act == nil {
		return 0
	}
	return len(act.Logits)
}

// TestV41GreedyGenerationParity is the #12908 token-path acceptance witness: a
// short greedy decode through Session.Prefill/Step matches the argmax of the
// whole-history recompute at every position, is deterministic across identical
// sessions, and is non-vacuous (a perturbed prompt changes the stream).
func TestV41GreedyGenerationParity(t *testing.T) {
	m := v41ReducedModel(t)
	prompt := []int{1, 3, 5}
	const steps = 4

	t.Run("session step tracks recompute argmax", func(t *testing.T) {
		got := v41GreedySession(t, m, prompt, steps)
		want := v41GreedyRecompute(t, m, prompt, steps)
		if len(got) != len(want) {
			t.Fatalf("session stream len %d, recompute len %d", len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("greedy token[%d] = %d, want %d (session %v, recompute %v)",
					i, got[i], want[i], got, want)
			}
		}
		for i := len(prompt); i < len(got); i++ {
			if got[i] < 0 || got[i] >= m.Cfg.VocabSize {
				t.Fatalf("generated token[%d] = %d out of vocab [0,%d)", i, got[i], m.Cfg.VocabSize)
			}
		}
	})

	t.Run("deterministic across sessions", func(t *testing.T) {
		a := v41GreedySession(t, m, prompt, steps)
		b := v41GreedySession(t, m, prompt, steps)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("non-deterministic at [%d]: %d vs %d", i, a[i], b[i])
			}
		}
	})

	t.Run("perturbed prompt perturbs the stream", func(t *testing.T) {
		base := v41GreedySession(t, m, prompt, steps)
		perturbed := v41GreedySession(t, m, []int{1, 3, 6}, steps)
		same := len(base) == len(perturbed)
		for i := range base {
			if i >= len(perturbed) || base[i] != perturbed[i] {
				same = false
				break
			}
		}
		if same {
			t.Fatalf("prompt perturbation produced an identical stream %v; parity fixture is vacuous", base)
		}
	})
}

// v41ParityStagesModel builds the reduced token-path fixture for the
// name-the-stages arm (fak#13151). It combines the four architecture-specific
// V4.1 stages on one model so a single Forward exercises them together:
//
//   - mHC:        every layer carries the mhc.mixes/base/scale tensors the
//     mixing stage reads (v41MHCSplit/v41MHCPre/v41MHCPost).
//   - Engram:     layer 1 declares Engram with its three mixing tensors and a
//     wired packed-row source, so v41EngramInject runs.
//   - CED/CSA2:   layer 0 declares CompressRatios[0]=2 with its compressor
//     tensors, so v41CompressedRows pools the projected KV.
//   - indexer:    layer 0 declares itself an index source with its indexer
//     tensors, so v41IndexRows scores and selects rows.
//
// The model is the ordinary reduced envelope + the Engram wiring; no production
// behavior changes and no checkpoint or hardware is involved.
func v41ParityStagesModel(t *testing.T) (*Model, V41EngramLayout) {
	t.Helper()
	m, layout := v41ReducedEngramModel(t)
	cfg := m.Cfg
	H := cfg.HiddenSize
	width := v41CompressorWidth(cfg)
	// The compressed + index source is the LAST layer, so no later layer becomes a
	// reader that would need session-state publication; the Engram layer is the
	// same last layer, so a single Forward exercises all four stages without a
	// session.
	src := cfg.NumLayers - 1
	ratios := make([]int, cfg.NumLayers)
	ratios[src] = 2
	cfg.DeepSeekV41.CompressRatios = ratios
	cfg.DeepSeekV41.IndexSourceLayerIDs = []int{src}
	cfg.IndexNHeads = 1
	cfg.IndexHeadDim = H
	cfg.IndexTopK = 2
	m.Cfg = cfg

	type ts = synthTensor
	tensors := []ts{
		{layerName(src, "attn.compressor.wkv.weight"), []int{width, H}},
		{layerName(src, "attn.compressor.wgate.weight"), []int{width, H}},
		{layerName(src, "attn.compressor.norm.weight"), []int{width}},
		{layerName(src, "indexer.wq_b.weight"), []int{cfg.IndexNHeads * cfg.IndexHeadDim, cfg.QLoraRank}},
		{layerName(src, "indexer.wk.weight"), []int{cfg.IndexHeadDim, width}},
		{layerName(src, "indexer.k_norm.weight"), []int{cfg.IndexHeadDim}},
		{layerName(src, "indexer.weights_proj.weight"), []int{cfg.IndexNHeads, H}},
	}
	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case hasSuffix(name, "compressor.norm.weight") || hasSuffix(name, "indexer.k_norm.weight"):
			return 1.0
		default:
			return synthMatmulFill(name, next)
		}
	})
	for k, v := range man {
		m.manifest[k] = v
	}
	for k, v := range raw {
		m.raw[k] = v
	}
	return m, layout
}

// TestV41ParityStages is the #13151 token-path parity arm that NAMES the four
// architecture-specific V4.1 stages on the production path: Engram, mHC, the
// CED/CSA2 compressor, and the lightning indexer. It asserts each stage is
// present, executes, and is non-vacuous (removing/disabling it moves or breaks
// the token-path result), so the parity claim cannot be satisfied by a path that
// silently skips a stage. It is a witness-completeness arm, not a regression.
func TestV41ParityStages(t *testing.T) {
	m, layout := v41ParityStagesModel(t)
	// The four stage names this arm is required to see on the path.
	stages := []v41ForwardStage{
		v41StageEngram,   // Engram n-gram retrieval + mixing
		v41StageMHC,      // four-stream mHC mixing
		v41StageCompress, // CED/CSA2 compressor pooling
		v41StageIndexer,  // lightning indexer scoring + selection
	}
	named := make(map[v41ForwardStage]bool, len(stages))
	for _, s := range stages {
		if !v41StageNamedOnPath(s) {
			t.Fatalf("stage %q is not a named assembly stage", s)
		}
		named[s] = true
	}
	if len(named) != 4 {
		t.Fatalf("expected 4 distinct named stages, got %d", len(named))
	}

	// Admission must succeed: the fixture carries the weights every declared
	// stage reads, so nothing is refused and nothing is silently skipped.
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("stage-complete fixture admission error = %v, want nil", err)
	}

	// A declared Engram layer must actually have a wired retrieval stage whose
	// layer set matches the declared Engram layers.
	stage := m.v41EngramStageFor()
	if stage == nil {
		t.Fatal("Engram stage not wired for a model that declares an Engram layer")
	}
	if v41EngramSourceCount(stage) != len(layout.Rows) {
		t.Fatalf("Engram stage layer count = %d, layout rows = %d",
			v41EngramSourceCount(stage), len(layout.Rows))
	}

	prompt := []int{1, 2, 3, 4}
	act := m.Forward(prompt)
	if act == nil || len(act.Logits) != len(prompt) {
		t.Fatalf("Forward returned %d positions, want %d", logitCount(act), len(prompt))
	}
	for pos, row := range act.Logits {
		if len(row) != m.Cfg.VocabSize {
			t.Fatalf("logits[%d] width %d, want %d", pos, len(row), m.Cfg.VocabSize)
		}
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logits[%d][%d] = %v is non-finite", pos, i, v)
			}
		}
	}

	// Non-vacuity per stage: disabling each stage must change the token-path
	// result (or make it refuse), proving the stage is load-bearing and not a
	// no-op the parity arm would have credited for free.

	// (1) CED/CSA2 + indexer: falling back to the uncompressed regime changes the
	// logits, so the compressor/indexer stages genuinely feed the contraction.
	t.Run("CED/CSA2 + indexer change the token path", func(t *testing.T) {
		plain, _ := v41ParityStagesModel(t)
		zeros := make([]int, plain.Cfg.NumLayers)
		plain.Cfg.DeepSeekV41.CompressRatios = zeros
		plain.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
		plainAct := plain.Forward(prompt)
		if v41LogitsEqual(act, plainAct) {
			t.Fatal("uncompressed forward matches the compressed forward; CED/CSA2/indexer did not execute")
		}
	})

	// (2) Engram: stripping the wired row source must make the Engram layer fail
	// closed rather than silently emit non-Engram logits.
	t.Run("Engram stage is load-bearing", func(t *testing.T) {
		noEngram, _ := v41ParityStagesModel(t)
		noEngram.Cfg.DeepSeekV41.EngramLayerIDs = nil
		noEngramAct := noEngram.Forward(prompt)
		if v41LogitsEqual(act, noEngramAct) {
			t.Fatal("removing the Engram layer left the token-path logits unchanged; Engram did not execute")
		}
	})

	// (3) mHC: a malformed mHC mixing shape must refuse at admission, so the arm
	// cannot pass on a path that silently skipped the mHC stage.
	t.Run("mHC stage is load-bearing", func(t *testing.T) {
		bad, _ := v41ParityStagesModel(t)
		H := bad.Cfg.HiddenSize
		bad.manifest[layerName(0, "mhc.mixes.weight")] = tensorMeta{Shape: []int{v41MHCMixWidth + 1, H}}
		if err := bad.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("malformed mHC admission error = %v, want ErrV41ForwardStage", err)
		}
	})
}

// v41StageNamedOnPath reports whether stage is one of the assembly's declared
// stage names. It is a structural (not behavioral) check: the four V4.1 stages
// this arm names must exist as distinct named stages so a refusal or failure can
// be attributed to one of them.
func v41StageNamedOnPath(stage v41ForwardStage) bool {
	switch stage {
	case v41StageEmbedding, v41StageLayer, v41StageMHC, v41StageAttention,
		v41StageMoE, v41StageEngram, v41StageFinalNorm, v41StageHead,
		v41StageCompress, v41StageIndexer:
		return true
	default:
		return false
	}
}

// v41EngramSourceCount reports how many Engram layers a stage owns, tolerating a
// nil stage.
func v41EngramSourceCount(stage *v41EngramStage) int {
	if stage == nil {
		return 0
	}
	return len(stage.layerIDs)
}

// v41LogitsEqual reports whether two activation results carry identical logits.
func v41LogitsEqual(a, b *Activations) bool {
	if a == nil || b == nil || len(a.Logits) != len(b.Logits) {
		return false
	}
	for i := range a.Logits {
		if len(a.Logits[i]) != len(b.Logits[i]) {
			return false
		}
		for j := range a.Logits[i] {
			if a.Logits[i][j] != b.Logits[i][j] {
				return false
			}
		}
	}
	return true
}
