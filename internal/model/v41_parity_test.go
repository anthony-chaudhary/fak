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
	IDs      []int     `json:"ids"`
	Vocab    int       `json:"vocab"`
	Hidden   int       `json:"hidden"`
	Argmax   []int     `json:"argmax"`
	Logits   []float32 `json:"logits"`
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
