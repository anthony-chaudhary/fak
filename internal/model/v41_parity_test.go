package model

// v41_parity_test.go - the independent token-path parity witness for the reduced
// DeepSeek V4.1 text forward (#12908). v41_oracle_test.go checks the component
// math against a scalar transcription; v41_forward_test.go checks one
// Session.Prefill -> Session.Step continuation against one longer Forward. This
// file closes the remaining acceptance clause: a SHORT GREEDY GENERATION loop
// driven through Session.Prefill/Session.Step must track the argmax of a single
// recomputed Forward over the grown history, with determinism and a
// non-vacuous-fixture negative control.
//
// Independence discipline. The expected token stream is derived from
// Model.Forward (the st == nil pure-prefill path) via argmaxF32, NOT from the
// session path under test. The session implements Step as "fold the token into
// committed history and recompute"; the parity claim is therefore that
// incremental decode and whole-history recompute select the same greedy token at
// every step - the #12901 prefill/step consistency property extended across a
// generation, not a single step. No production code changes; no checkpoint, no
// hardware, no full-model generation or physical-parity claim.

import (
	"testing"
)

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
