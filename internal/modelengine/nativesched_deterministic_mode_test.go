package modelengine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// TestNativeSchedulerDeterministicModeSkipsRadixInsert pins the SGLang
// deterministic-serving gate (radix_cache.py:483-484, flag disable_finished_insert,
// Apache-2.0): with deterministic mode on, identical prompts must both run a cold
// prefill and leave the radix tree empty; with it off, the second prompt records a
// hit. Adapted, not copied.
func TestNativeSchedulerDeterministicModeSkipsRadixInsert(t *testing.T) {
	run := func(deterministic bool) int {
		m := nativeSchedulerPrefillModel(t)
		tree := radixkv.New(0)
		s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{}))
		s.SetRadixKV(tree)
		s.SetDeterministicMode(deterministic)

		prompt := nativeSchedulerQwenPrompt(24)
		sess := m.NewSession()
		sess.Quant = true
		sess.Q4K = true
		defer sess.Close()
		logits := sess.Prefill(prompt)

		// First prompt: miss, then attempt to insert the finished prompt.
		if matched, b, owner := s.lookupPrefix(prompt); matched != 0 {
			t.Fatalf("first lookup matched=%d, want 0 (cold)", matched)
		} else if owner != nil {
			owner.Done(b)
		}
		s.maybeInsertRadixKV(prompt, sess, logits)

		// Second identical prompt: hit only when reuse is not disabled.
		matched, b, owner := s.lookupPrefix(prompt)
		if owner != nil {
			owner.Done(b)
		}
		return matched
	}

	if got := run(true); got != 0 {
		t.Fatalf("deterministic mode: second lookup matched=%d, want 0 (tree must stay empty)", got)
	}
	if got := run(false); got != 24 {
		t.Fatalf("reuse mode: second lookup matched=%d, want full 24-token hit", got)
	}
}
