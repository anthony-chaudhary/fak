package modelengine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestInBatchPrefixDedupReachable is the PRODUCTION reachability witness for fak#1914's
// in-batch cold-prefix dedup. Every mechanism test in this package builds its scheduler
// with the test-only newNativeScheduler constructor and calls SetInBatchPrefixDedup
// itself, so the whole suite stayed green while the feature was unreachable from any
// real engine: Engine.nativeScheduler — the sole production construction — never called
// the setter, so inBatchDedup was false in every serving process and prefillCoalesced
// was dead code on the live path.
//
// Part 1 pins the switch: the production constructor must arm the coalescing path from
// FAK_NATIVE_IN_BATCH_PREFIX_DEDUP or an explicit SetInBatchPrefixDedup, and must leave
// it disarmed by default so the historical per-lane prefill stays byte-for-byte. Part 2
// proves arming is not cosmetic: on the production-constructed scheduler a burst of two
// identical concurrent cold admissions runs exactly ONE shared prefill and the twin
// adopts it, so the coalescing branch actually executes on the serving seam.
//
// Both parts drive New + Preload + nativeScheduler rather than newNativeScheduler: a
// regression that un-wires the setter from the production constructor reds here even
// while every mechanism test still passes.
//
// fak-test:runtime fast est=1200ms
func TestInBatchPrefixDedupReachable(t *testing.T) {
	t.Run("default_is_disarmed", func(t *testing.T) {
		// An unset switch must leave the feature off. This is the back-compat
		// guarantee fak#13661 acceptance item 3 pins for the synchronous path.
		t.Setenv("FAK_NATIVE_IN_BATCH_PREFIX_DEDUP", "")
		e := New()
		e.Preload(model.NewSynthetic(SyntheticConfig()))
		if e.InBatchPrefixDedupArmed() {
			t.Fatal("production engine armed in-batch prefix dedup with the switch unset")
		}
	})

	t.Run("env_arms_production_constructor", func(t *testing.T) {
		t.Setenv("FAK_NATIVE_IN_BATCH_PREFIX_DEDUP", "1")
		e := New()
		e.Preload(model.NewSynthetic(SyntheticConfig()))
		if !e.InBatchPrefixDedupArmed() {
			t.Fatal("FAK_NATIVE_IN_BATCH_PREFIX_DEDUP=1 did not arm the production scheduler; the coalescing branch is unreachable from a serving engine")
		}
	})

	t.Run("explicit_setter_arms_production_constructor", func(t *testing.T) {
		t.Setenv("FAK_NATIVE_IN_BATCH_PREFIX_DEDUP", "")
		e := New()
		e.Preload(model.NewSynthetic(SyntheticConfig()))
		e.SetInBatchPrefixDedup(true)
		if !e.InBatchPrefixDedupArmed() {
			t.Fatal("SetInBatchPrefixDedup(true) did not arm the production scheduler")
		}
	})

	t.Run("unrecognized_env_value_stays_off", func(t *testing.T) {
		// A typo must degrade to today's behaviour, never silently widen what a
		// serving process coalesces.
		for _, raw := range []string{"0", "off", "no", "maybe", "1.5"} {
			t.Setenv("FAK_NATIVE_IN_BATCH_PREFIX_DEDUP", raw)
			e := New()
			e.Preload(model.NewSynthetic(SyntheticConfig()))
			if e.InBatchPrefixDedupArmed() {
				t.Fatalf("FAK_NATIVE_IN_BATCH_PREFIX_DEDUP=%q armed the coalescing path; want default-off", raw)
			}
		}
	})

	t.Run("production_scheduler_coalesces_a_twin", func(t *testing.T) {
		t.Setenv("FAK_NATIVE_IN_BATCH_PREFIX_DEDUP", "1")
		prompt := nativeSchedulerQwenPrompt(40) // >= nativeInBatchPrefixDedupMinShared (32)
		e := New()
		e.Preload(model.NewSynthetic(SyntheticConfig()))
		s := e.nativeScheduler()
		if !s.InBatchPrefixDedupArmed() {
			t.Fatal("production scheduler not armed")
		}
		// A generic synthetic checkpoint is not a resident Q4_K Qwen lane, so its
		// admission stays on the synchronous branch the coalesced path gates on. The
		// resident chunked lane is the separate fak#13661 gap.
		if s.qwenPrefillCap != nil {
			t.Fatalf("fixture minted a resident chunked prefill capability; this arm must exercise the synchronous branch")
		}
		// Keep the production scheduler and swap only the byte-level prepare hook for
		// the exact prompt under test, as the production-arming parity test does.
		s.prepare = prefixDedupPlainPrepare(map[string][]int{"a": prompt, "b": prompt})
		nativeSchedulerBeginManualDrain(t, s)
		defer nativeSchedulerEndManualDrain(s)

		lanes, leaderPrefills := prefixDedupAdmitPairGated(t, s, "a", "b")
		if leaderPrefills != 1 {
			t.Fatalf("two identical concurrent cold admissions ran %d shared leader prefills, want exactly 1", leaderPrefills)
		}
		stats := s.InBatchPrefixDedupStats()
		if stats.Leaders != 1 || stats.Followers != 1 {
			t.Fatalf("coalescing stats leaders=%d followers=%d, want 1/1", stats.Leaders, stats.Followers)
		}
		if stats.CoalescedPrefills != 1 || stats.ExactReuses != 1 {
			t.Fatalf("coalescing stats coalesced=%d exact=%d, want 1/1", stats.CoalescedPrefills, stats.ExactReuses)
		}
		// The twin must adopt the leader's KV and receive the identical last-token
		// logits, i.e. fusion stayed behavior-preserving.
		if len(lanes[0].logits) != len(lanes[1].logits) {
			t.Fatalf("leader logits=%d twin logits=%d, want equal length", len(lanes[0].logits), len(lanes[1].logits))
		}
		for i := range lanes[0].logits {
			if lanes[0].logits[i] != lanes[1].logits[i] {
				t.Fatalf("twin logit %d = %v, leader = %v; coalescing must be bit-identical", i, lanes[1].logits[i], lanes[0].logits[i])
			}
		}
	})
}
