package radixkv

import (
	"errors"
	"testing"
)

// TestSessionDecodeRetention is the oss-port-trtllm-cross-turn-gen-reuse-20261106 witness
// for the ADAPTED TensorRT-LLM KVCacheManagerV2 cross-turn generation-block reuse
// (PRs #19710 + #20086, tag v1.4.0rc2 @ b96ba434e8a05e2ab52d2082bd8d70b1d844739b,
// Apache-2.0): a finished decode lane's generated suffix is retained under a session key
// with priority/TTL and a bounded byte budget, so the session's next turn prefix-extends
// into the prior-turn KV instead of re-prefilling it. It pins the closed contract:
//
//   - (a) finished-lane blocks are retained under the session key and hit on next-turn lookup;
//   - (b) TTL / byte-budget eviction reclaims retained blocks;
//   - (c) default-off reproduces the exact baseline free behavior;
//   - sentinel errors and config validation fail closed.
//
// All evidence here is software (in-memory tree, logical clock); no hardware or
// whole-model throughput claim is made.
func TestSessionDecodeRetention(t *testing.T) {
	const (
		promptLen = 16
		genLen    = 16
		budget    = 48 // tokens: holds one 32-token transcript, forces eviction on a 40-token filler
	)
	turn1 := func() []int { return cat(seqTokens(1, promptLen), seqTokens(2, genLen)) }
	filler := func() []int { return seqTokens(9, 40) }

	// walkSuffix finds the deepest node on the full transcript path (read-only: no clock,
	// no lease, no recency mutation) so tier assertions do not disturb LRU ordering.
	walkSuffix := func(tr *Tree, tokens []int) (*node, int) {
		n, nlen, _, _ := tr.walk(tr.rootFor(""), tokens)
		return n, nlen
	}

	t.Run("retained_suffix_hits_next_turn", func(t *testing.T) {
		tr := New(budget)
		sr, err := NewSessionRetention(tr, SessionRetentionConfig{
			Enabled: true, Priority: DefaultSessionRetentionPriority,
			TTL: DefaultSessionRetentionTTL, ByteBudget: 64 << 20,
		})
		if err != nil {
			t.Fatal(err)
		}

		// Turn 1: lane finishes; its full transcript is cached and the lease released.
		insertUnleased(tr, turn1())
		if err := sr.Retain("sess-a", "", turn1(), promptLen); err != nil {
			t.Fatalf("Retain: %v", err)
		}
		if !sr.Has("sess-a") || sr.Sessions() != 1 {
			t.Fatalf("registry state: Has=%v Sessions=%d, want sess-a recorded", sr.Has("sess-a"), sr.Sessions())
		}
		if got := sr.RetainedBytes(); got != int64(genLen*sessionRetentionBytesPerToken) {
			t.Fatalf("RetainedBytes=%d, want %d (generated suffix only)", got, genLen*sessionRetentionBytesPerToken)
		}
		// The retained suffix node sits in the Tier 0 protection segment while live.
		child, nlen := walkSuffix(tr, turn1())
		if nlen != promptLen+genLen {
			t.Fatalf("precondition: full transcript resident, matched %d/%d", nlen, promptLen+genLen)
		}
		if tier := tr.NodeTier(child); tier != Tier0PinnedRoot {
			t.Fatalf("retained suffix NodeTier=%v, want Tier0PinnedRoot while the window is live", tier)
		}

		// Pressure: a newer 40-token filler pushes the tree past budget. The retained
		// suffix is immune, so the filler is evicted instead of the prior-turn KV.
		insertUnleased(tr, filler())

		// Next turn of the same session: prompt + prior assistant output + new input.
		nextTurn := cat(turn1(), seqTokens(3, 8))
		if m := tr.MatchLen(nextTurn); m != promptLen+genLen {
			t.Fatalf("next-turn lookup hit %d/%d, want the full prior-turn path retained", m, promptLen+genLen)
		}

		// Baseline contrast: the identical sequence WITHOUT retention loses the
		// oldest leaf (the finished lane's blocks) to the same pressure.
		ctl := New(budget)
		insertUnleased(ctl, turn1())
		insertUnleased(ctl, filler())
		if m := ctl.MatchLen(turn1()); m != 0 {
			t.Fatalf("control tree kept %d/%d of the unretained transcript, want 0 (baseline free behavior)", m, promptLen+genLen)
		}
	})

	t.Run("ttl_expiry_reclaims_retained_blocks", func(t *testing.T) {
		const ttl = 8
		tr := New(budget)
		sr, err := NewSessionRetention(tr, SessionRetentionConfig{
			Enabled: true, Priority: DefaultSessionRetentionPriority,
			TTL: ttl, ByteBudget: 64 << 20,
		})
		if err != nil {
			t.Fatal(err)
		}
		insertUnleased(tr, turn1())
		if err := sr.Retain("sess-b", "", turn1(), promptLen); err != nil {
			t.Fatal(err)
		}
		child, _ := walkSuffix(tr, turn1())
		if tier := tr.NodeTier(child); tier != Tier0PinnedRoot {
			t.Fatalf("live window: NodeTier=%v, want Tier0PinnedRoot", tier)
		}

		// Advance the LOGICAL clock strictly past the window.
		for i := 0; i < int(ttl)+4; i++ {
			tr.tick()
		}
		// Expired retention auto-demotes the suffix to probationary (evict-first) —
		// the closed TTL contract, asserted through the exported tier reader.
		if tier := tr.NodeTier(child); tier != Tier3Probationary {
			t.Fatalf("past TTL: NodeTier=%v, want Tier3Probationary", tier)
		}

		// Under pressure the expired blocks are reclaimed like any cold leaf.
		insertUnleased(tr, filler())
		if m := tr.MatchLen(turn1()); m != 0 {
			t.Fatalf("expired retained blocks survived pressure: matched %d/%d, want 0", m, promptLen+genLen)
		}
	})

	t.Run("reclaim_sweeps_expired_entries", func(t *testing.T) {
		const ttl = 8
		tr := New(0) // unbounded: isolate the registry sweep from tree eviction
		sr, err := NewSessionRetention(tr, SessionRetentionConfig{
			Enabled: true, Priority: DefaultSessionRetentionPriority,
			TTL: ttl, ByteBudget: 0,
		})
		if err != nil {
			t.Fatal(err)
		}
		insertUnleased(tr, turn1())
		if err := sr.Retain("sess-c", "", turn1(), promptLen); err != nil {
			t.Fatal(err)
		}
		// Inside the window: nothing is reclaimable.
		if freed := sr.Reclaim(tr.clock); freed != 0 {
			t.Fatalf("Reclaim inside window freed %d bytes, want 0", freed)
		}
		for i := 0; i < int(ttl)+4; i++ {
			tr.tick()
		}
		want := int64(genLen * sessionRetentionBytesPerToken)
		if freed := sr.Reclaim(tr.clock); freed != want {
			t.Fatalf("Reclaim past window freed %d bytes, want %d", freed, want)
		}
		if sr.Has("sess-c") || sr.Sessions() != 0 || sr.RetainedBytes() != 0 {
			t.Fatalf("post-Reclaim registry: Has=%v Sessions=%d Bytes=%d, want empty",
				sr.Has("sess-c"), sr.Sessions(), sr.RetainedBytes())
		}
		child, _ := walkSuffix(tr, turn1())
		if tier := tr.NodeTier(child); tier != Tier3Probationary {
			t.Fatalf("post-Reclaim NodeTier=%v, want Tier3Probationary (descriptor cleared)", tier)
		}
	})

	t.Run("byte_budget_reclaims_oldest_sessions", func(t *testing.T) {
		tr := New(64) // holds two 32-token transcripts exactly
		sr, err := NewSessionRetention(tr, SessionRetentionConfig{
			Enabled: true, Priority: DefaultSessionRetentionPriority,
			TTL: DefaultSessionRetentionTTL, ByteBudget: 32, // budget below one entry's 64-byte footprint
		})
		if err != nil {
			t.Fatal(err)
		}
		turnX := cat(seqTokens(10, promptLen), seqTokens(11, genLen))
		turnY := cat(seqTokens(20, promptLen), seqTokens(21, genLen))
		insertUnleased(tr, turnX)
		if err := sr.Retain("sess-x", "", turnX, promptLen); err != nil {
			t.Fatal(err)
		}
		// A single oversized retain is never self-reclaimed: it stays over budget.
		if got := sr.RetainedBytes(); got != 64 {
			t.Fatalf("after first Retain: RetainedBytes=%d, want 64", got)
		}
		insertUnleased(tr, turnY)
		if err := sr.Retain("sess-y", "", turnY, promptLen); err != nil {
			t.Fatal(err)
		}
		// The budget displaces the OLDER session (equal priority, earlier admission).
		if sr.Has("sess-x") {
			t.Fatal("byte budget must reclaim the older session first")
		}
		if !sr.Has("sess-y") || sr.Sessions() != 1 || sr.RetainedBytes() != 64 {
			t.Fatalf("post-budget registry: HasY=%v Sessions=%d Bytes=%d, want sess-y alone at 64",
				sr.Has("sess-y"), sr.Sessions(), sr.RetainedBytes())
		}
		xChild, xn := walkSuffix(tr, turnX)
		yChild, yn := walkSuffix(tr, turnY)
		if xn != promptLen+genLen || yn != promptLen+genLen {
			t.Fatalf("precondition: both transcripts resident (%d/%d)", xn, yn)
		}
		if tier := tr.NodeTier(xChild); tier != Tier3Probationary {
			t.Fatalf("displaced session NodeTier=%v, want Tier3Probationary (descriptor cleared)", tier)
		}
		if tier := tr.NodeTier(yChild); tier != Tier0PinnedRoot {
			t.Fatalf("surviving session NodeTier=%v, want Tier0PinnedRoot", tier)
		}

		// Pressure: the displaced session's blocks are now ordinary cold leaves and go
		// first; the surviving session's suffix stays immune.
		insertUnleased(tr, seqTokens(30, 40))
		if m := tr.MatchLen(turnX); m != 0 {
			t.Fatalf("displaced session kept %d/%d under pressure, want 0", m, promptLen+genLen)
		}
		if m := tr.MatchLen(turnY); m != promptLen+genLen {
			t.Fatalf("surviving session lost %d tokens under pressure (matched %d/%d)",
				promptLen+genLen-m, m, promptLen+genLen)
		}
	})

	t.Run("default_off_reproduces_baseline_free_behavior", func(t *testing.T) {
		// The disabled default: involving the registry changes nothing about the
		// tree's behavior.
		disabled := DefaultSessionRetentionConfig()
		disabled.Enabled = false
		withRegistry := New(budget)
		sr, err := NewSessionRetention(withRegistry, disabled)
		if err != nil {
			t.Fatalf("disabled default config must be valid: %v", err)
		}
		if sr.Enabled() {
			t.Fatal("disabled default config must report Enabled=false")
		}
		insertUnleased(withRegistry, turn1())
		if err := sr.Retain("sess-d", "", turn1(), promptLen); err != nil {
			t.Fatalf("disabled Retain must be a nil no-op: %v", err)
		}
		if sr.Has("sess-d") || sr.RetainedBytes() != 0 {
			t.Fatalf("disabled Retain must record nothing: Has=%v Bytes=%d", sr.Has("sess-d"), sr.RetainedBytes())
		}
		insertUnleased(withRegistry, filler())

		// Pure baseline: the identical sequence with no registry at all.
		baseline := New(budget)
		insertUnleased(baseline, turn1())
		insertUnleased(baseline, filler())

		got, want := withRegistry.Stats(), baseline.Stats()
		if got.Tokens != want.Tokens || got.Evictions != want.Evictions {
			t.Fatalf("disabled registry diverged from baseline: Tokens=%d/%d Evictions=%d/%d",
				got.Tokens, want.Tokens, got.Evictions, want.Evictions)
		}
		if m := withRegistry.MatchLen(turn1()); m != 0 {
			t.Fatalf("default-off: finished-lane blocks survived pressure (matched %d), want baseline free behavior (0)", m)
		}
		if m := baseline.MatchLen(turn1()); m != 0 {
			t.Fatalf("baseline precondition failed: matched %d, want 0", m)
		}
	})

	t.Run("env_switch_default_off_and_fail_closed", func(t *testing.T) {
		cases := map[string]bool{
			"":       false,
			"0":      false,
			"off":    false,
			"FALSE":  false,
			"banana": false, // unknown value fails closed to disabled
			"1":      true,
			"on":     true,
			"TRUE":   true,
		}
		for val, wantEnabled := range cases {
			t.Setenv(SessionRetentionEnvVar, val)
			cfg := SessionRetentionConfigFromEnv()
			if cfg.Enabled != wantEnabled {
				t.Errorf("FAK_RADIX_SESSION_RETENTION=%q: Enabled=%v, want %v", val, cfg.Enabled, wantEnabled)
			}
			if wantEnabled {
				if cfg.Priority != DefaultSessionRetentionPriority || cfg.TTL != DefaultSessionRetentionTTL ||
					cfg.ByteBudget != DefaultSessionRetentionByteBudget {
					t.Errorf("enabled defaults drifted: %+v", cfg)
				}
			}
		}
	})

	t.Run("closed_contract_sentinels", func(t *testing.T) {
		def := DefaultSessionRetentionConfig()
		if _, err := NewSessionRetention(nil, def); !errors.Is(err, ErrSessionNilTree) {
			t.Fatalf("nil tree: err=%v, want ErrSessionNilTree", err)
		}
		for name, bad := range map[string]SessionRetentionConfig{
			"priority_high": {Enabled: true, Priority: MaxRetentionPriority + 1},
			"priority_low":  {Enabled: true, Priority: MinRetentionPriority - 1},
			"ttl_negative":  {Enabled: true, TTL: -1},
			// A never-expiring window is refused: it would take SetNodeRetention's
			// permanent-pin path at Tier-strength priority, pinning beyond the suffix.
			"ttl_forever": {Enabled: true, TTL: RetainForever},
			"budget_neg":  {Enabled: true, ByteBudget: -1},
		} {
			if _, err := NewSessionRetention(New(0), bad); !errors.Is(err, ErrSessionConfig) {
				t.Errorf("%s: err=%v, want ErrSessionConfig", name, err)
			}
		}

		tr := New(0)
		sr, err := NewSessionRetention(tr, def)
		if err != nil {
			t.Fatal(err)
		}
		tokens := turn1()
		check := func(name string, got error, want error) {
			t.Helper()
			if !errors.Is(got, want) {
				t.Errorf("%s: err=%v, want %v", name, got, want)
			}
		}
		check("empty_session", sr.Retain("", "", tokens, promptLen), ErrSessionEmpty)
		check("gen_start_negative", sr.Retain("s", "", tokens, -1), ErrSessionGenRange)
		check("gen_start_past_end", sr.Retain("s", "", tokens, len(tokens)+1), ErrSessionGenRange)
		check("path_not_cached", sr.Retain("s", "", tokens, promptLen), ErrSessionPathNotCached)

		// Partial residency: only the prompt is cached, the generated half is not.
		insertUnleased(tr, seqTokens(1, promptLen))
		check("partial_path_not_cached", sr.Retain("s", "", tokens, promptLen), ErrSessionPathNotCached)

		// No-op shapes admit nothing.
		if err := sr.Retain("s", "", nil, 0); err != nil {
			t.Fatalf("empty transcript no-op: %v", err)
		}
		if err := sr.Retain("s", "", tokens, len(tokens)); err != nil {
			t.Fatalf("zero-generated no-op: %v", err)
		}
		if sr.Sessions() != 0 || sr.RetainedBytes() != 0 {
			t.Fatalf("failed/no-op Retains must leave the registry empty: Sessions=%d Bytes=%d",
				sr.Sessions(), sr.RetainedBytes())
		}
	})
}
