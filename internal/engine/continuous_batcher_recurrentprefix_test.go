package engine

import (
	"context"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestRecurrentPrefixReuseReportedWhenYieldedBeforeStep(t *testing.T) {
	t.Parallel()
	const sessionID = "recurrent-prefix-yield"
	prefix := []int{1, 2, 3}
	fullPrompt := []int{1, 2, 3, 4, 5, 6}
	ctx := context.Background()

	newBatcher := func(disable bool) *ContinuousBatcher {
		t.Helper()
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 1
		cfg.PrefillBudget = 0 // Admission eagerly completes prefill before yielding.
		cfg.Model = rpHybridModel()
		cfg.DisableRecurrentPrefixReuse = disable
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher: %v", err)
		}
		t.Cleanup(func() { _ = cb.Close() })
		return cb
	}
	submit := func(cb *ContinuousBatcher, prompt []int, target int) {
		t.Helper()
		id, err := cb.Submit(&SubagentRequest{SessionID: sessionID, PromptTokens: prompt, TargetTokens: target})
		if err != nil || id != sessionID {
			t.Fatalf("Submit: id=%q err=%v", id, err)
		}
	}
	step := func(cb *ContinuousBatcher, phase BatchPhase, hits, reused int) *BatchStepResult {
		t.Helper()
		res, err := cb.StepPhase(ctx)
		if err != nil {
			t.Fatalf("StepPhase: %v", err)
		}
		if res.Phase != phase {
			t.Fatalf("phase=%s, want %s", res.Phase, phase)
		}
		if res.PrefixHits != hits || res.PrefixReuseTokens != reused {
			t.Errorf("%s reuse=%d hits/%d tokens, want %d/%d", phase, res.PrefixHits, res.PrefixReuseTokens, hits, reused)
		}
		if res.PrefillTokens != 0 || res.TotalSlots != 1 || len(res.PromotedSessionIDs) != 0 {
			t.Fatalf("unexpected prefill/slots/promotion: %+v", res)
		}
		if phase == PhaseIdle {
			if res.ActiveSlots != 0 || res.TokensGenerated != 0 || res.DecodeTokens != 0 ||
				len(res.GeneratedTokens) != 0 || len(res.DecodeResidentUIDs) != 0 || len(res.RetiredSessionIDs) != 0 {
				t.Fatalf("idle performed generation or retirement: %+v", res)
			}
		} else if res.ActiveSlots != 1 || res.TokensGenerated != 1 || res.DecodeTokens != 1 ||
			len(res.GeneratedTokens) != 1 || len(res.DecodeResidentUIDs) != 1 {
			t.Fatalf("decode did not advance exactly one resident: %+v", res)
		}
		return res
	}
	finishedTokens := func(cb *ContinuousBatcher, target int) []int {
		t.Helper()
		slot, ok := cb.GetSlot(sessionID)
		if !ok || slot.State != SlotStateFinished || slot.TokensGenerated() != target || slot.Err() != nil {
			t.Fatalf("request did not finish with %d tokens: slot=%+v found=%v", target, slot, ok)
		}
		select {
		case <-slot.Done():
		default:
			t.Fatal("finished request's Done channel remains open")
		}
		var streamed []int
		for token := range slot.Tokens() {
			streamed = append(streamed, token)
		}
		if !reflect.DeepEqual(streamed, slot.GeneratedTokens) {
			t.Fatalf("stream=%v, generated=%v", streamed, slot.GeneratedTokens)
		}
		if cb.ActiveSlotCount() != 0 || cb.YieldedSlotCount() != 0 || cb.EmptySlotCount() != 1 || cb.CurrentKVCacheBytes() != 0 {
			t.Fatal("finished request did not release the resident slot and KV allocation")
		}
		return streamed
	}

	// A fresh cache-disabled batcher supplies the full-prompt generation baseline.
	baseline := newBatcher(true)
	if _, enabled := baseline.PrefixCacheStats(); enabled {
		t.Fatal("baseline unexpectedly has a prefix cache")
	}
	submit(baseline, fullPrompt, 2)
	step(baseline, PhaseDecode, 0, 0)
	step(baseline, PhaseDecode, 0, 0)
	wantTokens := finishedTokens(baseline, 2)

	cb := newBatcher(false)
	submit(cb, prefix, 1)
	step(cb, PhaseDecode, 0, 0)
	finishedTokens(cb, 1)
	stats, enabled := cb.PrefixCacheStats()
	if !enabled || stats.Puts != 1 || stats.Hits != 0 || cb.PrefixCacheLen() != 1 {
		t.Fatalf("completed warm request did not retain one real prefix: stats=%+v enabled=%v len=%d", stats, enabled, cb.PrefixCacheLen())
	}

	submit(cb, fullPrompt, 2)
	admitted := cb.Slots()[0]
	if admitted.State != SlotStateActiveDecode || admitted.SessionID != sessionID || admitted.PrefixHits != 1 ||
		admitted.PrefixMatchedTokens != 3 || admitted.TokensGenerated() != 0 || admitted.sess == nil || admitted.sess.Cache == nil {
		t.Fatalf("eager extension did not restore the three-token prefix: %+v", admitted)
	}
	stats, _ = cb.PrefixCacheStats()
	if stats.Hits != 1 {
		t.Fatalf("real cache hits=%d, want 1", stats.Hits)
	}
	cacheBefore := admitted.sess.Cache.Clone()
	kvBytes := cb.CurrentKVCacheBytes()
	if kvBytes == 0 || kvBytes != admitted.KVCacheBytes {
		t.Fatalf("admission KV bytes=%d, slot=%d", kvBytes, admitted.KVCacheBytes)
	}
	assertStationary := func(want *Slot) {
		t.Helper()
		got := cb.Slots()[0]
		if !reflect.DeepEqual(got, want) || cb.CurrentKVCacheBytes() != kvBytes ||
			!reflect.DeepEqual(got.sess.Cache.Clone(), cacheBefore) {
			t.Fatal("yield/idle/resume changed slot or complete hybrid KV state beyond the lifecycle transition")
		}
		select {
		case <-got.Done():
			t.Fatal("undecoded request completed")
		default:
		}
		select {
		case token, open := <-got.Tokens():
			t.Fatalf("undecoded stream changed: token=%d open=%v", token, open)
		default:
		}
	}

	// No StepPhase may intervene between eager cache admission and YieldSlot.
	if err := cb.YieldSlot(sessionID); err != nil {
		t.Fatalf("YieldSlot: %v", err)
	}
	admitted.State = SlotStateYieldedIO
	admitted.YieldCount = 1
	assertStationary(admitted)
	if cb.YieldedSlotCount() != 1 || cb.ActiveSlotCount() != 0 || len(cb.DecodeResident()) != 0 || cb.WaitingQueueLength() != 0 {
		t.Fatal("yielded request remains decode resident or queued")
	}

	totalHits, totalReuse, totalQueried := 0, 0, 0
	for i := 0; i < 2; i++ {
		hits, reused := 0, 0
		if i == 0 {
			hits, reused = 1, 3
		}
		res := step(cb, PhaseIdle, hits, reused)
		totalHits += res.PrefixHits
		totalReuse += res.PrefixReuseTokens
		totalQueried += res.PrefixQueriedTokens
		if res.YieldedSlots != 1 || res.EmptySlots != 0 || res.KVCacheBytesUsed != kvBytes || cb.TotalTokensGenerated() != 1 {
			t.Fatalf("idle changed yielded lifecycle or generation: %+v", res)
		}
		assertStationary(admitted)
	}

	if err := cb.ResumeSlot(sessionID); err != nil {
		t.Fatalf("ResumeSlot: %v", err)
	}
	admitted.State = SlotStateActiveDecode
	admitted.ResumeCount = 1
	assertStationary(admitted)
	if cb.YieldedSlotCount() != 0 || cb.ActiveSlotCount() != 1 || len(cb.DecodeResident()) != 1 {
		t.Fatal("resumed request is not the sole decode resident")
	}
	for i := 0; i < 2; i++ {
		res := step(cb, PhaseDecode, 0, 0)
		totalHits += res.PrefixHits
		totalReuse += res.PrefixReuseTokens
		totalQueried += res.PrefixQueriedTokens
		if res.DecodeResidentUIDs[0] != admitted.SubmissionSeq {
			t.Fatal("resume changed request residency identity")
		}
		if i == 0 && len(res.RetiredSessionIDs) != 0 || i == 1 && !reflect.DeepEqual(res.RetiredSessionIDs, []string{sessionID}) {
			t.Fatalf("unexpected retirement at decode %d: %v", i+1, res.RetiredSessionIDs)
		}
	}
	gotTokens := finishedTokens(cb, 2)
	if !reflect.DeepEqual(gotTokens, wantTokens) {
		t.Fatalf("reused/yielded generation=%v, cache-disabled full-prompt baseline=%v", gotTokens, wantTokens)
	}
	res := step(cb, PhaseIdle, 0, 0)
	totalHits += res.PrefixHits
	totalReuse += res.PrefixReuseTokens
	totalQueried += res.PrefixQueriedTokens
	// The hit-rate denominator is reported once per admission lookup and bounds
	// the matched tokens from above (vLLM prefix_cache_queries >= hits).
	if totalQueried < totalReuse || totalQueried == 0 {
		t.Fatalf("prefix queried tokens=%d, want >= reused %d and > 0", totalQueried, totalReuse)
	}
	if totalHits != 1 || totalReuse != 3 || cb.TotalTokensGenerated() != 3 || res.YieldedSlots != 0 || res.EmptySlots != 1 || res.KVCacheBytesUsed != 0 {
		t.Fatalf("final accounting: hits=%d reused=%d generated=%d result=%+v", totalHits, totalReuse, cb.TotalTokensGenerated(), res)
	}
}
