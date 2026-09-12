package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

func TestFeatureProofReceipts_Verification(t *testing.T) {
	before, after := []byte("HEAD1234567890TAIL"), []byte("HEADxTAIL")
	pre, post := 12, 4
	witness := FeatureRewriteWitness{PrefixBytes: 4, SuffixBytes: 4, ShedBytes: 9, PreTokens: &pre, PostTokens: &post, TokenMethod: FeatureProofAnthropicEstimate}
	proof, err := VerifyFeatureRewrite(before, after, witness)
	if err != nil {
		t.Fatal(err)
	}
	wantHash := sha256.Sum256(before)
	if proof.PreSHA256 != hex.EncodeToString(wantHash[:]) || proof.ShedBytes != 9 || proof.PrePrefixSHA256 != proof.PostPrefixSHA256 || proof.PreSuffixSHA256 != proof.PostSuffixSHA256 {
		t.Fatalf("wrong quantitative proof: %+v", proof)
	}
	for _, tc := range []struct {
		name  string
		after []byte
		w     FeatureRewriteWitness
	}{
		{"prefix", []byte("DEADxTAIL"), witness},
		{"tail", []byte("HEADxFAIL"), witness},
		{"wrong-delta", after, FeatureRewriteWitness{PrefixBytes: 4, SuffixBytes: 4, ShedBytes: 8}},
		{"overlap", after, FeatureRewriteWitness{PrefixBytes: 5, SuffixBytes: 5, ShedBytes: 9}},
		{"unpaired-tokens", after, FeatureRewriteWitness{PrefixBytes: 4, SuffixBytes: 4, ShedBytes: 9, PreTokens: &pre}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := VerifyFeatureRewrite(before, tc.after, tc.w); err == nil {
				t.Fatal("accepted corrupt proof")
			}
		})
	}
	c, err := NewFeatureProofCollector(2)
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Snapshot().Receipts) != 0 {
		t.Fatal("idle collector invented execution")
	}
	if err := c.RecordRewrite(FeatureCompactHistory, before, after, witness); err != nil {
		t.Fatal(err)
	}
	pre = 999
	first := c.Snapshot()
	if *first.Receipts[0].Rewrite.PreTokens != 12 {
		t.Fatal("retained caller token pointer")
	}
	*first.Receipts[0].Rewrite.PreTokens = 123
	first.Receipts[0].Rewrite.PreSHA256 = "corrupted"
	if got := c.Snapshot().Receipts[0].Rewrite; *got.PreTokens != 12 || got.PreSHA256 == "corrupted" {
		t.Fatal("snapshot mutation changed retained proof")
	}
	if err := c.RecordRewrite(FeatureElideResults, before, after, FeatureRewriteWitness{PrefixBytes: 4, SuffixBytes: 4, ShedBytes: 9}); err != nil {
		t.Fatal(err)
	}
	if err := c.RecordRewrite(FeatureCompactHistory, before, []byte("DEADxTAIL"), witness); err == nil {
		t.Fatal("bad prefix did not report telemetry error")
	}
	got := c.Snapshot()
	if len(got.Receipts) != 2 || got.EvictedReceipts != 1 || got.Receipts[0].Sequence != 2 || got.Receipts[1].Sequence != 3 {
		t.Fatalf("ring chronology=%+v", got)
	}
	failure := got.Receipts[1]
	if failure.Outcome != FeatureProofFailed || failure.Reason != "PROOF_GENERATION_FAILED" || failure.Rewrite != nil || failure.Cache != nil {
		t.Fatalf("failed proof retained partial evidence: %+v", failure)
	}
	for _, counter := range got.Counters {
		if counter.Feature == FeatureCompactHistory {
			if counter.Count != 1 {
				t.Errorf("compaction outcome count=%+v", counter)
			}
			if counter.Outcome == FeatureProofFailed && counter.SavedBytes != 0 {
				t.Error("failed proof claimed savings")
			}
			if counter.Outcome == FeatureProofVerified && counter.SavedBytes != 9 {
				t.Error("verified savings lost on eviction")
			}
		}
	}
	count := got.Receipts[1].Sequence
	if err := c.RecordRewrite(FeatureElideResults, after, after, FeatureRewriteWitness{}); err != nil {
		t.Fatal(err)
	}
	if err := c.RecordFailure(ServeFeature("unbounded-label"), FeatureProofInvalidBody); err == nil {
		t.Fatal("accepted unknown feature")
	}
	if c.Snapshot().Receipts[1].Sequence != count {
		t.Fatal("no-op or unknown feature created receipt")
	}
}

func TestFeatureProofCacheAndConcurrentRetention(t *testing.T) {
	for _, capacity := range []int{0, -1, 257} {
		if _, err := NewFeatureProofCollector(capacity); err == nil {
			t.Errorf("accepted capacity %d", capacity)
		}
	}
	c, err := NewFeatureProofCollector(8)
	if err != nil {
		t.Fatal(err)
	}
	identity := sha256.Sum256([]byte("canonical-call"))
	payload := []byte("served-payload")
	verdict := WireVerdict{Kind: "ALLOW", Reason: "SERVED_INLINE", By: "vdso"}
	if err := c.RecordVDSOServe(identity, payload, verdict); err != nil {
		t.Fatal(err)
	}
	cache := c.Snapshot().Receipts[0].Cache
	if cache.HitCount != 1 || cache.DurationSavedNS != nil || cache.DurationReason != "MATCHED_TIMING_UNAVAILABLE" || cache.WitnessKind != "gateway_call_result_sha256" {
		t.Fatalf("cache proof makes unsupported timing claim: %+v", cache)
	}
	encoded, err := json.Marshal(cache)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"duration_saved_ns":null`) || strings.Contains(string(encoded), "served-payload") || strings.Contains(string(encoded), "canonical-call") {
		t.Fatalf("cache proof leaked inputs or fabricated timing: %s", encoded)
	}
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := c.RecordVDSOServe(identity, payload, verdict); err != nil {
				t.Error(err)
			}
			_ = c.Snapshot()
		}()
	}
	wg.Wait()
	snapshot := c.Snapshot()
	if len(snapshot.Receipts) != 8 || snapshot.EvictedReceipts != 25 {
		t.Fatalf("concurrent bounded ring=%d evicted=%d", len(snapshot.Receipts), snapshot.EvictedReceipts)
	}
	for _, counter := range snapshot.Counters {
		if counter.Feature == FeatureVDSO && counter.Outcome == FeatureProofVerified && counter.Count != 33 {
			t.Errorf("concurrent hits=%d", counter.Count)
		}
	}
}

func TestFeatureProofRetainedWindowTracksOnlyLiveReceipts(t *testing.T) {
	c, err := NewFeatureProofCollector(2)
	if err != nil {
		t.Fatal(err)
	}
	empty := c.Snapshot()
	if empty.RetainedWindow.ReceiptCount != 0 || empty.RetainedWindow.FirstSequence != 0 || empty.RetainedWindow.LastSequence != 0 {
		t.Fatalf("empty retained window invented bounds: %+v", empty.RetainedWindow)
	}
	for _, counter := range empty.RetainedWindow.Counters {
		if counter.Count != 0 || counter.SavedBytes != 0 {
			t.Fatalf("empty retained window invented quantities: %+v", counter)
		}
	}

	before, after := []byte("HEAD1234567890TAIL"), []byte("HEADxTAIL")
	witness := FeatureRewriteWitness{PrefixBytes: 4, SuffixBytes: 4, ShedBytes: 9}
	if err := c.RecordRewrite(FeatureCompactHistory, before, after, witness); err != nil {
		t.Fatal(err)
	}
	if err := c.RecordFailure(FeatureCompactHistory, FeatureProofInvalidBody); err != nil {
		t.Fatal(err)
	}
	if err := c.RecordRewrite(FeatureElideResults, before, after, witness); err != nil {
		t.Fatal(err)
	}

	snapshot := c.Snapshot()
	if snapshot.EvictedReceipts != 1 || snapshot.RetainedWindow.ReceiptCount != 2 ||
		snapshot.RetainedWindow.FirstSequence != 2 || snapshot.RetainedWindow.LastSequence != 3 {
		t.Fatalf("wrong retained window after eviction: %+v", snapshot.RetainedWindow)
	}
	assertFeatureProofCounter(t, snapshot.Counters, FeatureCompactHistory, FeatureProofVerified, 1, 9)
	assertFeatureProofCounter(t, snapshot.Counters, FeatureCompactHistory, FeatureProofFailed, 1, 0)
	assertFeatureProofCounter(t, snapshot.Counters, FeatureElideResults, FeatureProofVerified, 1, 9)
	assertFeatureProofCounter(t, snapshot.RetainedWindow.Counters, FeatureCompactHistory, FeatureProofVerified, 0, 0)
	assertFeatureProofCounter(t, snapshot.RetainedWindow.Counters, FeatureCompactHistory, FeatureProofFailed, 1, 0)
	assertFeatureProofCounter(t, snapshot.RetainedWindow.Counters, FeatureElideResults, FeatureProofVerified, 1, 9)

	if err := c.RecordFailure(FeatureVDSO, FeatureProofInvalidCache); err != nil {
		t.Fatal(err)
	}
	snapshot = c.Snapshot()
	if snapshot.EvictedReceipts != 2 || snapshot.RetainedWindow.ReceiptCount != 2 ||
		snapshot.RetainedWindow.FirstSequence != 3 || snapshot.RetainedWindow.LastSequence != 4 {
		t.Fatalf("retained window did not roll forward: %+v", snapshot.RetainedWindow)
	}
	assertFeatureProofCounter(t, snapshot.RetainedWindow.Counters, FeatureCompactHistory, FeatureProofFailed, 0, 0)
	assertFeatureProofCounter(t, snapshot.RetainedWindow.Counters, FeatureElideResults, FeatureProofVerified, 1, 9)
	assertFeatureProofCounter(t, snapshot.RetainedWindow.Counters, FeatureVDSO, FeatureProofFailed, 1, 0)
	assertFeatureProofCounter(t, snapshot.Counters, FeatureCompactHistory, FeatureProofFailed, 1, 0)
	assertFeatureProofCounter(t, snapshot.Counters, FeatureVDSO, FeatureProofFailed, 1, 0)
}

func assertFeatureProofCounter(t *testing.T, counters []FeatureProofCounter, feature ServeFeature, outcome FeatureProofOutcome, count, saved uint64) {
	t.Helper()
	for _, counter := range counters {
		if counter.Feature == feature && counter.Outcome == outcome {
			if counter.Count != count || counter.SavedBytes != saved {
				t.Fatalf("counter %s/%s = count %d saved %d, want count %d saved %d", feature, outcome, counter.Count, counter.SavedBytes, count, saved)
			}
			return
		}
	}
	t.Fatalf("missing counter %s/%s", feature, outcome)
}
