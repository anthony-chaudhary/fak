package session

import (
	"sync"
	"testing"
)

func TestTokenSource_String(t *testing.T) {
	tests := []struct {
		source TokenSource
		want   string
	}{
		{source: TokenSourceFallback, want: "fallback"},
		{source: TokenSourceNativeTranscript, want: "transcript"},
		{source: TokenSourceStdout, want: "stdout"},
		{source: TokenSourceNetwork, want: "network"},
		{source: TokenSource(-1), want: "unknown"},
		{source: TokenSource(100), want: "unknown"},
	}

	for _, tt := range tests {
		if got := tt.source.String(); got != tt.want {
			t.Errorf("TokenSource(%d).String() = %q, want %q", int(tt.source), got, tt.want)
		}
	}
}

func TestTokenSource_NoDoubleCounting(t *testing.T) {
	merger := NewMultiSourceTokenMerger()

	transcript := TokenRecord{
		Source:       TokenSourceNativeTranscript,
		SessionID:    "s1",
		TurnIndex:    1,
		CallID:       "c1",
		PromptTokens: 100,
		OutputTokens: 25,
		TotalTokens:  125,
	}
	network := TokenRecord{
		Source:       TokenSourceNetwork,
		SessionID:    "s1",
		TurnIndex:    1,
		CallID:       "c1",
		PromptTokens: 110,
		OutputTokens: 30,
		TotalTokens:  140,
	}

	// Ingest transcript then network.
	merger.Ingest(transcript)
	merger.Ingest(network)

	// Replay both.
	merger.Ingest(transcript)
	merger.Ingest(network)

	prompt, output, total, cached, created := merger.Totals()
	if prompt != 110 {
		t.Fatalf("prompt = %d, want 110 (no double count)", prompt)
	}
	if output != 30 {
		t.Fatalf("output = %d, want 30 (no double count)", output)
	}
	if total != 140 {
		t.Fatalf("total = %d, want 140 (no double count)", total)
	}
	if cached != 0 || created != 0 {
		t.Fatalf("cached = %d, created = %d, want 0, 0", cached, created)
	}

	records := merger.Records()
	if len(records) != 1 {
		t.Fatalf("len(records) = %d, want 1", len(records))
	}
	if records[0].Source != TokenSourceNetwork {
		t.Fatalf("source = %v, want TokenSourceNetwork", records[0].Source)
	}
}

func TestTokenSource_HigherPriorityOverwritesLower(t *testing.T) {
	merger := NewMultiSourceTokenMerger()

	rFallback := TokenRecord{
		Source:       TokenSourceFallback,
		SessionID:    "s1",
		TurnIndex:    1,
		PromptTokens: 50,
		OutputTokens: 10,
		TotalTokens:  60,
	}
	rTranscript := TokenRecord{
		Source:       TokenSourceNativeTranscript,
		SessionID:    "s1",
		TurnIndex:    1,
		PromptTokens: 60,
		OutputTokens: 15,
		TotalTokens:  75,
	}
	rStdout := TokenRecord{
		Source:       TokenSourceStdout,
		SessionID:    "s1",
		TurnIndex:    1,
		PromptTokens: 70,
		OutputTokens: 20,
		TotalTokens:  90,
	}
	rNetwork := TokenRecord{
		Source:       TokenSourceNetwork,
		SessionID:    "s1",
		TurnIndex:    1,
		PromptTokens: 80,
		OutputTokens: 25,
		TotalTokens:  105,
	}

	merger.Ingest(rFallback)
	if p, o, tot, _, _ := merger.Totals(); p != 50 || o != 10 || tot != 60 {
		t.Fatalf("fallback mismatch: p=%d, o=%d, tot=%d", p, o, tot)
	}

	merger.Ingest(rTranscript)
	if p, o, tot, _, _ := merger.Totals(); p != 60 || o != 15 || tot != 75 {
		t.Fatalf("transcript mismatch: p=%d, o=%d, tot=%d", p, o, tot)
	}

	merger.Ingest(rStdout)
	if p, o, tot, _, _ := merger.Totals(); p != 70 || o != 20 || tot != 90 {
		t.Fatalf("stdout mismatch: p=%d, o=%d, tot=%d", p, o, tot)
	}

	merger.Ingest(rNetwork)
	if p, o, tot, _, _ := merger.Totals(); p != 80 || o != 25 || tot != 105 {
		t.Fatalf("network mismatch: p=%d, o=%d, tot=%d", p, o, tot)
	}

	recs := merger.Records()
	if len(recs) != 1 || recs[0].Source != TokenSourceNetwork {
		t.Fatalf("records mismatch: %+v", recs)
	}
}

func TestTokenSource_EnrichMissingFieldsLowerAfterHigher(t *testing.T) {
	merger := NewMultiSourceTokenMerger()

	network := TokenRecord{
		Source:        TokenSourceNetwork,
		SessionID:     "s1",
		TurnIndex:     1,
		PromptTokens:  200,
		OutputTokens:  50,
		TotalTokens:   250,
		CachedTokens:  0,
		CreatedTokens: 0,
	}
	transcript := TokenRecord{
		Source:        TokenSourceNativeTranscript,
		SessionID:     "s1",
		TurnIndex:     1,
		PromptTokens:  180,
		OutputTokens:  40,
		TotalTokens:   220,
		CachedTokens:  64,
		CreatedTokens: 32,
	}

	// Higher priority arrives first.
	merger.Ingest(network)
	// Lower priority arrives second.
	merger.Ingest(transcript)

	p, o, tot, cached, created := merger.Totals()
	if p != 200 {
		t.Fatalf("prompt = %d, want 200 (not overwritten)", p)
	}
	if o != 50 {
		t.Fatalf("output = %d, want 50 (not overwritten)", o)
	}
	if tot != 250 {
		t.Fatalf("total = %d, want 250 (not overwritten)", tot)
	}
	if cached != 64 {
		t.Fatalf("cached = %d, want 64 (enriched)", cached)
	}
	if created != 32 {
		t.Fatalf("created = %d, want 32 (enriched)", created)
	}

	recs := merger.Records()
	if len(recs) != 1 {
		t.Fatalf("len(recs) = %d, want 1", len(recs))
	}
	if recs[0].Source != TokenSourceNetwork {
		t.Fatalf("source = %v, want TokenSourceNetwork", recs[0].Source)
	}
	if recs[0].CachedTokens != 64 || recs[0].CreatedTokens != 32 {
		t.Fatalf("missing fields enrichment mismatch: %+v", recs[0])
	}
}

func TestTokenSource_PreserveMissingFieldsLowerBeforeHigher(t *testing.T) {
	merger := NewMultiSourceTokenMerger()

	stdout := TokenRecord{
		Source:        TokenSourceStdout,
		SessionID:     "s1",
		TurnIndex:     1,
		PromptTokens:  180,
		OutputTokens:  40,
		TotalTokens:   220,
		CachedTokens:  128,
		CreatedTokens: 64,
	}
	network := TokenRecord{
		Source:        TokenSourceNetwork,
		SessionID:     "s1",
		TurnIndex:     1,
		PromptTokens:  190,
		OutputTokens:  45,
		TotalTokens:   235,
		CachedTokens:  0,
		CreatedTokens: 0,
	}

	// Lower priority arrives first.
	merger.Ingest(stdout)
	// Higher priority arrives second with 0 cached/created.
	merger.Ingest(network)

	p, o, tot, cached, created := merger.Totals()
	if p != 190 {
		t.Fatalf("prompt = %d, want 190", p)
	}
	if o != 45 {
		t.Fatalf("output = %d, want 45", o)
	}
	if tot != 235 {
		t.Fatalf("total = %d, want 235", tot)
	}
	if cached != 128 {
		t.Fatalf("cached = %d, want 128 (preserved)", cached)
	}
	if created != 64 {
		t.Fatalf("created = %d, want 64 (preserved)", created)
	}

	recs := merger.Records()
	if len(recs) != 1 || recs[0].Source != TokenSourceNetwork {
		t.Fatalf("records mismatch: %+v", recs)
	}
}

func TestTokenSource_IndependentTurns(t *testing.T) {
	merger := NewMultiSourceTokenMerger()

	r1 := TokenRecord{
		Source:       TokenSourceStdout,
		SessionID:    "s1",
		TurnIndex:    1,
		CallID:       "",
		PromptTokens: 10,
		OutputTokens: 5,
		TotalTokens:  15,
	}
	r2 := TokenRecord{
		Source:       TokenSourceStdout,
		SessionID:    "s1",
		TurnIndex:    2,
		CallID:       "",
		PromptTokens: 20,
		OutputTokens: 10,
		TotalTokens:  30,
	}
	r3 := TokenRecord{
		Source:       TokenSourceStdout,
		SessionID:    "s1",
		TurnIndex:    2,
		CallID:       "call_123",
		PromptTokens: 30,
		OutputTokens: 15,
		TotalTokens:  45,
	}
	r4 := TokenRecord{
		Source:       TokenSourceStdout,
		SessionID:    "s2",
		TurnIndex:    1,
		CallID:       "",
		PromptTokens: 40,
		OutputTokens: 20,
		TotalTokens:  60,
	}

	merger.Ingest(r1)
	merger.Ingest(r2)
	merger.Ingest(r3)
	merger.Ingest(r4)

	recs := merger.Records()
	if len(recs) != 4 {
		t.Fatalf("len(recs) = %d, want 4", len(recs))
	}

	// Verify deterministic sorting:
	// 1. s1, TurnIndex 1, CallID ""
	// 2. s1, TurnIndex 2, CallID ""
	// 3. s1, TurnIndex 2, CallID "call_123"
	// 4. s2, TurnIndex 1, CallID ""
	if recs[0].SessionID != "s1" || recs[0].TurnIndex != 1 || recs[0].CallID != "" {
		t.Errorf("recs[0] = %+v", recs[0])
	}
	if recs[1].SessionID != "s1" || recs[1].TurnIndex != 2 || recs[1].CallID != "" {
		t.Errorf("recs[1] = %+v", recs[1])
	}
	if recs[2].SessionID != "s1" || recs[2].TurnIndex != 2 || recs[2].CallID != "call_123" {
		t.Errorf("recs[2] = %+v", recs[2])
	}
	if recs[3].SessionID != "s2" || recs[3].TurnIndex != 1 || recs[3].CallID != "" {
		t.Errorf("recs[3] = %+v", recs[3])
	}

	p, o, tot, _, _ := merger.Totals()
	if p != 100 || o != 50 || tot != 150 {
		t.Fatalf("totals = (%d, %d, %d), want (100, 50, 150)", p, o, tot)
	}

	u := merger.TotalUsage()
	if u.ContextTokens != 100 || u.OutputTokens != 50 {
		t.Fatalf("usage = %+v, want ContextTokens: 100, OutputTokens: 50", u)
	}
}

func TestTokenSource_Concurrency(t *testing.T) {
	merger := NewMultiSourceTokenMerger()
	const workers = 16
	const turns = 50

	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		workerID := w
		go func() {
			defer wg.Done()
			for i := 0; i < turns; i++ {
				source := TokenSourceFallback
				switch workerID % 4 {
				case 1:
					source = TokenSourceNativeTranscript
				case 2:
					source = TokenSourceStdout
				case 3:
					source = TokenSourceNetwork
				}

				merger.Ingest(TokenRecord{
					Source:       source,
					SessionID:    "shared_session",
					TurnIndex:    i,
					PromptTokens: 10 * (int(source) + 1),
					OutputTokens: 5 * (int(source) + 1),
					TotalTokens:  15 * (int(source) + 1),
				})

				_ = merger.Records()
				_, _, _, _, _ = merger.Totals()
				_ = merger.TotalUsage()
			}
		}()
	}

	wg.Wait()

	recs := merger.Records()
	if len(recs) != turns {
		t.Fatalf("len(recs) = %d, want %d", len(recs), turns)
	}
	for _, r := range recs {
		if r.Source != TokenSourceNetwork {
			t.Errorf("turn %d source = %v, want TokenSourceNetwork", r.TurnIndex, r.Source)
		}
	}
}

func TestTokenSource_MergeHelper(t *testing.T) {
	records := []TokenRecord{
		{
			Source:       TokenSourceStdout,
			SessionID:    "s1",
			TurnIndex:    2,
			PromptTokens: 20,
			OutputTokens: 10,
		},
		{
			Source:       TokenSourceFallback,
			SessionID:    "s1",
			TurnIndex:    1,
			PromptTokens: 10,
			OutputTokens: 5,
		},
		{
			Source:       TokenSourceNetwork,
			SessionID:    "s1",
			TurnIndex:    1,
			PromptTokens: 15,
			OutputTokens: 8,
		},
	}

	merged := MergeTokenRecords(records)
	if len(merged) != 2 {
		t.Fatalf("len(merged) = %d, want 2", len(merged))
	}

	// Ordered deterministically by TurnIndex:
	if merged[0].TurnIndex != 1 || merged[0].Source != TokenSourceNetwork || merged[0].PromptTokens != 15 {
		t.Errorf("merged[0] mismatch: %+v", merged[0])
	}
	if merged[1].TurnIndex != 2 || merged[1].Source != TokenSourceStdout || merged[1].PromptTokens != 20 {
		t.Errorf("merged[1] mismatch: %+v", merged[1])
	}
}
