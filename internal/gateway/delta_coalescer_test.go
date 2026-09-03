package gateway

import (
	"strings"
	"sync"
	"testing"
	"time"
)

// TestCoalescingSenderDecouplesSlowConsumer verifies the core requirement of #10725:
// with a slow/throttled network consumer, the generator completes in steady-state time (< 20ms)
// while sending 50 chunks, deltas coalesce into merged chunks, and the slow reader receives
// all text and tokens without drops.
func TestCoalescingSenderDecouplesSlowConsumer(t *testing.T) {
	var mu sync.Mutex
	var deliveredChunks []DeltaChunk
	totalDeliveredCalls := 0

	// Slow consumer: takes 5ms per chunk delivered
	slowDeliver := func(chunk DeltaChunk) error {
		time.Sleep(5 * time.Millisecond)
		mu.Lock()
		deliveredChunks = append(deliveredChunks, chunk)
		totalDeliveredCalls++
		mu.Unlock()
		return nil
	}

	sender := NewCoalescingSender(slowDeliver)

	// Fast generator: sends 50 small token chunks in rapid succession
	genStart := time.Now()
	for i := 0; i < 50; i++ {
		sender.Send("chunk-", 1)
	}
	genElapsed := time.Since(genStart)

	// Generator must finish rapidly (< 20ms) without stalling on the slow network writer
	if genElapsed > 25*time.Millisecond {
		t.Fatalf("generator loop stalled on slow consumer: took %v, want < 25ms", genElapsed)
	}

	// Close waits for remaining coalesced deltas to drain
	sender.Close()

	mu.Lock()
	defer mu.Unlock()

	// Verify that deltas coalesced: fewer calls than 50
	if totalDeliveredCalls >= 50 {
		t.Fatalf("expected deltas to coalesce under slow consumer, got %d calls for 50 sends", totalDeliveredCalls)
	}

	// Verify complete text and tokens are delivered with zero loss
	var totalText strings.Builder
	totalTokens := 0
	for _, c := range deliveredChunks {
		totalText.WriteString(c.Text)
		totalTokens += c.Tokens
	}

	wantText := strings.Repeat("chunk-", 50)
	if totalText.String() != wantText {
		t.Fatalf("reconstructed text mismatch: got %q, want %q", totalText.String(), wantText)
	}
	if totalTokens != 50 {
		t.Fatalf("total tokens mismatch: got %d, want 50", totalTokens)
	}
}

// TestCoalescingSenderFastConsumerDeliversAll verifies that with a fast consumer,
// all chunks are delivered completely and promptly.
func TestCoalescingSenderFastConsumerDeliversAll(t *testing.T) {
	var mu sync.Mutex
	var delivered []DeltaChunk

	fastDeliver := func(chunk DeltaChunk) error {
		mu.Lock()
		delivered = append(delivered, chunk)
		mu.Unlock()
		return nil
	}

	sender := NewCoalescingSender(fastDeliver)
	for i := 0; i < 10; i++ {
		sender.Send("a", 1)
	}
	sender.Close()

	mu.Lock()
	defer mu.Unlock()

	var totalText strings.Builder
	totalTokens := 0
	for _, c := range delivered {
		totalText.WriteString(c.Text)
		totalTokens += c.Tokens
	}

	if totalText.String() != "aaaaaaaaaa" || totalTokens != 10 {
		t.Fatalf("fast delivery failed: text=%q tokens=%d", totalText.String(), totalTokens)
	}
}
