package modelengine

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ExactHTTPFixtureForTest exposes only test coordination; Engine preparation and
// the scheduler run loop are the production implementations.
func ExactHTTPFixtureForTest(t *testing.T, tok NLTokenizer, enabled bool) (*Engine, func(context.Context, int, []int, int) error, func(), func() (int64, uint64), func(context.Context) error) {
	t.Helper()
	e := New()
	e.PreloadQ4K(nativeSchedulerPrefillModel(t))
	e.SetTokenizer(tok)
	e.SetInBatchPrefixDedup(enabled)
	s := e.nativeScheduler()
	var chunks atomic.Int64
	s.observeNativeEvent = func(ev nativeSchedulerEvent) {
		if ev.Kind == nativeSchedulerEventPrefill {
			chunks.Add(1)
		}
	}
	// Hold only execution, allowing real HTTP admissions to enqueue independently.
	s.executor.Lock()
	var once sync.Once
	release := func() { once.Do(s.executor.Unlock) }
	await := func(ctx context.Context, n int, prompt []int, canceled int) error {
		ticker := time.NewTicker(time.Millisecond)
		defer ticker.Stop()
		for {
			s.mu.Lock()
			count := len(s.waiting)
			cancelCount := 0
			var mismatch error
			for _, ln := range s.waiting {
				if !slices.Equal(ln.prompt, prompt) {
					mismatch = fmt.Errorf("queued prompt differs from actual tokenizer IDs: got %v want %v", ln.prompt, prompt)
				}
				if ln.ctx.Err() != nil {
					cancelCount++
				}
			}
			s.mu.Unlock()
			if mismatch != nil {
				return mismatch
			}
			if count >= n && cancelCount >= canceled {
				return nil
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-ticker.C:
			}
		}
	}
	stats := func() (int64, uint64) { return chunks.Load(), s.ChunkedExactPrefillReuses() }
	close := func(ctx context.Context) error { release(); return s.CloseAndWait(ctx) }
	return e, await, release, stats, close
}
