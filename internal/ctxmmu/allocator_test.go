package ctxmmu

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"
)

func TestSharedTokenPool_AdmissionLimit(t *testing.T) {
	pool := NewSharedTokenPool()

	// Verify defaults per Strix Halo APU specification
	if pool.MaxTokens() != DefaultMaxTokens {
		t.Fatalf("expected MaxTokens %d, got %d", DefaultMaxTokens, pool.MaxTokens())
	}
	if pool.TotalGTTBytes() != DefaultTotalGTTBytes {
		t.Fatalf("expected TotalGTTBytes %d, got %d", DefaultTotalGTTBytes, pool.TotalGTTBytes())
	}
	if pool.BytesPerToken() != DefaultPoolBytesPerToken {
		t.Fatalf("expected BytesPerToken %d, got %d", DefaultPoolBytesPerToken, pool.BytesPerToken())
	}
	if pool.CommittedTokens() != 0 || pool.ReservedTokens() != 0 {
		t.Fatalf("expected initial tokens 0, got committed=%d reserved=%d", pool.CommittedTokens(), pool.ReservedTokens())
	}
	if pool.FreeTokens() != DefaultMaxTokens {
		t.Fatalf("expected FreeTokens %d, got %d", DefaultMaxTokens, pool.FreeTokens())
	}
	if pool.AllocatedBytes() != 0 {
		t.Fatalf("expected AllocatedBytes 0, got %d", pool.AllocatedBytes())
	}
	if pool.ActiveStreams() != 0 {
		t.Fatalf("expected ActiveStreams 0, got %d", pool.ActiveStreams())
	}

	// Validate input sanity checks
	if err := pool.Reserve("", 100); !errors.Is(err, ErrEmptyStreamID) {
		t.Fatalf("expected ErrEmptyStreamID, got %v", err)
	}
	if err := pool.Reserve("s1", 0); !errors.Is(err, ErrInvalidTokens) {
		t.Fatalf("expected ErrInvalidTokens for 0, got %v", err)
	}
	if err := pool.Reserve("s1", -5); !errors.Is(err, ErrInvalidTokens) {
		t.Fatalf("expected ErrInvalidTokens for negative, got %v", err)
	}

	// Normal reservation
	if err := pool.Reserve("stream-1", 10000); err != nil {
		t.Fatalf("unexpected error on valid reserve: %v", err)
	}
	if pool.ReservedTokens() != 10000 {
		t.Fatalf("expected 10000 reserved, got %d", pool.ReservedTokens())
	}
	if pool.FreeTokens() != DefaultMaxTokens-10000 {
		t.Fatalf("expected FreeTokens %d, got %d", DefaultMaxTokens-10000, pool.FreeTokens())
	}
	expectedBytes := int64(10000) * DefaultPoolBytesPerToken
	if pool.AllocatedBytes() != expectedBytes {
		t.Fatalf("expected AllocatedBytes %d, got %d", expectedBytes, pool.AllocatedBytes())
	}
	if pool.ActiveStreams() != 1 {
		t.Fatalf("expected ActiveStreams 1, got %d", pool.ActiveStreams())
	}

	// Reserve remaining capacity
	remaining := pool.FreeTokens()
	if err := pool.Reserve("stream-2", remaining); err != nil {
		t.Fatalf("unexpected error reserving remainder: %v", err)
	}
	if pool.FreeTokens() != 0 {
		t.Fatalf("expected 0 free tokens, got %d", pool.FreeTokens())
	}

	// Attempting 1 more token must trigger ErrPoolExhausted
	if err := pool.Reserve("stream-3", 1); !errors.Is(err, ErrPoolExhausted) {
		t.Fatalf("expected ErrPoolExhausted, got %v", err)
	}

	// Test byte limit enforcement independent of maxTokens limit
	byteLimitedPool := NewSharedTokenPoolWithConfig(10000, 100*DefaultPoolBytesPerToken, DefaultPoolBytesPerToken, 1.0)
	if err := byteLimitedPool.Reserve("b1", 100); err != nil {
		t.Fatalf("expected reservation to succeed, got %v", err)
	}
	if err := byteLimitedPool.Reserve("b2", 1); !errors.Is(err, ErrPoolExhausted) {
		t.Fatalf("expected ErrPoolExhausted when GTT byte budget is exceeded, got %v", err)
	}
}

func TestSharedTokenPool_TwoPhaseCommit(t *testing.T) {
	pool := NewSharedTokenPoolWithConfig(10000, 10000*DefaultPoolBytesPerToken, DefaultPoolBytesPerToken, 1.0)
	streamID := "session-abc-turn-1"

	// Phase 1: Reserve output headroom
	headroom := 4096
	if err := pool.Reserve(streamID, headroom); err != nil {
		t.Fatalf("failed to reserve headroom: %v", err)
	}

	if pool.ReservedTokens() != headroom {
		t.Fatalf("expected reserved tokens %d, got %d", headroom, pool.ReservedTokens())
	}
	if pool.CommittedTokens() != 0 {
		t.Fatalf("expected committed tokens 0, got %d", pool.CommittedTokens())
	}

	// Phase 2: Incremental commit as tokens decode
	step1 := 1024
	if err := pool.Commit(streamID, step1); err != nil {
		t.Fatalf("failed to commit step 1: %v", err)
	}
	if pool.CommittedTokens() != step1 {
		t.Fatalf("expected committed tokens %d, got %d", step1, pool.CommittedTokens())
	}
	if pool.ReservedTokens() != headroom-step1 {
		t.Fatalf("expected reserved tokens %d, got %d", headroom-step1, pool.ReservedTokens())
	}

	// Invalid commit attempts
	if err := pool.Commit(streamID, 0); !errors.Is(err, ErrInvalidTokens) {
		t.Fatalf("expected ErrInvalidTokens for 0 tokens, got %v", err)
	}
	if err := pool.Commit("unknown-stream", 100); !errors.Is(err, ErrStreamNotFound) {
		t.Fatalf("expected ErrStreamNotFound, got %v", err)
	}
	if err := pool.Commit(streamID, 5000); !errors.Is(err, ErrExceedsReservation) {
		t.Fatalf("expected ErrExceedsReservation, got %v", err)
	}

	// Commit step 2
	step2 := 2048
	if err := pool.Commit(streamID, step2); err != nil {
		t.Fatalf("failed to commit step 2: %v", err)
	}
	if pool.CommittedTokens() != step1+step2 {
		t.Fatalf("expected committed tokens %d, got %d", step1+step2, pool.CommittedTokens())
	}

	// Commit remainder
	step3 := headroom - step1 - step2
	if err := pool.Commit(streamID, step3); err != nil {
		t.Fatalf("failed to commit step 3: %v", err)
	}
	if pool.ReservedTokens() != 0 {
		t.Fatalf("expected reserved tokens 0, got %d", pool.ReservedTokens())
	}
	if pool.CommittedTokens() != headroom {
		t.Fatalf("expected committed tokens %d, got %d", headroom, pool.CommittedTokens())
	}

	// Release stream entirely
	pool.Release(streamID)
	if pool.CommittedTokens() != 0 || pool.ReservedTokens() != 0 {
		t.Fatalf("expected pool to be clear after release, committed=%d reserved=%d",
			pool.CommittedTokens(), pool.ReservedTokens())
	}
	if pool.ActiveStreams() != 0 {
		t.Fatalf("expected 0 active streams, got %d", pool.ActiveStreams())
	}
}

func TestSharedTokenPool_DynamicDownscaling_HalogenFit(t *testing.T) {
	// Test 1: Env var HALOGEN_KV_POOL_FIT downscaling on initialization
	t.Setenv(EnvHalogenKVPoolFit, "0.5")
	pool := NewSharedTokenPool()

	if pool.DownscaleFactor() != 0.5 {
		t.Fatalf("expected DownscaleFactor 0.5, got %f", pool.DownscaleFactor())
	}

	expectedEffective := int(float64(DefaultMaxTokens) * 0.5)
	if pool.EffectiveMaxTokens() != expectedEffective {
		t.Fatalf("expected EffectiveMaxTokens %d, got %d", expectedEffective, pool.EffectiveMaxTokens())
	}
	if pool.FreeTokens() != expectedEffective {
		t.Fatalf("expected FreeTokens %d, got %d", expectedEffective, pool.FreeTokens())
	}

	// Admission must enforce the downscaled capacity
	if err := pool.Reserve("s1", expectedEffective); err != nil {
		t.Fatalf("failed to reserve exact effective capacity: %v", err)
	}
	if err := pool.Reserve("s2", 1); !errors.Is(err, ErrPoolExhausted) {
		t.Fatalf("expected ErrPoolExhausted beyond downscaled capacity, got %v", err)
	}

	// Test 2: Dynamic FitPoolToAvailableMemory clamping
	t.Setenv(EnvHalogenKVPoolFit, "1.0")
	pool2 := NewSharedTokenPool()

	// Suppose host OS reports only 16 GiB available GTT aperture
	availableGTT := int64(16) * 1024 * 1024 * 1024
	fittedTokens := pool2.FitPoolToAvailableMemory(availableGTT)

	expectedTokens := int(availableGTT / DefaultPoolBytesPerToken)
	if fittedTokens != expectedTokens {
		t.Fatalf("expected fitted tokens %d, got %d", expectedTokens, fittedTokens)
	}
	if pool2.MaxTokens() != expectedTokens {
		t.Fatalf("expected MaxTokens clamped to %d, got %d", expectedTokens, pool2.MaxTokens())
	}
	if pool2.FreeTokens() != expectedTokens {
		t.Fatalf("expected FreeTokens %d, got %d", expectedTokens, pool2.FreeTokens())
	}

	// Test 3: FitPoolToAvailableMemory with downscaleFactor active
	t.Setenv(EnvHalogenKVPoolFit, "0.8")
	pool3 := NewSharedTokenPool()
	availBytes := int64(1000) * DefaultPoolBytesPerToken
	fitted := pool3.FitPoolToAvailableMemory(availBytes)

	// Available tokens = 1000, scaled by 0.8 = 800
	if fitted != 800 {
		t.Fatalf("expected fitted 800 tokens, got %d", fitted)
	}
	if pool3.FreeTokens() != 800 {
		t.Fatalf("expected FreeTokens 800, got %d", pool3.FreeTokens())
	}
	if err := pool3.Reserve("strix-fit", 800); err != nil {
		t.Fatalf("expected reservation of 800 tokens to succeed, got %v", err)
	}
	if err := pool3.Reserve("overflow", 1); !errors.Is(err, ErrPoolExhausted) {
		t.Fatalf("expected ErrPoolExhausted, got %v", err)
	}
}

func TestSharedTokenPool_ImmediateHeadroomReturn(t *testing.T) {
	const poolSize = 5000
	pool := NewSharedTokenPoolWithConfig(poolSize, int64(poolSize)*DefaultPoolBytesPerToken, DefaultPoolBytesPerToken, 1.0)

	stream1 := "stream-stop-early"
	stream2 := "stream-waiting"

	// Stream 1 reserves large generation headroom (4000 tokens)
	if err := pool.Reserve(stream1, 4000); err != nil {
		t.Fatalf("failed to reserve for stream 1: %v", err)
	}

	// Stream 2 requires 2000 tokens; currently only 1000 free tokens exist
	if err := pool.Reserve(stream2, 2000); !errors.Is(err, ErrPoolExhausted) {
		t.Fatalf("expected ErrPoolExhausted for stream 2 before headroom release, got %v", err)
	}

	// Stream 1 finishes generation early after decoding only 250 tokens
	if err := pool.Commit(stream1, 250); err != nil {
		t.Fatalf("failed to commit decoded tokens: %v", err)
	}
	com, res := pool.StreamUsage(stream1)
	if com != 250 || res != 3750 {
		t.Fatalf("expected stream1 com=250 res=3750, got com=%d res=%d", com, res)
	}

	// Terminal finish reason (e.g. stop token or EOS): immediate headroom return
	pool.ReleaseHeadroom(stream1)

	com, res = pool.StreamUsage(stream1)
	if com != 250 || res != 0 {
		t.Fatalf("expected stream1 com=250 res=0 after headroom release, got com=%d res=%d", com, res)
	}
	if pool.ReservedTokens() != 0 {
		t.Fatalf("expected reservedTokens 0, got %d", pool.ReservedTokens())
	}
	if pool.CommittedTokens() != 250 {
		t.Fatalf("expected committedTokens 250, got %d", pool.CommittedTokens())
	}

	// Pool now has 5000 - 250 = 4750 free tokens. Stream 2's reservation must now succeed!
	if err := pool.Reserve(stream2, 2000); err != nil {
		t.Fatalf("expected stream 2 reserve to succeed after headroom release, got %v", err)
	}

	if pool.ActiveStreams() != 2 {
		t.Fatalf("expected 2 active streams, got %d", pool.ActiveStreams())
	}

	// Releasing stream 1 frees its committed tokens
	pool.Release(stream1)
	if pool.ActiveStreams() != 1 {
		t.Fatalf("expected 1 active stream after stream 1 release, got %d", pool.ActiveStreams())
	}
	if pool.CommittedTokens() != 0 {
		t.Fatalf("expected committedTokens 0 after stream 1 release, got %d", pool.CommittedTokens())
	}
}

func TestSharedTokenPool_ConcurrentContention(t *testing.T) {
	const poolSize = 50000
	pool := NewSharedTokenPoolWithConfig(poolSize, int64(poolSize)*DefaultPoolBytesPerToken, DefaultPoolBytesPerToken, 1.0)

	const numWorkers = 50
	const iterationsPerWorker = 20

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		workerID := w
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(workerID)*1000))

			for i := 0; i < iterationsPerWorker; i++ {
				streamID := fmt.Sprintf("stream-w%d-it%d", workerID, i)
				headroom := rng.Intn(100) + 20 // 20 to 119 tokens

				// Attempt reservation
				err := pool.Reserve(streamID, headroom)
				if err != nil {
					if errors.Is(err, ErrPoolExhausted) {
						// Under contention, back off slightly and retry with smaller size
						time.Sleep(time.Duration(rng.Intn(5)) * time.Millisecond)
						continue
					}
					t.Errorf("worker %d unexpected reserve error: %v", workerID, err)
					return
				}

				// Simulate incremental decode / commit
				commitTokens := headroom / 2
				if commitTokens > 0 {
					if err := pool.Commit(streamID, commitTokens); err != nil {
						t.Errorf("worker %d unexpected commit error: %v", workerID, err)
						return
					}
				}

				// Sample telemetry getters concurrently
				_ = pool.FreeTokens()
				_ = pool.CommittedTokens()
				_ = pool.ReservedTokens()
				_ = pool.AllocatedBytes()
				_ = pool.ActiveStreams()

				// Release unused headroom
				pool.ReleaseHeadroom(streamID)

				// Small work simulation
				time.Sleep(time.Duration(rng.Intn(2)) * time.Millisecond)

				// Finish stream lifecycle
				pool.Release(streamID)
			}
		}()
	}

	wg.Wait()

	// Invariant verification: pool must return completely clean after all workers exit
	if pool.CommittedTokens() != 0 {
		t.Fatalf("expected committedTokens 0 after all workers released, got %d", pool.CommittedTokens())
	}
	if pool.ReservedTokens() != 0 {
		t.Fatalf("expected reservedTokens 0 after all workers released, got %d", pool.ReservedTokens())
	}
	if pool.ActiveStreams() != 0 {
		t.Fatalf("expected 0 active streams, got %d", pool.ActiveStreams())
	}
	if pool.FreeTokens() != poolSize {
		t.Fatalf("expected FreeTokens %d, got %d", poolSize, pool.FreeTokens())
	}
	if pool.AllocatedBytes() != 0 {
		t.Fatalf("expected AllocatedBytes 0, got %d", pool.AllocatedBytes())
	}
}
