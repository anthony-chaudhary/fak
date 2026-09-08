package pacing

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

// BenchmarkTokenBucket_Allow measures rate-limiting evaluation throughput with active limits.
func BenchmarkTokenBucket_Allow(b *testing.B) {
	tb := NewTokenBucket(TokenBucketConfig{
		TPMLimit: 1_000_000_000_000,
		RPMLimit: 1_000_000_000_000,
	})

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if !tb.Allow(1) {
			b.Fatal("unexpected rate limit denial")
		}
	}
}

// BenchmarkTokenBucket_AllowUnlimited measures rate-limiting evaluation throughput when limits are disabled.
func BenchmarkTokenBucket_AllowUnlimited(b *testing.B) {
	tb := NewTokenBucket(TokenBucketConfig{
		TPMLimit: 0,
		RPMLimit: 0,
	})

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if !tb.Allow(10) {
			b.Fatal("unexpected rate limit denial")
		}
	}
}

// BenchmarkTokenBucket_Reserve measures reservation throughput under available capacity.
func BenchmarkTokenBucket_Reserve(b *testing.B) {
	tb := NewTokenBucket(TokenBucketConfig{
		TPMLimit: 1_000_000_000_000,
		RPMLimit: 1_000_000_000_000,
	})
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := tb.Reserve(ctx, 10); err != nil {
			b.Fatalf("Reserve failed: %v", err)
		}
	}
}

// BenchmarkTokenBucket_Stats measures snapshot statistics generation throughput.
func BenchmarkTokenBucket_Stats(b *testing.B) {
	tb := NewTokenBucket(TokenBucketConfig{
		TPMLimit: 60000,
		RPMLimit: 1000,
	})

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tb.Stats()
	}
}

// BenchmarkGovernor_AcquireYieldResume measures a complete acquire, yield, and priority resume cycle.
func BenchmarkGovernor_AcquireYieldResume(b *testing.B) {
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: 4,
	})
	ctx := context.Background()
	ticketID := "ticket-bench"

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		lease, err := gov.AcquireInference(ctx, ticketID, 100)
		if err != nil {
			b.Fatalf("AcquireInference failed: %v", err)
		}
		_ = lease
		if err := gov.YieldInference(ticketID); err != nil {
			b.Fatalf("YieldInference failed: %v", err)
		}
		lease, err = gov.ResumeInference(ctx, ticketID, 100)
		if err != nil {
			b.Fatalf("ResumeInference failed: %v", err)
		}
		_ = lease
		if err := gov.YieldInference(ticketID); err != nil {
			b.Fatalf("YieldInference failed: %v", err)
		}
	}
}

// BenchmarkGovernor_AcquireYield measures single-worker inference slot acquisition and yield cycle.
func BenchmarkGovernor_AcquireYield(b *testing.B) {
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: 4,
	})
	ctx := context.Background()
	ticketID := "ticket-bench"

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		lease, err := gov.AcquireInference(ctx, ticketID, 100)
		if err != nil {
			b.Fatalf("AcquireInference failed: %v", err)
		}
		_ = lease
		if err := gov.YieldInference(ticketID); err != nil {
			b.Fatalf("YieldInference failed: %v", err)
		}
	}
}

// BenchmarkGovernor_AcquireRelease measures single-worker inference slot acquisition and release cycle.
func BenchmarkGovernor_AcquireRelease(b *testing.B) {
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: 4,
	})
	ctx := context.Background()
	ticketID := "ticket-bench"

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		lease, err := gov.AcquireInference(ctx, ticketID, 100)
		if err != nil {
			b.Fatalf("AcquireInference failed: %v", err)
		}
		_ = lease
		gov.Release(ticketID)
	}
}

// BenchmarkGovernor_ResumeYield measures priority queue resumption and yield throughput.
func BenchmarkGovernor_ResumeYield(b *testing.B) {
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: 4,
	})
	ctx := context.Background()
	ticketID := "ticket-bench"

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		lease, err := gov.ResumeInference(ctx, ticketID, 100)
		if err != nil {
			b.Fatalf("ResumeInference failed: %v", err)
		}
		_ = lease
		if err := gov.YieldInference(ticketID); err != nil {
			b.Fatalf("YieldInference failed: %v", err)
		}
	}
}

// BenchmarkGovernor_ReportFeedback measures adaptive capacity feedback signal processing throughput.
func BenchmarkGovernor_ReportFeedback(b *testing.B) {
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: 8,
		MinInferenceSlots: 1,
	})
	signals := []FeedbackSignal{SignalSuccess, SignalHighLatency, SignalRateLimit}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		gov.ReportFeedback(signals[i%len(signals)])
	}
}

// BenchmarkGovernor_Stats measures governor state snapshot generation throughput.
func BenchmarkGovernor_Stats(b *testing.B) {
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: 4,
	})

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = gov.Stats()
	}
}

// BenchmarkGovernor_ContendedAcquireYield measures multiplexed slot competition across concurrent workers.
func BenchmarkGovernor_ContendedAcquireYield(b *testing.B) {
	gov := NewGovernor(GovernorConfig{
		MaxInferenceSlots: 4,
	})
	ctx := context.Background()
	const workers = 8
	var wg sync.WaitGroup
	tickets := make([]string, workers)
	for i := 0; i < workers; i++ {
		tickets[i] = fmt.Sprintf("ticket-worker-%d", i)
	}
	workCh := make(chan struct{})

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(ticket string) {
			defer wg.Done()
			for range workCh {
				lease, err := gov.AcquireInference(ctx, ticket, 100)
				if err == nil {
					_ = gov.YieldInference(ticket)
					_ = lease
				}
			}
		}(tickets[w])
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		workCh <- struct{}{}
	}
	close(workCh)
	wg.Wait()
}
