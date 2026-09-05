// Copyright 2026 The fak Authors
// Licensed under the Apache License, Version 2.0.

package compactcohere

import (
	"fmt"
	"testing"
	"time"
)

var (
	benchSinkEvent     PrefixEvent
	benchSinkDecision  Decision
	benchSinkPrune     PruneResult
	benchSinkSelection ToolResultSelection
	benchSinkToolRes   ToolResult
)

// BenchmarkClassify_Stable measures prefix event classification on the happy-path
// warm unchanged prefix.
func BenchmarkClassify_Stable(b *testing.B) {
	prev := TurnObservation{InboundPrefixDigest: "prefix-hash-001"}
	cur := TurnObservation{
		InboundPrefixDigest: "prefix-hash-001",
		IdleSinceLastTurn:   30 * time.Second,
		CacheReadTokens:     50000,
	}
	const ttl = 5 * time.Minute

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkEvent = Classify(prev, cur, ttl)
	}
}

// BenchmarkClassify_FakCut measures attribution when fak's cache-preserving cut fired.
func BenchmarkClassify_FakCut(b *testing.B) {
	prev := TurnObservation{InboundPrefixDigest: "prefix-hash-001"}
	cur := TurnObservation{
		InboundPrefixDigest: "prefix-hash-001",
		FakCompactFired:     true,
		IdleSinceLastTurn:   15 * time.Second,
		CacheReadTokens:     45000,
	}
	const ttl = 5 * time.Minute

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkEvent = Classify(prev, cur, ttl)
	}
}

// BenchmarkClassify_HarnessRewrite measures attribution when the harness changed
// the inbound protected prefix digest.
func BenchmarkClassify_HarnessRewrite(b *testing.B) {
	prev := TurnObservation{InboundPrefixDigest: "prefix-hash-001"}
	cur := TurnObservation{
		InboundPrefixDigest: "prefix-hash-002",
		IdleSinceLastTurn:   10 * time.Second,
	}
	const ttl = 5 * time.Minute

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkEvent = Classify(prev, cur, ttl)
	}
}

// BenchmarkClassify_ColdTTL measures classification when prompt cache expired by idle time.
func BenchmarkClassify_ColdTTL(b *testing.B) {
	prev := TurnObservation{InboundPrefixDigest: "prefix-hash-001"}
	cur := TurnObservation{
		InboundPrefixDigest: "prefix-hash-001",
		IdleSinceLastTurn:   6 * time.Minute,
	}
	const ttl = 5 * time.Minute

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkEvent = Classify(prev, cur, ttl)
	}
}

// BenchmarkClassify_WorldBreak measures classification when fak injected a deliberate world break.
func BenchmarkClassify_WorldBreak(b *testing.B) {
	prev := TurnObservation{InboundPrefixDigest: "prefix-hash-001"}
	cur := TurnObservation{
		InboundPrefixDigest: "prefix-hash-002",
		FakWorldBreak:       true,
		FakCompactFired:     true,
	}
	const ttl = 5 * time.Minute

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkEvent = Classify(prev, cur, ttl)
	}
}

// BenchmarkCoordinator_Observe_SteadyState measures rolling per-turn state tracking
// during normal ongoing multi-turn conversations within budget.
func BenchmarkCoordinator_Observe_SteadyState(b *testing.B) {
	coord := New(DefaultProviderCacheTTL)
	turn := TurnObservation{
		InboundPrefixDigest: "stable-prefix-1",
		CacheReadTokens:     45000,
		InputTokens:         1200,
		IdleSinceLastTurn:   20 * time.Second,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkDecision = coord.Observe(turn)
	}
}

// BenchmarkCoordinator_Observe_CeilingAndEscalation measures coordinator tracking
// through ceiling breach, harness rewrite, and reset escalation cycles.
func BenchmarkCoordinator_Observe_CeilingAndEscalation(b *testing.B) {
	turns := []TurnObservation{
		{InboundPrefixDigest: "p1", CacheReadTokens: 170000},
		{InboundPrefixDigest: "p1", CacheReadTokens: 175000},
		{InboundPrefixDigest: "p1", CacheReadTokens: 180000},
		{InboundPrefixDigest: "p2", CacheReadTokens: 25000},
		{InboundPrefixDigest: "p2", CacheReadTokens: 165000},
		{InboundPrefixDigest: "p3", CacheReadTokens: 30000},
		{InboundPrefixDigest: "p3", CacheReadTokens: 170000},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		coord := NewConfig(Config{
			ResidentCeiling:        160000,
			CeilingStreakToYield:   3,
			NonHoldRewritesToReset: 2,
		})
		for _, t := range turns {
			benchSinkDecision = coord.Observe(t)
		}
	}
}

// BenchmarkCheckPrune_Allowed measures prune verification on a profitable span
// with a balanced transcript.
func BenchmarkCheckPrune_Allowed(b *testing.B) {
	transcript := []TranscriptItem{
		{Kind: TranscriptOther},
		{Kind: TranscriptToolCall, CallID: "call-1"},
		{Kind: TranscriptToolResult, CallID: "call-1"},
		{Kind: TranscriptOther},
		{Kind: TranscriptToolCall, CallID: "call-2"},
		{Kind: TranscriptToolResult, CallID: "call-2"},
		{Kind: TranscriptOther},
	}
	const cut = 3 // cuts after call-1 and its result

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPrune = CheckPrune(120, transcript, cut)
	}
}

// BenchmarkCheckPrune_BelowMinimum measures the fast arithmetic rejection of
// unprofitable prune spans.
func BenchmarkCheckPrune_BelowMinimum(b *testing.B) {
	transcript := []TranscriptItem{
		{Kind: TranscriptOther},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPrune = CheckPrune(PlaceholderTokenCost-1, transcript, 0)
	}
}

// BenchmarkCheckPrune_OrphanCheck measures structural orphan scanning across
// a moderately sized transcript (40 items).
func BenchmarkCheckPrune_OrphanCheck(b *testing.B) {
	transcript := make([]TranscriptItem, 40)
	for i := 0; i < 20; i++ {
		id := fmt.Sprintf("call-%d", i)
		transcript[i*2] = TranscriptItem{Kind: TranscriptToolCall, CallID: id}
		transcript[i*2+1] = TranscriptItem{Kind: TranscriptToolResult, CallID: id}
	}
	const cut = 20

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPrune = CheckPrune(500, transcript, cut)
	}
}

// BenchmarkCoordinator_CheckPrune measures coordinator CheckPrune which wraps
// CheckPrune and maintains RefusalCounters.
func BenchmarkCoordinator_CheckPrune(b *testing.B) {
	coord := New(DefaultProviderCacheTTL)
	transcript := []TranscriptItem{
		{Kind: TranscriptToolCall, CallID: "call-1"},
		{Kind: TranscriptToolResult, CallID: "call-1"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPrune = coord.CheckPrune(50, transcript, 2)
	}
}

// BenchmarkSelectToolResults_CacheSafe measures sorting and selecting tool results
// by producer preference on a cache-safe prefix event.
func BenchmarkSelectToolResults_CacheSafe(b *testing.B) {
	candidates := make([]ToolResultEntry, 30)
	for i := range candidates {
		candidates[i] = ToolResultEntry{
			ID:       fmt.Sprintf("tool-%d", i),
			Result:   ToolResult{Useless: i%3 == 0},
			AgeTurns: uint64(i),
			Tokens:   150 + i*10,
		}
	}
	const limit = 10

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkSelection = SelectToolResults(EventStable, candidates, limit)
	}
}

// BenchmarkSelectToolResults_CacheUnsafe measures the early return when the prefix
// event does not permit tool result pruning.
func BenchmarkSelectToolResults_CacheUnsafe(b *testing.B) {
	candidates := make([]ToolResultEntry, 20)
	for i := range candidates {
		candidates[i] = ToolResultEntry{
			ID:     fmt.Sprintf("tool-%d", i),
			Result: ToolResult{Useless: true},
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkSelection = SelectToolResults(EventHarnessRewrite, candidates, 5)
	}
}

// BenchmarkCoordinator_SelectToolResults measures coordinator SelectToolResults
// with prune counter aggregation.
func BenchmarkCoordinator_SelectToolResults(b *testing.B) {
	coord := New(DefaultProviderCacheTTL)
	candidates := []ToolResultEntry{
		{ID: "t1", Result: ToolResult{Useless: true}, Tokens: 200},
		{ID: "t2", Result: ToolResult{Useless: false}, Tokens: 300},
		{ID: "t3", Result: ToolResult{Useless: true}, Tokens: 100},
		{ID: "t4", Result: ToolResult{Useless: false}, Tokens: 400},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkSelection = coord.SelectToolResults(EventFakCut, candidates, 2)
	}
}

// BenchmarkProducerResults measures the constructor helpers that mark empty results useless.
func BenchmarkProducerResults(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkToolRes = ProduceSearchResult(0, true)
		benchSinkToolRes = ProduceStatusResult(0, true)
		benchSinkToolRes = ProduceReadResult(0, 0, true)
	}
}

// BenchmarkLiveReplaySession measures replaying an end-to-end multi-turn session
// modeled on production telemetry (ceiling breach -> yield -> rewrite -> reset escalation).
func BenchmarkLiveReplaySession(b *testing.B) {
	const (
		preCompactResident  = int64(162356)
		postCompactResident = int64(18667)
	)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		coord := NewConfig(Config{
			ResidentCeiling:        20000,
			CeilingStreakToYield:   2,
			NonHoldRewritesToReset: 2,
		})

		// Turn 1 & 2: Over ceiling streak -> PostureAllow
		coord.Observe(TurnObservation{InboundPrefixDigest: "p1", CacheReadTokens: preCompactResident})
		coord.Observe(TurnObservation{InboundPrefixDigest: "p1", CacheReadTokens: preCompactResident})

		// Compaction 1: Harness rewrites -> post-compact resident, then climbs back over ceiling
		coord.Observe(TurnObservation{InboundPrefixDigest: "p2", CacheReadTokens: postCompactResident})
		coord.Observe(TurnObservation{InboundPrefixDigest: "p2", CacheReadTokens: preCompactResident})

		// Compaction 2: Second non-holding rewrite -> escalates to ActionRecommendReset
		coord.Observe(TurnObservation{InboundPrefixDigest: "p3", CacheReadTokens: postCompactResident})
		benchSinkDecision = coord.Observe(TurnObservation{InboundPrefixDigest: "p3", CacheReadTokens: preCompactResident})
	}
}
