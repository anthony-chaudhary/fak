package memq

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// BenchmarkMemStore_Push measures memory cell ingestion throughput across
// small and medium payloads, evaluating digest computation, extractive descriptor
// building, CAS blob storage, and cell append costs.
func BenchmarkMemStore_Push(b *testing.B) {
	payload256 := []byte(strings.Repeat("a", 256))
	payload4k := []byte(strings.Repeat("b", 4096))

	b.Run("256B", func(b *testing.B) {
		m := NewMemStore()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			m.Add("tool", "tool_result", DurabilitySession, payload256, false)
		}
	})

	b.Run("4KB", func(b *testing.B) {
		m := NewMemStore()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			m.Add("tool", "tool_result", DurabilitySession, payload4k, false)
		}
	})
}

// BenchmarkMemStore_PushPromoted measures cell ingestion with promotion ledger
// recording and metadata auditing.
func BenchmarkMemStore_PushPromoted(b *testing.B) {
	m := NewMemStore()
	payload := []byte("Standing user configuration: prefer concise responses and high concurrency.")
	meta := PromotionMeta{
		Consent:  ConsentExplicit,
		Producer: "user",
		Reason:   "standing user preference",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.AddPromoted("user", "preference", DurabilityDurable, payload, false, meta)
	}
}

// BenchmarkMemStore_Pop measures cell materialization (paging bytes in through
// trust gate and CAS lookup).
func BenchmarkMemStore_Pop(b *testing.B) {
	m := NewMemStore()
	ctx := context.Background()
	cellCount := 100
	ids := make([]string, cellCount)
	for i := 0; i < cellCount; i++ {
		c := m.Add("tool", "tool_result", DurabilityTurn, []byte(fmt.Sprintf("result payload for step %d", i)), false)
		ids[i] = c.ID
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := ids[i%cellCount]
		data, err := m.Materialize(ctx, id)
		if err != nil || len(data) == 0 {
			b.Fatalf("Materialize failed: %v", err)
		}
	}
}

// BenchmarkMemStore_CellsSnapshot measures page table snapshot extraction.
func BenchmarkMemStore_CellsSnapshot(b *testing.B) {
	m := NewMemStore()
	ctx := context.Background()
	for i := 0; i < 200; i++ {
		m.Add("agent", "reasoning", DurabilitySession, []byte(fmt.Sprintf("reasoning trace %d", i)), false)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cells, err := m.Cells(ctx)
		if err != nil || len(cells) != 200 {
			b.Fatalf("Cells failed: %v", err)
		}
	}
}

// BenchmarkDrainage_CASPrune measures CAS orphan blob identification and GC reclamation.
func BenchmarkDrainage_CASPrune(b *testing.B) {
	ctx := context.Background()

	b.Run("DryRun", func(b *testing.B) {
		m := NewMemStore()
		for i := 0; i < 50; i++ {
			m.Add("tool", "tool_result", DurabilityTurn, []byte(fmt.Sprintf("live cell %d", i)), false)
			m.AddOrphanBlob([]byte(fmt.Sprintf("orphan blob %d", i)))
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = m.Prune(ctx, false)
		}
	})

	b.Run("Reclaim", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			m := NewMemStore()
			for j := 0; j < 50; j++ {
				m.Add("tool", "tool_result", DurabilityTurn, []byte(fmt.Sprintf("live cell %d", j)), false)
				m.AddOrphanBlob([]byte(fmt.Sprintf("orphan blob %d", j)))
			}
			b.StartTimer()
			blobs, _, err := m.Prune(ctx, true)
			if err != nil || blobs != 50 {
				b.Fatalf("Prune failed: %v, blobs=%d", err, blobs)
			}
		}
	})
}

// BenchmarkDrainage_CurateEviction measures budget-curated forgetting under a hard
// byte cap with value-ranked eviction and protected-floor enforcement.
func BenchmarkDrainage_CurateEviction(b *testing.B) {
	for _, count := range []int{50, 200} {
		b.Run(fmt.Sprintf("%d_cells", count), func(b *testing.B) {
			cells := make([]Cell, count)
			valMap := make(map[string]int, count)
			for i := 0; i < count; i++ {
				dur := DurabilityTurn
				if i%5 == 0 {
					dur = DurabilityDurable
				}
				cells[i] = Cell{
					ID:         fmt.Sprintf("cell:%d", i),
					Step:       i,
					Role:       "tool",
					Descriptor: fmt.Sprintf("tool result step %d", i),
					Bytes:      100,
					Durability: dur,
				}
				valMap[cells[i].ID] = (i * 7) % 50
			}
			budget := int64(count * 50)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rep := BudgetCurate(cells, budget, valMap)
				if rep.Reason != CurateReason {
					b.Fatalf("unexpected curate reason: %s", rep.Reason)
				}
			}
		})
	}
}

// BenchmarkDrainage_Tombstone measures negative-only suppression marking.
func BenchmarkDrainage_Tombstone(b *testing.B) {
	ctx := context.Background()
	m := NewMemStore()
	cellCount := 100
	ids := make([]string, cellCount)
	for i := 0; i < cellCount; i++ {
		c := m.Add("tool", "tool_result", DurabilityTurn, []byte(fmt.Sprintf("item %d", i)), false)
		ids[i] = c.ID
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := ids[i%cellCount]
		_, _ = m.Tombstone(ctx, id, "benchmark tombstone", "bench")
	}
}

// BenchmarkCapacity_BudgetOverflow measures pipeline execution when the working
// set exceeds capacity, triggering cutline splitting and typed MEMORY_INDEX_OVERFLOW emission.
func BenchmarkCapacity_BudgetOverflow(b *testing.B) {
	ctx := context.Background()
	count := 100
	m := NewMemStore()
	for i := 0; i < count; i++ {
		m.Add("tool", "tool_result", DurabilitySession, []byte(fmt.Sprintf("session payload entry %d with sufficient text", i)), false)
	}
	q := Query{
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpBudget, Bytes: 1000},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Run(ctx, m, q, Caps{})
		if err != nil {
			b.Fatalf("Run failed: %v", err)
		}
		if res.Overflow == nil {
			b.Fatalf("expected overflow report")
		}
	}
}

// BenchmarkCapacity_AntiStarvation measures anti-starvation pass tracking (StarveK)
// across pipeline cutline evaluations.
func BenchmarkCapacity_AntiStarvation(b *testing.B) {
	ctx := context.Background()
	count := 40
	m := NewMemStore()
	for i := 0; i < count; i++ {
		m.Add("tool", "tool_result", DurabilitySession, []byte(fmt.Sprintf("starve test payload entry %d", i)), false)
	}
	q := Query{
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpLimit, K: 10, StarveK: 2},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Run(ctx, m, q, Caps{})
		if err != nil {
			b.Fatalf("Run failed: %v", err)
		}
		if len(res.Working) == 0 {
			b.Fatalf("expected working cells")
		}
	}
}

// BenchmarkCapacity_ProtectedFloor measures capacity enforcement with protected floors
// separating durable/recent cells from the ephemeral remainder.
func BenchmarkCapacity_ProtectedFloor(b *testing.B) {
	ctx := context.Background()
	count := 60
	m := NewMemStore()
	for i := 0; i < count; i++ {
		dur := DurabilityTurn
		if i%4 == 0 {
			dur = DurabilityDurable
		}
		m.Add("tool", "tool_result", dur, []byte(fmt.Sprintf("payload item %d", i)), false)
	}
	q := Query{
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpBudget, Bytes: 1500, Protect: true, Recent: 5},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Run(ctx, m, q, Caps{})
		if err != nil {
			b.Fatalf("Run failed: %v", err)
		}
		if len(res.Working) == 0 {
			b.Fatalf("expected working cells")
		}
	}
}

// BenchmarkCapacity_MergeOnEvict measures merge-on-evict similarity folding when
// memory cells fall below budget capacity.
func BenchmarkCapacity_MergeOnEvict(b *testing.B) {
	ctx := context.Background()
	count := 30
	m := NewMemStore()
	for i := 0; i < count; i++ {
		m.Add("tool", "tool_result", DurabilitySession, []byte(fmt.Sprintf("refund payment transaction record %d", i)), false)
	}
	q := Query{
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpBudget, Bytes: 500, MergeFloor: 0.1},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Run(ctx, m, q, Caps{})
		if err != nil {
			b.Fatalf("Run failed: %v", err)
		}
		if len(res.Working) == 0 {
			b.Fatalf("expected working cells")
		}
	}
}

// BenchmarkPipeline_ProductionSelect measures an end-to-end memory retrieval pipeline:
// Scan -> Filter -> Rank (Relevance) -> Limit -> Render.
func BenchmarkPipeline_ProductionSelect(b *testing.B) {
	ctx := context.Background()
	m := NewDemoStore()
	q := Query{
		Intent: "refund fee",
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpFilter, Pred: &Pred{Op: PredEq, Field: "sealed", Value: "false"}},
			{Kind: OpRank, By: RankRelevance, Desc: true},
			{Kind: OpLimit, K: 5},
			{Kind: OpRender},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Run(ctx, m, q, Caps{})
		if err != nil {
			b.Fatalf("Run failed: %v", err)
		}
		if len(res.Rendered) == 0 {
			b.Fatalf("expected rendered items")
		}
	}
}

// BenchmarkAdmit_WriteAdjudication measures deny-by-structure write gating against
// legitimate facts and common junk patterns (transient errors, oversize copies).
func BenchmarkAdmit_WriteAdjudication(b *testing.B) {
	legitFact := []byte("The database connection pool timeout is configured to 30 seconds across workers.")
	junkTransient := []byte("Connection reset by peer while trying to reach remote endpoint.")
	oversize := []byte(strings.Repeat("Large verbatim document text ", 700))

	b.Run("LegitimateFact", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := AdjudicateMemoryWrite(legitFact)
			if !v.Admit || v.Reason != AdmitOK {
				b.Fatalf("expected AdmitOK, got %+v", v)
			}
		}
	})

	b.Run("TransientError", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := AdjudicateMemoryWrite(junkTransient)
			if v.Reason != RefuseTransientError {
				b.Fatalf("expected RefuseTransientError, got %+v", v)
			}
		}
	})

	b.Run("OversizeVerbatim", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := AdjudicateMemoryWrite(oversize)
			if v.Reason != RefuseOversizeVerbatim {
				b.Fatalf("expected RefuseOversizeVerbatim, got %+v", v)
			}
		}
	})
}

// BenchmarkExplain_Plan measures static query explanation and validation.
func BenchmarkExplain_Plan(b *testing.B) {
	q := Query{
		Intent: "refund payment fee",
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpFilter, Pred: &Pred{Op: PredEq, Field: "sealed", Value: "false"}},
			{Kind: OpRank, By: RankRelevance, Desc: true},
			{Kind: OpBudget, Bytes: 2000, Protect: true, Recent: 3},
			{Kind: OpRender},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan := Explain(q)
		if !plan.Valid {
			b.Fatalf("expected valid plan: %s", plan.Error)
		}
	}
}
