package agentquery

import (
	"fmt"
	"testing"
	"time"
)

func makeBenchmarkRows(count int, baseTime time.Time) []Row {
	lanes := []string{"cmd", "docs", "gateway", "orchestrator", "eval"}
	states := []string{"LIVE", "CLOSED", "STALE", "CRASHED"}
	owners := []string{"alice", "bob", "carol", "dave"}
	hosts := []string{"host-01", "host-02", "host-03"}
	models := []string{"claude-3-5-sonnet", "gpt-4o", "gemini-1.5-pro"}
	providers := []string{"anthropic", "openai", "google"}

	rows := make([]Row, count)
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("agent-%06d", i)
		lane := lanes[i%len(lanes)]
		state := states[i%len(states)]
		owner := owners[i%len(owners)]
		host := hosts[i%len(hosts)]
		model := models[i%len(models)]
		provider := providers[i%len(providers)]
		elapsed := int64((i % 1000) * 150)
		turns := int64(10 + (i % 50))
		toolCalls := int64(20 + (i % 100))
		cost := float64(i%100) * 0.05
		liveness := state
		if state == "STALE" || state == "CRASHED" {
			liveness = "CLOSED"
		}
		started := baseTime.Add(-time.Duration(i%72) * time.Hour).Format(time.RFC3339)
		progress := baseTime.Add(-time.Duration(i%30) * time.Minute).Format(time.RFC3339)
		observed := baseTime.Format(time.RFC3339)

		rows[i] = Row{
			AgentID:          id,
			LogicalSessionID: fmt.Sprintf("session-%06d", i/2),
			Lane:             &lane,
			Owner:            &owner,
			Host:             &host,
			State:            state,
			Liveness:         liveness,
			StartedAt:        &started,
			LastProgressAt:   &progress,
			ObservedAt:       observed,
			ElapsedMS:        &elapsed,
			Model:            &model,
			Provider:         &provider,
			Turns:            &turns,
			ToolCalls:        &toolCalls,
			Cost:             &cost,
			Source:           "history",
			SourceVersion:    "v1",
		}
	}
	return rows
}

// BenchmarkParseQuery measures parsing and validation of aggregate query text into query plans.
func BenchmarkParseQuery(b *testing.B) {
	queries := map[string]string{
		"Standard":        "SELECT lane, state, count(*) AS agents, max(elapsed_ms) AS max_elapsed_ms FROM agents WHERE started_at >= now()-interval '7 day' GROUP BY lane,state ORDER BY max_elapsed_ms DESC",
		"FullAggregates":  "SELECT lane,state,count(*) AS agents,min(elapsed_ms) AS min_elapsed_ms,max(elapsed_ms) AS max_elapsed_ms,sum(elapsed_ms) AS sum_elapsed_ms,avg(elapsed_ms) AS avg_elapsed_ms FROM agents WHERE started_at >= now()-interval '7 day' GROUP BY lane,state ORDER BY max_elapsed_ms DESC",
		"ObservedAtAlias": "SELECT lane,state,count(*) AS agents,max(elapsed_ms) AS max_elapsed_ms FROM agents WHERE observed_at >= now()-interval '14 days' GROUP BY lane,state ORDER BY max_elapsed_ms DESC",
	}

	for name, q := range queries {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				plan, err := ParseQuery(q)
				if err != nil {
					b.Fatalf("unexpected parse error: %v", err)
				}
				if plan.Schema != QueryPlanSchema {
					b.Fatalf("unexpected plan schema: %s", plan.Schema)
				}
			}
		})
	}
}

// BenchmarkQueryMatch measures predicate evaluation across row fields in list plans.
func BenchmarkQueryMatch(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	rows := makeBenchmarkRows(100, now)

	lane := "gateway"
	owner := "alice"
	after := now.Add(-48 * time.Hour).Format(time.RFC3339)
	before := now.Format(time.RFC3339)

	plan := ListPlan{
		Schema:        ListPlanSchema,
		State:         "LIVE",
		Liveness:      "LIVE",
		Lane:          lane,
		Owner:         owner,
		StartedAfter:  &after,
		StartedBefore: &before,
		OrderBy:       "elapsed_desc",
		Limit:         50,
	}

	b.ReportAllocs()
	b.ResetTimer()
	matched := 0
	for i := 0; i < b.N; i++ {
		r := rows[i%len(rows)]
		if rowMatches(r, plan) {
			matched++
		}
	}
	_ = matched
}

// BenchmarkApplyListPlan measures filtering, sorting, and pagination over fleet rows.
func BenchmarkApplyListPlan(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	rows := makeBenchmarkRows(1000, now)

	sortOrders := []string{
		"elapsed_desc",
		"progress_age_desc",
		"cost_desc",
		"identity_asc",
	}

	for _, order := range sortOrders {
		b.Run(order, func(b *testing.B) {
			plan := ListPlan{
				Schema:   ListPlanSchema,
				State:    "LIVE",
				Liveness: "LIVE",
				Lane:     "gateway",
				OrderBy:  order,
				Limit:    50,
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, _, err := ApplyListPlan(rows, plan, now)
				if err != nil {
					b.Fatalf("ApplyListPlan failed: %v", err)
				}
				if len(out) == 0 {
					b.Fatal("expected matching rows")
				}
			}
		})
	}
}

// BenchmarkGroupLaneState measures multi-dimensional aggregation across lanes and states.
func BenchmarkGroupLaneState(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	rows := makeBenchmarkRows(1000, now)
	since := now.Add(-7 * 24 * time.Hour)

	b.Run("BaseAggregates", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := GroupLaneState(rows, since, now, "history", nil)
			if len(res.Rows) == 0 {
				b.Fatal("expected grouped rows")
			}
		}
	})

	b.Run("FullAggregates", func(b *testing.B) {
		plan, err := GroupedPlan(7 * 24 * time.Hour)
		if err != nil {
			b.Fatal(err)
		}
		plan.Aggregates = append([]string(nil), fullAggregates...)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := GroupLaneStatePlan(rows, plan, now, "history", nil)
			if len(res.Rows) == 0 {
				b.Fatal("expected grouped rows")
			}
		}
	})
}

// BenchmarkUnion measures deduplication, active-only filtering, and ordering of combined live and history rows.
func BenchmarkUnion(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	history := makeBenchmarkRows(800, now)

	live := make([]Row, 200)
	for i := 0; i < 200; i++ {
		id := fmt.Sprintf("agent-%06d", i*2)
		elapsed := int64(50000 + i)
		live[i] = Row{
			AgentID:          id,
			LogicalSessionID: fmt.Sprintf("live-session-%06d", i),
			State:            "LIVE",
			Liveness:         "LIVE",
			Source:           "live",
			ObservedAt:       now.Format(time.RFC3339),
			ElapsedMS:        &elapsed,
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Union(live, history, "union", false, 100, now)
		if len(res.Rows) == 0 {
			b.Fatal("expected union rows")
		}
	}
}

// BenchmarkValidateResult measures schema and structural validation of query result sets.
func BenchmarkValidateResult(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	rows := makeBenchmarkRows(100, now)
	res := Result{
		Metadata: Metadata{
			Schema:     Schema,
			Source:     "history",
			ObservedAt: now.Format(time.RFC3339),
			Freshness:  "recorded",
			Limit:      100,
		},
		Rows: rows,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ValidateResult(res); err != nil {
			b.Fatalf("validation failed: %v", err)
		}
	}
}
