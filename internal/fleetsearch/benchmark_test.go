package fleetsearch

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

var (
	benchReportSink Report
	benchQuerySink  Query
)

func buildBenchmarkInput(count int, now time.Time, rawQuery string, limit int) Input {
	lifecycle := make([]sessionjournal.Event, count)
	registrations := make([]sessionregistry.Record, count)
	tools := make([]toolproc.Event, count)

	states := []sessionregistry.State{
		sessionregistry.StateActive,
		sessionregistry.StateCompleted,
		sessionregistry.StateLost,
	}

	for i := 0; i < count; i++ {
		sessionID := fmt.Sprintf("session-%04d", i)
		regID := fmt.Sprintf("reg-%04d", i)
		attemptID := fmt.Sprintf("attempt-%04d", i)
		taskName := fmt.Sprintf("migration pipeline %d", i%15)
		created := now.Add(-time.Duration(i+1) * time.Minute)

		lifecycle[i] = sessionjournal.Event{
			Schema: sessionjournal.Schema,
			Kind:   sessionjournal.KindOpen,
			ID:     fmt.Sprintf("life-%04d", i),
			TS:     created.Format(time.RFC3339),
			CWD:    fmt.Sprintf("/work/service-%d", i%8),
			Registration: &sessionjournal.RegistrationCarry{
				RegistrationID: regID,
				SessionID:      sessionID,
				AttemptID:      attemptID,
				TaskID:         taskName,
				State:          string(states[i%len(states)]),
			},
		}

		registrations[i] = sessionregistry.Record{
			Schema:             sessionregistry.Schema,
			RegistrationID:     regID,
			RootRegistrationID: regID,
			AttemptID:          attemptID,
			LaunchKind:         "guard",
			Scope:              []string{fmt.Sprintf("pkg/service-%d/**", i%8)},
			Identity: sessionregistry.Identity{
				Runtime:   "codex",
				SessionID: sessionID,
			},
			State:     states[i%len(states)],
			CreatedAt: created,
		}

		tools[i] = toolproc.Event{
			Kind:             toolproc.EvSpawn,
			CallID:           fmt.Sprintf("tool-%04d", i),
			Session:          sessionID,
			Tool:             fmt.Sprintf("mcp__service_%d__query", i%6),
			AtMS:             created.Add(20 * time.Second).UnixMilli(),
			HeartbeatEveryMS: 120_000,
		}
	}

	q, _ := ParseQuery(rawQuery, limit)
	return Input{
		Query:         q,
		Lifecycle:     lifecycle,
		Registrations: registrations,
		ToolProcesses: tools,
		Coverage: []Coverage{
			{Store: StoreLifecycle, Status: CoverageComplete},
			{Store: StoreRegistration, Status: CoverageComplete},
			{Store: StoreToolProcess, Status: CoverageComplete},
		},
		Now:        now,
		StaleAfter: 30 * time.Minute,
	}
}

func writeBenchmarkFiles(b *testing.B, dir string, count int, now time.Time) (string, string, string) {
	b.Helper()
	lifecyclePath := filepath.Join(dir, "lifecycle.jsonl")
	registrationPath := filepath.Join(dir, "registrations.jsonl")
	toolPath := filepath.Join(dir, "tool-processes.jsonl")

	lifecycleRows := make([]any, count)
	registrationRows := make([]any, count)
	toolRows := make([]any, count)

	states := []sessionregistry.State{
		sessionregistry.StateActive,
		sessionregistry.StateCompleted,
		sessionregistry.StateLost,
	}

	for i := 0; i < count; i++ {
		sessionID := fmt.Sprintf("session-%04d", i)
		regID := fmt.Sprintf("reg-%04d", i)
		attemptID := fmt.Sprintf("attempt-%04d", i)
		created := now.Add(-time.Duration(i+1) * time.Minute)

		lifecycleRows[i] = sessionjournal.Event{
			Schema: sessionjournal.Schema,
			Kind:   sessionjournal.KindOpen,
			ID:     fmt.Sprintf("life-%04d", i),
			TS:     created.Format(time.RFC3339),
			CWD:    fmt.Sprintf("/work/service-%d", i%8),
			Registration: &sessionjournal.RegistrationCarry{
				RegistrationID: regID,
				SessionID:      sessionID,
				AttemptID:      attemptID,
				TaskID:         fmt.Sprintf("migration pipeline %d", i%15),
				State:          string(states[i%len(states)]),
			},
		}

		registrationRows[i] = sessionregistry.Event{
			Schema: sessionregistry.Schema,
			At:     created,
			Record: sessionregistry.Record{
				Schema:             sessionregistry.Schema,
				RegistrationID:     regID,
				RootRegistrationID: regID,
				AttemptID:          attemptID,
				LaunchKind:         "guard",
				Scope:              []string{fmt.Sprintf("pkg/service-%d/**", i%8)},
				Identity: sessionregistry.Identity{
					Runtime:   "codex",
					SessionID: sessionID,
				},
				State:     states[i%len(states)],
				CreatedAt: created,
			},
		}

		toolRows[i] = toolproc.Event{
			Kind:             toolproc.EvSpawn,
			CallID:           fmt.Sprintf("tool-%04d", i),
			Session:          sessionID,
			Tool:             fmt.Sprintf("mcp__service_%d__query", i%6),
			AtMS:             created.Add(20 * time.Second).UnixMilli(),
			HeartbeatEveryMS: 120_000,
		}
	}

	writeBenchmarkJSONLines(b, lifecyclePath, lifecycleRows)
	writeBenchmarkJSONLines(b, registrationPath, registrationRows)
	writeBenchmarkJSONLines(b, toolPath, toolRows)

	return lifecyclePath, registrationPath, toolPath
}

func writeBenchmarkJSONLines(b *testing.B, path string, rows []any) {
	b.Helper()
	f, err := os.Create(path)
	if err != nil {
		b.Fatal(err)
	}
	enc := json.NewEncoder(f)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			_ = f.Close()
			b.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		b.Fatal(err)
	}
}

// BenchmarkFleetSearch measures joining, filtering, and scoring across multi-store session datasets.
func BenchmarkFleetSearch(b *testing.B) {
	now := time.Date(2026, 8, 22, 18, 0, 0, 0, time.UTC)
	input := buildBenchmarkInput(100, now, "migration is:active", 20)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := Search(input)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
		benchReportSink = report
	}
}

// BenchmarkParseQuery measures query tokenization, facet extraction, and deduplication.
func BenchmarkParseQuery(b *testing.B) {
	queries := []string{
		`"confluence migration" is:active store:tool-process limit:10`,
		`deploy is:stale store:lifecycle`,
		`incident-response is:crashed store:registration limit:5`,
		`"complex query with multiple words" is:completed`,
		`bare-search-term`,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q, err := ParseQuery(queries[i%len(queries)], 20)
		if err != nil {
			b.Fatalf("ParseQuery failed: %v", err)
		}
		benchQuerySink = q
	}
}

// BenchmarkSearchScaling measures disjoint-set grouping and ranking as session volume increases.
func BenchmarkSearchScaling(b *testing.B) {
	now := time.Date(2026, 8, 22, 18, 0, 0, 0, time.UTC)
	scales := []int{20, 100, 300}
	for _, scale := range scales {
		b.Run(fmt.Sprintf("sessions_%d", scale), func(b *testing.B) {
			input := buildBenchmarkInput(scale, now, "migration is:active", 50)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				report, err := Search(input)
				if err != nil {
					b.Fatalf("Search failed: %v", err)
				}
				benchReportSink = report
			}
		})
	}
}

// BenchmarkRunDiskStores measures end-to-end file ingestion and search over on-disk store files.
func BenchmarkRunDiskStores(b *testing.B) {
	dir := b.TempDir()
	now := time.Date(2026, 8, 22, 18, 0, 0, 0, time.UTC)
	lifePath, regPath, toolPath := writeBenchmarkFiles(b, dir, 50, now)

	cfg := Config{
		LifecyclePath:    lifePath,
		RegistrationPath: regPath,
		ToolProcessPath:  toolPath,
		Now:              now,
		StaleAfter:       30 * time.Minute,
		Limit:            20,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := Run("migration is:active", cfg)
		if err != nil {
			b.Fatalf("Run failed: %v", err)
		}
		benchReportSink = report
	}
}

// TestBenchmarkFleetSearchSanity validates that benchmark fixtures produce valid matches.
func TestBenchmarkFleetSearchSanity(t *testing.T) {
	now := time.Date(2026, 8, 22, 18, 0, 0, 0, time.UTC)
	input := buildBenchmarkInput(30, now, "migration is:active", 10)

	report, err := Search(input)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if report.TotalMatches == 0 {
		t.Fatalf("expected matches, got 0")
	}
	if len(report.Hits) == 0 {
		t.Fatalf("expected non-empty hits")
	}
	hit := report.Hits[0]
	if hit.SessionID == "" {
		t.Fatalf("expected non-empty session ID")
	}
	if hit.Liveness != LivenessActive {
		t.Fatalf("expected active liveness, got %v", hit.Liveness)
	}
}
