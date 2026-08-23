package fleetsearch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

var fixtureNow = time.Date(2026, 8, 22, 18, 0, 0, 0, time.UTC)

func TestParseQuerySeparatesTermsFacetsAndLimit(t *testing.T) {
	q, err := ParseQuery(`"confluence migration" is:active store:tool-process limit:1`, 25)
	if err != nil {
		t.Fatalf("ParseQuery: %v", err)
	}
	if len(q.Terms) != 1 || q.Terms[0] != "confluence migration" {
		t.Fatalf("terms = %#v", q.Terms)
	}
	if len(q.Liveness) != 1 || q.Liveness[0] != LivenessActive {
		t.Fatalf("liveness = %#v", q.Liveness)
	}
	if len(q.Stores) != 1 || q.Stores[0] != StoreToolProcess {
		t.Fatalf("stores = %#v", q.Stores)
	}
	if q.Limit != 1 {
		t.Fatalf("limit = %d, want 1", q.Limit)
	}

	if _, err := ParseQuery("confluence is:maybe", 10); err == nil {
		t.Fatal("unknown liveness facet must be rejected")
	}
	if _, err := ParseQuery("confluence limit:0", 10); err == nil {
		t.Fatal("unsafe limit must be rejected")
	}
}

func TestSearchJoinsThreeStoreIdentitiesAndReturnsSoleMatch(t *testing.T) {
	report, err := Search(Input{
		Query: mustQuery(t, "confluence is:active", 10),
		Now:   fixtureNow,
		Lifecycle: []sessionjournal.Event{{
			Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "life-1",
			TS: fixtureNow.Add(-2 * time.Minute).Format(time.RFC3339), CWD: "/work/confluence-export",
			Registration: &sessionjournal.RegistrationCarry{
				RegistrationID: "reg-1", SessionID: "session-1", AttemptID: "attempt-1",
				TaskID: "migrate confluence space", State: "active",
			},
		}},
		Registrations: []sessionregistry.Record{{
			Schema: sessionregistry.Schema, RegistrationID: "reg-1", RootRegistrationID: "reg-1",
			AttemptID: "attempt-1", LaunchKind: "guard", Scope: []string{"docs/confluence/**"},
			Identity: sessionregistry.Identity{Runtime: "codex", SessionID: "session-1"},
			State:    sessionregistry.StateActive, CreatedAt: fixtureNow.Add(-2 * time.Minute),
		}},
		ToolProcesses: []toolproc.Event{{
			Kind: toolproc.EvSpawn, CallID: "tool-1", Session: "session-1",
			Tool: "mcp__confluence__search", AtMS: fixtureNow.Add(-time.Minute).UnixMilli(),
			HeartbeatEveryMS: 120_000,
		}},
		Coverage: completeCoverage(),
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if report.Verdict != VerdictSoleMatch || report.TotalMatches != 1 || len(report.Hits) != 1 {
		t.Fatalf("report = %+v", report)
	}
	hit := report.Hits[0]
	if hit.SessionID != "session-1" || hit.Liveness != LivenessActive {
		t.Fatalf("joined hit = %+v", hit)
	}
	if !contains(hit.Identifiers, "life-1") || !contains(hit.Identifiers, "reg-1") || !contains(hit.Identifiers, "attempt-1") {
		t.Fatalf("joined identities = %#v", hit.Identifiers)
	}
	if !contains(hit.Tools, "mcp__confluence__search") || len(hit.Evidence) != 3 {
		t.Fatalf("joined evidence = %+v", hit)
	}
}

func TestSearchFacetsAndLimitCannotManufactureUniqueness(t *testing.T) {
	lifecycle := []sessionjournal.Event{
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "active-a", TS: fixtureNow.Add(-time.Minute).Format(time.RFC3339), CWD: "deploy confluence"},
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "active-b", TS: fixtureNow.Add(-2 * time.Minute).Format(time.RFC3339), CWD: "deploy confluence"},
		{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "stale-c", TS: fixtureNow.Add(-2 * time.Hour).Format(time.RFC3339), CWD: "deploy confluence"},
	}
	report, err := Search(Input{
		Query:      mustQuery(t, "deploy is:active limit:1", 10),
		Now:        fixtureNow,
		StaleAfter: 30 * time.Minute,
		Lifecycle:  lifecycle,
		Coverage:   completeCoverage(),
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if report.Verdict != VerdictMatches || report.TotalMatches != 2 || len(report.Hits) != 1 {
		t.Fatalf("limited multi-match = %+v", report)
	}
	if report.Hits[0].Liveness != LivenessActive {
		t.Fatalf("is:active returned %+v", report.Hits[0])
	}

	stale, err := Search(Input{
		Query:      mustQuery(t, "deploy is:stale", 10),
		Now:        fixtureNow,
		StaleAfter: 30 * time.Minute,
		Lifecycle:  lifecycle,
		Coverage:   completeCoverage(),
	})
	if err != nil || stale.Verdict != VerdictSoleMatch || stale.Hits[0].SessionID != "stale-c" {
		t.Fatalf("stale facet = report %+v err %v", stale, err)
	}
}

func TestSearchLivenessFacetsCoverOperationalStates(t *testing.T) {
	terminal := fixtureNow.Add(-time.Minute)
	input := Input{
		Now: fixtureNow, StaleAfter: 30 * time.Minute, Coverage: completeCoverage(),
		Lifecycle: []sessionjournal.Event{{
			Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "stale-session",
			TS: fixtureNow.Add(-2 * time.Hour).Format(time.RFC3339), CWD: "work stale",
		}},
		Registrations: []sessionregistry.Record{
			{Schema: sessionregistry.Schema, RegistrationID: "active-reg", RootRegistrationID: "active-reg", AttemptID: "active-attempt", LaunchKind: "guard", TaskID: "work active", Identity: sessionregistry.Identity{Runtime: "codex", SessionID: "active-session"}, State: sessionregistry.StateActive, CreatedAt: fixtureNow.Add(-time.Hour)},
			{Schema: sessionregistry.Schema, RegistrationID: "lost-reg", RootRegistrationID: "lost-reg", AttemptID: "lost-attempt", LaunchKind: "guard", TaskID: "work crashed", Identity: sessionregistry.Identity{Runtime: "codex", SessionID: "crashed-session"}, State: sessionregistry.StateLost, CreatedAt: fixtureNow.Add(-time.Hour), TerminalAt: terminal},
			{Schema: sessionregistry.Schema, RegistrationID: "done-reg", RootRegistrationID: "done-reg", AttemptID: "done-attempt", LaunchKind: "guard", TaskID: "work completed", Identity: sessionregistry.Identity{Runtime: "codex", SessionID: "completed-session"}, State: sessionregistry.StateCompleted, CreatedAt: fixtureNow.Add(-time.Hour), TerminalAt: terminal},
		},
	}
	for facet, wantID := range map[string]string{
		"active": "active-session", "stale": "stale-session", "crashed": "crashed-session", "completed": "completed-session",
	} {
		input.Query = mustQuery(t, "work is:"+facet, 10)
		report, err := Search(input)
		if err != nil || report.Verdict != VerdictSoleMatch || len(report.Hits) != 1 || report.Hits[0].SessionID != wantID {
			t.Errorf("is:%s = report %+v err %v; want %s", facet, report, err, wantID)
		}
	}
}

func TestVerdictsNoTermsNoMatchAndPartialCoverage(t *testing.T) {
	noTerms, err := Search(Input{Query: mustQuery(t, "is:active", 10), Now: fixtureNow, Coverage: completeCoverage()})
	if err != nil || noTerms.Verdict != VerdictNoContentTerms {
		t.Fatalf("no-content verdict = %+v err=%v", noTerms, err)
	}

	noMatch, err := Search(Input{Query: mustQuery(t, "missing", 10), Now: fixtureNow, Coverage: completeCoverage()})
	if err != nil || noMatch.Verdict != VerdictNoMatch {
		t.Fatalf("no-match verdict = %+v err=%v", noMatch, err)
	}

	partial := completeCoverage()
	partial[1] = Coverage{Store: StoreRegistration, Status: CoverageUnavailable, Detail: "permission denied"}
	partialReport, err := Search(Input{
		Query: mustQuery(t, "confluence", 10), Now: fixtureNow,
		Lifecycle: []sessionjournal.Event{{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "only", TS: fixtureNow.Format(time.RFC3339), CWD: "confluence"}},
		Coverage:  partial,
	})
	if err != nil || partialReport.Verdict != VerdictPartialCoverage || partialReport.TotalMatches != 1 {
		t.Fatalf("partial sole-match guard = %+v err=%v", partialReport, err)
	}

	partialNoMatch, err := Search(Input{Query: mustQuery(t, "absent", 10), Now: fixtureNow, Coverage: partial})
	if err != nil || partialNoMatch.Verdict != VerdictPartialCoverage || partialNoMatch.TotalMatches != 0 {
		t.Fatalf("partial no-match guard = %+v err=%v", partialNoMatch, err)
	}
}

func TestRunFixtureStoreFailurePreventsFalseUniqueness(t *testing.T) {
	paths := writeFixtureStores(t)
	full, err := Run("confluence is:active", Config{
		LifecyclePath: paths.lifecycle, RegistrationPath: paths.registration, ToolProcessPath: paths.tool,
		Now: fixtureNow, StaleAfter: 30 * time.Minute, Limit: 10,
	})
	if err != nil {
		t.Fatalf("Run full: %v", err)
	}
	if full.Verdict != VerdictSoleMatch || full.TotalMatches != 1 {
		t.Fatalf("full report = %+v", full)
	}
	if _, err := os.Stat(paths.registration + ".lock"); !os.IsNotExist(err) {
		t.Fatalf("read-only search created registration lock: %v", err)
	}

	partial, err := Run("confluence is:active", Config{
		LifecyclePath: paths.lifecycle, RegistrationPath: t.TempDir(), ToolProcessPath: paths.tool,
		Now: fixtureNow, StaleAfter: 30 * time.Minute, Limit: 10,
	})
	if err != nil {
		t.Fatalf("Run partial: %v", err)
	}
	if partial.Verdict != VerdictPartialCoverage || partial.TotalMatches != 1 {
		t.Fatalf("partial report = %+v", partial)
	}
	if got := coverageFor(partial.Coverage, StoreRegistration); got.Status != CoverageUnavailable || got.Detail == "" {
		t.Fatalf("registration coverage = %+v", got)
	}

	skipped, err := Run("confluence is:active", Config{
		LifecyclePath: paths.lifecycle, RegistrationPath: paths.registration, ToolProcessPath: paths.tool,
		SkipToolProcess: true, Now: fixtureNow, StaleAfter: 30 * time.Minute, Limit: 10,
	})
	if err != nil || skipped.Verdict != VerdictPartialCoverage {
		t.Fatalf("skipped store report = %+v err=%v", skipped, err)
	}
	if got := coverageFor(skipped.Coverage, StoreToolProcess); got.Status != CoverageSkipped {
		t.Fatalf("skipped tool-process coverage = %+v", got)
	}
}

type fixturePaths struct{ lifecycle, registration, tool string }

func writeFixtureStores(t *testing.T) fixturePaths {
	t.Helper()
	dir := t.TempDir()
	paths := fixturePaths{
		lifecycle:    filepath.Join(dir, "lifecycle.jsonl"),
		registration: filepath.Join(dir, "registrations.jsonl"),
		tool:         filepath.Join(dir, "tool-processes.jsonl"),
	}
	writeJSONLines(t, paths.lifecycle, []any{sessionjournal.Event{
		Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "life-1",
		TS: fixtureNow.Add(-2 * time.Minute).Format(time.RFC3339), CWD: "/work/confluence-export",
		Registration: &sessionjournal.RegistrationCarry{RegistrationID: "reg-1", SessionID: "session-1", AttemptID: "attempt-1", TaskID: "confluence migration", State: "active"},
	}})
	writeJSONLines(t, paths.registration, []any{sessionregistry.Event{
		Schema: sessionregistry.Schema, At: fixtureNow.Add(-2 * time.Minute),
		Record: sessionregistry.Record{
			Schema: sessionregistry.Schema, RegistrationID: "reg-1", RootRegistrationID: "reg-1", AttemptID: "attempt-1",
			LaunchKind: "guard", Scope: []string{"docs/confluence/**"}, Identity: sessionregistry.Identity{Runtime: "codex", SessionID: "session-1"},
			State: sessionregistry.StateActive, CreatedAt: fixtureNow.Add(-2 * time.Minute),
		},
	}})
	writeJSONLines(t, paths.tool, []any{toolproc.Event{
		Kind: toolproc.EvSpawn, CallID: "tool-1", Session: "session-1", Tool: "mcp__confluence__search",
		AtMS: fixtureNow.Add(-time.Minute).UnixMilli(), HeartbeatEveryMS: 120_000,
	}})
	return paths
}

func writeJSONLines(t *testing.T, path string, rows []any) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	enc := json.NewEncoder(f)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			_ = f.Close()
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func mustQuery(t *testing.T, raw string, limit int) Query {
	t.Helper()
	q, err := ParseQuery(raw, limit)
	if err != nil {
		t.Fatal(err)
	}
	return q
}

func completeCoverage() []Coverage {
	return []Coverage{
		{Store: StoreLifecycle, Status: CoverageComplete},
		{Store: StoreRegistration, Status: CoverageComplete},
		{Store: StoreToolProcess, Status: CoverageComplete},
	}
}

func coverageFor(rows []Coverage, store Store) Coverage {
	for _, row := range rows {
		if row.Store == store {
			return row
		}
	}
	return Coverage{}
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
