package benchloop

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchcatalog"
	"github.com/anthony-chaudhary/fak/internal/benchruns"
	"github.com/anthony-chaudhary/fak/internal/nightrun"
)

func benchLoopTime(t *testing.T, s string) time.Time {
	t.Helper()
	tm, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatal(err)
	}
	return tm.UTC()
}

func TestStatusChoosesLocalCollectionEntry(t *testing.T) {
	now := benchLoopTime(t, "2026-06-30T00:00:00Z")
	rep := StatusFromParts(Parts{
		Root: "/repo",
		Now:  now,
		Benchmarks: []benchcatalog.Bench{
			{Name: "a", Need: benchcatalog.NeedNone},
			{Name: "b", Need: benchcatalog.NeedWeights, Level: benchcatalog.LevelServing},
			{Name: "c", Need: benchcatalog.NeedDataset, Level: benchcatalog.LevelE2E, Manual: true},
		},
		Catalog: benchruns.Catalog{Runs: []benchruns.Run{
			{"run_id": "old", "machine_id": "box-a", "timestamp": "2026-06-28T00:00:00Z", "model": "A", "precision": "q8"},
			{"run_id": "new", "machine_id": "box-b", "timestamp": "2026-06-29T00:00:00Z", "model": "B", "precision": "q4"},
		}},
		Ledger: []nightrun.CollectRow{{
			Schema: nightrun.CollectSchema, Date: "2026-06-30", Box: "box-a",
			TaskID: "fresh", Outcome: string(nightrun.OutcomeCollected), GeneratedAt: "2026-06-30T00:00:00Z",
		}},
		Caps: nightrun.Capabilities{Box: "box-a", GPU: "none", Net: true, Creds: map[string]bool{}},
		Tasks: []nightrun.Task{
			{ID: "fresh", Value: nightrun.ValueSmoke, Run: "echo fresh", RecheckDays: 14},
			{ID: "collect-me", Value: nightrun.ValueCoverage, Run: "echo 12 tok/s", Acceptance: "12 tok/s"},
			{ID: "blocked-cuda", Value: nightrun.ValueFrontier, Requires: []nightrun.Requirement{nightrun.ReqCUDA}, Run: "echo cuda"},
		},
		Authority: nightrun.LedgerGapReport{AuthorityDate: "2026-06-29", TotalRows: 1, TotalCollected: 1},
	})

	if rep.Schema != StatusSchema {
		t.Fatalf("schema = %q", rep.Schema)
	}
	if rep.Registry.Benchmarks != 3 || rep.Registry.Offline != 1 || rep.Registry.Serving != 1 || rep.Registry.E2E != 1 || rep.Registry.Manual != 1 {
		t.Fatalf("registry counts wrong: %+v", rep.Registry)
	}
	if rep.Catalog.RunCount != 2 || rep.Catalog.MachineCount != 2 || rep.Catalog.LatestRunID != "new" {
		t.Fatalf("catalog status wrong: %+v", rep.Catalog)
	}
	if rep.Local.Feasible != 2 || rep.Local.Blocked != 1 || rep.Local.Saturated != 1 {
		t.Fatalf("local status wrong: %+v", rep.Local)
	}
	if rep.Local.Next == nil || rep.Local.Next.ID != "collect-me" {
		t.Fatalf("next = %+v, want collect-me", rep.Local.Next)
	}
	if rep.NextAction.Kind != "collect_local" || rep.NextAction.Command != "fak bench-loop run --apply" {
		t.Fatalf("action = %+v, want collect_local via nightrun", rep.NextAction)
	}
	human := RenderStatus(rep)
	for _, want := range []string{"benchmark super-loop status", "next action: collect_local", "collect-me"} {
		if !strings.Contains(human, want) {
			t.Fatalf("status render missing %q:\n%s", want, human)
		}
	}
}

func TestStatusSurfacesManualAndCatalogRefresh(t *testing.T) {
	now := benchLoopTime(t, "2026-06-30T00:00:00Z")
	manual := StatusFromParts(Parts{
		Now:     now,
		Catalog: benchruns.Catalog{Runs: []benchruns.Run{{"run_id": "r", "timestamp": "2026-06-29T00:00:00Z"}}},
		Caps:    nightrun.Capabilities{Box: "box-a", GPU: "none", Net: true, Creds: map[string]bool{}},
		Tasks: []nightrun.Task{{
			ID: "operator", Value: nightrun.ValueFrontier, Run: "run <model>", Manual: true,
		}},
	})
	if manual.NextAction.Kind != "manual_collect" || manual.NextAction.Command != "run <model>" {
		t.Fatalf("manual action = %+v", manual.NextAction)
	}

	missing := StatusFromParts(Parts{
		Now:        now,
		CatalogErr: errors.New("missing catalog"),
		Caps:       nightrun.Capabilities{Box: "box-a", Net: true, Creds: map[string]bool{}},
	})
	if missing.NextAction.Kind != "refresh_catalog" || !strings.Contains(missing.NextAction.Command, "bench_catalog.py build") {
		t.Fatalf("missing catalog action = %+v", missing.NextAction)
	}
}

func TestWalkNamesBenchmarkSurfaces(t *testing.T) {
	walk := RenderWalk(Walk())
	for _, want := range []string{"fak bench-loop status", "fak bench-loop run --apply --loop", "fak bench-runs summary", "fak bench request"} {
		if !strings.Contains(walk, want) {
			t.Fatalf("walk missing %q:\n%s", want, walk)
		}
	}
	if strings.Index(walk, "status") > strings.Index(walk, "enter") {
		t.Fatalf("walk should preserve workflow order (status before enter):\n%s", walk)
	}
}
