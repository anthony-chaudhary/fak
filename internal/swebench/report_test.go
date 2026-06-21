package swebench

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func sampleDataset() *Dataset {
	return NewDataset([]Instance{
		{InstanceID: "django__django-1", Difficulty: "<15min", ProblemStatement: "fix a"},
		{InstanceID: "sympy__sympy-2", Difficulty: "1-4hr", ProblemStatement: "fix b"},
	})
}

func TestBuildComparisonFamilies(t *testing.T) {
	c := BuildComparison(CompareInputs{
		Dataset:  sampleDataset(),
		Geometry: DefaultGeometryModel(),
		Workers:  []int{1, 4},
	})
	if len(c.Families) != 4 {
		t.Fatalf("expected 4 families, got %d", len(c.Families))
	}
	// family 1 + 2 are computed; 3 + 4 gated without inputs
	byName := map[string]MetricFamily{}
	for _, f := range c.Families {
		byName[f.Name] = f
	}
	fam1 := byName["prefill / KV-reuse work-elimination (deterministic)"]
	if fam1.Provenance != ProvComputed {
		t.Errorf("family1 provenance: %+v", fam1)
	}
	if fam1.Kind != KindFakNative {
		t.Errorf("family1 must be fak-native (A/C, B/C are fak-vs-fak ablation arms), got %q", fam1.Kind)
	}
	if byName["in-process adjudication cost"].Kind != KindFakNative {
		t.Errorf("adjudication should be fak-native")
	}
	if byName["resolve-rate + safety"].Provenance != ProvGated {
		t.Errorf("resolve should be gated without an eval")
	}
	if len(c.Honesty) == 0 {
		t.Errorf("comparison must carry honesty notes")
	}
}

func TestBuildComparisonWithEvalAndAdj(t *testing.T) {
	c := BuildComparison(CompareInputs{
		Dataset:      sampleDataset(),
		Geometry:     DefaultGeometryModel(),
		Workers:      []int{1},
		Eval:         &EvalResult{Available: true, Resolved: 3, Total: 10, ResolveRatePct: 30},
		Adjudication: &AdjCost{InProcessP50Ns: 1300, SpawnHookP50Ns: 6500000, SpeedupX: 5000},
	})
	for _, f := range c.Families {
		if f.Name == "resolve-rate + safety" && f.Provenance != ProvLive {
			t.Errorf("resolve should be live with an available eval")
		}
		if f.Name == "in-process adjudication cost" && f.Provenance != ProvLive {
			t.Errorf("adjudication should be live with supplied p50s")
		}
	}
	md := RenderMarkdown(c)
	if !strings.Contains(md, "SWE-bench Verified comparison") || !strings.Contains(md, "pass_rate_pct") {
		t.Errorf("markdown missing expected content")
	}
}

func TestParseBenchResultLocal(t *testing.T) {
	// synth a minimal bench-shaped result and parse it
	dir := t.TempDir()
	p := filepath.Join(dir, "results_x.json")
	os.WriteFile(p, []byte(`{
	  "schema_version": 6, "profile_name": "swebench-verified-mini-workers-sweep",
	  "total_run_time": {"seconds": 1234.5},
	  "cache_verdict": {"status": "CACHE_SATURATED", "per_server": {"DGX3": {"token_hit_ratio_pct": 0.0}}}
	}`), 0o644)
	bs, err := ParseBenchResult(p)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !bs.Present || bs.SchemaVersion != 6 || bs.CacheStatus != "CACHE_SATURATED" || bs.TotalRunSeconds != 1234.5 {
		t.Errorf("bench-side parse wrong: %+v", bs)
	}
}
