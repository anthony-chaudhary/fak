package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func withDispatchPriceTaxonomy(t *testing.T) {
	t.Helper()
	old := dispatchLoadLaneTaxonomy
	dispatchLoadLaneTaxonomy = func(root string) (dispatchtick.LaneTaxonomy, error) {
		return dispatchtick.LaneTaxonomy{
			Concurrent: []string{"gateway", "docs"},
			Trees: map[string][]string{
				"gateway": {"internal/gateway/**"},
				"docs":    {"docs/**"},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchLoadLaneTaxonomy = old })
}

func TestDispatchPriceJSONPricesProposedFanoutBeforeLaunch(t *testing.T) {
	withDispatchPriceTaxonomy(t)
	path := writeCandidates(t, `{"agents":[
		{"name":"gateway","lane":"gateway"},
		{"name":"gateway-http","lane":"gateway","tree":["internal/gateway/http.go"]},
		{"name":"docs","lane":"docs"}
	]}`)

	out, errb, code := runDispatchAt("price", "--workspace", t.TempDir(), "--in", path, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got dispatchPriceReport
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got.Schema != dispatchPriceSchema || !got.OK {
		t.Fatalf("price header = %+v, want schema %s ok", got, dispatchPriceSchema)
	}
	if !strings.HasPrefix(got.PlanID, "plan-") {
		t.Fatalf("plan_id = %q, want plan digest", got.PlanID)
	}
	if got.Action != "LAUNCH_SAFE_SET" || got.SafeConcurrency != 2 ||
		got.CollisionsAvoided != 1 || got.SerializationWasted != 1 || got.ExpectedRework == 0 {
		t.Fatalf("price metrics = action %s safe %d avoided %d wasted %d rework %d",
			got.Action, got.SafeConcurrency, got.CollisionsAvoided, got.SerializationWasted, got.ExpectedRework)
	}
	if strings.Join(got.SafeNow, ",") != "gateway,docs" {
		t.Fatalf("safe_now = %#v, want gateway/docs", got.SafeNow)
	}
	if got.WaveCount != 2 || len(got.Waves) != 2 ||
		strings.Join(got.Waves[0].Agents, ",") != "gateway,docs" ||
		strings.Join(got.Waves[1].Agents, ",") != "gateway-http" {
		t.Fatalf("waves = %+v, want [gateway,docs] then [gateway-http]", got.Waves)
	}
	if len(got.LaunchPlan) != 2 || len(got.LaunchPlan[0].Targets) != 2 || len(got.LaunchPlan[1].Targets) != 1 {
		t.Fatalf("launch plan = %+v, want two waves with 2 then 1 target", got.LaunchPlan)
	}
	if got.LaunchPlan[0].Targets[0].LeaseID == "" || len(got.LaunchPlan[0].Targets[0].Tree) == 0 {
		t.Fatalf("first launch target = %+v, want resolved lease/tree", got.LaunchPlan[0].Targets[0])
	}
	if got.LaunchPlan[1].Targets[0].ID != "gateway-http" ||
		got.LaunchPlan[1].Targets[0].Disposition != dispatchorder.DispCollisionRisk {
		t.Fatalf("second-wave launch target = %+v, want held gateway-http collision-risk target", got.LaunchPlan[1].Targets[0])
	}
	if got.LaneSerialWaveCount != 2 || got.ScopedParallelGain != 0 || got.CollisionWavePenalty != 0 {
		t.Fatalf("baseline metrics = lane_serial %d gain %d penalty %d, want 2/0/0",
			got.LaneSerialWaveCount, got.ScopedParallelGain, got.CollisionWavePenalty)
	}
	if len(got.Repartition) != 1 || got.Repartition[0].Candidate != "gateway-http" {
		t.Fatalf("repartition = %+v, want gateway-http row", got.Repartition)
	}
	if !strings.Contains(got.Log, "reactive floor") {
		t.Fatalf("log should remind callers that leases remain required, got %q", got.Log)
	}
}

func TestDispatchPriceAllowsSameLaneDisjointExplicitScopes(t *testing.T) {
	withDispatchPriceTaxonomy(t)
	path := writeCandidates(t, `{"agents":[
		{"name":"http","lane":"gateway","tree":["internal/gateway/http.go"]},
		{"name":"mcp","lane":"gateway","tree":["internal/gateway/mcp.go"]}
	]}`)

	out, errb, code := runDispatchAt("price", "--workspace", t.TempDir(), "--in", path, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got dispatchPriceReport
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got.Action != "LAUNCH_ALL" || got.SafeConcurrencyPct != 100 ||
		got.SameLaneParallelism != 1 || len(got.Collisions) != 0 {
		t.Fatalf("same-lane scoped price = action %s safe_pct %d same_lane %d collisions %d",
			got.Action, got.SafeConcurrencyPct, got.SameLaneParallelism, len(got.Collisions))
	}
	if got.WaveCount != 1 || len(got.Waves) != 1 || strings.Join(got.Waves[0].Agents, ",") != "http,mcp" {
		t.Fatalf("waves = %+v, want one same-lane scoped wave", got.Waves)
	}
	if len(got.LaunchPlan) != 1 || len(got.LaunchPlan[0].Targets) != 2 {
		t.Fatalf("launch plan = %+v, want one two-target wave", got.LaunchPlan)
	}
	targets := got.LaunchPlan[0].Targets
	if targets[0].Lane != "gateway" || targets[1].Lane != "gateway" ||
		targets[0].LeaseID == targets[1].LeaseID ||
		!targets[0].Scoped || !targets[1].Scoped ||
		len(targets[0].Tree) != 1 || len(targets[1].Tree) != 1 {
		t.Fatalf("launch targets = %+v, want same display lane with distinct scoped leases/trees", targets)
	}
	if got.LaneSerialWaveCount != 2 || got.ScopedParallelGain != 1 || got.CollisionWavePenalty != 0 {
		t.Fatalf("baseline metrics = lane_serial %d gain %d penalty %d, want 2/1/0",
			got.LaneSerialWaveCount, got.ScopedParallelGain, got.CollisionWavePenalty)
	}
	for _, cand := range got.Candidates {
		if cand.Disposition != dispatchorder.DispKeep || cand.TreeSource != "input" {
			t.Fatalf("candidate = %+v, want kept input-scoped candidate", cand)
		}
	}
}

func TestDispatchPriceUnknownScopeCollidesConservatively(t *testing.T) {
	withDispatchPriceTaxonomy(t)
	path := writeCandidates(t, `{"agents":[
		{"name":"unknown","lane":"mystery"},
		{"name":"docs","lane":"docs"}
	]}`)

	out, errb, code := runDispatchAt("price", "--workspace", t.TempDir(), "--in", path, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got dispatchPriceReport
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got.Action != "LAUNCH_SAFE_SET" || got.CollisionsAvoided != 1 || got.SafeConcurrency != 1 {
		t.Fatalf("unknown-scope price = action %s avoided %d safe %d", got.Action, got.CollisionsAvoided, got.SafeConcurrency)
	}
	if got.Candidates[0].Name != "unknown" || got.Candidates[0].TreeSource != "unknown" {
		t.Fatalf("first candidate = %+v, want kept unknown-scope candidate", got.Candidates[0])
	}
	if got.WaveCount != 2 || strings.Join(got.Waves[0].Agents, ",") != "unknown" || strings.Join(got.Waves[1].Agents, ",") != "docs" {
		t.Fatalf("waves = %+v, want unknown then docs", got.Waves)
	}
	if len(got.LaunchPlan) != 2 || got.LaunchPlan[0].Targets[0].ScopeSource != "unknown" {
		t.Fatalf("launch plan = %+v, want unknown-scope first target", got.LaunchPlan)
	}
	if got.LaneSerialWaveCount != 1 || got.ScopedParallelGain != 0 || got.CollisionWavePenalty != 1 {
		t.Fatalf("baseline metrics = lane_serial %d gain %d penalty %d, want 1/0/1",
			got.LaneSerialWaveCount, got.ScopedParallelGain, got.CollisionWavePenalty)
	}
	if len(got.Repartition) != 1 || got.Repartition[0].Action != "peer_declare_tree_scope" {
		t.Fatalf("repartition = %+v, want peer_declare_tree_scope", got.Repartition)
	}
}

func TestDispatchPriceHumanRenderReturnsActionLine(t *testing.T) {
	withDispatchPriceTaxonomy(t)
	path := writeCandidates(t, `{"agents":[
		{"name":"http","lane":"gateway","tree":["internal/gateway/http.go"]},
		{"name":"mcp","lane":"gateway","tree":["internal/gateway/mcp.go"]}
	]}`)

	out, errb, code := runDispatchAt("price", "--workspace", t.TempDir(), "--in", path)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	for _, want := range []string{"dispatch price:", "plan_id=plan-", "waves=1", "lane_serial=2", "scoped_gain=1", "schedule: wave1[http,mcp]", "same_lane_parallelism=1", "Action: LAUNCH_ALL"} {
		if !strings.Contains(out, want) {
			t.Fatalf("human render missing %q:\n%s", want, out)
		}
	}
}

func TestDispatchPriceLargeDryRunWaveSelectionBenchmark(t *testing.T) {
	const candidates = 512
	agents := largeDispatchPriceAgents(candidates)

	start := time.Now()
	got := buildDispatchPriceReport(agents, dispatchtick.LaneTaxonomy{})
	elapsed := time.Since(start)
	perCandidate := elapsed / time.Duration(candidates)
	t.Logf("planned %d dry-run dispatch candidates in %s (%s/candidate)", candidates, elapsed, perCandidate)

	if got.Requested != candidates ||
		got.SafeConcurrency != candidates ||
		got.WaveCount != 1 ||
		got.SafeConcurrencyPct != 100 ||
		got.CollisionWavePenalty != 0 {
		t.Fatalf("large dry-run plan = requested %d safe %d waves %d safe_pct %d penalty %d",
			got.Requested, got.SafeConcurrency, got.WaveCount, got.SafeConcurrencyPct, got.CollisionWavePenalty)
	}
	if perCandidate > 5*time.Millisecond {
		t.Fatalf("dry-run wave selection too slow: %s/candidate over %d candidates", perCandidate, candidates)
	}
}

func BenchmarkDispatchPriceLargeDryRunWaveSelection(b *testing.B) {
	const candidates = 512
	agents := largeDispatchPriceAgents(candidates)
	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		got := buildDispatchPriceReport(agents, dispatchtick.LaneTaxonomy{})
		if got.WaveCount != 1 || got.SafeConcurrency != candidates {
			b.Fatalf("large dry-run plan = waves %d safe %d, want one wave and %d safe", got.WaveCount, got.SafeConcurrency, candidates)
		}
	}
	elapsed := time.Since(start)
	b.StopTimer()
	b.ReportMetric(float64(candidates), "candidates/op")
	b.ReportMetric(float64(elapsed.Nanoseconds())/float64(b.N*candidates), "ns/candidate")
}

func largeDispatchPriceAgents(n int) []dispatchPriceAgent {
	agents := make([]dispatchPriceAgent, 0, n)
	for i := 0; i < n; i++ {
		agents = append(agents, dispatchPriceAgent{
			Name: fmt.Sprintf("worker-%03d", i),
			Lane: "gateway",
			Tree: []string{fmt.Sprintf("internal/gateway/dryrun/%03d.go", i)},
		})
	}
	return agents
}

func TestDispatchLaneTaxonomyFromFileReadsLaneTrees(t *testing.T) {
	root := t.TempDir()
	dosToml := `[lanes]
concurrent = ["gateway", "docs"]
exclusive = ["dos"]

[lanes.trees]
gateway = ["internal/gateway/**"]
docs = ["docs/**", "README.md"]
`
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dosToml), 0o644); err != nil {
		t.Fatal(err)
	}

	taxonomy, err := dispatchLaneTaxonomyFromFile(root)
	if err != nil {
		t.Fatalf("dispatchLaneTaxonomyFromFile: %v", err)
	}
	if strings.Join(taxonomy.Trees["gateway"], ",") != "internal/gateway/**" {
		t.Fatalf("gateway tree = %#v", taxonomy.Trees["gateway"])
	}
	if strings.Join(taxonomy.Trees["docs"], ",") != "docs/**,README.md" {
		t.Fatalf("docs tree = %#v", taxonomy.Trees["docs"])
	}
	if len(taxonomy.Concurrent) == 0 {
		t.Fatalf("expected declared lanes from [lanes], got none")
	}
}
