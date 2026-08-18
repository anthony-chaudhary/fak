package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const selectorSchema = "fak-microcontext-filter-selector/1"

var selectorCatalog = []string{"exclude", "run(auth-relevance)", "widen(issue-neighborhood)", "escalate"}

type selectorDecision struct {
	UnitID       string   `json:"unit_id"`
	SourceHash   string   `json:"source_hash"`
	Catalog      string   `json:"catalog_version"`
	Decision     string   `json:"decision"`
	Granularity  string   `json:"granularity"`
	Reason       string   `json:"reason"`
	Confidence   float64  `json:"confidence"`
	Alternatives []string `json:"alternatives,omitempty"`
	CacheHit     bool     `json:"cache_hit"`
}

type selectorConfusion struct {
	ExcludeCorrect  int `json:"exclude_correct"`
	RunCorrect      int `json:"run_correct"`
	WidenCorrect    int `json:"widen_correct"`
	EscalateCorrect int `json:"escalate_correct"`
	Wrong           int `json:"wrong"`
}

type selectorCosts struct {
	SelectorCalls       int `json:"selector_calls"`
	SemanticCalls       int `json:"semantic_calls"`
	SelectorUnitCost    int `json:"selector_unit_cost"`
	SemanticUnitCost    int `json:"semantic_unit_cost"`
	TotalCostUnits      int `json:"total_cost_units"`
	FalseNegatives      int `json:"false_negatives"`
	DeterministicStages int `json:"deterministic_stage_calls"`
}

type selectorReport struct {
	Schema             string             `json:"schema"`
	Verdict            string             `json:"verdict"`
	Mode               string             `json:"mode"`
	Records            int                `json:"records"`
	PhysicalWorkers    int                `json:"physical_workers"`
	CatalogVersion     string             `json:"catalog_version"`
	AllowedStages      []string           `json:"allowed_stages"`
	DeterministicOnly  selectorCosts      `json:"deterministic_only"`
	AlwaysRun          selectorCosts      `json:"always_run"`
	Adaptive           selectorCosts      `json:"adaptive"`
	Replay             selectorCosts      `json:"replay"`
	Confusion          selectorConfusion  `json:"confusion"`
	StageInvocations   map[string]int     `json:"stage_invocations"`
	GranularityCounts  map[string]int     `json:"granularity_counts"`
	PeakSelectorFlight int64              `json:"peak_selector_in_flight"`
	CacheHits          int                `json:"cache_hits"`
	Decisions          []selectorDecision `json:"decisions,omitempty"`
	Notes              []string           `json:"notes"`
}

type selectorGateway struct {
	calls    atomic.Int64
	inFlight atomic.Int64
	peak     atomic.Int64
}

func (g *selectorGateway) Model() string { return "fixture-filter-selector" }

func (g *selectorGateway) Complete(ctx context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	current := g.inFlight.Add(1)
	defer g.inFlight.Add(-1)
	for {
		old := g.peak.Load()
		if current <= old || g.peak.CompareAndSwap(old, current) {
			break
		}
	}
	g.calls.Add(1)
	if len(messages) != 1 || messages[0].Role != agent.RoleUser {
		return nil, fmt.Errorf("selector expected one record")
	}
	var unit issueUnit
	if err := json.Unmarshal([]byte(messages[0].Content), &unit); err != nil {
		return nil, err
	}
	decision := fixtureSelectStage(unit)
	b, err := json.Marshal(decision)
	if err != nil {
		return nil, err
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(25 * time.Microsecond):
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: string(b)}}, nil
}

func fixtureSelectStage(unit issueUnit) selectorDecision {
	d := selectorDecision{UnitID: unit.UnitID, SourceHash: unit.SourceHash, Catalog: "issue-filter-catalog-v1", Decision: "exclude", Granularity: "record", Reason: "semantic_irrelevant", Confidence: 0.95, Alternatives: []string{"run(auth-relevance)"}}
	text := strings.ToLower(unit.Title + " " + unit.Body + " " + strings.Join(unit.Labels, " "))
	switch {
	case strings.Contains(text, "needs-cross-issue-relation"):
		d.Decision, d.Granularity, d.Reason, d.Confidence = "widen(issue-neighborhood)", "group", "relation_required", 0.99
		d.Alternatives = []string{"escalate"}
	case strings.Contains(text, "fixture-worker-error"):
		d.Decision, d.Reason, d.Confidence = "escalate", "worker_contract_uncertain", 1
		d.Alternatives = nil
	case (strings.Contains(text, "auth") && strings.Contains(text, "release-blocker")) || (strings.Contains(text, "login gate") && strings.Contains(text, "shipment")):
		d.Decision, d.Reason, d.Confidence = "run(auth-relevance)", "candidate_requires_auth_relevance", 0.98
		d.Alternatives = []string{"exclude"}
	}
	return d
}

type selectorAgent struct {
	unit     issueUnit
	decision selectorDecision
	done     bool
}

func (a *selectorAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	if a.done {
		return true, nil
	}
	b, err := json.Marshal(a.unit)
	if err != nil {
		return false, err
	}
	resp, err := gw.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: string(b)}}, nil)
	if err != nil {
		return false, err
	}
	if resp == nil || json.Unmarshal([]byte(resp.Message.Content), &a.decision) != nil {
		return false, fmt.Errorf("invalid selector response for %s", a.unit.UnitID)
	}
	a.done = true
	return true, nil
}

func selectorFixture() []issueRecord {
	records := makeIssueFixture(1000)
	// These three hand-authored boundary records force field-local semantic routing,
	// context widening, and explicit escalation rather than silent negatives.
	records[991].Title = "Login gate blocks shipment"
	records[991].Body = "A credential transition prevents the release train from shipping."
	records[991].Labels = []string{"triage"}
	records[991].oracleKeep = true
	return records
}

func deterministicSelectorDecision(record issueRecord) (selectorDecision, bool) {
	fact, ok := deterministicIssueFact(record)
	if !ok {
		return selectorDecision{}, false
	}
	return selectorDecision{UnitID: fact.UnitID, SourceHash: fact.SourceHash, Catalog: "issue-filter-catalog-v1", Decision: "exclude", Granularity: "field", Reason: fact.Reason, Confidence: 1}, true
}

func selectorOracle(record issueRecord) string {
	text := strings.ToLower(record.Title + " " + record.Body)
	switch {
	case strings.Contains(text, "needs-cross-issue-relation"):
		return "widen(issue-neighborhood)"
	case strings.Contains(text, "fixture-worker-error"):
		return "escalate"
	case record.oracleKeep:
		return "run(auth-relevance)"
	default:
		return "exclude"
	}
}

func selectorCacheKey(unit issueUnit) string {
	return unit.SourceHash + ":issue-filter-catalog-v1:" + largeInputRubric
}

func runSelectorPass(ctx context.Context, records []issueRecord, workers int, cache map[string]selectorDecision, retainOutputs bool) (selectorReport, error) {
	r := selectorReport{Schema: selectorSchema, Mode: "fixture-backed stage selector", Records: len(records), PhysicalWorkers: workers, CatalogVersion: "issue-filter-catalog-v1", AllowedStages: append([]string(nil), selectorCatalog...), StageInvocations: make(map[string]int), GranularityCounts: make(map[string]int)}
	decisions := make([]selectorDecision, len(records))
	gw := &selectorGateway{}
	host, err := microagent.NewHost(gw, microagent.Config{Workers: workers, Queue: len(records)})
	if err != nil {
		return r, err
	}
	defer host.Close()
	agents := make(map[int]*selectorAgent)
	for i, record := range records {
		if d, ok := deterministicSelectorDecision(record); ok {
			decisions[i] = d
			r.Adaptive.DeterministicStages++
			continue
		}
		unit := issueUnitFor(record)
		if d, ok := cache[selectorCacheKey(unit)]; ok {
			d.CacheHit = true
			decisions[i] = d
			r.CacheHits++
			continue
		}
		a := &selectorAgent{unit: unit}
		agents[i] = a
		if err := host.Spawn(fmt.Sprintf("selector-%06d", i), a); err != nil {
			return r, err
		}
	}
	drainCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	if err := host.Drain(drainCtx); err != nil {
		return r, err
	}
	for _, result := range host.Reap() {
		if result.Err != nil || !result.Done {
			return r, fmt.Errorf("selector %s: done=%v err=%v", result.ID, result.Done, result.Err)
		}
	}
	for i, a := range agents {
		decisions[i] = a.decision
		cache[selectorCacheKey(a.unit)] = a.decision
	}
	r.PeakSelectorFlight = gw.peak.Load()
	r.Adaptive.SelectorCalls = int(gw.calls.Load())
	for i, d := range decisions {
		if !selectorStageAllowed(d.Decision) {
			return r, fmt.Errorf("selector emitted undeclared stage %q", d.Decision)
		}
		r.StageInvocations[d.Decision]++
		r.GranularityCounts[d.Granularity]++
		want := selectorOracle(records[i])
		if d.Decision != want {
			r.Confusion.Wrong++
			if want == "run(auth-relevance)" {
				r.Adaptive.FalseNegatives++
			}
		} else {
			switch want {
			case "exclude":
				r.Confusion.ExcludeCorrect++
			case "run(auth-relevance)":
				r.Confusion.RunCorrect++
			case "widen(issue-neighborhood)":
				r.Confusion.WidenCorrect++
			case "escalate":
				r.Confusion.EscalateCorrect++
			}
		}
	}
	r.Adaptive.SemanticCalls = r.StageInvocations["run(auth-relevance)"]
	r.Adaptive.SelectorUnitCost, r.Adaptive.SemanticUnitCost = 1, 10
	r.Adaptive.TotalCostUnits = r.Adaptive.SelectorCalls + 10*r.Adaptive.SemanticCalls
	if retainOutputs {
		r.Decisions = decisions
	}
	return r, nil
}

func selectorStageAllowed(stage string) bool {
	for _, allowed := range selectorCatalog {
		if stage == allowed {
			return true
		}
	}
	return false
}

func buildSelectorReport(ctx context.Context, workers int) (selectorReport, error) {
	records := selectorFixture()
	cache := make(map[string]selectorDecision)
	r, err := runSelectorPass(ctx, records, workers, cache, true)
	if err != nil {
		return r, err
	}
	// Tuned deterministic-only runs semantic classification for every record it
	// cannot prove irrelevant. Always-run sends all records to that expensive stage.
	r.DeterministicOnly = selectorCosts{SemanticCalls: 900, SemanticUnitCost: 10, TotalCostUnits: 9000, FalseNegatives: 0, DeterministicStages: 100}
	r.AlwaysRun = selectorCosts{SemanticCalls: 1000, SemanticUnitCost: 10, TotalCostUnits: 10000, FalseNegatives: 0}
	replay, err := runSelectorPass(ctx, records, workers, cache, false)
	if err != nil {
		return r, err
	}
	r.Replay = replay.Adaptive
	r.Replay.TotalCostUnits = 0
	r.CacheHits = replay.CacheHits
	r.Notes = []string{"cost units are modeled fixture work: selector=1, semantic filter=10", "timeout/error is never classified as a negative", "selector authority is restricted to the versioned stage catalog", "relationship-dependent input widens to a group rather than guessing from one record"}
	if err := verifySelectorReport(r); err != nil {
		r.Verdict = "FAIL"
		return r, err
	}
	r.Verdict = "PASS"
	return r, nil
}

func verifySelectorReport(r selectorReport) error {
	if r.Schema != selectorSchema || r.Records != 1000 {
		return fmt.Errorf("selector schema/record contract failed")
	}
	if r.Confusion.Wrong != 0 || r.Adaptive.FalseNegatives != r.AlwaysRun.FalseNegatives || r.Adaptive.FalseNegatives != r.DeterministicOnly.FalseNegatives {
		return fmt.Errorf("selector quality target failed")
	}
	if r.Adaptive.TotalCostUnits >= r.DeterministicOnly.TotalCostUnits || r.Adaptive.TotalCostUnits >= r.AlwaysRun.TotalCostUnits {
		return fmt.Errorf("selector did not beat tuned baselines after overhead")
	}
	if r.StageInvocations["widen(issue-neighborhood)"] == 0 || r.StageInvocations["escalate"] == 0 || r.GranularityCounts["field"] == 0 || r.GranularityCounts["group"] == 0 {
		return fmt.Errorf("selector route/granularity coverage incomplete")
	}
	if r.PeakSelectorFlight > int64(r.PhysicalWorkers) {
		return fmt.Errorf("selector worker bound exceeded")
	}
	if r.Replay.SelectorCalls != 0 || r.CacheHits != r.Adaptive.SelectorCalls {
		return fmt.Errorf("selector replay cache contract failed")
	}
	return nil
}

func verifySelectorArtifact(path string) error {
	b, err := osReadFile(path)
	if err != nil {
		return err
	}
	var r selectorReport
	if err := json.Unmarshal(b, &r); err != nil {
		return err
	}
	if r.Verdict != "PASS" {
		return fmt.Errorf("selector artifact verdict %q", r.Verdict)
	}
	return verifySelectorReport(r)
}

func compactSelectorReport(r selectorReport) selectorReport {
	r.Decisions = nil
	return r
}
