package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const (
	largeInputSchema = "fak-microcontext-large-input/1"
	largeInputRubric = "release-blocking-auth-v1"
	largeInputFanIn  = 32
)

type issueRecord struct {
	Number     int      `json:"number"`
	Title      string   `json:"title"`
	Body       string   `json:"body"`
	State      string   `json:"state"`
	Labels     []string `json:"labels"`
	Milestone  string   `json:"milestone"`
	oracleKeep bool
}

type issueUnit struct {
	UnitID     string   `json:"unit_id"`
	SourceHash string   `json:"source_hash"`
	Number     int      `json:"number"`
	Title      string   `json:"title"`
	Body       string   `json:"body"`
	Labels     []string `json:"labels"`
	Milestone  string   `json:"milestone"`
	Rubric     string   `json:"rubric"`
}

type issueFact struct {
	UnitID     string `json:"unit_id"`
	SourceHash string `json:"source_hash"`
	Status     string `json:"status"`
	Keep       bool   `json:"keep"`
	Reason     string `json:"reason"`
	Evidence   string `json:"evidence,omitempty"`
	Operator   string `json:"operator"`
	CacheHit   bool   `json:"cache_hit"`
}

type largeInputPass struct {
	Name           string      `json:"name"`
	SourceHash     string      `json:"source_hash"`
	Records        int         `json:"records"`
	Deterministic  int         `json:"deterministic_exclusions"`
	SemanticCalls  int         `json:"semantic_calls"`
	CacheHits      int         `json:"cache_hits"`
	Kept           int         `json:"kept"`
	Excluded       int         `json:"excluded"`
	Abstained      int         `json:"abstained"`
	Errored        int         `json:"errored"`
	PeakInFlight   int64       `json:"peak_in_flight"`
	FoldFanIn      int         `json:"fold_fan_in"`
	FoldMaxInput   int         `json:"fold_max_input"`
	FoldLevels     int         `json:"fold_levels"`
	FoldNodes      int         `json:"fold_nodes"`
	CitedIssueIDs  []int       `json:"cited_issue_ids"`
	OracleIssueIDs []int       `json:"oracle_issue_ids"`
	Facts          []issueFact `json:"facts,omitempty"`
}

type largeInputReport struct {
	Schema                   string         `json:"schema"`
	Verdict                  string         `json:"verdict"`
	Mode                     string         `json:"mode"`
	Scope                    string         `json:"scope"`
	Records                  int            `json:"records"`
	PhysicalWorkers          int            `json:"physical_workers"`
	Rubric                   string         `json:"rubric"`
	Baseline                 largeInputPass `json:"baseline"`
	Unchanged                largeInputPass `json:"unchanged"`
	Mutated                  largeInputPass `json:"mutated"`
	MutationUnitID           string         `json:"mutation_unit_id"`
	PreciseInvalidations     int            `json:"precise_invalidations"`
	UnchangedPureStageReuse  int            `json:"unchanged_pure_stage_reuse"`
	SourceAccountingVerified bool           `json:"source_accounting_verified"`
	OracleVerified           bool           `json:"oracle_verified"`
	ReducerBounded           bool           `json:"reducer_bounded"`
	Notes                    []string       `json:"notes"`
}

type semanticIssueGateway struct {
	calls    atomic.Int64
	inFlight atomic.Int64
	peak     atomic.Int64
}

func (g *semanticIssueGateway) Model() string { return "fixture-semantic-worker" }

func (g *semanticIssueGateway) Complete(ctx context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
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
		return nil, fmt.Errorf("large-input worker expected one user record")
	}
	var unit issueUnit
	if err := json.Unmarshal([]byte(messages[0].Content), &unit); err != nil {
		return nil, fmt.Errorf("decode issue unit: %w", err)
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(50 * time.Microsecond):
	}
	fact := classifyIssueFixture(unit)
	encoded, err := json.Marshal(fact)
	if err != nil {
		return nil, err
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: string(encoded)}}, nil
}

func classifyIssueFixture(unit issueUnit) issueFact {
	fact := issueFact{UnitID: unit.UnitID, SourceHash: unit.SourceHash, Status: "excluded", Reason: "semantic_not_relevant", Operator: largeInputRubric}
	text := strings.ToLower(unit.Title + " " + unit.Body + " " + strings.Join(unit.Labels, " "))
	if strings.Contains(text, "fixture-worker-error") {
		fact.Status, fact.Reason = "error", "fixture_worker_error"
		return fact
	}
	if strings.Contains(text, "needs-cross-issue-relation") {
		fact.Status, fact.Reason = "abstain", "missing_required_relation"
		return fact
	}
	if strings.Contains(text, "auth") && strings.Contains(text, "release-blocker") {
		fact.Status, fact.Keep, fact.Reason, fact.Evidence = "kept", true, "auth_release_blocker", "title/body contains auth + release-blocker"
	}
	return fact
}

type issueOperatorAgent struct {
	unit issueUnit
	fact issueFact
	done bool
}

func (a *issueOperatorAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	if a.done {
		return true, nil
	}
	payload, err := json.Marshal(a.unit)
	if err != nil {
		return false, err
	}
	resp, err := gw.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: string(payload)}}, nil)
	if err != nil {
		return false, err
	}
	if resp == nil {
		return false, fmt.Errorf("nil semantic response for %s", a.unit.UnitID)
	}
	if err := json.Unmarshal([]byte(resp.Message.Content), &a.fact); err != nil {
		return false, fmt.Errorf("decode semantic fact %s: %w", a.unit.UnitID, err)
	}
	a.done = true
	return true, nil
}

func makeIssueFixture(n int) []issueRecord {
	records := make([]issueRecord, n)
	for i := range records {
		number := 10000 + i
		r := issueRecord{Number: number, State: "open", Milestone: "v-next", Labels: []string{"triage"}, Title: fmt.Sprintf("Routine issue %04d", i), Body: "Ordinary maintenance report."}
		switch {
		case i%10 == 0:
			r.Labels = []string{"documentation"}
			r.Title = fmt.Sprintf("Documentation cleanup %04d", i)
		case i%37 == 0:
			r.Title = fmt.Sprintf("Auth regression %04d", i)
			r.Body = "Authentication path regression is a release-blocker."
			r.Labels = []string{"auth", "release-blocker"}
			r.oracleKeep = true
		case i == 997:
			r.Body = "needs-cross-issue-relation"
		case i == 998:
			r.Body = "fixture-worker-error"
		}
		records[i] = r
	}
	return records
}

func issueSourceHash(r issueRecord) string {
	public := struct {
		Number    int      `json:"number"`
		Title     string   `json:"title"`
		Body      string   `json:"body"`
		State     string   `json:"state"`
		Labels    []string `json:"labels"`
		Milestone string   `json:"milestone"`
	}{r.Number, r.Title, r.Body, r.State, r.Labels, r.Milestone}
	b, _ := json.Marshal(public)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func issueCorpusHash(records []issueRecord) string {
	h := sha256.New()
	for _, r := range records {
		h.Write([]byte(issueSourceHash(r)))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func deterministicIssueFact(r issueRecord) (issueFact, bool) {
	for _, label := range r.Labels {
		if label == "documentation" {
			return issueFact{UnitID: fmt.Sprintf("issue-%d", r.Number), SourceHash: issueSourceHash(r), Status: "excluded", Reason: "deterministic_label_documentation", Operator: "exact-prefilter-v1"}, true
		}
	}
	return issueFact{}, false
}

func issueUnitFor(r issueRecord) issueUnit {
	return issueUnit{UnitID: fmt.Sprintf("issue-%d", r.Number), SourceHash: issueSourceHash(r), Number: r.Number, Title: r.Title, Body: r.Body, Labels: append([]string(nil), r.Labels...), Milestone: r.Milestone, Rubric: largeInputRubric}
}

func semanticCacheKey(unit issueUnit) string { return unit.SourceHash + ":" + unit.Rubric }

func runLargeInputPass(ctx context.Context, name string, records []issueRecord, workers int, cache map[string]issueFact, includeFacts bool) (largeInputPass, error) {
	p := largeInputPass{Name: name, SourceHash: issueCorpusHash(records), Records: len(records), FoldFanIn: largeInputFanIn}
	facts := make([]issueFact, len(records))
	gateway := &semanticIssueGateway{}
	host, err := microagent.NewHost(gateway, microagent.Config{Workers: workers, Queue: len(records)})
	if err != nil {
		return p, err
	}
	defer host.Close()
	agents := make(map[int]*issueOperatorAgent)
	for i, record := range records {
		if fact, ok := deterministicIssueFact(record); ok {
			facts[i] = fact
			p.Deterministic++
			continue
		}
		unit := issueUnitFor(record)
		if cached, ok := cache[semanticCacheKey(unit)]; ok {
			cached.CacheHit = true
			facts[i] = cached
			p.CacheHits++
			continue
		}
		a := &issueOperatorAgent{unit: unit}
		agents[i] = a
		if err := host.Spawn(fmt.Sprintf("%s-%06d", name, i), a); err != nil {
			return p, err
		}
	}
	drainCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	if err := host.Drain(drainCtx); err != nil {
		return p, err
	}
	for _, result := range host.Reap() {
		if result.Err != nil || !result.Done {
			return p, fmt.Errorf("semantic worker %s: done=%v err=%v", result.ID, result.Done, result.Err)
		}
	}
	for i, a := range agents {
		facts[i] = a.fact
		cache[semanticCacheKey(a.unit)] = a.fact
	}
	p.SemanticCalls = int(gateway.calls.Load())
	p.PeakInFlight = gateway.peak.Load()
	seen := make(map[string]struct{}, len(facts))
	for _, fact := range facts {
		if fact.UnitID == "" {
			return p, fmt.Errorf("unaccounted source record")
		}
		if _, duplicate := seen[fact.UnitID]; duplicate {
			return p, fmt.Errorf("duplicate fact %s", fact.UnitID)
		}
		seen[fact.UnitID] = struct{}{}
		switch fact.Status {
		case "kept":
			p.Kept++
		case "excluded":
			p.Excluded++
		case "abstain":
			p.Abstained++
		case "error":
			p.Errored++
		default:
			return p, fmt.Errorf("unknown status %q", fact.Status)
		}
	}
	p.CitedIssueIDs, p.FoldLevels, p.FoldNodes, p.FoldMaxInput = foldIssueFacts(facts, largeInputFanIn)
	p.OracleIssueIDs = oracleIssueIDs(records)
	if includeFacts {
		p.Facts = facts
	}
	return p, nil
}

func foldIssueFacts(facts []issueFact, fanIn int) ([]int, int, int, int) {
	type node struct{ ids []int }
	level := make([]node, 0, len(facts))
	for _, fact := range facts {
		n := node{}
		if fact.Keep {
			var id int
			fmt.Sscanf(fact.UnitID, "issue-%d", &id)
			n.ids = []int{id}
		}
		level = append(level, n)
	}
	levels, nodes, maxInput := 0, 0, 0
	for len(level) > 1 {
		levels++
		next := make([]node, 0, (len(level)+fanIn-1)/fanIn)
		for start := 0; start < len(level); start += fanIn {
			end := start + fanIn
			if end > len(level) {
				end = len(level)
			}
			if end-start > maxInput {
				maxInput = end - start
			}
			merged := node{}
			for _, child := range level[start:end] {
				merged.ids = append(merged.ids, child.ids...)
			}
			sort.Ints(merged.ids)
			next = append(next, merged)
			nodes++
		}
		level = next
	}
	if len(level) == 0 {
		return nil, levels, nodes, maxInput
	}
	return level[0].ids, levels, nodes, maxInput
}

func oracleIssueIDs(records []issueRecord) []int {
	ids := make([]int, 0)
	for _, record := range records {
		if record.oracleKeep {
			ids = append(ids, record.Number)
		}
	}
	sort.Ints(ids)
	return ids
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func buildLargeInputReport(ctx context.Context, records, workers int) (largeInputReport, error) {
	r := largeInputReport{Schema: largeInputSchema, Mode: "fixture-backed semantic worker", Scope: "synthetic correctness/concurrency witness; not model throughput", Records: records, PhysicalWorkers: workers, Rubric: largeInputRubric, MutationUnitID: "issue-10996"}
	if records != 1000 {
		return r, fmt.Errorf("large-input selfcheck requires exactly 1000 records")
	}
	if workers < 1 || workers >= records {
		return r, fmt.Errorf("workers must be bounded below record count")
	}
	fixture := makeIssueFixture(records)
	cache := make(map[string]issueFact)
	var err error
	if r.Baseline, err = runLargeInputPass(ctx, "baseline", fixture, workers, cache, true); err != nil {
		return r, err
	}
	if r.Unchanged, err = runLargeInputPass(ctx, "unchanged", fixture, workers, cache, false); err != nil {
		return r, err
	}
	mutated := append([]issueRecord(nil), fixture...)
	mutated[996] = fixture[996]
	mutated[996].Title = "Auth regression discovered during release"
	mutated[996].Body = "Authentication path regression is a release-blocker."
	mutated[996].Labels = []string{"auth", "release-blocker"}
	mutated[996].oracleKeep = true
	if r.Mutated, err = runLargeInputPass(ctx, "mutated", mutated, workers, cache, false); err != nil {
		return r, err
	}
	r.PreciseInvalidations = r.Mutated.SemanticCalls
	r.UnchangedPureStageReuse = r.Unchanged.CacheHits
	r.SourceAccountingVerified = passAccounted(r.Baseline) && passAccounted(r.Unchanged) && passAccounted(r.Mutated)
	r.OracleVerified = sameInts(r.Baseline.CitedIssueIDs, r.Baseline.OracleIssueIDs) && sameInts(r.Unchanged.CitedIssueIDs, r.Unchanged.OracleIssueIDs) && sameInts(r.Mutated.CitedIssueIDs, r.Mutated.OracleIssueIDs)
	r.ReducerBounded = r.Baseline.FoldMaxInput <= largeInputFanIn && r.Unchanged.FoldMaxInput <= largeInputFanIn && r.Mutated.FoldMaxInput <= largeInputFanIn
	r.Notes = []string{"one immutable issue record per semantic micro-context", "deterministic documentation-label exclusions run before semantic work", "negative, abstain, error, and kept remain distinct", "all reducers have bounded fan-in; final IDs cite source issue numbers"}
	if err := verifyLargeInputReport(r); err != nil {
		r.Verdict = "FAIL"
		return r, err
	}
	r.Verdict = "PASS"
	return r, nil
}

func passAccounted(p largeInputPass) bool {
	return p.Records == p.Kept+p.Excluded+p.Abstained+p.Errored
}

func verifyLargeInputReport(r largeInputReport) error {
	if r.Schema != largeInputSchema || r.Records != 1000 {
		return fmt.Errorf("large-input schema/record contract failed")
	}
	if !r.SourceAccountingVerified || !r.OracleVerified || !r.ReducerBounded {
		return fmt.Errorf("large-input correctness witness failed")
	}
	if r.Baseline.SemanticCalls == 0 || r.Baseline.CacheHits != 0 {
		return fmt.Errorf("baseline cache witness failed")
	}
	if r.Unchanged.SemanticCalls != 0 || r.Unchanged.CacheHits != r.Baseline.SemanticCalls {
		return fmt.Errorf("unchanged reuse witness failed: calls=%d hits=%d want_hits=%d", r.Unchanged.SemanticCalls, r.Unchanged.CacheHits, r.Baseline.SemanticCalls)
	}
	if r.Mutated.SemanticCalls != 1 || r.PreciseInvalidations != 1 {
		return fmt.Errorf("one-record invalidation witness failed")
	}
	if r.Baseline.PeakInFlight > int64(r.PhysicalWorkers) || r.Mutated.PeakInFlight > int64(r.PhysicalWorkers) {
		return fmt.Errorf("physical worker bound exceeded")
	}
	if r.Baseline.Abstained == 0 || r.Baseline.Errored == 0 || r.Baseline.Excluded == 0 || r.Baseline.Kept == 0 {
		return fmt.Errorf("typed outcome coverage incomplete")
	}
	wantMutated := append(append([]int(nil), r.Baseline.CitedIssueIDs...), 10996)
	sort.Ints(wantMutated)
	if !sameInts(r.Mutated.CitedIssueIDs, wantMutated) {
		return fmt.Errorf("mutated citation set did not add issue 10996: got=%v want=%v", r.Mutated.CitedIssueIDs, wantMutated)
	}
	return nil
}

func verifyLargeInputArtifact(path string) error {
	var r largeInputReport
	b, err := osReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return err
	}
	if r.Verdict != "PASS" {
		return fmt.Errorf("artifact verdict %q", r.Verdict)
	}
	return verifyLargeInputReport(r)
}

// osReadFile is a test seam kept here so the operator implementation remains one file.
var osReadFile = func(path string) ([]byte, error) { return os.ReadFile(path) }
