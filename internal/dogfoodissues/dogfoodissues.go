package dogfoodissues

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuecohort"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
	"github.com/anthony-chaudhary/fak/internal/windowgate"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// Schema is the stable schema tag stamped on the machine-readable result.
const Schema = "fak.dogfood-action-issues.v1"

var markerRE = regexp.MustCompile(`<!--\s*fak-dogfood-action-key:\s*([^>\s]+)\s*-->`)
var issueURLRE = regexp.MustCompile(`https?://\S+/issues/([0-9]+)`)

// DefaultGhTimeout bounds each gh subprocess so a stuck network call cannot
// wedge the dogfood issue sync indefinitely.
const DefaultGhTimeout = 30 * time.Second

// DefaultMaxReportAge is the freshness ceiling before live issue sync is refused
// unless the operator explicitly overrides it.
const DefaultMaxReportAge = 24 * time.Hour

// DefaultMilestone is the intake/default roadmap bucket for machine-created
// dogfood issues. It matches docs/generation.md's current-generation milestone.
const DefaultMilestone = "Generation G0 - Now / Immediate"

// DefaultProjectSignalLabel keeps the missing-project fact visible without
// requiring this issue bridge to own GitHub Projects API plumbing.
const DefaultProjectSignalLabel = "needs-project"

// SeenCache is the persisted dedup state of ONE issue-filing surface: a schema tag plus a
// key->record map, keyed by the same stable dedup key the filed issue carries in its marker.
// R is the surface's own record type, and it is the ONLY thing that differs between the
// debt-dispatch siblings -- each records different provenance (a doc/class/exact triple, a
// scaffold id and grade, ...) while the file format, the defaulting and the write path are
// one shared contract. Two dispatchers that disagreed about what an absent or empty cache
// means would refile issues they had already filed.
type SeenCache[R any] struct {
	Schema string       `json:"schema"`
	Seen   map[string]R `json:"seen"`
}

// LoadSeen reads a SeenCache from path. A missing file, an empty file, an absent schema tag
// and a null map all default to an empty cache stamped with schema: the FIRST run of any
// dispatcher has no cache file, and a zero-length file is what a killed writer leaves behind,
// so neither may read as an error and block filing. A corrupt (non-empty, non-JSON) cache is
// reported -- the already-defaulted cache is returned alongside the error so a caller that
// chooses to continue past it still holds a usable value.
func LoadSeen[R any](path, schema string) (SeenCache[R], error) {
	cache := SeenCache[R]{Schema: schema, Seen: map[string]R{}}
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return cache, nil
	}
	if err != nil {
		return cache, err
	}
	if strings.TrimSpace(string(b)) == "" {
		return cache, nil
	}
	if err := json.Unmarshal(b, &cache); err != nil {
		return cache, err
	}
	if cache.Schema == "" {
		cache.Schema = schema
	}
	if cache.Seen == nil {
		cache.Seen = map[string]R{}
	}
	return cache, nil
}

// SaveSeen writes cache to path as indented JSON with a trailing newline, creating the parent
// directory first (the cache lives under a per-surface .fak/ subdirectory that need not exist
// yet). The schema tag and the map are defaulted on the way out for the same reason LoadSeen
// defaults them on the way in: a cache written without its schema tag would read back as an
// unversioned file.
func SaveSeen[R any](path, schema string, cache SeenCache[R]) error {
	if cache.Schema == "" {
		cache.Schema = schema
	}
	if cache.Seen == nil {
		cache.Seen = map[string]R{}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(cache, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

// ReportFreshness records the timestamp/age of the selected dogfood report.
type ReportFreshness struct {
	Timestamp     string `json:"timestamp"`
	Source        string `json:"source"`
	AgeSeconds    int64  `json:"age_seconds"`
	Age           string `json:"age"`
	MaxAgeSeconds int64  `json:"max_age_seconds"`
	MaxAge        string `json:"max_age"`
	Stale         bool   `json:"stale"`
	StaleAllowed  bool   `json:"stale_allowed"`
}

// ActionItem is one scorecard ACTION item extracted from a dogfood report.json.
type ActionItem struct {
	Key                string
	Title              string
	SourceProbe        string
	ScoreName          string
	Score              string
	Grade              string
	DebtName           string
	DebtCount          int
	EvidencePath       string
	NextAction         string
	Finding            string
	ParentRef          string
	CurrentState       string
	WhyNow             string
	WorkingSpine       string
	WorkUnit           string
	ExpectedSteps      int
	WorkEstimate       string
	ScopeContribution  string
	CompletionStandard string
	Assumptions        []string
	ConfusionRisks     []string
	Coordination       []string
	Trigger            string
	BatchPolicy        string
	InScope            string
	OutOfScope         string
	DoneCondition      string
	Witness            string
	AcceptanceGate     string
	Lane               string
	Paths              []string
	Labels             []string
	BoundaryNotes      []string
	ClosureBinding     string
}

// PlanRow is one create/update decision for a single ActionItem.
type PlanRow struct {
	Action       string             `json:"action"`
	Key          string             `json:"key"`
	Number       *int               `json:"number"`
	State        string             `json:"state"`
	Title        string             `json:"title"`
	Milestone    string             `json:"milestone,omitempty"`
	Body         string             `json:"-"`
	Score        string             `json:"score"`
	Grade        string             `json:"grade"`
	DebtCount    int                `json:"debt_count"`
	EvidencePath string             `json:"evidence_path"`
	NextAction   string             `json:"next_action"`
	Lane         string             `json:"lane,omitempty"`
	Paths        []string           `json:"paths,omitempty"`
	Labels       []string           `json:"labels,omitempty"`
	Review       issuepolicy.Review `json:"review,omitempty"`
}

// SkippedRow records a scorecard ACTION item that remains visible in the
// dogfood report, but is not scoped enough to create/update a dispatchable
// public GitHub issue.
type SkippedRow struct {
	Key             string             `json:"key"`
	Title           string             `json:"title"`
	Reason          string             `json:"reason"`
	Dispatchability string             `json:"dispatchability"`
	Review          issuepolicy.Review `json:"review,omitempty"`
}

// SyncRow is one gh create/edit outcome on a --live run.
type SyncRow struct {
	Key      string `json:"key"`
	Action   string `json:"action"`
	OK       bool   `json:"ok"`
	Number   *int   `json:"number,omitempty"`
	URL      string `json:"url,omitempty"`
	Verified bool   `json:"verified"`
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
}

// Result is the machine-readable plan/result fold.
type Result struct {
	Schema          string           `json:"schema"`
	Mode            string           `json:"mode"`
	Report          string           `json:"report"`
	ReportFreshness *ReportFreshness `json:"report_freshness,omitempty"`
	Planned         []PlanRow        `json:"planned"`
	Synced          []SyncRow        `json:"synced"`
	Skipped         []SkippedRow     `json:"skipped,omitempty"`
	Refused         bool             `json:"refused,omitempty"`
	Error           string           `json:"error,omitempty"`
	// Receipt is the path of the bridge receipt written beside the report on a
	// tracker-consulting run (--live / --fetch-existing); "" otherwise.
	Receipt string `json:"receipt,omitempty"`
	// Cohort is the batch-level wave/collision fold of the action items, computed
	// by running issuecohort.Build over the batch so a --live run's dispatch-scope
	// collisions are visible before sync rather than only at wave-launch time. nil
	// when no cohort was computed.
	Cohort *issuecohort.Plan `json:"cohort,omitempty"`
}

// Issue is the subset of a `gh issue list --json ...` row this tool reads.
type Issue struct {
	Number    int             `json:"number"`
	Title     string          `json:"title"`
	Body      string          `json:"body"`
	State     string          `json:"state"`
	URL       string          `json:"url"`
	Milestone *IssueMilestone `json:"milestone"`
}

// IssueMilestone is the subset of a GitHub milestone read through gh JSON.
type IssueMilestone struct {
	Title string `json:"title"`
}

// BuildOptions carries the producer context needed to turn scorecard ACTION
// rows into dispatchable issues. The legacy BuildPlan path is left unreviewed
// for tests and older callers; effectful callers should use BuildPlanWithOptions.
type BuildOptions struct {
	Live               bool
	DedupeChecked      bool
	DedupeCap          int
	DefaultMilestone   string
	ParentIssue        int
	ParentBaseline     float64
	CompletionStandard string
	TargetEnvelope     string
	WitnessedEnvelope  string
}

func toInt(v any, def int) int {
	switch n := v.(type) {
	case bool:
		return def
	case float64: // JSON numbers decode to float64
		return int(n)
	case int:
		return n
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return def
		}
		return int(i)
	case string:
		i, err := strconv.Atoi(strings.TrimSpace(n))
		if err != nil {
			return def
		}
		return i
	default:
		return def
	}
}

func toStr(v any, def string) string {
	if v == nil {
		return def
	}
	switch s := v.(type) {
	case string:
		return s
	case float64:
		// Mirror Python str(float): drop a trailing ".0" so 71.5 stays "71.5"
		// and 12.0 becomes "12".
		return strconv.FormatFloat(s, 'f', -1, 64)
	case bool:
		if s {
			return "True"
		}
		return "False"
	default:
		return fmt.Sprintf("%v", s)
	}
}

func asMap(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return nil
}

// LoadReport parses a dogfood report.json file into a generic object fold.
func LoadReport(path string) (map[string]any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var data any
	if err := json.Unmarshal(b, &data); err != nil {
		return nil, err
	}
	m, ok := data.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("report must be a JSON object")
	}
	return m, nil
}

// ReportFreshnessForFile computes freshness from the selected report's mtime,
// matching the newest-report selection semantics used by the Python bridge.
func ReportFreshnessForFile(path string, now time.Time, maxAge time.Duration, allowStale bool) (ReportFreshness, error) {
	info, err := os.Stat(path)
	if err != nil {
		return ReportFreshness{}, err
	}
	if now.IsZero() {
		now = time.Now()
	}
	if maxAge < 0 {
		maxAge = 0
	}
	mtime := info.ModTime().UTC()
	age := now.UTC().Sub(mtime)
	if age < 0 {
		age = 0
	}
	return ReportFreshness{
		Timestamp:     mtime.Format(time.RFC3339),
		Source:        "mtime",
		AgeSeconds:    int64(age / time.Second),
		Age:           FormatReportDuration(age),
		MaxAgeSeconds: int64(maxAge / time.Second),
		MaxAge:        FormatReportDuration(maxAge),
		Stale:         age > maxAge,
		StaleAllowed:  allowStale,
	}, nil
}

// StaleReportMessage is the human-readable gate explanation shared by the CLI
// and renderer.
func StaleReportMessage(f ReportFreshness) string {
	return fmt.Sprintf("selected dogfood report is stale (age %s > max %s); rerun dogfood or pass --allow-stale-report to continue",
		f.Age, f.MaxAge)
}

// FormatReportDuration renders a short duration suitable for terminal output.
func FormatReportDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	seconds := int64(d / time.Second)
	if seconds < 60 {
		return fmt.Sprintf("%ds", seconds)
	}
	minutes := seconds / 60
	sec := seconds % 60
	if minutes < 60 {
		if sec == 0 {
			return fmt.Sprintf("%dm", minutes)
		}
		return fmt.Sprintf("%dm %ds", minutes, sec)
	}
	hours := minutes / 60
	minute := minutes % 60
	if hours < 24 {
		if minute == 0 {
			return fmt.Sprintf("%dh", hours)
		}
		return fmt.Sprintf("%dh %dm", hours, minute)
	}
	days := hours / 24
	hour := hours % 24
	if hour == 0 {
		return fmt.Sprintf("%dd", days)
	}
	return fmt.Sprintf("%dd %dh", days, hour)
}

// Invariant: dogfood issues extraction is fail-closed and deterministic.
// Missing or corrupt probes, non-ACTION probes, and zero-debt findings produce zero action items.
//
// ExtractActionItems folds the report's probes into the scorecard ACTION items
// that warrant a tracked backlog issue. reportPath is recorded as the evidence
// path on each item.
func ExtractActionItems(report map[string]any, reportPath string) []ActionItem {
	evidence := reportPath
	var items []ActionItem
	probes, _ := report["probes"].([]any)
	for _, p := range probes {
		probe := asMap(p)
		if probe == nil {
			continue
		}
		key := toStr(probe["key"], "")
		payload := asMap(probe["payload"])
		if payload == nil {
			continue
		}
		schema := toStr(payload["schema"], "")

		if key == "code-slop-scorecard" || schema == "fleet-code-slop-scorecard/1" {
			corpus := asMap(payload["corpus"])
			if corpus == nil {
				corpus = map[string]any{}
			}
			debt := toInt(corpus["slop_debt"], toInt(payload["slop_debt"], 0))
			verdict := toStr(payload["verdict"], "")
			if verdict != "ACTION" && debt <= 0 {
				continue
			}
			finding := toStr(payload["finding"], "code_slop")
			items = append(items, ActionItem{
				Key:          "recent-feature-dogfood/code-slop-scorecard/" + finding,
				Title:        "dogfood ACTION: code-slop scorecard debt",
				SourceProbe:  key,
				ScoreName:    "slop_score",
				Score:        toStr(corpus["score"], toStr(payload["score"], "?")),
				Grade:        toStr(corpus["grade"], toStr(payload["grade"], "?")),
				DebtName:     "slop_debt",
				DebtCount:    debt,
				EvidencePath: evidence,
				NextAction: toStr(payload["next_action"],
					"Retire code-slop debt worst-first, then rerun the dogfood packet."),
				Finding:            finding,
				ParentRef:          "#6738",
				CurrentState:       fmt.Sprintf("The code-slop scorecard reports %d hard defect(s); the current worst KPI is duplication.", debt),
				WhyNow:             "The recent-feature dogfood packet is red until one bounded, witnessed slop slice is dispatched and retired.",
				WorkingSpine:       "Select the highest-payoff extractable duplication family, remove exactly that clone family without behavior change, and rerun the scorecard.",
				WorkUnit:           "leaf",
				ExpectedSteps:      4,
				WorkEstimate:       "Estimate: 4 points",
				ScopeContribution:  "Contribution: 4/4 points",
				CompletionStandard: "development",
				InScope:            "One highest-payoff extractable duplication family, its behavior-preserving helper/deletion, affected package tests, and before/after scorecard JSON.",
				OutOfScope:         "Do not redesign the scorecard, re-pin its baseline, clear the full portfolio, or sweep unrelated cleanup.",
				DoneCondition:      "One witnessed duplication family is removed and duplication debt strictly decreases without increasing another hard KPI.",
				Witness:            "Run affected Go package tests in a clean overlay and compare `python tools/code_slop_scorecard.py --json --no-toolchain` before/after.",
				AcceptanceGate:     "Affected package tests pass and the after scorecard reports strictly lower duplication debt with no new hard defect.",
				Lane:               "tools",
				Paths:              []string{"tools/code_slop_scorecard.py", "internal/**", "cmd/**"},
				Labels:             []string{"gen/now", "priority/P1", "class:dev"},
				BoundaryNotes:      []string{"Choose and name one clone family before editing; do not let the generated issue become an unbounded cleanup epic."},
			})
			continue
		}

		if key == "dogfood-coverage-scorecard" || schema == "dogfood-coverage/1" {
			hardDebt := toInt(payload["dogfood_debt"], 0)
			worstFirst, _ := payload["worst_first"].([]any)
			softDebt := 0
			for _, x := range worstFirst {
				if strings.TrimSpace(toStr(x, "")) != "" {
					softDebt++
				}
			}
			grade := toStr(payload["grade"], "")
			if hardDebt <= 0 && softDebt <= 0 && (grade == "" || grade == "A") {
				continue
			}
			debt := softDebt
			if hardDebt > 0 {
				debt = hardDebt
			}
			action := "Raise dogfood coverage to grade A, then rerun the dogfood packet."
			if len(worstFirst) > 0 {
				n := len(worstFirst)
				if n > 5 {
					n = 5
				}
				parts := make([]string, 0, n)
				for _, x := range worstFirst[:n] {
					parts = append(parts, toStr(x, ""))
				}
				action = "Address dogfood coverage gaps worst-first: " + strings.Join(parts, ", ")
			}
			gradeOut := grade
			if gradeOut == "" {
				gradeOut = "?"
			}
			items = append(items, ActionItem{
				Key:          "recent-feature-dogfood/dogfood-coverage-scorecard/dogfood_coverage",
				Title:        "dogfood ACTION: dogfood coverage gap",
				SourceProbe:  key,
				ScoreName:    "coverage",
				Score:        toStr(payload["coverage"], "?"),
				Grade:        gradeOut,
				DebtName:     "dogfood_debt_or_gaps",
				DebtCount:    debt,
				EvidencePath: evidence,
				NextAction:   action,
				Finding:      "dogfood_coverage",
			})
		}
	}
	return items
}

// IssueBody renders the stable, marker-stamped issue body for an item.
func IssueBody(item ActionItem) string { return IssueBodyWithOptions(item, BuildOptions{}) }
func IssueBodyWithOptions(item ActionItem, opt BuildOptions) string {
	c := actionCandidate(item, opt)
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- fak-dogfood-action-key: %s -->\n", item.Key)
	fmt.Fprintln(&b, "# Dogfood scorecard ACTION")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- Stable key: `%s`\n", item.Key)
	fmt.Fprintf(&b, "- Source probe: `%s`\n", item.SourceProbe)
	fmt.Fprintf(&b, "- Finding: `%s`\n", item.Finding)
	fmt.Fprintf(&b, "- %s: `%s`\n", item.ScoreName, item.Score)
	fmt.Fprintf(&b, "- grade: `%s`\n", item.Grade)
	fmt.Fprintf(&b, "- %s: `%d`\n", item.DebtName, item.DebtCount)
	fmt.Fprintf(&b, "- Evidence path: `%s`\n", item.EvidencePath)
	if item.Lane != "" {
		fmt.Fprintf(&b, "- Lane: `%s`\n", item.Lane)
	}
	if len(item.Paths) > 0 {
		fmt.Fprintln(&b, "- Path hints:")
		for _, p := range item.Paths {
			fmt.Fprintf(&b, "  - `%s`\n", p)
		}
	}
	fmt.Fprintln(&b)
	issueSection(&b, "Current state", c.CurrentState)
	issueSection(&b, "Why this is next", c.WhyNow)
	issueSection(&b, "Working spine", item.WorkingSpine)
	issueSection(&b, "Priority context", c.PriorityContext)
	issueSection(&b, "Work unit", c.WorkUnit)
	if c.ExpectedSteps > 0 {
		issueSection(&b, "Expected steps", strconv.Itoa(c.ExpectedSteps))
	} else {
		issueSection(&b, "Expected steps", "Not specified.")
	}
	issueListSection(&b, "Assumptions", c.Assumptions)
	issueListSection(&b, "Confusion risks", c.ConfusionRisks)
	issueListSection(&b, "Coordination", c.Coordination)
	issueSection(&b, "Trigger", c.Trigger)
	issueSection(&b, "Batch policy", c.BatchPolicy)
	issueSection(&b, "Core through-line", item.InScope)
	issueSection(&b, "Gold-plating boundary", item.OutOfScope)
	issueProblemFrame(&b, c.ProblemFrame)
	issueSection(&b, "Done condition", item.DoneCondition)
	issueSection(&b, "Witness", item.Witness)
	issueSection(&b, "Acceptance gate", item.AcceptanceGate)
	issueSection(&b, "Work estimate", c.WorkEstimate)
	issueSection(&b, "Overall completion contribution", c.ScopeContribution)
	issueSection(&b, "Completion standard", c.CompletionStandard)
	if c.TargetEnvelope != "" {
		issueSection(&b, "Target operating envelope", c.TargetEnvelope)
	}
	if c.WitnessedEnvelope != "" {
		issueSection(&b, "Witnessed operating envelope", c.WitnessedEnvelope)
	}
	issueListSection(&b, "Boundary notes", item.BoundaryNotes)
	issueSection(&b, "Closure binding", c.ClosureBinding)
	fmt.Fprintln(&b, "Suggested next action:")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, strmatch.FirstTrimmed(item.NextAction, "Triage this scorecard ACTION row into a scoped, witness-backed leaf before dispatch."))
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "This issue is managed by `fak dogfood-issues`. Re-running the helper updates this issue in place instead of opening duplicates.")
	return b.String()
}

// ReviewActionItem grades one ACTION item against the shared machine-created
// issue contract.
func ReviewActionItem(item ActionItem, opt BuildOptions) issuepolicy.Review {
	c := actionCandidate(item, opt)
	review := issuepolicy.ReviewCandidate(c, strictScopeOptions(opt))
	return applyStrictScopeHold(review, strictScopeHold(item, opt))
}

func actionCandidate(item ActionItem, opt BuildOptions) issuepolicy.Candidate {
	scoreState := fmt.Sprintf("Source probe `%s` reported finding `%s`, grade `%s`, and %s `%d`.",
		item.SourceProbe, item.Finding, item.Grade, item.DebtName, item.DebtCount)
	c := issuepolicy.Candidate{
		Schema:             issuepolicy.Schema,
		ProblemFrame:       dogfoodProblemFrame(),
		Key:                item.Key,
		Title:              item.Title,
		ParentRef:          strmatch.FirstTrimmed(item.ParentRef, "fak dogfood-issues"),
		CurrentState:       strmatch.FirstTrimmed(item.CurrentState, scoreState),
		WhyNow:             strmatch.FirstTrimmed(item.WhyNow, "The recent-feature dogfood report emitted an ACTION/debt row for this scorecard."),
		WorkingSpine:       item.WorkingSpine,
		PriorityContext:    dogfoodPriorityContext(item),
		WorkUnit:           strmatch.FirstTrimmed(item.WorkUnit, "leaf"),
		ExpectedSteps:      firstPositive(item.ExpectedSteps, 4),
		WorkEstimate:       item.WorkEstimate,
		ScopeContribution:  item.ScopeContribution,
		CompletionStandard: item.CompletionStandard,
		Assumptions: appendDefault(item.Assumptions,
			"The source scorecard row is still current when the worker starts."),
		ConfusionRisks: appendDefault(item.ConfusionRisks,
			"Do not treat this generated action row as a broad scorecard epic."),
		Coordination: appendDefault(item.Coordination,
			"Use the stable marker key before creating siblings so reruns update in place."),
		Trigger:        strmatch.FirstTrimmed(item.Trigger, dogfoodIssueTrigger(item)),
		BatchPolicy:    strmatch.FirstTrimmed(item.BatchPolicy, "One issue per stable dogfood action key; reruns update the existing marker instead of opening duplicates."),
		InScope:        item.InScope,
		OutOfScope:     item.OutOfScope,
		DoneCondition:  item.DoneCondition,
		Witness:        item.Witness,
		AcceptanceGate: item.AcceptanceGate,
		Lane:           item.Lane,
		Paths:          append([]string(nil), item.Paths...),
		Labels:         append([]string(nil), item.Labels...),
		BoundaryNotes:  append([]string(nil), item.BoundaryNotes...),
		ClosureBinding: strmatch.FirstTrimmed(item.ClosureBinding, "Resolving commit must cite `#N` and carry a matching `(fak <leaf>)` trailer."),
	}
	if opt.ParentIssue > 0 {
		c.ParentRef = fmt.Sprintf("#%d", opt.ParentIssue)
	}
	if opt.ParentBaseline > 0 {
		points := float64(c.ExpectedSteps)
		c.WorkEstimate = fmt.Sprintf("Estimate: %g points", points)
		c.ScopeContribution = fmt.Sprintf("Contribution: %g/%g points", points, opt.ParentBaseline)
		c.CompletionStandard = strmatch.FirstTrimmed(opt.CompletionStandard, "production")
	}
	if c.CompletionStandard != "" {
		c.TargetEnvelope = opt.TargetEnvelope
		c.WitnessedEnvelope = opt.WitnessedEnvelope
	}
	return c
}
func dogfoodProblemFrame() issuepolicy.ProblemFrame {
	checks := map[string]issuepolicy.ProblemCheck{
		"p1": {ID: "p1", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "the action preserves its source probe, finding, debt, and stable marker context", Valid: true},
		"p2": {ID: "p2", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "measured dogfood debt becomes one bounded improvement instead of recurring manual triage", Valid: true},
		"p3": {ID: "p3", Status: issuepolicy.ProblemCheckPreserved, Evidence: "the generated leaf remains independently reviewable and reruns update by stable key", Valid: true},
		"p4": {ID: "p4", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "the scorecard action flows through shared candidate review and dispatch", Valid: true},
	}
	return issuepolicy.ProblemFrame{Schema: issuepolicy.ProblemFrameSchema, Ready: true, Enforced: true, Centrality: issuepolicy.CentralityEnabling, CentralityTarget: "improve the kernel outcome measured by the source dogfood scorecard", Checks: checks}
}

func dogfoodIssueTrigger(item ActionItem) string {
	probe := strmatch.FirstTrimmed(item.SourceProbe, "dogfood scorecard")
	finding := strmatch.FirstTrimmed(item.Finding, item.Key, "ACTION row")
	return fmt.Sprintf("Scorecard probe `%s` emitted ACTION finding `%s`.", probe, finding)
}

func dogfoodPriorityContext(item ActionItem) string {
	spine := strmatch.FirstTrimmed(item.WorkingSpine, "Retire the scorecard ACTION row on the smallest witnessed path.")
	current := fmt.Sprintf("Scorecard `%s` reports `%s` with %s `%d`.", item.SourceProbe, item.Finding, item.DebtName, item.DebtCount)
	return strings.Join([]string{
		"Working path: " + spine,
		"Current blocker: " + current,
		"Unblocks: resolving this ACTION row keeps recent-feature dogfood from hiding a real breakage.",
		"Not polish: address the named probe row before broad dogfood expansion or optimization.",
	}, "\n")
}

func appendDefault(items []string, fallback string) []string {
	out := append([]string(nil), items...)
	if len(out) == 0 && strings.TrimSpace(fallback) != "" {
		out = append(out, fallback)
	}
	return out
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func issueProblemFrame(b *strings.Builder, frame issuepolicy.ProblemFrame) {
	fmt.Fprintln(b, "## Problem frame")
	fmt.Fprintln(b)
	fmt.Fprintf(b, "- Centrality: %s (%s)\n", frame.Centrality, frame.CentralityTarget)
	for _, id := range []string{"p1", "p2", "p3", "p4"} {
		check := frame.Checks[id]
		fmt.Fprintf(b, "- %s: %s - %s\n", strings.ToUpper(id), check.Status, check.Evidence)
	}
	fmt.Fprintln(b)
}

func issueSection(b *strings.Builder, title, body string) {
	fmt.Fprintf(b, "## %s\n\n", title)
	body = strings.TrimSpace(body)
	if body == "" {
		body = "Not specified."
	}
	fmt.Fprintln(b, body)
	fmt.Fprintln(b)
}

func issueListSection(b *strings.Builder, title string, items []string) {
	fmt.Fprintf(b, "## %s\n\n", title)
	if len(items) == 0 {
		fmt.Fprintln(b, "No boundary notes supplied.")
		fmt.Fprintln(b)
		return
	}
	for _, item := range items {
		if s := strings.TrimSpace(item); s != "" {
			fmt.Fprintf(b, "- %s\n", s)
		}
	}
	fmt.Fprintln(b)
}

// MarkerKey extracts the stable key from an issue body's HTML-comment marker,
// or "" when absent.
func MarkerKey(body string) string {
	m := markerRE.FindStringSubmatch(body)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

// existingByKey indexes existing issues by their marker key.
func existingByKey(issues []Issue) map[string]Issue {
	out := map[string]Issue{}
	for _, issue := range issues {
		key := MarkerKey(issue.Body)
		if key != "" {
			out[key] = issue
		}
	}
	return out
}

// BuildPlan decides create vs update for each item against the existing issues
// (matched by marker key).
func BuildPlan(items []ActionItem, existing []Issue) []PlanRow {
	byKey := existingByKey(existing)
	plan := make([]PlanRow, 0, len(items))
	for _, item := range items {
		found, ok := byKey[item.Key]
		row := planRow(item)
		if ok {
			row.Action = "update"
			n := found.Number
			row.Number = &n
			row.State = found.State
		}
		plan = append(plan, row)
	}
	return plan
}

// BuildPlanWithOptions is BuildPlan plus the shared issue-candidate contract.
// Non-OK candidates are returned as skipped rows instead of being synced as vague
// public issues.
func BuildPlanWithOptions(items []ActionItem, existing []Issue, opt BuildOptions) ([]PlanRow, []SkippedRow) {
	byKey := existingByKey(existing)
	plan := make([]PlanRow, 0, len(items))
	skipped := []SkippedRow{}
	for _, item := range items {
		review := ReviewActionItem(item, opt)
		if !review.OK {
			skipped = append(skipped, SkippedRow{
				Key:             item.Key,
				Title:           item.Title,
				Reason:          strings.Join(review.Reasons, ","),
				Dispatchability: review.Dispatchability,
				Review:          review,
			})
			continue
		}
		row := planRowWithOptions(item, opt)
		row.Review = review
		if found, ok := byKey[item.Key]; ok {
			row.Action = "update"
			n := found.Number
			row.Number = &n
			row.State = found.State
		}
		plan = append(plan, row)
	}
	return plan, skipped
}

func planRow(item ActionItem) PlanRow {
	return planRowWithOptions(item, BuildOptions{})
}

// CohortPlan folds the batch's action items through issuecohort.Build so a
// producer Result can carry the wave/collision structure of the planned batch.
// The cohort planner re-reviews each item through the same issuecontract the
// per-row path uses, so its dispatchable leaves are exactly the planned rows,
// partitioned into concurrency-safe waves; collision_pairs counts the
// dispatch-scope overlaps a --live sync would otherwise discover only at
// wave-launch time.
func CohortPlan(items []ActionItem, opt BuildOptions) issuecohort.Plan {
	candidates := make([]issuepolicy.Candidate, 0, len(items))
	for _, item := range items {
		candidate := actionCandidate(item, opt)
		if len(strictScopeHold(item, opt)) > 0 {
			// issuecohort reviews candidates directly, so make the stricter
			// lane-and-path hold visible to its shared route check too.
			candidate.Lane = ""
			candidate.Paths = nil
		}
		candidates = append(candidates, candidate)
	}
	return issuecohort.Build(candidates, issuecohort.Options{
		Options: strictScopeOptions(opt),
	})
}

func planRowWithOptions(item ActionItem, opt BuildOptions) PlanRow {
	milestone := strings.TrimSpace(opt.DefaultMilestone)
	if milestone == "" {
		milestone = DefaultMilestone
	}
	return PlanRow{
		Action:       "create",
		Key:          item.Key,
		Title:        item.Title,
		Milestone:    milestone,
		Body:         IssueBodyWithOptions(item, opt),
		Score:        item.Score,
		Grade:        item.Grade,
		DebtCount:    item.DebtCount,
		EvidencePath: item.EvidencePath,
		NextAction:   item.NextAction,
		Lane:         item.Lane,
		Paths:        append([]string(nil), item.Paths...),
		Labels:       mergeDogfoodIssueLabels(item.Labels, []string{DefaultProjectSignalLabel}),
	}
}

// Runner runs a `gh` subprocess and returns its stdout, stderr, and an ok flag
// (true when the process exited 0). It is injectable so Sync is testable without
// a real gh.
type Runner func(args []string) (stdout, stderr string, ok bool)

// SyncOptions tunes effectful sync behavior. Zero values use conservative
// defaults.
type SyncOptions struct {
	Timeout time.Duration
}

// defaultRunner shells out to the real `gh` CLI.
func defaultRunner(args []string) (string, string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), DefaultGhTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		stderr := strings.TrimSpace(errb.String())
		if stderr != "" {
			stderr += "\n"
		}
		stderr += fmt.Sprintf("gh timed out after %s", DefaultGhTimeout)
		return out.String(), stderr, false
	}
	return out.String(), errb.String(), err == nil
}

// FetchExistingIssues queries `gh issue list` for the existing issues to classify
// create vs update. repo "" uses the current repo.
func FetchExistingIssues(repo string, limit int) ([]Issue, error) {
	args := []string{"issue", "list", "--state", "all", "--limit", strconv.Itoa(limit),
		"--json", "number,title,body,state,url,milestone"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	stdout, stderr, ok := defaultRunner(args)
	if !ok {
		return nil, fmt.Errorf("gh %s: %s", strings.Join(args, " "), strings.TrimSpace(stderr))
	}
	if strings.TrimSpace(stdout) == "" {
		return nil, nil
	}
	var issues []Issue
	if err := json.Unmarshal([]byte(stdout), &issues); err != nil {
		return nil, err
	}
	return issues, nil
}

// Sync creates or edits each planned issue via gh. The body is passed inline with
// --body so no temp file is needed. runner defaults to the real gh CLI when nil.
func Sync(plan []PlanRow, repo string, labels []string, runner Runner) []SyncRow {
	return SyncWithOptions(plan, repo, labels, runner, SyncOptions{})
}

// SyncWithOptions is Sync plus testable timeout control. Each create/edit is
// followed by a marker read-back (`gh issue view`) before the row is marked OK.
func SyncWithOptions(plan []PlanRow, repo string, labels []string, runner Runner, opt SyncOptions) []SyncRow {
	run := runner
	if run == nil {
		run = defaultRunner
	}
	timeout := opt.Timeout
	if timeout <= 0 {
		timeout = DefaultGhTimeout
	}
	call := func(args []string) (string, string, bool) {
		return runWithTimeout(run, args, timeout)
	}
	results := make([]SyncRow, 0, len(plan))
	for _, row := range plan {
		var args []string
		if row.Action == "update" {
			if row.Number == nil {
				results = append(results, SyncRow{
					Key:    row.Key,
					Action: row.Action,
					OK:     false,
					Stderr: "cannot update issue: missing issue number",
				})
				continue
			}
			num := strconv.Itoa(*row.Number)
			args = []string{"issue", "edit", num, "--title", row.Title, "--body", row.Body}
		} else {
			args = []string{"issue", "create", "--title", row.Title, "--body", row.Body}
			for _, label := range mergeDogfoodIssueLabels(row.Labels, labels) {
				args = append(args, "--label", label)
			}
		}
		if strings.TrimSpace(row.Milestone) != "" {
			args = append(args, "--milestone", strings.TrimSpace(row.Milestone))
		}
		if repo != "" {
			args = append(args, "--repo", repo)
		}
		stdout, stderr, ok := call(args)
		number := row.Number
		url := ""
		verified := false
		if ok {
			if row.Action == "create" {
				var parsed int
				var parsedURL string
				var parsedOK bool
				parsed, parsedURL, parsedOK = CreatedIssue(stdout)
				if !parsedOK {
					ok = false
					stderr = appendSyncStderr(stderr, "gh issue create succeeded but stdout did not contain a created issue URL")
				} else {
					number = &parsed
					url = parsedURL
				}
			}
		}
		if ok && number != nil {
			var verifyErr string
			var verifiedURL string
			verifiedURL, verifyErr, verified = verifySyncedIssue(call, repo, row, *number)
			if verifiedURL != "" {
				url = verifiedURL
			}
			if !verified {
				ok = false
				stderr = appendSyncStderr(stderr, verifyErr)
			}
		}
		results = append(results, SyncRow{
			Key:      row.Key,
			Action:   row.Action,
			OK:       ok,
			Number:   number,
			URL:      url,
			Verified: verified,
			Stdout:   strings.TrimSpace(stdout),
			Stderr:   strings.TrimSpace(stderr),
		})
	}
	return results
}

func runWithTimeout(run Runner, args []string, timeout time.Duration) (string, string, bool) {
	if timeout <= 0 {
		return run(args)
	}
	type result struct {
		stdout, stderr string
		ok             bool
	}
	ch := make(chan result, 1)
	go func() {
		stdout, stderr, ok := run(args)
		ch <- result{stdout: stdout, stderr: stderr, ok: ok}
	}()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case r := <-ch:
		return r.stdout, r.stderr, r.ok
	case <-timer.C:
		return "", fmt.Sprintf("gh timed out after %s", timeout), false
	}
}

// CreatedIssue parses the URL gh prints after `gh issue create`.
func CreatedIssue(stdout string) (number int, url string, ok bool) {
	match := issueURLRE.FindStringSubmatch(strings.TrimSpace(stdout))
	if match == nil {
		return 0, "", false
	}
	n, err := strconv.Atoi(match[1])
	if err != nil {
		return 0, "", false
	}
	return n, match[0], true
}

func verifySyncedIssue(run Runner, repo string, row PlanRow, number int) (url, stderr string, ok bool) {
	fields := "number,title,body,state,url"
	if strings.TrimSpace(row.Milestone) != "" {
		fields += ",milestone"
	}
	args := []string{"issue", "view", strconv.Itoa(number), "--json", fields}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	stdout, stderr, runOK := run(args)
	if !runOK {
		return "", strings.TrimSpace(stderr), false
	}
	var issue Issue
	if err := json.Unmarshal([]byte(stdout), &issue); err != nil {
		return "", fmt.Sprintf("verify issue #%d: parse gh issue view JSON: %v", number, err), false
	}
	if issue.Number != 0 && issue.Number != number {
		return issue.URL, fmt.Sprintf("verify issue #%d: gh returned issue #%d", number, issue.Number), false
	}
	if got := MarkerKey(issue.Body); got != row.Key {
		return issue.URL, fmt.Sprintf("verify issue #%d: marker key %q != planned key %q", number, got, row.Key), false
	}
	if want := strings.TrimSpace(row.Milestone); want != "" {
		if issue.Milestone == nil || strings.TrimSpace(issue.Milestone.Title) != want {
			got := ""
			if issue.Milestone != nil {
				got = issue.Milestone.Title
			}
			return issue.URL, fmt.Sprintf("verify issue #%d: milestone %q != planned milestone %q", number, got, want), false
		}
	}
	return issue.URL, "", true
}

func appendSyncStderr(stderr, msg string) string {
	stderr = strings.TrimSpace(stderr)
	msg = strings.TrimSpace(msg)
	if stderr == "" {
		return msg
	}
	if msg == "" {
		return stderr
	}
	return stderr + "\n" + msg
}

func mergeDogfoodIssueLabels(base, extra []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, group := range [][]string{base, extra} {
		for _, label := range group {
			label = strings.TrimSpace(label)
			if label == "" || seen[label] {
				continue
			}
			seen[label] = true
			out = append(out, label)
		}
	}
	return out
}

// Render produces the human-readable summary of a plan/result.
func Render(r Result) string {
	lines := []string{
		fmt.Sprintf("dogfood-action-issues: %s  %d item(s)", r.Mode, len(r.Planned)),
		fmt.Sprintf("  report: %s", r.Report),
	}
	if r.ReportFreshness != nil {
		stale := "no"
		if r.ReportFreshness.Stale {
			stale = "yes"
		}
		lines = append(lines,
			fmt.Sprintf("  report timestamp: %s (source=%s)", r.ReportFreshness.Timestamp, r.ReportFreshness.Source),
			fmt.Sprintf("  report age: %s  max=%s  stale=%s", r.ReportFreshness.Age, r.ReportFreshness.MaxAge, stale),
		)
		if r.ReportFreshness.Stale {
			msg := StaleReportMessage(*r.ReportFreshness)
			if r.ReportFreshness.StaleAllowed {
				msg += " (--allow-stale-report override active)"
			}
			lines = append(lines, "  STALE report: "+msg)
		}
	}
	if r.Refused {
		if strings.TrimSpace(r.Error) != "" {
			lines = append(lines, "  refused: "+r.Error)
		}
		return strings.Join(lines, "\n")
	}
	if len(r.Skipped) > 0 {
		lines = append(lines, fmt.Sprintf("  skipped-contract: %d item(s)", len(r.Skipped)))
		for _, row := range r.Skipped {
			lines = append(lines, fmt.Sprintf("    key=%s: %s", row.Key, row.Reason))
		}
	}
	if len(r.Planned) == 0 {
		lines = append(lines, "  no dispatchable scorecard ACTION items found")
		if r.Receipt != "" {
			lines = append(lines, "  receipt: "+r.Receipt)
		}
		return strings.Join(lines, "\n")
	}
	for _, row := range r.Planned {
		target := "new issue"
		if row.Number != nil {
			target = "#" + strconv.Itoa(*row.Number)
		}
		lines = append(lines, fmt.Sprintf("  [%s] %s: %s (grade=%s debt=%d)",
			row.Action, target, row.Title, row.Grade, row.DebtCount))
	}
	if r.Cohort != nil {
		lines = append(lines, fmt.Sprintf("  cohort: %d wave(s), peak %d at once, %d colliding pair(s)",
			r.Cohort.NumWaves, r.Cohort.PeakConcurrency, r.Cohort.CollisionPairs))
		if r.Cohort.CollisionPairs > 0 {
			lines = append(lines, "  cohort: colliding batch - review wave split before --live")
		}
	}
	if r.Mode == "dry-run" {
		lines = append(lines, "  dry-run: pass --live to create/update issues with gh")
	}
	if r.Receipt != "" {
		lines = append(lines, "  receipt: "+r.Receipt)
	}
	return strings.Join(lines, "\n")
}
