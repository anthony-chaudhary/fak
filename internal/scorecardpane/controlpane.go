// Package scorecardpane is the native Go port of the two highest-frequency
// scorecard folds the family still ran in Python: the portfolio control-pane
// fold (tools/scorecard_control_pane.py) and the repo-hygiene scorecard fold
// (tools/repo_hygiene_scorecard.py).
//
// This file ports the CONTROL-PANE fold: read each per-scorecard control-pane
// payload (schema/ok/verdict/finding/reason/next_action + a *_debt integer),
// preserve every raw debt in its metric-specific unit, compute an observational
// heterogeneous total plus a scale-invariant grade_debt companion, compare only
// same-metric/same-detector-version baselines, and emit the ratchet verdict.
//
// The pure surface (the tested core) is: MetricFromPayload, Fold, ComputeTrend,
// CheckGate, BaselineDoc, and the three-tier grade derivation. The impure shell
// (running each scorecard as a subprocess) lives in collect.go.
package scorecardpane

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema identifiers, byte-identical to the Python tool's constants so a
// consumer keyed on the schema string does not need to special-case the port.
const (
	Schema         = "fak-scorecard-control-pane/1"
	BaselineSchema = "fak-scorecard-control-pane.baseline/2"
	// LegacyBaselineSchema predates per-metric detector-version pins. It remains
	// readable so an existing baseline can be migrated intentionally on --pin.
	LegacyBaselineSchema = "fak-scorecard-control-pane.baseline/1"
	// TotalDebtSemantics labels the compatibility total everywhere it appears. The
	// component integers have different detector-specific units, so their sum is a
	// useful observational inventory only, never a quality quantity or gate.
	TotalDebtSemantics = "observational heterogeneous sum; not a comparable quality unit or gate"
	// MetricComparisonBasis is the only valid raw-debt ratchet comparison.
	MetricComparisonBasis = "same metric + same detector version"
	// BaselineRel is the tracked baseline file the trend is pinned in.
	BaselineRel = "tools/scorecard_baseline.json"
	// GradeRatchetEnv demotes the native grade ratchet to advisory when set to 0/false/no/off.
	GradeRatchetEnv = "FAK_SCORECARD_GRADE_RATCHET"
)

// gradeDebt maps a letter grade to the severity weight one metric contributes to
// grade_debt. Identical to the Python GRADE_DEBT table: a slop A->B regression
// weighs exactly as much as a stability A->B regression (units-invariant).
var gradeDebt = map[string]int{"A": 0, "B": 1, "C": 2, "D": 4, "F": 8}

// scoreKeys maps the metric key -> the legacy corpus-level aggregate score field for
// Python-era scorecards that grade per-item but emit no corpus-level letter
// (docs/seo/demo/robustness/learning). New scorecards should emit corpus.value; this
// map remains only as a fallback while Python cards are migrated.
var scoreKeys = map[string]string{
	"doc":        "mean_score",
	"seo":        "overall_score",
	"demo":       "mean_score",
	"robustness": "mean_score",
	"learning":   "mean_score",
}

// Card binds a scorecard's debt key to the script or go-run command that emits it.
// The fold cares only about Key/Label/Debt; Script/Cmd drive the impure runner.
type Card struct {
	Key    string
	Debt   string
	Label  string
	Script string
	Cmd    string
	// Corpus is the set of repo-relative file globs whose change could move this
	// card's debt — its shift-left carry key. On a --since fold a card whose Corpus
	// is disjoint from the diff is CARRIED from the baseline instead of re-measured.
	// It MUST be a SUPERSET of everything the card actually reads: an over-broad glob
	// only costs an unnecessary re-measure, but an under-broad one would silently
	// carry a stale number (a false skip). A card is annotated only once its corpus
	// is a verified pure function of tracked tree files; an empty Corpus means
	// "always measure" (the fail-safe default — never carried), which is correct for
	// cards that read out-of-tree inputs (session logs, git history, live state).
	Corpus []string
	// Origin identifies the producer seam where a cheap Go-backed card can run
	// before the aggregate pane. It is deliberately limited to a command and a
	// repo-relative owning file: no argv, environment, or payload is exposed.
	Origin *ProducerOrigin
}

// ProducerOrigin is the stable at-origin hook metadata consumed by producers.
type ProducerOrigin struct {
	Command string `json:"command"`
	File    string `json:"file"`
}

// Result is one collected scorecard payload paired with the card metadata needed
// by downstream autopost/dedupe sinks.
type Result struct {
	Card Card
	Raw  []byte
}

// Cards is the scorecard family in the canonical order the Python tool lists them.
// The fold folds every Debt key into one portfolio number. GoBacked is derived
// from a non-empty Cmd containing "go run".
var Cards = []Card{
	{Key: "doc", Debt: "doc_debt", Script: "docs_scorecard.py", Label: "docs", Corpus: []string{"**/*.md", "**/*.txt"}},
	{Key: "readme", Debt: "readme_debt", Script: "readme_freshness_audit.py", Label: "readme-freshness"},
	{Key: "code", Debt: "code_debt", Script: "code_quality_scorecard.py", Label: "code", Corpus: []string{"**/*.go", "go.mod", "go.sum"}},
	{Key: "appeal", Debt: "appeal_debt", Script: "doc_appeal_scorecard.py", Label: "doc-appeal"},
	{Key: "seo", Debt: "seo_debt", Cmd: "go run ./cmd/fak score seo --json", Label: "seo"},
	{Key: "demo", Debt: "demo_debt", Script: "demo_quality_scorecard.py", Label: "demo-quality"},
	{Key: "robustness", Debt: "robustness_debt", Script: "demo_robustness_scorecard.py", Label: "demo-robustness"},
	{Key: "hygiene", Debt: "hygiene_debt", Script: "repo_hygiene_scorecard.py", Label: "repo-hygiene"},
	{Key: "parity", Debt: "parity_debt", Script: "industry_scorecard.py", Label: "industry-parity"},
	{Key: "agent", Debt: "friction_debt", Cmd: "go run ./cmd/fak score agent-readiness --json", Label: "agent-readiness"},
	{Key: "product", Debt: "product_debt", Cmd: "go run ./cmd/fak product-scorecard --json", Label: "product"},
	{Key: "persona", Debt: "persona_debt", Script: "persona_readiness_scorecard.py", Label: "persona"},
	{Key: "stability", Debt: "stability_debt", Script: "stability_scorecard.py", Label: "stability"},
	{Key: "slop", Debt: "slop_debt", Script: "code_slop_scorecard.py", Label: "code-slop", Corpus: []string{"**/*.go", "**/CLAIMS.md"}},
	{Key: "steer", Debt: "steerability_debt", Script: "steerability_scorecard.py", Label: "steerability"},
	// The operator-steerability RESIDUAL card (#5022): how many forming units in the
	// pending dev->release delta are banded RESIDUAL — a claim the kernel could not
	// witness, i.e. exactly the pile that owes an operator a look. Debt IS that count,
	// so a clean overlay holds it at/near 0. It reads `fak steer prs --json` (schema
	// fak.steerpr.v1), whose residual_count is the same number the overlay's own
	// worst-first render leads with — one number, one source, no second oracle.
	//
	// It is ORTHOGONAL to the neighbouring cards and must not be read as a second
	// take on either: `steer` (steerability_debt) grades package-graph shape and
	// `heaviness` grades the CLI surface, while this one owns units awaiting
	// attention and nothing else.
	//
	// Corpus is deliberately EMPTY (never carried on a --since fold): the number is a
	// function of git history plus the witness ledger, not of tracked tree files, so
	// a diff-disjoint carry would reproduce a stale residual pile as if it were fresh.
	//
	// UNMEASURED-not-0 fence: when the payload cannot be read at all (an unresolvable
	// base/head ref exits non-zero with no JSON on stdout) the card errors, Debt stays
	// nil, BaselineDoc omits the key, and the superloop walk reports UNMEASURED. A
	// broken read must never fold as a clean zero — see TestOSPResidualUnreadableIsUnmeasuredNeverZero.
	{Key: "osp_residual", Debt: "residual_count", Cmd: "go run ./cmd/fak steer prs --json", Label: "osp-residual"},
	{Key: "conflation", Debt: "conflation_debt", Cmd: "go run ./cmd/fak conflation-scorecard --json", Label: "conflation"},
	{Key: "ui_quality", Debt: "ui_quality_debt", Cmd: "go run ./cmd/fak ui-quality-scorecard --json", Label: "ui-quality", Corpus: []string{"cmd/fak/"}},
	{Key: "disambiguation", Debt: "disambiguation_debt", Script: "concept_disambiguation_scorecard.py", Label: "concept-disambiguation"},
	{Key: "intent_literal", Debt: "intent_literal_debt", Script: "intent_literal_scorecard.py", Label: "intent-literal"},
	{Key: "tokendefaults", Debt: "token_defaults_debt", Cmd: "go run ./cmd/fak token-defaults-scorecard --json", Label: "token-defaults", Origin: &ProducerOrigin{Command: "token-defaults-scorecard", File: "cmd/fak/token_defaults.go"}},
	{Key: "guard_rsi", Debt: "guard_rsi_debt", Cmd: "go run ./cmd/fak guard-rsi-scorecard --json", Label: "guard-rsi", Origin: &ProducerOrigin{Command: "guard-rsi-scorecard", File: "cmd/fak/guardrsi.go"}},
	{Key: "dogfood", Debt: "dogfood_debt", Cmd: "go run ./cmd/fak dogfood-score --json", Label: "dogfood-loop"},
	{Key: "conceptusage", Debt: "conceptusage_debt", Cmd: "go run ./cmd/fak concept-usage-score --json", Label: "concept-usage", Origin: &ProducerOrigin{Command: "concept-usage-score", File: "cmd/fak/conceptusagescore.go"}},
	{Key: "maturity", Debt: "maturity_debt", Cmd: "go run ./cmd/fak maturity --json", Label: "maturity"},
	{Key: "growth", Debt: "growth_debt", Cmd: "go run ./cmd/fak coverage-matrix --json", Label: "growth-debt"},
	{Key: "support_maturity", Debt: "support_maturity_debt", Cmd: "go run ./cmd/fak support-maturity-scorecard --json", Label: "support-maturity"},
	{Key: "milestone", Debt: "milestone_debt", Cmd: "go run ./cmd/fak milestone-scorecard --json", Label: "milestone"},
	{Key: "milestone_climb", Debt: "climb_ratchet_debt", Cmd: "go run ./cmd/fak milestone-scorecard --ratchet --json", Label: "milestone-climb"},
	{Key: "loopindex", Debt: "loopindex_debt", Cmd: "go run ./cmd/fak loop-index-scorecard --json", Label: "loop-index"},
	{Key: "heaviness", Debt: "heaviness_debt", Cmd: "go run ./cmd/fak operator heaviness --json", Label: "operator-heaviness"},
	{Key: "propagation", Debt: "propagation_debt", Cmd: "go run ./cmd/fak propagation-scorecard --json", Label: "propagation"},
	{Key: "antipattern", Debt: "antipattern_debt", Cmd: "go run ./cmd/fak antipattern-scorecard --json", Label: "antipattern"},
	{Key: "brittleness", Debt: "brittleness_debt", Cmd: "go run ./cmd/fak score brittleness --json", Label: "brittleness"},
	{Key: "negframe", Debt: "negframe_debt", Cmd: "go run ./cmd/fak score negframe --json", Label: "negframe", Origin: &ProducerOrigin{Command: "score negframe", File: "cmd/fak/negframescore.go"}},
	{Key: "negation_tax", Debt: "negation_tax_debt", Cmd: "go run ./cmd/fak score negation-tax --json", Label: "negation-tax", Origin: &ProducerOrigin{Command: "score negation-tax", File: "cmd/fak/negationtaxscore.go"}},
	{Key: "negation_operator", Debt: "negation_operator_debt", Cmd: "go run ./cmd/fak score negation_operator --json", Label: "negation-operator"},
	{Key: "claim_repro", Debt: "claim_repro_debt", Script: "claim_repro_scorecard.py", Label: "claim-repro"},
	{Key: "release", Debt: "release_debt", Script: "release_readiness_scorecard.py", Label: "release-readiness"},
	{Key: "sota_coverage", Debt: "sota_debt", Cmd: "go run ./cmd/fak sota-coverage-scorecard --json", Label: "sota-coverage"},
	{Key: "observability", Debt: "observability_debt", Script: "observability_scorecard.py", Label: "observability"},
	{Key: "learning", Debt: "learning_debt", Script: "learning_scorecard.py", Label: "learning"},
	{Key: "rsi_maturity", Debt: "rsi_debt", Script: "rsi_maturity_scorecard.py", Label: "rsi-maturity"},
	{Key: "tooling_quality", Debt: "py_debt", Script: "tooling_quality_scorecard.py", Label: "tooling-quality"},
	{Key: "bench_dx", Debt: "bench_dx_debt", Script: "bench_dx_scorecard.py", Label: "bench-dx"},
	{Key: "cuda_dev", Debt: "process_debt", Script: "cuda_dev_scorecard.py", Label: "cuda-dev"},
	{Key: "persona_fit", Debt: "persona_fit_debt", Script: "persona_fit_scorecard.py", Label: "persona-fit"},
	// The FLOW card (#6198, epic #6194): eight Little's-Law axes — flow efficiency, queue
	// time, unstarted backlog, aging WIP, atomicity, arrival-vs-service, witnessed progress
	// and local (working-tree) WIP — folded by internal/flowmetrics. flow_debt is the COUNT
	// of tripped axes, not a percentage.
	//
	// It is ORTHOGONAL to the neighbouring cards: `osp_residual` counts units awaiting an
	// operator's look and `hygiene` grades repo shape, while this one grades how work MOVES
	// — how long it waits, how much of it was never started, and how much is unlanded on
	// disk. Nothing else here reads issue timestamps against commit timestamps.
	//
	// Corpus is deliberately EMPTY (never carried on a --since fold): the KPIs are a
	// function of git history plus live issue state, not of tracked tree files, so a
	// diff-disjoint carry would replay a stale reading as though it were fresh.
	{Key: "flow", Debt: "flow_debt", Cmd: "go run ./cmd/fak score flow --json", Label: "flow-metrics"},
}

// goBackedKey reports whether a card key is a go run ./cmd/fak card. A simultaneous
// error across these is almost always a working tree that does not compile, not a
// card bug — buildBreakHint operationalizes that distinction.
func goBackedKey(key string) bool {
	for _, c := range Cards {
		if c.Key == key {
			return strings.Contains(c.Cmd, "go run")
		}
	}
	return false
}

// gradeFromValue maps a continuous quality value onto the family's shared
// 0.90/0.80/0.70/0.60 ladder. This is the primary severity lens for new payloads.
func gradeFromValue(value float64) string {
	switch {
	case value >= 0.90:
		return "A"
	case value >= 0.80:
		return "B"
	case value >= 0.70:
		return "C"
	case value >= 0.60:
		return "D"
	default:
		return "F"
	}
}

// gradeFromScore maps a legacy 0-100 score onto the same ladder. It exists only for
// old Python payloads that have not grown corpus.value yet.
func gradeFromScore(score float64) string {
	return gradeFromValue(score / 100)
}

// deriveGrade is the last-resort grade for a scorecard that emits neither a letter
// NOR a score (readme-freshness). Maps debt onto the A-F ladder by magnitude. It is
// SCALE-VARIANT, so it is the lowest-precedence source.
func deriveGrade(debt int) string {
	switch {
	case debt <= 0:
		return "A"
	case debt <= 2:
		return "B"
	case debt <= 5:
		return "C"
	case debt <= 10:
		return "D"
	default:
		return "F"
	}
}

// Metric is one scorecard's extracted control-pane row. Field tags match the
// Python dict keys so the JSON shape stays compatibility-friendly. Debt, Value, and
// Score are pointers so a missing/null value serializes as JSON null (the Python
// contract: an errored card has "debt": null), distinct from a measured zero.
type Metric struct {
	Key             string   `json:"key"`
	Label           string   `json:"label"`
	DebtKey         string   `json:"debt_key"`
	Debt            *int     `json:"debt"`
	DetectorVersion string   `json:"detector_version"`
	Grade           *string  `json:"grade"`
	RatchetGrade    *string  `json:"ratchet_grade,omitempty"`
	RatchetScore    *float64 `json:"ratchet_score,omitempty"`
	Value           *float64 `json:"value"`
	Score           *float64 `json:"score"`
	OK              bool     `json:"ok"`
	Verdict         string   `json:"verdict"`
	Error           string   `json:"error,omitempty"`
	EffGrade        string   `json:"eff_grade,omitempty"`
	GradeWeight     *int     `json:"grade_weight,omitempty"`
	// Carried marks a metric NOT freshly measured this run but reproduced from the
	// pinned baseline because its corpus was untouched by the --since diff (its debt
	// provably could not have moved). Zero-value (a measured metric) omits the field,
	// so a full run's --json bytes are unchanged.
	Carried bool `json:"carried,omitempty"`
	// Origin is present only for cards with an at-producer hook.
	Origin *ProducerOrigin `json:"origin,omitempty"`
}

// displayGrade is the single source of truth for a metric's effective letter grade.
// Precedence: an explicitly emitted ratchet letter/score, when a scorecard has a
// stable gate-specific lens, then the scorecard's own emitted letter, then a
// continuous VALUE-derived letter, then a legacy SCORE-derived letter, then a
// debt-derived fallback.
func displayGrade(m Metric) string {
	if m.RatchetGrade != nil {
		g := strings.ToUpper(*m.RatchetGrade)
		if _, ok := gradeDebt[g]; ok {
			return g
		}
	}
	if m.RatchetScore != nil {
		return gradeFromScore(*m.RatchetScore)
	}
	if m.Grade != nil {
		g := strings.ToUpper(*m.Grade)
		if _, ok := gradeDebt[g]; ok {
			return g
		}
	}
	if m.Value != nil {
		return gradeFromValue(*m.Value)
	}
	if m.Score != nil {
		return gradeFromScore(*m.Score)
	}
	if m.Debt != nil {
		return deriveGrade(*m.Debt)
	}
	return "F"
}

// MetricFromPayload extracts one Metric from a scorecard's parsed JSON payload (or
// records an error row when the payload is missing/non-dict). It mirrors the Python
// metric_from_payload byte-for-byte: the same debt search, grade/score derivation,
// and error string shapes.
func MetricFromPayload(card Card, payload map[string]any, errMsg string) Metric {
	if errMsg != "" || payload == nil {
		e := errMsg
		if e == "" {
			e = "no payload"
		}
		return Metric{
			Key: card.Key, Label: card.Label, DebtKey: card.Debt, Origin: card.Origin,
			Debt: nil, Grade: nil, Score: nil, OK: false, Verdict: "ERROR", Error: e,
		}
	}
	debt := findInt(payload, card.Debt)
	valuePtr := findValue(payload)
	ratchetScorePtr := findScore(payload, "ratchet_score")
	var scorePtr *float64
	if sk := scoreKeys[card.Key]; sk != "" {
		scorePtr = findScore(payload, sk)
		if valuePtr == nil && scorePtr != nil {
			v := *scorePtr / 100
			valuePtr = &v
		}
	}
	m := Metric{
		Key: card.Key, Label: card.Label, DebtKey: card.Debt, Origin: card.Origin,
		Debt:            debt,
		DetectorVersion: asString(payload["schema"]),
		Grade:           findGrade(payload),
		RatchetGrade:    findString(payload, "ratchet_grade"),
		RatchetScore:    ratchetScorePtr,
		Value:           valuePtr,
		Score:           scorePtr,
		OK:              scorecard.True(payload["ok"]),
		Verdict:         asString(payload["verdict"]),
	}
	if debt == nil {
		m.Error = "missing " + card.Debt + " in payload"
	}
	return m
}

// buildBreakHint distinguishes a working-tree BUILD BREAK from a real card bug.
// When any errored card is go-backed (shells go run ./cmd/fak), the usual cause is
// a working tree that does not compile, not a card bug. Returns "" when no errored
// card is go-backed. Byte-identical guidance to the Python build_break_hint.
func buildBreakHint(errored []Metric) string {
	var goErrs []string
	for _, m := range errored {
		if goBackedKey(m.Key) {
			goErrs = append(goErrs, m.Label)
		}
	}
	if len(goErrs) == 0 {
		return ""
	}
	sort.Strings(goErrs)
	return fmt.Sprintf(" — note: %d Go-backed card(s) errored (%s); "+
		"these shell `go run ./cmd/fak …`, so the usual cause is a working tree that "+
		"does NOT compile, not a card bug. Triage with `go build ./...`: if it FAILS, "+
		"commit or stash your WIP (or measure a pristine HEAD checkout that keeps .git, "+
		"e.g. `git worktree add --detach <dir> HEAD`) and re-run; if `go build ./...` "+
		"PASSES yet a card still errors, it is a real card bug — debug that card's "+
		"--json. (clean-read recipe: .claude/skills/scorecard/SKILL.md)",
		len(goErrs), strings.Join(goErrs, ", "))
}

// EarlyWarning is one per-metric rise vs its pinned baseline value, surfaced advisory
// even when the portfolio total holds. Field tags match the Python dict.
type EarlyWarning struct {
	Key   string `json:"key"`
	Label string `json:"label"`
	Delta int    `json:"delta"`
	From  int    `json:"from"`
	To    int    `json:"to"`
}

// GradeRegression is one metric whose severity weight rose vs the pinned baseline
// even if its raw debt stayed flat. This is the native parity for the Python
// grade_regressed list that powers the hard A->B scorecard ratchet.
type GradeRegression struct {
	Key        string `json:"key"`
	Label      string `json:"label"`
	FromWeight int    `json:"from_weight"`
	ToWeight   int    `json:"to_weight"`
	ToGrade    string `json:"to_grade"`
}

// DetectorVersionChange identifies a raw-debt series that cannot be compared to
// its pin. A detector/schema revision can legitimately reveal a different amount
// of debt, so it must be reviewed and re-pinned rather than called a regression or
// improvement.
type DetectorVersionChange struct {
	Key      string `json:"key"`
	Label    string `json:"label"`
	Baseline string `json:"baseline"`
	Current  string `json:"current"`
}

// Trend is the per-metric + portfolio delta vs a pinned baseline. Field tags and
// shape match the Python compute_trend return dict.
type Trend struct {
	Direction          string                  `json:"direction"`
	Summary            string                  `json:"summary"`
	ComparisonBasis    string                  `json:"comparison_basis"`
	Comparable         bool                    `json:"comparable"`
	TotalDelta         *int                    `json:"total_delta"`
	TotalDebtSemantics string                  `json:"total_debt_semantics"`
	GradeDelta         *int                    `json:"grade_delta"`
	BaselineCommit     string                  `json:"baseline_commit"`
	BaselineTotal      *int                    `json:"baseline_total"`
	BaselineGrade      *int                    `json:"baseline_grade"`
	GradeDebt          int                     `json:"grade_debt"`
	Deltas             map[string]int          `json:"deltas"`
	Worsened           []string                `json:"worsened"`
	Improved           []string                `json:"improved"`
	EarlyWarning       []EarlyWarning          `json:"early_warning"`
	GradeRegressed     []GradeRegression       `json:"grade_regressed"`
	VersionChanged     []DetectorVersionChange `json:"version_changed"`
	OutOfControl       *PortfolioOutOfControl  `json:"out_of_control,omitempty"`
}

// Baseline is the pinned per-metric baseline body (the tracked baseline file shape).
type Baseline struct {
	Schema             string            `json:"schema"`
	Commit             string            `json:"commit"`
	TotalDebt          int               `json:"total_debt"`
	TotalDebtSemantics string            `json:"total_debt_semantics,omitempty"`
	GradeDebt          int               `json:"grade_debt"`
	Metrics            map[string]int    `json:"metrics"`
	GradeWeights       map[string]int    `json:"grade_weights,omitempty"`
	DetectorVersions   map[string]string `json:"detector_versions,omitempty"`
	Doc                string            `json:"_doc,omitempty"`
}

// Payload is the folded control-pane payload. Field order/tags match the Python
// fold() return dict so the --json bytes are contract-compatible.
type Payload struct {
	Schema             string                 `json:"schema"`
	OK                 bool                   `json:"ok"`
	Verdict            string                 `json:"verdict"`
	Finding            string                 `json:"finding"`
	Reason             string                 `json:"reason"`
	NextAction         string                 `json:"next_action"`
	Workspace          string                 `json:"workspace"`
	Commit             string                 `json:"commit"`
	TotalDebt          int                    `json:"total_debt"`
	TotalDebtSemantics string                 `json:"total_debt_semantics"`
	GradeDebt          int                    `json:"grade_debt"`
	GradeBreakdown     string                 `json:"grade_breakdown"`
	Measured           int                    `json:"measured"`
	Errored            int                    `json:"errored"`
	EarlyWarning       []EarlyWarning         `json:"early_warning"`
	OutOfControl       *PortfolioOutOfControl `json:"out_of_control,omitempty"`
	Metrics            []Metric               `json:"metrics"`
	Trend              Trend                  `json:"trend"`
	// GateExit/GateMessage are populated only under --check (the ratchet contract),
	// matching the Python gated payload.
	GateExit    *int   `json:"gate_exit,omitempty"`
	GateMessage string `json:"gate_message,omitempty"`
	// Incremental is populated only on a --since (shift-left) fold: it records that
	// some cards were CARRIED from the pinned baseline rather than re-measured. Its
	// presence marks the payload as an incremental read, NOT a full measurement — so
	// it must not be pinned or posted as a new floor. A full run leaves it nil (the
	// field omits), keeping the full-run --json bytes unchanged.
	Incremental *IncrementalInfo `json:"incremental,omitempty"`
}

// Fold folds per-scorecard metrics into one control-pane payload + trend. It is a
// faithful port of the Python fold(): the same verdict ladder (error > regressed >
// debt > clear), the same early-warning note, the same reason/next_action strings.
func Fold(metrics []Metric, baseline *Baseline, workspace, commit string) Payload {
	var measured []*Metric
	var errors []Metric
	for i := range metrics {
		if metrics[i].Debt != nil {
			measured = append(measured, &metrics[i])
		} else {
			errors = append(errors, metrics[i])
		}
	}
	totalDebt := 0
	for _, m := range measured {
		totalDebt += *m.Debt
	}
	gradeDebtTotal := 0
	for _, m := range measured {
		eff := displayGrade(*m)
		m.EffGrade = eff
		w := gradeDebt[eff]
		wcopy := w
		m.GradeWeight = &wcopy
		gradeDebtTotal += w
	}

	trend := ComputeTrend(metrics, baseline, totalDebt, gradeDebtTotal)

	byDebt := append([]*Metric(nil), measured...)
	sort.SliceStable(byDebt, func(i, j int) bool { return *byDebt[i].Debt > *byDebt[j].Debt })
	breakdown := "none"
	if len(byDebt) > 0 {
		parts := make([]string, len(byDebt))
		for i, m := range byDebt {
			parts[i] = fmt.Sprintf("%s %d", m.Label, *m.Debt)
		}
		breakdown = strings.Join(parts, ", ")
	}
	byGrade := append([]*Metric(nil), measured...)
	sort.SliceStable(byGrade, func(i, j int) bool { return *byGrade[i].GradeWeight > *byGrade[j].GradeWeight })
	var gbParts []string
	for _, m := range byGrade {
		if *m.GradeWeight > 0 {
			gbParts = append(gbParts, fmt.Sprintf("%s %s(%d)", m.Label, m.EffGrade, *m.GradeWeight))
		}
	}
	gradeBreakdown := "all A"
	if len(gbParts) > 0 {
		gradeBreakdown = strings.Join(gbParts, ", ")
	}

	regressed := trend.Direction == "regressed"
	incomparable := trend.Direction == "incomparable"
	earlyWarning := trend.EarlyWarning
	ewNote := ""
	if len(earlyWarning) > 0 && !regressed {
		var ws []string
		for _, e := range earlyWarning {
			ws = append(ws, fmt.Sprintf("%s %d->%d (+%d)", e.Label, e.From, e.To, e.Delta))
		}
		ewNote = "; EARLY-WARNING (advisory): " + strings.Join(ws, ", ") +
			" rose vs baseline under a green portfolio — a hidden per-metric " +
			"regression; review before --pin re-floors it"
	}

	var ok bool
	var verdict, finding, reason, nextAction string
	switch {
	case len(errors) > 0:
		ok, verdict, finding = false, "ACTION", "scorecard_unmeasured"
		var labels []string
		for _, m := range errors {
			labels = append(labels, m.Label)
		}
		reason = fmt.Sprintf("%d scorecard(s) failed to report a debt integer "+
			"(%s); observational heterogeneous total %d across %d measured",
			len(errors), strings.Join(labels, ", "), totalDebt, len(measured))
		nextAction = "repair the failing scorecard(s) so the fold is complete; " +
			"re-run python tools/scorecard_control_pane.py" + buildBreakHint(errors)
	case incomparable:
		ok, verdict, finding = false, "ACTION", "scorecard_incomparable"
		var changed []string
		for _, v := range trend.VersionChanged {
			changed = append(changed, fmt.Sprintf("%s %s->%s", v.Label, v.Baseline, v.Current))
		}
		reason = "detector version changed for " + strings.Join(changed, ", ") +
			"; raw debt deltas are incomparable"
		nextAction = "review the detector change, then intentionally re-pin: " +
			"fak scorecard control-pane --pin"
	case regressed:
		ok, verdict, finding = false, "ACTION", "scorecard_regressed"
		worsened := strings.Join(trend.Worsened, ", ")
		if worsened == "" {
			worsened = "see deltas"
		}
		reason = fmt.Sprintf("same-version per-metric debt rose vs baseline @%s; "+
			"worsened: %s; observational heterogeneous total %d (%s)",
			trend.BaselineCommit, worsened, totalDebt, breakdown)
		nextAction = "retire the regressed metric(s) worst-first with the owning " +
			"scorecard's skill, then re-pin: " +
			"fak scorecard control-pane --pin"
	case totalDebt > 0:
		ok, verdict, finding = false, "ACTION", "scorecard_debt"
		reason = fmt.Sprintf("observational heterogeneous total debt %d across %d scorecards "+
			"(%s); per-metric trend %s",
			totalDebt, len(measured), breakdown, trend.Summary)
		nextAction = fmt.Sprintf("retire debt worst-first (heaviest: %s %d) with that "+
			"scorecard's skill; re-run to prove the portfolio drop", byDebt[0].Label, *byDebt[0].Debt)
	default:
		ok, verdict, finding = true, "OK", "all_clear"
		reason = fmt.Sprintf("zero raw debt in all %d measured scorecards; per-metric trend %s",
			len(measured), trend.Summary)
		nextAction = "hold the line; re-pin the baseline to lock the clean state"
	}

	reason += ewNote
	if ewNote != "" {
		var labels []string
		for _, e := range earlyWarning {
			labels = append(labels, e.Label)
		}
		nextAction = "review the per-metric early-warning (" + strings.Join(labels, ", ") +
			") with that scorecard's skill BEFORE `--pin`, so a hidden regression " +
			"isn't blessed as the new floor; then: " + nextAction
	}

	return Payload{
		Schema: Schema, OK: ok, Verdict: verdict, Finding: finding,
		Reason: reason, NextAction: nextAction, Workspace: workspace, Commit: commit,
		TotalDebt: totalDebt, TotalDebtSemantics: TotalDebtSemantics,
		GradeDebt: gradeDebtTotal, GradeBreakdown: gradeBreakdown,
		Measured: len(measured), Errored: len(errors), EarlyWarning: earlyWarning,
		OutOfControl: trend.OutOfControl,
		Metrics:      metrics,
		Trend:        trend,
	}
}

// ComputeTrend compares raw debt only within the same metric and detector version.
// totalDebt is retained as an observational compatibility summary; it does not
// decide Direction because unlike metric units cannot cancel each other honestly.
func ComputeTrend(metrics []Metric, baseline *Baseline, totalDebt, gradeDebtTotal int) Trend {
	baseMetrics := map[string]int{}
	baseVersions := map[string]string{}
	baseCommit := ""
	var baseTotal, baseGrade *int
	if baseline != nil {
		baseMetrics = baseline.Metrics
		baseVersions = baseline.DetectorVersions
		baseCommit = baseline.Commit
		bt := baseline.TotalDebt
		baseTotal = &bt
		bg := baseline.GradeDebt
		baseGrade = &bg
	}

	if len(baseMetrics) == 0 || baseTotal == nil {
		zero := 0
		return Trend{
			Direction: "unpinned", Summary: "unpinned (no baseline; run --pin)",
			ComparisonBasis: MetricComparisonBasis, Comparable: false,
			TotalDelta: &zero, TotalDebtSemantics: TotalDebtSemantics, GradeDelta: &zero,
			BaselineCommit: baseCommit,
			BaselineTotal:  baseTotal, BaselineGrade: baseGrade, GradeDebt: gradeDebtTotal,
			Deltas: map[string]int{}, Worsened: []string{}, Improved: []string{},
			EarlyWarning: []EarlyWarning{}, GradeRegressed: []GradeRegression{},
			VersionChanged: []DetectorVersionChange{},
		}
	}

	deltas := map[string]int{}
	worsened := []string{}
	improved := []string{}
	earlyWarning := []EarlyWarning{}
	gradeRegressed := []GradeRegression{}
	versionChanged := []DetectorVersionChange{}
	for _, m := range metrics {
		if m.Debt == nil {
			continue
		}
		prior, ok := baseMetrics[m.Key]
		if !ok {
			continue
		}
		priorVersion := baseVersions[m.Key]
		// A legacy pin has no detector_versions map. Keep it readable for one
		// intentional migration pin; once both sides name versions, equality is
		// mandatory and a change is neither improvement nor regression.
		if priorVersion != "" && m.DetectorVersion != "" && priorVersion != m.DetectorVersion {
			versionChanged = append(versionChanged, DetectorVersionChange{
				Key: m.Key, Label: m.Label, Baseline: priorVersion, Current: m.DetectorVersion,
			})
			continue
		}
		delta := *m.Debt - prior
		deltas[m.Key] = delta
		if delta > 0 {
			worsened = append(worsened, m.Label)
			earlyWarning = append(earlyWarning, EarlyWarning{
				Key: m.Key, Label: m.Label, Delta: delta, From: prior, To: *m.Debt,
			})
		} else if delta < 0 {
			improved = append(improved, m.Label)
		}
		if priorW, ok := baseGradeWeight(baseline, m.Key); ok && m.GradeWeight != nil && *m.GradeWeight > priorW {
			gradeRegressed = append(gradeRegressed, GradeRegression{
				Key: m.Key, Label: m.Label, FromWeight: priorW, ToWeight: *m.GradeWeight, ToGrade: m.EffGrade,
			})
		}
	}

	totalDeltaValue := totalDebt - *baseTotal
	gradeDeltaValue := 0
	if baseGrade != nil {
		gradeDeltaValue = gradeDebtTotal - *baseGrade
	}
	var totalDelta, gradeDelta *int
	if len(versionChanged) == 0 {
		totalDelta = &totalDeltaValue
		gradeDelta = &gradeDeltaValue
	}
	direction := "flat"
	if len(versionChanged) > 0 {
		direction = "incomparable"
	} else if len(worsened) > 0 {
		direction = "regressed"
	} else if len(improved) > 0 {
		direction = "improved"
	}
	bc := baseCommit
	if bc == "" {
		bc = "baseline"
	}
	summary := "incomparable: detector version changed; review and re-pin"
	if direction != "incomparable" {
		summary = fmt.Sprintf("%s by %s vs @%s; observational total %d->%d (%+d)",
			direction, MetricComparisonBasis, bc, *baseTotal, totalDebt, totalDeltaValue)
	}
	if baseGrade != nil && gradeDelta != nil && *gradeDelta != 0 {
		summary += fmt.Sprintf("; grade-debt %d->%d (%+d)", *baseGrade, gradeDebtTotal, *gradeDelta)
	}
	ooc := AssessOutOfControl(metrics, baseline, totalDebt, gradeDebtTotal, DefaultOutOfControlBounds())
	var oocPtr *PortfolioOutOfControl
	if ooc.Status != StatusUnpinned {
		oocPtr = &ooc
	}
	return Trend{
		Direction: direction, Summary: summary, ComparisonBasis: MetricComparisonBasis,
		Comparable: len(versionChanged) == 0, TotalDelta: totalDelta,
		TotalDebtSemantics: TotalDebtSemantics, GradeDelta: gradeDelta,
		BaselineCommit: baseCommit, BaselineTotal: baseTotal, BaselineGrade: baseGrade,
		GradeDebt: gradeDebtTotal, Deltas: deltas, Worsened: worsened, Improved: improved,
		EarlyWarning: earlyWarning, GradeRegressed: gradeRegressed, VersionChanged: versionChanged,
		OutOfControl: oocPtr,
	}
}

func baseGradeWeight(baseline *Baseline, key string) (int, bool) {
	if baseline == nil || baseline.GradeWeights == nil {
		return 0, false
	}
	v, ok := baseline.GradeWeights[key]
	return v, ok
}

// BaselineDoc builds the baseline file body to pin from a folded payload. Ported
// from the Python baseline_doc.
func BaselineDoc(p Payload) Baseline {
	metrics := map[string]int{}
	gradeWeights := map[string]int{}
	detectorVersions := map[string]string{}
	for _, m := range p.Metrics {
		if m.Debt != nil {
			metrics[m.Key] = *m.Debt
			if m.GradeWeight != nil {
				gradeWeights[m.Key] = *m.GradeWeight
			}
			if m.DetectorVersion != "" {
				detectorVersions[m.Key] = m.DetectorVersion
			}
		}
	}
	return Baseline{
		Schema: BaselineSchema, Commit: p.Commit, TotalDebt: p.TotalDebt,
		TotalDebtSemantics: TotalDebtSemantics, GradeDebt: p.GradeDebt,
		Metrics: metrics, GradeWeights: gradeWeights, DetectorVersions: detectorVersions,
		Doc: "Pinned per-metric scorecard-debt baseline for the unified " +
			"control pane. Each raw debt is unbounded by the pane and gates only " +
			"against the same metric and detector version. total_debt is an observational " +
			"heterogeneous sum, not a gate or universal quality unit. grade_debt " +
			"is the scale-invariant severity companion; grade_weights pins " +
			"each metric's letter-grade severity so an A->B slip reds the gate " +
			"even at flat raw debt. Review detector-version changes before re-pinning: " +
			"fak scorecard control-pane --pin",
	}
}

// CheckGate is the CI ratchet decision over a folded payload (pure: exit code +
// message). The raw-debt gate is per metric and detector version: any comparable
// metric rise is red even when another metric falls by more. Version changes are
// incomparable until reviewed/re-pinned. The heterogeneous total is observational.
//
//	0  flat / improved   — every comparable per-metric ratchet held
//	1  regressed         — at least one comparable metric rose (or is unmeasured)
//	2  unpinned/incomparable — establish or intentionally refresh the pin
func CheckGate(p Payload) (int, string) {
	if p.Errored > 0 {
		var errored []Metric
		for _, m := range p.Metrics {
			if m.Debt == nil {
				errored = append(errored, m)
			}
		}
		return 1, fmt.Sprintf("RATCHET FAIL: %d scorecard(s) unmeasured — %s",
			p.Errored, p.Reason) + buildBreakHint(errored)
	}
	switch p.Trend.Direction {
	case "incomparable":
		var changed []string
		for _, v := range p.Trend.VersionChanged {
			changed = append(changed, fmt.Sprintf("%s %s->%s", v.Label, v.Baseline, v.Current))
		}
		return 2, "RATCHET INCOMPARABLE: detector version changed for " +
			strings.Join(changed, ", ") + "; review the detector and intentionally re-pin"
	case "unpinned":
		return 2, "RATCHET UNPINNED: no baseline to ratchet against; run " +
			"`python tools/scorecard_control_pane.py --pin` to set one"
	case "regressed":
		worsened := strings.Join(p.Trend.Worsened, ", ")
		if worsened == "" {
			worsened = "see deltas"
		}
		extra := ""
		if p.OutOfControl != nil && p.OutOfControl.IsOutOfControl {
			extra = fmt.Sprintf(" [OUT OF CONTROL: %s]", strings.Join(p.OutOfControl.Reasons, "; "))
		}
		return 1, fmt.Sprintf("RATCHET FAIL: same-version per-metric debt regressed; "+
			"worsened: %s; %s (total_debt=%d)%s",
			worsened, TotalDebtSemantics, p.TotalDebt, extra)
	}
	if p.OutOfControl != nil && p.OutOfControl.IsOutOfControl {
		return 1, fmt.Sprintf("RATCHET FAIL: OUT OF CONTROL debt dynamics (%s): %s", p.OutOfControl.Status, p.Reason)
	}
	gradeDelta := 0
	if p.Trend.GradeDelta != nil {
		gradeDelta = *p.Trend.GradeDelta
	}
	if (len(p.Trend.GradeRegressed) > 0 || gradeDelta > 0) && gradeRatchetHard() {
		who := p.GradeBreakdown
		if len(p.Trend.GradeRegressed) > 0 {
			parts := make([]string, 0, len(p.Trend.GradeRegressed))
			for _, g := range p.Trend.GradeRegressed {
				parts = append(parts, fmt.Sprintf("%s %s->%s", g.Label, weightLetter(g.FromWeight), g.ToGrade))
			}
			who = strings.Join(parts, ", ")
		}
		return 1, fmt.Sprintf("GRADE-RATCHET FAIL: grade-debt rose %+d to %d vs baseline @%s "+
			"-- a scorecard slipped a letter the raw-unit total held flat: %s. Retire it with "+
			"the owning scorecard's skill, then re-pin (`--pin`); or set %s=0 to demote this "+
			"gate to advisory for a deliberate one-off pin.",
			gradeDelta, p.GradeDebt, p.Trend.BaselineCommit, who, GradeRatchetEnv)
	}
	msg := fmt.Sprintf("RATCHET OK: every comparable per-metric debt held "+
		"(%s); total_debt=%d is %s", MetricComparisonBasis, p.TotalDebt, TotalDebtSemantics)
	if len(p.Trend.EarlyWarning) > 0 {
		var ws []string
		for _, e := range p.Trend.EarlyWarning {
			ws = append(ws, fmt.Sprintf("%s +%d", e.Label, e.Delta))
		}
		msg += "; EARLY-WARNING (advisory, gate still green): " + strings.Join(ws, ", ") +
			" rose vs baseline — a hidden per-metric regression; review before --pin"
	}
	if gradeDelta > 0 && !gradeRatchetHard() {
		msg += fmt.Sprintf("; GRADE-DEBT WARN (advisory -- ratchet demoted via %s=0): severity rose "+
			"%+d to %d vs baseline (%s) -- review before --pin",
			GradeRatchetEnv, gradeDelta, p.GradeDebt, p.GradeBreakdown)
	}
	return 0, msg
}

func gradeRatchetHard() bool {
	raw, ok := os.LookupEnv(GradeRatchetEnv)
	if !ok {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

func weightLetter(weight int) string {
	for letter, w := range gradeDebt {
		if w == weight {
			return letter
		}
	}
	return "?"
}

// Render is the human control-pane snapshot, ported from the Python render().
func Render(p Payload) string {
	var b strings.Builder
	fmt.Fprintf(&b, "scorecard control pane — %s (%s)\n", p.Verdict, p.Finding)
	fmt.Fprintf(&b, "  observational total debt: %d (heterogeneous raw-unit sum; not a gate)  "+
		"grade-debt: %d (severity, scale-invariant)  "+
		"(%d measured, %d errored)  @%s\n", p.TotalDebt, p.GradeDebt, p.Measured, p.Errored, p.Commit)
	fmt.Fprintf(&b, "  raw-debt gate: %s\n", MetricComparisonBasis)
	fmt.Fprintf(&b, "  grade severity: %s\n", p.GradeBreakdown)
	fmt.Fprintf(&b, "  trend: %s\n\n", p.Trend.Summary)
	for _, m := range p.Metrics {
		debt := ""
		if m.Debt != nil {
			debt = fmt.Sprintf("%d", *m.Debt)
		} else {
			debt = fmt.Sprintf("ERR (%s)", m.Error)
		}
		grade := ""
		if m.Grade != nil && *m.Grade != "" {
			grade = " [" + *m.Grade + "]"
		}
		fmt.Fprintf(&b, "  %-16s %-16s %s%s\n", m.Label, m.DebtKey, debt, grade)
	}
	if len(p.EarlyWarning) > 0 {
		b.WriteString("\n")
		for _, e := range p.EarlyWarning {
			fmt.Fprintf(&b, "  WARN early-warning: %s rose %d->%d (+%d) vs baseline — hidden under a green portfolio\n",
				e.Label, e.From, e.To, e.Delta)
		}
	}
	if p.OutOfControl != nil && p.OutOfControl.IsOutOfControl {
		b.WriteString("\n  OUT-OF-CONTROL DEBT ALERT: portfolio debt movement exceeds safety bounds\n")
	}
	if len(p.Trend.GradeRegressed) > 0 {
		b.WriteString("\n")
		for _, g := range p.Trend.GradeRegressed {
			fmt.Fprintf(&b, "  GRADE REGRESSION: %s slipped to %s vs pinned grade — reds the grade ratchet\n",
				g.Label, g.ToGrade)
		}
	}
	if len(p.Trend.VersionChanged) > 0 {
		b.WriteString("\n")
		for _, v := range p.Trend.VersionChanged {
			fmt.Fprintf(&b, "  INCOMPARABLE: %s detector changed %s->%s — review, then re-pin\n",
				v.Label, v.Baseline, v.Current)
		}
	}
	if p.OutOfControl != nil && p.OutOfControl.IsOutOfControl {
		b.WriteString("\n")
		fmt.Fprintf(&b, "  OUT-OF-CONTROL DEBT ALERT (status %s):\n", p.OutOfControl.Status)
		for _, r := range p.OutOfControl.Reasons {
			fmt.Fprintf(&b, "    ! %s\n", r)
		}
		for _, om := range p.OutOfControl.Metrics {
			fmt.Fprintf(&b, "    * %s: %s (%d->%d, %+d) — %s\n",
				om.Label, om.Severity, om.From, om.To, om.Delta, strings.Join(om.Reasons, "; "))
		}
	} else if p.OutOfControl != nil && p.OutOfControl.Status == StatusElevated {
		b.WriteString("\n")
		fmt.Fprintf(&b, "  ELEVATED DEBT ALERT (status %s — %d metric(s) exceeding rate/velocity bounds):\n",
			p.OutOfControl.Status, len(p.OutOfControl.Metrics))
		for _, om := range p.OutOfControl.Metrics {
			fmt.Fprintf(&b, "    * %s: %s (%d->%d, %+d) — %s\n",
				om.Label, om.Severity, om.From, om.To, om.Delta, strings.Join(om.Reasons, "; "))
		}
	}
	fmt.Fprintf(&b, "\n  → %s\n", p.NextAction)
	return b.String()
}
