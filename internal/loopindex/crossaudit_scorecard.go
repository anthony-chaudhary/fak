package loopindex

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"io"
	"sort"
	"strings"
)

// crossAuditSchema is the stable control-pane schema identifier the cross-audit
// observability scorecard emits. The JSON shape under this schema is pinned by a
// golden test — a consumer (cmd/TUI/Slack) may bind to these keys.
const crossAuditSchema = "fak.crossaudit.scorecard.v1"

// Cross-audit OUTCOME vocabulary. This fold reports on the whole background
// audit LOOP — attempts AND durable receipts — so it is broader than the
// modelroute receipt verdict set: a same-family attempt that was REFUSED before
// inference produces no receipt, and an UNAVAILABLE auditor produced no verdict.
// Both are audit facts the operator must see, so both are first-class outcomes
// here rather than hidden as "missing".
const (
	OutcomePass         = "PASS"         // audit completed, no finding
	OutcomeRefute       = "REFUTE"       // audit completed, finding raised
	OutcomeInconclusive = "INCONCLUSIVE" // audit completed, no decision
	OutcomeUnavailable  = "UNAVAILABLE"  // auditor/provider was unavailable — a dark-loop signal
	OutcomeRefused      = "REFUSED"      // independence not admitted (e.g. same family) — no audit ran
	OutcomeUnknown      = "UNKNOWN"      // out-of-vocabulary outcome — never counted as audited
)

// Independence states, kept transparent so a refused or unproven audit is never
// laundered into an admitted denominator (the named confusion risk).
const (
	IndependenceAdmitted = "admitted"
	IndependenceRefused  = "refused"
	IndependenceUnknown  = "unknown"
)

// AuditRecord is ONE recorded audit outcome — the pure projection the impure
// shell fills from a modelroute receipt row (#3850) or a background-loop event
// (#3856). Keeping it a plain struct (no import of modelroute) holds this fold
// at foundation tier and lets a test supply fixtures without a live loop.
type AuditRecord struct {
	IssueNumber        int    `json:"issue_number"`
	Class              string `json:"class"` // issue class, for per-class yield
	Outcome            string `json:"outcome"`
	Severity           string `json:"severity,omitempty"` // only meaningful for REFUTE
	Independence       string `json:"independence"`
	AuthorModel        string `json:"author_model,omitempty"`
	AuditorModel       string `json:"auditor_model,omitempty"`
	CalibrationVersion string `json:"calibration_version,omitempty"`
	RecordedAtUnixNano int64  `json:"recorded_at_unix_nano,omitempty"` // for age/freshness
	DurationNanos      int64  `json:"duration_nanos,omitempty"`        // latency
	// Cost provenance: Measured says a real usage reading exists; CostBasis is the
	// modelroute basis vocabulary (unreported | provider-reported | host-calculated).
	// A cost total is only summed when it carries a non-"unreported" basis, so a
	// modeled or absent cost never masquerades as a measured one.
	CostMeasured  bool   `json:"cost_measured,omitempty"`
	InputTokens   int64  `json:"input_tokens,omitempty"`
	OutputTokens  int64  `json:"output_tokens,omitempty"`
	TotalTokens   int64  `json:"total_tokens,omitempty"`
	CostMicrosUSD int64  `json:"cost_micros_usd,omitempty"`
	CostBasis     string `json:"cost_basis,omitempty"`
}

// ProviderHealth is one auditor provider's liveness as the loop last observed it.
type ProviderHealth struct {
	Name      string `json:"name"`
	Available bool   `json:"available"`
}

// LoopHealth is the background-loop projection: liveness, backlog, retry/dead-
// letter accounting, and provider health. It is filled by the loop owner (#3856);
// until that lands the shell can pass a zero value and the fold still scores the
// receipt side honestly (an absent loop reads as not-running — a dark loop).
type LoopHealth struct {
	Present          bool             `json:"present"` // false => loop state unavailable (treated as dark)
	Running          bool             `json:"running"`
	LastTickUnixNano int64            `json:"last_tick_unix_nano,omitempty"`
	PendingIssues    int              `json:"pending_issues"`
	InflightIssues   int              `json:"inflight_issues"`
	Retries          int              `json:"retries"`
	DeadLetters      int              `json:"dead_letters"`
	Providers        []ProviderHealth `json:"providers,omitempty"`
}

// CrossAuditInput is the whole fold input. NowUnixNano and StaleAfterNanos are
// passed in (never read from a clock) so the fold stays deterministic — two
// callers with the same input score identically.
type CrossAuditInput struct {
	NowUnixNano     int64         `json:"now_unix_nano,omitempty"`
	StaleAfterNanos int64         `json:"stale_after_nanos,omitempty"`
	EligibleIssues  int           `json:"eligible_issues"`
	Records         []AuditRecord `json:"records"`
	Loop            LoopHealth    `json:"loop"`
}

// NameCount is a deterministic (count desc, then name asc) tally row used for
// model mix, calibration versions, verdicts, and severities.
type NameCount struct {
	Name  string `json:"name"`
	Count int    `json:"count"`
}

// CoverageBlock keeps every denominator explicit: an UNAVAILABLE or REFUSED
// attempt is NOT counted as audited, and issues with no record at all surface as
// Missing rather than silently shrinking the denominator.
type CoverageBlock struct {
	Eligible        int     `json:"eligible"`
	Records         int     `json:"records"`          // total recorded outcomes
	Attempted       int     `json:"attempted"`        // distinct issues with any record
	Audited         int     `json:"audited"`          // distinct issues with a COMPLETED audit
	Pending         int     `json:"pending"`          // eligible - audited
	Missing         int     `json:"missing"`          // eligible with no record at all
	UnavailableOnly int     `json:"unavailable_only"` // attempted but only ever UNAVAILABLE/REFUSED
	AuditedRate     float64 `json:"audited_rate"`
	AttemptedRate   float64 `json:"attempted_rate"`
	RateBasis       string  `json:"rate_basis"` // names the denominator
}

// IndependenceBlock reports admitted/refused/unknown side by side. AdmittedRate
// is over ALL records, so refused/unknown can never hide inside a healthy number.
type IndependenceBlock struct {
	Total        int     `json:"total"`
	Admitted     int     `json:"admitted"`
	Refused      int     `json:"refused"`
	Unknown      int     `json:"unknown"`
	AdmittedRate float64 `json:"admitted_rate"`
	RateBasis    string  `json:"rate_basis"`
}

// ClassYield is per-issue-class finding yield: refutes over completed audits in
// that class. Yield is a discovery rate, never a correctness claim.
type ClassYield struct {
	Class     string  `json:"class"`
	Completed int     `json:"completed"`
	Refutes   int     `json:"refutes"`
	YieldRate float64 `json:"yield_rate"`
}

// QualityBlock. PassRate is pass over COMPLETED audits and is explicitly labelled
// not-a-correctness-rate: a PASS verdict means an independent auditor found no
// refutation on the supplied evidence, not that the work is correct.
type QualityBlock struct {
	Verdicts      []NameCount  `json:"verdicts"`
	Severities    []NameCount  `json:"severities"` // over REFUTE findings
	Completed     int          `json:"completed"`  // sample count for PassRate
	Passes        int          `json:"passes"`
	Findings      int          `json:"findings"` // REFUTE count
	PassRate      float64      `json:"pass_rate"`
	PassRateBasis string       `json:"pass_rate_basis"`
	PerClassYield []ClassYield `json:"per_class_yield"`
}

// ModelMixBlock names who authored and who audited, and which calibration
// versions ran — the independence and drift audit trail.
type ModelMixBlock struct {
	Authors      []NameCount `json:"authors"`
	Auditors     []NameCount `json:"auditors"`
	Calibrations []NameCount `json:"calibrations"`
}

// EconomicsBlock. Every total carries its SampleCount and provenance so a partial
// or modeled measurement is never read as a complete one (the witness rule: all
// cost and quality fields display provenance and sample count).
type EconomicsBlock struct {
	Records            int         `json:"records"`
	TokenSampleCount   int         `json:"token_sample_count"` // records with a measured usage reading
	InputTokens        int64       `json:"input_tokens"`
	OutputTokens       int64       `json:"output_tokens"`
	TotalTokens        int64       `json:"total_tokens"`
	CostSampleCount    int         `json:"cost_sample_count"` // records with a non-"unreported" cost basis
	CostMicrosUSD      int64       `json:"cost_micros_usd"`
	CostBasisCounts    []NameCount `json:"cost_basis_counts"`
	LatencySampleCount int         `json:"latency_sample_count"`
	TotalDurationNanos int64       `json:"total_duration_nanos"`
	AvgDurationNanos   int64       `json:"avg_duration_nanos"`
}

// HealthBlock carries the loop liveness and the dark-loop verdict. DarkLoop is
// the headline alarm: coverage is zero on nonzero eligible, the loop is not
// running or has gone stale, work is dead-lettered, a provider is down, or an
// auditor was unavailable.
type HealthBlock struct {
	LoopPresent          bool             `json:"loop_present"`
	LoopRunning          bool             `json:"loop_running"`
	LastTickAgeNanos     int64            `json:"last_tick_age_nanos"`
	Stale                bool             `json:"stale"`
	PendingIssues        int              `json:"pending_issues"`
	InflightIssues       int              `json:"inflight_issues"`
	Retries              int              `json:"retries"`
	DeadLetters          int              `json:"dead_letters"`
	Providers            []ProviderHealth `json:"providers,omitempty"`
	UnavailableProviders int              `json:"unavailable_providers"`
	UnavailableAudits    int              `json:"unavailable_audits"` // UNAVAILABLE outcomes
	NewestRecordAgeNanos int64            `json:"newest_record_age_nanos"`
	OldestRecordAgeNanos int64            `json:"oldest_record_age_nanos"`
	FreshnessSampleCount int              `json:"freshness_sample_count"`
	DarkLoop             bool             `json:"dark_loop"`
}

// CrossAuditScorecard is the control-pane envelope. OK is true only when there is
// no coverage, independence, finding, or dark-loop debt; Debts names every active
// one and Alarms carries the human-readable dark-loop reasons.
type CrossAuditScorecard struct {
	Schema       string            `json:"schema"`
	OK           bool              `json:"ok"`
	Verdict      string            `json:"verdict"`
	Grade        string            `json:"grade"`
	Finding      string            `json:"finding"`
	Reason       string            `json:"reason"`
	NextAction   string            `json:"next_action"`
	Debts        []string          `json:"debts"`
	Alarms       []string          `json:"alarms"`
	Coverage     CoverageBlock     `json:"coverage"`
	Independence IndependenceBlock `json:"independence"`
	Quality      QualityBlock      `json:"quality"`
	ModelMix     ModelMixBlock     `json:"model_mix"`
	Economics    EconomicsBlock    `json:"economics"`
	Health       HealthBlock       `json:"health"`
}

// ScoreCrossAudit is the pure, deterministic fold from recorded audit outcomes +
// loop health to the control-pane scorecard. No clock, no RNG, no I/O.
func ScoreCrossAudit(in CrossAuditInput) CrossAuditScorecard {
	verdicts := map[string]int{}
	severities := map[string]int{}
	authors := map[string]int{}
	auditors := map[string]int{}
	calibrations := map[string]int{}
	basisCounts := map[string]int{}

	// Per-issue aggregation so an UNAVAILABLE/REFUSED attempt never counts an issue
	// as audited when a completed audit for it does not exist.
	issueCompleted := map[int]bool{}
	issueSeen := map[int]bool{}

	// Per-class completed/refute for yield.
	classCompleted := map[string]int{}
	classRefutes := map[string]int{}

	var indAdmitted, indRefused, indUnknown int
	var passes, findings, completed, unavailableAudits int
	var inTok, outTok, totTok, costMicros int64
	var tokenSamples, costSamples, latencySamples int
	var totalDuration int64
	var newestAge, oldestAge int64
	var freshnessSamples int
	haveAge := false

	for _, r := range in.Records {
		outcome := normalizeOutcome(r.Outcome)
		verdicts[outcome]++
		issueSeen[r.IssueNumber] = true
		if completedOutcome(outcome) {
			issueCompleted[r.IssueNumber] = true
			completed++
			cls := normalizeClass(r.Class)
			classCompleted[cls]++
			if outcome == OutcomePass {
				passes++
			}
			if outcome == OutcomeRefute {
				findings++
				classRefutes[cls]++
				severities[normalizeSeverity(r.Severity)]++
			}
		}
		if outcome == OutcomeUnavailable {
			unavailableAudits++
		}

		switch normalizeIndependence(r.Independence) {
		case IndependenceAdmitted:
			indAdmitted++
		case IndependenceRefused:
			indRefused++
		default:
			indUnknown++
		}

		if m := strings.TrimSpace(r.AuthorModel); m != "" {
			authors[m]++
		}
		if m := strings.TrimSpace(r.AuditorModel); m != "" {
			auditors[m]++
		}
		if c := strings.TrimSpace(r.CalibrationVersion); c != "" {
			calibrations[c]++
		}

		if r.CostMeasured {
			tokenSamples++
			inTok += r.InputTokens
			outTok += r.OutputTokens
			totTok += r.TotalTokens
			basis := normalizeBasis(r.CostBasis)
			basisCounts[basis]++
			if basis != "unreported" {
				costSamples++
				costMicros += r.CostMicrosUSD
			}
		}
		if r.DurationNanos > 0 {
			latencySamples++
			totalDuration += r.DurationNanos
		}
		if in.NowUnixNano > 0 && r.RecordedAtUnixNano > 0 {
			age := in.NowUnixNano - r.RecordedAtUnixNano
			if age < 0 {
				age = 0
			}
			freshnessSamples++
			if !haveAge {
				newestAge, oldestAge, haveAge = age, age, true
			} else {
				if age < newestAge {
					newestAge = age
				}
				if age > oldestAge {
					oldestAge = age
				}
			}
		}
	}

	audited := len(issueCompleted)
	attempted := len(issueSeen)
	eligible := in.EligibleIssues
	pending := eligible - audited
	if pending < 0 {
		pending = 0
	}
	missing := eligible - attempted
	if missing < 0 {
		missing = 0
	}
	unavailableOnly := attempted - audited
	if unavailableOnly < 0 {
		unavailableOnly = 0
	}

	avgDuration := int64(0)
	if latencySamples > 0 {
		avgDuration = totalDuration / int64(latencySamples)
	}

	unavailableProviders := 0
	for _, p := range in.Loop.Providers {
		if !p.Available {
			unavailableProviders++
		}
	}
	lastTickAge := int64(0)
	stale := false
	if in.Loop.Present && in.NowUnixNano > 0 && in.Loop.LastTickUnixNano > 0 {
		lastTickAge = in.NowUnixNano - in.Loop.LastTickUnixNano
		if lastTickAge < 0 {
			lastTickAge = 0
		}
		if in.StaleAfterNanos > 0 && lastTickAge > in.StaleAfterNanos {
			stale = true
		}
	}
	loopRunning := in.Loop.Present && in.Loop.Running

	// Dark loop: the audit machinery is silently not covering the work.
	coverageZero := eligible > 0 && audited == 0
	darkLoop := coverageZero || !loopRunning || stale ||
		in.Loop.DeadLetters > 0 || unavailableProviders > 0 || unavailableAudits > 0

	perClass := make([]ClassYield, 0, len(classCompleted))
	for cls, n := range classCompleted {
		perClass = append(perClass, ClassYield{
			Class:     cls,
			Completed: n,
			Refutes:   classRefutes[cls],
			YieldRate: rate(classRefutes[cls], n),
		})
	}
	sort.Slice(perClass, func(i, j int) bool { return perClass[i].Class < perClass[j].Class })

	sc := CrossAuditScorecard{
		Schema: crossAuditSchema,
		Coverage: CoverageBlock{
			Eligible: eligible, Records: len(in.Records), Attempted: attempted, Audited: audited,
			Pending: pending, Missing: missing, UnavailableOnly: unavailableOnly,
			AuditedRate: rate(audited, eligible), AttemptedRate: rate(attempted, eligible),
			RateBasis: "audited (completed) / eligible; UNAVAILABLE and REFUSED are not audited",
		},
		Independence: IndependenceBlock{
			Total: len(in.Records), Admitted: indAdmitted, Refused: indRefused, Unknown: indUnknown,
			AdmittedRate: rate(indAdmitted, len(in.Records)),
			RateBasis:    "admitted / all records; refused and unknown shown separately",
		},
		Quality: QualityBlock{
			Verdicts: tallySorted(verdicts), Severities: tallySorted(severities),
			Completed: completed, Passes: passes, Findings: findings,
			PassRate:      rate(passes, completed),
			PassRateBasis: "PASS / completed audits — an independence-checked no-refutation rate, NOT a correctness rate",
			PerClassYield: perClass,
		},
		ModelMix: ModelMixBlock{
			Authors: tallySorted(authors), Auditors: tallySorted(auditors), Calibrations: tallySorted(calibrations),
		},
		Economics: EconomicsBlock{
			Records: len(in.Records), TokenSampleCount: tokenSamples,
			InputTokens: inTok, OutputTokens: outTok, TotalTokens: totTok,
			CostSampleCount: costSamples, CostMicrosUSD: costMicros, CostBasisCounts: tallySorted(basisCounts),
			LatencySampleCount: latencySamples, TotalDurationNanos: totalDuration, AvgDurationNanos: avgDuration,
		},
		Health: HealthBlock{
			LoopPresent: in.Loop.Present, LoopRunning: loopRunning, LastTickAgeNanos: lastTickAge, Stale: stale,
			PendingIssues: in.Loop.PendingIssues, InflightIssues: in.Loop.InflightIssues,
			Retries: in.Loop.Retries, DeadLetters: in.Loop.DeadLetters,
			Providers: in.Loop.Providers, UnavailableProviders: unavailableProviders,
			UnavailableAudits:    unavailableAudits,
			NewestRecordAgeNanos: newestAge, OldestRecordAgeNanos: oldestAge, FreshnessSampleCount: freshnessSamples,
			DarkLoop: darkLoop,
		},
	}

	// Debts (worst-first order): dark loop, coverage, findings, independence.
	var debts, alarms []string
	if coverageZero {
		debts = append(debts, "coverage-dark")
		alarms = append(alarms, fmt.Sprintf("zero audited of %d eligible — the audit loop is dark", eligible))
	}
	if !loopRunning {
		alarms = append(alarms, "background audit loop is not running")
	}
	if stale {
		alarms = append(alarms, fmt.Sprintf("loop last ticked %dns ago, past the %dns freshness budget", lastTickAge, in.StaleAfterNanos))
	}
	if in.Loop.DeadLetters > 0 {
		alarms = append(alarms, fmt.Sprintf("%d dead-lettered audit(s)", in.Loop.DeadLetters))
	}
	if unavailableProviders > 0 {
		alarms = append(alarms, fmt.Sprintf("%d auditor provider(s) unavailable", unavailableProviders))
	}
	if unavailableAudits > 0 {
		alarms = append(alarms, fmt.Sprintf("%d audit(s) recorded UNAVAILABLE", unavailableAudits))
	}
	if darkLoop {
		debts = appendUnique(debts, "dark-loop")
	}
	if pending > 0 {
		debts = append(debts, "coverage-incomplete")
	}
	if findings > 0 {
		debts = append(debts, "open-findings")
	}
	if indRefused > 0 || indUnknown > 0 {
		debts = append(debts, "independence-unproven")
	}

	sc.Debts = debts
	sc.Alarms = alarms
	sc.OK = len(debts) == 0
	sc.Grade = crossAuditGrade(sc)
	sc.Verdict, sc.Finding, sc.Reason, sc.NextAction = crossAuditVerdict(sc)
	return sc
}

// crossAuditGrade folds the debt set into an A..F letter. A dark loop is an
// automatic F — an unobserved loop is the worst state regardless of what little
// coverage exists. Otherwise the grade tracks the audited coverage rate.
func crossAuditGrade(sc CrossAuditScorecard) string {
	if sc.Health.DarkLoop {
		return "F"
	}
	return mathx.Grade100(int(round(100 * sc.Coverage.AuditedRate)))
}

func crossAuditVerdict(sc CrossAuditScorecard) (verdict, finding, reason, next string) {
	if sc.OK {
		return "OK",
			fmt.Sprintf("audit loop healthy: %d/%d eligible audited, %d finding(s), independence admitted on all %d records",
				sc.Coverage.Audited, sc.Coverage.Eligible, sc.Quality.Findings, sc.Independence.Total),
			"coverage complete, loop live, every audit independence-admitted",
			"hold coverage; raise the freshness budget or add per-class targets to push observability toward now"
	}
	// Worst-first next action.
	switch {
	case contains(sc.Debts, "dark-loop"):
		next = "restore the audit loop: bring the auditor provider back, restart the loop, or clear dead letters — an unobserved loop is a security-theater claim"
	case contains(sc.Debts, "coverage-incomplete"):
		next = fmt.Sprintf("audit the %d pending eligible issue(s) (%d missing a receipt, %d only unavailable/refused)", sc.Coverage.Pending, sc.Coverage.Missing, sc.Coverage.UnavailableOnly)
	case contains(sc.Debts, "open-findings"):
		next = fmt.Sprintf("triage %d open refute finding(s)", sc.Quality.Findings)
	default:
		next = fmt.Sprintf("prove independence: %d refused, %d unknown of %d records", sc.Independence.Refused, sc.Independence.Unknown, sc.Independence.Total)
	}
	finding = fmt.Sprintf("audit loop NOT green: %s (audited %d/%d, dark_loop=%v)",
		strings.Join(sc.Debts, ","), sc.Coverage.Audited, sc.Coverage.Eligible, sc.Health.DarkLoop)
	if len(sc.Alarms) > 0 {
		reason = strings.Join(sc.Alarms, "; ")
	} else {
		reason = "coverage or independence debt without a live dark-loop alarm"
	}
	return "ACTION", finding, reason, next
}

// RenderCrossAudit writes the human scorecard: headline, coverage, independence,
// quality (with the pass-rate provenance label), economics with sample counts,
// and the dark-loop / alarm block worst-first.
func RenderCrossAudit(w io.Writer, sc CrossAuditScorecard) {
	fmt.Fprintf(w, "cross-audit scorecard: %s grade %s (dark_loop=%v)\n", sc.Verdict, sc.Grade, sc.Health.DarkLoop)
	c := sc.Coverage
	fmt.Fprintf(w, "  coverage: audited %d/%d eligible (rate %.3f); pending %d (missing %d, unavailable/refused-only %d) [%s]\n",
		c.Audited, c.Eligible, c.AuditedRate, c.Pending, c.Missing, c.UnavailableOnly, c.RateBasis)
	i := sc.Independence
	fmt.Fprintf(w, "  independence: admitted %d, refused %d, unknown %d of %d (rate %.3f) [%s]\n",
		i.Admitted, i.Refused, i.Unknown, i.Total, i.AdmittedRate, i.RateBasis)
	q := sc.Quality
	fmt.Fprintf(w, "  quality: %d finding(s); pass_rate %.3f over %d completed [%s]\n",
		q.Findings, q.PassRate, q.Completed, q.PassRateBasis)
	e := sc.Economics
	fmt.Fprintf(w, "  economics: tokens %d (n=%d), cost %dµ USD (n=%d), latency avg %dns (n=%d) — sample counts are provenance\n",
		e.TotalTokens, e.TokenSampleCount, e.CostMicrosUSD, e.CostSampleCount, e.AvgDurationNanos, e.LatencySampleCount)
	fmt.Fprintf(w, "  finding: %s\n", sc.Finding)
	for _, a := range sc.Alarms {
		fmt.Fprintf(w, "  ALARM: %s\n", a)
	}
	fmt.Fprintf(w, "  next: %s\n", sc.NextAction)
}

func normalizeOutcome(s string) string {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case OutcomePass:
		return OutcomePass
	case OutcomeRefute:
		return OutcomeRefute
	case OutcomeInconclusive:
		return OutcomeInconclusive
	case OutcomeUnavailable:
		return OutcomeUnavailable
	case OutcomeRefused:
		return OutcomeRefused
	default:
		return OutcomeUnknown
	}
}

func completedOutcome(o string) bool {
	return o == OutcomePass || o == OutcomeRefute || o == OutcomeInconclusive
}

func normalizeIndependence(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "admitted", "admit":
		return IndependenceAdmitted
	case "refused", "refuse":
		return IndependenceRefused
	default:
		return IndependenceUnknown
	}
}

func normalizeSeverity(s string) string {
	s = strings.ToUpper(strings.TrimSpace(s))
	switch s {
	case "NONE", "UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL":
		return s
	default:
		return "UNKNOWN"
	}
}

func normalizeBasis(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "provider-reported", "host-calculated", "unreported":
		return s
	default:
		return "unreported"
	}
}

func normalizeClass(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return "unclassified"
	}
	return s
}

func rate(num, den int) float64 {
	if den <= 0 {
		return 0
	}
	return round3(float64(num) / float64(den))
}

func tallySorted(m map[string]int) []NameCount {
	out := make([]NameCount, 0, len(m))
	for k, v := range m {
		out = append(out, NameCount{Name: k, Count: v})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

func appendUnique(ss []string, want string) []string {
	if contains(ss, want) {
		return ss
	}
	return append(ss, want)
}
