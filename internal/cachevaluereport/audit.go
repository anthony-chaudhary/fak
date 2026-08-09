package cachevaluereport

// The Track-2 savings AUDIT (#2780) — a per-date + cumulative reconciliation of the
// OBSERVED provider-$ cache economics the durable ledger records, folded here as a
// reusable library so the throwaway Python fold that produced "net API cost reduction
// = 80.8%" becomes a versioned, tested command anyone can re-run.
//
// The audit answers three questions the raw ledger does not, without ever blending a
// projection into a witness (the Track-1/Track-2 provenance fence still holds):
//
//   - #2780  reconcile the dollar axes per date and cumulatively: cache-read fraction,
//     rebate vs write-premium, the no-cache COUNTERFACTUAL, the net dollar reduction,
//     and the cold-write turns that read NET-NEGATIVE until reuse repays the write.
//   - #2781  split the fold by FIDELITY (lossless provider hits vs bounded-lossy
//     compaction shedding) so a "lossless-or-better" SLO can be gated, not asserted.
//   - #2782  keep DOLLAR-BLIND rows (token evidence, no trusted price) OUT of the
//     dollar-reduction denominator so a bootstrap day of unpriced rows cannot silently
//     understate the cumulative saving — while still surfacing their token volume.
//
// Everything here is PURE (rows in, report out; the only time input is `now`, used to
// stamp GeneratedAt) so the fold is deterministic and testable.

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// AuditSchema tags the reconciliation report shape.
const AuditSchema = "fak-cache-savings-audit/1"

// Fidelity classifies a savings mechanism by how faithfully its saving preserves the
// original context. It is the SINGLE source of truth the write path (NewSavingsRows)
// and the audit/gate folds share, so a producer can never drift from a consumer:
//
//   - lossless    — byte-identical reuse (provider prompt cache); no context is lost.
//   - bounded     — bounded-lossy shed within a budget (fak compaction); recoverable
//     only up to the dropped tokens, but the drop is capped and WITNESSED.
//   - recoverable — lost but reconstructable on demand (reserved; no Track-2 mechanism
//     emits it today).
//   - passive     — no fidelity claim (usage counters).
//   - unknown     — an unrecognized/blank mechanism; treated as worse than any floor.
const (
	FidelityLossless    = "lossless"
	FidelityBounded     = "bounded"
	FidelityRecoverable = "recoverable"
	FidelityPassive     = "passive"
	FidelityUnknown     = "unknown"
)

// Fidelity maps a Track-2 mechanism to its fidelity class. Kept deliberately small and
// total so both the writer and every consumer resolve it identically.
func Fidelity(mechanism string) string {
	switch strings.TrimSpace(mechanism) {
	case "provider_prompt_cache":
		return FidelityLossless
	case "compaction_shed":
		return FidelityBounded
	case "":
		return FidelityUnknown
	default:
		return FidelityUnknown
	}
}

// rowFidelity returns a row's stamped Fidelity, falling back to deriving it from the
// mechanism for rows written before the field existed (#2781 is additive).
func rowFidelity(r SavingsRow) string {
	if f := strings.TrimSpace(r.Fidelity); f != "" {
		return f
	}
	return Fidelity(r.Mechanism)
}

// isDollarBlindRow reports whether a row carries real token evidence but no trusted
// dollar figure — the bootstrap-day case #2782 guards. It trusts the explicit
// dollar_blind marker first (the writer and the parse-time normalizer both set it),
// then falls back to the structural test (token axes present, every dollar axis zero,
// no base price) so a row written before the marker existed is still caught.
func isDollarBlindRow(r SavingsRow) bool {
	if r.DollarStatus == SavingsDollarStatusBlind {
		return true
	}
	hasTokens := r.CacheReadTokens > 0 || r.CacheCreationTokens > 0 || r.CompactionShedTokens > 0
	unpriced := r.InputPerMTokUSD == 0 && r.OutputPerMTokUSD == 0 &&
		r.RebateUSD == 0 && r.WritePremiumUSD == 0 && r.SpendUSD == 0 && r.CompactionSavedUSD == 0
	return hasTokens && unpriced
}

// counterfactualUSD is the no-cache full cost of a provider prompt-cache row: what the
// session would have paid if the prompt cache did not exist, so every prompt token
// (fresh input + cache_read + cache_creation) is billed at the base input rate and the
// output at the base output rate. It is the denominator of the dollar-reduction ratio.
// It is 0 for a dollar-blind row (no base price) — which is exactly why such rows fall
// out of both the numerator and the denominator (#2782).
func counterfactualUSD(r SavingsRow) float64 {
	if r.Mechanism != "provider_prompt_cache" || isDollarBlindRow(r) {
		return 0
	}
	promptTokens := float64(r.InputTokens) + float64(r.CacheReadTokens) + float64(r.CacheCreationTokens)
	return perMTok(r.InputPerMTokUSD, promptTokens) + perMTok(r.OutputPerMTokUSD, float64(r.OutputTokens))
}

// AuditTotals is one reconciliation account — either a single date or the cumulative
// roll-up. Token axes are summed OBSERVED counts; dollar axes are the priced
// projection. The derived fields (fraction, reduction, net-negative count) are folded,
// not stored on rows, so they can never drift from the axes they summarize.
type AuditTotals struct {
	Rows              int     `json:"rows"`
	InputTokens       uint64  `json:"input_tokens"`
	CacheReadTokens   uint64  `json:"cache_read_tokens"`
	CacheCreationToks uint64  `json:"cache_creation_tokens"`
	OutputTokens      uint64  `json:"output_tokens"`
	CompactionShed    uint64  `json:"compaction_shed_tokens"`
	PromptTokens      uint64  `json:"prompt_tokens"` // input + cache_read + cache_creation
	SavedTokenEquiv   float64 `json:"saved_token_equiv"`

	RebateUSD          float64 `json:"rebate_usd"`
	WritePremiumUSD    float64 `json:"write_premium_usd"`
	SpendUSD           float64 `json:"spend_usd"`
	CompactionSavedUSD float64 `json:"compaction_saved_usd"`
	NetUSD             float64 `json:"net_usd"`
	CounterfactualUSD  float64 `json:"counterfactual_usd"` // priced provider rows only

	// CacheReadFraction is cache_read / prompt_tokens — the share of prompt tokens
	// served from cache. Nil when there were no prompt tokens.
	CacheReadFraction *float64 `json:"cache_read_fraction,omitempty"`
	// DollarReductionPct is 100 * (rebate - write_premium) / counterfactual over PRICED
	// provider rows only — the "net API cost reduction" headline. Note (rebate -
	// write_premium) equals exactly (counterfactual - actual_spend), so this is the true
	// dollar saving vs a no-cache baseline. Nil when no priced counterfactual exists
	// (e.g. a fully dollar-blind window), so a blind day never renders as "0% reduction".
	DollarReductionPct *float64 `json:"dollar_reduction_pct,omitempty"`
	// ColdWriteRows is the count of priced provider rows whose write premium outweighs
	// the read rebate (RebateUSD < WritePremiumUSD) — cold cache-write turns where the
	// write is not yet repaid by reuse. It is the write-vs-read signal the issue names,
	// independent of the residual API spend the full NET also subtracts.
	ColdWriteRows int `json:"cold_write_rows"`
	// DollarBlindRows / DollarBlindSavedTokenEquiv surface the token volume that has NO
	// dollar figure, so the blind rows excluded from the ratio stay visible (#2782).
	DollarBlindRows            int     `json:"dollar_blind_rows"`
	DollarBlindSavedTokenEquiv float64 `json:"dollar_blind_saved_token_equiv"`
}

func (t *AuditTotals) add(r SavingsRow) {
	t.Rows++
	t.InputTokens += r.InputTokens
	t.CacheReadTokens += r.CacheReadTokens
	t.CacheCreationToks += r.CacheCreationTokens
	t.OutputTokens += r.OutputTokens
	t.CompactionShed += r.CompactionShedTokens
	t.PromptTokens += r.InputTokens + r.CacheReadTokens + r.CacheCreationTokens
	t.SavedTokenEquiv += r.SavedTokenEquiv
	t.RebateUSD += r.RebateUSD
	t.WritePremiumUSD += r.WritePremiumUSD
	t.SpendUSD += r.SpendUSD
	t.CompactionSavedUSD += r.CompactionSavedUSD
	t.NetUSD += r.NetUSD
	t.CounterfactualUSD += counterfactualUSD(r)
	if isDollarBlindRow(r) {
		t.DollarBlindRows++
		t.DollarBlindSavedTokenEquiv += r.SavedTokenEquiv
	} else if r.Mechanism == "provider_prompt_cache" && r.RebateUSD < r.WritePremiumUSD {
		t.ColdWriteRows++
	}
}

// derive computes the ratio fields from the summed axes. Called once per account after
// all rows are added.
func (t *AuditTotals) derive() {
	if t.PromptTokens > 0 {
		f := float64(t.CacheReadTokens) / float64(t.PromptTokens)
		t.CacheReadFraction = &f
	}
	if t.CounterfactualUSD > 0 {
		pct := 100 * (t.RebateUSD - t.WritePremiumUSD) / t.CounterfactualUSD
		t.DollarReductionPct = &pct
	}
}

// AuditDateRow is one calendar day's reconciliation plus the running cumulative NET
// through that day (dollar-blind rows do not move the cumulative dollar line).
type AuditDateRow struct {
	Date             string  `json:"date"`
	AuditTotals              // embedded per-date account
	CumulativeNetUSD float64 `json:"cumulative_net_usd"`
}

// AuditMechanismRow / AuditFidelityRow are the two roll-up splits the report carries so
// a reader can see provider-cache vs compaction (mechanism) and lossless vs bounded
// (fidelity) side by side without re-folding.
type AuditMechanismRow struct {
	Mechanism       string  `json:"mechanism"`
	Fidelity        string  `json:"fidelity"`
	Rows            int     `json:"rows"`
	SavedTokenEquiv float64 `json:"saved_token_equiv"`
	NetUSD          float64 `json:"net_usd"`
	DollarBlindRows int     `json:"dollar_blind_rows"`
}

type AuditFidelityRow struct {
	Fidelity        string  `json:"fidelity"`
	Rows            int     `json:"rows"`
	SavedTokenEquiv float64 `json:"saved_token_equiv"`
	ShedTokens      uint64  `json:"shed_tokens,omitempty"`
}

// AuditReport is the whole reconciliation: per-date rows, the cumulative account, the
// mechanism/fidelity splits, and a verdict. It is JSON-serializable for `--json`.
type AuditReport struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	Since       string `json:"since,omitempty"`

	Dates          []AuditDateRow      `json:"dates"`
	Cumulative     AuditTotals         `json:"cumulative"`
	MechanismSplit []AuditMechanismRow `json:"mechanism_split"`
	FidelitySplit  []AuditFidelityRow  `json:"fidelity_split"`

	// EfficiencyOutliers are per-session cache-efficiency regressions (#1992): provider-
	// cache sessions whose token hit-rate falls well below the window's session-median or
	// whose write-amplification is high, so a churny session (prefix instability) cannot
	// hide inside a healthy day-aggregate hit-rate. Empty (and omitted) when every session
	// tracks the baseline.
	EfficiencyOutliers []SessionEfficiency `json:"efficiency_outliers,omitempty"`

	// NetReconciliationDriftUSD is Σ(stored NetUSD − re-derived NetUSDComputed) across
	// all rows: it must be ~0, and a non-zero value means a producer wrote a NET that
	// does not equal its own component dollars — an honesty tripwire, not a metric.
	NetReconciliationDriftUSD float64 `json:"net_reconciliation_drift_usd"`

	Verdict    string `json:"verdict"` // MEASURED | INSUFFICIENT
	Finding    string `json:"finding"`
	NextAction string `json:"next_action,omitempty"`
	OK         bool   `json:"ok"`
}

// FoldAudit reconciles Track-2 savings rows into the per-date + cumulative audit. It is
// pure: bucketing comes from each row's own Date; rows with an unparseable date are
// skipped (mirroring the Track-1/Track-2 folds). `now` only stamps GeneratedAt.
func FoldAudit(rows []SavingsRow, now time.Time) AuditReport {
	byDate := map[string]*AuditTotals{}
	mechAgg := map[string]*AuditMechanismRow{}
	fidAgg := map[string]*AuditFidelityRow{}
	var drift float64

	for _, r := range rows {
		normalizeSavingsDimensions(&r)
		if _, err := time.Parse("2006-01-02", r.Date); err != nil {
			continue
		}
		drift += r.NetUSD - r.NetUSDComputed()

		acc := byDate[r.Date]
		if acc == nil {
			acc = &AuditTotals{}
			byDate[r.Date] = acc
		}
		acc.add(r)

		m := mechAgg[r.Mechanism]
		if m == nil {
			m = &AuditMechanismRow{Mechanism: r.Mechanism, Fidelity: rowFidelity(r)}
			mechAgg[r.Mechanism] = m
		}
		m.Rows++
		m.SavedTokenEquiv += r.SavedTokenEquiv
		m.NetUSD += r.NetUSD
		if isDollarBlindRow(r) {
			m.DollarBlindRows++
		}

		fid := rowFidelity(r)
		f := fidAgg[fid]
		if f == nil {
			f = &AuditFidelityRow{Fidelity: fid}
			fidAgg[fid] = f
		}
		f.Rows++
		f.SavedTokenEquiv += r.SavedTokenEquiv
		f.ShedTokens += r.CompactionShedTokens
	}

	dates := make([]string, 0, len(byDate))
	for d := range byDate {
		dates = append(dates, d)
	}
	sort.Strings(dates)

	rep := AuditReport{
		Schema:                    AuditSchema,
		GeneratedAt:               now.UTC().Format(time.RFC3339),
		NetReconciliationDriftUSD: drift,
		OK:                        true,
	}

	var cum AuditTotals
	var cumulativeNet float64
	for _, d := range dates {
		acc := byDate[d]
		acc.derive()

		// Fold this date into the cumulative account.
		cum.Rows += acc.Rows
		cum.InputTokens += acc.InputTokens
		cum.CacheReadTokens += acc.CacheReadTokens
		cum.CacheCreationToks += acc.CacheCreationToks
		cum.OutputTokens += acc.OutputTokens
		cum.CompactionShed += acc.CompactionShed
		cum.PromptTokens += acc.PromptTokens
		cum.SavedTokenEquiv += acc.SavedTokenEquiv
		cum.RebateUSD += acc.RebateUSD
		cum.WritePremiumUSD += acc.WritePremiumUSD
		cum.SpendUSD += acc.SpendUSD
		cum.CompactionSavedUSD += acc.CompactionSavedUSD
		cum.NetUSD += acc.NetUSD
		cum.CounterfactualUSD += acc.CounterfactualUSD
		cum.ColdWriteRows += acc.ColdWriteRows
		cum.DollarBlindRows += acc.DollarBlindRows
		cum.DollarBlindSavedTokenEquiv += acc.DollarBlindSavedTokenEquiv

		cumulativeNet += acc.NetUSD
		rep.Dates = append(rep.Dates, AuditDateRow{
			Date:             d,
			AuditTotals:      *acc,
			CumulativeNetUSD: cumulativeNet,
		})
	}
	cum.derive()
	rep.Cumulative = cum
	rep.EfficiencyOutliers = foldSessionOutliers(rows)

	rep.MechanismSplit = sortedMechanismSplit(mechAgg)
	rep.FidelitySplit = sortedFidelitySplit(fidAgg)
	rep.finalize()
	return rep
}

func (rep *AuditReport) finalize() {
	if len(rep.Dates) == 0 {
		rep.Verdict = "INSUFFICIENT"
		rep.Finding = "no OBSERVED-$ savings rows to reconcile"
		rep.NextAction = "append provider cache_read/cache_creation or compaction rows to the Track-2 savings ledger, then re-run"
		return
	}
	rep.Verdict = "MEASURED"
	c := rep.Cumulative
	reduction := "no priced counterfactual (all dollar-blind)"
	if c.DollarReductionPct != nil {
		reduction = fmt.Sprintf("net API $ reduction %.1f%% (rebate $%.2f − write-premium $%.2f vs counterfactual $%.2f)",
			*c.DollarReductionPct, c.RebateUSD, c.WritePremiumUSD, c.CounterfactualUSD)
	}
	rep.Finding = fmt.Sprintf("%d date(s), %d row(s); %s; cumulative NET $%.2f",
		len(rep.Dates), c.Rows, reduction, c.NetUSD)
	if c.DollarBlindRows > 0 {
		rep.NextAction = fmt.Sprintf("%d dollar-blind row(s) (%.0f saved-token-equiv) are excluded from the $ reduction; configure a trusted base price to value them",
			c.DollarBlindRows, c.DollarBlindSavedTokenEquiv)
	}
	if n := len(rep.EfficiencyOutliers); n > 0 {
		w := rep.EfficiencyOutliers[0]
		msg := fmt.Sprintf("%d session(s) flagged for cache-efficiency regression (worst %.1f%% hit-rate on %s); see efficiency_outliers",
			n, w.HitRatePct, w.Date)
		if rep.NextAction == "" {
			rep.NextAction = msg
		} else {
			rep.NextAction += "; " + msg
		}
	}
}

func sortedMechanismSplit(m map[string]*AuditMechanismRow) []AuditMechanismRow {
	out := make([]AuditMechanismRow, 0, len(m))
	for _, v := range m {
		out = append(out, *v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Mechanism < out[j].Mechanism })
	return out
}

func sortedFidelitySplit(m map[string]*AuditFidelityRow) []AuditFidelityRow {
	out := make([]AuditFidelityRow, 0, len(m))
	for _, v := range m {
		out = append(out, *v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Fidelity < out[j].Fidelity })
	return out
}

// --- Lint (#2782) ------------------------------------------------------------------

// LintFinding is one dollar-blind row the lint flags: token evidence with no trusted
// price, so its saving is real on the token axis but invisible on the dollar axis.
type LintFinding struct {
	Date            string  `json:"date"`
	Provider        string  `json:"provider"`
	Mechanism       string  `json:"mechanism"`
	GeneratedAt     string  `json:"generated_at,omitempty"`
	SavedTokenEquiv float64 `json:"saved_token_equiv"`
	CacheReadTokens uint64  `json:"cache_read_tokens,omitempty"`
	CompactionShed  uint64  `json:"compaction_shed_tokens,omitempty"`
	PricingSource   string  `json:"pricing_source,omitempty"`
	Kind            string  `json:"kind"`   // "dollar_blind"
	Reason          string  `json:"reason"` // human one-liner
}

// LintReport is the dollar-blind scan result, JSON-serializable for `--json`.
type LintReport struct {
	Schema          string        `json:"schema"`
	GeneratedAt     string        `json:"generated_at"`
	Since           string        `json:"since,omitempty"`
	RowsScanned     int           `json:"rows_scanned"`
	Findings        []LintFinding `json:"findings"`
	DollarBlindRows int           `json:"dollar_blind_rows"`
	// BlindSavedTokenEquiv is the total token-equiv saving that has no dollar figure —
	// the magnitude a bootstrap day would silently drop from a dollar roll-up (#2782).
	BlindSavedTokenEquiv float64 `json:"blind_saved_token_equiv"`
	Verdict              string  `json:"verdict"` // CLEAN | DOLLAR_BLIND
	Finding              string  `json:"finding"`
	NextAction           string  `json:"next_action,omitempty"`
	OK                   bool    `json:"ok"`
}

const lintSchema = "fak-cache-savings-lint/1"

// LintSavings flags every dollar-blind row (token evidence, no trusted price) so a
// bootstrap day of unpriced rows cannot silently understate cumulative savings. It is
// pure and read-only; findings are ordered by date then provider for determinism.
func LintSavings(rows []SavingsRow, now time.Time) LintReport {
	rep := LintReport{Schema: lintSchema, GeneratedAt: now.UTC().Format(time.RFC3339), OK: true}
	for _, r := range rows {
		normalizeSavingsDimensions(&r)
		if r.Date == "" {
			continue
		}
		rep.RowsScanned++
		if !isDollarBlindRow(r) {
			continue
		}
		rep.Findings = append(rep.Findings, LintFinding{
			Date:            r.Date,
			Provider:        r.Provider,
			Mechanism:       r.Mechanism,
			GeneratedAt:     r.GeneratedAt,
			SavedTokenEquiv: r.SavedTokenEquiv,
			CacheReadTokens: r.CacheReadTokens,
			CompactionShed:  r.CompactionShedTokens,
			PricingSource:   r.PricingSource,
			Kind:            "dollar_blind",
			Reason:          "token evidence recorded but no trusted base price; dollar axes are placeholders at zero, not a priced no-savings result",
		})
		rep.DollarBlindRows++
		rep.BlindSavedTokenEquiv += r.SavedTokenEquiv
	}
	sort.SliceStable(rep.Findings, func(i, j int) bool {
		if rep.Findings[i].Date != rep.Findings[j].Date {
			return rep.Findings[i].Date < rep.Findings[j].Date
		}
		return rep.Findings[i].Provider < rep.Findings[j].Provider
	})
	if rep.DollarBlindRows == 0 {
		rep.Verdict = "CLEAN"
		rep.Finding = fmt.Sprintf("no dollar-blind rows across %d scanned row(s)", rep.RowsScanned)
		return rep
	}
	rep.Verdict = "DOLLAR_BLIND"
	rep.Finding = fmt.Sprintf("%d dollar-blind row(s) across %d scanned (%.0f saved-token-equiv with no dollar figure)",
		rep.DollarBlindRows, rep.RowsScanned, rep.BlindSavedTokenEquiv)
	rep.NextAction = "configure a trusted base price (FAK_CACHEVALUE_INPUT_PER_MTOK_USD/OUTPUT or a known provider/context default) so these rows carry dollars, and exclude them from any dollar-reduction denominator until then"
	return rep
}

// --- Gate (#2781) ------------------------------------------------------------------

// SLOLosslessOrBetter is the one SLO string the gate enforces today: every saving must
// be a lossless provider hit, with bounded-lossy compaction shedding tolerated only up
// to a token budget.
const SLOLosslessOrBetter = "lossless-or-better"

const gateSchema = "fak-cache-savings-gate/1"

// GateViolation is one row (or the window total) that breaches the SLO.
type GateViolation struct {
	Date       string `json:"date,omitempty"`
	Provider   string `json:"provider,omitempty"`
	Mechanism  string `json:"mechanism,omitempty"`
	Fidelity   string `json:"fidelity,omitempty"`
	ShedTokens uint64 `json:"shed_tokens,omitempty"`
	Reason     string `json:"reason"`
}

// GateReport is the SLO verdict, JSON-serializable for `--json`. Pass=false is the
// non-zero exit condition.
type GateReport struct {
	Schema                 string          `json:"schema"`
	GeneratedAt            string          `json:"generated_at"`
	Since                  string          `json:"since,omitempty"`
	SLO                    string          `json:"slo"`
	CompactionBudgetTokens uint64          `json:"compaction_budget_tokens"`
	RowsScanned            int             `json:"rows_scanned"`
	BoundedShedTokens      uint64          `json:"bounded_shed_tokens"`
	Violations             []GateViolation `json:"violations,omitempty"`
	Pass                   bool            `json:"pass"`
	Verdict                string          `json:"verdict"` // PASS | FAIL
	Finding                string          `json:"finding"`
	NextAction             string          `json:"next_action,omitempty"`
}

// QualityFloor configures the opt-in Track-2 lossless-or-better SLO gate.
// MinRows prevents a thin corpus from being mistaken for evidence.
type QualityFloor struct {
	Floor   float64 `json:"floor"`
	MinRows int     `json:"min_rows"`
}

// QualityObservation is the minimal Track-2 evidence consumed by the gate.
// Callers project their audit rows into this stable, opt-in gate boundary.
type QualityObservation struct {
	ManagedDollarKnown  bool `json:"managed_cost_known"`
	BaselineDollarKnown bool `json:"baseline_dollar_known"`
	Lossless            bool `json:"lossless"`
}

// QualityFloorResult is the independently inspectable Track-2 judgment.
type QualityFloorResult struct {
	Passed       bool    `json:"passed"`
	Code         string  `json:"code,omitempty"`
	Message      string  `json:"message,omitempty"`
	EligibleRows int     `json:"eligible_rows"`
	LosslessRows int     `json:"lossless_rows"`
	Fidelity     float64 `json:"fidelity"`
	Floor        float64 `json:"floor"`
	MinRows      int     `json:"min_rows"`
}

// CheckTrack2Quality applies an opt-in lossless-or-better floor. Only rows with
// known gate and baseline dollars are judgment-grade, keeping the fidelity SLO
// bound to the same provider-dollar evidence as the Track-2 economics audit.
func CheckTrack2Quality(rows []QualityObservation, gate QualityFloor) QualityFloorResult {
	result := QualityFloorResult{Floor: gate.Floor, MinRows: gate.MinRows}
	if gate.Floor < 0 || gate.Floor > 1 || gate.MinRows < 1 {
		result.Code = "invalid_quality_floor"
		result.Message = "quality floor check requires floor in [0,1] and min_rows >= 1"
		return result
	}
	for _, row := range rows {
		if !row.ManagedDollarKnown || !row.BaselineDollarKnown {
			continue
		}
		result.EligibleRows++
		if row.Lossless {
			result.LosslessRows++
		}
	}
	if result.EligibleRows < gate.MinRows {
		result.Code = "fidelity_corpus_too_thin"
		result.Message = fmt.Sprintf("quality floor check has %d eligible rows; requires at least %d", result.EligibleRows, gate.MinRows)
		return result
	}
	result.Fidelity = float64(result.LosslessRows) / float64(result.EligibleRows)
	if result.Fidelity < gate.Floor {
		result.Code = "fidelity_below_floor"
		result.Message = fmt.Sprintf("Track-2 fidelity %.4f is below floor %.4f", result.Fidelity, gate.Floor)
		return result
	}
	result.Passed = true
	result.Message = "Track-2 fidelity meets the configured floor"
	return result
}

// GateSavings enforces the fidelity SLO over a window of rows: it fails when a row's
// fidelity is worse than the floor (anything but lossless or bounded), or when the
// bounded-lossy compaction shedding in the window exceeds compactionBudgetTokens. A
// budget of 0 means no bounded shedding is tolerated — the strict reading of a
// "lossless-or-better" claim. Pure; the caller maps Pass onto the process exit code.
func GateSavings(rows []SavingsRow, slo string, compactionBudgetTokens uint64, now time.Time) GateReport {
	if strings.TrimSpace(slo) == "" {
		slo = SLOLosslessOrBetter
	}
	rep := GateReport{
		Schema:                 gateSchema,
		GeneratedAt:            now.UTC().Format(time.RFC3339),
		SLO:                    slo,
		CompactionBudgetTokens: compactionBudgetTokens,
	}
	for _, r := range rows {
		normalizeSavingsDimensions(&r)
		if r.Date == "" {
			continue
		}
		rep.RowsScanned++
		fid := rowFidelity(r)
		switch fid {
		case FidelityLossless:
			// Byte-identical reuse always clears a lossless-or-better floor.
		case FidelityBounded:
			rep.BoundedShedTokens += r.CompactionShedTokens
		default:
			rep.Violations = append(rep.Violations, GateViolation{
				Date:      r.Date,
				Provider:  r.Provider,
				Mechanism: r.Mechanism,
				Fidelity:  fid,
				Reason:    fmt.Sprintf("fidelity %q is worse than the %q floor", fid, slo),
			})
		}
	}
	if rep.BoundedShedTokens > compactionBudgetTokens {
		rep.Violations = append(rep.Violations, GateViolation{
			Mechanism:  "compaction_shed",
			Fidelity:   FidelityBounded,
			ShedTokens: rep.BoundedShedTokens,
			Reason: fmt.Sprintf("bounded-lossy shedding %d tokens exceeds the compaction budget of %d",
				rep.BoundedShedTokens, compactionBudgetTokens),
		})
	}
	rep.Pass = len(rep.Violations) == 0
	if rep.Pass {
		rep.Verdict = "PASS"
		rep.Finding = fmt.Sprintf("SLO %q holds across %d row(s); bounded shed %d/%d tokens",
			slo, rep.RowsScanned, rep.BoundedShedTokens, compactionBudgetTokens)
		return rep
	}
	rep.Verdict = "FAIL"
	rep.Finding = fmt.Sprintf("SLO %q breached: %d violation(s) across %d row(s)", slo, len(rep.Violations), rep.RowsScanned)
	rep.NextAction = "raise --compaction-budget to accept the bounded shedding, or reduce compaction/lossy mechanisms until the window is lossless-or-better"
	return rep
}

// --- Session efficiency (#1992) ----------------------------------------------------

// Per-session cache-efficiency tripwires (#1992). The hit-rate flag is RELATIVE: a
// session is churny when its token hit-rate falls sessionHitRegressionPts points below
// the window's own session-median hit-rate, so a fleet running at 98% flags its 80%
// outlier while a fleet genuinely at 80% does not false-positive. The median (not the
// prompt-weighted mean) is the baseline so one pathological high-volume cold session
// cannot drag the reference down and mask milder regressions. The write-amp flag is
// ABSOLUTE: cache_creation exceeding sessionWriteAmpCeiling of cache_read means the
// prefix churned enough to re-write a large share of what it reused (prefix instability),
// independent of the baseline. sessionMinPromptTokens drops trivially small sessions
// whose ratios are noise.
const (
	sessionHitRegressionPts = 15.0
	sessionWriteAmpCeiling  = 0.5
	sessionMinPromptTokens  = 1000
)

// SessionEfficiency is one savings row scored on its own cache efficiency, so a churny
// session cannot hide inside a healthy day-aggregate hit-rate (#1992). HitRatePct and
// WriteAmp are token ratios — pricing-independent, so a dollar-blind row is still
// scored — and Reason names the tripwire(s) that flagged the session.
type SessionEfficiency struct {
	Date         string  `json:"date"`
	GeneratedAt  string  `json:"generated_at,omitempty"`
	Provider     string  `json:"provider,omitempty"`
	PromptTokens uint64  `json:"prompt_tokens"`
	HitRatePct   float64 `json:"hit_rate_pct"` // 100 * cache_read / (input + cache_read + cache_creation)
	WriteAmp     float64 `json:"write_amp"`    // cache_creation / cache_read (0 when no reuse)
	Reason       string  `json:"reason"`
}

// foldSessionOutliers scores each dated provider-cache row on its per-session hit-rate
// and write-amp, then returns only the regressions, worst-first. The hit-rate tripwire
// is measured against the window's session-median hit-rate, so "regression" means "below
// this fleet's own norm" rather than an arbitrary absolute floor. Pure and deterministic:
// the sort is total (hit-rate, then write-amp, then generated_at, then date).
func foldSessionOutliers(rows []SavingsRow) []SessionEfficiency {
	scored := make([]SessionEfficiency, 0, len(rows))
	hits := make([]float64, 0, len(rows))
	for _, r := range rows {
		normalizeSavingsDimensions(&r)
		if _, err := time.Parse("2006-01-02", r.Date); err != nil {
			continue
		}
		if r.Mechanism != "provider_prompt_cache" {
			continue // hit-rate / write-amp are provider prompt-cache concepts
		}
		prompt := r.InputTokens + r.CacheReadTokens + r.CacheCreationTokens
		if prompt < sessionMinPromptTokens {
			continue
		}
		hitPct := 100 * float64(r.CacheReadTokens) / float64(prompt)
		var writeAmp float64
		if r.CacheReadTokens > 0 {
			writeAmp = float64(r.CacheCreationTokens) / float64(r.CacheReadTokens)
		}
		scored = append(scored, SessionEfficiency{
			Date:         r.Date,
			GeneratedAt:  r.GeneratedAt,
			Provider:     r.Provider,
			PromptTokens: prompt,
			HitRatePct:   hitPct,
			WriteAmp:     writeAmp,
		})
		hits = append(hits, hitPct)
	}
	if len(scored) == 0 {
		return nil
	}
	baseline := medianFloats(hits)

	var out []SessionEfficiency
	for _, s := range scored {
		var reasons []string
		if s.HitRatePct < baseline-sessionHitRegressionPts {
			reasons = append(reasons, fmt.Sprintf("hit-rate %.1f%% is %.1f pts below the session-median baseline %.1f%%",
				s.HitRatePct, baseline-s.HitRatePct, baseline))
		}
		if s.WriteAmp > sessionWriteAmpCeiling {
			reasons = append(reasons, fmt.Sprintf("write-amp %.2f exceeds %.2f (cache re-written faster than reused)",
				s.WriteAmp, sessionWriteAmpCeiling))
		}
		if len(reasons) == 0 {
			continue
		}
		s.Reason = strings.Join(reasons, "; ")
		out = append(out, s)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].HitRatePct != out[j].HitRatePct {
			return out[i].HitRatePct < out[j].HitRatePct
		}
		if out[i].WriteAmp != out[j].WriteAmp {
			return out[i].WriteAmp > out[j].WriteAmp
		}
		if out[i].GeneratedAt != out[j].GeneratedAt {
			return out[i].GeneratedAt < out[j].GeneratedAt
		}
		return out[i].Date < out[j].Date
	})
	return out
}

// medianFloats returns the median of xs without mutating it. Empty input yields 0.
func medianFloats(xs []float64) float64 {
	n := len(xs)
	if n == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// --- Renderers ---------------------------------------------------------------------

// RenderAudit renders the reconciliation as a compact, deterministic terminal table.
func RenderAudit(r AuditReport) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "cache-savings audit (Track-2 OBSERVED $, per-date reconciliation) — %s\n", r.Verdict)
	if r.Since != "" {
		fmt.Fprintf(&sb, "  since: %s\n", r.Since)
	}
	fmt.Fprintf(&sb, "  %s\n", r.Finding)
	if r.NextAction != "" {
		fmt.Fprintf(&sb, "  next: %s\n", r.NextAction)
	}
	if len(r.Dates) == 0 {
		return sb.String()
	}
	fmt.Fprintf(&sb, "\n  %-10s  %5s  %12s  %10s  %10s  %10s  %12s  %8s  %6s  %s\n",
		"date", "rows", "prompt_tok", "read_frac", "rebate$", "writeprem$", "net$", "reduct%", "blind", "cum_net$")
	for _, d := range r.Dates {
		fmt.Fprintf(&sb, "  %-10s  %5d  %12d  %10s  %10.2f  %10.2f  %12.2f  %8s  %6d  %.2f\n",
			d.Date, d.Rows, d.PromptTokens, fracStr(d.CacheReadFraction), d.RebateUSD, d.WritePremiumUSD,
			d.NetUSD, pctStr(d.DollarReductionPct), d.DollarBlindRows, d.CumulativeNetUSD)
	}
	c := r.Cumulative
	fmt.Fprintf(&sb, "  %-10s  %5d  %12d  %10s  %10.2f  %10.2f  %12.2f  %8s  %6d  %.2f\n",
		"CUMULATIVE", c.Rows, c.PromptTokens, fracStr(c.CacheReadFraction), c.RebateUSD, c.WritePremiumUSD,
		c.NetUSD, pctStr(c.DollarReductionPct), c.DollarBlindRows, c.NetUSD)
	fmt.Fprintf(&sb, "  counterfactual (no-cache) $%.2f; cold-write turns (write>rebate) %d; NET reconciliation drift $%.6f\n",
		c.CounterfactualUSD, c.ColdWriteRows, r.NetReconciliationDriftUSD)

	if len(r.FidelitySplit) > 0 {
		fmt.Fprintf(&sb, "\n  fidelity split (context-faithfulness of the saving)\n")
		fmt.Fprintf(&sb, "  %-12s  %5s  %16s  %10s\n", "fidelity", "rows", "saved_tok_equiv", "shed_tok")
		for _, f := range r.FidelitySplit {
			fmt.Fprintf(&sb, "  %-12s  %5d  %16.0f  %10d\n", f.Fidelity, f.Rows, f.SavedTokenEquiv, f.ShedTokens)
		}
	}
	if len(r.MechanismSplit) > 0 {
		fmt.Fprintf(&sb, "\n  mechanism split\n")
		fmt.Fprintf(&sb, "  %-22s  %-10s  %5s  %16s  %12s  %6s\n", "mechanism", "fidelity", "rows", "saved_tok_equiv", "net$", "blind")
		for _, m := range r.MechanismSplit {
			fmt.Fprintf(&sb, "  %-22s  %-10s  %5d  %16.0f  %12.2f  %6d\n",
				m.Mechanism, m.Fidelity, m.Rows, m.SavedTokenEquiv, m.NetUSD, m.DollarBlindRows)
		}
	}
	if len(r.EfficiencyOutliers) > 0 {
		fmt.Fprintf(&sb, "\n  cache-efficiency outliers (per-session hit-rate/write-amp regressions)\n")
		fmt.Fprintf(&sb, "  %-10s  %-20s  %-16s  %8s  %9s  %s\n",
			"date", "generated_at", "provider", "hit%", "write_amp", "reason")
		for _, e := range r.EfficiencyOutliers {
			fmt.Fprintf(&sb, "  %-10s  %-20s  %-16s  %8.1f  %9.2f  %s\n",
				e.Date, e.GeneratedAt, e.Provider, e.HitRatePct, e.WriteAmp, e.Reason)
		}
	}
	return sb.String()
}

// RenderLint renders the dollar-blind scan.
func RenderLint(r LintReport) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "cache-savings lint (dollar-blind rows) — %s\n", r.Verdict)
	if r.Since != "" {
		fmt.Fprintf(&sb, "  since: %s\n", r.Since)
	}
	fmt.Fprintf(&sb, "  %s\n", r.Finding)
	if r.NextAction != "" {
		fmt.Fprintf(&sb, "  next: %s\n", r.NextAction)
	}
	if len(r.Findings) == 0 {
		return sb.String()
	}
	fmt.Fprintf(&sb, "\n  %-10s  %-16s  %-22s  %16s  %14s  %s\n",
		"date", "provider", "mechanism", "saved_tok_equiv", "cache_read_tok", "pricing_source")
	for _, f := range r.Findings {
		src := f.PricingSource
		if src == "" {
			src = "<none>"
		}
		fmt.Fprintf(&sb, "  %-10s  %-16s  %-22s  %16.0f  %14d  %s\n",
			f.Date, f.Provider, f.Mechanism, f.SavedTokenEquiv, f.CacheReadTokens, src)
	}
	return sb.String()
}

// RenderGate renders the SLO verdict.
func RenderGate(r GateReport) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "cache-savings gate (SLO %s) — %s\n", r.SLO, r.Verdict)
	if r.Since != "" {
		fmt.Fprintf(&sb, "  since: %s\n", r.Since)
	}
	fmt.Fprintf(&sb, "  %s\n", r.Finding)
	if r.NextAction != "" {
		fmt.Fprintf(&sb, "  next: %s\n", r.NextAction)
	}
	for _, v := range r.Violations {
		loc := strings.TrimSpace(fmt.Sprintf("%s %s %s", v.Date, v.Provider, v.Mechanism))
		if loc == "" {
			loc = "window"
		}
		fmt.Fprintf(&sb, "  - [%s] %s\n", loc, v.Reason)
	}
	return sb.String()
}

func fracStr(p *float64) string {
	if p == nil {
		return "-"
	}
	return fmt.Sprintf("%.4f", *p)
}

func pctStr(p *float64) string {
	if p == nil {
		return "-"
	}
	return fmt.Sprintf("%.1f", *p)
}
