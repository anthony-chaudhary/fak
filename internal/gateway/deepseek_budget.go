package gateway

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// deepseek_budget.go — DeepSeek V4 long-context memory/compute BUDGET calculator
// (#3015, under the DeepSeek V4 support program #3006).
//
// This is a DETERMINISTIC estimator, NOT a benchmark. It turns DeepSeek's
// paper-claimed and card-documented model constants into per-context-length
// weight-storage and compute-proxy budgets so fak can reason about what a 1M
// context COSTS before ever claiming to serve it natively.
//
// PROVENANCE DISCIPLINE is the load-bearing property. Every emitted number carries
// exactly one label from a CLOSED set, and the calculator FAILS CLOSED if it cannot
// honestly attach one:
//
//   - SOURCE_DOCUMENTED: a constant published by DeepSeek or the HF model card
//     (total/active parameter counts, the 1M context ceiling, precision widths).
//   - PAPER_CLAIMED: a RELATIVE ratio DeepSeek's V4 paper asserts versus V3.2 (the
//     27%/10% FLOP and 10%/7% KV reductions at 1M). DeepSeek's OWN claim — never a
//     fak measurement, and never multiplied into an absolute fak number without a
//     witnessed baseline.
//   - MODELED: a value this calculator DERIVES from documented constants under a
//     stated assumption (e.g. an FP4/FP8 mixed-precision weight-storage bound).
//   - WITNESSED: a value fak measured under its own gateway path. NONE exist yet;
//     this deterministic calculator NEVER emits this label — a witnessed row can
//     only come from the #3013/#3014 serving telemetry.
//
// The honest-refusal rule (ClaimNativeSupport) is the whole point of the seam: no
// row and no report may assert "1M native support" while every load-bearing number
// is MODELED or PAPER_CLAIMED. The claim is REFUSED until resident-memory,
// cache-layout, and throughput WITNESSES exist. See the paper's ratios recorded
// here as PAPER_CLAIMED, not as fak benchmarks — they are not a fak result until
// measured under fak.

// DeepSeekBudgetProvenance is the closed label vocabulary above.
type DeepSeekBudgetProvenance string

const (
	// ProvenanceSourceDocumented — published by DeepSeek / the HF card.
	ProvenanceSourceDocumented DeepSeekBudgetProvenance = "SOURCE_DOCUMENTED"
	// ProvenancePaperClaimed — a relative ratio DeepSeek's paper asserts vs V3.2.
	ProvenancePaperClaimed DeepSeekBudgetProvenance = "PAPER_CLAIMED"
	// ProvenanceModeled — derived by this calculator under a stated assumption.
	ProvenanceModeled DeepSeekBudgetProvenance = "MODELED"
	// ProvenanceWitnessed — measured under fak's own path. Never emitted here.
	ProvenanceWitnessed DeepSeekBudgetProvenance = "WITNESSED"
)

// validBudgetProvenance is the closed set a row may carry; anything else is a bug.
var validBudgetProvenance = map[DeepSeekBudgetProvenance]bool{
	ProvenanceSourceDocumented: true,
	ProvenancePaperClaimed:     true,
	ProvenanceModeled:          true,
	ProvenanceWitnessed:        true,
}

// DeepSeekBudgetProvenanceValid reports whether label is a member of the closed
// provenance vocabulary (and non-empty). The report's own invariant test uses it to
// FAIL if any emitted row is unlabeled or carries an off-vocabulary label.
func DeepSeekBudgetProvenanceValid(label DeepSeekBudgetProvenance) bool {
	return validBudgetProvenance[label]
}

// DeepSeek V4 configured constants, sourced as the #3015 grounding records them.
const (
	// DeepSeekV4ProTotalParams / ...ActiveParams — V4 Pro is 1.6T total / 49B active.
	DeepSeekV4ProTotalParams  = 1.6e12
	DeepSeekV4ProActiveParams = 49e9
	// DeepSeekV4FlashTotalParams / ...ActiveParams — V4 Flash is 284B total / 13B active.
	DeepSeekV4FlashTotalParams  = 284e9
	DeepSeekV4FlashActiveParams = 13e9
	// DeepSeekV4MaxContextTokens — both V4 tiers are 1M-context models.
	DeepSeekV4MaxContextTokens = 1_000_000

	// Paper-claimed relative reductions at 1M context vs DeepSeek-V3.2.
	DeepSeekV4ProKVRatioVsV32     = 0.10 // V4 Pro uses 10% of V3.2 KV cache.
	DeepSeekV4ProFLOPRatioVsV32   = 0.27 // V4 Pro uses 27% of V3.2 single-token FLOPs.
	DeepSeekV4FlashKVRatioVsV32   = 0.07 // V4 Flash uses 7% of V3.2 KV cache.
	DeepSeekV4FlashFLOPRatioVsV32 = 0.10 // V4 Flash uses 10% of V3.2 single-token FLOPs.

	// Precision widths (bytes per parameter). FP4 = 4 bits, FP8 = 8 bits. The HF card
	// lists V4 Pro as FP4 (MoE experts) + FP8 (most other params) mixed, so the true
	// resident weight size sits between the all-FP4 floor and the all-FP8 ceiling.
	bytesPerParamFP4 = 0.5
	bytesPerParamFP8 = 1.0

	// DeepSeekBudgetSourceV4Paper / ...V4ProCard name the provenance sources, in the
	// same source-string convention deepseek_pricing.go uses.
	DeepSeekBudgetSourceV4Paper   = "https://arxiv.org/html/2606.19348v1"
	DeepSeekBudgetSourceV4ProCard = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro"
)

// deepSeekBudgetNote is the honest header the report always carries: the paper's
// relative ratios are DeepSeek's claims, not fak benchmarks, until measured under fak.
const deepSeekBudgetNote = "DeepSeek's relative KV/FLOP ratios (PAPER_CLAIMED) are DeepSeek's own vs-V3.2 figures, " +
	"NOT a fak benchmark. No row is WITNESSED until measured under fak's gateway path (#3013/#3014). " +
	"Absolute long-context KV is only estimated (MODELED) when a witnessed V3.2 per-token KV baseline is supplied."

// DeepSeekModelSpec is one V4 tier's configured constants with their provenance.
type DeepSeekModelSpec struct {
	Model            string  `json:"model"`
	TotalParams      float64 `json:"total_params"`
	ActiveParams     float64 `json:"active_params"`
	MaxContextTokens int     `json:"max_context_tokens"`
	// KVCacheRatioVsV32 / FLOPRatioVsV32 are the paper's 1M-context relative reductions.
	KVCacheRatioVsV32 float64                  `json:"kv_cache_ratio_vs_v32"`
	FLOPRatioVsV32    float64                  `json:"flop_ratio_vs_v32"`
	Provenance        DeepSeekBudgetProvenance `json:"provenance"`
	RatioProvenance   DeepSeekBudgetProvenance `json:"ratio_provenance"`
	Sources           []string                 `json:"sources"`
}

// deepSeekBudgetSpecFor resolves the built-in V4 budget spec for a model id. It
// FAILS CLOSED exactly like DeepSeekCachePricingFor: only the two current V4 tiers
// (and the documented pre-V4 aliases mapping to the flash tier) resolve; any other
// id returns ok=false so no caller budgets off a guessed or stale row.
func deepSeekBudgetSpecFor(model string) (DeepSeekModelSpec, bool) {
	switch strings.ToLower(strings.TrimSpace(model)) {
	case "deepseek-v4-pro":
		return DeepSeekModelSpec{
			Model:             "deepseek-v4-pro",
			TotalParams:       DeepSeekV4ProTotalParams,
			ActiveParams:      DeepSeekV4ProActiveParams,
			MaxContextTokens:  DeepSeekV4MaxContextTokens,
			KVCacheRatioVsV32: DeepSeekV4ProKVRatioVsV32,
			FLOPRatioVsV32:    DeepSeekV4ProFLOPRatioVsV32,
			Provenance:        ProvenanceSourceDocumented,
			RatioProvenance:   ProvenancePaperClaimed,
			Sources:           []string{DeepSeekBudgetSourceV4Paper, DeepSeekBudgetSourceV4ProCard},
		}, true
	case "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner":
		return DeepSeekModelSpec{
			Model:             "deepseek-v4-flash",
			TotalParams:       DeepSeekV4FlashTotalParams,
			ActiveParams:      DeepSeekV4FlashActiveParams,
			MaxContextTokens:  DeepSeekV4MaxContextTokens,
			KVCacheRatioVsV32: DeepSeekV4FlashKVRatioVsV32,
			FLOPRatioVsV32:    DeepSeekV4FlashFLOPRatioVsV32,
			Provenance:        ProvenanceSourceDocumented,
			RatioProvenance:   ProvenancePaperClaimed,
			Sources:           []string{DeepSeekBudgetSourceV4Paper},
		}, true
	}
	return DeepSeekModelSpec{}, false
}

// BudgetInputs tunes the deterministic budget. The zero value is a valid,
// maximally-honest default: default context buckets, and KV left ratio-only
// (no absolute long-context KV invented without a witnessed baseline).
type BudgetInputs struct {
	// V32KVBytesPerToken, if > 0, is a DeepSeek-V3.2-class KV-cache size per context
	// token (bytes). Supplying it turns the paper's PAPER_CLAIMED KV ratio into a
	// MODELED absolute per-context KV estimate (ratio × baseline × tokens). Left 0,
	// the calculator REFUSES to invent an absolute long-context memory number.
	V32KVBytesPerToken float64
	// Contexts overrides the default context-length buckets (tokens). Values above a
	// model's documented context ceiling are dropped.
	Contexts []int
}

// defaultBudgetContexts are the 4K / 32K / 128K / 512K / 1M buckets the #3015 scope
// names, clamped per model to the documented context ceiling. The top bucket is the
// decimal 1,000,000-token ceiling itself (DeepSeekV4MaxContextTokens), not 2^20, so
// the "1M" row is actually emitted rather than clamped away as over-ceiling.
var defaultBudgetContexts = []int{4096, 32768, 131072, 524288, DeepSeekV4MaxContextTokens}

// WeightStorageBudget bounds resident weight size for the FP4+FP8 mixed layout. The
// exact expert/dense split is not published, so the calculator emits documented
// bounds instead of a fabricated point estimate.
type WeightStorageBudget struct {
	FloorBytes   float64                  `json:"floor_bytes"`   // all-FP4 lower bound
	CeilingBytes float64                  `json:"ceiling_bytes"` // all-FP8 upper bound
	Provenance   DeepSeekBudgetProvenance `json:"provenance"`
	Note         string                   `json:"note"`
}

// ComputeBudget is the activated-parameter single-token compute proxy plus the
// paper's claimed relative FLOP reduction (kept separate, never multiplied in).
type ComputeBudget struct {
	ActiveParamFLOPsPerToken float64                  `json:"active_param_flops_per_token"` // 2N proxy over active params
	Provenance               DeepSeekBudgetProvenance `json:"provenance"`
	PaperFLOPRatioVsV32      float64                  `json:"paper_flop_ratio_vs_v32"`
	PaperRatioProvenance     DeepSeekBudgetProvenance `json:"paper_ratio_provenance"`
	Note                     string                   `json:"note"`
}

// ContextBudgetRow is one context-length bucket's KV budget. AbsoluteKVBytes is only
// populated (and only MODELED, never WITNESSED) when a V3.2 KV baseline was supplied.
type ContextBudgetRow struct {
	ContextTokens     int                      `json:"context_tokens"`
	KVRatioVsV32      float64                  `json:"kv_ratio_vs_v32"`
	AbsoluteKVBytes   float64                  `json:"absolute_kv_bytes"`
	KVBaselineWitness bool                     `json:"kv_baseline_witness"`
	Provenance        DeepSeekBudgetProvenance `json:"provenance"`
	Note              string                   `json:"note"`
}

// NativeSupportClaim is the honest-refusal verdict. Granted is true only when the
// three load-bearing witnesses exist; the deterministic calculator supplies none,
// so a default report ALWAYS refuses.
type NativeSupportClaim struct {
	Model            string   `json:"model"`
	Claim            string   `json:"claim"`
	Granted          bool     `json:"granted"`
	Reason           string   `json:"reason"`
	MissingWitnesses []string `json:"missing_witnesses"`
}

// DeepSeekBudgetReport is the full per-model deterministic budget.
type DeepSeekBudgetReport struct {
	Model         string              `json:"model"`
	Note          string              `json:"note"`
	Spec          DeepSeekModelSpec   `json:"spec"`
	WeightStorage WeightStorageBudget `json:"weight_storage"`
	Compute       ComputeBudget       `json:"compute"`
	Contexts      []ContextBudgetRow  `json:"contexts"`
	NativeSupport NativeSupportClaim  `json:"native_support"`
}

// DeepSeekBudget builds the deterministic long-context budget for a model id. It
// FAILS CLOSED (ok=false) for any id outside the built-in V4 table.
func DeepSeekBudget(model string, in BudgetInputs) (DeepSeekBudgetReport, bool) {
	spec, ok := deepSeekBudgetSpecFor(model)
	if !ok {
		return DeepSeekBudgetReport{}, false
	}

	weights := WeightStorageBudget{
		FloorBytes:   spec.TotalParams * bytesPerParamFP4,
		CeilingBytes: spec.TotalParams * bytesPerParamFP8,
		Provenance:   ProvenanceModeled,
		Note: "resident weight size bounded by documented FP4 (MoE experts) and FP8 (dense) " +
			"precision widths; exact mixed split not published, so a point estimate is refused.",
	}

	compute := ComputeBudget{
		ActiveParamFLOPsPerToken: 2 * spec.ActiveParams,
		Provenance:               ProvenanceModeled,
		PaperFLOPRatioVsV32:      spec.FLOPRatioVsV32,
		PaperRatioProvenance:     ProvenancePaperClaimed,
		Note: "2N forward-pass proxy over activated parameters (attention-over-context term omitted); " +
			"the paper's relative FLOP reduction is kept separate and never multiplied into a fak number.",
	}

	contexts := in.Contexts
	if len(contexts) == 0 {
		contexts = defaultBudgetContexts
	}
	rows := make([]ContextBudgetRow, 0, len(contexts))
	for _, tokens := range contexts {
		if tokens <= 0 || tokens > spec.MaxContextTokens {
			continue
		}
		row := ContextBudgetRow{
			ContextTokens: tokens,
			KVRatioVsV32:  spec.KVCacheRatioVsV32,
			Provenance:    ProvenancePaperClaimed,
			Note:          "KV ratio-only: no absolute KV without a witnessed V3.2 baseline.",
		}
		if in.V32KVBytesPerToken > 0 {
			// A supplied baseline turns the paper ratio into a MODELED absolute — still
			// a model (ratio × baseline), never WITNESSED for the V4 tier itself.
			row.AbsoluteKVBytes = spec.KVCacheRatioVsV32 * in.V32KVBytesPerToken * float64(tokens)
			row.Provenance = ProvenanceModeled
			row.Note = "absolute KV = paper KV ratio × supplied V3.2 per-token baseline × context tokens (MODELED)."
		}
		rows = append(rows, row)
	}
	// Deterministic order: ascending context length.
	sort.Slice(rows, func(i, j int) bool { return rows[i].ContextTokens < rows[j].ContextTokens })

	report := DeepSeekBudgetReport{
		Model:         spec.Model,
		Note:          deepSeekBudgetNote,
		Spec:          spec,
		WeightStorage: weights,
		Compute:       compute,
		Contexts:      rows,
	}
	report.NativeSupport = report.claimNativeSupport()
	return report, true
}

// claimNativeSupport is the honest-refusal kernel. The deterministic report carries
// no witnessed rows, so it enumerates the three missing witnesses and REFUSES the
// "1M native support" claim rather than launder MODELED/PAPER_CLAIMED numbers into one.
func (r DeepSeekBudgetReport) claimNativeSupport() NativeSupportClaim {
	var missing []string
	if !r.hasWitnessedRow() {
		// Resident-memory witness: any absolute-memory row proven under fak.
		missing = append(missing, "resident-memory witness (measured KV/weights under fak)")
	}
	// Cache-layout and throughput witnesses are structurally absent from a
	// deterministic estimate — they can only come from serving telemetry.
	missing = append(missing,
		"cache-layout witness (observed hybrid KV / state-cache behavior)",
		"throughput witness (measured TTFT/TPOT at the target context)")
	granted := len(missing) == 0
	reason := "REFUSED: every load-bearing figure is MODELED or PAPER_CLAIMED; a native-support claim needs measured witnesses."
	if granted {
		reason = "granted: memory, cache-layout, and throughput witnesses all present."
	}
	return NativeSupportClaim{
		Model:            r.Model,
		Claim:            "1M native support",
		Granted:          granted,
		Reason:           reason,
		MissingWitnesses: missing,
	}
}

// hasWitnessedRow reports whether any emitted row carries the WITNESSED label. The
// deterministic calculator never emits it, so this is always false today — but the
// refusal kernel is written to flip automatically once a witnessed row appears.
func (r DeepSeekBudgetReport) hasWitnessedRow() bool {
	for _, row := range r.Rows() {
		if row.Provenance == ProvenanceWitnessed {
			return true
		}
	}
	return false
}

// BudgetProvenanceRow is one flattened (name, provenance) pair for the every-row
// -is-labeled invariant. The acceptance test asserts every row here is labeled.
type BudgetProvenanceRow struct {
	Name       string
	Provenance DeepSeekBudgetProvenance
}

// Rows flattens every provenance-bearing figure in the report into a single list so
// a caller (and the invariant test) can assert NO row is emitted unlabeled.
func (r DeepSeekBudgetReport) Rows() []BudgetProvenanceRow {
	rows := []BudgetProvenanceRow{
		{Name: "spec.params", Provenance: r.Spec.Provenance},
		{Name: "spec.ratios", Provenance: r.Spec.RatioProvenance},
		{Name: "weight_storage", Provenance: r.WeightStorage.Provenance},
		{Name: "compute.active_param_proxy", Provenance: r.Compute.Provenance},
		{Name: "compute.paper_flop_ratio", Provenance: r.Compute.PaperRatioProvenance},
	}
	for _, c := range r.Contexts {
		rows = append(rows, BudgetProvenanceRow{
			Name:       fmt.Sprintf("context.%d", c.ContextTokens),
			Provenance: c.Provenance,
		})
	}
	return rows
}

// RenderJSON emits the report as indented JSON.
func (r DeepSeekBudgetReport) RenderJSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// RenderMarkdown emits the report as a provenance-labeled markdown page: the honest
// header note, the spec, the weight-storage bounds, the compute proxy, a per-context
// KV table with a provenance column per row, and the native-support refusal line.
func (r DeepSeekBudgetReport) RenderMarkdown() string {
	var b strings.Builder
	fmt.Fprintf(&b, "# DeepSeek %s — long-context budget (MODELED)\n\n", r.Model)
	fmt.Fprintf(&b, "> %s\n\n", r.Note)
	fmt.Fprintf(&b, "## Spec (%s)\n\n", r.Spec.Provenance)
	fmt.Fprintf(&b, "- total params: %.0f\n", r.Spec.TotalParams)
	fmt.Fprintf(&b, "- active params: %.0f\n", r.Spec.ActiveParams)
	fmt.Fprintf(&b, "- max context: %d tokens\n", r.Spec.MaxContextTokens)
	fmt.Fprintf(&b, "- KV ratio vs V3.2: %.2f, FLOP ratio vs V3.2: %.2f (%s)\n",
		r.Spec.KVCacheRatioVsV32, r.Spec.FLOPRatioVsV32, r.Spec.RatioProvenance)
	fmt.Fprintf(&b, "- sources: %s\n\n", strings.Join(r.Spec.Sources, ", "))

	fmt.Fprintf(&b, "## Weight storage (%s)\n\n", r.WeightStorage.Provenance)
	fmt.Fprintf(&b, "- floor (all-FP4): %.3g bytes\n", r.WeightStorage.FloorBytes)
	fmt.Fprintf(&b, "- ceiling (all-FP8): %.3g bytes\n", r.WeightStorage.CeilingBytes)
	fmt.Fprintf(&b, "- note: %s\n\n", r.WeightStorage.Note)

	fmt.Fprintf(&b, "## Compute (%s)\n\n", r.Compute.Provenance)
	fmt.Fprintf(&b, "- active-param FLOPs/token (2N proxy): %.3g\n", r.Compute.ActiveParamFLOPsPerToken)
	fmt.Fprintf(&b, "- paper FLOP ratio vs V3.2: %.2f (%s)\n", r.Compute.PaperFLOPRatioVsV32, r.Compute.PaperRatioProvenance)
	fmt.Fprintf(&b, "- note: %s\n\n", r.Compute.Note)

	fmt.Fprintf(&b, "## Context KV budget\n\n")
	fmt.Fprintf(&b, "| context tokens | KV ratio vs V3.2 | absolute KV bytes | provenance |\n")
	fmt.Fprintf(&b, "|---:|---:|---:|:--|\n")
	for _, c := range r.Contexts {
		fmt.Fprintf(&b, "| %d | %.2f | %.3g | %s |\n", c.ContextTokens, c.KVRatioVsV32, c.AbsoluteKVBytes, c.Provenance)
	}
	fmt.Fprintf(&b, "\n## Native-support claim\n\n")
	verdict := "REFUSED"
	if r.NativeSupport.Granted {
		verdict = "GRANTED"
	}
	fmt.Fprintf(&b, "- %s claim: **%s** — %s\n", r.NativeSupport.Claim, verdict, r.NativeSupport.Reason)
	if len(r.NativeSupport.MissingWitnesses) > 0 {
		fmt.Fprintf(&b, "- missing witnesses:\n")
		for _, m := range r.NativeSupport.MissingWitnesses {
			fmt.Fprintf(&b, "  - %s\n", m)
		}
	}
	return b.String()
}
