package dojo

// claims.go is the dojo's pure CLAIM REGISTRY: the single anchored home for every
// theory number a lever declares. Each (lever, metric) cell carries exactly one
// `claim(...) = <float>` literal, the IntentionalFloor bit that tells a genuine
// estimate apart from a guard the dojo exists to defend, and the prose Basis.
//
// Why a registry and not inline `Claimed: 0.85` composite-literal fields: the
// dojo-RSI loop (docs/fak/dojo-rsi-loop.md) re-points a genuine estimate's claim
// at its corpus central tendency by rewriting ONE anchored literal and proving
// the re-measurement gained — its keep-bit demands `treeChangedOnly(claims.go)`,
// exactly one file and exactly one recalibrated literal. Inline claims scattered
// across the 900-line shell defeat that discipline; one literal per cell here is
// the seam the worktree-rewrite arm rewrites. Pure: no I/O, no dependency on the
// shell, so the registry is unit-testable and the rewrite target is unambiguous.

// Claim is one registered theory cell: the number a lever claims for a metric,
// whether that number is a genuine estimate or an intentional floor, the metric's
// direction, and the prose basis. It mirrors the Prediction fields the builder
// fills from it, so a builder copies a Claim straight into a Prediction.
type Claim struct {
	// Claimed is the theory number — one anchored literal per cell, the only field
	// the RSI loop's RECALIBRATE arm rewrites.
	Claimed float64
	// IntentionalFloor marks a claim that is a guard, not an estimate: a value the
	// dojo asserts reality must NOT breach (false_warm_rate must stay 0.0), as
	// opposed to a best-guess central tendency the loop may recalibrate toward its
	// measured mean. FoldCalibrable folds estimates by calib_err but a floor only
	// by its breach, so closing a floor's gap can never look like a "gain". The bit
	// is the structural reason the loop cannot optimise itself into dishonesty.
	IntentionalFloor bool
	// LowerIsBetter names the metric's direction so the verdict (and a floor's
	// breach side) match the metric's polarity. The default false keeps the
	// higher-is-better scoring most metrics want; set it true for a metric where a
	// lower realized value is the good outcome (false_warm_rate). It carries onto
	// the Prediction unchanged.
	LowerIsBetter bool
	// Basis is the prose justification carried onto the Prediction.
	Basis string
}

// claim is the single anchored-literal constructor every estimate cell uses, so
// each Claimed number appears exactly once and the RSI rewrite target is one
// `claim("lever","metric") = <float>` per cell.
func claim(claimed float64, basis string) Claim {
	return Claim{Claimed: claimed, Basis: basis}
}

// floor is claim's intentional-floor sibling: a guard the dojo defends, not an
// estimate it recalibrates. Identical literal shape so the rewrite anchor is the
// same; the IntentionalFloor bit flips the loop's incentive sign for this cell.
// lowerIsBetter names the breach side — true for an upper-bound floor reality must
// stay BELOW (false_warm_rate), false for a lower-bound default reality may rise
// above harmlessly (the bimodal cross-session warm-hit default).
func floor(claimed float64, lowerIsBetter bool, basis string) Claim {
	return Claim{Claimed: claimed, IntentionalFloor: true, LowerIsBetter: lowerIsBetter, Basis: basis}
}

// claimKey identifies a registry cell. Keeping (lever, metric) the composite key
// matches every consumer (the board, the fold, the candidate picker) which all
// address a cell by that pair.
type claimKey struct {
	Lever  string
	Metric string
}

// ClaimRegistry maps each (lever, metric) cell to its single registered Claim.
// It is the canonical home for every dojo theory number; the cmd/fak builders
// read from it instead of inlining the literal. A cell absent from the registry
// is a programming error surfaced by Lookup's ok=false, never a silent zero.
type ClaimRegistry map[claimKey]Claim

// Registry is the live dojo claim registry — one anchored literal per cell. Every
// cache/compaction/resume number here was lifted verbatim from the inline
// `Claimed:` field it replaced; the pinned-claim tests in cmd/fak/dojo_test.go
// prove the extraction preserved each value. The dispatch-yield (#4497),
// provider-turns (#4505), provider-cache (#4504), provider-toolcall (#4507),
// provider-cost (#4488), provider-tokens (#4503), provider-completion (#4506),
// and cache-read-share (#4498/#4484)
// KPI cells are SEEDED estimates instead — they live in this central literal
// (not the additive RegisterClaim seam) because the RSI recalibrate arm's
// anchored rewriter targets only this file (dojocal.ClaimsRelPath), and those
// cells exist precisely to be recalibrated toward their corpus-measured values.
// false_warm_rate and cross_session_warm_hit_rate are floors (the
// lethal false-warm class and the bimodal 0.0 default the loop must not recalibrate
// up to its empirical rate); every other cell is a genuine estimate.
var Registry = ClaimRegistry{
	{"vdso-ablation", "engine_call_elision"}: claim(1.0,
		"vDSO ON serves every fast-path call locally, eliding it from the engine"),

	{"resume-posture", "posture_accuracy"}: claim(1.0,
		"the resume projection's per-boundary cold/warm posture call assumed correct"),
	{"resume-posture", "cold_write_share"}: claim(0.85,
		"the projection prices ~85% of the resident at the cold-write premium (share = 0.85)"),
	{"resume-posture", "cross_session_warm_hit_rate"}: floor(0.0, false,
		"~0% of large first-turn resumes hit a still-warm cross-session prefix by default; the rate is workload-dependent and bimodal across corpora (0.00→0.65 observed)"),

	{"vcache-warmth", "false_warm_rate"}: floor(0.0, true,
		"the warmth belief never predicts warm on a call the provider bills cache_read=0"),
	{"vcache-warmth", "warm_recall"}: claim(1.0,
		"the warmth belief calls warm every read the provider bills cache_read>0"),

	{"compaction", "token_shed_ratio"}: claim(1.0,
		"the projected shed (WITNESSED shed_tokens) matches the billed input_tokens delta (OFF - ON)"),
	{"compaction", "cache_prefix_preserved"}: claim(1.0,
		"a fired compaction ships the protected prefix byte-identical"),

	{"dispatch-yield", "verified_ship_rate"}: claim(0.5,
		"seed theory (#4497): about half of dispatched workers reconcile as a diff-witnessed VERIFIED close over the loop-ledger window; a genuine estimate the RSI loop recalibrates toward the measured spawn-to-close yield"),

	{"provider-turns", "turns_per_task"}: claim(20.0,
		"seed theory (#4505): a provider completes a task in about twenty assistant turns — the median turns per completed session in the multi-provider session corpus; a genuine estimate the RSI loop recalibrates toward the measured medians, and the per-provider spread of the same cell is the cross-provider leaderboard"),

	{"provider-cache", "cache_read_share"}: claim(0.8,
		"seed theory (#4504): ~80% of the input tokens a provider bills on a warm agentic session corpus (input + cache_read + cache_creation) are served as cache reads; a genuine estimate the RSI loop recalibrates toward the measured shares, and the per-provider spread of the same cell is the cross-provider cache-economy leaderboard"),

	{"provider-toolcall", "tool_call_success_rate"}: claim(0.9,
		"seed theory (#4507): ~90% of tool calls on an agentic session corpus succeed on the first try (non-errored tool_result / total tool_result); a genuine estimate the RSI loop recalibrates toward the measured rates, and the per-provider spread of the same cell is the cross-provider tool-reliability leaderboard"),

	{"cache-read-share", "billed_cache_read_share"}: claim(0.8,
		"seed theory (#4498/#4484): ~80% of the billed input-side tokens across the WHOLE multi-provider session corpus (input + cache_read + cache_creation) are served as cache reads — the WITNESSED top-line cache-read fraction folded across ALL providers into one headline number, a genuine estimate the RSI loop recalibrates toward the measured share. Distinct from the per-provider provider-cache/cache_read_share leaderboard (#4504): same billed tokens, aggregated corpus-wide into a single cell instead of split per provider, so it is the one number an operator cites for cache economy — not a double-count of provider-cache but its top-line aggregate"),

	{"provider-cost", "cost_per_completed_issue"}: claim(3.0,
		"seed theory (#4488): a provider spends about three billed US dollars per completed issue — the mean billed USD per completed session (the corpus proxy for a verified close) across the multi-provider session corpus, priced by the existing sessionaudit per-model table (not a new pricing table); a genuine estimate the RSI loop recalibrates toward the measured per-provider means, and the per-provider spread of the same cell is the cross-provider cost-to-close economics leaderboard. A provider whose completed sessions carry no priced billing scores UNMEASURED, never a fabricated $0.00"),

	{"provider-tokens", "tokens_per_completed_issue"}: claim(1000000.0,
		"seed theory (#4503): a provider bills about one million total tokens (input + output + cache_read + cache_creation) per completed issue — the mean total billed tokens per completed session (the corpus proxy for a verified close) across the multi-provider session corpus; a genuine estimate the RSI loop recalibrates toward the measured per-provider means, and the per-provider spread of the same cell is the cross-provider tokens-to-close leaderboard. This is TOTAL billed tokens keyed by provider (not USD cost, not output-only tokens); a provider whose completed sessions carry no billed tokens scores UNMEASURED, never a fabricated 0"),

	{"provider-completion", "verified_completion_rate"}: claim(0.5,
		"seed theory (#4506): about half of the sessions a provider is dispatched reconcile as a verified close (a completed, non-interrupted session — the corpus proxy for a closed task) over the multi-provider session corpus; a genuine estimate the RSI loop recalibrates toward the measured per-provider rates, and the per-provider spread of the same cell is the cross-provider verified-completion-rate leaderboard. This is verified closes / dispatched keyed BY PROVIDER — the per-provider analog of the fak-aggregate dispatch-yield/verified_ship_rate cell (#4497), seeded at the same 0.5, not that cell; a provider with no dispatched sessions in the corpus scores UNMEASURED, never a fabricated rate"),
}

// registered is the additive claim seam: the composed home for cells a KPI leaf
// declares in its OWN file via RegisterClaim, kept separate from the Registry
// literal so adding a cell never edits — and never conflicts on — that central
// literal. Lookup folds it in behind the receiver, so Registry.Lookup/Predict
// resolve an additively registered cell while len(Registry) (the extraction-fidelity
// witness pinned in claims_test.go) stays the count of cells lifted from the inline
// builders. One anchored literal per cell still holds: each RegisterClaim call
// carries exactly one claim(...)/floor(...) literal, the single RSI rewrite target
// for that cell — now in the cell's own file instead of this shared map.
var registered = ClaimRegistry{}

// RegisterClaim adds one (lever, metric) cell to the additive seam so a KPI cell
// lands in its own file — a plain package-level `var _ = RegisterClaim(...)` — with
// no edit to the central Registry literal, letting parallel KPI-cell workers avoid
// colliding on one file and one map. It panics on a duplicate (the same cell already
// present in either Registry or the additive seam): a doubly-registered cell is a
// programming error, surfaced loudly at init rather than silently shadowing a claim.
// It returns the Claim so it composes as a var initializer. The one anchored literal
// for the cell is the Claim the caller passes (built with claim(...)/floor(...)).
func RegisterClaim(lever, metric string, c Claim) Claim {
	key := claimKey{Lever: lever, Metric: metric}
	if _, taken := Registry[key]; taken {
		panic("dojo: RegisterClaim on a cell already in the central Registry: " + lever + "/" + metric)
	}
	if _, taken := registered[key]; taken {
		panic("dojo: RegisterClaim on an already-registered cell: " + lever + "/" + metric)
	}
	registered[key] = c
	return c
}

// Lookup returns the registered Claim for a (lever, metric) cell. It resolves the
// receiver's own cells first, then folds in the additive seam (cells declared in
// their own files via RegisterClaim) so a KPI leaf resolves without ever editing the
// central Registry literal. ok is false for an unregistered cell so a builder fails
// loud (a missing registry entry is a programming error) rather than scoring against
// a silent zero claim.
func (r ClaimRegistry) Lookup(lever, metric string) (Claim, bool) {
	key := claimKey{Lever: lever, Metric: metric}
	if c, ok := r[key]; ok {
		return c, true
	}
	if c, ok := registered[key]; ok {
		return c, true
	}
	return Claim{}, false
}

// Predict builds the Prediction for a (lever, metric) cell straight from the
// registry, so a builder declares only the cell + its unit and the Claimed /
// IntentionalFloor / LowerIsBetter / Basis come from the one anchored literal. The
// bool ok mirrors Lookup: an unregistered cell yields a zero Prediction and
// ok=false, never a silent zero claim.
func (r ClaimRegistry) Predict(lever, metric, unit string) (Prediction, bool) {
	c, ok := r.Lookup(lever, metric)
	if !ok {
		return Prediction{}, false
	}
	return Prediction{
		Lever:            lever,
		Metric:           metric,
		Claimed:          c.Claimed,
		Unit:             unit,
		Basis:            c.Basis,
		LowerIsBetter:    c.LowerIsBetter,
		IntentionalFloor: c.IntentionalFloor,
	}, true
}

// MustPredict is Predict for the in-tree builders, where an unregistered cell is a
// programming error (the registry and the builders are edited together). It panics
// on a missing cell so a typo surfaces at the first call, not as a mis-scored
// episode. cmd/fak builders use this; external callers use Predict and handle ok.
func (r ClaimRegistry) MustPredict(lever, metric, unit string) Prediction {
	p, ok := r.Predict(lever, metric, unit)
	if !ok {
		panic("dojo: no registered claim for cell " + lever + "/" + metric)
	}
	return p
}
