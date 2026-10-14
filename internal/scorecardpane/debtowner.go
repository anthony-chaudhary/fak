package scorecardpane

// debtowner.go — the scorecard debt-key OWNERSHIP registry (#1974, QA-dogfood
// spine #1961/QD-014).
//
// A control-pane row names a debt key and a raw magnitude, but nothing in the
// folded payload said WHO owns retiring that debt: the debt key describes the
// symptom while the owning package leaf, lane, and the pathspec a worker should
// touch stayed implicit. That ambiguity is what makes a "retire the regressed
// metric worst-first" next_action un-actionable: the operator has a number and a
// label but no destination.
//
// This is that destination as DATA: one row per debt key mapping it to its owning
// package leaf, its declared dos.toml lane label, and the repo-relative pathspec
// the fix lands in. It is deliberately a static, auditable registry rather than a
// derived one: the mapping is a human decision (which leaf owns a scorecard's
// debt), and pinning it in one table keeps it reviewable and lets a test prove
// every folded card has an owner and every lane it names is really declared.
//
// The registry is keyed on the DEBT KEY (what actually sums into total_debt and
// pins per-key in the baseline), not the cosmetic card key, matching the parity
// gate's own keying (paneparity_test.go).

// DebtOwner is the owning destination of one scorecard debt key: the package leaf
// that owns the detector, the dos.toml lane label a worker arbitrates on, and the
// repo-relative pathspec the fix lands in. It is emitted on a control-pane metric
// (see Metric.OwnerLeaf/Lane/Pathspec) so a regressed row carries its own worklist
// target instead of a naked number.
type DebtOwner struct {
	// Leaf is the owning package leaf: the internal/<leaf> package for a Go-backed
	// card, or the coarse surface ("tools", "cmd", "docs") for a Python or inline
	// detector that has no dedicated internal leaf.
	Leaf string `json:"leaf"`
	// Lane is the declared dos.toml lane label a worker arbitrates against. It is
	// the lane the owning leaf is registered under, never a free-text guess.
	Lane string `json:"lane"`
	// Pathspec is the repo-relative glob (or comma-joined globs) the owner's fix
	// lands in — the destination the next_action points at.
	Pathspec string `json:"pathspec"`
}

// DebtOwners maps a scorecard debt key to its owning leaf, lane, and pathspec.
// Every debt key folded by Cards (or carried in DebtCategories) MUST appear here;
// TestDebtOwnerRegistryCoversEveryCard makes that a gate, and
// TestDebtOwnerLanesAreDeclared proves each Lane is a real dos.toml roster entry.
var DebtOwners = map[string]DebtOwner{
	// --- Python tools/ detectors: leaf "tools", lane "tools" ---------------------
	"doc_debt":            {Leaf: "tools", Lane: "tools", Pathspec: "tools/docs_scorecard.py,docs/**"},
	"readme_debt":         {Leaf: "tools", Lane: "tools", Pathspec: "tools/readme_freshness_audit.py,README.md"},
	"code_debt":           {Leaf: "tools", Lane: "tools", Pathspec: "tools/code_quality_scorecard.py"},
	"appeal_debt":         {Leaf: "tools", Lane: "tools", Pathspec: "tools/doc_appeal_scorecard.py"},
	"demo_debt":           {Leaf: "tools", Lane: "tools", Pathspec: "tools/demo_quality_scorecard.py"},
	"robustness_debt":     {Leaf: "tools", Lane: "tools", Pathspec: "tools/demo_robustness_scorecard.py"},
	"hygiene_debt":        {Leaf: "tools", Lane: "tools", Pathspec: "tools/repo_hygiene_scorecard.py"},
	"parity_debt":         {Leaf: "tools", Lane: "tools", Pathspec: "tools/industry_scorecard.py"},
	"persona_debt":        {Leaf: "tools", Lane: "tools", Pathspec: "tools/persona_readiness_scorecard.py"},
	"popularization_debt": {Leaf: "tools", Lane: "tools", Pathspec: "tools/popularization_readiness_scorecard.py"},
	"stability_debt":      {Leaf: "tools", Lane: "tools", Pathspec: "tools/stability_scorecard.py"},
	"slop_debt":           {Leaf: "tools", Lane: "tools", Pathspec: "tools/code_slop_scorecard.py"},
	"steerability_debt":   {Leaf: "tools", Lane: "tools", Pathspec: "tools/steerability_scorecard.py"},
	"disambiguation_debt": {Leaf: "tools", Lane: "tools", Pathspec: "tools/concept_disambiguation_scorecard.py"},
	"intent_literal_debt": {Leaf: "tools", Lane: "tools", Pathspec: "tools/intent_literal_scorecard.py"},
	"claim_repro_debt":    {Leaf: "tools", Lane: "tools", Pathspec: "tools/claim_repro_scorecard.py"},
	"release_debt":        {Leaf: "tools", Lane: "tools", Pathspec: "tools/release_readiness_scorecard.py"},
	"observability_debt":  {Leaf: "tools", Lane: "tools", Pathspec: "tools/observability_scorecard.py"},
	"learning_debt":       {Leaf: "tools", Lane: "tools", Pathspec: "tools/learning_scorecard.py"},
	"rsi_debt":            {Leaf: "tools", Lane: "tools", Pathspec: "tools/rsi_maturity_scorecard.py"},
	"py_debt":             {Leaf: "tools", Lane: "tools", Pathspec: "tools/tooling_quality_scorecard.py"},
	"bench_dx_debt":       {Leaf: "tools", Lane: "tools", Pathspec: "tools/bench_dx_scorecard.py"},
	"process_debt":        {Leaf: "tools", Lane: "tools", Pathspec: "tools/cuda_dev_scorecard.py"},
	"persona_fit_debt":    {Leaf: "tools", Lane: "tools", Pathspec: "tools/persona_fit_scorecard.py"},
	"commit_debt":         {Leaf: "tools", Lane: "tools", Pathspec: "tools/commit_subject_coverage.py"},

	// --- Go-backed detectors: leaf = the owning internal package -----------------
	"seo_debt":               {Leaf: "seoaeoscore", Lane: "seoaeoscore", Pathspec: "internal/seoaeoscore/**"},
	"friction_debt":          {Leaf: "agentreadinessscore", Lane: "agentreadinessscore", Pathspec: "internal/agentreadinessscore/**"},
	"product_debt":           {Leaf: "productscorecard", Lane: "productscorecard", Pathspec: "internal/productscorecard/**"},
	"brittleness_debt":       {Leaf: "brittleness", Lane: "brittleness", Pathspec: "internal/brittleness/**"},
	"conflation_debt":        {Leaf: "conflationscore", Lane: "conflationscore", Pathspec: "internal/conflationscore/**"},
	"ui_quality_debt":        {Leaf: "uiquality", Lane: "uiquality", Pathspec: "internal/uiquality/**"},
	"guard_rsi_debt":         {Leaf: "guardrsi", Lane: "guardrsi", Pathspec: "internal/guardrsi/**"},
	"guard_accuracy_debt":    {Leaf: "guardaccuracy", Lane: "guardaccuracy", Pathspec: "internal/guardaccuracy/**"},
	"dogfood_debt":           {Leaf: "dogfoodscore", Lane: "dogfoodscore", Pathspec: "internal/dogfoodscore/**"},
	"conceptusage_debt":      {Leaf: "conceptusage", Lane: "conceptusage", Pathspec: "internal/conceptusage/**"},
	"lightgap_debt":          {Leaf: "lightgapscore", Lane: "lightgapscore", Pathspec: "internal/lightgapscore/**"},
	"maturity_debt":          {Leaf: "maturity", Lane: "maturity", Pathspec: "internal/maturity/**"},
	"growth_debt":            {Leaf: "covmatrix", Lane: "covmatrix", Pathspec: "internal/covmatrix/**"},
	"support_maturity_debt":  {Leaf: "supportmaturityscore", Lane: "supportmaturityscore", Pathspec: "internal/supportmaturityscore/**"},
	"milestone_debt":         {Leaf: "milestonereport", Lane: "milestonereport", Pathspec: "internal/milestonereport/**"},
	"climb_ratchet_debt":     {Leaf: "milestonereport", Lane: "milestonereport", Pathspec: "internal/milestonereport/**"},
	"loopindex_debt":         {Leaf: "loopindex", Lane: "loopindex", Pathspec: "internal/loopindex/**"},
	"heaviness_debt":         {Leaf: "heavinessscore", Lane: "heavinessscore", Pathspec: "internal/heavinessscore/**"},
	"propagation_debt":       {Leaf: "propagationscore", Lane: "propagationscore", Pathspec: "internal/propagationscore/**"},
	"sota_debt":              {Leaf: "sotacoverage", Lane: "sotacoverage", Pathspec: "internal/sotacoverage/**"},
	"flow_debt":              {Leaf: "flowmetrics", Lane: "flowmetrics", Pathspec: "internal/flowmetrics/**"},
	"antipattern_debt":       {Leaf: "antipattern", Lane: "antipattern", Pathspec: "internal/antipattern/**"},
	"negframe_debt":          {Leaf: "negframe", Lane: "negframe", Pathspec: "internal/negframe/**"},
	"negation_tax_debt":      {Leaf: "negframe", Lane: "negframe", Pathspec: "internal/negframe/**"},
	"negation_operator_debt": {Leaf: "negframe", Lane: "negframe", Pathspec: "internal/negframe/**"},
	"residual_count":         {Leaf: "steerpr", Lane: "steerpr", Pathspec: "internal/steerpr/**"},
	"token_defaults_debt":    {Leaf: "cmd", Lane: "cmd", Pathspec: "cmd/fak/token_defaults.go"},
}

// OwnerForDebt returns the registered DebtOwner for a scorecard debt key. The
// bool is false for a debt key with no registry row, so a caller never invents an
// owner: an unmapped key is an honest "unknown", not a silent default.
func OwnerForDebt(debtKey string) (DebtOwner, bool) {
	o, ok := DebtOwners[debtKey]
	return o, ok
}
