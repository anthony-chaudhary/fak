package modelroute

import (
	"fmt"
	"sort"
	"strings"
)

// This file folds verified cross-audit receipts into failure clusters grouped by
// normalized failure MECHANISM and AUTHOR provenance (provider/family), with
// minimum-sample, base-rate, uncertainty, and confounder guards. It is the
// deterministic, pure spine behind #3855: it produces route-policy *proposals*
// only and never emits a causal, blame, or intent claim. Two structural fences
// enforce that:
//
//   - Rows carry a closed MechanismClass vocabulary, never the auditor's
//     free-text reason. An untrusted auditor cannot smuggle intent-attribution
//     prose into a generated row because the prose is never rendered.
//   - Grouping is keyed on author provenance. When a receipt carries no author
//     provenance, its finding collapses into a single unattributed bucket that
//     is always marked insufficient (NO_PROVENANCE) — stripping provenance
//     structurally prevents any model-specific claim.

const AuditClusterReportSchema = "fak-crossaudit-failure-clusters/v1"

// MechanismClass is the closed vocabulary of failure mechanisms a finding is
// normalized into. It describes HOW a change failed audit, never WHY (no motive)
// and never WHO-is-at-fault (no blame). MechanismUnclassified is the sink for a
// finding whose reason does not match a known mechanism — it is preserved, not
// dropped, so an uncalibrated mechanism never silently disappears.
type MechanismClass string

const (
	MechanismUnclassified         MechanismClass = "unclassified"
	MechanismSilentOmission       MechanismClass = "silent_omission"
	MechanismIncompleteFix        MechanismClass = "incomplete_fix"
	MechanismFabricatedEvidence   MechanismClass = "fabricated_evidence"
	MechanismStaleContext         MechanismClass = "stale_context"
	MechanismScopeViolation       MechanismClass = "scope_violation"
	MechanismRegressionIntroduced MechanismClass = "regression_introduced"
	MechanismUnverifiedClaim      MechanismClass = "unverified_claim"
)

// mechanismKeywords maps reason substrings onto mechanism classes in priority
// order — the FIRST class with a matching needle wins, so the mapping is
// deterministic regardless of how many needles a reason happens to contain.
var mechanismKeywords = []struct {
	class   MechanismClass
	needles []string
}{
	{MechanismFabricatedEvidence, []string{"fabricat", "nonexistent", "hallucinat", "invented", "cited a test that does not", "phantom"}},
	{MechanismSilentOmission, []string{"silently", "silent omission", "omitted", "dropped the", "missing required", "left out"}},
	{MechanismIncompleteFix, []string{"incomplete", "partial fix", "did not fully", "unaddressed", "not all acceptance", "half-implemented"}},
	{MechanismStaleContext, []string{"stale", "overwrote", "clobber", "outdated base", "reverted a peer", "lost a peer"}},
	{MechanismScopeViolation, []string{"out of scope", "out-of-scope", "unrelated change", "scope creep", "touched an unrelated"}},
	{MechanismRegressionIntroduced, []string{"regress", "broke the build", "introduced a failure", "new test failure", "broke a passing"}},
	{MechanismUnverifiedClaim, []string{"unverified", "no witness", "without proof", "overclaim", "claimed done", "unproven"}},
}

// NormalizeMechanism maps a finding reason onto a closed mechanism class. It is a
// deterministic keyword fold; an unrecognized reason returns MechanismUnclassified
// rather than being interpreted or dropped.
func NormalizeMechanism(reason string) MechanismClass {
	r := strings.ToLower(strings.TrimSpace(reason))
	if r == "" {
		return MechanismUnclassified
	}
	for _, m := range mechanismKeywords {
		for _, needle := range m.needles {
			if strings.Contains(r, needle) {
				return m.class
			}
		}
	}
	return MechanismUnclassified
}

// Closed insufficiency-reason vocabulary. A row is a claim ONLY when it carries
// none of these.
const (
	ClusterInsufficientLowSample    = "LOW_SAMPLE"
	ClusterInsufficientLowAudits    = "LOW_AUDITS"
	ClusterInsufficientNoProvenance = "NO_PROVENANCE"
	ClusterInsufficientConfounded   = "CONFOUNDED"
)

// ClusterConfig sets the sample floors below which a group is insufficient to
// support any claim.
type ClusterConfig struct {
	// MinFindings is the minimum finding count in a (mechanism, provenance)
	// group before the group may support a claim.
	MinFindings int `json:"min_findings"`
	// MinAudits is the minimum denominator (total audits of that provenance)
	// before a rate is trustworthy.
	MinAudits int `json:"min_audits"`
}

// DefaultClusterConfig is the conservative floor: at least three findings over
// at least five audits before a provenance group is anything but anecdote.
func DefaultClusterConfig() ClusterConfig { return ClusterConfig{MinFindings: 3, MinAudits: 5} }

func (c ClusterConfig) normalized() ClusterConfig {
	if c.MinFindings <= 0 {
		c.MinFindings = DefaultClusterConfig().MinFindings
	}
	if c.MinAudits <= 0 {
		c.MinAudits = DefaultClusterConfig().MinAudits
	}
	return c
}

// AuditClusterRow is one (mechanism, provider, family) group. Rates are integer
// permille (parts per thousand) to keep the fold float-free and its output
// stable — the same discipline the receipt ledger uses for its hash preimages.
type AuditClusterRow struct {
	Mechanism           MechanismClass `json:"mechanism"`
	Provider            string         `json:"provider,omitempty"`
	Family              string         `json:"family,omitempty"`
	Findings            int            `json:"findings"`
	ProvenanceAudits    int            `json:"provenance_audits"`
	GroupRatePermille   int            `json:"group_rate_permille"`
	BaseFindings        int            `json:"base_findings"`
	BaseAudits          int            `json:"base_audits"`
	BaseRatePermille    int            `json:"base_rate_permille"`
	Harnesses           []string       `json:"harnesses,omitempty"`
	Efforts             []string       `json:"efforts,omitempty"`
	Insufficient        bool           `json:"insufficient"`
	InsufficientReasons []string       `json:"insufficient_reasons,omitempty"`
	Confounded          bool           `json:"confounded"`
	ConfounderNotes     []string       `json:"confounder_notes,omitempty"`
	UncertaintyNotes    []string       `json:"uncertainty_notes,omitempty"`
}

// AuditClusterProposal is a route-policy PROPOSAL — never a mutation, ban, or
// blame. It is emitted only for sufficient, unconfounded rows whose group rate
// exceeds the corpus base rate.
type AuditClusterProposal struct {
	Mechanism MechanismClass `json:"mechanism"`
	Provider  string         `json:"provider"`
	Family    string         `json:"family"`
	Proposal  string         `json:"proposal"`
}

// AuditClusterResult is the deterministic fold output.
type AuditClusterResult struct {
	Schema        string                 `json:"schema"`
	Config        ClusterConfig          `json:"config"`
	TotalReceipts int                    `json:"total_receipts"`
	TotalFindings int                    `json:"total_findings"`
	Rows          []AuditClusterRow      `json:"rows"`
	Proposals     []AuditClusterProposal `json:"proposals"`
}

const clusterKeySep = "\x00"

func clusterProvKey(provider, family string) string { return provider + clusterKeySep + family }

// clusterFindingAgg accumulates one (mechanism, provenance) finding group.
type clusterFindingAgg struct {
	mechanism MechanismClass
	provider  string
	family    string
	findings  int
	harnesses map[string]bool
	efforts   map[string]bool
}

// ClusterAuditFailures folds verified audit receipts into failure clusters.
//
// A receipt's AUTHOR identity is the provenance of the audited change; a REFUTE
// verdict is a finding against that author, normalized to a mechanism class.
// Every receipt (any verdict) counts toward its provenance denominator, so a
// group rate is findings/audits for that provenance and the base rate is the
// same mechanism's findings over ALL audits. Groups below the sample floors, or
// whose author provenance perfectly co-varies with a non-model axis (harness or
// effort), are marked insufficient and cannot support a claim.
func ClusterAuditFailures(receipts []IssueAuditReceipt, cfg ClusterConfig) AuditClusterResult {
	cfg = cfg.normalized()
	result := AuditClusterResult{Schema: AuditClusterReportSchema, Config: cfg}

	provAudits := map[string]int{} // provenance key -> total audits (denominator)
	baseByMech := map[MechanismClass]int{}
	groups := map[string]*clusterFindingAgg{} // (mech + provKey) -> aggregate
	// Per mechanism: which provenance groups used each harness/effort value —
	// the raw material for confounder detection.
	harnessProv := map[MechanismClass]map[string]map[string]bool{}
	effortProv := map[MechanismClass]map[string]map[string]bool{}
	mechProvs := map[MechanismClass]map[string]bool{} // distinct finding provenances per mechanism

	for _, receipt := range receipts {
		author := normalizeAuditIdentityFields(receipt.Author)
		provKey := clusterProvKey(author.Provider, author.Family)
		provAudits[provKey]++
		result.TotalReceipts++

		if receipt.Verdict != CrossAuditRefute {
			continue
		}
		mech := NormalizeMechanism(receipt.Reason)
		baseByMech[mech]++
		result.TotalFindings++

		gk := string(mech) + clusterKeySep + provKey
		agg := groups[gk]
		if agg == nil {
			agg = &clusterFindingAgg{
				mechanism: mech, provider: author.Provider, family: author.Family,
				harnesses: map[string]bool{}, efforts: map[string]bool{},
			}
			groups[gk] = agg
		}
		agg.findings++
		harness := author.Harness
		effort := author.ReasoningPosture
		agg.harnesses[harness] = true
		agg.efforts[effort] = true

		if mechProvs[mech] == nil {
			mechProvs[mech] = map[string]bool{}
			harnessProv[mech] = map[string]map[string]bool{}
			effortProv[mech] = map[string]map[string]bool{}
		}
		mechProvs[mech][provKey] = true
		if harness != "" {
			if harnessProv[mech][harness] == nil {
				harnessProv[mech][harness] = map[string]bool{}
			}
			harnessProv[mech][harness][provKey] = true
		}
		if effort != "" {
			if effortProv[mech][effort] == nil {
				effortProv[mech][effort] = map[string]bool{}
			}
			effortProv[mech][effort][provKey] = true
		}
	}

	for _, agg := range groups {
		provKey := clusterProvKey(agg.provider, agg.family)
		row := AuditClusterRow{
			Mechanism:        agg.mechanism,
			Provider:         agg.provider,
			Family:           agg.family,
			Findings:         agg.findings,
			ProvenanceAudits: provAudits[provKey],
			BaseFindings:     baseByMech[agg.mechanism],
			BaseAudits:       result.TotalReceipts,
			Harnesses:        sortedKnown(agg.harnesses),
			Efforts:          sortedKnown(agg.efforts),
		}
		row.GroupRatePermille = permille(row.Findings, row.ProvenanceAudits)
		row.BaseRatePermille = permille(row.BaseFindings, row.BaseAudits)

		// Uncertainty: an entirely unrecorded confounder axis cannot be ruled
		// out, so note it — but do not force insufficiency, or every group with
		// unrecorded harness/effort would be unusable.
		if len(row.Harnesses) == 0 {
			row.UncertaintyNotes = append(row.UncertaintyNotes, "HARNESS_UNKNOWN: harness not recorded; a harness/tooling confounder cannot be excluded")
		}
		if len(row.Efforts) == 0 {
			row.UncertaintyNotes = append(row.UncertaintyNotes, "EFFORT_UNKNOWN: reasoning effort not recorded; an effort confounder cannot be excluded")
		}

		// Confounder: within this mechanism, if the family's findings all share a
		// single value on a non-model axis that NO other audited author used,
		// the base-model effect is inseparable from that axis. Requires >=2
		// distinct provenances in the mechanism, else there is nothing to
		// confound against.
		if len(mechProvs[agg.mechanism]) >= 2 {
			if note := confoundNote("harness", row.Harnesses, harnessProv[agg.mechanism], provKey); note != "" {
				row.Confounded = true
				row.ConfounderNotes = append(row.ConfounderNotes, note)
			}
			if note := confoundNote("effort", row.Efforts, effortProv[agg.mechanism], provKey); note != "" {
				row.Confounded = true
				row.ConfounderNotes = append(row.ConfounderNotes, note)
			}
		}

		if agg.provider == "" && agg.family == "" {
			row.InsufficientReasons = append(row.InsufficientReasons, ClusterInsufficientNoProvenance)
		}
		if row.Findings < cfg.MinFindings {
			row.InsufficientReasons = append(row.InsufficientReasons, ClusterInsufficientLowSample)
		}
		if row.ProvenanceAudits < cfg.MinAudits {
			row.InsufficientReasons = append(row.InsufficientReasons, ClusterInsufficientLowAudits)
		}
		if row.Confounded {
			row.InsufficientReasons = append(row.InsufficientReasons, ClusterInsufficientConfounded)
		}
		row.Insufficient = len(row.InsufficientReasons) > 0
		result.Rows = append(result.Rows, row)
	}

	sort.Slice(result.Rows, func(i, j int) bool { return lessClusterRow(result.Rows[i], result.Rows[j]) })
	result.Proposals = clusterProposals(result.Rows)
	return result
}

// confoundNote returns a note when the family's findings occupy exactly one
// known value on the axis and that value is unique to this provenance within the
// mechanism (perfect co-variation).
func confoundNote(axis string, values []string, valueProv map[string]map[string]bool, provKey string) string {
	if len(values) != 1 {
		return ""
	}
	v := values[0]
	holders := valueProv[v]
	if len(holders) == 1 && holders[provKey] {
		return fmt.Sprintf("%s_CONFOUND: every finding used %s %q, which no other audited author used for this mechanism; a base-model effect cannot be separated from a %s effect",
			strings.ToUpper(axis), axis, v, axis)
	}
	return ""
}

func clusterProposals(rows []AuditClusterRow) []AuditClusterProposal {
	var out []AuditClusterProposal
	for _, row := range rows {
		if row.Insufficient || row.Confounded {
			continue
		}
		if row.GroupRatePermille <= row.BaseRatePermille {
			continue
		}
		out = append(out, AuditClusterProposal{
			Mechanism: row.Mechanism,
			Provider:  row.Provider,
			Family:    row.Family,
			Proposal: fmt.Sprintf(
				"Provenance %s/%s shows %s at %s vs a %s corpus base across %d audits — consider adding independent-auditor coverage for this provenance on this mechanism. Correlation only; not causation.",
				row.Provider, row.Family, row.Mechanism,
				permilleString(row.GroupRatePermille), permilleString(row.BaseRatePermille), row.ProvenanceAudits),
		})
	}
	return out
}

func lessClusterRow(a, b AuditClusterRow) bool {
	if a.Mechanism != b.Mechanism {
		return a.Mechanism < b.Mechanism
	}
	if a.Provider != b.Provider {
		return a.Provider < b.Provider
	}
	return a.Family < b.Family
}

func permille(num, den int) int {
	if den <= 0 {
		return 0
	}
	return num * 1000 / den
}

// permilleString renders integer permille as a percent with one decimal, e.g.
// 500 -> "50.0%".
func permilleString(p int) string {
	return fmt.Sprintf("%d.%d%%", p/10, p%10)
}

func sortedKnown(set map[string]bool) []string {
	var out []string
	for v := range set {
		if v == "" {
			continue
		}
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

// RenderAuditClusterReport renders the dogfood report section. It leads with an
// explicit correlation-not-causation fence, and renders ONLY closed-vocabulary
// fields (mechanism class, provenance, counts, rates, typed flags) — never the
// auditor's free-text reason — so an intent-attribution phrase in a receipt can
// never reach a generated row.
func RenderAuditClusterReport(result AuditClusterResult) string {
	var sb strings.Builder
	sb.WriteString("## Cross-model failure clustering (correlation, not causation)\n\n")
	sb.WriteString("> Correlation only — not causation. These rows relate audit findings to author\n")
	sb.WriteString("> provenance; a higher group rate can be explained by harness, tooling, task\n")
	sb.WriteString("> mix, or small samples, and such confounders are noted per row. Do not read\n")
	sb.WriteString("> any row as a causal claim, a motive, or grounds to ban or penalize a model.\n")
	sb.WriteString("> Use rows solely to propose independent-auditor routing coverage.\n\n")
	sb.WriteString(fmt.Sprintf("Receipts: %d · findings: %d · sample floor: findings ≥ %d, audits ≥ %d\n\n",
		result.TotalReceipts, result.TotalFindings, result.Config.MinFindings, result.Config.MinAudits))

	var sufficient, insufficient []AuditClusterRow
	for _, row := range result.Rows {
		if row.Insufficient {
			insufficient = append(insufficient, row)
		} else {
			sufficient = append(sufficient, row)
		}
	}

	sb.WriteString("### Sufficient clusters\n\n")
	if len(sufficient) == 0 {
		sb.WriteString("_none — no provenance group cleared the sample and confounder floors._\n\n")
	} else {
		sb.WriteString("| mechanism | provider | family | findings | audits | group rate | base rate | notes |\n")
		sb.WriteString("|---|---|---|---|---|---|---|---|\n")
		for _, row := range sufficient {
			sb.WriteString(fmt.Sprintf("| %s | %s | %s | %d | %d | %s | %s | %s |\n",
				row.Mechanism, provDisplay(row.Provider), provDisplay(row.Family),
				row.Findings, row.ProvenanceAudits,
				permilleString(row.GroupRatePermille), permilleString(row.BaseRatePermille),
				noteCell(row.UncertaintyNotes)))
		}
		sb.WriteString("\n")
	}

	sb.WriteString("### Insufficient / confounded clusters (not a claim)\n\n")
	if len(insufficient) == 0 {
		sb.WriteString("_none._\n\n")
	} else {
		sb.WriteString("| mechanism | provider | family | findings | audits | flags |\n")
		sb.WriteString("|---|---|---|---|---|---|\n")
		for _, row := range insufficient {
			flags := append(append([]string{}, row.InsufficientReasons...), row.ConfounderNotes...)
			flags = append(flags, row.UncertaintyNotes...)
			sb.WriteString(fmt.Sprintf("| %s | %s | %s | %d | %d | %s |\n",
				row.Mechanism, provDisplay(row.Provider), provDisplay(row.Family),
				row.Findings, row.ProvenanceAudits, noteCell(flags)))
		}
		sb.WriteString("\n")
	}

	sb.WriteString("### Route-policy proposals (proposals only — no mutation)\n\n")
	if len(result.Proposals) == 0 {
		sb.WriteString("_none._\n")
	} else {
		for _, p := range result.Proposals {
			sb.WriteString("- " + p.Proposal + "\n")
		}
	}
	return sb.String()
}

func provDisplay(s string) string {
	if strings.TrimSpace(s) == "" {
		return "(unattributed)"
	}
	return s
}

func noteCell(notes []string) string {
	if len(notes) == 0 {
		return "—"
	}
	return strings.Join(notes, "; ")
}

// intentVocabulary is the closed set of intent-attribution WORDS a generated row
// must never carry: motive/blame/deception terms. Matched on whole tokens so
// benign substrings ("believe", "client", "reliable") never false-positive.
var intentVocabularyWords = map[string]bool{
	"malicious": true, "maliciously": true, "malice": true, "deliberate": true, "deliberately": true,
	"intentional": true, "intentionally": true, "sabotage": true, "sabotaged": true,
	"sabotages": true, "sabotaging": true, "lie": true, "lied": true, "lying": true, "deceive": true,
	"deceived": true, "deceptive": true, "deception": true, "dishonest": true, "dishonestly": true,
	"blame": true, "blamed": true, "guilty": true, "willful": true, "wilful": true,
	"evil": true, "nefarious": true, "cheat": true, "cheated": true, "cheating": true,
	"treacherous": true, "conspiracy": true, "conspired": true, "framed": true,
}

// intentVocabularyPhrases are multiword intent-attribution phrases matched as
// substrings.
var intentVocabularyPhrases = []string{
	"on purpose", "bad actor", "to blame", "at fault", "bad faith", "adversarial intent", "acted in bad",
}

// IntentVocabularyViolations returns the sorted, de-duplicated intent-attribution
// terms present in text. Empty means the text is free of intent language. It is
// the reusable linter behind the acceptance gate's prose check.
func IntentVocabularyViolations(text string) []string {
	lower := strings.ToLower(text)
	hits := map[string]bool{}
	var token strings.Builder
	flush := func() {
		if token.Len() == 0 {
			return
		}
		w := token.String()
		token.Reset()
		if intentVocabularyWords[w] {
			hits[w] = true
		}
	}
	for _, r := range lower {
		if (r >= 'a' && r <= 'z') || r == '\'' {
			token.WriteRune(r)
			continue
		}
		flush()
	}
	flush()
	for _, phrase := range intentVocabularyPhrases {
		if strings.Contains(lower, phrase) {
			hits[phrase] = true
		}
	}
	out := make([]string, 0, len(hits))
	for h := range hits {
		out = append(out, h)
	}
	sort.Strings(out)
	return out
}
