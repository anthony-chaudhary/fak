// contamination.go is the benchmark-contamination risk audit (issue #4571, one
// leaf of the #4509 "missing middle" quality ladder). It answers a question the
// coarse end-benchmark score cannot: is this quality corpus even *entitled* to a
// confirmatory claim, or is the score inflated by cases the model already saw in
// training, by duplicate cases counted twice, or by cases whose provenance is too
// thin to trust?
//
// The lm-evaluation-harness / HELM lineage the issue cites treats contamination
// and provenance as a first-class gate: a benchmark number is only meaningful once
// duplicates are deduped, exposed-to-training cases are held out, and every case
// carries enough provenance (model, tokenizer, engine, seed/oracle, code revision,
// tolerance/baseline) to be independently replayed. This audit is that gate for the
// fak quality ladder, fail-closed on the repo's net-true doctrine: MISSING OR
// INCONCLUSIVE EVIDENCE IS NEVER A PASS. A case that cannot prove it is clean is
// excluded from the confirmatory claim set rather than silently counted.
//
// It operates only on content HASHES plus metadata — never raw prompt/expected
// text — so the emitted replay artifact is already scrubbed: publishing it cannot
// itself leak (and thereby further contaminate) the eval. Each excluded case names
// the FIRST actionable divergence (the first failing check in a fixed order:
// provenance completeness -> dedupe -> pre-cutoff exposure), so an operator gets a
// single next step, not a pile of symptoms.
//
// Every case is assigned an explicit run tier (pr | nightly | release) and carries
// its runtime/resource cost, so the report also documents what each tier costs to
// run — the operator-facing "where does this case belong and what does it cost"
// the issue's acceptance criterion names.
//
// The report is deterministic (no clock, no map iteration in output), so it is a
// re-derivable, independently replayable witness:
//
//	go test ./internal/bench -run TestContaminationAudit -count=1
//
// (the golden artifact regenerates into testdata/contamination_report.json with
// UPDATE_GOLDEN=1).
package bench

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ContaminationCase is one benchmark case plus the provenance the audit needs to
// decide whether it may back a confirmatory quality claim. It carries a CONTENT
// HASH, never the raw prompt/expected text: the audit — and every artifact it
// emits — stays scrubbed by construction.
type ContaminationCase struct {
	// ID identifies the case; Suite is the benchmark suite it belongs to.
	ID    string `json:"id"`
	Suite string `json:"suite"`
	// ContentHash is the dedupe key: a hash of the (prompt, expected) pair. Two
	// cases with the same hash are the same case counted twice.
	ContentHash string `json:"content_hash"`

	// The per-case provenance acceptance criterion #2 names, one field each.
	Model        string `json:"model"`               // model under test
	Tokenizer    string `json:"tokenizer"`           // tokenizer/vocab
	Engine       string `json:"engine"`              // engine/backend
	SeedOrOracle string `json:"seed_or_oracle"`      // a seed value or a named deterministic oracle
	CodeRevision string `json:"code_revision"`       // code/module revision the case ran against
	Baseline     string `json:"baseline_provenance"` // tolerance + baseline provenance

	// Exposure evidence. PublishedAt is the ISO-8601 date the case content was
	// first publicly exposed; TrainCutoff is the model's training-data cutoff. A
	// case published on/before the cutoff and not held out may be in the training
	// set. Holdout marks a fresh private case never exposed to training, which does
	// not need a publish date to be judged clean.
	PublishedAt string `json:"published_at,omitempty"`
	TrainCutoff string `json:"train_cutoff,omitempty"`
	Holdout     bool   `json:"holdout"`

	// Tier assigns the case to an explicit run tier; RuntimeCostMS documents its
	// runtime/resource cost.
	Tier          string `json:"tier"`
	RuntimeCostMS int    `json:"runtime_cost_ms"`
}

// Per-case audit outcomes. Exactly one is assigned; the three excluded_* statuses
// keep the case OUT of the confirmatory claim set.
const (
	StatusAdmitted     = "admitted"              // complete provenance, unique, no pre-cutoff exposure
	StatusIncomplete   = "excluded_incomplete"   // inconclusive evidence — never pass (the fail-closed rule)
	StatusDuplicate    = "excluded_duplicate"    // same content hash as an earlier case
	StatusContaminated = "excluded_contaminated" // exposed to training (published on/before the model cutoff, not holdout)
)

// validTiers is the closed vocabulary of run tiers, in escalating cost order; the
// report documents cost per tier in this order.
var validTiers = []string{"pr", "nightly", "release"}

func validTier(t string) bool {
	for _, v := range validTiers {
		if t == v {
			return true
		}
	}
	return false
}

// firstMissingField returns the first missing/invalid provenance field in the
// order acceptance criterion #2 lists them, or "" when the case is fully evidenced.
// An unknown publish date (for a non-holdout case) is INCONCLUSIVE exposure
// evidence and therefore counts as missing — the audit will not certify a case it
// cannot prove was published after the cutoff.
func firstMissingField(c ContaminationCase) string {
	switch {
	case strings.TrimSpace(c.ID) == "":
		return "id"
	case strings.TrimSpace(c.Model) == "":
		return "model"
	case strings.TrimSpace(c.Tokenizer) == "":
		return "tokenizer"
	case strings.TrimSpace(c.Engine) == "":
		return "engine/backend"
	case strings.TrimSpace(c.SeedOrOracle) == "":
		return "seed_or_deterministic_oracle"
	case strings.TrimSpace(c.CodeRevision) == "":
		return "code_revision"
	case strings.TrimSpace(c.Baseline) == "":
		return "tolerance_baseline_provenance"
	case strings.TrimSpace(c.ContentHash) == "":
		return "content_hash"
	case !validTier(c.Tier):
		return "tier (want pr|nightly|release)"
	case c.RuntimeCostMS <= 0:
		return "runtime_cost_ms"
	case !c.Holdout && strings.TrimSpace(c.PublishedAt) == "":
		return "published_at (or mark holdout)"
	case !c.Holdout && strings.TrimSpace(c.TrainCutoff) == "":
		return "train_cutoff (or mark holdout)"
	}
	return ""
}

// classify decides one case's outcome and its first actionable divergence, in the
// fixed check order provenance-completeness -> dedupe -> exposure. firstByHash maps
// a content hash to the ID of the FIRST case that carried it (a prior case only —
// the caller registers the current case after classifying, so a case is never a
// duplicate of itself).
func classify(c ContaminationCase, firstByHash map[string]string) (status, divergence, dupeOf string) {
	if miss := firstMissingField(c); miss != "" {
		return StatusIncomplete, "missing/inconclusive provenance: " + miss, ""
	}
	if canon, ok := firstByHash[c.ContentHash]; ok {
		return StatusDuplicate, fmt.Sprintf("duplicate of case %q (content hash %s)", canon, c.ContentHash), canon
	}
	if !c.Holdout && c.PublishedAt <= c.TrainCutoff {
		return StatusContaminated,
			fmt.Sprintf("exposed: published %s on/before train cutoff %s and not a fresh holdout", c.PublishedAt, c.TrainCutoff),
			""
	}
	return StatusAdmitted, "", ""
}

// ReplayArtifact is the scrubbed, independently-replayable record of one case's
// audit: the content hash and provenance needed to re-run it, its tier and cost,
// its status, and — when excluded — the first actionable divergence. It never
// carries raw eval content, so it is safe to publish.
type ReplayArtifact struct {
	Case          string `json:"case"`
	Suite         string `json:"suite"`
	Status        string `json:"status"`
	Divergence    string `json:"first_divergence,omitempty"`
	DuplicateOf   string `json:"duplicate_of,omitempty"`
	ContentHash   string `json:"content_hash"`
	Model         string `json:"model"`
	Tokenizer     string `json:"tokenizer"`
	Engine        string `json:"engine"`
	SeedOrOracle  string `json:"seed_or_oracle"`
	CodeRevision  string `json:"code_revision"`
	Baseline      string `json:"baseline_provenance"`
	Tier          string `json:"tier"`
	RuntimeCostMS int    `json:"runtime_cost_ms"`
}

func artifactFor(c ContaminationCase, status, divergence, dupeOf string) ReplayArtifact {
	return ReplayArtifact{
		Case:          c.ID,
		Suite:         c.Suite,
		Status:        status,
		Divergence:    divergence,
		DuplicateOf:   dupeOf,
		ContentHash:   c.ContentHash,
		Model:         c.Model,
		Tokenizer:     c.Tokenizer,
		Engine:        c.Engine,
		SeedOrOracle:  c.SeedOrOracle,
		CodeRevision:  c.CodeRevision,
		Baseline:      c.Baseline,
		Tier:          c.Tier,
		RuntimeCostMS: c.RuntimeCostMS,
	}
}

// TierCost documents how many cases and how much runtime cost a run tier carries —
// the acceptance criterion "assign the case to an explicit PR/nightly/release tier
// and document runtime/resource cost."
type TierCost struct {
	Tier          string `json:"tier"`
	Cases         int    `json:"cases"`
	RuntimeCostMS int    `json:"runtime_cost_ms"`
}

// Verdicts the audit can reach.
const (
	// VerdictContaminationClean: every case carries complete provenance, is unique,
	// and shows no pre-cutoff exposure — the whole corpus may back a confirmatory
	// claim.
	VerdictContaminationClean = "contamination_clean"
	// VerdictContaminationRisk: one or more cases were flagged and excluded from the
	// confirmatory claim set. A real finding: the audit did its job.
	VerdictContaminationRisk = "contamination_risk_flagged"
)

// ContaminationReport is the full audit: which cases may back a confirmatory claim,
// which were excluded and why, the per-tier cost, and the verdict.
type ContaminationReport struct {
	Schema     string     `json:"schema"`
	Provenance Provenance `json:"provenance"`
	Cases      int        `json:"cases"`
	// ConfirmatoryEligible lists the admitted case IDs, in input order — the ONLY
	// cases a confirmatory quality claim may be computed over.
	ConfirmatoryEligible []string         `json:"confirmatory_eligible"`
	Admitted             []ReplayArtifact `json:"admitted"`
	Excluded             []ReplayArtifact `json:"excluded"`
	TierCosts            []TierCost       `json:"tier_costs"`
	Verdict              string           `json:"verdict"`
	Finding              string           `json:"finding"`
}

// AuditContamination runs the audit over a corpus and folds the per-case outcomes
// into the report. Cases are processed in input order; the report is deterministic.
func AuditContamination(cases []ContaminationCase) ContaminationReport {
	return auditContamination(cases, Provenance{
		Kind:        ProvenanceSimulated,
		Command:     "go test ./internal/bench -run TestContaminationAudit -count=1",
		GeneratedBy: "fak/internal/bench.AuditContamination",
		Note: "Corpus is a labeled fixture of content HASHES + provenance (never raw eval text): the audit " +
			"witnesses the dedupe/exposure/provenance gate and the scrubbed replay artifact. Real corpora feed " +
			"the same report shape by supplying observed hashes, publish dates, and per-case run cost.",
	})
}

func auditContamination(cases []ContaminationCase, provenance Provenance) ContaminationReport {
	firstByHash := map[string]string{}
	admitted := make([]ReplayArtifact, 0, len(cases))
	excluded := make([]ReplayArtifact, 0)
	eligible := make([]string, 0, len(cases))

	type acc struct {
		cases int
		cost  int
	}
	tierAcc := map[string]*acc{}
	var extraTiers []string

	var nDup, nContam, nIncomplete int
	for _, c := range cases {
		status, divergence, dupeOf := classify(c, firstByHash)
		if strings.TrimSpace(c.ContentHash) != "" {
			if _, seen := firstByHash[c.ContentHash]; !seen {
				firstByHash[c.ContentHash] = c.ID
			}
		}

		art := artifactFor(c, status, divergence, dupeOf)
		if status == StatusAdmitted {
			admitted = append(admitted, art)
			eligible = append(eligible, c.ID)
		} else {
			excluded = append(excluded, art)
			switch status {
			case StatusDuplicate:
				nDup++
			case StatusContaminated:
				nContam++
			case StatusIncomplete:
				nIncomplete++
			}
		}

		// Cost accounting per tier — count every case, even excluded ones, since a
		// case that was RUN still cost its resources whether or not it was admitted.
		tier := c.Tier
		if !validTier(tier) {
			tier = "unassigned"
		}
		a := tierAcc[tier]
		if a == nil {
			a = &acc{}
			tierAcc[tier] = a
			if !validTier(tier) {
				extraTiers = append(extraTiers, tier)
			}
		}
		a.cases++
		a.cost += c.RuntimeCostMS
	}

	// Emit tier costs deterministically: the known tiers in escalating-cost order,
	// then any others (e.g. "unassigned") sorted, and only tiers that had cases.
	sort.Strings(extraTiers)
	tierCosts := make([]TierCost, 0, len(tierAcc))
	for _, t := range append(append([]string{}, validTiers...), extraTiers...) {
		if a := tierAcc[t]; a != nil {
			tierCosts = append(tierCosts, TierCost{Tier: t, Cases: a.cases, RuntimeCostMS: a.cost})
		}
	}

	verdict := VerdictContaminationClean
	if len(excluded) > 0 {
		verdict = VerdictContaminationRisk
	}

	return ContaminationReport{
		Schema:               "contamination.v1",
		Provenance:           provenance,
		Cases:                len(cases),
		ConfirmatoryEligible: eligible,
		Admitted:             admitted,
		Excluded:             excluded,
		TierCosts:            tierCosts,
		Verdict:              verdict,
		Finding:              contaminationFinding(len(cases), len(eligible), nDup, nContam, nIncomplete),
	}
}

func contaminationFinding(total, eligible, nDup, nContam, nIncomplete int) string {
	flagged := nDup + nContam + nIncomplete
	if flagged == 0 {
		return fmt.Sprintf(
			"all %d case(s) carry complete provenance, are unique, and show no pre-cutoff exposure: "+
				"the confirmatory claim set is the full corpus.",
			total)
	}
	return fmt.Sprintf(
		"flagged %d of %d case(s) — %d duplicate, %d contaminated (pre-cutoff exposure), %d incomplete "+
			"(inconclusive evidence, never pass); excluded from the confirmatory claim set, leaving %d eligible case(s).",
		flagged, total, nDup, nContam, nIncomplete, eligible)
}

// DefaultContaminationCorpus is the fixed, clean reference corpus: a small set of
// cases spanning the three run tiers, each fully evidenced, unique, and either
// published after the model cutoff or a fresh private holdout. It is the "passes
// after the fix" half of the witness; the failing half plants duplicate,
// contaminated, and incomplete cases on top of it.
func DefaultContaminationCorpus() []ContaminationCase {
	return []ContaminationCase{
		{
			ID: "decode-parity-ascii", Suite: "decode",
			ContentHash: "sha256:decode-01", Model: "fak-decode-ref", Tokenizer: "bpe-v3",
			Engine: "cpu-q8", SeedOrOracle: "seed:42", CodeRevision: "rev-abc123",
			Baseline:    "tol=1e-6 vs ref-golden@rev-abc123",
			PublishedAt: "2025-03-01", TrainCutoff: "2024-10-01", Holdout: false,
			Tier: "pr", RuntimeCostMS: 120,
		},
		{
			ID: "sampling-ci-holdout", Suite: "sampling",
			ContentHash: "sha256:sample-02", Model: "fak-decode-ref", Tokenizer: "bpe-v3",
			Engine: "cpu-q8", SeedOrOracle: "oracle:closed-form-binomial", CodeRevision: "rev-abc123",
			Baseline: "CI=[0.48,0.52] vs analytic mean", Holdout: true, // fresh private holdout: no publish date needed
			Tier: "nightly", RuntimeCostMS: 800,
		},
		{
			ID: "engine-parity-attn", Suite: "parity",
			ContentHash: "sha256:parity-03", Model: "fak-decode-ref", Tokenizer: "bpe-v3",
			Engine: "cuda-fp16", SeedOrOracle: "seed:7", CodeRevision: "rev-abc123",
			Baseline:    "max-abs-logit-delta < 5e-3 vs cpu-q8 baseline@rev-abc123",
			PublishedAt: "2025-06-01", TrainCutoff: "2024-10-01", Holdout: false,
			Tier: "release", RuntimeCostMS: 5000,
		},
	}
}

// JSON renders the report as stable, indented JSON (no clock, no map iteration), so
// it is a re-derivable, independently replayable witness artifact.
func (r ContaminationReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}
