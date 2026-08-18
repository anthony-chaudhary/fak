package rungobs

// cost_ledger.go is the validation-cost / marginal-defect-yield child of the
// quality spine (#4586, sibling of the loader-parity child #4545 in
// internal/modelengine): it proves that a fixture "month" of validation
// suite-runs folds into a cost-and-defect ledger that attributes compute,
// generation- and judge-token spend, wall time, flakes, and UNIQUE defects to
// each suite WITHOUT double-counting a defect that several suites caught. The
// canonical fold is the reference; an alternate fold is the engine. A faithful
// fold is invisible (row-identical regardless of replay order); a double-count,
// a dropped cost component, or a mis-tiered suite surfaces as the FIRST divergent
// ledger row, localized to its exact suite, and is refused before the ledger is
// published.
//
// The oracle is deterministic and self-contained (no clock, no network, no
// randomness): the fixture month is a pinned list of suite-runs, each carrying a
// cost vector and the set of defect IDs it caught. A defect is attributed to
// exactly ONE suite — the cheapest tier that caught it (PR < nightly < release),
// ties broken by suite name — so the summed per-suite unique-defect counts equal
// the number of DISTINCT defects in the month. That equality is the "without
// double count" acceptance criterion, checkable independently of the row compare.
// Runtime/resource cost: pure in-process, microseconds per case, no fixtures on
// disk. Tier: PR (runs in the package unit gate).
//
// Scrubbing: the replay artifact carries suite identity, tier, summed cost, and
// unique-defect COUNTS — never the raw per-suite defect-ID membership, which
// could leak which internal check found what. It is the cost-ledger analog of
// the loader child recording tensor names and sizes but never raw weights.

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/stringset"
	"sort"
	"strings"
)

// costLedgerTier assigns this child's case to the PR gate: it is a pure
// in-process unit test with no external suite execution, so it runs on every
// pull request rather than being deferred to nightly or release.
const costLedgerTier = "PR"

// costLedgerSeed is the fixed deterministic-oracle seed recorded in provenance.
// The fold uses no randomness; the seed pins the fixture identity so a replay in
// a clean environment reproduces byte-identical rows.
const costLedgerSeed uint64 = 0xC0571ED6E42D

// costTier orders the three validation tiers by ascending spend, so a defect is
// attributed to the CHEAPEST tier that caught it.
type costTier int

const (
	costTierPR costTier = iota
	costTierNightly
	costTierRelease
)

func (t costTier) String() string {
	switch t {
	case costTierPR:
		return "PR"
	case costTierNightly:
		return "nightly"
	case costTierRelease:
		return "release"
	}
	return "unknown"
}

// costVector is the per-suite spend the ledger folds: compute (ns), tokens spent
// on generation, tokens spent on the LLM judge, wall time (ns), and the flake
// count. Every field is additive; a suite's cost is attributed to that suite
// alone, so summing runs never double-counts across suites.
type costVector struct {
	computeNanos int64
	genTokens    int64
	judgeTokens  int64
	wallNanos    int64
	flakes       int64
}

func (c costVector) add(o costVector) costVector {
	return costVector{
		computeNanos: c.computeNanos + o.computeNanos,
		genTokens:    c.genTokens + o.genTokens,
		judgeTokens:  c.judgeTokens + o.judgeTokens,
		wallNanos:    c.wallNanos + o.wallNanos,
		flakes:       c.flakes + o.flakes,
	}
}

// suiteRun is one run of one validation suite in the fixture month: its suite
// name, tier, cost, and the defect IDs it caught (a set; order-insensitive).
type suiteRun struct {
	suite   string
	tier    costTier
	cost    costVector
	defects []string
}

// costLedgerFixtureMonth is the pinned fixture: a month of validation suite-runs
// across all three tiers. Two facts make it a witness rather than a toy:
//
//   - The PR "unit" suite ran twice and caught D-101 in BOTH runs — run-level
//     dedup must count D-101 once for the suite.
//   - D-101 is ALSO caught by the nightly "accuracy" suite — cross-suite dedup
//     must attribute it to PR (the cheaper tier), not to both.
//
// Distinct defects: D-101, D-102, D-201, D-301 (4). Canonical attribution: unit
// gets D-101 + D-102 (2), accuracy gets D-201 (1), hardware gets D-301 (1) —
// summing to 4, the no-double-count invariant.
func costLedgerFixtureMonth() []suiteRun {
	return []suiteRun{
		{
			suite: "unit", tier: costTierPR,
			cost:    costVector{computeNanos: 3_000_000_000, genTokens: 600, judgeTokens: 300, wallNanos: 3_500_000_000, flakes: 1},
			defects: []string{"D-101"},
		},
		{
			suite: "unit", tier: costTierPR,
			cost:    costVector{computeNanos: 2_000_000_000, genTokens: 400, judgeTokens: 200, wallNanos: 2_500_000_000, flakes: 0},
			defects: []string{"D-102", "D-101"},
		},
		{
			suite: "accuracy", tier: costTierNightly,
			cost:    costVector{computeNanos: 80_000_000_000, genTokens: 40_000, judgeTokens: 12_000, wallNanos: 90_000_000_000, flakes: 3},
			defects: []string{"D-101", "D-201"},
		},
		{
			suite: "hardware", tier: costTierRelease,
			cost:    costVector{computeNanos: 200_000_000_000, genTokens: 5_000, judgeTokens: 2_000, wallNanos: 300_000_000_000, flakes: 0},
			defects: []string{"D-301"},
		},
	}
}

// ledgerRow is one suite's folded line: its identity and tier, its summed cost,
// and its MARGINAL (unique) defect count — the defects this suite caught that no
// cheaper tier already accounted for.
type ledgerRow struct {
	suite         string
	tier          costTier
	cost          costVector
	uniqueDefects int
}

// costLedger is the published fold: the per-suite rows in canonical order plus
// the month-wide distinct-defect count the no-double-count invariant checks.
type costLedger struct {
	rows            []ledgerRow
	distinctDefects int
}

// foldLedger is the canonical, replay-order-independent fold. It groups runs by
// suite (summing cost, unioning defect sets), orders suites by (tier asc, name
// asc), then attributes each defect to the FIRST suite in that order that caught
// it — cheapest tier wins, ties by name. The result is deterministic regardless
// of the order runs were recorded/replayed in.
func foldLedger(month []suiteRun) costLedger {
	type agg struct {
		tier    costTier
		cost    costVector
		defects map[string]struct{}
	}
	bySuite := map[string]*agg{}
	for _, r := range month {
		a := bySuite[r.suite]
		if a == nil {
			a = &agg{tier: r.tier, defects: map[string]struct{}{}}
			bySuite[r.suite] = a
		}
		a.cost = a.cost.add(r.cost)
		for _, d := range r.defects {
			a.defects[d] = struct{}{}
		}
	}

	suites := make([]string, 0, len(bySuite))
	for s := range bySuite {
		suites = append(suites, s)
	}
	sort.Slice(suites, func(i, j int) bool {
		ai, aj := bySuite[suites[i]], bySuite[suites[j]]
		if ai.tier != aj.tier {
			return ai.tier < aj.tier
		}
		return suites[i] < suites[j]
	})

	claimed := map[string]struct{}{}
	distinct := map[string]struct{}{}
	unique := map[string]int{}
	for _, s := range suites {
		for _, d := range sortedDefects(bySuite[s].defects) {
			distinct[d] = struct{}{}
			if _, seen := claimed[d]; !seen {
				claimed[d] = struct{}{}
				unique[s]++
			}
		}
	}

	rows := make([]ledgerRow, 0, len(suites))
	for _, s := range suites {
		a := bySuite[s]
		rows = append(rows, ledgerRow{suite: s, tier: a.tier, cost: a.cost, uniqueDefects: unique[s]})
	}
	return costLedger{rows: rows, distinctDefects: len(distinct)}
}

func sortedDefects(set map[string]struct{}) []string { return stringset.Sorted(set) }

// --- faithful folds: same ledger regardless of replay order -----------------

// ledgerFold is one fold implementation ("engine/backend"): a name, the backend
// it models, and the fold transform it applies to the fixture month.
type ledgerFold struct {
	name    string
	backend string
	fold    func(month []suiteRun) costLedger
}

// costLedgerFaithfulFolds are the in-scope faithful folds. Each arrives at the
// canonical ledger by a genuinely different replay ORDER of the same month, so a
// correct fold is row-identical: the cost ledger must be replay-order-independent
// (the "independently replayed environment" the witness demands).
func costLedgerFaithfulFolds() []ledgerFold {
	return []ledgerFold{
		{"as-recorded", "batched-fold", foldLedger},
		{"reversed-replay", "reverse-stream", func(m []suiteRun) costLedger { return foldLedger(reverseRuns(m)) }},
		{"rotated-replay", "rotated-stream", func(m []suiteRun) costLedger { return foldLedger(rotateRuns(m, 2)) }},
	}
}

func reverseRuns(m []suiteRun) []suiteRun {
	out := make([]suiteRun, len(m))
	for i, r := range m {
		out[len(m)-1-i] = r
	}
	return out
}

func rotateRuns(m []suiteRun, by int) []suiteRun {
	if len(m) == 0 {
		return m
	}
	by %= len(m)
	out := make([]suiteRun, 0, len(m))
	out = append(out, m[by:]...)
	out = append(out, m[:by]...)
	return out
}

// --- planted representative defects -----------------------------------------

// costDoubleCountFold plants the marquee defect: it attributes each caught defect
// to EVERY suite that caught it, so a defect several suites caught is counted more
// than once — the exact double count the ledger must never do. It inflates the
// nightly "accuracy" row (which shares D-101 with PR "unit") and breaks the
// no-double-count invariant.
func costDoubleCountFold(month []suiteRun) costLedger {
	l := foldLedger(month)
	bySuite := map[string]map[string]struct{}{}
	for _, r := range month {
		set := bySuite[r.suite]
		if set == nil {
			set = map[string]struct{}{}
			bySuite[r.suite] = set
		}
		for _, d := range r.defects {
			set[d] = struct{}{}
		}
	}
	for i := range l.rows {
		l.rows[i].uniqueDefects = len(bySuite[l.rows[i].suite])
	}
	return l
}

// costDroppedComponentFold plants a dropped cost component: it forgets the judge
// spend when summing, so every suite that paid for an LLM judge under-reports its
// cost. It bites at the first suite with nonzero judge tokens ("unit").
func costDroppedComponentFold(month []suiteRun) costLedger {
	scrubbed := make([]suiteRun, len(month))
	for i, r := range month {
		r.cost.judgeTokens = 0
		scrubbed[i] = r
	}
	return foldLedger(scrubbed)
}

// costMisTierFold plants a mis-tiered suite: it promotes the nightly "accuracy"
// suite to the PR tier, changing the attribution order so a defect is credited to
// the wrong suite and the canonical row order shifts. It bites at row 0.
func costMisTierFold(month []suiteRun) costLedger {
	retiered := make([]suiteRun, len(month))
	for i, r := range month {
		if r.suite == "accuracy" {
			r.tier = costTierPR
		}
		retiered[i] = r
	}
	return foldLedger(retiered)
}

// --- canonical bytes, fingerprint, and the publish gate ---------------------

// costLedgerCanonicalBytes serializes the ledger in canonical row order: each
// row's suite, tier, five cost fields, and unique-defect count, then the
// month-wide distinct count. The bytes change only when a folded value changes.
func costLedgerCanonicalBytes(l costLedger) []byte {
	var b []byte
	var v [8]byte
	putI := func(x int64) {
		binary.BigEndian.PutUint64(v[:], uint64(x))
		b = append(b, v[:]...)
	}
	for _, r := range l.rows {
		b = append(b, r.suite...)
		putI(int64(r.tier))
		putI(r.cost.computeNanos)
		putI(r.cost.genTokens)
		putI(r.cost.judgeTokens)
		putI(r.cost.wallNanos)
		putI(r.cost.flakes)
		putI(int64(r.uniqueDefects))
	}
	putI(int64(l.distinctDefects))
	return b
}

func costLedgerFingerprint(l costLedger) [32]byte { return sha256.Sum256(costLedgerCanonicalBytes(l)) }

// costPublishGate is the fast pre-publish guard: it refuses to publish a month
// ledger whose canonical fingerprint does not match the pinned baseline. Every
// planted defect changes the fingerprint and is refused here, before the ledger
// is reported as the month's cost-and-defect-yield summary.
func costPublishGate(l costLedger, baseline [32]byte) error {
	if got := costLedgerFingerprint(l); got != baseline {
		return fmt.Errorf("pre-publish fingerprint mismatch: ledger %x != baseline %x", got[:6], baseline[:6])
	}
	return nil
}

// --- no-double-count invariant ----------------------------------------------

func sumUnique(l costLedger) int {
	s := 0
	for _, r := range l.rows {
		s += r.uniqueDefects
	}
	return s
}

// noDoubleCount reports the load-bearing invariant: the summed per-suite unique
// (marginal) defect counts must equal the month's distinct-defect count. If a
// defect were credited to two suites, the sum would exceed the distinct count.
func noDoubleCount(l costLedger) bool { return sumUnique(l) == l.distinctDefects }

// --- provenance, replay artifact, and the differential oracle ---------------

// costProvenance records everything the acceptance criteria require per case:
// model (the fixture-month identity), tokenizer (the ledger schema), engine/
// backend (the fold), seed/oracle, code revision, and tolerance/baseline. Suites
// are recorded scrubbed — name, tier, and unique-defect COUNT only, never the raw
// per-suite defect-ID membership.
type costProvenance struct {
	CaseID    string
	Model     string
	Tokenizer string
	Backend   string
	Seed      uint64
	Revision  string
	Baseline  string
	Tolerance string
	Tier      string
	Suites    []string
}

func costProvenanceOf(caseID string, l costLedger, backend string, baseline [32]byte) costProvenance {
	suites := make([]string, 0, len(l.rows))
	for _, r := range l.rows {
		suites = append(suites, fmt.Sprintf("%s:%s:u%d", r.suite, r.tier, r.uniqueDefects))
	}
	return costProvenance{
		CaseID: caseID, Model: "fak-quality-fixture-month", Tokenizer: "cost-ledger-v1", Backend: backend,
		Seed: costLedgerSeed, Revision: "rungobs@cost-ledger-1", Baseline: fmt.Sprintf("%x", baseline[:6]),
		Tolerance: "exact (deterministic fold, no clock)", Tier: costLedgerTier, Suites: suites,
	}
}

// complete reports whether every required provenance field is populated — an
// unprovenanced case is inconclusive and must never be reported as pass.
func (p costProvenance) complete() bool {
	return p.Model != "" && p.Tokenizer != "" && p.Backend != "" && p.Revision != "" &&
		p.Baseline != "" && p.Tolerance != "" && p.Tier != "" && p.Seed != 0
}

// costDivergence is the first actionable divergence: the ledger row index, the
// field that first differs there, and the reference vs candidate values.
type costDivergence struct {
	Index     int
	Field     string
	Reference string
	Candidate string
}

// costReplayArtifact is the scrubbed, independently-replayable failure bundle:
// full provenance plus the first divergence, carrying suite names/tiers/counts
// but never raw defect-ID membership.
type costReplayArtifact struct {
	Provenance costProvenance
	FailPath   string
	Reason     string
	Divergence *costDivergence
}

func (a costReplayArtifact) String() string {
	idx, field, ref, cand := -1, "<none>", "<none>", "<none>"
	if a.Divergence != nil {
		idx, field, ref, cand = a.Divergence.Index, a.Divergence.Field, a.Divergence.Reference, a.Divergence.Candidate
	}
	p := a.Provenance
	return fmt.Sprintf("replay{case=%s model=%s tok=%s backend=%s seed=%#x rev=%s baseline=%s tol=%q tier=%s fail=%s reason=%s divergence=@%d field=%s ref=%q cand=%q suites=%s}",
		p.CaseID, p.Model, p.Tokenizer, p.Backend, p.Seed, p.Revision, p.Baseline, p.Tolerance, p.Tier,
		a.FailPath, a.Reason, idx, field, ref, cand, strings.Join(p.Suites, ","))
}

type costVerdict struct {
	Pass     bool
	Detail   string
	Artifact *costReplayArtifact
}

// rowDiff reports the first field at which two ledger rows differ, or nil if the
// rows are identical. Field order (suite, tier, cost components, unique count)
// makes the reported field the most actionable one.
func rowDiff(i int, ref, cand ledgerRow) *costDivergence {
	mk := func(field, r, c string) *costDivergence {
		return &costDivergence{Index: i, Field: field, Reference: r, Candidate: c}
	}
	if ref.suite != cand.suite {
		return mk("suite", ref.suite, cand.suite)
	}
	if ref.tier != cand.tier {
		return mk("tier", ref.tier.String(), cand.tier.String())
	}
	if ref.cost.computeNanos != cand.cost.computeNanos {
		return mk("compute_nanos", fmt.Sprint(ref.cost.computeNanos), fmt.Sprint(cand.cost.computeNanos))
	}
	if ref.cost.genTokens != cand.cost.genTokens {
		return mk("gen_tokens", fmt.Sprint(ref.cost.genTokens), fmt.Sprint(cand.cost.genTokens))
	}
	if ref.cost.judgeTokens != cand.cost.judgeTokens {
		return mk("judge_tokens", fmt.Sprint(ref.cost.judgeTokens), fmt.Sprint(cand.cost.judgeTokens))
	}
	if ref.cost.wallNanos != cand.cost.wallNanos {
		return mk("wall_nanos", fmt.Sprint(ref.cost.wallNanos), fmt.Sprint(cand.cost.wallNanos))
	}
	if ref.cost.flakes != cand.cost.flakes {
		return mk("flakes", fmt.Sprint(ref.cost.flakes), fmt.Sprint(cand.cost.flakes))
	}
	if ref.uniqueDefects != cand.uniqueDefects {
		return mk("unique_defects", fmt.Sprint(ref.uniqueDefects), fmt.Sprint(cand.uniqueDefects))
	}
	return nil
}

// costJudge is the differential oracle: a candidate fold must reproduce the
// reference ledger exactly AND satisfy the no-double-count invariant. An empty
// candidate is never a pass; any divergence is reported as the first row/field
// with a scrubbed replay artifact.
func costJudge(ref, cand costLedger, prov costProvenance) costVerdict {
	mk := func(reason string, d *costDivergence) *costReplayArtifact {
		return &costReplayArtifact{Provenance: prov, FailPath: prov.Backend, Reason: reason, Divergence: d}
	}
	if len(cand.rows) == 0 {
		return costVerdict{Pass: false, Detail: "candidate produced no ledger rows — inconclusive evidence is never pass",
			Artifact: mk("no-evidence", &costDivergence{Index: 0, Field: "rows", Reference: fmt.Sprintf("%d rows", len(ref.rows)), Candidate: "0 rows"})}
	}
	n := len(ref.rows)
	if len(cand.rows) < n {
		n = len(cand.rows)
	}
	for i := 0; i < n; i++ {
		if d := rowDiff(i, ref.rows[i], cand.rows[i]); d != nil {
			return costVerdict{Pass: false,
				Detail:   fmt.Sprintf("ledger diverged at row %d field %s: reference %q, candidate %q — the fold does not reproduce the month", d.Index, d.Field, d.Reference, d.Candidate),
				Artifact: mk("divergence", d)}
		}
	}
	if len(ref.rows) != len(cand.rows) {
		d := &costDivergence{Index: n, Field: "row-count", Reference: fmt.Sprint(len(ref.rows)), Candidate: fmt.Sprint(len(cand.rows))}
		return costVerdict{Pass: false,
			Detail:   fmt.Sprintf("suite-row count diverged at %d: reference has %d, candidate has %d — a suite was dropped or duplicated", n, len(ref.rows), len(cand.rows)),
			Artifact: mk("row-count-divergence", d)}
	}
	// Belt-and-suspenders: even a ledger that matched the reference so far is
	// refused if it violates the no-double-count invariant, so the property holds
	// independently of the reference it was compared against.
	if !noDoubleCount(cand) {
		d := &costDivergence{Index: 0, Field: "no-double-count", Reference: fmt.Sprint(cand.distinctDefects), Candidate: fmt.Sprint(sumUnique(cand))}
		return costVerdict{Pass: false,
			Detail:   fmt.Sprintf("no-double-count invariant violated: summed unique defects %d != distinct defects %d", sumUnique(cand), cand.distinctDefects),
			Artifact: mk("double-count", d)}
	}
	if ref.distinctDefects != cand.distinctDefects {
		d := &costDivergence{Index: n, Field: "distinct_defects", Reference: fmt.Sprint(ref.distinctDefects), Candidate: fmt.Sprint(cand.distinctDefects)}
		return costVerdict{Pass: false,
			Detail:   fmt.Sprintf("distinct-defect count diverged: reference %d, candidate %d", ref.distinctDefects, cand.distinctDefects),
			Artifact: mk("distinct-divergence", d)}
	}
	return costVerdict{Pass: true, Detail: fmt.Sprintf("fold reproduced the reference: %d suites, %d distinct defects, no double count", len(ref.rows), ref.distinctDefects)}
}

// costFirstDiff returns the first row index at which two ledgers differ, the min
// row count if one is a prefix of the other, or -1 if identical. It lets the
// defect tests assert the oracle's localization without hard-coding an index.
func costFirstDiff(a, b costLedger) int {
	n := len(a.rows)
	if len(b.rows) < n {
		n = len(b.rows)
	}
	for i := 0; i < n; i++ {
		if rowDiff(i, a.rows[i], b.rows[i]) != nil {
			return i
		}
	}
	if len(a.rows) != len(b.rows) || a.distinctDefects != b.distinctDefects {
		return n
	}
	return -1
}
