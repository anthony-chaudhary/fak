package qaprocessscore

// flakequarantine.go closes the detect-without-enforce loop on flaky tests (#3836, epic #3831).
//
// internal/brittleness already CAPTURES a flake while it is fresh: `fak affected --rerun-fail N`
// reruns an initially-red package on the same tree, and a package that then passes is
// FLAKY_PASSED_ON_RETRY -- exonerated, exit dropped to 0, folded into a FLAKY_RETRY_PASS Finding
// with the tree sha it flaked on. But that finding rides only as a SOFT brittleness_pressure
// observation and never gates, and --rerun-fail carries NO cap: a chronically non-deterministic
// package may flake forever, silently retried green, with nothing tracking it and nothing filed.
//
// This file adds the enforcement half, as a pure fold with no I/O:
//
//   - a durable QUARANTINE LEDGER keyed by per-test identity (pkg, or pkg.Test when the producer
//     records a test name), accumulating FlakeObservation rows across runs;
//   - a RERUN BUDGET: an identity that exceeds Budget flakes ACROSS A SOAK (Soak or more distinct
//     trees) is quarantined and surfaces as a HARD flake_quarantine defect -- one unit of
//     qa_process_debt, which the qa-process card gates on;
//   - a deduped FAN-OUT: FlakeQuarantineGaps projects each quarantined identity onto the shared
//     Gap shape, so the existing dispatcher (dispatch.go -> dogfoodissues) files exactly one
//     content-stable "deflake" ticket per flaky test.
//
// Why the soak requirement, and why this may gate when brittleness may not: brittleness reads a
// git-history window, and a landed commit cannot be un-shipped on a no-rewrite trunk, so its
// findings can never be HARD, in-tree-mendable debt. A rerun-masked flake is the opposite: it is
// a CURRENT property of the working tree, and the mend (make the test deterministic) is in-tree
// and available to whoever owns the package. Requiring the flakes to span multiple distinct trees
// before quarantine is what keeps that honest -- a burst on one bad tree is transient, while the
// same identity flaking across several trees is chronic non-determinism that poisons the shared
// fast gate for every peer.
//
// The ledger is durable but this leaf holds no disk: EncodeObservations/DecodeObservations are the
// append-only JSONL wire format over an io.Writer/io.Reader, so a shell appends a run's rows to a
// soak file and the fold stays pure and fixture-testable -- the same pure-core/shell-owns-I/O split
// the rest of the scorecard family uses.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/brittleness"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// FlakeQuarantineKey is both the KPI key and the qa-process dogfood finding name
// (qa-process-score/flake-quarantine-ledger). #3836.
const FlakeQuarantineKey = "flake_quarantine"

// flakeOverBudgetClass is the closed-vocabulary work-list token a quarantined identity renders
// under -- the same grep-able "<CLASS> <ref>: <detail>" discipline the sibling KPIs use.
const flakeOverBudgetClass = "FLAKE_OVER_BUDGET"

// DefaultRerunBudget is how many rerun-masked flakes one test identity may spend before it is
// quarantined: exceeding it (strictly more than Budget) is HARD debt. Two is the smallest budget
// that still tolerates a genuine one-off (a loaded CI box, a single unlucky scheduling) without
// tolerating a pattern.
const DefaultRerunBudget = 2

// DefaultSoakTrees is how many DISTINCT trees an identity must have flaked on before the budget
// may quarantine it. One tree is a burst (possibly one broken commit, mendable by a revert); two
// or more is chronic non-determinism that survived a code change -- the "after a soak" condition
// that keeps the gate off transient history.
const DefaultSoakTrees = 2

// modulePathPrefix is trimmed off a Go import path to get the repo-relative package dir the
// generated issue routes to, so a finding recorded as an import path still lands in the lane that
// owns the files.
const modulePathPrefix = "github.com/anthony-chaudhary/fak/"

// FlakeObservation is ONE rerun-masked flake event: a test identity that failed and then passed on
// a same-tree rerun, stamped with the tree it flaked on. Test is optional -- today's producer
// (internal/affectedtests via brittleness.FromFlakyPackages) records package-level identity, the
// finest grain the rerun classifier keeps -- so an empty Test means "the whole package", and a
// producer that later records test names gets per-test quarantine for free. Json-tagged because
// these rows ARE the durable ledger's wire format (one JSON object per line).
type FlakeObservation struct {
	Pkg  string `json:"pkg"`
	Test string `json:"test,omitempty"`
	Tree string `json:"tree,omitempty"` // freshness stamp: the tree sha the rerun ran on
}

// Identity is the STABLE per-test dedup anchor: "pkg" for a package-level observation, or
// "pkg.Test" when a test name was recorded. It is what the ledger keys on and what the generated
// issue's content-stable key is built from, so the same flaky test always folds to one row and
// files one ticket.
func (o FlakeObservation) Identity() string {
	pkg := strings.TrimSpace(o.Pkg)
	test := strings.TrimSpace(o.Test)
	if test == "" {
		return pkg
	}
	return pkg + "." + test
}

// QuarantineEntry is one folded ledger row: a test identity, how many times it flaked, the distinct
// trees it flaked on (the soak evidence, sorted), and whether the budget quarantined it.
type QuarantineEntry struct {
	ID          string   `json:"id"`
	Pkg         string   `json:"pkg"`
	Test        string   `json:"test,omitempty"`
	Flakes      int      `json:"flakes"`
	Trees       []string `json:"trees,omitempty"`
	Quarantined bool     `json:"quarantined"`
}

// workItem renders an entry as a single closed-vocabulary work-list line.
func (e QuarantineEntry) workItem() string {
	return fmt.Sprintf("%s %s: %s -- make it deterministic or skip it with a tracking ticket; until then --rerun-fail masks it green and it poisons the shared fast gate for every peer",
		flakeOverBudgetClass, e.ID, e.detail())
}

// detail is the run-varying half of the work-list line (counts and tree stamps). It is
// deliberately NOT part of any dedup key.
func (e QuarantineEntry) detail() string {
	d := fmt.Sprintf("%s across %s", scorecard.CountNoun(e.Flakes, "rerun-masked flake"), scorecard.CountNoun(len(e.Trees), "tree"))
	if len(e.Trees) > 0 {
		d += " [" + strings.Join(e.Trees, ", ") + "]"
	}
	return d
}

// QuarantineLedger is the durable quarantine state: the budget it was folded under, the soak
// threshold, and one entry per observed test identity (worst-first). It is pure data -- json-tagged
// so a shell can persist and re-read the folded view, while the append-only observation rows stay
// the source of truth.
type QuarantineLedger struct {
	Budget  int               `json:"budget"`
	Soak    int               `json:"soak"`
	Entries []QuarantineEntry `json:"entries"`
}

// FoldQuarantineLedger is the pure fold: observations in, ledger out. No disk, no clock, no git.
//
// Observations are grouped by Identity; Flakes counts every observation (a package that flaked
// three times in one run is three flakes) while Trees de-duplicates the freshness stamps (that
// same run is one tree). An identity is Quarantined when it exceeds the budget AND spans at least
// Soak distinct trees -- both dials must trip, so neither a single loud run nor a long-tail
// once-per-tree flake alone can gate. A budget <= 0 falls back to DefaultRerunBudget and a soak
// <= 0 to DefaultSoakTrees, so a zero-value caller gets the documented defaults rather than a
// gate that fires on the first flake. Entries sort worst-first (most flakes, then id) for a
// deterministic work list.
func FoldQuarantineLedger(obs []FlakeObservation, budget, soak int) QuarantineLedger {
	if budget <= 0 {
		budget = DefaultRerunBudget
	}
	if soak <= 0 {
		soak = DefaultSoakTrees
	}

	byID := map[string]*QuarantineEntry{}
	trees := map[string]map[string]bool{}
	var order []string
	for _, o := range obs {
		id := o.Identity()
		if id == "" {
			continue // an observation with no package names nothing -- skip, never invent an identity
		}
		e, ok := byID[id]
		if !ok {
			e = &QuarantineEntry{ID: id, Pkg: strings.TrimSpace(o.Pkg), Test: strings.TrimSpace(o.Test)}
			byID[id] = e
			trees[id] = map[string]bool{}
			order = append(order, id)
		}
		e.Flakes++
		if t := strings.TrimSpace(o.Tree); t != "" {
			trees[id][t] = true
		}
	}

	entries := make([]QuarantineEntry, 0, len(order))
	for _, id := range order {
		e := byID[id]
		for t := range trees[id] {
			e.Trees = append(e.Trees, t)
		}
		sort.Strings(e.Trees)
		e.Quarantined = e.Flakes > budget && len(e.Trees) >= soak
		entries = append(entries, *e)
	}
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].Flakes != entries[j].Flakes {
			return entries[i].Flakes > entries[j].Flakes
		}
		return entries[i].ID < entries[j].ID
	})
	return QuarantineLedger{Budget: budget, Soak: soak, Entries: entries}
}

// OverBudget returns just the quarantined entries, worst-first -- the HARD half of the ledger.
func (l QuarantineLedger) OverBudget() []QuarantineEntry {
	var out []QuarantineEntry
	for _, e := range l.Entries {
		if e.Quarantined {
			out = append(out, e)
		}
	}
	return out
}

// ObservationsFromFindings is the bridge from the sibling capture leaf: every brittleness
// FLAKY_RETRY_PASS finding becomes Weight observations of its Ref, each stamped with the finding's
// captured-fresh tree sha. Other brittleness classes (RECURRING_FIX, REVERTED_LANDING) are history
// observations, not rerun-masked flakes, and are ignored -- history never feeds this gate. A
// finding carrying several Fresh stamps contributes one observation per stamp so a re-captured
// flake widens the soak rather than inflating a single tree.
func ObservationsFromFindings(fs []brittleness.Finding) []FlakeObservation {
	var out []FlakeObservation
	for _, f := range fs {
		if f.Class != brittleness.ClassFlakyRetryPass {
			continue
		}
		pkg := strings.TrimSpace(f.Ref)
		if pkg == "" {
			continue
		}
		n := f.Weight
		if n <= 0 {
			n = 1
		}
		stamps := f.Fresh
		if len(stamps) == 0 {
			stamps = []string{""}
		}
		for i := 0; i < n; i++ {
			// Spread the observations over the finding's stamps: one stamp -> all on that tree.
			out = append(out, FlakeObservation{Pkg: pkg, Tree: strings.TrimSpace(stamps[i%len(stamps)])})
		}
	}
	return out
}

// EncodeObservations writes rows as append-only JSONL (one JSON object per line) -- the durable
// ledger's wire format. Append a run's rows to the soak file; DecodeObservations reads the whole
// window back. The leaf never opens a file: the caller owns the io.Writer, so the fold stays pure.
func EncodeObservations(w io.Writer, obs []FlakeObservation) error {
	enc := json.NewEncoder(w)
	for _, o := range obs {
		if o.Identity() == "" {
			continue
		}
		if err := enc.Encode(o); err != nil {
			return fmt.Errorf("encoding flake observation %q: %w", o.Identity(), err)
		}
	}
	return nil
}

// DecodeObservations reads an append-only JSONL observation stream back. Blank lines are skipped;
// a malformed line is an error rather than a silent drop, because a ledger that quietly loses rows
// under-counts flakes and fails OPEN -- the exact failure this KPI exists to stop.
func DecodeObservations(r io.Reader) ([]FlakeObservation, error) {
	var out []FlakeObservation
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for line := 1; sc.Scan(); line++ {
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var o FlakeObservation
		if err := json.Unmarshal([]byte(raw), &o); err != nil {
			return nil, fmt.Errorf("flake ledger line %d: %w", line, err)
		}
		out = append(out, o)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("reading flake ledger: %w", err)
	}
	return out, nil
}

// FlakeQuarantine folds the ledger into the flake_quarantine KPI. Every quarantined identity is one
// HARD defect (one unit of qa_process_debt): unlike a brittleness history finding it is a CURRENT,
// in-tree-mendable property of the working tree, so it may gate. Identities that flaked but stayed
// within budget ride as SOFT notes -- visible in the work list, never gating -- so the card shows
// the whole flake surface, not just the part over the line.
//
// An empty ledger scores 100 with an explicit "no rerun-masked flakes observed" detail: the fold
// never manufactures debt it did not measure. Per the family's anti-fail-open discipline (#3833)
// the CALLER omits this KPI entirely when no ledger was supplied, so an unmeasured card shows no
// flake_quarantine row rather than a hollow green.
func FlakeQuarantine(l QuarantineLedger) scorecard.KPI {
	over := l.OverBudget()
	defects := make([]string, 0, len(over))
	for _, e := range over {
		defects = append(defects, e.workItem())
	}

	var soft []string
	for _, e := range l.Entries {
		if e.Quarantined {
			continue
		}
		soft = append(soft, fmt.Sprintf("FLAKE_WITHIN_BUDGET %s: %s (budget %d flakes over %d tree(s))",
			e.ID, e.detail(), l.Budget, l.Soak))
	}

	total := len(l.Entries)
	score := 100.0
	detail := "no rerun-masked flakes observed in the soak window"
	if total > 0 {
		score = scorecard.Round1(100 * float64(total-len(over)) / float64(total))
		detail = fmt.Sprintf("%d/%d flaky identities within the rerun budget (%d flakes over %d distinct tree(s)); %s quarantined",
			total-len(over), total, l.Budget, l.Soak, scorecard.CountNoun(len(over), "identity"))
	}
	return scorecard.KPI{
		Key:     FlakeQuarantineKey,
		Group:   "flake",
		Score:   score,
		Detail:  detail,
		Defects: defects,
		Soft:    soft,
	}
}

// FlakeQuarantineGaps projects each quarantined identity onto the shared Gap shape, so the existing
// dispatcher fans out exactly ONE deduped "deflake" ticket per flaky test. Ref is the per-test
// Identity, which is what makes Gap.Key content-stable across runs: the flake counts and tree
// stamps live only in Detail, so a re-run with more flakes UPDATES the same issue instead of
// opening a duplicate. Routing rides the package tree, so the ticket lands in the lane that owns
// the non-deterministic test.
func FlakeQuarantineGaps(l QuarantineLedger) []Gap {
	over := l.OverBudget()
	gaps := make([]Gap, 0, len(over))
	for _, e := range over {
		gaps = append(gaps, Gap{
			KPI:    FlakeQuarantineKey,
			Class:  flakeOverBudgetClass,
			Ref:    e.ID,
			Detail: e.workItem(),
			Paths:  []string{packageTree(e.Pkg) + "/**"},
			Grade:  "D",
		})
	}
	return gaps
}

// packageTree turns a Go import path into the repo-relative package dir a generated issue routes
// on, by trimming the module prefix. A path that is already repo-relative (or belongs to another
// module) is returned unchanged -- routing degrades to the literal ref rather than guessing.
func packageTree(pkg string) string {
	pkg = strings.TrimSpace(pkg)
	pkg = strings.TrimSuffix(pkg, "/")
	if t := strings.TrimPrefix(pkg, modulePathPrefix); t != pkg {
		return t
	}
	return pkg
}
