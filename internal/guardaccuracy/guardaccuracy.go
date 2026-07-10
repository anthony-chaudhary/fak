// Package guardaccuracy folds a labeled command corpus through the real guard
// reversibility classifier and scores its accuracy: the false-positive rate
// (benign calls the guard escalated) and the false-negative rate (dangerous
// calls the guard let through as reversible). It is the accuracy dual of the
// guard-verdict-rsi loop -- where that loop scores whether verdicts are
// EXPLAINED (honesty), this scores whether the escalate/don't-escalate boundary
// is CORRECT (tuning), as a number with a witness rather than a vibe.
//
// The design is the "guard-accuracy scorecard" named as the cheapest missing
// next step in docs/notes/CONCEPT-GUARD-FALSE-POSITIVE-DURABILITY-2026-07-02.md:
// a shared, growing corpus (testdata/corpus.json) that the garden / control-pane
// ratchet folds worst-first, so "a guard misfired" becomes "the fleet measured
// and retired guard debt". Every misfire found in the wild is added as a
// permanent row, never patched locally -- a fix is a ratchet, not a patch.
//
// The classifier is imported read-only; this package never edits the hard-self
// internal/adjudicator lane. Same corpus + same classifier => same payload
// (deterministic), and the fold is proven non-vacuous by a test that injects a
// known FP and FN and confirms both surface as debt.
package guardaccuracy

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	// Schema is the control-pane payload schema tag.
	Schema = "fak-guard-accuracy/1"
	// DebtKey is the corpus key the shared fold writes the headline debt count into.
	DebtKey = "guard_accuracy_debt"
	// minPolarity is the smallest number of rows of each polarity (benign and
	// dangerous) the corpus must carry before an accuracy claim is non-vacuous:
	// a green rate on a corpus that cannot exhibit one of the two failure modes
	// is not evidence. Mirrors the empty-journal honesty gate in guard-verdict-rsi.
	minPolarity = 5
)

//go:embed testdata/corpus.json
var seedCorpusJSON []byte

// CorpusRow is one labeled (tool,args) call with the preview class the guard
// MUST assign it. Expect is one of the adjudicator.Reversibility* string values.
type CorpusRow struct {
	Name      string         `json:"name"`
	Tool      string         `json:"tool"`
	Args      map[string]any `json:"args"`
	Expect    string         `json:"expect"`
	Rationale string         `json:"rationale"`
}

type corpusFile struct {
	Rows []CorpusRow `json:"rows"`
}

// SeedCorpus returns the rows embedded at build time. It panics only if the
// embedded asset is corrupt, which a test would catch immediately.
func SeedCorpus() []CorpusRow {
	rows, err := parseCorpus(seedCorpusJSON)
	if err != nil {
		panic("guardaccuracy: embedded corpus is unparseable: " + err.Error())
	}
	return rows
}

// LoadCorpus parses a corpus file's bytes into rows.
func LoadCorpus(data []byte) ([]CorpusRow, error) { return parseCorpus(data) }

func parseCorpus(data []byte) ([]CorpusRow, error) {
	var cf corpusFile
	if err := json.Unmarshal(data, &cf); err != nil {
		return nil, err
	}
	return cf.Rows, nil
}

// outcome labels how one row's actual class compares to its expected class.
type outcome int

const (
	outcomeCorrect       outcome = iota // classifier matched the expected class exactly
	outcomeFalsePositive                // expected reversible, classifier escalated
	outcomeFalseNegative                // expected escalation, classifier returned reversible
	outcomeClassDrift                   // both escalated but the subclass differs (advisory only)
)

func escalated(class string) bool { return class != string(adjudicator.ReversibilityReversible) }

// grade classifies one row and returns its outcome plus the class the guard
// actually assigned.
func grade(r CorpusRow) (outcome, string) {
	got := string(adjudicator.ClassifyReversibility(r.Tool, r.Args).Class)
	switch {
	case got == r.Expect:
		return outcomeCorrect, got
	case !escalated(r.Expect) && escalated(got):
		return outcomeFalsePositive, got
	case escalated(r.Expect) && !escalated(got):
		return outcomeFalseNegative, got
	default:
		// Both escalate (e.g. expected irreversible, got outward-facing): the
		// preview-confirm pause fires either way, so the security floor holds --
		// the subclass label is cosmetic. Surfaced as a soft advisory, never debt.
		return outcomeClassDrift, got
	}
}

// Result is the folded accuracy corpus, exposed for callers that want the raw
// numbers alongside the scorecard payload.
type Result struct {
	Total         int
	Correct       int
	BenignRows    int
	DangerousRows int
	FalsePositive []string // "<name> (expect reversible, got <class>)"
	FalseNegative []string
	ClassDrift    []string
	FPRate        float64
	FNRate        float64
}

// Fold runs every row through the classifier and tallies the outcomes.
func Fold(rows []CorpusRow) Result {
	res := Result{Total: len(rows)}
	for _, r := range rows {
		if escalated(r.Expect) {
			res.DangerousRows++
		} else {
			res.BenignRows++
		}
		oc, got := grade(r)
		switch oc {
		case outcomeCorrect:
			res.Correct++
		case outcomeFalsePositive:
			res.FalsePositive = append(res.FalsePositive,
				fmt.Sprintf("%s (expect %s, got %s)", r.Name, r.Expect, got))
		case outcomeFalseNegative:
			res.FalseNegative = append(res.FalseNegative,
				fmt.Sprintf("%s (expect %s, got %s)", r.Name, r.Expect, got))
		case outcomeClassDrift:
			res.ClassDrift = append(res.ClassDrift,
				fmt.Sprintf("%s (expect %s, got %s)", r.Name, r.Expect, got))
		}
	}
	if res.BenignRows > 0 {
		res.FPRate = float64(len(res.FalsePositive)) / float64(res.BenignRows)
	}
	if res.DangerousRows > 0 {
		res.FNRate = float64(len(res.FalseNegative)) / float64(res.DangerousRows)
	}
	return res
}

func offenders(names []string) string {
	if len(names) == 0 {
		return ""
	}
	sort.Strings(names)
	const max = 4
	if len(names) > max {
		return strings.Join(names[:max], "; ") + fmt.Sprintf("; +%d more", len(names)-max)
	}
	return strings.Join(names, "; ")
}

// BuildScorecard folds the embedded seed corpus into the control-pane payload.
func BuildScorecard(root string) scorecard.Payload {
	return BuildScorecardFromRows(root, SeedCorpus())
}

// BuildScorecardFromRows folds an arbitrary corpus, so a caller can score an
// override corpus and a test can inject a known FP/FN to prove the fold detects
// them. The KPIs, weights, and grade come from the shared pkg/scorecard kernel.
func BuildScorecardFromRows(root string, rows []CorpusRow) scorecard.Payload {
	return BuildScorecardWithComplaints(root, rows, nil)
}

// BuildScorecardWithComplaints folds the labeled corpus AND the agent-authored
// guard-complaint intake (#2821). The corpus half is unchanged hard debt (the
// escalate/don't boundary scored against ground truth); the complaint half is an
// advisory field-false-positive triage queue (scorecard Soft) that never counts
// as debt and never reds the gate. That split is deliberate: a complaint is a
// cheap-to-file self-report (byte-identical in the journal to a correct refusal),
// so scoring it as debt would let an agent lower the guard's assertiveness by
// appealing. A confirmed complaint becomes a measured FP only once it is promoted
// into testdata/corpus.json, where the corpus fold scores it. Passing a nil/empty
// complaint slice yields a payload byte-identical to BuildScorecardFromRows.
func BuildScorecardWithComplaints(root string, rows []CorpusRow, complaints []FieldComplaint) scorecard.Payload {
	res := Fold(rows)

	fpKPI := scorecard.KPI{
		Key:    "no_false_positives",
		Group:  "accuracy",
		Score:  100 * (1 - res.FPRate),
		Detail: fmt.Sprintf("%d/%d benign call(s) the guard wrongly escalated", len(res.FalsePositive), res.BenignRows),
	}
	if len(res.FalsePositive) > 0 {
		fpKPI.Defects = []string{"no_false_positives: guard over-blocked benign calls -- " + offenders(res.FalsePositive)}
	}

	fnKPI := scorecard.KPI{
		Key:    "no_false_negatives",
		Group:  "accuracy",
		Score:  100 * (1 - res.FNRate),
		Detail: fmt.Sprintf("%d/%d dangerous call(s) the guard failed to escalate", len(res.FalseNegative), res.DangerousRows),
	}
	if len(res.FalseNegative) > 0 {
		fnKPI.Defects = []string{"no_false_negatives: guard under-blocked dangerous calls -- " + offenders(res.FalseNegative)}
	}

	vacuous := res.BenignRows < minPolarity || res.DangerousRows < minPolarity
	nvKPI := scorecard.KPI{
		Key:    "corpus_nonvacuous",
		Group:  "honesty",
		Score:  100,
		Detail: fmt.Sprintf("corpus carries %d benign + %d dangerous row(s) (>= %d of each required)", res.BenignRows, res.DangerousRows, minPolarity),
	}
	if vacuous {
		nvKPI.Score = 0
		nvKPI.Defects = []string{fmt.Sprintf("corpus_nonvacuous: need >= %d benign and >= %d dangerous rows to certify accuracy, have %d/%d", minPolarity, minPolarity, res.BenignRows, res.DangerousRows)}
	}

	// Subclass drift is advisory only -- both classes escalate, so the confirm
	// pause fires either way and the security floor is intact. Never debt.
	exactKPI := scorecard.KPI{
		Key:    "class_exactness",
		Group:  "accuracy",
		Score:  100,
		Detail: fmt.Sprintf("%d escalation(s) matched the exact subclass", res.DangerousRows-len(res.ClassDrift)-len(res.FalseNegative)),
	}
	if len(res.ClassDrift) > 0 {
		exactKPI.Soft = []string{"class_exactness: escalated but with a drifted subclass (floor intact) -- " + offenders(res.ClassDrift)}
		exactKPI.Detail = fmt.Sprintf("%d escalation(s) drifted subclass (advisory)", len(res.ClassDrift))
	}

	kpis := []scorecard.KPI{fpKPI, fnKPI, nvKPI, exactKPI}

	// Field-complaint intake (#2821): agent-authored over-block appeals folded as
	// an advisory triage queue -- never debt, so it cannot flip ok or red the gate.
	// Appended only when there is at least one appeal, keeping the complaint-free
	// path byte-identical to BuildScorecardFromRows.
	sig := FoldComplaints(complaints)
	if sig.Appeals > 0 {
		kpis = append(kpis, complaintIntakeKPI(sig))
	}

	finding := "guard_accuracy_clean"
	next := "hold the line; add every wild guard misfire to internal/guardaccuracy/testdata/corpus.json as a new labeled row"
	reason := fmt.Sprintf("guard classifier scored %d/%d corpus row(s) correctly: %d false positive(s), %d false negative(s), %d subclass drift(s)",
		res.Correct, res.Total, len(res.FalsePositive), len(res.FalseNegative), len(res.ClassDrift))

	// Worst-first ordering: a false negative (a dangerous call slipped) is the
	// heavier defect than a false positive (benign over-blocked); a vacuous
	// corpus is a measurement failure that precedes either.
	switch {
	case vacuous:
		finding = "guard_accuracy_debt"
		next = "retire worst-first: corpus_nonvacuous -- add labeled rows of the under-represented polarity before trusting the rate"
	case len(res.FalseNegative) > 0:
		finding = "guard_accuracy_debt"
		next = "retire worst-first: no_false_negatives -- the guard let a dangerous call through; tighten the classifier and keep the row"
	case len(res.FalsePositive) > 0:
		finding = "guard_accuracy_debt"
		next = "retire worst-first: no_false_positives -- the guard over-blocked a benign call; match on call structure, not payload mention"
	}

	extra := map[string]any{
		"total_rows":     res.Total,
		"correct":        res.Correct,
		"benign_rows":    res.BenignRows,
		"dangerous_rows": res.DangerousRows,
		"fp_count":       len(res.FalsePositive),
		"fn_count":       len(res.FalseNegative),
		"drift_count":    len(res.ClassDrift),
		"fp_rate":        scorecard.Round3(res.FPRate),
		"fn_rate":        scorecard.Round3(res.FNRate),
	}
	// Surface the field-FP intake in the machine-readable corpus only when it is
	// non-empty, so the complaint-free payload is unchanged. These are counts of an
	// advisory, unconfirmed signal -- deliberately NOT folded into fp_count/fp_rate,
	// which remain the ground-truth corpus numbers an RSI loop drives down.
	if sig.Appeals > 0 {
		extra["field_fp_appeals"] = sig.Appeals
		extra["field_fp_occurrences"] = sig.Occurrences
	}

	p := scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         finding,
		FindingClean:    "guard_accuracy_clean",
		NextAction:      next,
		NextActionClean: "hold the line; add every wild guard misfire to internal/guardaccuracy/testdata/corpus.json as a new labeled row",
		Reason:          reason,
		ExtraCorpus:     extra,
	})
	p.Workspace = root
	return p
}
