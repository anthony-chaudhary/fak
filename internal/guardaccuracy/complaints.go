package guardaccuracy

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// The corpus fold in guardaccuracy.go scores the guard's escalate/don't-escalate
// boundary against a HAND-LABELED corpus -- ground truth. This file adds the
// second evidence stream #2821 names: the agent-authored guard complaints filed
// through `fak complain` (internal/guardcomplaint). A complaint of kind
// "false-positive" or "over-broad" is an agent asserting the reversibility gate
// over-blocked a legitimate call -- a FIELD false positive the seed corpus does
// not yet contain.
//
// Crucially this signal is SUBJECTIVE. As internal/guardcomplaint states, a
// false-positive refusal is byte-identical in the decision journal to a correct
// one -- only the filing agent knows the call was legitimate, and filing a
// complaint is cheap. So the intake is surfaced as an advisory triage queue
// (scorecard Soft), never as hard debt: it points the RSI loop at candidate
// over-blocks to confirm, and the anti-gaming rule (a soft signal can never red
// a gate) means an agent cannot lower the guard's assertiveness score by filing
// appeals. A complaint becomes a measured FP only once it is confirmed and
// promoted into testdata/corpus.json, where the corpus fold scores it as debt.

// overBlockKinds is the closed set of guardcomplaint kinds that bear on the
// escalate/don't-escalate boundary -- an appeal that the gate refused a
// legitimate call. It mirrors the false-positive/over-broad entries in
// guardcomplaint.Kinds; latency/confusing/other are real complaints but not
// accuracy signals (they do not claim the escalate decision itself was wrong),
// so they are excluded from the field-FP intake.
var overBlockKinds = map[string]bool{
	"false-positive": true,
	"over-broad":     true,
}

// FieldComplaint is one agent-authored over-block appeal reduced to the fields
// that bear on accuracy: its kind and its recurrence count. It deliberately
// mirrors guardcomplaint.Complaint/PlanRow WITHOUT importing that package, so
// the accuracy fold stays dependency-light and deterministic; the cmd layer,
// which already reads the complaint issues, translates them into this shape.
type FieldComplaint struct {
	Kind        string `json:"kind"`        // guardcomplaint kind (false-positive, over-broad, latency, ...)
	Summary     string `json:"summary"`     // one-line headline, for the triage-queue offender list
	Occurrences int    `json:"occurrences"` // recurrence weight; a re-filed appeal is a stronger signal (>=1)
}

type complaintFile struct {
	Complaints []FieldComplaint `json:"complaints"`
}

// LoadComplaints parses a complaints file's bytes into field complaints, mirroring
// LoadCorpus. The shape is {"complaints":[{"kind":...,"summary":...,"occurrences":N}]}
// so a caller (a garden/RSI harness that fetched the guard-complaint issues however
// it likes) can feed the intake in as pure data, keeping this package network-free.
func LoadComplaints(data []byte) ([]FieldComplaint, error) {
	var cf complaintFile
	if err := json.Unmarshal(data, &cf); err != nil {
		return nil, err
	}
	return cf.Complaints, nil
}

// ComplaintSignal is the folded field-false-positive intake: the agent-reported
// over-block appeals not yet in the labeled corpus. Appeals counts distinct
// over-block complaints; Occurrences sums their recurrence (a complaint re-filed
// N times is a stronger candidate than a one-off). Offenders is the triage queue,
// ordered most-recurrent-first for a stable, worst-first read.
type ComplaintSignal struct {
	Appeals     int
	Occurrences int
	Offenders   []string // "<summary> (<kind>, xN)" -- most-recurrent first
}

// isOverBlock reports whether a (normalized) complaint kind is an escalate-boundary
// over-block appeal, i.e. field false-positive evidence.
func isOverBlock(kind string) bool {
	return overBlockKinds[strings.ToLower(strings.TrimSpace(kind))]
}

// FoldComplaints reduces field complaints to the over-block intake signal. Only
// over-block kinds are counted; every other kind (and an empty input) folds to a
// zero signal. Occurrences below 1 are clamped to 1, matching guardcomplaint's
// own floor, so a malformed count still registers as one appeal.
func FoldComplaints(cs []FieldComplaint) ComplaintSignal {
	var sig ComplaintSignal
	type row struct {
		text string
		occ  int
	}
	rows := make([]row, 0, len(cs))
	for _, c := range cs {
		if !isOverBlock(c.Kind) {
			continue
		}
		occ := c.Occurrences
		if occ < 1 {
			occ = 1
		}
		sig.Appeals++
		sig.Occurrences += occ
		summary := strings.TrimSpace(c.Summary)
		if summary == "" {
			summary = "(no summary)"
		}
		rows = append(rows, row{
			text: fmt.Sprintf("%s (%s, x%d)", summary, strings.ToLower(strings.TrimSpace(c.Kind)), occ),
			occ:  occ,
		})
	}
	// Most-recurrent first, then lexicographic -- a total order, so the queue is
	// deterministic regardless of the caller's complaint ordering.
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].occ != rows[j].occ {
			return rows[i].occ > rows[j].occ
		}
		return rows[i].text < rows[j].text
	})
	for _, r := range rows {
		sig.Offenders = append(sig.Offenders, r.text)
	}
	return sig
}

// complaintIntakeKPI renders the field-FP intake as an advisory KPI. It scores a
// perfect 100 and carries its queue only in Soft (never Defects), so it NEVER adds
// debt and NEVER reds the accuracy gate -- the deliberate anti-gaming rule for a
// cheap-to-file self-reported signal. Callers append it only when Appeals > 0, so
// the complaint-free path (BuildScorecard / BuildScorecardFromRows) is unchanged.
func complaintIntakeKPI(sig ComplaintSignal) scorecard.KPI {
	k := scorecard.KPI{
		Key:   "field_fp_intake",
		Group: "accuracy",
		Score: 100,
		Detail: fmt.Sprintf("%d agent-appealed over-block(s) (%d occurrence(s)) awaiting triage into the corpus",
			sig.Appeals, sig.Occurrences),
	}
	k.Soft = []string{"field_fp_intake: agent-reported over-block appeals to triage (confirm -> add to testdata/corpus.json) -- " + capOffenders(sig.Offenders, 4)}
	return k
}

// capOffenders joins the pre-ordered queue, keeping the first max entries and
// summarizing the rest as "+N more" -- like offenders() but without re-sorting,
// so the most-recurrent-first order from FoldComplaints is preserved.
func capOffenders(items []string, max int) string {
	if len(items) == 0 {
		return ""
	}
	if len(items) > max {
		return strings.Join(items[:max], "; ") + fmt.Sprintf("; +%d more", len(items)-max)
	}
	return strings.Join(items, "; ")
}
