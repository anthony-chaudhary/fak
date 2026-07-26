// Package modedebt is the CONSUMER half of the mode-debt scorer/dispatcher pair
// (epic #4397, under harness-native #2387 / permission regimes #2389). The sibling
// scorer leaf enumerates each permission dial with a HARD/soft grade and a
// lifted/un-lifted state and emits it as a stable JSON scorecard; this package
// reads that scorecard and maps every HARD un-lifted dial onto the EXISTING
// internal/dogfoodissues backlog bridge -- the same ToActionItem -> BuildPlan ->
// Sync path the propagation-/unwired-/qa-process-debt siblings established, so
// there is no new gh-issue code here.
//
// This leaf owns only the CONSUMER view of the scorecard contract: the struct the
// dispatcher unmarshals and the HARD-un-lifted selection. It does NOT compute the
// grades or the lifted state -- that is the scorer leaf's job. If the scorecard is
// absent, the dispatcher fails closed rather than inventing dials.
package modedebt

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

// Schema is the stable schema tag the scorer stamps on the scorecard JSON and the
// dispatcher checks for provenance. An empty schema is tolerated (older scorer
// snapshots) so long as the dials decode.
const Schema = "fak.mode-debt-scorecard.v1"

// DebtKey is the headline HARD integer the dispatcher folds onto each ActionItem.
const DebtKey = "mode_debt"

// Permission-regime routing targets. Filed issues carry these so they land on the
// permission-regime backlog (#2389 / #2405) rather than a stray lane, per #4416.
const (
	// TargetRegime is the permission-regimes program epic (#2389).
	TargetRegime = "#2389"
	// TargetRegimeBacklog is the concrete permission-regime backlog epic (#2405).
	TargetRegimeBacklog = "#2405"
	// ParentEpic is the mode-debt scorer/dispatcher pair epic (#4397).
	ParentEpic = "#4397"
)

// Dial is one permission dial as scored by the sibling scorer leaf: a slug, an
// optional human name, a HARD/soft grade, and whether the regime has LIFTED it.
// The dispatcher selects on Grade+Lifted only; it never re-derives the regime.
type Dial struct {
	Slug   string `json:"slug"`
	Name   string `json:"name,omitempty"`
	Grade  string `json:"grade"`
	Lifted bool   `json:"lifted"`
	Regime string `json:"regime,omitempty"`
	Detail string `json:"detail,omitempty"`

	// The census half (#4415). The dispatcher selects on Grade+Lifted+Slug only, so
	// every field below is additive and omitempty: an older scorecard that carries
	// none of them still decodes and still dispatches identically. They exist so the
	// `fak mode-debt` dump can show WHY a dial got its grade (Criteria), WHERE the
	// dial lives (Surface/File/Line), and which reviewed safety hold excluded it.
	Surface  string   `json:"surface,omitempty"`
	File     string   `json:"file,omitempty"`
	Line     int      `json:"line,omitempty"`
	Criteria Criteria `json:"criteria"`
	Excluded string   `json:"excluded,omitempty"`
}

// Scorecard is the stable JSON payload the scorer leaf emits and this dispatcher
// consumes. Only Dials is load-bearing for selection.
type Scorecard struct {
	Schema string `json:"schema,omitempty"`
	Dials  []Dial `json:"dials"`

	// Roll-up counts derived by Score (#4415) so no consumer re-folds the grades.
	// Debt is the headline DebtKey integer: RANKED debt only (Hard+Soft), so a CLEAN
	// dial and a harness-held safety dial both contribute zero.
	Clean   int `json:"clean"`
	Hard    int `json:"hard"`
	Soft    int `json:"soft"`
	NotDebt int `json:"not_debt"`
	Debt    int `json:"debt"`
}

// IsHard reports whether the dial is graded HARD (case-insensitive), i.e. a dial
// the permission regime forces rather than one a worker can soft-approve.
func (d Dial) IsHard() bool {
	return strings.EqualFold(strings.TrimSpace(d.Grade), "HARD")
}

// Display is the human name for the dial (Name when the scorer set one, else the
// slug), used in the issue title and body.
func (d Dial) Display() string {
	if s := strings.TrimSpace(d.Name); s != "" {
		return s
	}
	return strings.TrimSpace(d.Slug)
}

// Key is the CONTENT-STABLE dedup identity (no run-id / timestamp): mode-debt plus
// the dial slug. A re-run folds onto the same issue, and a recurrence after a
// "lifted" close re-files the same key.
func (d Dial) Key() string {
	return "mode-debt/" + slug(d.Slug)
}

// LoadScorecard reads and decodes the scorer leaf's scorecard JSON. A missing or
// unparseable file is an error, not an empty scorecard: absence must fail closed
// rather than silently dispatch nothing.
func LoadScorecard(path string) (Scorecard, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Scorecard{}, err
	}
	var sc Scorecard
	if err := json.Unmarshal(b, &sc); err != nil {
		return Scorecard{}, fmt.Errorf("scorecard %s: %w", path, err)
	}
	return sc, nil
}

// SelectHardUnlifted returns every HARD un-lifted dial, sorted by slug so the
// dispatcher's cap and dedup are stable across runs. Soft dials and already-lifted
// dials produce no candidate -- a CLEAN (fully lifted) regime yields an empty set.
func SelectHardUnlifted(sc Scorecard) []Dial {
	var out []Dial
	for _, d := range sc.Dials {
		if d.IsHard() && !d.Lifted && strings.TrimSpace(d.Slug) != "" {
			out = append(out, d)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Key() < out[j].Key() })
	return out
}

// ToActionItem maps a HARD un-lifted dial onto the EXISTING dogfoodissues.ActionItem
// so the GitHub-issue half is pure reuse (IssueBody / BuildPlanWithOptions / Sync).
// evidencePath is the scorecard JSON the issue cites. ParentRef and Coordination
// route the filed issue to the permission-regime backlog (#2389 / #2405).
func (d Dial) ToActionItem(evidencePath string) dogfoodissues.ActionItem {
	name := d.Display()
	regime := strings.TrimSpace(d.Regime)
	regimeNote := ""
	if regime != "" {
		regimeNote = " (regime `" + regime + "`)"
	}
	detail := strings.TrimSpace(d.Detail)
	if detail == "" {
		detail = "The scorer graded this dial HARD and un-lifted."
	}
	return dogfoodissues.ActionItem{
		Key:          d.Key(),
		Title:        "lift the HARD un-lifted permission dial " + name,
		SourceProbe:  "mode-debt-scorecard",
		ScoreName:    "dial_state",
		Score:        "HARD/un-lifted",
		Grade:        "F",
		DebtName:     DebtKey,
		DebtCount:    1,
		EvidencePath: evidencePath,
		NextAction:   "Lift the HARD permission dial `" + name + "` so the regime no longer forces it, then re-run the mode-debt scorecard.",
		Finding:      d.Key(),
		ParentRef: fmt.Sprintf("permission regimes %s / regime backlog %s (epic %s)",
			TargetRegime, TargetRegimeBacklog, ParentEpic),
		CurrentState: fmt.Sprintf("The mode-debt scorecard grades permission dial `%s`%s HARD and un-lifted: %s The debt stays invisible in the backlog until a tracked issue owns lifting it, which is what this generated issue does.",
			name, regimeNote, detail),
		WhyNow:         "A HARD un-lifted dial is inert until something turns it into a tracked, assignable backlog item against the permission-regime epics " + TargetRegime + " / " + TargetRegimeBacklog + "; filing it now closes the scorer->dispatcher loop so the regime backlog converges on re-runs.",
		WorkingSpine:   "Lift the single named permission dial `" + name + "` on the smallest witnessed path -- do not redesign the regime; retire exactly this dial's HARD un-lifted state.",
		WorkUnit:       "leaf",
		ExpectedSteps:  5,
		Assumptions:    []string{"The mode-debt scorer's grade/lifted flags for this dial are still current when the worker starts."},
		ConfusionRisks: []string{"Do not batch multiple dials into one worker issue; one issue owns one dial gap.", "Lifting the dial is the fix; do not merely re-grade it soft to make the scorecard clean."},
		Coordination:   []string{"Targets the permission-regime backlog " + TargetRegime + " / " + TargetRegimeBacklog + " (parent epic " + ParentEpic + "); coordinate the lift there, not a stray lane.", "One generated issue owns one permission dial."},
		Trigger:        "The mode-debt scorecard reports permission dial `" + slug(d.Slug) + "` as HARD and un-lifted.",
		BatchPolicy:    "One issue per HARD un-lifted dial, keyed on the content-stable `" + d.Key() + "` marker; reruns update by marker and converge, capped at the family --cap so a noisy scorecard cannot spam the backlog.",
		InScope:        "Lift the permission dial `" + name + "` so the regime no longer forces it, and prove the mode-debt scorecard re-scores it lifted (or soft).",
		OutOfScope:     "Do not change the mode-debt scorer's dial roster or grade thresholds to make the scorecard clean, and do not touch other dials in the same change.",
		DoneCondition:  "A re-run of the mode-debt scorecard no longer reports dial `" + slug(d.Slug) + "` as HARD un-lifted (the `" + d.Key() + "` gap is gone).",
		Witness:        "The mode-debt scorecard re-scores dial `" + slug(d.Slug) + "` as lifted or soft, with the regime change that lifted it committed.",
		AcceptanceGate: "go build ./... && the mode-debt scorecard reports dial `" + slug(d.Slug) + "` lifted",
		Lane:           "",
		Paths:          []string{"internal/modedebt/**"},
		Labels:         []string{"mode-debt", "permission-regime"},
		BoundaryNotes:  []string{"Public permission-regime dial only; no private or lab-local evidence."},
		ClosureBinding: "Resolving commit cites `#N` in the subject and carries a matching `(fak <leaf>)` trailer for the dial's lane.",
	}
}

// ActionItems maps a set of dials onto dogfoodissues.ActionItems, the input to the
// backlog bridge.
func ActionItems(dials []Dial, evidencePath string) []dogfoodissues.ActionItem {
	items := make([]dogfoodissues.ActionItem, 0, len(dials))
	for _, d := range dials {
		items = append(items, d.ToActionItem(evidencePath))
	}
	return items
}

// slug lowercases s and collapses any run of non-[a-z0-9] into a single dash,
// trimming leading/trailing dashes, so a dial name yields a stable key part.
func slug(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "dial"
	}
	return out
}
