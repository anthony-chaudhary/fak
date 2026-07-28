package quality

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ActionCompleteness is the actionability oracle for executive reports (#4554):
// every risk or blocker a report surfaces must carry an OWNER and a NEXT ACTION,
// and — where the item itself declares a decision is needed — a DECISION ASK. A
// report that names a blocker with no owner is not status, it is ambient anxiety:
// nothing in it tells the reader who moves next or what they move on. Where
// material-omission (#4552) catches an item that was DROPPED, this oracle catches
// an item that was RAISED but left un-actionable.
//
// Report items travel structured, as a JSON array of item objects in the engine
// trace's Text (the richer-than-Tokens seam the spine documents):
//
//	[
//	  {"kind": "risk", "title": "vendor API deprecation in Q3",
//	   "owner": "dana", "next_action": "draft the migration plan by Friday"},
//	  {"kind": "blocker", "title": "staging migration blocked on DBA review",
//	   "owner": "lee", "next_action": "escalate the review queue",
//	   "needs_decision": true, "decision_ask": "approve contractor DBA hours"}
//	]
//
// Rules (deterministic, documented):
//
//   - Items of kind "risk" or "blocker" (case-insensitive) are ACTIONABLE and
//     must carry a non-placeholder owner and next_action. Other kinds (wins,
//     info, decisions already made) carry no such obligation here.
//   - An actionable item with needs_decision=true must also carry a
//     decision_ask — that is the "where applicable" clause: the item itself
//     declares that a decision is being requested, so the ask must be stated.
//   - A field that is empty, whitespace, or a placeholder ("TBD", "todo",
//     "unassigned", "n/a", "none", "?") is MISSING — an owner of "TBD" is the
//     canonical incomplete blocker, not an owner.
//
// Score = complete actionable items / actionable items; Pass iff Score >=
// Rubric.MinScore (default 1: every raised risk/blocker must be actionable). On
// failure the Detail lists EVERY incomplete item with the exact field(s) it is
// missing, so the fix is a named list, not a hunch. Edge behavior (defined and
// tested): a report with no risk/blocker items passes vacuously; a Text that
// does not parse as a JSON item array fails closed with score 0 — an
// unparseable report cannot prove its blockers are owned.
type ActionCompleteness struct{}

func (ActionCompleteness) Name() string { return "action-completeness" }
func (ActionCompleteness) Kind() string { return "rubric" }

func init() { Register(ActionCompleteness{}) }

// actItem is one structured report item as carried in the engine trace's Text.
// Kind and Title identify it; Owner, NextAction, and (when NeedsDecision)
// DecisionAsk are the completeness obligations this oracle enforces on risks
// and blockers.
type actItem struct {
	Kind          string `json:"kind"`
	Title         string `json:"title"`
	Owner         string `json:"owner,omitempty"`
	NextAction    string `json:"next_action,omitempty"`
	NeedsDecision bool   `json:"needs_decision,omitempty"`
	DecisionAsk   string `json:"decision_ask,omitempty"`
}

// actItemsText renders items as the JSON array an action-completeness trace
// carries in Text — the authoring surface for cases and scripted engines.
func actItemsText(items []actItem) string {
	b, err := json.Marshal(items)
	if err != nil {
		// []actItem of plain strings/bools cannot fail to marshal; keep the
		// helper total anyway.
		return "[]"
	}
	return string(b)
}

// actParseItems parses a report's Text as the JSON item array. The error is
// surfaced (not swallowed) so Judge can fail closed on an unparseable report.
func actParseItems(text string) ([]actItem, error) {
	var items []actItem
	if err := json.Unmarshal([]byte(strings.TrimSpace(text)), &items); err != nil {
		return nil, err
	}
	return items, nil
}

// actActionable reports whether an item's kind obliges it to carry an owner
// and a next action.
func (it actItem) actActionable() bool {
	switch strings.ToLower(strings.TrimSpace(it.Kind)) {
	case "risk", "blocker":
		return true
	}
	return false
}

// actMissingFields returns the human names of the completeness fields this
// item lacks, in a fixed order (owner, next action, decision ask).
func (it actItem) actMissingFields() []string {
	var missing []string
	if actBlankField(it.Owner) {
		missing = append(missing, "owner")
	}
	if actBlankField(it.NextAction) {
		missing = append(missing, "next action")
	}
	if it.NeedsDecision && actBlankField(it.DecisionAsk) {
		missing = append(missing, "decision ask")
	}
	return missing
}

// actLabel names an item for a failure Detail: its quoted title, or its index
// when the title itself is blank.
func (it actItem) actLabel(index int) string {
	kind := strings.ToLower(strings.TrimSpace(it.Kind))
	if t := strings.TrimSpace(it.Title); t != "" {
		return fmt.Sprintf("%s %q", kind, t)
	}
	return fmt.Sprintf("%s item %d", kind, index)
}

// actPlaceholders are the non-answers that do not count as a filled field: a
// blocker owned by "TBD" has no owner.
var actPlaceholders = map[string]bool{
	"tbd": true, "todo": true, "unassigned": true,
	"n/a": true, "na": true, "none": true, "?": true,
}

// actBlankField reports whether a field value is missing: empty, whitespace,
// or a placeholder non-answer (case-insensitive).
func actBlankField(s string) bool {
	s = strings.ToLower(strings.TrimSpace(s))
	return s == "" || actPlaceholders[s]
}

// Judge parses the engine report's structured items and checks every risk and
// blocker for an owner, a next action, and — when the item flags
// needs_decision — a decision ask. Score is the fraction of actionable items
// that are complete; on failure Detail lists each incomplete item with the
// exact fields it is missing.
func (ActionCompleteness) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "action-completeness", Kind: "rubric", Pass: true, Score: 1}
	items, err := actParseItems(eng.Text)
	if err != nil {
		return rubricFail(v, fmt.Sprintf("report items not parseable as a JSON item array: %v", err))
	}
	actionable, complete := 0, 0
	var incomplete []string
	for i, it := range items {
		if !it.actActionable() {
			continue
		}
		actionable++
		missing := it.actMissingFields()
		if len(missing) == 0 {
			complete++
			continue
		}
		incomplete = append(incomplete,
			fmt.Sprintf("%s missing %s", it.actLabel(i), strings.Join(missing, ", ")))
	}
	if actionable == 0 {
		v.Detail = "no risk/blocker items in the report; nothing to require actions for"
		return v
	}
	min, short := rubricScore(&v, c, complete, actionable)
	if short {
		v.Detail = fmt.Sprintf("action-completeness %.2f < %.2f (%d/%d actionable items complete); incomplete: %s",
			v.Score, min, complete, actionable, strings.Join(incomplete, "; "))
		return v
	}
	if len(incomplete) > 0 {
		v.Detail = fmt.Sprintf("action-completeness %.2f >= %.2f (%d/%d complete; tolerated incomplete: %s)",
			v.Score, min, complete, actionable, strings.Join(incomplete, "; "))
		return v
	}
	v.Detail = fmt.Sprintf("all %d risk/blocker item(s) carry an owner, a next action, and any required decision ask", actionable)
	return v
}
