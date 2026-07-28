package quality

import (
	"fmt"
	"strings"
)

// MaterialOmission is the omission-side complement to GroundingRubric (#4552):
// where grounding catches FABRICATED claims, this oracle catches DROPPED ones. A
// status report that reads fluently but silently omits a material win, risk,
// blocker, or decision must fail a gate, not merely read fine. The case declares
// its material items via Rubric.Required — either plain must-mention entries, or
// categorized entries built by MaterialItems.Rubric — and the oracle scores the
// fraction present in the engine text; Pass iff Score >= Rubric.MinScore (default
// 1: nothing material may be omitted). On failure the Detail names EVERY omitted
// item (with its category when declared), so the fix is a list, not a hunch.
type MaterialOmission struct{}

func (MaterialOmission) Name() string { return "material-omission" }
func (MaterialOmission) Kind() string { return "rubric" }

func (MaterialOmission) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "material-omission", Kind: "rubric", Pass: true, Score: 1}
	items := materialItems(c.Rubric.Required)
	if len(items) == 0 {
		v.Detail = "no material items declared; nothing to omit"
		return v
	}
	text := strings.ToLower(eng.Text)
	present := 0
	var omitted []MaterialItem
	for _, it := range items {
		if strings.Contains(text, strings.ToLower(it.Text)) {
			present++
		} else {
			omitted = append(omitted, it)
		}
	}
	min, short := rubricScore(&v, c, present, len(items))
	if short {
		v.Detail = fmt.Sprintf("material-omission score %.2f < %.2f; omitted: %s",
			v.Score, min, describeMaterialItems(omitted))
		return v
	}
	v.Detail = fmt.Sprintf("material-omission score %.2f >= %.2f; %d/%d material items covered",
		v.Score, min, present, len(items))
	return v
}

func init() { Register(MaterialOmission{}) }

// The four material categories a report-quality case may tag items with. They are
// the encoding vocabulary of MaterialItems.Required and the parse vocabulary of
// materialItems; any other prefix is treated as literal item text.
const (
	categoryWin      = "win"
	categoryRisk     = "risk"
	categoryBlocker  = "blocker"
	categoryDecision = "decision"
)

// MaterialItem is one piece of material content a report must cover: its text
// (matched case-insensitively as a substring of the engine output) and an
// optional category used when reporting an omission.
type MaterialItem struct {
	Category string `json:"category,omitempty"`
	Text     string `json:"text"`
}

// MaterialItems is the categorized declaration of everything material a report
// must cover: wins, risks, blockers, and decisions. It is the authoring surface
// for material-omission cases — build one, then carry it on the case via Rubric.
type MaterialItems struct {
	Wins      []string
	Risks     []string
	Blockers  []string
	Decisions []string
}

// Required flattens the categorized items into Rubric.Required entries using the
// "category: item" encoding materialItems parses back out. Order is
// deterministic: wins, risks, blockers, decisions, each in declaration order.
// Empty item strings are dropped — an empty item is not material content.
func (m MaterialItems) Required() []string {
	var out []string
	add := func(cat string, items []string) {
		for _, it := range items {
			if it != "" {
				out = append(out, cat+": "+it)
			}
		}
	}
	add(categoryWin, m.Wins)
	add(categoryRisk, m.Risks)
	add(categoryBlocker, m.Blockers)
	add(categoryDecision, m.Decisions)
	return out
}

// Rubric is the helper constructor for a case judged by material-omission: it
// carries the categorized items in Rubric.Required and the pass threshold in
// MinScore (0 means the default — nothing material may be omitted). A case that
// ALSO names grounding-rubric should carry plain entries instead, since grounding
// matches each Required entry literally, category prefix and all.
func (m MaterialItems) Rubric(minScore float64) RubricSpec {
	return RubricSpec{Required: m.Required(), MinScore: minScore}
}

// materialItems parses Rubric.Required entries into material items. An entry with
// a known "category:" prefix (win, risk, blocker, decision — case-insensitive)
// becomes a categorized item; any other non-empty entry is an uncategorized item
// taken verbatim, so plain Required lists work unchanged.
func materialItems(required []string) []MaterialItem {
	items := make([]MaterialItem, 0, len(required))
	for _, r := range required {
		r = strings.TrimSpace(r)
		if r == "" {
			continue
		}
		if cat, text, ok := splitMaterialCategory(r); ok {
			items = append(items, MaterialItem{Category: cat, Text: text})
			continue
		}
		items = append(items, MaterialItem{Text: r})
	}
	return items
}

// splitMaterialCategory splits "category: item" when the prefix is one of the
// four known categories and the item text is non-empty; anything else is not an
// encoded entry (a colon inside ordinary item text stays literal).
func splitMaterialCategory(entry string) (cat, text string, ok bool) {
	i := strings.Index(entry, ":")
	if i < 0 {
		return "", "", false
	}
	cat = strings.ToLower(strings.TrimSpace(entry[:i]))
	switch cat {
	case categoryWin, categoryRisk, categoryBlocker, categoryDecision:
	default:
		return "", "", false
	}
	text = strings.TrimSpace(entry[i+1:])
	if text == "" {
		return "", "", false
	}
	return cat, text, true
}

// describeMaterialItems renders omitted items for a failure Detail: categorized
// items as `category "text"`, uncategorized as the bare quoted text, in the
// case's declaration order.
func describeMaterialItems(items []MaterialItem) string {
	parts := make([]string, len(items))
	for i, it := range items {
		if it.Category != "" {
			parts[i] = fmt.Sprintf("%s %q", it.Category, it.Text)
		} else {
			parts[i] = fmt.Sprintf("%q", it.Text)
		}
	}
	return strings.Join(parts, ", ")
}
