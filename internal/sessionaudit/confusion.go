package sessionaudit

// Confusion lens (#3565): a prose-level counterpart to the tool-I/O Behavior lens
// (behavior.go). Every existing detector reads what the agent DID — tool_use inputs
// and errored tool_results — or aggregate token counts. None reads what the agent
// SAID. An agent can thrash SEMANTICALLY — "I misread that", "still failing", "that
// doesn't make sense" — while every tool call succeeds, so the churn/loop/error
// detectors stay silent. This lens reads the assistant's own "text" content blocks
// (the committed narration, NOT the "thinking" scratchpad where reconsideration is
// healthy and expected) for high-precision markers of self-correction, dead-ends,
// and expressed confusion. It is the "read the trace for issues or confusion" rung
// the audit never had.
//
// PRECISION OVER RECALL. An empirical scan of 8,361 real assistant text blocks showed
// the generic words are almost entirely false positives — "what actually exists",
// "let me wait for the background task" — while the genuine markers are rare and
// specific: misread (12), "still broken/red" (14), "let me reconsider" (3), "i was
// wrong" (2). So bare "wait"/"actually" are DROPPED entirely; only reversal-anchored
// or intrinsically-unambiguous phrases count. The base rate is ~0.5%, so a single
// session carrying 3+ markers is a real outlier, not noise. Every marker below either
// survived the false-positive scan or fires only in a self-referential/failure frame.
//
// Like the Behavior lens it stays pure and deterministic: stdlib-only, no clock, no
// RNG, stable ordering, capped top-N — same corpus in, same Confusion out.

import (
	"encoding/json"
	"regexp"
	"sort"
	"strings"
)

const (
	// Confusion category keys (stable wire names).
	catSelfCorrection = "self_correction"
	catDeadEnd        = "dead_end"
	catConfusion      = "confusion"

	// confusionExampleCap bounds a captured marker snippet (runes), matching the
	// normHead discipline the Behavior lens uses for its signatures.
	confusionExampleCap = 140
	// confusionTopMarkers caps the per-session marker roster so one verbose session
	// cannot emit an unbounded list.
	confusionTopMarkers = 10
)

// confusionMarker binds a stable label + category to the regexp that detects it. The
// slice order is fixed so first-seen ordering is deterministic before the count sort.
type confusionMarker struct {
	category string
	label    string
	re       *regexp.Regexp
}

// confusionMarkers is the FIXED, ordered marker table. Each regexp is case-insensitive
// and matched against the whitespace-collapsed text block. Comments record why each
// survives the false-positive test (see the empirical scan noted in the file header).
var confusionMarkers = []confusionMarker{
	// --- self_correction: the agent reverses its own prior claim or approach -------
	// "I misread/misunderstood/misinterpreted/mistook/misjudged ..." — intrinsically
	// self-referential; 12 real hits, zero benign.
	{catSelfCorrection, "misread", regexp.MustCompile(`(?i)\bmis(?:read|understood|understand|interpreted|took|judged)\b`)},
	// "I was wrong / I'm mistaken / my mistake / my bad" — first-person error owning.
	{catSelfCorrection, "i-was-wrong", regexp.MustCompile(`(?i)\bi(?: was| am|'m) (?:wrong|mistaken|incorrect)\b|\bmy (?:mistake|bad)\b|\bi stand corrected\b`)},
	// "let me reconsider/rethink/re-examine/back up/step back/start over" — an explicit
	// restart of the current line of reasoning.
	{catSelfCorrection, "reconsider", regexp.MustCompile(`(?i)\blet me (?:reconsider|rethink|re-?examine|re-?think|back up|step back|revisit that|start over)\b`)},
	// "scratch that / strike that / never mind / disregard that / forget what I said" —
	// an explicit retraction of the immediately preceding text.
	{catSelfCorrection, "scratch-that", regexp.MustCompile(`(?i)\b(?:scratch that|strike that|never ?mind|disregard (?:that|the last)|forget (?:that|what i said)|ignore what i (?:said|wrote))\b`)},
	// Reversal-anchored "wait" / "correction:" — bare "wait" is dropped (all FPs:
	// "let me wait for the build"); only "no wait" or "wait, that/this/I ..." count.
	{catSelfCorrection, "no-wait", regexp.MustCompile(`(?i)\bno,? wait\b|\bwait[,—-] (?:that|this|i|no|the|it|actually|scratch)\b`)},
	// "correction: I ..." / "to correct myself" — anchored to a first-person reversal.
	{catSelfCorrection, "correction", regexp.MustCompile(`(?i)\bcorrection[:—-] (?:i|that|the above|my)\b|\bto correct myself\b`)},
	// "I got that/it wrong/backwards" / "that was backwards" — self-referential error.
	{catSelfCorrection, "got-it-wrong", regexp.MustCompile(`(?i)\bi got (?:that|it|this) (?:wrong|backwards|backward|inverted)\b|\bthat was (?:backwards|backward|inverted)\b`)},

	// --- dead_end: a repair attempt visibly failed or the same failure recurs -------
	// "still failing/fails/broken/red/not working/the same/wrong" — restricted to the
	// failure vocabulary so "still running"/"still pending" (in-progress) do NOT fire.
	{catDeadEnd, "still-broken", regexp.MustCompile(`(?i)\bstill (?:failing|fails|failed|not working|broken|red|erroring|errors|the same|wrong|does ?n'?t work|not fixed)\b`)},
	// "that didn't work / didn't help / didn't fix it" — an attempt that came back empty.
	{catDeadEnd, "didnt-work", regexp.MustCompile(`(?i)\bthat (?:did ?n'?t|did not|does ?n'?t) (?:work|help|fix)\b|\b(?:still|again) did ?n'?t work\b`)},
	// "same error again / no luck / back to square one" — recurrence of a known failure.
	{catDeadEnd, "same-again", regexp.MustCompile(`(?i)\bsame (?:error|failure|problem|issue) (?:again|as before|still)\b|\bno luck\b|\bback to (?:the same|square one)\b`)},

	// --- confusion: expressed puzzlement or surprise at unexpected behavior ---------
	// "I'm confused/puzzled/baffled/stumped/lost" / "not sure why" — first-person
	// puzzlement.
	{catConfusion, "confused", regexp.MustCompile(`(?i)\bi(?: am|'m) (?:confused|puzzled|baffled|stumped|lost)\b|\bnot sure why\b|\bi do ?n'?t understand why\b`)},
	// "doesn't make sense / makes no sense" — an observation the agent cannot reconcile.
	{catConfusion, "no-sense", regexp.MustCompile(`(?i)\b(?:does ?n'?t|do ?n'?t|did ?n'?t) make sense\b|\bmakes no sense\b`)},
	// "that's strange/weird/odd/unexpected/surprising" — surprise at behavior that
	// contradicts the agent's model. The weakest marker; kept because surprise at
	// unexpected output is genuine confusion signal, and the conservative gate below
	// (>= 3 markers/session) prevents an occasional "that's odd" from tripping alone.
	{catConfusion, "unexpected", regexp.MustCompile(`(?i)\bthat(?:'s| is| was)? (?:strange|weird|odd|bizarre|puzzling|surprising|unexpected|suspicious)\b|\bthis is (?:strange|weird|odd|bizarre|unexpected|surprising)\b`)},
}

// ConfusionMarkerRow is one detected marker family with its total count and one
// example snippet — mirrors RepeatFailureRow in the Behavior lens.
type ConfusionMarkerRow struct {
	Category string `json:"category"`
	Label    string `json:"label"`
	Count    int64  `json:"count"`
	Example  string `json:"example"`
}

// Confusion is the per-session prose-friction summary emitted onto Session. Counts are
// TURN-level (a text block is counted once per category regardless of how many markers
// it carries) so a single verbose turn cannot dominate; TotalMarkers keeps the raw hit
// count for detail. Score is turns_with_confusion / text_turns — a base-rate-normalized
// rate that is comparable across sessions of different lengths.
type Confusion struct {
	TextTurns           int64                `json:"text_turns"`
	TurnsWithConfusion  int64                `json:"turns_with_confusion"`
	SelfCorrectionTurns int64                `json:"self_correction_turns"`
	DeadEndTurns        int64                `json:"dead_end_turns"`
	ConfusionTurns      int64                `json:"confusion_turns"`
	TotalMarkers        int64                `json:"total_markers"`
	Score               float64              `json:"score"`
	Markers             []ConfusionMarkerRow `json:"markers"`
}

// confusionLens accumulates prose-friction markers across one transcript walk. It is
// fed one assistant "text" block at a time via noteText, in transcript order.
type confusionLens struct {
	textTurns          int64
	turnsWithConfusion int64
	catTurns           map[string]int64  // category -> turns carrying >=1 marker
	counts             map[string]int64  // label -> total hit count
	order              []string          // first-seen label order (determinism)
	examples           map[string]string // label -> first example snippet
	labelCat           map[string]string // label -> category
}

func newConfusionLens() *confusionLens {
	return &confusionLens{
		catTurns: map[string]int64{},
		counts:   map[string]int64{},
		examples: map[string]string{},
		labelCat: map[string]string{},
	}
}

// noteText folds one assistant "text" content block into the lens. Harness-injected
// "API Error:" blocks are the transport surface, not the model's reasoning, so they are
// excluded from both the numerator and the denominator.
func (l *confusionLens) noteText(raw json.RawMessage) {
	text := strings.TrimSpace(txtStr(raw, 24000))
	if text == "" || strings.HasPrefix(text, "API Error") {
		return
	}
	norm := strings.Join(strings.Fields(text), " ")
	l.textTurns++
	turnCats := map[string]bool{}
	for _, m := range confusionMarkers {
		locs := m.re.FindAllStringIndex(norm, -1)
		if len(locs) == 0 {
			continue
		}
		if _, seen := l.counts[m.label]; !seen {
			l.order = append(l.order, m.label)
			l.labelCat[m.label] = m.category
			l.examples[m.label] = confusionExample(norm, locs[0])
		}
		l.counts[m.label] += int64(len(locs))
		turnCats[m.category] = true
	}
	if len(turnCats) == 0 {
		return
	}
	l.turnsWithConfusion++
	for cat := range turnCats {
		l.catTurns[cat]++
	}
}

func (l *confusionLens) summary() Confusion {
	c := Confusion{
		TextTurns:           l.textTurns,
		TurnsWithConfusion:  l.turnsWithConfusion,
		SelfCorrectionTurns: l.catTurns[catSelfCorrection],
		DeadEndTurns:        l.catTurns[catDeadEnd],
		ConfusionTurns:      l.catTurns[catConfusion],
		Markers:             []ConfusionMarkerRow{},
	}
	for _, label := range l.order {
		n := l.counts[label]
		c.TotalMarkers += n
		c.Markers = append(c.Markers, ConfusionMarkerRow{
			Category: l.labelCat[label],
			Label:    label,
			Count:    n,
			Example:  l.examples[label],
		})
	}
	// Rank by count desc; ties broken by category then label so the order is total and
	// deterministic regardless of first-seen order.
	sort.SliceStable(c.Markers, func(i, j int) bool {
		if c.Markers[i].Count != c.Markers[j].Count {
			return c.Markers[i].Count > c.Markers[j].Count
		}
		if c.Markers[i].Category != c.Markers[j].Category {
			return c.Markers[i].Category < c.Markers[j].Category
		}
		return c.Markers[i].Label < c.Markers[j].Label
	})
	if len(c.Markers) > confusionTopMarkers {
		c.Markers = c.Markers[:confusionTopMarkers]
	}
	if l.textTurns > 0 {
		c.Score = round(float64(l.turnsWithConfusion)/float64(l.textTurns), 3)
	}
	return c
}

// confusionExample returns a whitespace-collapsed window around a marker match, capped
// at confusionExampleCap runes — the human-readable evidence for one marker.
func confusionExample(norm string, loc []int) string {
	start := loc[0] - 40
	if start < 0 {
		start = 0
	}
	end := loc[1] + 60
	if end > len(norm) {
		end = len(norm)
	}
	return normHead(norm[start:end], confusionExampleCap)
}
