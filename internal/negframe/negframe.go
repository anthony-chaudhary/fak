// Package negframe is the negation-lexicon + reframe-classification stick.
//
// The lesson it generalizes: prose that steers an agent lands better when it LEADS WITH THE
// AFFORDANCE -- the action to take -- instead of the prohibition. "don't forget to stamp the
// commit" tells the reader what to avoid; "remember to stamp the commit" tells them what to
// do. The first spends a clause building a negative the reader must invert; the second is the
// instruction already in positive form. Across a whole steer-prose corpus (the guard runtime
// prose injected into every session, AGENTS.md, the skills, the refusal vocabulary) that
// inversion tax compounds -- so this card measures it and, where a mechanical rewrite is
// unambiguous, hands back the positive reframe.
//
// It is a TEXT-READING card: the prose IS the data. Findings are re-derived from the bytes on
// disk, so the card cannot be gamed by editing a JSON file -- only by reframing the prose.
// The scoring fold/grade/markdown machinery lives in pkg/scorecard; this package holds the
// negation lexicon, the reframe rules, per-document classification, and the diff helper the
// ratchet (#3545) gates on.
//
// Two confidence tiers keep the debt integer honest and the gate sane:
//
//   - A MECHANICAL finding matches a reframe rule that carries an unambiguous positive rewrite
//     (e.g. "don't forget to X" -> "remember to X"). These are the HARD debt: cheap, concrete
//     wins with a suggestion attached.
//   - A JUDGEMENT finding is negatively framed but has no confident mechanical rewrite (a bare
//     "never merge without review"); it is SOFT -- advisory, never gating -- because the cheap
//     way to move a soft signal is prose spam, and a soft signal must not be able to red a gate.
package negframe

import (
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id, in the shared fak-<name>-scorecard/<v> shape so the
// fold and any downstream consumer read this card like every other.
const Schema = "fak-negframe-scorecard/1"

// DebtKey is the headline integer the control-pane folds: the count of MECHANICAL (confidently
// reframable) negatives outstanding across the scored corpus.
const DebtKey = "negframe_debt"

// Category is the class of negative framing a finding falls under. Each carries a one-line hint
// for the JUDGEMENT tier -- the positive shape to aim the prose at when no mechanical rewrite
// applies.
type Category string

const (
	// Prohibition: an imperative that says what NOT to do ("don't", "do not", "never", "avoid").
	// The positive shape is the action to take instead.
	Prohibition Category = "prohibition"
	// Absence: framing by what is MISSING ("no", "without", "lack of", "fails to").
	// The positive shape names what is present or required.
	Absence Category = "absence"
	// Refusal: a boundary framed as a wall ("not allowed", "forbidden", "may not", "refuse to").
	// The positive shape states the permitted path first, then the boundary.
	Refusal Category = "refusal"
	// Hedge: a property asserted by double negative ("not un-", "not impossible", "not without").
	// The positive shape asserts the property directly.
	Hedge Category = "hedge"
	// Exception: selection narrowed by an exception frame ("only", "except").
	// The positive shape enumerates the selected or remaining set.
	Exception Category = "exception"
)

// categoryHint is the JUDGEMENT-tier reframe hint shown when a finding has no mechanical rewrite.
var categoryHint = map[Category]string{
	Prohibition: "lead with the action to take instead of the one to avoid",
	Absence:     "name what is present or required, not what is missing",
	Refusal:     "state the permitted path first, then the boundary",
	Hedge:       "assert the property directly instead of by double negative",
	Exception:   "enumerate the selected or remaining positive set",
}

// Hint returns the positive-shape hint for a category (empty for an unknown one).
func Hint(c Category) string { return categoryHint[c] }

// Finding is one negatively-framed span located in a document.
type Finding struct {
	Path     string        `json:"path"`
	Line     int           `json:"line"` // 1-based line number within Path
	Category Category      `json:"category"`
	Span     string        `json:"span"`    // exact source bytes matched on Line
	Text     string        `json:"text"`    // clipped source line containing Span
	Suggest  string        `json:"suggest"` // the positive rewrite (mechanical tier) or "" (judgement)
	Hint     string        `json:"hint"`    // the category hint (judgement tier) or "" (mechanical)
	Tier     BroadcastTier `json:"broadcast_tier"`
	Weight   int           `json:"broadcast_weight"`
}

// Mechanical reports whether this finding carries a confident positive rewrite (HARD debt).
// A judgement-tier finding (Suggest == "") is advisory only.
func (f Finding) Mechanical() bool { return f.Suggest != "" }

// reframeRule is one entry in the lexicon: a case-insensitive pattern for a negatively-framed
// span, the category it classifies to, and -- for the mechanical tier -- a replacement template
// applied to the matched span to produce the positive rewrite. An empty Template marks a
// judgement-tier rule (detected and classified, but reframed only with the category hint).
type reframeRule struct {
	Pattern  *regexp.Regexp
	Category Category
	Template string // regexp replacement ($1.. refer to Pattern's capture groups); "" == judgement tier
}

// rules is the negation lexicon, ordered most-specific-first so a mechanical rule ("don't
// forget to X") is tried before the generic prohibition rule that would also match "don't X".
// classifyLine stops at the first rule that matches a given span, so ordering is the precedence.
//
// The mechanical rules are deliberately high-precision: each names a fixed idiom whose positive
// inverse is unambiguous. The generic rules at the tail catch the broader negative framings for
// the judgement tier, where the reframe is a human call and the card only points.
var rules = []reframeRule{
	// --- mechanical tier: fixed idioms with an unambiguous positive inverse -------------------
	{regexp.MustCompile(`(?i)\bdo\s+not\s+forget\s+to\s+(\w+)`), Prohibition, "remember to $1"},
	{regexp.MustCompile(`(?i)\bdon'?t\s+forget\s+to\s+(\w+)`), Prohibition, "remember to $1"},
	{regexp.MustCompile(`(?i)\bdo\s+not\s+hesitate\s+to\s+(\w+)`), Prohibition, "feel free to $1"},
	{regexp.MustCompile(`(?i)\bdon'?t\s+hesitate\s+to\s+(\w+)`), Prohibition, "feel free to $1"},
	{regexp.MustCompile(`(?i)\bno\s+need\s+to\s+(\w+)`), Absence, "you can skip $1"},
	// The double-negative reframe ("not un-X" -> "X") holds ONLY when "un" is the negating prefix
	// of a genuine antonym. A bare `not\s+un(\w+)` also fires on words where "un" is part of the
	// ROOT -- "not unique", "not universal", "not unless", "does not unlock" -- and emits garbage
	// ("less", "lock") at the MECHANICAL (gating) tier, so a real "does not unlock" would falsely
	// red the ratchet. The stem is therefore an explicit allowlist of adjectives that actually
	// negate with "un-"; anything outside it stays un-suggested (and is caught, if at all, by the
	// generic judgement rules). Extend the list rather than loosening it back to `\w+`.
	{regexp.MustCompile(`(?i)\bnot\s+un(readable|usual|common|clear|likely|necessary|important|safe|able|available|aware|certain|reasonable|realistic|helpful|related|expected|documented|reachable|recoverable|ambiguous|desirable|acceptable|familiar|known|wise|fair|kind|true|real|stable|bounded|intended|wanted|used|changed|tested|defined|limited|restricted|planned|warranted|justified|founded|biased|happy|healthy|even)\b`), Hedge, "$1"},
	{regexp.MustCompile(`(?i)\bmake\s+sure\s+(?:that\s+)?you\s+do\s+not\s+(\w+)`), Prohibition, "make sure you avoid $1ing"},

	// --- judgement tier: negatively framed, reframe is a human call ---------------------------
	{regexp.MustCompile(`(?i)\bnever\b`), Prohibition, ""},
	{regexp.MustCompile(`(?i)\bdo\s+not\b`), Prohibition, ""},
	{regexp.MustCompile(`(?i)\bdon'?t\b`), Prohibition, ""},
	{regexp.MustCompile(`(?i)\bavoid\b`), Prohibition, ""},
	{regexp.MustCompile(`(?i)\bnot\s+allowed\b`), Refusal, ""},
	{regexp.MustCompile(`(?i)\bforbidden\b`), Refusal, ""},
	{regexp.MustCompile(`(?i)\bmay\s+not\b`), Refusal, ""},
	{regexp.MustCompile(`(?i)\brefuse\s+to\b`), Refusal, ""},
	{regexp.MustCompile(`(?i)\bfails?\s+to\b`), Absence, ""},
	{regexp.MustCompile(`(?i)\bwithout\b`), Absence, ""},
}

// Categories lists the categories in a stable order (used for per-category KPI folding).
var Categories = []Category{Prohibition, Absence, Refusal, Hedge, Exception}

// Classify locates every negatively-framed span in text, tagging each with path for the
// finding's provenance. Fenced code blocks (``` / ~~~) and blank lines are skipped -- the card
// gardens PROSE, not code samples that legitimately show a `!x` or a "don't" in a string. Each
// line yields at most one finding per rule position so a line with two distinct negatives is
// two findings, but the same idiom is not double-counted by an overlapping generic rule.
func Classify(path, text string) []Finding {
	var out []Finding
	tier := broadcastTierForPath(path)
	inFence := false
	for i, raw := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(raw)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			continue
		}
		if inFence || trimmed == "" {
			continue
		}
		for _, f := range classifyLine(raw) {
			f.Path = path
			f.Line = i + 1
			f.Tier = tier
			f.Weight = tier.Weight()
			out = append(out, f)
		}
	}
	return out
}

// classifyLine returns the findings in a single line. It walks the rules in precedence order and,
// for each match, records the span and marks its byte range consumed so a later, more generic
// rule cannot re-report the same idiom (the "don't forget to" span is not also counted as a bare
// "don't"). Overlap is resolved by first-match-wins, which -- given the most-specific-first rule
// order -- keeps the mechanical rewrite when one applies.
func classifyLine(line string) []Finding {
	type span struct{ start, end int }
	var claimed []span
	overlaps := func(s, e int) bool {
		for _, c := range claimed {
			if s < c.end && c.start < e {
				return true
			}
		}
		return false
	}
	type placed struct {
		start int
		f     Finding
	}
	var found []placed
	for _, r := range rules {
		for _, loc := range r.Pattern.FindAllStringIndex(line, -1) {
			s, e := loc[0], loc[1]
			if overlaps(s, e) {
				continue
			}
			claimed = append(claimed, span{s, e})
			f := Finding{
				Category: r.Category,
				Span:     line[s:e],
				Text:     scorecard.Clip(line, 100),
			}
			if r.Template != "" {
				f.Suggest = r.Pattern.ReplaceAllString(line[s:e], r.Template)
			} else {
				f.Hint = categoryHint[r.Category]
			}
			found = append(found, placed{start: s, f: f})
		}
	}
	// Report findings left-to-right by their position in the line for deterministic ordering,
	// independent of the rule-precedence order they were discovered in.
	sort.SliceStable(found, func(a, b int) bool { return found[a].start < found[b].start })
	out := make([]Finding, len(found))
	for i, p := range found {
		out[i] = p.f
	}
	return out
}

// DocResult is the per-document classification: its findings, the sentence-ish denominator the
// density score divides by, and the derived positivity value (100 == no reframable negatives).
type DocResult struct {
	Path       string    `json:"path"`
	Sentences  int       `json:"sentences"`
	Mechanical int       `json:"mechanical"` // count of HARD (confidently reframable) findings
	Judgement  int       `json:"judgement"`  // count of SOFT (advisory) findings
	Findings   []Finding `json:"findings"`
}

// Negatives is the total finding count (both tiers) in a document.
func (d DocResult) Negatives() int { return d.Mechanical + d.Judgement }

// ScoreDoc classifies text and folds it into a per-document result. Sentences is a coarse
// denominator (non-empty prose lines outside code fences) -- enough to normalize a long doc's
// negative count against its size without a full NLP sentence splitter, which would add
// dependency weight for a density estimate the card only needs to be monotone.
func ScoreDoc(path, text string) DocResult {
	d := DocResult{Path: path, Findings: Classify(path, text)}
	for _, f := range d.Findings {
		if f.Mechanical() {
			d.Mechanical++
		} else {
			d.Judgement++
		}
	}
	d.Sentences = countProseLines(text)
	return d
}

// countProseLines counts non-blank lines outside code fences -- the density denominator.
func countProseLines(text string) int {
	n := 0
	inFence := false
	for _, raw := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(raw)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			continue
		}
		if inFence || trimmed == "" {
			continue
		}
		n++
	}
	return n
}

// NewFindings returns the mechanical-tier findings present in after whose (category, normalized
// text) does NOT appear in before -- the reframable negatives a change INTRODUCED. This is the
// ratchet primitive (#3545): a diff-scoped gate reds only on newly-added confidently-reframable
// negatives, so a change is never blocked by debt it did not create. Only the mechanical tier is
// diffed, keeping the gate on the unambiguous-win class.
func NewFindings(path, before, after string) []Finding {
	seen := map[string]bool{}
	for _, f := range Classify(path, before) {
		if f.Mechanical() {
			seen[findingKey(f)] = true
		}
	}
	var out []Finding
	for _, f := range Classify(path, after) {
		if f.Mechanical() && !seen[findingKey(f)] {
			out = append(out, f)
		}
	}
	return out
}

// findingKey normalizes a finding to a line-independent identity (category + collapsed text) so
// the diff compares WHAT the negative is, not where it moved to when surrounding lines shifted.
func findingKey(f Finding) string {
	return string(f.Category) + "\x1f" + strings.ToLower(strings.Join(strings.Fields(f.Text), " "))
}
