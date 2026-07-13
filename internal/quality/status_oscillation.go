package quality

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// oscStatusOscillation is the status-oscillation oracle for executive reports
// (#4560): a workstream's status must not flip report-to-report without an
// accompanying rationale. The failure mode this catches is the confidence-
// eroding green -> red -> green oscillation where each week's report simply
// asserts a new color and never explains what changed — the report reads fine
// in isolation, and only the report-to-report axis exposes the churn.
//
// The PREVIOUS report's statuses travel in the case as a status block in
// Reference.Text, one workstream per line:
//
//	workstream | status
//	workstream | older -> newer            (history, oldest first)
//
// The LAST status in a line's "->" chain is what the previous report
// asserted; earlier entries are that workstream's declared history and let
// the oracle name a round trip (green -> red -> green) as an oscillation
// rather than a bare flip. Lines that do not parse (no "|", empty name, a
// status outside the vocabulary) are skipped, never panicked on.
//
// Statuses are a closed, case-insensitive vocabulary folded to a canonical
// color: green ("green", "on track"), yellow ("yellow", "amber", "at risk"),
// red ("red", "blocked", "off track"). All matching is word-bounded so "red"
// never matches inside "delivered".
//
// The CURRENT status of a workstream is read from eng.Text: the first
// sentence mentioning the workstream that carries a status phrase decides,
// and the LAST status phrase in that sentence is the current status — so
// "moved from red back to green" reads as green. A workstream the report
// never gives a checkable status for is skipped here (dropping an item is
// material-omission's concern, not oscillation's).
//
// Flip rule: a current status differing from the previous one is a flip, and
// a flip must carry a rationale — one of a closed set of explanation cues
// ("because", "due to", "after", "root cause", ...) word-bounded in the flip
// sentence or the immediately following sentence. A flip WITH a rationale is
// legitimate change narration and passes; a flip WITHOUT one is the defect.
//
// Score = (unchanged + explained) / checked; Pass iff Score >=
// Rubric.MinScore (default 1: no unexplained flip tolerated). On failure
// Detail names the FIRST unexplained flip — the workstream and its full
// status chain — localizing the churn per the spine contract. A case with no
// parseable previous statuses, or a report asserting no checkable current
// status, passes at score 1 with a Detail note.
type oscStatusOscillation struct{}

func (oscStatusOscillation) Name() string { return "status-oscillation" }
func (oscStatusOscillation) Kind() string { return "rubric" }

func init() { Register(oscStatusOscillation{}) }

func (oscStatusOscillation) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "status-oscillation", Kind: "rubric", Pass: true, Score: 1}
	prev := oscParseWorkstreams(ref.Text)
	if len(prev) == 0 {
		v.Detail = "no previous statuses declared in the reference; nothing to compare"
		return v
	}
	sentences := oscSentences(eng.Text)
	checked, ok, explained := 0, 0, 0
	firstViolation := ""
	for _, ws := range prev {
		cur, idx, found := oscCurrentStatus(sentences, ws.name)
		if !found {
			continue // no checkable current status for this workstream
		}
		checked++
		if cur == ws.previous() {
			ok++
			continue
		}
		if oscHasRationale(sentences, idx) {
			ok++
			explained++
			continue
		}
		if firstViolation == "" {
			firstViolation = fmt.Sprintf("unexplained status %s: %q went %s with no accompanying rationale",
				ws.flipKind(cur), ws.name, ws.chainWith(cur))
		}
	}
	if checked == 0 {
		v.Detail = fmt.Sprintf("%d previous status(es) declared but the report asserts no checkable current status",
			len(prev))
		return v
	}
	v.Score = float64(ok) / float64(checked)
	min := c.Rubric.MinScore
	if min == 0 {
		min = 1 // default: every status flip must carry a rationale
	}
	if v.Score < min {
		v.Pass = false
		v.Detail = fmt.Sprintf("status-oscillation score %.2f < %.2f (%d/%d consistent or explained); %s",
			v.Score, min, ok, checked, firstViolation)
		return v
	}
	switch {
	case firstViolation != "":
		v.Detail = fmt.Sprintf("status-oscillation score %.2f >= %.2f (tolerated: %s)", v.Score, min, firstViolation)
	case explained > 0:
		v.Detail = fmt.Sprintf("all %d checked workstream status(es) consistent with the previous report (%d flip(s) explained)",
			checked, explained)
	default:
		v.Detail = fmt.Sprintf("all %d checked workstream status(es) unchanged from the previous report", checked)
	}
	return v
}

// oscWorkstream is one workstream's declared status history from the previous
// report block: its name plus canonical statuses oldest-first. The last entry
// is the status the previous report asserted.
type oscWorkstream struct {
	name    string
	history []string
}

func (w oscWorkstream) previous() string { return w.history[len(w.history)-1] }

// flipKind labels an unexplained change: a current status that returns to the
// status held just before the previous report is an "oscillation" (the round
// trip the oracle is named for); any other change is a "flip".
func (w oscWorkstream) flipKind(current string) string {
	if len(w.history) >= 2 && w.history[len(w.history)-2] == current {
		return "oscillation"
	}
	return "flip"
}

// chainWith renders the declared history plus the current status as an arrow
// chain, e.g. "green -> red -> green".
func (w oscWorkstream) chainWith(current string) string {
	return strings.Join(append(append([]string(nil), w.history...), current), " -> ")
}

// oscParseWorkstreams parses the previous-report status block: one
// "workstream | status" (or "workstream | older -> newer" history) per line.
// Unparseable lines — no "|", empty name, any status outside the vocabulary —
// are skipped so a malformed block can never fail a report it does not
// actually constrain.
func oscParseWorkstreams(text string) []oscWorkstream {
	var out []oscWorkstream
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "|", 2)
		if len(parts) != 2 {
			continue
		}
		name := strings.TrimSpace(parts[0])
		if name == "" {
			continue
		}
		var history []string
		valid := true
		for _, seg := range strings.Split(parts[1], "->") {
			canon, okc := oscCanonicalStatus(seg)
			if !okc {
				valid = false
				break
			}
			history = append(history, canon)
		}
		if !valid || len(history) == 0 {
			continue
		}
		out = append(out, oscWorkstream{name: name, history: history})
	}
	return out
}

// oscStatusVocab is the closed set of recognized status phrases and the
// canonical color each folds to. Multi-word phrases let prose say "on track"
// / "at risk" / "blocked" and still land on the green/yellow/red axis the
// previous report block declares. Order is fixed for deterministic scanning.
var oscStatusVocab = []struct{ phrase, canon string }{
	{"green", "green"},
	{"on track", "green"},
	{"yellow", "yellow"},
	{"amber", "yellow"},
	{"at risk", "yellow"},
	{"red", "red"},
	{"blocked", "red"},
	{"off track", "red"},
}

// oscCanonicalStatus folds a declared status token to its canonical color;
// ok is false for anything outside the vocabulary.
func oscCanonicalStatus(s string) (string, bool) {
	s = strings.ToLower(strings.TrimSpace(s))
	for _, e := range oscStatusVocab {
		if s == e.phrase || s == e.canon {
			return e.canon, true
		}
	}
	return "", false
}

// oscRationaleMarkers is the closed set of explanation cues. A flip counts as
// explained iff one appears (word-bounded) in the flip sentence or the
// immediately following sentence — the span report prose attaches its "why" to.
var oscRationaleMarkers = []string{
	"because", "due to", "after", "since", "caused by", "root cause",
	"owing to", "as a result", "driven by", "following", "explained by",
	"fixed by", "resolved by", "thanks to",
}

// oscSentences splits report text into sentences on newlines and ". ",
// trimming whitespace and a trailing period and dropping empties (the same
// split discipline as splitClaims; duplicated so sibling oracle files stay
// edit-disjoint).
func oscSentences(text string) []string {
	var out []string
	for _, line := range strings.Split(text, "\n") {
		for _, part := range strings.Split(line, ". ") {
			s := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(part), "."))
			if s != "" {
				out = append(out, s)
			}
		}
	}
	return out
}

// oscCurrentStatus finds the workstream's current status in the report: the
// first sentence mentioning the workstream (word-bounded, case-insensitive)
// that also carries a status phrase decides, and the LAST status phrase in
// that sentence is the current status — so "moved from red back to green"
// reads as green. idx is the deciding sentence, the anchor for the rationale
// scope.
func oscCurrentStatus(sentences []string, name string) (status string, idx int, ok bool) {
	lname := strings.ToLower(name)
	for i, s := range sentences {
		ls := strings.ToLower(s)
		if oscLastBoundedIndex(ls, lname) < 0 {
			continue
		}
		if st, found := oscLastStatus(ls); found {
			return st, i, true
		}
	}
	return "", 0, false
}

// oscLastStatus returns the canonical status of the last word-bounded status
// phrase in the (already lowercased) sentence.
func oscLastStatus(ls string) (string, bool) {
	best, canon := -1, ""
	for _, e := range oscStatusVocab {
		if i := oscLastBoundedIndex(ls, e.phrase); i > best {
			best, canon = i, e.canon
		}
	}
	return canon, best >= 0
}

// oscHasRationale reports whether a rationale marker appears (word-bounded)
// in the flip sentence at idx or the immediately following sentence.
func oscHasRationale(sentences []string, idx int) bool {
	end := idx + 1
	if end >= len(sentences) {
		end = len(sentences) - 1
	}
	for i := idx; i <= end; i++ {
		ls := strings.ToLower(sentences[i])
		for _, m := range oscRationaleMarkers {
			if oscLastBoundedIndex(ls, m) >= 0 {
				return true
			}
		}
	}
	return false
}

// oscLastBoundedIndex returns the byte index of the last occurrence of phrase
// in text whose neighbors on both sides are not letters or digits, or -1.
// Both arguments must already be lowercased. Word-bounding is what keeps
// "red" from matching inside "delivered".
func oscLastBoundedIndex(text, phrase string) int {
	if phrase == "" {
		return -1
	}
	end := len(text)
	for {
		i := strings.LastIndex(text[:end], phrase)
		if i < 0 {
			return -1
		}
		if oscBoundedAt(text, i, len(phrase)) {
			return i
		}
		end = i
	}
}

// oscBoundedAt reports whether text[i:i+n] is word-bounded: the runes
// immediately before and after are absent or non-alphanumeric.
func oscBoundedAt(text string, i, n int) bool {
	if i > 0 {
		r, _ := utf8.DecodeLastRuneInString(text[:i])
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return false
		}
	}
	if i+n < len(text) {
		r, _ := utf8.DecodeRuneInString(text[i+n:])
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return false
		}
	}
	return true
}
