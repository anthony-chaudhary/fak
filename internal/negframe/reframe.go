package negframe

// reframe.go is the EMIT-TIME half of the negation stick (#3566). Classify (negframe.go) is the
// static-tree LINT: it reads prose on disk and counts reframable negatives so the ratchet can
// gate a diff. Reframe is the runtime GENERALIZATION of the same lexicon: it takes a string fak
// is about to push toward the model — a SessionStart rule, a step-advice directive, a refusal
// note, a resume-recovery prompt — and hands back the same string with every UNAMBIGUOUS negative
// idiom already flipped to its positive inverse, so the steer leads with the affordance without a
// human having to reword each runtime-assembled string.
//
// It is deliberately the SAME machine as the lint, aimed the other way: only the MECHANICAL tier
// (a finding that carries a confident `Suggest` template) is rewritten. A JUDGEMENT-tier negative
// ("do not re-propose", "never merge without review") is LOAD-BEARING prose whose positive shape
// is a human call — Reframe leaves it byte-identical and only counts it (ResidualNegatives), never
// guesses. Three properties make it safe to wire onto the request path:
//
//   - TOKEN-SUPERSET FAIL-SAFE. A rewrite is applied ONLY when every must-keep contract token
//     (an ALL-CAPS reason token like OFF_TRUNK, or a `backticked` span) present in the original
//     survives it. A candidate that would fold a token away (e.g. "do not DEPLOY" -> "avoid
//     DEPLOYing", dropping the standalone DEPLOY token) is REFUSED and the original span is emitted
//     verbatim. Reframe can only ever make prose more positive, never drop a token the runtime
//     relies on.
//   - IDEMPOTENT. Reframe(Reframe(x)) == Reframe(x): a positive rewrite carries no mechanical
//     negative idiom, so a second pass finds nothing left to flip.
//   - PURE. No I/O, no clock, no model call, no network — a deterministic regex fold, so it is safe
//     to call at emit time and its output is a golden-testable function of its input.

import (
	"regexp"
	"sort"
	"strings"
)

// ReframeResult is the outcome of one emit-time pass. Text is always safe to emit (it equals the
// input when nothing applied or every candidate was refused). The three integers are the telemetry
// the per-turn journal row folds (#3568 consumes them): how many idioms were flipped, how many
// mechanical candidates were refused by the token guard and emitted verbatim, and how many
// judgement-tier negatives remain (the soft residual a human could still reword).
type ReframeResult struct {
	Text              string
	Applied           int               // mechanical idioms flipped to their positive inverse
	VerbatimFallback  int               // mechanical candidates refused (would drop a must-keep token) -> original span kept
	ResidualNegatives int               // judgement-tier negatives left in place (advisory; never auto-rewritten)
	ComplementClasses []ComplementClass // optional complement-routing telemetry, one class per routed span
}

// mustKeepTokenRE matches a contract token a reframe must preserve byte-for-byte: a `backticked`
// span (a literal command/flag/identifier the runtime quotes), or a bare ALL-CAPS run of 3+ chars
// (the structured reason vocabulary — OFF_TRUNK splits into OFF and TRUNK, both must survive;
// POLICY_BLOCK, TERMINAL, DENY, QUARANTINE, ...). The 3-char floor keeps ordinary two-letter caps
// (ON, IO) out while catching every real reason token. Being a superset of the true contract
// vocabulary only makes the guard MORE conservative (it refuses more rewrites), never less safe.
var mustKeepTokenRE = regexp.MustCompile("`[^`]+`|\\b[A-Z0-9]{3,}\\b")

// mustKeepSet returns the multiset of must-keep tokens in s (counts matter: dropping one of two
// identical tokens must be caught).
func mustKeepSet(s string) map[string]int {
	m := map[string]int{}
	for _, t := range mustKeepTokenRE.FindAllString(s, -1) {
		m[t]++
	}
	return m
}

// tokenSuperset reports whether after preserves every must-keep token of before at its full
// multiplicity — the fail-safe predicate: a rewrite passes only when it drops no contract token.
func tokenSuperset(before, after map[string]int) bool {
	for t, n := range before {
		if after[t] < n {
			return false
		}
	}
	return true
}

// prohibitionMarkerRE finds load-bearing negative operators. A token is considered
// bound when it occurs later in the same clause. The narrow do-not-forget family is
// excluded: it expresses a positive obligation, not a prohibition on the token.
var prohibitionMarkerRE = regexp.MustCompile(`(?i)\b(?:never|must\s+not|do\s+not)\b|\bno\b`)
var obligationMarkerRE = regexp.MustCompile(`(?i)^do\s+not\s+(?:forget|fail|neglect)\b`)
var clauseEndRE = regexp.MustCompile(`[.!?;\n]`)

// polarityProtectedSet returns the multiset of must-keep tokens governed by a
// prohibition in s. Counts matter for lines that state the same contract twice.
func polarityProtectedSet(s string) map[string]int {
	protected := map[string]int{}
	for _, marker := range prohibitionMarkerRE.FindAllStringIndex(s, -1) {
		tail := s[marker[0]:]
		if obligationMarkerRE.MatchString(tail) {
			continue
		}
		end := len(tail)
		if loc := clauseEndRE.FindStringIndex(tail); loc != nil {
			end = loc[0]
		}
		clause := tail[:end]
		for _, token := range mustKeepTokenRE.FindAllString(clause, -1) {
			protected[token]++
		}
	}
	return protected
}

// polarityPreserved reports whether every must-keep token protected by a
// load-bearing prohibition before remains protected after. Keeping the literal
// token is insufficient: "never deploy `OFF_TRUNK`" must not become "deploy
// `OFF_TRUNK`".
func polarityPreserved(before, after string) bool {
	return tokenSuperset(polarityProtectedSet(before), polarityProtectedSet(after))
}

// Reframe returns text with every unambiguous negative idiom flipped to its positive inverse,
// leaving load-bearing judgement-tier prose and every must-keep token untouched. It is the
// convenience wrapper over ReframePass for callers that want only the string.
// For transforming refusal summaries into affordance-first prose, see ReframeRefusalProse (#11044).
func Reframe(text string) string { return ReframePass(text).Text }

// ReframePass runs the emit-time reframe and returns the rewritten text plus the telemetry counts.
// It walks lines exactly as Classify does — fenced code blocks (``` / ~~~) and blank lines are
// copied verbatim and never reframed (the card gardens prose, not code samples) — so the reframe
// and the lint agree on what counts as prose.
func ReframePass(text string) ReframeResult {
	lines := strings.Split(text, "\n")
	var b strings.Builder
	res := ReframeResult{}
	inFence := false
	for i, raw := range lines {
		trimmed := strings.TrimSpace(raw)
		isFence := strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~")
		if isFence {
			inFence = !inFence
		}
		if isFence || inFence || trimmed == "" {
			b.WriteString(raw)
		} else {
			out, applied, refused, residual := reframeLine(raw)
			b.WriteString(out)
			res.Applied += applied
			res.VerbatimFallback += refused
			res.ResidualNegatives += residual
		}
		if i < len(lines)-1 {
			b.WriteByte('\n')
		}
	}
	res.Text = b.String()
	return res
}

// reframeLine rewrites a single prose line and reports (newLine, applied, verbatimFallback,
// residual). It reuses the same first-match-wins, most-specific-first non-overlapping walk as
// classifyLine so a mechanical idiom ("do not forget to X") claims its span before the generic
// "do not" judgement rule can re-report it. For each claimed span:
//
//   - a judgement-tier rule (empty Template) contributes one ResidualNegative and is left in place;
//   - a mechanical rule computes its positive rewrite and applies it ONLY when the token-superset
//     guard holds for the whole line; otherwise the span is left verbatim (VerbatimFallback).
//
// Accepted edits are applied right-to-left so earlier byte offsets stay valid.
func reframeLine(line string) (string, int, int, int) {
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
	type edit struct {
		start, end int
		repl       string
	}
	var edits []edit
	applied, refused, residual := 0, 0, 0
	before := mustKeepSet(line)
	for _, r := range rules {
		for _, loc := range r.Pattern.FindAllStringIndex(line, -1) {
			s, e := loc[0], loc[1]
			if overlaps(s, e) {
				continue
			}
			claimed = append(claimed, span{s, e})
			if r.Template == "" {
				residual++ // judgement-tier negative: load-bearing, left in place
				continue
			}
			repl := r.Pattern.ReplaceAllString(line[s:e], r.Template)
			candidate := line[:s] + repl + line[e:]
			if !tokenSuperset(before, mustKeepSet(candidate)) || !polarityPreserved(line, candidate) {
				refused++ // would drop a token or its prohibition -> emit the original span verbatim
				continue
			}
			edits = append(edits, edit{s, e, repl})
			applied++
		}
	}
	if len(edits) == 0 {
		return line, applied, refused, residual
	}
	sort.Slice(edits, func(i, j int) bool { return edits[i].start > edits[j].start })
	out := line
	for _, ed := range edits {
		out = out[:ed.start] + ed.repl + out[ed.end:]
	}
	return out, applied, refused, residual
}
