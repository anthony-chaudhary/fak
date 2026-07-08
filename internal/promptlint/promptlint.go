// Package promptlint is the durable freshness monitor for the dispatch worker-issue
// prompts. A rendered worker prompt makes executable claims about the fak surface: it
// tells the worker which `fak <verb>` commands to run, and it names the UPPER_SNAKE
// refusal tokens that will gate the commit "below the agent" (OFF_TRUNK, PATHSPEC_RACE,
// ARCH_LAYER_VIOLATION, …). Those claims silently rot: a verb is renamed, a reason
// token is retired, or one of the two renderers (the live Go RenderIssuePrompt and the
// legacy Python tools/issue_worker_prompt.py) lags the other. Nothing watches for that
// today — a whole fleet of workers can be dispatched against guidance that names a
// command or token which no longer resolves.
//
// promptlint is that watcher. It extracts the executable claims from a rendered prompt
// and flags any that no longer resolve against fak's own authoritative registries:
//   - a `fak <verb>` the prompt names that matches no live verb  -> STALE_VERB
//   - an UPPER_SNAKE refusal token no fak registry declares       -> STALE_REFUSAL_TOKEN
//
// It is a loop-runnable observable: a dispatch preflight or `fak index freshness` pass
// lints the prompt it is about to hand a worker and refuses on any finding, so drift is
// caught at the source instead of by a confused worker.
//
// # Authority
//
// The load-bearing design choice is WHICH registry decides "resolves". The reason
// tokens a worker prompt names are fak's OWN gate tokens — the safecommit closed
// vocabulary (internal/safecommit: OFF_TRUNK, PATHSPEC_RACE, CORE_SELF_MODIFY, …), the
// pythongate ratchet (REASON_NEW_PYTHON_TOOL), the architest layer gate
// (ARCH_LAYER_VIOLATION) — NOT the DOS-kernel arbitration vocabulary. A monitor that
// checked prompt tokens against only the DOS closed set would raise a false
// STALE_REFUSAL_TOKEN on every real fak commit-gate token. So promptlint never bakes a
// vocabulary in: the caller supplies the authoritative Known sets from the live
// registries, and this package only does the extraction + set membership.
//
// # Layering
//
// Pure leaf, stdlib only. It deliberately does NOT import the devindex verb catalog or
// the safecommit/pythongate reason vocabularies (that would be a lateral cross-leaf
// import, the ARCH_LAYER_VIOLATION class it exists to help catch). The composition root
// (cmd/fak) wires devindex.Verbs() + the reason registries into a Known and passes it
// in. That keeps promptlint a low, dependency-free leaf any surface can reuse.
package promptlint

import (
	"regexp"
	"sort"
	"strings"
)

// Kind is the closed vocabulary of prompt-freshness findings. Keeping it a small closed
// set (not free text) is what lets a loop route on the finding the same way the guard
// routes on a refusal reason.
type Kind string

const (
	// StaleVerb: the prompt tells a worker to run `fak <verb>` but no live fak verb
	// (canonical name or alias) matches — a renamed/removed verb, or a typo.
	StaleVerb Kind = "STALE_VERB"
	// StaleRefusalToken: the prompt names an UPPER_SNAKE refusal token as an enforced
	// gate, but no fak reason registry declares it — a retired or misspelled token.
	StaleRefusalToken Kind = "STALE_REFUSAL_TOKEN"
)

// Finding is one drift the monitor found: what kind, the offending token exactly as the
// prompt named it, and a short surrounding snippet so a human (or a repair loop) can
// locate and fix it without re-deriving where it came from.
type Finding struct {
	Kind    Kind   `json:"kind"`
	Token   string `json:"token"`
	Context string `json:"context,omitempty"`
}

// Known is the authoritative "these resolve" evidence the caller supplies from fak's
// live registries. Every set is lowercased for verbs and upper-cased for reasons by
// convention of its source; NewKnown normalizes for you. A nil set means "do not lint
// that dimension" — so a caller that only has the verb catalog can check verbs without
// having to assemble the full reason union first.
type Known struct {
	// Verbs is the set of live `fak` verb spellings (canonical names + aliases),
	// lowercased. Wire from devindex.Catalog.Verbs() -> Verb.Spellings().
	Verbs map[string]bool
	// Reasons is the union of fak's real refusal-reason tokens, upper-cased. Wire from
	// safecommit's closed vocabulary + pythongate + architest (+ the DOS closed set for
	// the arbitration tokens the prompt also names, e.g. OFF_TRUNK).
	Reasons map[string]bool
}

// NewKnown builds a Known from raw slices, normalizing case (verbs lowercased, reasons
// upper-cased) and skipping blanks, so callers can pass registry values verbatim.
func NewKnown(verbs, reasons []string) Known {
	k := Known{}
	if verbs != nil {
		k.Verbs = map[string]bool{}
		for _, v := range verbs {
			if v = strings.ToLower(strings.TrimSpace(v)); v != "" {
				k.Verbs[v] = true
			}
		}
	}
	if reasons != nil {
		k.Reasons = map[string]bool{}
		for _, r := range reasons {
			if r = strings.ToUpper(strings.TrimSpace(r)); r != "" {
				k.Reasons[r] = true
			}
		}
	}
	return k
}

// fakVerbRe matches a `fak <verb>` command mention: the literal `fak`, whitespace, then
// a verb token (lowercase, digits, dashes). The leading (?:^|[^(\w]) capture ensures we
// do NOT match the `(fak <lane>)` commit-stamp trailer — `(fak docs)` names the owning
// LANE, not a verb, and must never be flagged as a stale verb. It also avoids matching
// inside a longer word (e.g. `unfak`). The verb is group 1.
var fakVerbRe = regexp.MustCompile(`(?:^|[^(\w])fak\s+([a-z][a-z0-9-]*)`)

// upperSnakeRe matches an UPPER_SNAKE refusal token: at least two SCREAMING segments
// joined by underscores (OFF_TRUNK, PATHSPEC_RACE, ARCH_LAYER_VIOLATION). Requiring the
// underscore excludes bare ALLCAPS prose (NEVER, ONLY, NOT, DCO) and dotted filenames
// (AGENTS.md, CLAIMS.md), which are not reason tokens.
var upperSnakeRe = regexp.MustCompile(`\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b`)

// Mention is one extracted claim: the token as written plus a short surrounding context
// snippet for the human fix. ExtractFakVerbs and ExtractRefusalTokens return these; Lint
// folds them against Known.
type Mention struct {
	Token   string
	Context string
}

// ExtractFakVerbs returns every distinct `fak <verb>` command the prompt names, in first
// -appearance order. The `(fak <lane>)` commit-trailer form is excluded by construction
// (see fakVerbRe), so a `(fak docs)` stamp is never mistaken for a `fak docs` command.
func ExtractFakVerbs(prompt string) []Mention {
	return extract(prompt, fakVerbRe, 1)
}

// ExtractRefusalTokens returns every distinct UPPER_SNAKE refusal token the prompt names,
// in first-appearance order.
func ExtractRefusalTokens(prompt string) []Mention {
	return extract(prompt, upperSnakeRe, 0)
}

// extract runs re over prompt and returns the distinct captures (group `grp`) with a
// bounded context snippet, preserving first-appearance order so findings read top-down.
func extract(prompt string, re *regexp.Regexp, grp int) []Mention {
	seen := map[string]bool{}
	var out []Mention
	for _, loc := range re.FindAllStringSubmatchIndex(prompt, -1) {
		tok := prompt[loc[2*grp]:loc[2*grp+1]]
		if tok == "" || seen[tok] {
			continue
		}
		seen[tok] = true
		out = append(out, Mention{Token: tok, Context: snippet(prompt, loc[2*grp], loc[2*grp+1])})
	}
	return out
}

// snippet returns up to ~40 chars of single-line context around [start,end) so a finding
// carries where it came from without dumping the whole prompt.
func snippet(s string, start, end int) string {
	const pad = 24
	lo, hi := start-pad, end+pad
	if lo < 0 {
		lo = 0
	}
	if hi > len(s) {
		hi = len(s)
	}
	ctx := strings.TrimSpace(s[lo:hi])
	ctx = strings.ReplaceAll(ctx, "\n", " ")
	return strings.Join(strings.Fields(ctx), " ")
}

// Lint folds the prompt's executable claims against Known and returns every claim that no
// longer resolves. Findings are sorted (kind, then token) for a stable, diffable report.
// A nil Known.Verbs skips the verb dimension; a nil Known.Reasons skips the reason
// dimension — so a caller with only one registry loaded lints only what it can vouch for,
// rather than flagging everything as stale.
func Lint(prompt string, known Known) []Finding {
	var out []Finding
	if known.Verbs != nil {
		for _, m := range ExtractFakVerbs(prompt) {
			if !known.Verbs[strings.ToLower(m.Token)] {
				out = append(out, Finding{Kind: StaleVerb, Token: m.Token, Context: m.Context})
			}
		}
	}
	if known.Reasons != nil {
		for _, m := range ExtractRefusalTokens(prompt) {
			if !known.Reasons[strings.ToUpper(m.Token)] {
				out = append(out, Finding{Kind: StaleRefusalToken, Token: m.Token, Context: m.Context})
			}
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return out[i].Token < out[j].Token
	})
	return out
}

// OK reports whether the prompt named nothing stale — the one-bit gate a loop or a
// dispatch preflight branches on.
func OK(prompt string, known Known) bool {
	return len(Lint(prompt, known)) == 0
}
