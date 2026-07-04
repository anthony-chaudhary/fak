package devindex

// Verb-catalog STYLE gate (#2249, epic #2245). The catalog is a live VIEW
// (verbs.go): coverage is derived from cmd/fak/main.go's dispatch switch, and the
// curated verbManifest supplies the synopsis/lane/doc QUALITY for the verbs it
// names. UndeclaredVerbs (freshness.go) is only an ADVISORY curation-drift signal,
// so "not yet cataloged" placeholders regrow as peers add verbs faster than they
// curate them. This gate makes the quality bar enforceable instead of advisory:
// CheckVerbCatalogStyle red-lights a NEW style defect — an uncataloged verb, an
// over-width synopsis, a sentence-case lead, a trailing period, or unbalanced
// parenthetical notation — while a frozen, shrinks-only baseline grandfathers the
// defects present when the gate landed. A devindex test (verbstyle_test.go) runs
// it in make ci; the cmd/fak wall-header argument-notation grammar (<required>,
// [optional], a|b) is the out-of-lane remainder, tracked on #2249.

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// VerbStyleKind classifies a verb-catalog style defect. The tokens are stable so a
// baseline entry keyed on one survives a synopsis reword that keeps the same defect.
type VerbStyleKind string

const (
	// VerbStyleUncataloged marks a dispatched verb carrying the derived fallback
	// synopsis (verbs.go's "not yet cataloged — …") — no curated verbManifest entry.
	VerbStyleUncataloged VerbStyleKind = "uncataloged"
	// VerbStyleSynopsisWidth marks a curated synopsis wider than VerbSynopsisMaxRunes.
	VerbStyleSynopsisWidth VerbStyleKind = "synopsis-width"
	// VerbStyleSynopsisLead marks a sentence-case leading capital (a single leading
	// upper-case letter) — the house voice is a lowercase lead; a proper noun or
	// acronym (two or more upper-case letters in the first token) is allowed.
	VerbStyleSynopsisLead VerbStyleKind = "synopsis-lead"
	// VerbStyleTrailingPeriod marks a synopsis that ends with a period (the manifest
	// voice is a bare clause, no terminal punctuation).
	VerbStyleTrailingPeriod VerbStyleKind = "trailing-period"
	// VerbStyleNotation marks unbalanced () or [] notation delimiters in a synopsis.
	VerbStyleNotation VerbStyleKind = "notation"
)

// VerbSynopsisMaxRunes is the width ceiling for a curated verb synopsis, counted in
// runes (not bytes) so a multi-byte character costs one column, matching the usage
// wall's own rendering width.
const VerbSynopsisMaxRunes = 110

// VerbStyleViolation is one style defect found on a catalog verb.
type VerbStyleViolation struct {
	Verb   string        `json:"verb"`
	Kind   VerbStyleKind `json:"kind"`
	Detail string        `json:"detail"`
}

// key is the baseline key: canonical verb name + kind. Keying on the kind (not the
// exact detail) lets a synopsis reword keep its grandfather as long as the SAME
// defect persists, while a NEW kind of defect on a baselined verb still reds.
func (v VerbStyleViolation) key() string { return v.Verb + "\t" + string(v.Kind) }

// CheckVerbCatalogStyle returns every style violation across the given catalog
// verbs (from Catalog.Verbs()) EXCLUDING the frozen grandfathered baseline. A nil
// result is the green state the gate enforces: no style debt has entered since the
// baseline was pinned. Findings are sorted by verb then kind for a stable diff.
func CheckVerbCatalogStyle(verbs []Verb) []VerbStyleViolation {
	var out []VerbStyleViolation
	for _, v := range verbs {
		for _, viol := range verbStyleViolations(v) {
			if verbStyleBaseline[viol.key()] {
				continue
			}
			out = append(out, viol)
		}
	}
	sortViolations(out)
	return out
}

// rawVerbCatalogStyle returns ALL style violations INCLUDING baselined ones. The
// shrinks-only test uses it to prove no baseline entry has gone stale (been fixed):
// a grandfathered defect that no longer occurs must be dropped from the baseline, so
// the baseline can only shrink as the catalog is cleaned up.
func rawVerbCatalogStyle(verbs []Verb) []VerbStyleViolation {
	var out []VerbStyleViolation
	for _, v := range verbs {
		out = append(out, verbStyleViolations(v)...)
	}
	sortViolations(out)
	return out
}

func sortViolations(out []VerbStyleViolation) {
	sort.Slice(out, func(i, j int) bool {
		if out[i].Verb != out[j].Verb {
			return out[i].Verb < out[j].Verb
		}
		return out[i].Kind < out[j].Kind
	})
}

// verbStyleViolations grades one verb against the style rules (baseline-agnostic).
func verbStyleViolations(v Verb) []VerbStyleViolation {
	syn := strings.TrimSpace(v.Synopsis)
	var out []VerbStyleViolation
	add := func(k VerbStyleKind, d string) {
		out = append(out, VerbStyleViolation{Verb: v.Name, Kind: k, Detail: d})
	}
	// A missing/placeholder synopsis is the headline defect (#2249): grade only that
	// and stop, so an uncataloged verb yields one clear finding, not a pile of noise.
	if syn == "" || strings.Contains(syn, "not yet cataloged") {
		add(VerbStyleUncataloged, "no curated synopsis (derived fallback placeholder)")
		return out
	}
	if n := utf8.RuneCountInString(syn); n > VerbSynopsisMaxRunes {
		add(VerbStyleSynopsisWidth, fmt.Sprintf("%d runes > %d", n, VerbSynopsisMaxRunes))
	}
	if !synopsisLeadOK(syn) {
		add(VerbStyleSynopsisLead, "sentence-case leading capital; use a lowercase lead (or a proper-noun/acronym)")
	}
	if strings.HasSuffix(syn, ".") {
		add(VerbStyleTrailingPeriod, "synopsis ends with a period")
	}
	if d, ok := balancedNotation(syn); !ok {
		add(VerbStyleNotation, d)
	}
	return out
}

// synopsisLeadOK reports whether the synopsis opens in the house voice: a lowercase
// lead, a symbol lead (e.g. "= fak ps --watch"), or a proper-noun/acronym lead. A
// proper noun/acronym is a first token with two or more upper-case letters
// (AILuminate, SWE-bench, OpenAI, FrontierSWE); a single leading capital on an
// otherwise-lowercase word (Run, Read) is sentence case and fails.
func synopsisLeadOK(syn string) bool {
	fields := strings.Fields(syn)
	if len(fields) == 0 {
		return true
	}
	w := strings.TrimLeftFunc(fields[0], func(r rune) bool { return !unicode.IsLetter(r) })
	if w == "" {
		return true // leads with punctuation/symbol, not a word
	}
	r0, _ := utf8.DecodeRuneInString(w)
	if !unicode.IsUpper(r0) {
		return true // lowercase lead — the house voice
	}
	uppers := 0
	for _, r := range w {
		if unicode.IsUpper(r) {
			uppers++
		}
	}
	return uppers >= 2 // proper noun/acronym, not sentence case
}

// balancedNotation reports whether the synopsis's () and [] delimiters nest and
// close cleanly. Angle brackets are NOT checked: synopses use "->" arrows freely
// (orient->plan->act), so a '>' is not a notation close. On imbalance it returns a
// short reason and false.
func balancedNotation(syn string) (string, bool) {
	var stack []rune
	closer := map[rune]rune{')': '(', ']': '['}
	for _, r := range syn {
		switch r {
		case '(', '[':
			stack = append(stack, r)
		case ')', ']':
			if len(stack) == 0 || stack[len(stack)-1] != closer[r] {
				return fmt.Sprintf("unbalanced %q", string(r)), false
			}
			stack = stack[:len(stack)-1]
		}
	}
	if len(stack) > 0 {
		return fmt.Sprintf("unclosed %q", string(stack[len(stack)-1])), false
	}
	return "", true
}
