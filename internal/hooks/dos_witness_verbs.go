package hooks

import (
	"regexp"
	"strings"
)

// dos_witness_verbs.go — a faithful, bounded mirror of the DOS commit-audit claim classifier
// (dos-kernel commit_audit.py: _CODE_VERBS / classify_claim), used ONLY to PREDICT an ABSTAIN
// before the commit lands. It is the missing half of the pre-commit lint.
//
// gate_commitmsg.go's commitVerbs is a deliberately WIDER accept-set (it admits descriptive
// verbs like "define"/"explain"/"describe"/"state"/"back"/"report" to keep fak's own false-flag
// rate ~1%). The DOS referee's _CODE_VERBS is TIGHTER, and the conventional type token
// `feat`/`perf` is not itself an effect verb — so a `feat(x): define Y` subject passes
// CommitMsgVerdict (gradeable) yet the referee returns claim_kind=none and ABSTAINs, and the code
// change lands UNWITNESSED. Audit finding (docs/notes, 2026-07-01): 3 of the last 30 commits
// (`back off` / `explain` / `define`) shipped real source + tests through this exact silent gap,
// scored `cleared_rate=1.0` by `dos review`, yet were never bound to a code-effect witness. This
// file lets abstainHazard warn on it at commit-preview time.
//
// Kept in lockstep with the kernel by convention — the same way commitstamp.go's stamp regexes
// are "ported verbatim from the oracle"; the unit tests pin the observed behavior against the
// real ABSTAINed subjects. Only the verbs/markers the prediction actually needs are mirrored:
// this is a predictor, not a reimplementation of the whole referee.

// dosCodeEffectVerbs mirrors commit_audit.py _CODE_VERBS: the verbs the DOS referee accepts as a
// CODE_EFFECT claim when they LEAD the conventional type or the scoped description. A leading verb
// outside this set (with no doc/test signal) makes a feat/perf commit ABSTAIN as claim_kind=none.
var dosCodeEffectVerbs = setOf(
	"fix", "fixes", "fixed", "add", "adds", "added", "implement", "implements",
	"implemented", "remove", "removes", "removed", "delete", "optimize",
	"optimise", "refactor", "rename", "move", "support", "handle", "resolve",
	"resolves", "patch", "correct", "introduce", "enable", "disable", "drop",
	"migrate", "upgrade", "downgrade", "wire", "hook",
	// Descriptive effect verbs the referee also witnesses (commit_audit.py _CODE_VERBS
	// second block) — each names a concrete behavioral effect on code.
	"expose", "exposes", "exposed", "surface", "surfaces", "surfaced",
	"extract", "extracts", "extracted", "enforce", "enforces", "enforced",
	"declare", "declares", "declared", "derive", "derives", "derived",
	"carry", "carries", "carried", "splice", "splices", "spliced",
	"attribute", "attributes", "attributed", "reserve", "reserves", "reserved",
	"consume", "consumes", "consumed", "route", "routes", "routed",
	"forward", "forwards", "forwarded", "feed", "feeds", "fed",
	"accumulate", "accumulates", "accumulated", "memoize", "memoizes", "memoized",
	"synthesize", "synthesizes", "synthesized", "bind", "binds", "bound",
	"ground", "grounds", "grounded", "price", "prices", "priced",
	"fold", "folds", "folded", "arm", "arms", "armed",
	"publish", "publishes", "published", "show", "shows", "showed",
	"standardize", "standardizes", "standardized", "unify", "unifies", "unified",
	"invert", "inverts", "inverted", "refresh", "refreshes", "refreshed",
	"require", "requires", "required", "reset", "resets",
	"warm", "warms", "warmed", "bridge", "bridges", "bridged",
	"floor", "floors", "floored", "drive", "drives", "drove", "driven",
	"refuse", "refuses", "refused", "land", "lands", "landed",
	"dequant", "dequantize", "dequantizes", "author", "authors", "authored",
)

// dosDocLeadMarkers / dosDocWindowNouns / dosTestLeadMarkers mirror the DOC and TEST rungs the
// referee classifies BEFORE code-effect (commit_audit.py classify_claim precedence). The
// predictor uses them ONLY to SUPPRESS a false warning: a feat/perf subject the referee would
// witness on its doc or test rung is not an unwitnessed-code hazard.
var (
	dosDocLeadMarkers  = setOf("doc", "docs", "documentation", "readme", "comment", "comments", "typo", "changelog", "wording", "rephrase", "clarify")
	dosDocWindowNouns  = setOf("docs", "documentation", "readme", "changelog", "guide", "guides", "guideline", "guidelines", "tutorial", "tutorials", "faq", "glossary", "docstring", "docstrings")
	dosTestLeadMarkers = setOf("test", "tests", "testing", "spec", "specs", "coverage", "assertion", "assertions")
)

var dosDocSuffixes = []string{".md", ".rst", ".txt", ".adoc", ".org"}

// dosDepManifestRE — a dependency manifest wearing a doc suffix (requirements.txt) is data, not
// prose; mirrors commit_audit.py _DEP_MANIFEST_TXT_RE so naming one does NOT read as a doc file.
var dosDepManifestRE = regexp.MustCompile(`(?i)^(requirements|constraints)[^/]*\.(txt|in)$`)

// dosWouldAbstainOnCodeEffect predicts whether the DOS commit-audit would ABSTAIN
// (claim_kind=none) on a gradeable feat/perf subject because its description verb is not one the
// referee witnesses as a code effect — the silent gap between fak's wider commitVerbs gate and
// the kernel's tighter _CODE_VERBS. rest is the description (subjectRE group 4); scope is the
// parenthesized scope with its parens (subjectRE group 2), "" if none.
//
// It returns the offending leading verb and true, or "" and false when the referee would witness
// the subject: a recognized effect verb, or a doc/test-shaped subject the referee grades on its
// own rung. Conservative — it errs toward NOT warning (an advisory false positive is the cost the
// kernel itself optimizes against), so it suppresses on any doc/test signal it can see.
func dosWouldAbstainOnCodeEffect(rest, scope string) (verb string, abstains bool) {
	rest = strings.TrimSpace(strings.ToLower(rest))
	first := strings.Trim(splitFirstWordOrColon(rest), "`*\"'")
	if first == "" || dosCodeEffectVerbs[first] {
		return "", false // the referee witnesses this as a code-effect claim
	}
	// Suppress when the referee would classify DOC or TEST — those rungs witness on their own,
	// so the code change is not silently unwitnessed.
	if dosDocLeadMarkers[first] || dosTestLeadMarkers[first] {
		return "", false
	}
	if dosScopeIsDoc(scope) || dosNamesDocFile(rest) {
		return "", false
	}
	for _, w := range dosPhraseWindow(rest, 8) {
		if dosDocWindowNouns[w] {
			return "", false
		}
	}
	return first, true
}

// dosScopeIsDoc reports whether a conventional-commit scope names documentation (`feat(docs): …`),
// the referee's _cc_scope_is_doc analogue. scope carries its parens, e.g. "(docs)".
func dosScopeIsDoc(scope string) bool {
	switch strings.ToLower(strings.Trim(scope, "()")) {
	case "doc", "docs", "readme", "changelog":
		return true
	}
	return false
}

// dosNamesDocFile reports whether a whitespace token in the subject is a doc FILENAME
// (README.md, notes.rst) — naming the doc artifact is the claim's own "this is documentation"
// signal (commit_audit.py _names_doc_file). A dependency manifest (requirements.txt) is data and
// does not engage.
func dosNamesDocFile(subject string) bool {
	for _, tok := range strings.Fields(strings.ToLower(subject)) {
		tok = strings.Trim(tok, "\"'`,;:()[]{}<>")
		if dosDepManifestRE.MatchString(tok) {
			continue
		}
		for _, suf := range dosDocSuffixes {
			if strings.HasSuffix(tok, suf) {
				return true
			}
		}
	}
	return false
}

// dosPhraseWindow returns the first n lowercase-letter words of rest, for multi-word doc-noun
// matching (commit_audit.py _phrase_window). rest is expected already lowercased.
func dosPhraseWindow(rest string, n int) []string {
	var words []string
	for _, w := range strings.FieldsFunc(rest, func(r rune) bool { return r < 'a' || r > 'z' }) {
		if w == "" {
			continue
		}
		words = append(words, w)
		if len(words) >= n {
			break
		}
	}
	return words
}
