package adjudicator

import "strings"

// ootInertMentionVerbs is the allow-list of command words for which a quoted
// operand is provably DATA: none of them executes an operand, and none writes a
// file except through a redirect — and a redirect is extracted separately by the
// quote-aware redirectWriteTargets, so a real one is never in the zero-target case
// this set gates. It is explicit rather than shape-based for the same reason
// powerShellCmdlets is: a guessed set is a guessed bypass.
//
// Deliberately ABSENT: eval, xargs, find (-exec), sed, awk, and the shells. Each
// re-executes or evaluates an operand, so a traversal quoted inside their arguments
// is a live write that rceShellSources does not unwrap — eval especially, since
// `eval "cp x ../../etc/y"` extracts no target and would otherwise land in exactly
// the carve-out below.
var ootInertMentionVerbs = map[string]bool{
	"echo":   true,
	"printf": true,
	"grep":   true,
	"egrep":  true,
	"fgrep":  true,
	"rg":     true,
	"git":    true,
}

// ootMentionOnly reports whether an out-of-tree raw match is provably an inert
// MENTION of a traversal rather than a write.
//
// The four out-of-tree rules match a literal `../`, so they fire on any command that
// merely QUOTES one — writing a commit message about the rule, echoing a warning,
// grepping the docs for it. Extraction correctly finds no write destination in those,
// but the caller's fail-closed default ("raw matched, no destination identified")
// then denies them anyway. That default is right in general and must stay: a shape
// the extractor does not understand has to keep the deny. It is only wrong when the
// absence of a destination is PROVEN rather than merely observed.
//
// All three conditions are required, and each closes a bypass the other two leave:
//
//   - no `..` traversal outside a quoted token in ANY source, so an unquoted
//     destination (`git init ../x`, `cp a ../b`) can never reach this path even
//     though its verb may be inert;
//   - every segment headed by a verb from ootInertMentionVerbs, so a launcher that
//     re-executes its quoted operand (`eval "cp x ../y"`) cannot launder one past
//     the quote test above;
//   - the caller has already extracted ZERO write destinations, so anything the
//     extractor did understand is still decided by containment, not by this.
//
// Sources are walked individually, so an `sh -c` / $() / backtick payload is tested
// on its own unwrapped text: `sh -c "cp x ../../etc/y"` exposes an unquoted `..` in
// the inner source and is refused.
//
// Strictly SUBTRACTIVE, like the rest of the package: it can only turn a deny into
// an admit, only when the raw regex has already matched, and only on a proof.
func ootMentionOnly(cmd string) bool {
	sources := rceShellSources(cmd)
	if len(sources) == 0 {
		return false
	}
	for _, src := range sources {
		if hasUnquotedTraversal(src) {
			return false
		}
		segs := rceShellSegments(src)
		if len(segs) == 0 {
			return false
		}
		for _, seg := range segs {
			i := rceCommandWord(seg.argv)
			if i < 0 {
				return false
			}
			if !ootInertMentionVerbs[rceProgramBasename(seg.argv[i])] {
				return false
			}
		}
	}
	return true
}

// hasUnquotedTraversal reports whether src contains a `..` path traversal outside
// any quoted token. Quote handling mirrors redirectWriteTargets byte for byte so
// the two agree on what "quoted" means; disagreeing would let a destination be
// invisible to one and inert to the other.
func hasUnquotedTraversal(src string) bool {
	var quote byte
	for i := 0; i < len(src); i++ {
		ch := src[i]
		if quote != 0 {
			if ch == '\\' && quote == '"' && i+1 < len(src) {
				i++
				continue
			}
			if ch == quote {
				quote = 0
			}
			continue
		}
		switch ch {
		case '\\':
			i++ // skip the escaped char
		case '\'', '"':
			quote = ch
		case '.':
			if strings.HasPrefix(src[i:], "../") || strings.HasPrefix(src[i:], `..\`) {
				return true
			}
		}
	}
	return false
}
