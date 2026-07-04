package adjudicator

import (
	"os"
	"path"
	"strings"
)

// canonicalizeArgValue folds an arg value to the ONE form arg predicates
// match against, closing the spelling-variant bypass class named in #1464 /
// #1466 (a documented instance already slipped `self_modify_globs`): a
// backslash separator, a redundant "./"/".." segment, a "~"/"$HOME" alias, or
// a value wrapped in quotes all denote the SAME call as their canonical
// spelling, so a rule written against the canonical form must still catch
// them. Pure and allocation-light — the only external read is the process
// env (the adjudicator has no richer per-session env view yet), never the
// filesystem or network.
//
// ok is false when value cannot be canonicalized safely (an unterminated
// quote) — the caller fails closed (MALFORMED) rather than risk a rule that
// would have matched the canonical form silently passing a raw one.
func canonicalizeArgValue(value string) (string, bool) {
	v, ok := unwrapQuotes(value)
	if !ok {
		return "", false
	}
	v = strings.ReplaceAll(v, `\`, "/")
	v = expandEnvAliases(v)
	v = collapseWhitespace(v)
	// Dot-segment resolution only applies to a bare, single-token value (a
	// path or URL, which is what ArgAllowGlob always targets and what a
	// path-shaped ArgDenyRegex rule usually names). A multi-token value is a
	// shell command line, not a path: path.Clean's rule 3 ("eliminate an
	// inner '..' along with the element that precedes it") does not know a
	// preceding token is a command/flag rather than a path segment, and would
	// delete it — turning `echo x >> ../../tmp/exfil` into `tmp/exfil` and
	// silently erasing the very command a traversal rule must still see. A
	// URL-scheme marker ("://") is excluded for the same reason path.Clean
	// would collapse "//" to a single slash and rewrite "http://host" to
	// "http:/host", breaking any rule matching an egress URL.
	if v != "" && !strings.Contains(v, " ") && !strings.Contains(v, "://") {
		v = path.Clean(v)
	}
	return v, true
}

// unwrapQuotes strips one layer of matching quotes wrapping the whole value —
// the "quote styles" bypass variant, e.g. an arg carrying `'/a/b'` or `"/a/b"`
// instead of the bare path. Only a value that STARTS with a quote is treated
// as a whole-value wrap candidate (a command string that merely ENDS in a
// quote, e.g. a shell fragment closing an inner quoted segment, is not one);
// a leading quote with no matching close is an unterminated quote —
// non-canonicalizable, so the caller fails closed.
func unwrapQuotes(v string) (string, bool) {
	if len(v) == 0 {
		return v, true
	}
	first := v[0]
	if first != '\'' && first != '"' {
		return v, true
	}
	if len(v) >= 2 && v[len(v)-1] == first {
		return v[1 : len(v)-1], true
	}
	return "", false
}

// expandEnvAliases resolves the well-known aliases the documented bypasses
// actually used ("~" and "$HOME"/"${HOME}") against the process HOME (falling
// back to USERPROFILE, where HOME is often unset). An unrecognized "$VAR" is
// left as-is: it is not a documented bypass class, and expanding an arbitrary
// variable would make the canonical form depend on unrelated process state.
func expandEnvAliases(v string) string {
	home := os.Getenv("HOME")
	if home == "" {
		home = os.Getenv("USERPROFILE")
	}
	if home == "" {
		return v
	}
	v = strings.ReplaceAll(v, "${HOME}", home)
	v = strings.ReplaceAll(v, "$HOME", home)
	switch {
	case v == "~":
		return home
	case strings.HasPrefix(v, "~/"):
		return home + v[1:]
	default:
		return v
	}
}

// collapseWhitespace folds runs of whitespace to a single space and trims the
// ends, so a command re-spaced or re-quoted still matches the same rule as
// its canonical spelling.
func collapseWhitespace(v string) string {
	return strings.Join(strings.Fields(v), " ")
}
