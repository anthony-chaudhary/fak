// Package policy carries pure, dependency-free policy primitives shared by
// every fak build.
//
// perm.go ADAPTS opencode's ordered wildcard tool-permission ruleset
// evaluation: packages/core/src/permission.ts evaluate() at line 76, revision
// 4eb29a64f0054672950acf789f2b09487ebfbb20 (MIT, Copyright (c) 2025 opencode):
// https://github.com/anomalyco/opencode/blob/4eb29a64f0054672950acf789f2b09487ebfbb20/packages/core/src/permission.ts#L76
// Upstream evaluates an ordered list of {action, resource, effect} rules with
// Wildcard.match on both dimensions and lets the LAST matching rule decide
// (findLast), falling back to a synthesized "ask" rule when none matches.
// Changed for Fak: the two rule dimensions are flattened into one
// "tool:target" pattern per rule, matching uses path.Match-style glob syntax
// over opaque strings (a '*' spans every byte, including '/') instead of
// upstream Wildcard.match, the effect is a free-form action string instead of
// the closed allow/deny/ask set, and a non-matching lookup returns
// matched=false instead of inventing a default. This is a pure evaluator: it
// is not wired into adjudication and persists no rules. See
// THIRD_PARTY_NOTICES.md.
package policy

import (
	"errors"
	"regexp"
	"strings"
)

// errBadPattern marks a malformed glob (an unterminated '[' class); such a
// pattern matches nothing.
var errBadPattern = errors.New("malformed permission pattern")

// PermissionRule is one ordered (pattern, action) row of a
// PermissionRuleset. Pattern is a path.Match-style glob over the composite
// "tool:target" space, matched against opaque strings:
//
//   - "name"    constrains the tool name only (any target),
//   - "name:g"  constrains the tool name AND the target against the two
//     globs; the first ":" splits, later colons belong to the target glob,
//   - "*"       matches every tool and every target.
//
// Action is the caller-defined effect string ("allow", "deny", "ask", ...)
// returned verbatim by Evaluate.
type PermissionRule struct {
	Pattern string
	Action  string
}

// PermissionRuleset is an ordered list of permission rules. Order is the
// contract: Evaluate scans front to back and the LAST matching rule decides
// (upstream findLast semantics); earlier matches are overridden by later
// ones. A nil or empty ruleset matches nothing.
type PermissionRuleset []PermissionRule

// splitPattern splits a rule pattern into its tool and target globs at the
// first ":". A pattern without ":" constrains the tool name only; its target
// glob is "*" (any target).
func splitPattern(pattern string) (toolPat, targetPat string) {
	if i := strings.IndexByte(pattern, ':'); i >= 0 {
		return pattern[:i], pattern[i+1:]
	}
	return pattern, "*"
}

// compileGlob translates a path.Match-style glob into an anchored regular
// expression over an OPAQUE string: '*' spans every byte (including '/'),
// '?' matches exactly one byte, and '[...]' is a character class whose
// leading '!' negates. Unlike path.Match, the value is never split on
// separators: tool targets are opaque strings, not filesystem paths, so a
// catch-all '*' must match every target. An unterminated '[' class is a
// malformed pattern.
func compileGlob(pat string) (*regexp.Regexp, error) {
	var b strings.Builder
	b.WriteString("\\A")
	for i := 0; i < len(pat); {
		switch c := pat[i]; c {
		case '*':
			b.WriteString("[\\s\\S]*")
			i++
		case '?':
			b.WriteString("[\\s\\S]")
			i++
		case '[':
			j := i + 1
			if j < len(pat) && (pat[j] == '!' || pat[j] == '^') {
				j++
			}
			if j < len(pat) && pat[j] == ']' {
				j++
			}
			for j < len(pat) && pat[j] != ']' {
				j++
			}
			if j >= len(pat) {
				return nil, errBadPattern
			}
			class := pat[i+1 : j]
			if strings.HasPrefix(class, "!") {
				class = "^" + class[1:]
			}
			b.WriteByte('[')
			b.WriteString(strings.ReplaceAll(class, `\`, `\\`))
			b.WriteByte(']')
			i = j + 1
		default:
			b.WriteString(regexp.QuoteMeta(string(c)))
			i++
		}
	}
	b.WriteString("\\z")
	re, err := regexp.Compile(b.String())
	if err != nil {
		return nil, errBadPattern
	}
	return re, nil
}

// globMatch reports whether value satisfies the path.Match-style glob pat
// over an opaque string. A malformed pattern never matches: it falls through
// to earlier rules or the caller default instead of surfacing an error at
// evaluation time.
func globMatch(pat, value string) bool {
	re, err := compileGlob(pat)
	if err != nil || re == nil {
		return false
	}
	return re.MatchString(value)
}

// matches reports whether the rule pattern applies to (tool, target): both
// the tool part and the target part must satisfy their globs.
func (r PermissionRule) matches(tool, target string) bool {
	toolPat, targetPat := splitPattern(r.Pattern)
	return globMatch(toolPat, tool) && globMatch(targetPat, target)
}

// Evaluate resolves (tool, target) against the ordered ruleset with
// last-match-wins: it returns the action of the LAST rule whose pattern
// matches, and matched=false with an empty action when no rule applies so
// callers keep their own default (upstream synthesizes an "ask" fallback;
// this seam refuses to invent one). Nil-safe.
func (rs PermissionRuleset) Evaluate(tool, target string) (action string, matched bool) {
	for _, r := range rs {
		if r.matches(tool, target) {
			action, matched = r.Action, true
		}
	}
	return action, matched
}

// Merge flattens rulesets into one ordered ruleset, preserving each set's
// internal order and the argument order (upstream merge is a flat concat).
// The result shares no backing array with its inputs.
func Merge(rulesets ...PermissionRuleset) PermissionRuleset {
	n := 0
	for _, rs := range rulesets {
		n += len(rs)
	}
	out := make(PermissionRuleset, 0, n)
	for _, rs := range rulesets {
		out = append(out, rs...)
	}
	return out
}
