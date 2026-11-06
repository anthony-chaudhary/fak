package policy

import "testing"

// TestPermissionRulesetWildcardLastMatchWins pins the override order the
// ticket requires: allow "*", then deny "bash", then allow "bash:ls*". The
// last matching rule decides; when nothing matches, matched=false so the
// caller keeps its own default.
func TestPermissionRulesetWildcardLastMatchWins(t *testing.T) {
	rs := PermissionRuleset{
		{Pattern: "*", Action: "allow"},
		{Pattern: "bash", Action: "deny"},
		{Pattern: "bash:ls*", Action: "allow"},
	}
	cases := []struct {
		name    string
		tool    string
		target  string
		wantAct string
		wantHit bool
	}{
		{"later specific allow overrides earlier deny and star", "bash", "ls -la", "allow", true},
		{"bare ls target still resolves to the later allow", "bash", "ls", "allow", true},
		{"non-ls target falls back to the bash deny row", "bash", "rm -rf /", "deny", true},
		{"empty target matches the bare bash row only", "bash", "", "deny", true},
		{"other tool resolves to the star allow", "python", "-c 'print(1)'", "allow", true},
		{"matching is case-sensitive", "Bash", "ls", "allow", true},
		{"star matches the empty pair", "", "", "allow", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, hit := rs.Evaluate(tc.tool, tc.target)
			if hit != tc.wantHit || got != tc.wantAct {
				t.Fatalf("Evaluate(%q,%q) = (%q,%v), want (%q,%v)", tc.tool, tc.target, got, hit, tc.wantAct, tc.wantHit)
			}
		})
	}
}

// TestPermissionRulesetGlobSyntax pins the path.Match-style syntax over
// OPAQUE strings: '*' spans every byte including '/', '?' is exactly one
// byte, and '[...]' is a character class whose leading '!' negates.
func TestPermissionRulesetGlobSyntax(t *testing.T) {
	cases := []struct {
		name    string
		pattern string
		tool    string
		target  string
		want    bool
	}{
		{"star spans slashes in the target", "*", "bash", "/etc/passwd", true},
		{"question mark is exactly one byte", "b?s?", "bash", "", true},
		{"question mark does not span", "b?s?", "basash", "", false},
		{"character class member", "cat:f[i]le*", "cat", "file.txt", true},
		{"character class rejects", "cat:f[i]le*", "cat", "fole.txt", false},
		{"negated class excludes the byte", "run:[!r]*m", "run", "rm", false},
		{"negated class admits others", "run:[!r]*m", "run", "xm", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rs := PermissionRuleset{{Pattern: tc.pattern, Action: "allow"}}
			got, hit := rs.Evaluate(tc.tool, tc.target)
			if hit != tc.want || (hit && got != "allow") {
				t.Fatalf("Evaluate(%q,%q) under %q = (%q,%v), want hit=%v", tc.tool, tc.target, tc.pattern, got, hit, tc.want)
			}
		})
	}
}

// TestPermissionRulesetNoMatchKeepsCallerDefault pins the absent-ruleset and
// no-match contract: matched=false with an empty action, never a synthesized
// effect.
func TestPermissionRulesetNoMatchKeepsCallerDefault(t *testing.T) {
	var nilRS PermissionRuleset
	if act, hit := nilRS.Evaluate("bash", "ls"); hit || act != "" {
		t.Fatalf("nil ruleset Evaluate = (%q,%v), want (\"\",false)", act, hit)
	}
	empty := PermissionRuleset{}
	if act, hit := empty.Evaluate("bash", "ls"); hit || act != "" {
		t.Fatalf("empty ruleset Evaluate = (%q,%v), want (\"\",false)", act, hit)
	}
	rs := PermissionRuleset{{Pattern: "bash", Action: "deny"}}
	if act, hit := rs.Evaluate("python", "x"); hit || act != "" {
		t.Fatalf("no-match Evaluate = (%q,%v), want (\"\",false)", act, hit)
	}
}

// TestPermissionRulesetDuplicateAndMalformedPatterns pins duplicate-row
// ordering (the later identical pattern wins), malformed-glob fall-through,
// and the first-colon split of "tool:target" patterns.
func TestPermissionRulesetDuplicateAndMalformedPatterns(t *testing.T) {
	dup := PermissionRuleset{
		{Pattern: "bash", Action: "allow"},
		{Pattern: "bash", Action: "deny"},
	}
	if act, hit := dup.Evaluate("bash", "x"); !hit || act != "deny" {
		t.Fatalf("duplicate rows: the later row must win, got (%q,%v)", act, hit)
	}
	malformed := PermissionRuleset{
		{Pattern: "b[ash", Action: "deny"},
		{Pattern: "*", Action: "allow"},
	}
	if act, hit := malformed.Evaluate("bash", "x"); !hit || act != "allow" {
		t.Fatalf("a malformed glob must not match; evaluation must fall through to *, got (%q,%v)", act, hit)
	}
	colon := PermissionRuleset{{Pattern: "git:push:*", Action: "deny"}}
	if act, hit := colon.Evaluate("git", "push:origin main"); !hit || act != "deny" {
		t.Fatalf("colons after the first belong to the target glob, got (%q,%v)", act, hit)
	}
	if act, hit := colon.Evaluate("git", "pull"); hit {
		t.Fatalf("the target glob must not match pull, got (%q,%v)", act, hit)
	}
}

// TestPermissionRulesetMergePreservesOrder pins that Merge is an ordered flat
// concat: later sets override earlier ones, inputs are untouched.
func TestPermissionRulesetMergePreservesOrder(t *testing.T) {
	a := PermissionRuleset{{Pattern: "*", Action: "allow"}}
	b := PermissionRuleset{{Pattern: "bash", Action: "deny"}}
	merged := Merge(a, b)
	if act, hit := merged.Evaluate("bash", "x"); !hit || act != "deny" {
		t.Fatalf("merged order must keep the later set's row winning, got (%q,%v)", act, hit)
	}
	if act, hit := merged.Evaluate("python", "x"); !hit || act != "allow" {
		t.Fatalf("merged ruleset lost the earlier star row, got (%q,%v)", act, hit)
	}
	if len(a) != 1 || len(b) != 1 || len(merged) != 2 {
		t.Fatalf("Merge must not mutate inputs: len(a)=%d len(b)=%d len(merged)=%d", len(a), len(b), len(merged))
	}
}
