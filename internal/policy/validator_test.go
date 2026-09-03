package policy

import (
	"errors"
	"testing"
)

func TestValidateRegexSafety(t *testing.T) {
	safePatterns := []struct {
		name    string
		pattern string
	}{
		{"anchored alphanumeric", `^[a-zA-Z0-9_-]+$`},
		{"wildcard span", `foo.*bar`},
		{"bounded digits with separators", `\d{3}-\d{2}-\d{4}`},
		{"disjoint alternation in repetition", `(abc|def)+`},
		{"file extension matching", `.*\.go`},
		{"whitespace separation", `hello\s+world`},
		{"cli flag rm pattern", `\brm\s+-[A-Za-z]*[rRfF]`},
		{"optional sudo in pipe", `\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh\b`},
		{"optional branch prefix and suffix", `(?i)^(refs/heads/)?(main|master|trunk|develop|release(/.+)?|prod(uction)?)$`},
		{"bounded repetition with limit", `git\s+push\b.{0,40}(--force|--force-with-lease)`},
		{"exact repetition", `AKIA[0-9A-Z]{16}`},
		{"disjoint single-char alternation", `(a|b)+`},
		{"single plus repetition", `[a-z]+`},
		{"consecutive disjoint repetitions", `a+b+`},
		{"separated digits", `\d+\s+\d+`},
		{"non-capturing group", `(?:abc|def)`},
		{"named capture group", `(?P<ident>[a-zA-Z_][a-zA-Z0-9_]*)`},
	}

	for _, tc := range safePatterns {
		t.Run("safe_"+tc.name, func(t *testing.T) {
			if err := ValidateRegexSafety(tc.pattern); err != nil {
				t.Errorf("expected pattern %q to be safe, got error: %v", tc.pattern, err)
			}
		})
	}

	dangerousPatterns := []struct {
		name    string
		pattern string
	}{
		{"nested plus", `(a+)+`},
		{"nested star", `(a*)*`},
		{"nested plus charclass", `([a-z]+)+`},
		{"star over plus", `(a+)*`},
		{"alternation containing plus", `(a|b+)+`},
		{"positive lookahead", `(?=abc)`},
		{"negative lookahead", `(?!abc)`},
		{"positive lookbehind", `(?<=abc)`},
		{"negative lookbehind", `(?<!abc)`},
		{"backreference group 1", `(a)\1`},
		{"bare backreference", `\1`},
		{"backreference group 2", `(a)\2`},
		{"nested digit plus", `([0-9]+)+`},
		{"quest in plus", `(a?)+`},
		{"quest in star", `(a?)*`},
		{"nested quest and plus", `((a+)?)+`},
		{"consecutive wildcards", `.*.*`},
		{"consecutive identical plus", `a+a+`},
		{"consecutive digit plus", `\d+\d+`},
		{"consecutive charclass plus", `[a-z]+[a-z]+`},
		{"syntax error unclosed bracket", `[unclosed`},
		{"syntax error unclosed paren", `(unclosed`},
	}

	for _, tc := range dangerousPatterns {
		t.Run("dangerous_"+tc.name, func(t *testing.T) {
			err := ValidateRegexSafety(tc.pattern)
			if err == nil {
				t.Fatalf("expected pattern %q to fail validation, but it passed", tc.pattern)
			}
			if !errors.Is(err, ErrInvalidRegexPattern) {
				t.Errorf("expected error to wrap ErrInvalidRegexPattern, got: %v", err)
			}
		})
	}
}

func TestPolicyAdmissionRejectsUnsafeRegex(t *testing.T) {
	// ReDoS deny_regex in arg_rules must fail policy Parse
	manifestDenyRegex := []byte(`{
		"allow": ["run_shell"],
		"arg_rules": [
			{"tool":"run_shell", "arg":"cmd", "deny_regex":"(a+)+", "reason":"POLICY_BLOCK"}
		]
	}`)
	if _, err := Parse(manifestDenyRegex); err == nil {
		t.Fatal("expected policy with ReDoS deny_regex to be rejected during admission, but passed")
	} else if !errors.Is(err, ErrInvalidRegexPattern) {
		t.Errorf("expected parse error to wrap ErrInvalidRegexPattern, got: %v", err)
	}

	// Lookahead in deny_regex must fail policy Parse
	manifestLookahead := []byte(`{
		"allow": ["run_shell"],
		"arg_rules": [
			{"tool":"run_shell", "arg":"cmd", "deny_regex":"(?=abc)", "reason":"POLICY_BLOCK"}
		]
	}`)
	if _, err := Parse(manifestLookahead); err == nil {
		t.Fatal("expected policy with lookahead deny_regex to be rejected during admission, but passed")
	}

	// ReDoS secret_patterns must fail policy Parse
	manifestSecret := []byte(`{
		"allow": ["read_file"],
		"secret_patterns": ["(a*)*"]
	}`)
	if _, err := Parse(manifestSecret); err == nil {
		t.Fatal("expected policy with ReDoS secret_patterns to be rejected during admission, but passed")
	} else if !errors.Is(err, ErrInvalidRegexPattern) {
		t.Errorf("expected parse error to wrap ErrInvalidRegexPattern, got: %v", err)
	}

	// Safe patterns should pass policy Parse
	manifestSafe := []byte(`{
		"allow": ["run_shell"],
		"arg_rules": [
			{"tool":"run_shell", "arg":"cmd", "deny_regex":"^[a-zA-Z0-9_-]+$", "reason":"POLICY_BLOCK"}
		],
		"secret_patterns": ["sk-[a-zA-Z0-9]{20}"]
	}`)
	if _, err := Parse(manifestSafe); err != nil {
		t.Fatalf("expected safe policy manifest to pass admission, got error: %v", err)
	}
}
