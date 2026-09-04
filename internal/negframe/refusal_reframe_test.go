package negframe

import (
	"strings"
	"testing"
)

func TestRefusalReframe(t *testing.T) {
	cases := []struct {
		name         string
		in           string
		wantContains []string
		tokens       []string
	}{
		{
			name: "denied-prefix-and-test-tamper",
			in:   "DENIED: tool call attempted to write to internal/adjudicator/test_immunity_test.go; rule=test_immunity; reason=TEST_TAMPER_REFUSED. Modify only implementation files; test modifications are forbidden.",
			wantContains: []string{
				"ACTION REQUIRED:",
				"Modify implementation files in your assigned lane; preserve test files intact",
			},
			tokens: []string{
				"internal/adjudicator/test_immunity_test.go",
				"rule=test_immunity",
				"reason=TEST_TAMPER_REFUSED",
			},
		},
		{
			name: "refused-do-not-edit-path",
			in:   "REFUSED: do not edit internal/witness/witness.go",
			wantContains: []string{
				"ACTION REQUIRED: preserve internal/witness/witness.go",
			},
			tokens: []string{
				"internal/witness/witness.go",
			},
		},
		{
			name: "refused-do-not-edit-token-x",
			in:   "REFUSED: do not edit X",
			wantContains: []string{
				"ACTION REQUIRED: preserve X",
			},
			tokens: []string{
				"X",
			},
		},
		{
			name: "cannot-execute-command",
			in:   "CANNOT execute shell command; command is not allowed for rule=test_immunity",
			wantContains: []string{
				"Use permitted tools instead of executing shell command",
				"command requires explicit authorization",
			},
			tokens: []string{
				"rule=test_immunity",
			},
		},
		{
			name: "modify-only-implementation-files-forbidden",
			in:   "modify only implementation files; test modifications are forbidden",
			wantContains: []string{
				"modify implementation files in your assigned lane; preserve test files intact",
			},
			tokens: nil,
		},
		{
			name: "dos-toml-test-tamper-fix",
			in:   "Modify only implementation files in your assigned lane; test modifications are forbidden.",
			wantContains: []string{
				"Modify implementation files in your assigned lane; preserve test files intact.",
			},
			tokens: nil,
		},
		{
			name: "reasoncode-and-backticked-command",
			in:   "DENIED: cannot execute `rm -rf /` under rule=safe_exec; ReasonCode=1100 reason=POLICY_BLOCK",
			wantContains: []string{
				"ACTION REQUIRED:",
				"use permitted tools instead of executing `rm -rf /`",
			},
			tokens: []string{
				"`rm -rf /`",
				"rule=safe_exec",
				"ReasonCode=1100",
				"reason=POLICY_BLOCK",
			},
		},
		{
			name: "wildcard-and-file-preservation",
			in:   "ReasonCode: 1100 ReasonTestTamperRefused: CANNOT edit *_test.go; do not touch testdata/**",
			wantContains: []string{
				"Preserve *_test.go",
				"preserve testdata/**",
			},
			tokens: []string{
				"1100",
				"ReasonTestTamperRefused",
				"*_test.go",
				"testdata/**",
			},
		},
		{
			name: "cannot-modify-without-authorization",
			in:   "Tool call cannot modify cmd/fak/main.go without authorization; reason=OFF_TRUNK",
			wantContains: []string{
				"preserve cmd/fak/main.go with required authorization",
			},
			tokens: []string{
				"cmd/fak/main.go",
				"reason=OFF_TRUNK",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := RefusalReframe(tc.in)

			// 1. Check alias equivalence
			if alias := ReframeRefusalProse(tc.in); alias != got {
				t.Fatalf("ReframeRefusalProse(%q) = %q, want %q (same as RefusalReframe)", tc.in, alias, got)
			}

			// 2. Check desired phrasing
			for _, want := range tc.wantContains {
				if !strings.Contains(got, want) {
					t.Errorf("RefusalReframe(%q)\n  got:  %q\n  missing substring: %q", tc.in, got, want)
				}
			}

			// 3. Check machine-readable tokens are preserved byte-for-byte
			for _, tok := range tc.tokens {
				if !strings.Contains(got, tok) {
					t.Errorf("token dropped: %q not found in reframed prose %q", tok, got)
				}
			}

			// 4. Verify reframed prose scores clean under Classify (zero negative findings)
			findings := Classify("refusal-test", got)
			if len(findings) != 0 {
				var spans []string
				for _, f := range findings {
					spans = append(spans, string(f.Category)+":"+f.Span)
				}
				t.Errorf("reframed prose has negative findings under Classify: %v\n  text: %q", spans, got)
			}

			// 5. Verify zero negative-frame debt under ScoreDoc
			doc := ScoreDoc("refusal-test", got)
			if doc.Mechanical != 0 {
				t.Errorf("reframed prose has %d mechanical debt (want 0)\n  text: %q", doc.Mechanical, got)
			}
			if doc.Judgement != 0 {
				t.Errorf("reframed prose has %d judgement findings (want 0)\n  text: %q", doc.Judgement, got)
			}

			// 6. Verify idempotency: RefusalReframe(RefusalReframe(x)) == RefusalReframe(x)
			twice := RefusalReframe(got)
			if twice != got {
				t.Fatalf("RefusalReframe not idempotent:\n  once:  %q\n  twice: %q", got, twice)
			}
		})
	}
}

func TestRefusalReframeEmptyAndClean(t *testing.T) {
	cleanProse := "Proceed with implementation in your assigned lane; preserve test files intact."
	if got := RefusalReframe(cleanProse); got != cleanProse {
		t.Fatalf("clean prose was modified: got %q, want %q", got, cleanProse)
	}

	if got := RefusalReframe(""); got != "" {
		t.Fatalf("empty input was modified: got %q", got)
	}
}

func TestRefusalReframePassTelemetry(t *testing.T) {
	in := "DENIED: do not edit internal/witness/witness.go; reason=TEST_TAMPER_REFUSED"
	res := RefusalReframePass(in)
	if res.Applied == 0 {
		t.Fatalf("Applied = 0, want > 0 for reframed idioms")
	}
	if len(res.PreservedTokens) == 0 {
		t.Fatalf("PreservedTokens is empty, want preserved tokens")
	}
	if !strings.Contains(res.Text, "ACTION REQUIRED: preserve internal/witness/witness.go") {
		t.Fatalf("Text = %q, want ACTION REQUIRED: preserve...", res.Text)
	}
}
