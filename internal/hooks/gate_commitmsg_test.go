package hooks

import "testing"

// TestSuggestGradeableSubject_correctsDeterministicFailures asserts that the two DETERMINISTIC
// gradeability failures — a near-miss conventional type and an inflected leading verb — earn a
// concrete, self-verified rewrite, while any case that would require a guess earns "".
func TestSuggestGradeableSubject_correctsDeterministicFailures(t *testing.T) {
	cases := []struct {
		name    string
		subject string
		want    string
	}{
		// Near-miss type: the description verb is already fine, only the type is wrong.
		{"feature->feat", "feature(gateway): add the reclaim path", "feat(gateway): add the reclaim path"},
		{"fixes-type->fix", "fixes(policy): correct the rule", "fix(policy): correct the rule"},
		{"documentation->docs", "documentation: clarify the runbook", "docs: clarify the runbook"},
		{"tests->test", "tests(gateway): cover the slot path", "test(gateway): cover the slot path"},
		{"bang preserved", "feature(api)!: add the breaking flag", "feat(api)!: add the breaking flag"},

		// Inflected leading verb: type is valid, the verb just needs its imperative base.
		{"past-ed", "feat(gateway): added a retry", "feat(gateway): add a retry"},
		{"past-wired", "feat(gateway): wired the seam", "feat(gateway): wire the seam"},
		{"gerund-caching", "perf(cache): caching the results", "perf(cache): cache the results"},
		{"gerund-wiring", "feat(x): wiring the panel", "feat(x): wire the panel"},
		{"third-person-fixes", "fix(x): fixes the leak", "fix(x): fix the leak"},
		{"doubled-pinning", "refactor(x): pinning the default", "refactor(x): pin the default"},

		// Both wrong at once: near-miss type AND inflected verb.
		{"type+verb", "feature(gateway): added a retry", "feat(gateway): add a retry"},

		// No safe suggestion — must stay "".
		{"empty", "", ""},
		{"no-conventional-prefix", "fixed the parser crash", ""},
		{"unknown-type-no-correction", "improvement(x): add a thing", ""},
		{"genuinely-noun-led", "feat(gateway): posture improvements", ""},
		{"decorated-lead", "feat(x): `added` a retry", ""},
		{"already-gradeable", "feat(gateway): add a retry", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := suggestGradeableSubject(tc.subject)
			if got != tc.want {
				t.Fatalf("suggestGradeableSubject(%q) = %q, want %q", tc.subject, got, tc.want)
			}
			// Every non-empty suggestion must actually be gradeable — the self-verify contract.
			if got != "" {
				if ok, why := CommitMsgVerdict(got); !ok {
					t.Fatalf("suggested subject %q is not gradeable: %s", got, why)
				}
			}
		})
	}
}

// TestImperativeBase_membershipDecides spot-checks the over-generative base resolver: a form that
// derives from a recognized verb resolves, an unrelated noun does not.
func TestImperativeBase_membershipDecides(t *testing.T) {
	cases := map[string]string{
		"added":   "add",
		"caching": "cache",
		"wiring":  "wire",
		"fixes":   "fix",
		"pinning": "pin",
		"built":   "build",
		"posture": "", // noun, no verb base
		"the":     "", // stopword
		"retry":   "", // noun
		"add":     "add",
	}
	for in, want := range cases {
		if got := imperativeBase(in); got != want {
			t.Errorf("imperativeBase(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestLintCommitMessage_gradeabilitySuggestionComposesTrailer is the end-to-end proof: a subject
// that is BOTH non-gradeable (wrong type) AND unstamped earns a single suggested subject that fixes
// both — the gradeable rewrite with the path-implied trailer appended.
func TestLintCommitMessage_gradeabilitySuggestionComposesTrailer(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("feature(gateway): added the slot reclaim path", []string{"internal/gateway/server.go"}, root)
	if r.OK {
		t.Fatalf("expected NOT ok for a non-gradeable, unstamped subject")
	}
	if r.Gradeable {
		t.Fatalf("subject leads with an unknown type; should not be gradeable")
	}
	want := "feat(gateway): add the slot reclaim path (fak gateway)"
	if r.SuggestedSubject != want {
		t.Fatalf("SuggestedSubject = %q, want %q", r.SuggestedSubject, want)
	}
	// The composed suggestion must itself pass the gate.
	if ok, why := CommitMsgVerdict(r.SuggestedSubject); !ok {
		t.Fatalf("composed suggestion %q is not gradeable: %s", r.SuggestedSubject, why)
	}
}

// TestLintCommitMessage_nounLedNoFalseSuggestion guards the negative: a genuinely noun-led subject
// (no deterministic fix) gets prose advice, not a fabricated rewrite.
func TestLintCommitMessage_nounLedNoFalseSuggestion(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("feat(gateway): posture improvements", []string{"internal/gateway/server.go"}, root)
	if r.Gradeable {
		t.Fatalf("noun-led description should not be gradeable")
	}
	if r.SuggestedSubject != "" {
		t.Fatalf("must not fabricate a rewrite for a noun-led subject, got %q", r.SuggestedSubject)
	}
	if !hasIssueContaining(r, "witness-gradeable") {
		t.Fatalf("want the witness-gradeable prose advice, got %v", r.Issues)
	}
}
