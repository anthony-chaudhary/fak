package headlesslint

import "sort"

// FixtureCase is one labeled sample of agent final-output text and what the
// scanner must say about it: whether it is headless-clean, and (when not) which
// Classes must appear. The corpus spans honest headless-complete outputs and
// the operator-directed shapes across every Class.
type FixtureCase struct {
	Name   string  `json:"name"`
	Text   string  `json:"text"`
	Clean  bool    `json:"clean"`
	Expect []Class `json:"expect,omitempty"`
}

// FixtureOutcome is the graded result of one FixtureCase.
type FixtureOutcome struct {
	Name    string  `json:"name"`
	Clean   bool    `json:"expect_clean"`
	Got     string  `json:"got_verdict"`
	Classes []Class `json:"got_classes"`
	OK      bool    `json:"ok"`
}

// Fixture is the built-in corpus, asserted by headlesslint_test.go and runnable
// from the CLI via `fak headless-lint --self-test` — the scanner's own
// re-derivable witness.
func Fixture() []FixtureCase {
	return []FixtureCase{
		// --- headless-clean: an autonomous worker that acted, not asked ---
		{
			Name:  "clean-shipped",
			Text:  "Implemented the parser and committed as abc123. Tests pass (go test ./internal/parser). Pushed to main.",
			Clean: true,
		},
		{
			Name:  "clean-honest-escalation",
			Text:  "The deploy step needs a production credential this environment does not hold. Filed #4821 to wire it and routed the blocker to the release agent.",
			Clean: true,
		},
		{
			Name:  "clean-not-yet-ticketed",
			Text:  "The migration is not yet complete; the remaining table is tracked in #900.",
			Clean: true,
		},
		{
			Name:  "clean-ticketed-todo",
			Text:  "Wired the core path. TODO(#901): handle the timeout case in the follow-up.",
			Clean: true,
		},

		// --- operator-directed: one shape per Class ---
		{
			Name:   "permission-push",
			Text:   "The change is ready. Do you want me to push it?",
			Expect: []Class{PermissionAsk},
		},
		{
			Name:   "preference-sync-async",
			Text:   "Both approaches work. Which would you prefer, the sync or async version?",
			Expect: []Class{PreferenceAsk},
		},
		{
			Name:   "clarification-retries",
			Text:   "I started the API client. Could you clarify whether you want retries enabled?",
			Expect: []Class{ClarificationRequest},
		},
		{
			Name:   "review-request",
			Text:   "Updated the module. Please review the changes and let me know if this looks right.",
			Expect: []Class{ReviewRequest},
		},
		{
			Name:   "confirmation-wait",
			Text:   "The build is staged. I'll wait for your confirmation before continuing.",
			Expect: []Class{ConfirmationWait},
		},
		{
			Name:   "deferred-todo",
			Text:   "Wired the happy path. TODO: handle the timeout case later.",
			Expect: []Class{DeferredWork},
		},
		{
			Name:   "suggestion-punt",
			Text:   "Added the endpoint. You may want to add rate limiting at some point.",
			Expect: []Class{SuggestionPunt},
		},
		{
			Name:   "open-offer-docs",
			Text:   "I updated the config. Let me know if you'd like me to also update the docs.",
			Expect: []Class{OpenOffer},
		},
	}
}

// RunFixture grades the corpus: a case passes when its verdict matches Clean and
// (when dirty) every expected Class appears. Returns the per-case outcomes and
// the number that passed.
func RunFixture() (cases []FixtureOutcome, passed int) {
	for _, fc := range Fixture() {
		rep := Scan(fc.Text)
		got := make([]Class, 0, len(rep.Classes))
		for c := range rep.Classes {
			got = append(got, c)
		}
		sort.Slice(got, func(i, j int) bool { return got[i] < got[j] })

		ok := (rep.Verdict == Clean) == fc.Clean
		if ok && !fc.Clean {
			for _, want := range fc.Expect {
				if rep.Classes[want] == 0 {
					ok = false
					break
				}
			}
		}
		cases = append(cases, FixtureOutcome{
			Name:    fc.Name,
			Clean:   fc.Clean,
			Got:     rep.Verdict,
			Classes: got,
			OK:      ok,
		})
		if ok {
			passed++
		}
	}
	return cases, passed
}
