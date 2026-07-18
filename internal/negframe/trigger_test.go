package negframe

import (
	"testing"
)

func TestTriggerLexicon(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		text     string
		token    string
		category Category
	}{
		{"not", "Select the mode that is not red.", "not", Prohibition},
		{"never", "Never expose the credential.", "never", Prohibition},
		{"avoid", "Avoid destructive operations.", "avoid", Prohibition},
		{"only", "Use only signed artifacts.", "only", Exception},
		{"except", "Allow every region except west.", "except", Exception},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Trigger(tt.text)
			if !got.Negation || got.Token != tt.token || got.Category != tt.category {
				t.Fatalf("Trigger(%q) = %+v, want token %q category %q", tt.text, got, tt.token, tt.category)
			}
		})
	}
}

func TestTriggerBenignAndCodeFixtures(t *testing.T) {
	t.Parallel()
	for _, text := range []string{
		"The notebook is on the desk.",
		"The avoidance score improved.",
		"`if value != nil { return }`",
		"Run `git show --format=only` for the fixture.",
		"```go\nif !allowed { panic(\"not allowed\") }\n```",
		"~~~text\nnever only except avoid not\n~~~",
	} {
		if got := Trigger(text); got.Negation {
			t.Errorf("Trigger(%q) = %+v, want no trigger", text, got)
		}
	}
}

func TestTriggerUsesSharedLexicon(t *testing.T) {
	t.Parallel()
	// "without" is intentionally present only in the established document
	// lexicon. Detecting it proves Trigger consumes that shared table.
	got := Trigger("Complete the handoff without losing the witness.")
	if !got.Negation || got.Token != "without" || got.Category != Absence {
		t.Fatalf("Trigger(shared rule) = %+v", got)
	}
}

func TestTriggerRegexpsArePrecompiled(t *testing.T) {
	for i, r := range append(append([]reframeRule(nil), rules...), triggerOnlyRules...) {
		if r.Pattern == nil {
			t.Fatalf("rule %d has nil pattern", i)
		}
	}
	if raceDetectorEnabled {
		// The race detector instruments every memory access, which inflates the
		// count testing.AllocsPerRun observes (~27 vs the real 2), so the steady-
		// state allocation budget is only meaningful in a non-race build. The
		// precompiled-pattern check above still runs under -race.
		t.Skip("allocation budget is not meaningful under go test -race instrumentation")
	}
	if allocs := testing.AllocsPerRun(100, func() { _ = Trigger("ordinary positive state") }); allocs > 4 {
		t.Fatalf("Trigger allocs = %.1f, want <= 4", allocs)
	}
}
