package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// The reasoning-posture vocabulary is spelled in three places on purpose: the interactive
// launcher (accounts_launch.go), the guard posture resolver (guard_effort.go), and the pure
// dispatch leaf (internal/dispatchtick), which mirrors BY VALUE rather than importing cmd/fak so
// it stays modelroute+stdlib clean. Mirrors drift silently, so these are the drift guards the
// mirror comments promise. A divergence here is not cosmetic: the dispatch leaf hoists the
// launcher's inline --settings payload by exact string match, and guard recognizes an
// operator-pinned posture by exact flag value — a one-character drift turns both into silent
// no-ops that hand a fleet worker the wrong posture, or strip guard's hook stack.
func TestUltracodeVocabularyMirrorsDoNotDrift(t *testing.T) {
	if ultracodeSettingsArg != dispatchtick.UltracodeSettingsArg {
		t.Fatalf("ultracode --settings payload drifted: launcher %q vs dispatchtick %q",
			ultracodeSettingsArg, dispatchtick.UltracodeSettingsArg)
	}
	if guardEffortModeUltracode != dispatchtick.GuardEffortUltracode {
		t.Fatalf("guard ultracode posture value drifted: guard %q vs dispatchtick %q",
			guardEffortModeUltracode, dispatchtick.GuardEffortUltracode)
	}
	// dispatchtick relays the posture through guard's OWN flag name, so the flag it emits must
	// be one `fak guard` actually parses.
	if got := dispatchtick.GuardEffortFlag; got != "--effort" {
		t.Fatalf("dispatchtick relays %q, but guard's posture flag is --effort", got)
	}
	// xhigh is the agent-session default, and it must stay a level the child CLI admits
	// (low|medium|high|xhigh|max) — "ultracode" is NOT an --effort level, which is exactly why
	// it travels as a settings key instead.
	if guardEffortLevelXHigh != dispatchtick.EffortXHigh {
		t.Fatalf("xhigh effort level drifted: guard %q vs dispatchtick %q",
			guardEffortLevelXHigh, dispatchtick.EffortXHigh)
	}
}
