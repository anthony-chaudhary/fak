package adjudicator

import "testing"

// TestScratchCarveOutSpellingAsymmetry pins WHICH spellings reach the scratchpad
// carve-out, because the answer differs by surface and the difference is invisible
// from a refusal.
//
// internal/session's deny-all breaker tells a blocked agent to re-target a recursive
// delete strictly inside a declared scratchpad root. On the PowerShell surface a
// Windows-spelled path gets there. On a POSIX surface it does NOT: rceShellSegments
// reads `\` as an escape, so `C:\scratch\sess\build` arrives as the single token
// `C:scratchsessbuild`, resolves under no root, and is refused a second time.
//
// The refusal is CORRECT and must stay — Git Bash would aim that delete at a
// different path than the one written, and admitting the INTENDED path while the
// shell executes another is the one move a containment decider must never make. What
// was wrong was only that nothing said so, so a correctly-followed remedy looked like
// a route that does not exist. This test is the standing record that the asymmetry is
// deliberate, so a future reader does not "fix" it by folding `\` to `/` here.
func TestScratchCarveOutSpellingAsymmetry(t *testing.T) {
	ws, ok := canonicalizeArgValue(`C:\work\fak`)
	if !ok {
		t.Fatal("canonicalizeArgValue rejected the workspace root")
	}
	root, ok := canonicalizeArgValue(`C:\agent-scratch\sess1`)
	if !ok {
		t.Fatal("canonicalizeArgValue rejected the scratchpad root")
	}
	scratch := []string{root}

	cases := []struct {
		name   string
		cmd    string
		denied bool
	}{
		// The route the breaker advertises, spelled as the receiving shell reads it.
		{"posix forward-slash target inside scratch", `rm -rf C:/agent-scratch/sess1/build`, false},
		{"posix nested target inside scratch", `rm -rf C:/agent-scratch/sess1/a/b`, false},
		// Same intent, Windows spelling: the backslashes are escapes, so this names
		// another path entirely and stays denied.
		{"posix backslash target cannot reach the carve-out", `rm -rf C:\agent-scratch\sess1\build`, true},
		// PowerShell operands keep their backslashes, so the route works there.
		{"powershell backslash target inside scratch", `Remove-Item -Recurse -Force C:\agent-scratch\sess1\build`, false},

		// The conditions that keep the deny, on the spelling that otherwise works.
		{"the scratch root itself is too broad", `rm -rf C:/agent-scratch/sess1`, true},
		{"a glob is not a literal target", `rm -rf C:/agent-scratch/sess1/*`, true},
		{"an unexpanded variable is not a literal target", `rm -rf $SCRATCH/build`, true},
		{"a dot-segment escape is clamped, not followed", `rm -rf C:/agent-scratch/sess1/../../work/fak`, true},
		{"one target outside the root denies the whole call", `rm -rf C:/agent-scratch/sess1/build C:/work/fak/internal`, true},

		// The fatal case the carve-out must never reach, on both surfaces.
		{"workspace tree stays denied on posix", `rm -rf C:/work/fak/internal`, true},
		{"workspace tree stays denied on powershell", `Remove-Item -Recurse -Force C:\work\fak\internal`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := commandHasUnsafeRecursiveForcedDelete(tc.cmd, ws, scratch); got != tc.denied {
				t.Errorf("commandHasUnsafeRecursiveForcedDelete(%q) = %v, want %v", tc.cmd, got, tc.denied)
			}
		})
	}
}
