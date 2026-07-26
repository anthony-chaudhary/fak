package main

import (
	"strings"
	"testing"
)

// guard_output_style_test.go covers the ACT half of the closing-shape rung:
// runGuardOutputStyleGate and its mode-resolution wiring, plus the applyClosingSignal
// sensor. The fold itself (internal/headlesslint.ScanClosing) is tested in that package;
// here we pin the guard-side vocabulary, the OFF-by-default cap, and the gate ladder.

// proseWallClose is a trailing prose paragraph well over the fold's 40-word threshold with
// no list marker — the shape ScanClosing refuses.
const proseWallClose = "I went through the authentication module and the session handling code and the token refresh path and reviewed each of the error branches carefully, then I updated the retry behavior and the backoff timing and confirmed the changes all behave the way the existing tests expect them to now, so everything looks consistent."

// TestNormalizeGuardOutputStyleMode covers the closed vocabulary: empty defaults to OFF (the
// shipped default, unlike the operator-directed gate's warn), warn/shadow/enforce are honored,
// and anything else errors.
func TestNormalizeGuardOutputStyleMode(t *testing.T) {
	cases := []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"", guardPreCompactModeOff, false},
		{"  ", guardPreCompactModeOff, false},
		{"OFF", guardPreCompactModeOff, false},
		{"shadow", guardPreCompactModeShadow, false},
		{"WARN", guardOutputStyleModeWarn, false},
		{"enforce", guardPreCompactModeEnforce, false},
		{"loud", "", true},
	}
	for _, tc := range cases {
		got, err := normalizeGuardOutputStyleMode(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("normalizeGuardOutputStyleMode(%q): want error, got %q", tc.in, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("normalizeGuardOutputStyleMode(%q): unexpected error %v", tc.in, err)
		}
		if got != tc.want {
			t.Errorf("normalizeGuardOutputStyleMode(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestGuardOutputStyleNormalizedOrOff pins the total env-boundary form: a bad string falls back
// to OFF (never enforce), and a good value round-trips.
func TestGuardOutputStyleNormalizedOrOff(t *testing.T) {
	if got := guardOutputStyleNormalizedOrOff("loud"); got != guardPreCompactModeOff {
		t.Errorf("guardOutputStyleNormalizedOrOff(bad) = %q, want off", got)
	}
	if got := guardOutputStyleNormalizedOrOff("enforce"); got != guardPreCompactModeEnforce {
		t.Errorf("guardOutputStyleNormalizedOrOff(enforce) = %q, want enforce", got)
	}
}

// TestGuardOutputStyleEffectiveMode covers the OFF-by-default resolution and the operator-absent
// cap: an unconfigured session is off; an explicit enforce is honored only for a headless child
// and capped to warn for an attended interactive one.
func TestGuardOutputStyleEffectiveMode(t *testing.T) {
	cases := []struct {
		name             string
		configured       string
		explicitlySet    bool
		childInteractive bool
		want             string
	}{
		{"unset headless -> off", guardPreCompactModeEnforce, false, false, guardPreCompactModeOff},
		{"unset interactive -> off", guardPreCompactModeEnforce, false, true, guardPreCompactModeOff},
		{"explicit enforce headless -> enforce", guardPreCompactModeEnforce, true, false, guardPreCompactModeEnforce},
		{"explicit enforce interactive -> warn", guardPreCompactModeEnforce, true, true, guardOutputStyleModeWarn},
		{"explicit shadow headless -> shadow", guardPreCompactModeShadow, true, false, guardPreCompactModeShadow},
		{"explicit warn interactive -> warn", guardOutputStyleModeWarn, true, true, guardOutputStyleModeWarn},
		{"explicit off headless -> off", guardPreCompactModeOff, true, false, guardPreCompactModeOff},
		{"bad string set headless -> off", "loud", true, false, guardPreCompactModeOff},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := guardOutputStyleEffectiveMode(tc.configured, tc.explicitlySet, tc.childInteractive)
			if got != tc.want {
				t.Errorf("guardOutputStyleEffectiveMode(%q, set=%v, interactive=%v) = %q, want %q",
					tc.configured, tc.explicitlySet, tc.childInteractive, got, tc.want)
			}
		})
	}
}

// wallTranscript is a stop transcript whose final turn closed on a prose wall.
func wallTranscript() *guardStopTranscript {
	return &guardStopTranscript{
		Read:             true,
		ClosingProseWall: true,
		ClosingResolve:   "re-cast the closing as bullets, verdict first; put the next checkable step as the final bullet",
	}
}

// TestRunGuardOutputStyleGate walks the ladder: off/nil/non-wall are inert; shadow and warn allow
// and soak; enforce blocks and feeds the remediation back; a bad mode fails open.
func TestRunGuardOutputStyleGate(t *testing.T) {
	cleanTr := &guardStopTranscript{Read: true} // no ClosingProseWall
	cases := []struct {
		name      string
		mode      string
		tr        *guardStopTranscript
		wantExit  int
		wantDisp  guardStopDisposition
		wantFired bool
		wantText  string
	}{
		{"off is inert", guardPreCompactModeOff, wallTranscript(), 0, "", false, ""},
		{"nil transcript inert", guardPreCompactModeEnforce, nil, 0, "", false, ""},
		{"clean close inert", guardPreCompactModeEnforce, cleanTr, 0, "", false, ""},
		{"bad mode fails open", "loud", wallTranscript(), 0, "", false, ""},
		{"shadow allows", guardPreCompactModeShadow, wallTranscript(), 0, stopDispOutputStyleShadow, true, "shadow"},
		{"warn allows + soaks", guardOutputStyleModeWarn, wallTranscript(), 0, stopDispOutputStyleWarn, true, "soak mode"},
		{"enforce blocks", guardPreCompactModeEnforce, wallTranscript(), 2, stopDispOutputStyleContinue, true, "prose wall"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stderr strings.Builder
			exit, disp, fired := runGuardOutputStyleGate(&stderr, tc.mode, tc.tr)
			if exit != tc.wantExit || disp != tc.wantDisp || fired != tc.wantFired {
				t.Errorf("runGuardOutputStyleGate(%q) = (%d, %q, %v), want (%d, %q, %v)",
					tc.mode, exit, disp, fired, tc.wantExit, tc.wantDisp, tc.wantFired)
			}
			if tc.wantText != "" && !strings.Contains(stderr.String(), tc.wantText) {
				t.Errorf("runGuardOutputStyleGate(%q) stderr = %q, want substring %q", tc.mode, stderr.String(), tc.wantText)
			}
		})
	}
}

// TestRunGuardOutputStyleGateBlankResolve pins the defensive fallback: an enforce block still
// produces a usable continue message when the sensor left ClosingResolve blank.
func TestRunGuardOutputStyleGateBlankResolve(t *testing.T) {
	var stderr strings.Builder
	blank := &guardStopTranscript{Read: true, ClosingProseWall: true}
	exit, disp, fired := runGuardOutputStyleGate(&stderr, guardPreCompactModeEnforce, blank)
	if exit != 2 || disp != stopDispOutputStyleContinue || !fired {
		t.Fatalf("blank-resolve enforce = (%d, %q, %v), want (2, continue, true)", exit, disp, fired)
	}
	if !strings.Contains(stderr.String(), "re-cast the closing as bullets") {
		t.Errorf("blank-resolve message missing generic remediation: %q", stderr.String())
	}
}

// TestApplyClosingSignal pins the SENSOR: a final turn that closes on a prose wall stamps
// ClosingProseWall + ClosingResolve; a scannable close leaves both zero.
func TestApplyClosingSignal(t *testing.T) {
	wall := &guardStopTranscript{Read: true}
	applyClosingSignal(wall, proseWallClose)
	if !wall.ClosingProseWall || strings.TrimSpace(wall.ClosingResolve) == "" {
		t.Errorf("prose wall not sensed: proseWall=%v resolve=%q", wall.ClosingProseWall, wall.ClosingResolve)
	}

	clean := &guardStopTranscript{Read: true}
	applyClosingSignal(clean, "Done. Pushed abc123.")
	if clean.ClosingProseWall || clean.ClosingResolve != "" {
		t.Errorf("short close falsely sensed as a wall: proseWall=%v resolve=%q", clean.ClosingProseWall, clean.ClosingResolve)
	}

	bulleted := &guardStopTranscript{Read: true}
	applyClosingSignal(bulleted, "- Verdict: shipped, tests green.\n- Next: run make ci before the push.")
	if bulleted.ClosingProseWall {
		t.Errorf("bulleted close falsely sensed as a wall")
	}
}
