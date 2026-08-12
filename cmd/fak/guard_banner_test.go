package main

import (
	"strings"
	"testing"
)

// TestGuardBannerModeDecision pins the --banner precedence: --quiet silences everything;
// an explicit mode wins over auto; AUTO plays the icon ANIMATION for the attended interactive
// launch (interactive stdin AND interactive child) and keeps the full report byte-for-byte for
// every headless/piped shape — the fleet-log contract; unknown values fail loud. (The animate
// mode degrades to the compact banner at the RENDER site off a color TTY / on opt-out; that
// fallback is guardLaunchAnimEnabled's job, tested separately.)
func TestGuardBannerModeDecision(t *testing.T) {
	for _, tc := range []struct {
		name                     string
		banner                   string
		quiet, stdinTTY, childUI bool
		want                     string
		wantErr                  bool
	}{
		{name: "auto attended interactive animates", banner: "auto", stdinTTY: true, childUI: true, want: guardBannerAnimate},
		{name: "auto piped stdin keeps full", banner: "auto", stdinTTY: false, childUI: true, want: guardBannerFull},
		{name: "auto headless -p child keeps full", banner: "auto", stdinTTY: true, childUI: false, want: guardBannerFull},
		{name: "empty means auto", banner: "", stdinTTY: true, childUI: true, want: guardBannerAnimate},
		{name: "explicit full wins over attended auto-animate", banner: "full", stdinTTY: true, childUI: true, want: guardBannerFull},
		{name: "explicit compact wins over headless auto-full", banner: "compact", stdinTTY: false, childUI: false, want: guardBannerCompact},
		{name: "explicit animate honored headless", banner: "animate", stdinTTY: false, childUI: false, want: guardBannerAnimate},
		{name: "explicit off", banner: "off", stdinTTY: true, childUI: true, want: guardBannerOff},
		{name: "quiet wins over explicit full", banner: "full", quiet: true, stdinTTY: true, childUI: true, want: guardBannerOff},
		{name: "quiet wins over attended auto-animate", banner: "auto", quiet: true, stdinTTY: true, childUI: true, want: guardBannerOff},
		{name: "case and space tolerated", banner: "  Full ", stdinTTY: true, childUI: true, want: guardBannerFull},
		{name: "unknown value fails loud", banner: "verbose", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := guardBannerModeDecision(tc.banner, tc.quiet, tc.stdinTTY, tc.childUI)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("want an error for %q, got mode %q", tc.banner, got)
				}
				if !strings.Contains(err.Error(), "--banner") {
					t.Errorf("error must name the flag: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("mode = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestPrintGuardCompactBannerIsCompact is the render-witness for the whole feature: the
// attended banner stays THREE lines (the wall of text was the reported problem), keeps
// the identity + gateway URL, and carries a copy-pasteable `fak info --startup` command
// pointing at THIS session's gateway — the on-demand door to the suppressed detail.
func TestPrintGuardCompactBannerIsCompact(t *testing.T) {
	var b strings.Builder
	printGuardCompactBanner(&b, "9.9.9", "abc123de+", "http://127.0.0.1:9", []string{"claude"}, nil)
	out := b.String()

	if n := strings.Count(out, "\n"); n != 3 {
		t.Fatalf("compact banner is %d lines, want exactly 3 (that is the point):\n%s", n, out)
	}
	for _, want := range []string{
		"fak guard 9.9.9 (abc123de+) — kernel-adjudicated: claude",
		"gateway http://127.0.0.1:9",
		"fak info --startup --gateway-url http://127.0.0.1:9",
		"--banner=full",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("compact banner missing %q:\n%s", want, out)
		}
	}
}

// TestPrintGuardCompactBannerKeepsHardBudget proves stale refusal history cannot
// grow the attended pre-child surface. Complete refusal detail remains in the
// startup report addressed by the third line.
func TestPrintGuardCompactBannerKeepsHardBudget(t *testing.T) {
	var b strings.Builder
	printGuardCompactBanner(&b, "9.9.9", "", "http://127.0.0.1:9", []string{"claude"},
		[]guardRefusalCarry{{Reason: "OFF_TRUNK", Count: 99, Fix: "commit directly to main"}})
	out := b.String()
	if n := strings.Count(out, "\n"); n != 3 {
		t.Fatalf("compact banner with refusal history is %d lines, want exactly 3:\n%s", n, out)
	}
	if strings.Contains(out, "OFF_TRUNK") || strings.Contains(out, "commit directly") {
		t.Fatalf("compact banner leaked prior refusal history into child scrollback:\n%s", out)
	}
}
