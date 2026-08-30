package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestGuardBannerModeDecision pins the --banner precedence: --quiet silences everything;
// an explicit mode wins over auto; AUTO/empty resolve to the private progress-only mode for
// interactive and noninteractive launches; unknown and internal values fail loud.
func TestGuardBannerModeDecision(t *testing.T) {
	for _, tc := range []struct {
		name                     string
		banner                   string
		quiet, stdinTTY, childUI bool
		want                     string
		wantErr                  bool
	}{
		{name: "auto attended interactive uses progress", banner: "auto", stdinTTY: true, childUI: true, want: guardBannerProgress},
		{name: "auto piped stdin uses progress", banner: "auto", stdinTTY: false, childUI: true, want: guardBannerProgress},
		{name: "auto headless child uses progress", banner: "auto", stdinTTY: true, childUI: false, want: guardBannerProgress},
		{name: "auto fully noninteractive uses progress", banner: "auto", stdinTTY: false, childUI: false, want: guardBannerProgress},
		{name: "empty interactive means progress", banner: "", stdinTTY: true, childUI: true, want: guardBannerProgress},
		{name: "empty noninteractive means progress", banner: "", stdinTTY: false, childUI: false, want: guardBannerProgress},
		{name: "explicit full wins over auto", banner: "full", stdinTTY: true, childUI: true, want: guardBannerFull},
		{name: "explicit compact wins over auto", banner: "compact", stdinTTY: false, childUI: false, want: guardBannerCompact},
		{name: "explicit animate honored headless", banner: "animate", stdinTTY: false, childUI: false, want: guardBannerAnimate},
		{name: "explicit off", banner: "off", stdinTTY: true, childUI: true, want: guardBannerOff},
		{name: "quiet wins over explicit full", banner: "full", quiet: true, stdinTTY: true, childUI: true, want: guardBannerOff},
		{name: "quiet wins over auto progress", banner: "auto", quiet: true, stdinTTY: true, childUI: true, want: "off"},
		{name: "case and space tolerated", banner: "  Full ", stdinTTY: true, childUI: true, want: guardBannerFull},
		{name: "private progress value is not selectable", banner: "progress", wantErr: true},
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

func TestGuardStartupProgressAndLaunchFailureDecisions(t *testing.T) {
	if !guardDumpStartupOnLaunchFail(guardBannerProgress) {
		t.Error("default progress mode must spill the full report on launch failure")
	}
	if guardDumpStartupOnLaunchFail(guardBannerFull) {
		t.Error("explicit full must not duplicate the report on launch failure")
	}
}

func TestGuardDefaultLaunchFailureSpillsFullReport(t *testing.T) {
	const report = "fak guard FULL STARTUP REPORT\nresponse profile\nwork profile\nidentity/configuration\n"
	for _, tc := range []struct {
		name                       string
		banner                     string
		stdinTTY, childInteractive bool
	}{
		{name: "auto interactive", banner: "auto", stdinTTY: true, childInteractive: true},
		{name: "auto noninteractive", banner: "auto"},
		{name: "empty interactive", stdinTTY: true, childInteractive: true},
		{name: "empty noninteractive"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mode, err := guardBannerModeDecision(tc.banner, false, tc.stdinTTY, tc.childInteractive)
			if err != nil {
				t.Fatal(err)
			}
			var out strings.Builder
			guardWriteLaunchFailReport(&out, report, guardDumpStartupOnLaunchFail(mode))
			if !strings.Contains(out.String(), "launch failed") || !strings.Contains(out.String(), report) {
				t.Fatalf("default launch failure did not spill full report: %q", out.String())
			}
		})
	}
}

// TestGuardHealthyDefaultLaunchKeepsStructuredProfilesOffStderr is the captured-render
// witness for #10187. The real guard entry point still resolves and injects the default
// profiles into a supported child, but a healthy AUTO launch must not dump their structured
// JSON capture into the child's terminal scrollback.
func TestGuardHealthyDefaultLaunchKeepsStructuredProfilesOffStderr(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the helper needs a basename recognized as claude; Windows batch shims retain .bat")
	}
	childDir := t.TempDir()
	child := filepath.Join(childDir, "claude")
	if err := os.WriteFile(child, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	code, out, timedOut := runGuardE2E(t,
		"--provider openai --base-url http://127.0.0.1:9 --audit off --lease mode=off --precompact-hook off --deny-all-continue off --toolproc-hooks off --task-handoff off --operator-directed off --mcp-register=false --fleet-bus=false --resource-stats=false --debug-stats=false -- "+child,
		map[string]string{"CLAUDE_CONFIG_DIR": t.TempDir(), "FAK_USAGE_LOG": "off"},
	)
	if timedOut || code != 0 {
		t.Fatalf("healthy default guard launch code=%d timedOut=%v\n%s", code, timedOut, out)
	}
	for _, unwanted := range []string{"response-profile {", `"schema": "fak.guard-profiles.v2"`} {
		if strings.Contains(out, unwanted) {
			t.Fatalf("healthy default launch leaked raw profile capture %q:\n%s", unwanted, out)
		}
	}
}

// TestWriteGuardStartupBannerExplicitModes pins the existing explicit render semantics while
// proving the internal progress mode emits no banner bytes. Animate is exercised through its
// established non-TTY compact fallback; its TTY frames are witnessed in guard_launch_anim_test.
func TestWriteGuardStartupBannerExplicitModes(t *testing.T) {
	const report = "FULL STARTUP REPORT\nresponse profile\nwork profile\nidentity/configuration\n"
	view := guardStartupView{gwURL: "http://127.0.0.1:9", command: []string{"codex"}}

	for _, tc := range []struct {
		name, mode string
		want       string
		silent     bool
	}{
		{name: "full", mode: guardBannerFull, want: report},
		{name: "compact", mode: guardBannerCompact, want: "fak info --startup"},
		{name: "animate non-TTY fallback", mode: guardBannerAnimate, want: "fak info --startup"},
		{name: "off", mode: "off", silent: true},
		{name: "progress", mode: guardBannerProgress, silent: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out strings.Builder
			view.bannerMode = tc.mode
			writeGuardStartupBanner(&out, view, report, false, false, "", 80)
			if tc.silent {
				if out.Len() != 0 {
					t.Fatalf("mode %q emitted banner bytes: %q", tc.mode, out.String())
				}
				return
			}
			if !strings.Contains(out.String(), tc.want) {
				t.Fatalf("mode %q output %q missing %q", tc.mode, out.String(), tc.want)
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
