package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cloudroute"
)

// scratchRootsHas reports whether the resolved FAK_GUARD_SCRATCHPAD_ROOTS list
// contains want as a whole entry (not a substring of a longer root).
func scratchRootsHas(got, want string) bool {
	for _, r := range strings.Split(got, string(os.PathListSeparator)) {
		if strings.EqualFold(strings.TrimSpace(r), want) {
			return true
		}
	}
	return false
}

func TestGuardCapabilityFloorDefaultsScratchpadRoot(t *testing.T) {
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", "")
	loadGuardCapabilityFloor("")
	want := filepath.Join(os.TempDir(), "claude")
	got := os.Getenv("FAK_GUARD_SCRATCHPAD_ROOTS")
	if !scratchRootsHas(got, want) {
		t.Fatalf("scratchpad roots=%q want to contain %q", got, want)
	}
	// The default must stay the narrow Claude scratch tree — never the OS temp
	// directory itself, which every root here sits strictly below.
	for _, r := range strings.Split(got, string(os.PathListSeparator)) {
		if strings.EqualFold(strings.TrimRight(strings.ReplaceAll(r, `\`, "/"), "/"),
			strings.TrimRight(strings.ReplaceAll(os.TempDir(), `\`, "/"), "/")) {
			t.Fatalf("scratchpad roots=%q must not declare the OS temp directory itself", got)
		}
	}
}

func TestGuardCapabilityFloorPreservesScratchpadOverride(t *testing.T) {
	want := filepath.Join(t.TempDir(), "session-scratch")
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", want)
	loadGuardCapabilityFloor("")
	if got := os.Getenv("FAK_GUARD_SCRATCHPAD_ROOTS"); !scratchRootsHas(got, want) {
		t.Fatalf("scratchpad override=%q want to contain %q", got, want)
	}
}

// TestScratchpadRootCarriesBothHostSpellings pins the fix for the defect that made
// the recursive-delete scratch carve-out the largest remaining refusal class in the
// guard-audit corpus (49 of 103 POLICY_BLOCKs, all dated after the carve-out
// shipped). The gates prove containment by string comparison, so a root declared
// only as `C:/…` could never match the `/c/…` spelling Git Bash — the shell behind
// the Bash tool — uses for that identical directory. A live probe of one throwaway
// directory inside the session scratchpad reproduced it: `/c/…` was hard-denied,
// byte-equivalent `C:/…` fell through to the preview-confirm gate.
func TestScratchpadRootCarriesBothHostSpellings(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("drive-letter/MSYS spelling duality is Windows-only")
	}
	got := guardScratchpadRootsValue(`C:\agent-scratch\claude`)
	for _, want := range []string{`C:\agent-scratch\claude`, "/c/agent-scratch/claude"} {
		if !scratchRootsHas(got, want) {
			t.Errorf("roots=%q want to contain %q", got, want)
		}
	}
	// Aliasing is symmetric: a root declared in the Git Bash spelling must also
	// cover a delete a PowerShell-backed surface spells with the drive letter.
	got = guardScratchpadRootsValue("/c/agent-scratch/claude")
	if !scratchRootsHas(got, "C:/agent-scratch/claude") {
		t.Errorf("roots=%q want to contain the drive-letter alias", got)
	}
}

// TestScratchpadRootAliasNeverAddsADirectory is the safety half: an alias is a
// second NAME for an already-declared root, never an extra directory. Every alias
// must therefore rewrite only the root prefix and keep the trailing path intact —
// if one ever mapped to a shorter or different tree it would silently widen the
// carve-out past what the operator declared.
func TestScratchpadRootAliasNeverAddsADirectory(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{`C:\agent-scratch\claude`, "/c/agent-scratch/claude"},
		{"C:/agent-scratch/claude/", "/c/agent-scratch/claude"},
		{"/c/agent-scratch/claude", "C:/agent-scratch/claude"},
		{"relative/not/a/root", ""},
		{"/agent-scratch/claude", ""}, // no drive component: nothing to alias
		{"", ""},
	} {
		got := scratchpadRootAlias(tc.in)
		if runtime.GOOS != "windows" {
			if got != "" {
				t.Errorf("scratchpadRootAlias(%q)=%q, want %q off Windows — a `C:`-prefixed alias would split a POSIX ':'-separated list into a bogus top-level root", tc.in, got, "")
			}
			continue
		}
		if got != tc.want {
			t.Errorf("scratchpadRootAlias(%q)=%q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestGuardStartupReportCarriesTheEnterprisePosture pins the two #8172 lines onto
// the DURABLE report rather than a stderr line a compact or --quiet launch drops.
// This is the one artifact a running session can be asked about for its whole life
// (`fak info --startup`, startup_report on /debug/vars), so it is where "which trust
// store am I validating with" and "is my model traffic adjudicated at all" have to
// be answerable. The waiver line matters most: every other line in this report
// describes adjudication fak IS performing, and a banner that stays silent about a
// waived request-signed route is the report lying by omission.
func TestGuardStartupReportCarriesTheEnterprisePosture(t *testing.T) {
	const note = "fak guard: upstream trust — validating against the platform trust store PLUS /etc/corp/ca-bundle.pem (1 certificate(s): Example Corp Root CA)\n"
	got := renderGuardStartupReport(guardStartupView{upstreamTrustNote: note, cloudRouteWaived: true})
	if !strings.Contains(got, note) {
		t.Fatalf("startup report dropped the trust-source line.\nwant: %q\ngot:\n%s", note, got)
	}
	if !strings.Contains(got, cloudroute.WaiverKey) {
		t.Fatalf("startup report never names %s, so an operator cannot tell the waiver is in force.\ngot:\n%s", cloudroute.WaiverKey, got)
	}
	for _, want := range []string{"NONE for model traffic", "fak serve --stdio --policy FILE"} {
		if !strings.Contains(got, want) {
			t.Fatalf("waived-route line missing %q — it must say what is unadjudicated and which route is not.\ngot:\n%s", want, got)
		}
	}
}

// TestGuardStartupReportSilentOnAnUnmanagedHost is the other half of the same
// property: neither line may appear on a host with no corporate CA bundle and no
// cloud-route selector, which is every non-managed host. A seam for managed hosts
// that adds two lines to everyone else's banner has taxed the whole population to
// serve a subset.
func TestGuardStartupReportSilentOnAnUnmanagedHost(t *testing.T) {
	got := renderGuardStartupReport(guardStartupView{})
	for _, unwanted := range []string{"upstream trust", cloudroute.WaiverKey, "NONE for model traffic"} {
		if strings.Contains(got, unwanted) {
			t.Fatalf("unmanaged host got %q on its startup report; the enterprise posture must be invisible here.\ngot:\n%s", unwanted, got)
		}
	}
}

func TestGuardStartupReportCarriesActiveConfigDigest(t *testing.T) {
	got := renderGuardStartupReport(guardStartupView{
		gwURL:        "http://127.0.0.1:8080",
		up:           "anthropic",
		floorSource:  "built-in guard floor",
		policyDigest: "sha256:abc123",
		command:      []string{"claude"},
	})
	if !strings.Contains(got, "fak guard: active config digest sha256:abc123\n") {
		t.Fatalf("startup report missing active config digest:\n%s", got)
	}
}

func TestGuardStartupProfileRowsRenderForFullAndLaunchFailure(t *testing.T) {
	view := guardStartupView{
		gwURL:           "http://127.0.0.1:8080",
		up:              "anthropic",
		floorSource:     "built-in guard floor",
		command:         []string{"claude"},
		responseProfile: "caveman:native:medium",
		workProfile:     "ponytail:native:medium",
		bannerMode:      guardBannerFull,
	}
	report := renderGuardStartupReport(view)
	for _, want := range []string{
		"  response profile : caveman:native:medium\n",
		"  work profile     : ponytail:native:medium\n",
	} {
		if !strings.Contains(report, want) {
			t.Fatalf("startup report missing human-readable profile row %q:\n%s", want, report)
		}
	}
	if strings.Contains(report, `"schema": "fak.guard-profiles.v2"`) || strings.Contains(report, "response-profile {") {
		t.Fatalf("startup report rendered raw profile JSON instead of rows:\n%s", report)
	}

	var full strings.Builder
	writeGuardStartupBanner(&full, view, report, false, false, "", 80)
	for _, want := range []string{"response profile : caveman:native:medium", "work profile     : ponytail:native:medium"} {
		if !strings.Contains(full.String(), want) {
			t.Fatalf("--banner full missing %q:\n%s", want, full.String())
		}
	}

	var failed strings.Builder
	guardWriteLaunchFailReport(&failed, report, true)
	for _, want := range []string{"launch failed", "response profile : caveman:native:medium", "work profile     : ponytail:native:medium"} {
		if !strings.Contains(failed.String(), want) {
			t.Fatalf("launch-failure startup report missing %q:\n%s", want, failed.String())
		}
	}
}
