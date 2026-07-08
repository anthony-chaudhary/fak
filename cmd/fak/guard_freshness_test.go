package main

import (
	"strings"
	"testing"
)

// TestGuardBuildStampUnattested pins the classifier that decides whether the guard banner
// should warn. The two "cannot attest a commit" strings guardBannerBuildStamp can produce
// must warn; a real commit stamp and a released module build must not.
func TestGuardBuildStampUnattested(t *testing.T) {
	for _, tc := range []struct {
		name  string
		stamp string
		want  bool
	}{
		{"no vcs stamp", "(no VCS stamp — built without module/VCS provenance; cannot confirm the commit)", true},
		{"no embedded build info", "(no embedded build info)", true},
		{"clean commit", "abc123def456  (committed 2026-06-30T00:00:00Z)", false},
		{"dirty commit", "abc123def456 +uncommitted  (committed 2026-06-30T00:00:00Z)", false},
		{"released module build", "module v0.37.0", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := guardBuildStampUnattested(tc.stamp); got != tc.want {
				t.Fatalf("guardBuildStampUnattested(%q) = %v, want %v", tc.stamp, got, tc.want)
			}
		})
	}
}

// TestGuardUnattestedBuildWarning: an unattested build yields a WARN that names the defect and
// BOTH durable fixes; an attested build yields "" so the banner emits nothing extra.
func TestGuardUnattestedBuildWarning(t *testing.T) {
	warn := guardUnattestedBuildWarning("(no VCS stamp — built without module/VCS provenance; cannot confirm the commit)")
	if warn == "" {
		t.Fatal("expected a warning for an unstamped build, got none")
	}
	for _, want := range []string{
		"no VCS stamp",
		"UNVERIFIABLE",
		"go build ./cmd/fak",
		"fak self-update --force",
	} {
		if !strings.Contains(warn, want) {
			t.Fatalf("warning missing %q:\n%s", want, warn)
		}
	}
	if !strings.HasSuffix(warn, "\n") {
		t.Fatalf("warning must be a complete line (trailing newline): %q", warn)
	}
	if got := guardUnattestedBuildWarning("abc123def456 +uncommitted  (committed 2026-06-30T00:00:00Z)"); got != "" {
		t.Fatalf("attested build must not warn, got: %q", got)
	}
}

// TestPrintGuardBannerWarnsOnUnstampedBuild is the render-witness that the warning reaches the
// full banner: feeding the "no VCS stamp" build row (what guardBannerBuildStamp returns on an
// unattested binary) makes the banner carry the loud staleness-UNVERIFIABLE warning, while a
// normal commit stamp leaves the banner clean.
func TestPrintGuardBannerWarnsOnUnstampedBuild(t *testing.T) {
	render := func(buildStamp string) string {
		var b strings.Builder
		printGuardBanner(&b,
			"9.9.9", buildStamp,
			"http://127.0.0.1:9", "anthropic", "https://api.anthropic.com", "examples/floor.json",
			"ANTHROPIC_BASE_URL=http://127.0.0.1:9", "off", "~/.fak/audit.jsonl",
			nil, false /*remoteServe*/, false /*local*/, "", []string{"claude"})
		return b.String()
	}

	unstamped := render("(no VCS stamp — built without module/VCS provenance; cannot confirm the commit)")
	if !strings.Contains(unstamped, "build WARN") || !strings.Contains(unstamped, "UNVERIFIABLE") {
		t.Fatalf("unstamped banner missing the freshness warning:\n%s", unstamped)
	}

	stamped := render("abc123def456 +uncommitted  (committed 2026-06-30T00:00:00Z)")
	if strings.Contains(stamped, "build WARN") {
		t.Fatalf("attested banner must not carry the warning:\n%s", stamped)
	}
}

// TestPrintGuardCompactBannerMarksUnstamped: the fixed-three-line compact banner surfaces an
// unattested binary in the identity itself (an empty short build id) rather than adding a row.
func TestPrintGuardCompactBannerMarksUnstamped(t *testing.T) {
	var b strings.Builder
	printGuardCompactBanner(&b, "9.9.9", "", "http://127.0.0.1:9", []string{"claude"}, nil)
	out := b.String()
	if !strings.Contains(out, "(no stamp)") {
		t.Fatalf("compact banner should mark an unstamped binary in the identity:\n%s", out)
	}
	if n := strings.Count(out, "\n"); n != 3 {
		t.Fatalf("compact banner must stay exactly 3 lines, got %d:\n%s", n, out)
	}
}
