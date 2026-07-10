package main

import (
	"strings"
	"testing"
	"time"
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

// TestGuardInfoStalenessNote pins the info-pane twin of the banner warning: an unattested build
// yields a single-line note that names the defect and BOTH durable fixes; an attested build
// yields "" so the pane header stays uncluttered.
func TestGuardInfoStalenessNote(t *testing.T) {
	note := guardInfoStalenessNote("(no VCS stamp — built without module/VCS provenance; cannot confirm the commit)")
	if note == "" {
		t.Fatal("expected a staleness note for an unstamped build, got none")
	}
	for _, want := range []string{
		"stale-build WARN",
		"UNVERIFIABLE",
		"go build ./cmd/fak",
		"fak self-update --force",
	} {
		if !strings.Contains(note, want) {
			t.Fatalf("note missing %q:\n%s", want, note)
		}
	}
	// The header supplies the line break; the note itself must be a single line so a width-trim
	// cannot split it across rows.
	if strings.Contains(note, "\n") {
		t.Fatalf("pane note must be a single line: %q", note)
	}
	if got := guardInfoStalenessNote("abc123def456 +uncommitted  (committed 2026-06-30T00:00:00Z)"); got != "" {
		t.Fatalf("attested build must not warn, got: %q", got)
	}
}

// TestGuardInfoStartupHeaderStalenessConsistency is the render-witness that the pane header
// surfaces the UNATTESTED note EXACTLY when the running binary is unattested. It ties the header's
// output to the pure helper's verdict on the SAME live build stamp, so the assertion holds whether
// or not the test binary itself happens to carry a VCS stamp. It keys on the unattested note's
// DISTINCTIVE phrase (not the shared "stale-build WARN" prefix) so the attested-but-behind skew
// note — which shares that prefix — cannot be mistaken for this one.
func TestGuardInfoStartupHeaderStalenessConsistency(t *testing.T) {
	header := guardInfoStartupHeader("anthropic", 2*time.Second, 0)
	wantNote := guardInfoStalenessNote(guardBannerBuildStamp()) != ""
	gotNote := strings.Contains(header, "cannot confirm which commit fak is running")
	if gotNote != wantNote {
		t.Fatalf("header staleness note present=%v, want %v (build stamp %q):\n%s",
			gotNote, wantNote, guardBannerBuildStamp(), header)
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
