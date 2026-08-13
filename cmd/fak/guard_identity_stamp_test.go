package main

import (
	"runtime/debug"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// selfInstalledBuildInfo is the build provenance a `fak self-update` / source-install binary
// actually carries: the Go toolchain embedded NO vcs.* settings (it built from an export, not
// the checkout), so the commit reaches the binary only through the -X-injected
// appversion.BuildCommit. Both the version surface and the guard identity are driven from this
// one fixture so a divergence between them is the test's only possible failure.
func selfInstalledBuildInfo(t *testing.T, commit string) *debug.BuildInfo {
	t.Helper()
	old := appversion.BuildCommit
	appversion.BuildCommit = commit
	t.Cleanup(func() { appversion.BuildCommit = old })
	return &debug.BuildInfo{Main: debug.Module{Version: "(devel)"}}
}

// TestGuardIdentityMatchesVersionSurfaceOnDeployedBuild is the render witness for #6537: a
// CLEAN DEPLOYED binary that `fak version` can attest ("build: 0c96937b61ac") must render that
// same short build id in BOTH attended guard identities — the compact banner and the animated
// settle line — instead of the "(no stamp)" staleness tell. The tell is only useful if it fires
// exclusively on binaries that genuinely cannot attest a commit; firing on a just-self-updated
// one sends an operator toward an update they already did.
func TestGuardIdentityMatchesVersionSurfaceOnDeployedBuild(t *testing.T) {
	const commit = "0c96937b61ac1122334455667788990011223344"
	bi := selfInstalledBuildInfo(t, commit)

	// The version surface, verbatim: what `fak version` prints for this binary.
	versionLine := buildProvenanceLine(bi)
	if !strings.Contains(versionLine, commit[:12]) {
		t.Fatalf("fak version build line = %q, want the deployed commit %s", versionLine, commit[:12])
	}

	// The guard identity, from the SAME provenance.
	id := buildIdentity(bi)
	short := guardShortBuildIDOf(id)
	if short == "" {
		t.Fatalf("guard short build id is empty while the version surface attests a commit (%q)", versionLine)
	}
	if !strings.HasPrefix(id.CommitShort, short) {
		t.Fatalf("guard short build id %q is not a prefix of the version surface's %q — the two surfaces disagree about the commit",
			short, id.CommitShort)
	}

	var b strings.Builder
	printGuardCompactBanner(&b, "0.43.0", short, "http://127.0.0.1:9", []string{"claude"}, nil)
	compact := b.String()
	if strings.Contains(compact, "(no stamp)") {
		t.Fatalf("compact banner claims no stamp for a build %q attests:\n%s", versionLine, compact)
	}
	if !strings.Contains(compact, short) {
		t.Fatalf("compact banner missing the short build id %q:\n%s", short, compact)
	}

	settle := strings.Join(guardLaunchSettleLines("0.43.0", short, "claude", "http://127.0.0.1:9", 200, nil), "\n")
	if strings.Contains(settle, "(no stamp)") {
		t.Fatalf("animated settle line claims no stamp for a build %q attests:\n%s", versionLine, settle)
	}
	if !strings.Contains(settle, short) {
		t.Fatalf("animated settle line missing the short build id %q:\n%s", short, settle)
	}
}

// TestGuardIdentityMarksGenuinelyUnstampedBuild is the other half of the done condition: the
// "(no stamp)" tell must SURVIVE for a build that really cannot attest a commit (a bare
// `go build ./cmd/fak` with buildvcs off — no vcs.* settings and no injected commit), because
// that binary is indistinguishable from a stale one.
func TestGuardIdentityMarksGenuinelyUnstampedBuild(t *testing.T) {
	bi := selfInstalledBuildInfo(t, "") // no injected commit either

	if short := guardShortBuildIDOf(buildIdentity(bi)); short != "" {
		t.Fatalf("guard short build id = %q, want empty for a binary with no commit stamp at all", short)
	}
	if line := buildProvenanceLine(bi); !strings.Contains(line, "no VCS stamp") {
		t.Fatalf("version surface = %q, want the explicit no-VCS-stamp note", line)
	}

	var b strings.Builder
	printGuardCompactBanner(&b, "0.43.0", guardShortBuildIDOf(buildIdentity(bi)), "http://127.0.0.1:9", []string{"claude"}, nil)
	if out := b.String(); !strings.Contains(out, "(no stamp)") {
		t.Fatalf("compact banner must keep the staleness tell for an unstamped binary:\n%s", out)
	}
}

// TestGuardShortBuildIDTracksVersionIdentity binds the LIVE helper (the one the launch path
// calls) to the LIVE identity `fak version --json` publishes, so the two cannot drift again
// whatever provenance this test binary happens to carry: the compact id is non-empty exactly
// when the identity is stamped, and it always abbreviates the SAME commit.
func TestGuardShortBuildIDTracksVersionIdentity(t *testing.T) {
	bi, _ := debug.ReadBuildInfo()
	id := buildIdentity(bi)
	got := guardShortBuildID()

	if id.Stamped != (got != "") {
		t.Fatalf("guardShortBuildID() = %q but version identity Stamped = %v — the guard banner and `fak version` disagree about whether this binary is stamped",
			got, id.Stamped)
	}
	if !id.Stamped {
		return
	}
	if strings.HasSuffix(got, "+") != id.Dirty {
		t.Errorf("guardShortBuildID() = %q, dirty marker disagrees with identity Dirty = %v", got, id.Dirty)
	}
	if rev := strings.TrimSuffix(got, "+"); !strings.HasPrefix(id.CommitShort, rev) {
		t.Errorf("guardShortBuildID() = %q abbreviates a different commit than the version surface's %q", got, id.CommitShort)
	}
}
