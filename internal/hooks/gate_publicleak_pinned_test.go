package hooks

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

// gate_publicleak_pinned_test.go — the PUBLIC_LEAK gate's ONE exemption: a checksum-pinned,
// generated third-party artifact (#5405). internal/egresslist owns the predicate; these tests
// pin the WIRING, which is the half that was missing — the Go predicate shipped with no caller,
// so the in-process gate still refused every bundled filter list.
//
// Each test states the property in the direction that matters. The first fails without the
// exemption (the gate reports the feed's own rule lines as leaks). The other two fail if the
// exemption is ever widened past its pin or past the generated line shape — the directions in
// which a mistake publishes a leak instead of merely refusing a commit.

// affiliationNeedle builds the vendor-brand needle at runtime, so this file carries no literal
// needle of its own. That is the same construction hooks_test.go uses for the private-address
// fixture, and it is why no test file appears in selfReferentialLeak.
func affiliationNeedle() string { return "sam" + "sung" }

const (
	pinnedListName    = "curated-feed"
	pinnedListRel     = egresslist.ListsDir + "/" + pinnedListName + ".txt"
	pinnedManifestRel = egresslist.ListsDir + "/" + egresslist.ManifestFile
)

// pinnedArtifactBytes renders an artifact in the shape RenderArtifact emits: the generator
// banner, a prose provenance line, then upstream rule lines — two of which name the affiliation
// needle, because naming a host is how a filter feed blocks it.
func pinnedArtifactBytes() string {
	n := affiliationNeedle()
	return egresslist.GeneratedHeader + "\n" +
		"! source: https://example.invalid/hosts\n" +
		"||ads." + n + "ads.com^\n" +
		"@@||config." + n + "ads.com^\n" +
		"||unrelated.example.com^\n"
}

// pinnedManifestBytes writes a manifest pinning pinnedListName to sum. Passing a sum that is
// NOT the artifact's own hash is how the mismatch case is built.
func pinnedManifestBytes(t *testing.T, sum string) string {
	t.Helper()
	b, err := json.Marshal(egresslist.Manifest{
		Version: egresslist.ManifestVersion,
		Lists: []egresslist.Source{{
			Name:   pinnedListName,
			URL:    "https://example.invalid/hosts",
			SHA256: sum,
		}},
	})
	if err != nil {
		t.Fatalf("marshal manifest fixture: %v", err)
	}
	return string(b)
}

// stagedPinned stages the artifact as an all-new file with the manifest readable beside it.
// The bytes go in fileCache rather than on disk so FileBytes resolves them without a git or a
// filesystem read — the same seeding gate_duplication_test.go uses.
func stagedPinned(artifact, manifest string) *StagedDiff {
	d := &StagedDiff{
		Root:        "/r",
		AddedByFile: map[string][]AddedLine{},
		fileCache: map[string]fileEntry{
			pinnedListRel:     {data: []byte(artifact), exists: true},
			pinnedManifestRel: {data: []byte(manifest), exists: true},
		},
	}
	for i, line := range strings.Split(strings.TrimSuffix(artifact, "\n"), "\n") {
		d.AddedByFile[pinnedListRel] = append(d.AddedByFile[pinnedListRel],
			AddedLine{File: pinnedListRel, New: i + 1, Text: line})
	}
	d.StagedPaths = []string{pinnedListRel}
	d.AddedPaths = []string{pinnedListRel}
	d.AddedRenamedPaths = []string{pinnedListRel}
	return d
}

func leakDetails(fs []Finding) string {
	var b strings.Builder
	for _, f := range fs {
		b.WriteString("\n  ")
		b.WriteString(f.File)
		b.WriteString(":")
		b.WriteString(f.Detail)
	}
	return b.String()
}

// The whole point of the exemption: a feed whose bytes hash to the manifest's pin commits
// cleanly, brand-named block rules and all. Without the wiring in gatePublicLeak this reports
// two findings and the commit is refused — which is exactly what #5405 opened on.
func TestPublicLeakExemptsRuleLinesOfAChecksumPinnedArtifact(t *testing.T) {
	artifact := pinnedArtifactBytes()
	d := stagedPinned(artifact, pinnedManifestBytes(t, egresslist.Checksum(artifact)))

	findings, err := gatePublicLeak(d)
	if err != nil {
		t.Fatalf("gatePublicLeak: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("a checksum-pinned upstream feed must commit clean; got %d finding(s):%s",
			len(findings), leakDetails(findings))
	}
}

// The pin is the load-bearing condition, so bytes the manifest does not vouch for get the
// ordinary strict scan. This is the case where someone edits a rule by hand without re-pinning:
// the hash stops matching and the needle lines are reported again.
func TestPublicLeakRefusesAnArtifactTheManifestDoesNotPin(t *testing.T) {
	artifact := pinnedArtifactBytes()
	stale := egresslist.Checksum(artifact + "||drifted.example.com^\n")
	d := stagedPinned(artifact, pinnedManifestBytes(t, stale))

	findings, err := gatePublicLeak(d)
	if err != nil {
		t.Fatalf("gatePublicLeak: %v", err)
	}
	if len(findings) != 2 {
		t.Fatalf("an unpinned artifact must be scanned strictly: want 2 needle findings, got %d:%s",
			len(findings), leakDetails(findings))
	}
}

// Even under a valid pin the exemption covers only the generated rule shape. A needle in the
// artifact's `!` prose header is a claim this repo wrote into a generated file, so it stays a
// leak — the narrow direction that keeps "generated" from becoming a blanket pass.
func TestPublicLeakRefusesANeedleInAPinnedArtifactsProse(t *testing.T) {
	n := affiliationNeedle()
	artifact := egresslist.GeneratedHeader + "\n" +
		"! curated for the " + n + " fleet\n" +
		"||ads." + n + "ads.com^\n"
	d := stagedPinned(artifact, pinnedManifestBytes(t, egresslist.Checksum(artifact)))

	findings, err := gatePublicLeak(d)
	if err != nil {
		t.Fatalf("gatePublicLeak: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("a needle in a generated file's prose header stays a leak: want 1 finding, got %d:%s",
			len(findings), leakDetails(findings))
	}
	if findings[0].Line != 2 {
		t.Fatalf("the finding must name the prose line (2), not a rule line; got line %d:%s",
			findings[0].Line, leakDetails(findings))
	}
}
