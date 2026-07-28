package egresslist

import (
	"strings"
	"testing"
)

// pinned_artifact_test.go — the witness for the #5405 exemption predicate: what the
// generator emits IS exemptible, and nothing else is.
//
// The positive half is a regression guard, not a formality. The exemption a leak gate
// grants a bundled list is conditioned on the generator's own output — the banner it
// writes and the line shape it renders. The day RenderArtifact starts emitting a shape
// IsUpstreamRuleLine does not recognise, or stops writing the banner, the exemption
// silently stops covering the artifact and the next `fak egresslist refresh` becomes
// uncommittable for a reason nobody can trace back to this package. These tests red the
// trunk at that moment instead.
//
// The negative half is the load-bearing one: a hand-written file that merely sits in the
// lists directory, or copies the banner, or was hand-edited after its pin was computed,
// must NOT qualify. That is the entire difference between "upstream data this repo
// records in order to block it" and "a claim this repo makes".

// fixtureManifest renders a one-entry manifest pinning `text` under `name`, via the real
// Manifest.Render so the fixture cannot drift from the on-disk schema.
func fixtureManifest(t *testing.T, name, text string) []byte {
	t.Helper()
	m := Manifest{Version: ManifestVersion}
	m.Set(Source{
		Name: name, URL: "https://feed.invalid/hosts", Format: "hosts",
		License: "MIT", SHA256: Checksum(text), Rules: 1,
	})
	b, err := m.Render()
	if err != nil {
		t.Fatalf("render fixture manifest: %v", err)
	}
	return b
}

// fixtureArtifact renders a real generated artifact from real compiled rules, so the
// bytes under test are produced by the same path `fak egresslist refresh` uses.
func fixtureArtifact(name string, hosts ...string) string {
	var b strings.Builder
	for _, h := range hosts {
		b.WriteString("||" + h + "^\n")
	}
	l := NewBuilder().AddFilterText(name, b.String()).Build()
	return RenderArtifact(Source{Name: name, URL: "https://feed.invalid/hosts", License: "MIT"}, l)
}

// TestGeneratedHeaderMatchesRenderArtifact pins the banner constant to the generator.
// Without this the exemption could be keyed on a header no artifact actually carries —
// a gate that never fires open, discovered only when a refresh cannot be committed.
func TestGeneratedHeaderMatchesRenderArtifact(t *testing.T) {
	got := fixtureArtifact("feed", "ads.example.net")
	if !strings.HasPrefix(got, GeneratedHeader+"\n") {
		first, _, _ := strings.Cut(got, "\n")
		t.Fatalf("RenderArtifact first line = %q, want the GeneratedHeader constant %q", first, GeneratedHeader)
	}
}

// TestRenderedArtifactIsPinnedAndRuleShaped is the positive witness: the exact bytes the
// generator produces, pinned by the exact manifest the refresh writes, qualify — and
// every data line in them is a shape the exemption recognises.
func TestRenderedArtifactIsPinnedAndRuleShaped(t *testing.T) {
	text := fixtureArtifact("feed", "ads.vendorads.example", "log.vendoracr.example", "cdn.example.net")
	rel := ListsDir + "/feed.txt"
	if !IsPinnedArtifact(rel, []byte(text), fixtureManifest(t, "feed", text)) {
		t.Fatalf("the generator's own output is not recognised as a pinned artifact")
	}
	data := 0
	for _, line := range strings.Split(text, "\n") {
		if line == "" || strings.HasPrefix(line, "!") {
			continue // provenance header: written by this repo, deliberately NOT exempt
		}
		if !IsUpstreamRuleLine(line) {
			t.Errorf("generated data line %q is not recognised as an upstream rule: the "+
				"exemption would not cover it and a refresh carrying a needle would be refused", line)
		}
		data++
	}
	if data != 3 {
		t.Fatalf("counted %d data lines, want 3 — the shape assertion above would be vacuous", data)
	}
}

// TestPinnedArtifactRefusesEverythingHandWritten is the negative half. Each case is a way
// a leak could try to inherit the exemption; every one must fail closed.
func TestPinnedArtifactRefusesEverythingHandWritten(t *testing.T) {
	text := fixtureArtifact("feed", "ads.vendorads.example", "cdn.example.net")
	good := fixtureManifest(t, "feed", text)
	rel := ListsDir + "/feed.txt"

	cases := []struct {
		name     string
		rel      string
		artifact string
		manifest []byte
	}{
		{
			// The load-bearing one: prose that merely LIVES in the lists directory.
			name: "hand-written file in the lists dir", rel: ListsDir + "/notes.txt",
			artifact: "our partnership notes\n", manifest: good,
		},
		{
			// "Looks generated" must never substitute for "is the pinned bytes".
			name: "forged banner with no manifest entry", rel: ListsDir + "/forged.txt",
			artifact: fixtureArtifact("forged", "tracker.example.net"), manifest: good,
		},
		{
			// The pin is what closes the smuggling path: edit the file, lose the exemption.
			name: "hand-edited after pinning", rel: rel,
			artifact: text + "||smuggled.example.net^\n", manifest: good,
		},
		{
			name: "banner missing", rel: rel,
			artifact: strings.TrimPrefix(text, GeneratedHeader+"\n"), manifest: good,
		},
		{
			// Scoped by path as well as by shape: the same pinned bytes elsewhere lose it.
			name: "declared shape outside the declared dir", rel: "docs/lists/feed.txt",
			artifact: text, manifest: good,
		},
		{
			name: "manifest unparseable", rel: rel,
			artifact: text, manifest: []byte("{not json"),
		},
		{
			name: "manifest pins a different name", rel: rel,
			artifact: text, manifest: fixtureManifest(t, "other-feed", text),
		},
		{
			name: "manifest pins an empty checksum", rel: rel,
			artifact: text,
			manifest: func() []byte {
				m := Manifest{Version: ManifestVersion}
				m.Set(Source{Name: "feed", SHA256: ""})
				b, _ := m.Render()
				return b
			}(),
		},
	}
	for _, c := range cases {
		if IsPinnedArtifact(c.rel, []byte(c.artifact), c.manifest) {
			t.Errorf("%s: qualified for the exemption, want refused", c.name)
		}
	}
}

// TestUpstreamRuleLineRejectsProseAndClaims pins the per-LINE half. A generated artifact
// is exempt only line by line: its provenance header is written by this repo, so a claim
// smuggled in there (with the manifest re-pinned so the file still verifies) is exactly
// the case the shape test exists to catch.
func TestUpstreamRuleLineRejectsProseAndClaims(t *testing.T) {
	for _, ok := range []string{
		"||ads.example.net^",
		"@@||allow.example.net^",
		"  ||indented.example.net^  ",
		"||0.0.0.0.creative.example.net^",
	} {
		if !IsUpstreamRuleLine(ok) {
			t.Errorf("IsUpstreamRuleLine(%q) = false, want true", ok)
		}
	}
	for _, bad := range []string{
		"",
		"!",
		"! Description: our partnership with a vendor",
		"! Name: feed",
		"||nodot^",             // not a host: no dot
		"||UPPER.example.net^", // the generator lower-cases every rule
		"||has space.example^", // not the rendered grammar
		"ads.example.net",      // a bare hosts-file token, not a rendered rule
		"0.0.0.0 ads.example.net",
		"||ads.example.net", // missing the domain anchor terminator
		"# a comment claiming something",
	} {
		if IsUpstreamRuleLine(bad) {
			t.Errorf("IsUpstreamRuleLine(%q) = true, want false", bad)
		}
	}
}

// TestArtifactNameParsesOnlyTheDeclaredClass pins the path condition on its own, so a
// future caller cannot widen the class by passing a cleverly shaped relative path.
func TestArtifactNameParsesOnlyTheDeclaredClass(t *testing.T) {
	for rel, want := range map[string]string{
		ListsDir + "/feed.txt":                               "feed",
		"fak/" + ListsDir + "/feed.txt":                      "feed",
		"./" + ListsDir + "/feed.txt":                        "feed",
		strings.ReplaceAll(ListsDir, "/", `\`) + `\feed.txt`: "feed",
	} {
		got, ok := ArtifactName(rel)
		if !ok || got != want {
			t.Errorf("ArtifactName(%q) = (%q, %v), want (%q, true)", rel, got, ok, want)
		}
	}
	for _, rel := range []string{
		"",
		"docs/lists/feed.txt",
		ListsDir + "/feed.md",
		ListsDir + "/nested/feed.txt",
		ListsDir + "/.txt",
		"internal/egresslistX/lists/feed.txt",
		"fak/fak/" + ListsDir + "/feed.txt",
	} {
		if name, ok := ArtifactName(rel); ok {
			t.Errorf("ArtifactName(%q) = (%q, true), want refused", rel, name)
		}
	}
}

// TestBundledGeneratedArtifactsStayExemptible checks the artifacts that ACTUALLY ship:
// any bundled list carrying the generator banner must still verify against its pinned
// checksum and consist of recognised rule lines. This is the test that catches a refresh
// which re-pins the manifest but changes the rendering — the artifact would stay green
// under TestBundledListsMatchPinnedChecksum while quietly losing its leak-gate exemption.
func TestBundledGeneratedArtifactsStayExemptible(t *testing.T) {
	for _, name := range BundledListNames() {
		text, ok := BundledList(name)
		if !ok || !strings.HasPrefix(text, GeneratedHeader) {
			continue // hand-authored bundled list (e.g. the sample placeholder): not this class
		}
		rel := ListsDir + "/" + name + ".txt"
		if !IsPinnedArtifact(rel, []byte(text), manifestJSON) {
			t.Errorf("bundled generated list %q does not verify against its pinned checksum: "+
				"it carries the generator banner but would get no leak-gate exemption", name)
		}
		for i, line := range strings.Split(text, "\n") {
			if line == "" || strings.HasPrefix(line, "!") {
				continue
			}
			if !IsUpstreamRuleLine(line) {
				t.Errorf("%s:%d: %q is not a recognised upstream rule line", rel, i+1, line)
			}
		}
	}
}
