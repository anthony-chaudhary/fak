package egresslist

import (
	"strings"
	"testing"
)

// TestBundledListsMatchPinnedChecksum is the gate that makes the pin real: every list that
// ships must hash to the checksum the manifest records, and the two sets must correspond
// exactly. It reds the trunk when a list is hand-edited without re-pinning — which is the
// whole point, since a quiet edit to a block list is a capability change to the
// adjudicator's egress rung and must never ride in unreviewed.
func TestBundledListsMatchPinnedChecksum(t *testing.T) {
	sources, err := Sources()
	if err != nil {
		t.Fatalf("Sources(): %v", err)
	}
	if len(sources) == 0 {
		t.Fatal("manifest records no lists at all")
	}
	for _, s := range sources {
		if err := VerifyBundled(s.Name); err != nil {
			t.Errorf("VerifyBundled(%q): %v", s.Name, err)
		}
		if s.SHA256 == "" {
			t.Errorf("list %q pins no checksum", s.Name)
		}
	}
	// Every shipped list is accounted for: a list added to lists/ without a manifest
	// entry has no provenance and no pin, so it must not compile silently.
	for _, name := range BundledListNames() {
		if _, ok := SourceFor(name); !ok {
			t.Errorf("list %q ships but has no manifest entry (no provenance, no checksum pin)", name)
		}
	}
	// ...and every manifest entry names a list that actually ships.
	for _, s := range sources {
		if _, ok := BundledList(s.Name); !ok {
			t.Errorf("manifest records %q but no such list ships", s.Name)
		}
	}
}

// TestPinnedRuleCountMatchesCompiledRules proves the manifest's rule count is the truth
// about what the list compiles to, not a stale annotation — the refresh truncation guard
// measures against it, so a wrong count would silently weaken that guard.
func TestPinnedRuleCountMatchesCompiledRules(t *testing.T) {
	sources, err := Sources()
	if err != nil {
		t.Fatalf("Sources(): %v", err)
	}
	for _, s := range sources {
		text, ok := BundledList(s.Name)
		if !ok {
			continue // covered by TestBundledListsMatchPinnedChecksum
		}
		block, allow := NewBuilder().AddFilterText(s.Name, text).Build().Counts()
		if got := block + allow; got != s.Rules {
			t.Errorf("list %q compiles to %d rules but the manifest pins %d", s.Name, got, s.Rules)
		}
	}
}

// TestVerifyBundledFailsClosed proves the verifier refuses rather than shrugs: an unknown
// name and a checksum mismatch are both errors that name what went wrong.
func TestVerifyBundledFailsClosed(t *testing.T) {
	if err := VerifyBundled("no-such-list"); err == nil {
		t.Error("VerifyBundled(no-such-list) = nil, want an error (unknown names must fail closed)")
	}
	// A tampered artifact must not hash to its pin. The embedded bytes cannot be
	// mutated at runtime, so prove the discriminating property directly: appending a
	// single extra block rule — the exact shape of a quiet widening — changes the hash.
	text, ok := BundledList("sample-malware")
	if !ok {
		t.Fatal("sample-malware does not ship")
	}
	src, ok := SourceFor("sample-malware")
	if !ok {
		t.Fatal("sample-malware has no manifest entry")
	}
	tampered := text + "0.0.0.0 sneaky-addition.example\n"
	if Checksum(tampered) == src.SHA256 {
		t.Error("a tampered list hashes to the pinned checksum: the pin does not discriminate")
	}
	if Checksum(text) != src.SHA256 {
		t.Error("the untampered list does not hash to its pin")
	}
}

// TestChecksumIsLineEndingStable pins the portability property the checksum depends on:
// the same content checked out with CRLF (Windows) or LF (Linux) hashes identically, so
// the gate cannot red on one platform only. A BOM is likewise not content.
func TestChecksumIsLineEndingStable(t *testing.T) {
	const lf = "0.0.0.0 a.example\n0.0.0.0 b.example\n"
	crlf := strings.ReplaceAll(lf, "\n", "\r\n")
	if Checksum(lf) != Checksum(crlf) {
		t.Error("CRLF and LF forms of the same list hash differently: the pin is not portable")
	}
	if Checksum("\xef\xbb\xbf"+lf) != Checksum(lf) {
		t.Error("a UTF-8 BOM changes the checksum: the pin is not content-addressed")
	}
	// Content changes must still move the hash.
	if Checksum(lf) == Checksum(lf+"0.0.0.0 c.example\n") {
		t.Error("adding a rule did not change the checksum")
	}
}

// TestRenderArtifactRoundTrips proves RenderArtifact is the inverse of AddFilterText for
// the grammar this leaf models: a refreshed artifact re-parses to exactly the rules it was
// rendered from. Without this, a refresh could pin a file whose text says something
// different from what the kernel compiled and checked.
func TestRenderArtifactRoundTrips(t *testing.T) {
	const upstream = `
# a messy upstream: hosts lines, anchors, an exception, and rules we do not model
0.0.0.0 ads.example.net
127.0.0.1 localhost
||tracker.example.com^
@@||docs.example.com^
||opts.example.com^$third-party
example.com##.banner
`
	src := Source{Name: "roundtrip", URL: "https://upstream.example/list.txt", Description: "round-trip fixture"}
	first := NewBuilder().AddFilterText(src.Name, upstream).Build()
	artifact := RenderArtifact(src, first)

	second := NewBuilder().AddFilterText(src.Name, artifact).Build()
	want, got := first.Rules(), second.Rules()
	if len(want) != len(got) {
		t.Fatalf("round-trip changed the rule count: %d -> %d\n--- artifact ---\n%s", len(want), len(got), artifact)
	}
	for i := range want {
		if want[i].Domain != got[i].Domain || want[i].Kind != got[i].Kind {
			t.Errorf("rule %d: rendered %v/%s, re-parsed %v/%s", i, want[i].Kind, want[i].Domain, got[i].Kind, got[i].Domain)
		}
	}
	// The artifact is the NORMALIZED form: unmodeled upstream lines are gone, hosts
	// lines are anchors, and the provenance header is present.
	if !strings.Contains(artifact, "! Source: https://upstream.example/list.txt") {
		t.Error("artifact does not record its provenance URL")
	}
	if !strings.Contains(artifact, "||ads.example.net^") || !strings.Contains(artifact, "@@||docs.example.com^") {
		t.Errorf("artifact lost a modeled rule:\n%s", artifact)
	}
	if strings.Contains(artifact, "localhost") || strings.Contains(artifact, "##") || strings.Contains(artifact, "$third-party") {
		t.Errorf("artifact carries an unmodeled/noise line:\n%s", artifact)
	}
	// Rendering is deterministic: same input, byte-identical output (this is what makes
	// an unchanged upstream produce an empty diff).
	if RenderArtifact(src, first) != artifact {
		t.Error("RenderArtifact is not deterministic")
	}
}

// TestRulesAreSortedDeterministically pins the ordering RenderArtifact relies on: block
// rules first, each group domain-sorted, regardless of insertion order.
func TestRulesAreSortedDeterministically(t *testing.T) {
	l := NewBuilder().
		AddRules("s", []string{"z.example", "a.example", "m.example"}, Block).
		AddRules("s", []string{"z-allow.example", "a-allow.example"}, Allow).
		Build()
	rules := l.Rules()
	var blocks, allows []string
	for _, r := range rules {
		switch r.Kind {
		case Block:
			if len(allows) > 0 {
				t.Fatal("a Block rule sorted after an Allow rule")
			}
			blocks = append(blocks, r.Domain)
		case Allow:
			allows = append(allows, r.Domain)
		}
	}
	if strings.Join(blocks, ",") != "a.example,m.example,z.example" {
		t.Errorf("block rules = %v, want domain-sorted", blocks)
	}
	if strings.Join(allows, ",") != "a-allow.example,z-allow.example" {
		t.Errorf("allow rules = %v, want domain-sorted", allows)
	}
	if len(NilListRules()) != 0 {
		t.Error("a nil List must yield no rules")
	}
}

// NilListRules exercises the nil-receiver path Rules() must tolerate (a policy with no
// list configured is the common case).
func NilListRules() []Rule {
	var l *List
	return l.Rules()
}

// TestParseManifestFailsClosed proves the manifest reader refuses ambiguity rather than
// letting a last-entry-wins duplicate decide which checksum pins a list.
func TestParseManifestFailsClosed(t *testing.T) {
	for name, body := range map[string]string{
		"duplicate name": `{"version":1,"lists":[{"name":"a","sha256":"x"},{"name":"a","sha256":"y"}]}`,
		"empty name":     `{"version":1,"lists":[{"name":"","sha256":"x"}]}`,
		"wrong version":  `{"version":99,"lists":[]}`,
		"not json":       `{`,
	} {
		if _, err := ParseManifest([]byte(body)); err == nil {
			t.Errorf("ParseManifest(%s) = nil error, want a refusal", name)
		}
	}
	m, err := ParseManifest([]byte(`{"version":1,"lists":[{"name":"b"},{"name":"a"}]}`))
	if err != nil {
		t.Fatalf("ParseManifest(valid): %v", err)
	}
	if len(m.Lists) != 2 || m.Lists[0].Name != "a" {
		t.Errorf("ParseManifest did not name-sort: %+v", m.Lists)
	}
}

// TestManifestRenderIsCanonical proves a re-render is stable and name-sorted, so a refresh
// that changes one checksum yields a one-line diff rather than a reshuffled file.
func TestManifestRenderIsCanonical(t *testing.T) {
	m := Manifest{Version: ManifestVersion, Lists: []Source{{Name: "z"}, {Name: "a"}}}
	out, err := m.Render()
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if !strings.HasSuffix(string(out), "\n") {
		t.Error("rendered manifest lacks a trailing newline")
	}
	back, err := ParseManifest(out)
	if err != nil {
		t.Fatalf("re-parse rendered manifest: %v", err)
	}
	again, err := back.Render()
	if err != nil {
		t.Fatalf("re-render: %v", err)
	}
	if string(again) != string(out) {
		t.Errorf("render is not canonical:\n--- first ---\n%s\n--- second ---\n%s", out, again)
	}
	if back.Lists[0].Name != "a" {
		t.Errorf("rendered manifest not name-sorted: %+v", back.Lists)
	}
	// Refreshability is provenance-driven, not a flag someone can forget to set.
	if (Source{URL: "https://x.example/l.txt"}).Refreshable() == false {
		t.Error("a list with a URL must be Refreshable")
	}
	if (Source{URL: "   "}).Refreshable() {
		t.Error("a blank URL must not be Refreshable")
	}
}

// TestSourceForUnknownIsClosed proves the provenance lookup does not invent an entry.
func TestSourceForUnknownIsClosed(t *testing.T) {
	if _, ok := SourceFor("definitely-not-a-list"); ok {
		t.Error("SourceFor(unknown) ok=true, want false")
	}
	if s, ok := SourceFor("sample-malware"); !ok || s.Name != "sample-malware" {
		t.Errorf("SourceFor(sample-malware) = %+v/%v, want the shipped entry", s, ok)
	}
}

// TestSampleListIsNotRefreshable documents the shipped placeholder's honest state: it
// records no upstream, so a refresh run reports it skipped instead of fabricating a feed.
// When the real community lists are ingested they arrive with a real url + last_refreshed.
func TestSampleListIsNotRefreshable(t *testing.T) {
	s, ok := SourceFor("sample-malware")
	if !ok {
		t.Fatal("sample-malware has no manifest entry")
	}
	if s.Refreshable() {
		t.Error("the sample placeholder claims an upstream it does not have")
	}
	if s.LastRefreshed != "" {
		t.Errorf("last_refreshed = %q, want empty: the placeholder has never been refreshed", s.LastRefreshed)
	}
}
