package egressrefresh

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

// seedDir builds a bundled-lists directory holding one list named "testlist", pinned at
// the canonical rendering of seedText with a matching checksum and rule count — i.e. the
// state a previous refresh would have left. Every test below starts from a correctly
// pinned artifact so that "the pinned artifact survived" is a meaningful assertion.
func seedDir(t *testing.T, url, seedText string) string {
	t.Helper()
	dir := t.TempDir()
	src := egresslist.Source{
		Name:          "testlist",
		URL:           url,
		Format:        "hosts",
		Description:   "refresh test fixture",
		LastRefreshed: "2026-01-01T00:00:00Z",
	}
	list := egresslist.NewBuilder().AddFilterText(src.Name, seedText).Build()
	block, allow := list.Counts()
	artifact := egresslist.RenderArtifact(src, list)
	src.SHA256 = egresslist.Checksum(artifact)
	src.Rules = block + allow

	man := egresslist.Manifest{Version: egresslist.ManifestVersion, Lists: []egresslist.Source{src}}
	out, err := man.Render()
	if err != nil {
		t.Fatalf("render seed manifest: %v", err)
	}
	if err := os.WriteFile(ManifestPath(dir), out, 0o644); err != nil {
		t.Fatalf("write seed manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "testlist.txt"), []byte(artifact), 0o644); err != nil {
		t.Fatalf("write seed artifact: %v", err)
	}
	return dir
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(b)
}

func manifestOf(t *testing.T, dir string) egresslist.Manifest {
	t.Helper()
	m, err := egresslist.ParseManifest([]byte(readFile(t, ManifestPath(dir))))
	if err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	return m
}

// staticFetcher returns fixed bytes for any URL.
type staticFetcher struct{ body string }

func (f staticFetcher) Fetch(context.Context, string) ([]byte, error) { return []byte(f.body), nil }

// errFetcher fails every fetch — the upstream-is-down case.
type errFetcher struct{ err error }

func (f errFetcher) Fetch(context.Context, string) ([]byte, error) { return nil, f.err }

const seedText = "0.0.0.0 old-a.example\n0.0.0.0 old-b.example\n0.0.0.0 old-c.example\n0.0.0.0 old-d.example\n"

// TestRefreshUpdatesArtifactChecksumAndDiff is acceptance #1, driven over a REAL HTTP
// round-trip (httptest + the production HTTPFetcher): refreshing pulls the upstream,
// rewrites the checked-in artifact, re-pins its checksum, stamps last_refreshed, and the
// artifact text actually changes — the reviewable diff.
func TestRefreshUpdatesArtifactChecksumAndDiff(t *testing.T) {
	const upstream = "! refreshed feed\n0.0.0.0 new-evil.example\n0.0.0.0 new-tracker.example\n" +
		"||anchor.example^\n@@||sanctioned.example^\n0.0.0.0 old-a.example\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(upstream))
	}))
	defer srv.Close()

	dir := seedDir(t, srv.URL, seedText)
	before := readFile(t, filepath.Join(dir, "testlist.txt"))
	stamp := time.Date(2026, 7, 16, 12, 0, 0, 0, time.UTC)

	results, err := Refresh(context.Background(), Options{Dir: dir, Now: stamp})
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if len(results) != 1 || results[0].Status != StatusUpdated {
		t.Fatalf("results = %+v, want one updated", results)
	}

	after := readFile(t, filepath.Join(dir, "testlist.txt"))
	if after == before {
		t.Fatal("artifact unchanged after refreshing a moved upstream: no reviewable diff was produced")
	}
	// The refreshed artifact carries the upstream's rules, normalized through the
	// kernel's own ingest path (hosts lines become domain anchors).
	for _, want := range []string{"||new-evil.example^", "||new-tracker.example^", "||anchor.example^", "@@||sanctioned.example^"} {
		if !strings.Contains(after, want) {
			t.Errorf("refreshed artifact missing %q\n--- artifact ---\n%s", want, after)
		}
	}
	// A rule dropped upstream is gone from the artifact — refresh replaces, never merges.
	if strings.Contains(after, "||old-b.example^") {
		t.Error("refreshed artifact still carries old-b.example, dropped by the upstream")
	}

	// The manifest re-pins the new bytes and records when we looked.
	m := manifestOf(t, dir)
	src, ok := m.Get("testlist")
	if !ok {
		t.Fatal("manifest lost the testlist entry")
	}
	if got := egresslist.Checksum(after); src.SHA256 != got {
		t.Errorf("manifest pins %s but the artifact hashes to %s", src.SHA256, got)
	}
	if src.SHA256 != results[0].NewSHA256 || results[0].NewSHA256 == results[0].OldSHA256 {
		t.Errorf("checksum not advanced: old=%s new=%s manifest=%s", results[0].OldSHA256, results[0].NewSHA256, src.SHA256)
	}
	if src.Rules != 5 {
		t.Errorf("manifest rules = %d, want 5 (4 block + 1 allow)", src.Rules)
	}
	if src.LastRefreshed != "2026-07-16T12:00:00Z" {
		t.Errorf("last_refreshed = %q, want the injected 2026-07-16T12:00:00Z", src.LastRefreshed)
	}
	// The refreshed artifact must still compile to the verdicts it claims.
	l := egresslist.NewBuilder().AddFilterText("testlist", after).Build()
	if d := l.Decide("new-evil.example"); d.Kind != egresslist.Block {
		t.Errorf("refreshed list Decide(new-evil.example) = %v, want Block", d.Kind)
	}
	if d := l.Decide("sanctioned.example"); d.Kind != egresslist.Allow {
		t.Errorf("refreshed list Decide(sanctioned.example) = %v, want Allow", d.Kind)
	}
}

// TestFetchFailureKeepsPinnedArtifact is acceptance #2, the headline fail-closed case: the
// upstream is down, so BOTH the artifact and the manifest that pins it must be untouched,
// byte for byte. A refresh that half-wrote here would leave the kernel compiling a list
// nobody vouched for.
func TestFetchFailureKeepsPinnedArtifact(t *testing.T) {
	dir := seedDir(t, "https://upstream.invalid/list.txt", seedText)
	artPath := filepath.Join(dir, "testlist.txt")
	artBefore := readFile(t, artPath)
	manBefore := readFile(t, ManifestPath(dir))

	results, err := Refresh(context.Background(), Options{
		Dir:     dir,
		Fetcher: errFetcher{err: errors.New("dial tcp: connection refused")},
		Now:     time.Date(2026, 7, 16, 12, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatalf("Refresh returned a run error for a per-list fetch failure: %v", err)
	}
	if len(results) != 1 || results[0].Status != StatusFailed {
		t.Fatalf("results = %+v, want one failed", results)
	}
	if !strings.Contains(results[0].Reason, "connection refused") {
		t.Errorf("reason = %q, want it to name the fetch error", results[0].Reason)
	}
	if got := readFile(t, artPath); got != artBefore {
		t.Errorf("PINNED ARTIFACT MUTATED by a failed fetch\n--- before ---\n%s\n--- after ---\n%s", artBefore, got)
	}
	if got := readFile(t, ManifestPath(dir)); got != manBefore {
		t.Errorf("manifest mutated by a failed fetch\n--- before ---\n%s\n--- after ---\n%s", manBefore, got)
	}
}

// TestNon200KeepsPinnedArtifact proves the HTTP layer refuses an error page rather than
// parsing it. A 500 body ("<html>Service Unavailable</html>") would otherwise parse to
// zero rules and silently empty the block list.
func TestNon200KeepsPinnedArtifact(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "<html>Service Unavailable</html>", http.StatusInternalServerError)
	}))
	defer srv.Close()

	dir := seedDir(t, srv.URL, seedText)
	artBefore := readFile(t, filepath.Join(dir, "testlist.txt"))

	results, err := Refresh(context.Background(), Options{Dir: dir})
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if results[0].Status != StatusFailed || !strings.Contains(results[0].Reason, "HTTP 500") {
		t.Fatalf("result = %+v, want failed naming HTTP 500", results[0])
	}
	if got := readFile(t, filepath.Join(dir, "testlist.txt")); got != artBefore {
		t.Error("pinned artifact mutated by a non-200 upstream")
	}
}

// TestEmptyUpstreamRefused is the all-permissive trap: an upstream that fetches fine (200)
// but parses to zero rules must NOT become the new list. This is the case the issue calls
// out by name — an empty block list silently blocks nothing.
func TestEmptyUpstreamRefused(t *testing.T) {
	dir := seedDir(t, "https://upstream.invalid/list.txt", seedText)
	artBefore := readFile(t, filepath.Join(dir, "testlist.txt"))
	manBefore := readFile(t, ManifestPath(dir))

	results, err := Refresh(context.Background(), Options{
		Dir:     dir,
		Fetcher: staticFetcher{body: "! header only, every rule gone\n! nothing parseable here\n"},
	})
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if results[0].Status != StatusFailed {
		t.Fatalf("result = %+v, want failed (an empty upstream must be refused)", results[0])
	}
	if !strings.Contains(results[0].Reason, "all-permissive") {
		t.Errorf("reason = %q, want it to explain the all-permissive hazard", results[0].Reason)
	}
	if got := readFile(t, filepath.Join(dir, "testlist.txt")); got != artBefore {
		t.Error("pinned artifact mutated by an empty upstream")
	}
	if got := readFile(t, ManifestPath(dir)); got != manBefore {
		t.Error("manifest mutated by an empty upstream")
	}
}

// TestTruncatedUpstreamRefused is the other half of fail-closed: a body that parses but
// collapsed against its pinned rule count is a broken fetch, not news. --allow-shrink is
// the deliberate operator override, proving the guard is a gate and not a wall.
func TestTruncatedUpstreamRefused(t *testing.T) {
	dir := seedDir(t, "https://upstream.invalid/list.txt", seedText) // pinned at 4 rules
	artBefore := readFile(t, filepath.Join(dir, "testlist.txt"))
	truncated := staticFetcher{body: "0.0.0.0 only-one.example\n"} // 1 rule: below 50% of 4

	results, err := Refresh(context.Background(), Options{Dir: dir, Fetcher: truncated})
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if results[0].Status != StatusFailed || !strings.Contains(results[0].Reason, "truncation guard") {
		t.Fatalf("result = %+v, want failed naming the truncation guard", results[0])
	}
	if got := readFile(t, filepath.Join(dir, "testlist.txt")); got != artBefore {
		t.Fatal("pinned artifact mutated by a truncated upstream")
	}

	// The same shrink lands once an operator has actually looked at it.
	results, err = Refresh(context.Background(), Options{
		Dir: dir, Fetcher: truncated, AllowShrink: true,
		Now: time.Date(2026, 7, 16, 12, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatalf("Refresh --allow-shrink: %v", err)
	}
	if results[0].Status != StatusUpdated {
		t.Fatalf("result = %+v, want updated under --allow-shrink", results[0])
	}
	if got := readFile(t, filepath.Join(dir, "testlist.txt")); got == artBefore {
		t.Error("artifact unchanged under --allow-shrink, want the shrink applied")
	}
	if m := manifestOf(t, dir); m.Lists[0].Rules != 1 {
		t.Errorf("manifest rules = %d, want 1 after the sanctioned shrink", m.Lists[0].Rules)
	}
}

// TestUnchangedUpstreamAdvancesOnlyStaleness pins the no-diff property: re-refreshing an
// unchanged upstream must leave the artifact byte-identical (no timestamp churn in the
// file) while still advancing last_refreshed — so "checked, current" is distinguishable
// from "nobody has looked".
func TestUnchangedUpstreamAdvancesOnlyStaleness(t *testing.T) {
	dir := seedDir(t, "https://upstream.invalid/list.txt", seedText)
	artBefore := readFile(t, filepath.Join(dir, "testlist.txt"))

	results, err := Refresh(context.Background(), Options{
		Dir:     dir,
		Fetcher: staticFetcher{body: seedText}, // same content the artifact is pinned at
		Now:     time.Date(2026, 7, 16, 12, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if results[0].Status != StatusUnchanged {
		t.Fatalf("result = %+v, want unchanged", results[0])
	}
	if got := readFile(t, filepath.Join(dir, "testlist.txt")); got != artBefore {
		t.Error("artifact rewritten for an unchanged upstream: the diff should have been empty")
	}
	m := manifestOf(t, dir)
	if m.Lists[0].LastRefreshed != "2026-07-16T12:00:00Z" {
		t.Errorf("last_refreshed = %q, want it advanced to the check time", m.Lists[0].LastRefreshed)
	}
	if m.Lists[0].SHA256 != egresslist.Checksum(artBefore) {
		t.Error("checksum drifted for an unchanged artifact")
	}
}

// TestSkipsListWithNoProvenanceURL proves a hand-authored list (the shipped
// sample-malware placeholder) is reported skipped, never fabricated a source or emptied.
func TestSkipsListWithNoProvenanceURL(t *testing.T) {
	dir := seedDir(t, "", seedText)
	artBefore := readFile(t, filepath.Join(dir, "testlist.txt"))

	results, err := Refresh(context.Background(), Options{Dir: dir, Fetcher: errFetcher{err: errors.New("must not be called")}})
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if results[0].Status != StatusSkipped || !strings.Contains(results[0].Reason, "no provenance URL") {
		t.Fatalf("result = %+v, want skipped naming the missing provenance URL", results[0])
	}
	if got := readFile(t, filepath.Join(dir, "testlist.txt")); got != artBefore {
		t.Error("artifact mutated for a non-refreshable list")
	}
}

// TestDryRunWritesNothing proves --dry-run previews the same verdict without touching disk.
func TestDryRunWritesNothing(t *testing.T) {
	dir := seedDir(t, "https://upstream.invalid/list.txt", seedText)
	artBefore := readFile(t, filepath.Join(dir, "testlist.txt"))
	manBefore := readFile(t, ManifestPath(dir))

	results, err := Refresh(context.Background(), Options{
		Dir: dir, DryRun: true, AllowShrink: true,
		Fetcher: staticFetcher{body: "0.0.0.0 brand-new.example\n"},
	})
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if results[0].Status != StatusUpdated {
		t.Fatalf("result = %+v, want updated (previewed)", results[0])
	}
	if readFile(t, filepath.Join(dir, "testlist.txt")) != artBefore || readFile(t, ManifestPath(dir)) != manBefore {
		t.Error("--dry-run wrote to disk")
	}
}

// TestUnknownNameIsHardError proves a typo'd --name fails loudly instead of reporting a
// green "0 lists refreshed" that an operator would read as success.
func TestUnknownNameIsHardError(t *testing.T) {
	dir := seedDir(t, "https://upstream.invalid/list.txt", seedText)
	_, err := Refresh(context.Background(), Options{Dir: dir, Names: []string{"nosuchlist"}})
	if err == nil {
		t.Fatal("Refresh with an unknown --name returned nil error, want a hard failure")
	}
	if !strings.Contains(err.Error(), "nosuchlist") || !strings.Contains(err.Error(), "testlist") {
		t.Errorf("error = %v, want it to name the bad list and the known ones", err)
	}
}

// TestOversizeBodyRefused proves the fetch cap refuses a runaway upstream rather than
// eating the box.
func TestOversizeBodyRefused(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Repeat("0.0.0.0 pad.example\n", 100)))
	}))
	defer srv.Close()

	dir := seedDir(t, srv.URL, seedText)
	artBefore := readFile(t, filepath.Join(dir, "testlist.txt"))

	results, err := Refresh(context.Background(), Options{Dir: dir, Fetcher: HTTPFetcher{MaxBytes: 64}})
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if results[0].Status != StatusFailed || !strings.Contains(results[0].Reason, "exceeds") {
		t.Fatalf("result = %+v, want failed naming the size cap", results[0])
	}
	if got := readFile(t, filepath.Join(dir, "testlist.txt")); got != artBefore {
		t.Error("pinned artifact mutated by an oversize upstream")
	}
}
