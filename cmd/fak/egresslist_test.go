package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

// seedEgresslistRoot builds a throwaway repo root holding
// internal/egresslist/lists/{manifest.json,testlist.txt} pinned at the canonical
// rendering of seedText with a matching checksum — the state a previous refresh would
// have left. url is the recorded provenance the verb re-fetches. It returns the root the
// verb's --root flag points at and the seeded artifact path.
func seedEgresslistRoot(t *testing.T, url, seedText string) (root, artifact string) {
	t.Helper()
	root = t.TempDir()
	dir := filepath.Join(root, "internal", "egresslist", "lists")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir lists: %v", err)
	}
	src := egresslist.Source{
		Name:          "testlist",
		URL:           url,
		Format:        "hosts",
		Description:   "refresh verb test fixture",
		LastRefreshed: "2026-01-01T00:00:00Z",
	}
	list := egresslist.NewBuilder().AddFilterText(src.Name, seedText).Build()
	block, allow := list.Counts()
	art := egresslist.RenderArtifact(src, list)
	src.SHA256 = egresslist.Checksum(art)
	src.Rules = block + allow

	man := egresslist.Manifest{Version: egresslist.ManifestVersion, Lists: []egresslist.Source{src}}
	out, err := man.Render()
	if err != nil {
		t.Fatalf("render seed manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), out, 0o644); err != nil {
		t.Fatalf("write seed manifest: %v", err)
	}
	artifact = filepath.Join(dir, "testlist.txt")
	if err := os.WriteFile(artifact, []byte(art), 0o644); err != nil {
		t.Fatalf("write seed artifact: %v", err)
	}
	return root, artifact
}

func readTextFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(b)
}

const verbSeedText = "0.0.0.0 old-a.example\n0.0.0.0 old-b.example\n0.0.0.0 old-c.example\n0.0.0.0 old-d.example\n"

// TestRefreshVerbUpdatesAndExitsZero drives the operator verb end to end over a real HTTP
// round-trip: `fak egresslist refresh --root <tmp> --json` re-fetches the moved upstream,
// rewrites the checked-in artifact (the reviewable diff), and exits 0. This binds the
// verb's success contract, not just the engine's.
func TestRefreshVerbUpdatesAndExitsZero(t *testing.T) {
	const upstream = "! refreshed feed\n0.0.0.0 new-evil.example\n0.0.0.0 new-tracker.example\n" +
		"||anchor.example^\n@@||sanctioned.example^\n0.0.0.0 old-a.example\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(upstream))
	}))
	defer srv.Close()

	root, artifact := seedEgresslistRoot(t, srv.URL, verbSeedText)
	before := readTextFile(t, artifact)

	var stdout, stderr bytes.Buffer
	code := runEgresslistRefresh(&stdout, &stderr, []string{"--root", root, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	if after := readTextFile(t, artifact); after == before {
		t.Fatal("artifact unchanged after refreshing a moved upstream: no reviewable diff produced")
	}
	if out := stdout.String(); !strings.Contains(out, `"status": "updated"`) {
		t.Errorf("json output does not report an updated list:\n%s", out)
	}
	if after := readTextFile(t, artifact); !strings.Contains(after, "||new-evil.example^") {
		t.Errorf("refreshed artifact missing the new upstream rule:\n%s", after)
	}
}

// TestRefreshVerbFetchFailureExitsThreeArtifactIntact is the verb-level fail-closed proof:
// a dead upstream makes the verb exit 3 (a refusal a script/CI can gate on) while leaving
// the previously pinned artifact byte-for-byte intact. The URL points at a closed listener
// so the fetch fails offline and deterministically — no network reachability assumed.
func TestRefreshVerbFetchFailureExitsThreeArtifactIntact(t *testing.T) {
	dead := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	deadURL := dead.URL
	dead.Close() // now a connection to deadURL is refused

	root, artifact := seedEgresslistRoot(t, deadURL, verbSeedText)
	before := readTextFile(t, artifact)

	var stdout, stderr bytes.Buffer
	code := runEgresslistRefresh(&stdout, &stderr, []string{"--root", root, "--json"})
	if code != 3 {
		t.Fatalf("exit = %d, want 3 on a fetch failure (stdout: %s)", code, stdout.String())
	}
	if after := readTextFile(t, artifact); after != before {
		t.Errorf("PINNED ARTIFACT MUTATED by a failed fetch\n--- before ---\n%s\n--- after ---\n%s", before, after)
	}
}
