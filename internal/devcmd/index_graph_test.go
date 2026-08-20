package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/docreach"
)

func TestIndexGraphReadsCommittedMarkdownFromRequestedRoot(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"dos.toml":           "[lanes.trees]\ndocs = [\"docs/**\"]\n",
		"README.md":          "# Fixture\n\n[guide](docs/guide.md)\n",
		"docs/guide.md":      "# Guide\n\n[spaced](with space.md)\n",
		"docs/with space.md": "# Spaced\n",
	}
	for rel, body := range files {
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGraphGit(t, root, "init")
	runGraphGit(t, root, "config", "user.email", "fixture@example.test")
	runGraphGit(t, root, "config", "user.name", "fixture")
	runGraphGit(t, root, "add", ".")
	runGraphGit(t, root, "commit", "-m", "fixture")
	commit := strings.TrimSpace(runGraphGit(t, root, "rev-parse", "HEAD"))

	// Neither a dirty tracked body nor an untracked Markdown file may enter the
	// census: the index is a committed-tree witness, not a peer-WIP snapshot.
	if err := os.WriteFile(filepath.Join(root, "docs", "guide.md"), []byte("# DIRTY\n\n[missing](missing.md)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "peer.md"), []byte("# Peer WIP\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if rc := RunIndex(&out, &errb, []string{"graph", "--root", root, "--json"}); rc != 0 {
		t.Fatalf("index graph rc=%d, stderr=%s", rc, errb.String())
	}
	var got docreach.Report
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode graph report: %v\n%s", err, out.String())
	}
	if got.Commit != commit || got.Documents != 3 {
		t.Fatalf("graph header=%+v, want commit=%s documents=3", got, commit)
	}
	if len(got.BrokenLinks) != 0 {
		t.Fatalf("graph read dirty tracked Markdown: broken_links=%+v", got.BrokenLinks)
	}
	for _, rule := range got.Rules {
		if rule.Denominator != 3 {
			t.Fatalf("rule denominator=%d, want committed corpus size 3: %+v", rule.Denominator, rule)
		}
	}
}

func runGraphGit(t *testing.T, root string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
	return string(out)
}
