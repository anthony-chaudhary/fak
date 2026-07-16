package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestLLMSFullIsolatesOwnedDocsAndReportsProvenance(t *testing.T) {
	root := llmsFullFixture(t)
	owned := filepath.Join(root, "docs", "owned.md")
	peer := filepath.Join(root, "docs", "peer.md")
	untrackedPeer := filepath.Join(root, "docs", "untracked-peer.md")
	if err := os.WriteFile(owned, []byte("# Owned\n\nOWNED DELTA\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(peer, []byte("# Peer\n\nPEER WIP\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(untrackedPeer, []byte("# Untracked peer\n\nUNTRACKED PEER WIP\n"), 0644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	old, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(old) })
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	code := runLLMSFull([]string{"--mine", "docs/owned.md", "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var got llmsFullResult
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.OK || len(got.Included) != 1 || got.Included[0] != "docs/owned.md" {
		t.Fatalf("result=%+v", got)
	}
	head := strings.TrimSpace(llmsGitOutput(t, root, "rev-parse", "HEAD"))
	if got.SourceCommit != head {
		t.Fatalf("source_commit=%q, want %q", got.SourceCommit, head)
	}
	for _, want := range []string{"docs/peer.md", "docs/untracked-peer.md"} {
		if !llmsContains(got.Excluded, want) {
			t.Fatalf("excluded=%v, want %s", got.Excluded, want)
		}
	}
	corpus, err := os.ReadFile(filepath.Join(root, "llms-full.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(corpus, []byte("OWNED DELTA")) {
		t.Fatal("owned overlay missing")
	}
	if bytes.Contains(corpus, []byte("PEER WIP")) || bytes.Contains(corpus, []byte("UNTRACKED PEER WIP")) {
		t.Fatal("peer WIP leaked into corpus")
	}
}

func TestLLMSFullCheckIgnoresUnrelatedDirtyFiles(t *testing.T) {
	root := llmsFullFixture(t)
	old, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(old) })
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "peer.md"), []byte("# Peer\n\nunrelated dirty input\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "peer-notes.md"), []byte("untracked peer file\n"), 0644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runLLMSFull([]string{"--check", "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var got llmsFullResult
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.OK || got.DriftCause != "" {
		t.Fatalf("result=%+v", got)
	}
	for _, want := range []string{"docs/peer.md", "peer-notes.md"} {
		if !llmsContains(got.Excluded, want) {
			t.Fatalf("excluded=%v, want %s", got.Excluded, want)
		}
	}
}

func TestLLMSFullCheckAttributesOwnedDriftAndCleanOutputIsDeterministic(t *testing.T) {
	root := llmsFullFixture(t)
	old, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(old) })
	_ = os.Chdir(root)
	before, _ := os.ReadFile(filepath.Join(root, "llms-full.txt"))
	var out, errout bytes.Buffer
	if code := runLLMSFull([]string{"--json"}, &out, &errout); code != 0 {
		t.Fatalf("clean generation: %d %s", code, errout.String())
	}
	after, _ := os.ReadFile(filepath.Join(root, "llms-full.txt"))
	if !bytes.Equal(before, after) {
		t.Fatal("clean generation changed deterministic output")
	}
	out.Reset()
	errout.Reset()
	if code := runLLMSFull([]string{"--json"}, &out, &errout); code != 0 {
		t.Fatalf("second clean generation: %d %s", code, errout.String())
	}
	again, _ := os.ReadFile(filepath.Join(root, "llms-full.txt"))
	if !bytes.Equal(after, again) {
		t.Fatal("two clean generations were not byte-identical")
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "owned.md"), []byte("# Owned\n\nchanged\n"), 0644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errout.Reset()
	code := runLLMSFull([]string{"--check", "--mine", "docs/owned.md", "--json"}, &out, &errout)
	if code != 1 {
		t.Fatalf("check code=%d stderr=%s", code, errout.String())
	}
	var got llmsFullResult
	if e := json.Unmarshal(out.Bytes(), &got); e != nil {
		t.Fatal(e)
	}
	if got.DriftCause != "owned_inputs" {
		t.Fatalf("drift=%q result=%+v", got.DriftCause, got)
	}
}

func llmsFullFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	mustMkdir := func(p string) {
		if e := os.MkdirAll(filepath.Join(root, p), 0755); e != nil {
			t.Fatal(e)
		}
	}
	mustMkdir("tools")
	mustMkdir("docs")
	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	script, err := os.ReadFile(filepath.Join(repoRoot, "tools", "gen_llms_full.py"))
	if err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{"tools/gen_llms_full.py": script, "llms.txt": []byte("# Map\n\n- [Owned](docs/owned.md)\n- [Peer](docs/peer.md)\n"), "docs/owned.md": []byte("# Owned\n\nbase\n"), "docs/peer.md": []byte("# Peer\n\nbase peer\n")}
	for p, b := range files {
		if e := os.WriteFile(filepath.Join(root, filepath.FromSlash(p)), b, 0644); e != nil {
			t.Fatal(e)
		}
	}
	run := func(name string, args ...string) {
		cmd := exec.Command(name, args...)
		cmd.Dir = root
		if b, e := cmd.CombinedOutput(); e != nil {
			t.Fatalf("%s %v: %v: %s", name, args, e, b)
		}
	}
	py := "python3"
	if filepath.Separator == '\\' {
		py = "python"
	}
	run(py, "tools/gen_llms_full.py", "--root", root)
	run("git", "init")
	run("git", "config", "user.email", "fixture@example.test")
	run("git", "config", "user.name", "fixture")
	run("git", "add", ".")
	run("git", "commit", "-m", "fixture")
	return root
}
func llmsGitOutput(t *testing.T, root string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	b, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, b)
	}
	return string(b)
}

func llmsContains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
