// Tests for managedinventory's CLI contract: flag/argument validation and the
// load-vs-validate exit codes through run(), the git-archaeology error paths,
// and the discovery replay that grounds the generated report at one immutable
// revision. The git fixtures live in temporary repositories; no tree in this
// repository is written and no binary is produced.
package main

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/managedinventory"
)

func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w
	done := make(chan string)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		done <- buf.String()
	}()
	fn()
	_ = w.Close()
	os.Stderr = old
	return <-done
}

// initTempRepo seeds a minimal git repository containing one committed
// documentation file, returning the root and the seed commit SHA.
func initTempRepo(t *testing.T) (root, rev string) {
	t.Helper()
	root = t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v failed: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	git("config", "user.email", "test@example.com")
	git("config", "user.name", "Managed Inventory Test")
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	seed := "the performance gate lives here\nand a second gate line\n"
	if err := os.WriteFile(filepath.Join(root, "docs", "gateway.md"), []byte(seed), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", ".")
	git("commit", "-q", "-m", "seed")
	return root, git("rev-parse", "HEAD")
}

func seedCatalog(rev string) managedinventory.Catalog {
	return managedinventory.Catalog{
		Schema: managedinventory.Schema,
		Discovery: managedinventory.Discovery{
			Revision: rev,
			Scope:    []string{"docs"},
			Queries: []managedinventory.DiscoveryQuery{{
				ID:            "gate-mentions",
				Pattern:       "gate",
				Paths:         []string{"docs"},
				ExpectedLines: 2,
				ExpectedFiles: 1,
			}},
		},
		Objects: []managedinventory.Object{{
			ID:       "session",
			Evidence: []string{"docs/gateway.md"},
		}},
	}
}

func TestRunFlagArgumentValidation(t *testing.T) {
	cases := []struct {
		name string
		args []string
	}{
		{"check and write conflict", []string{"--check", "--write"}},
		{"unexpected positional", []string{"extra-arg"}},
		{"unknown flag", []string{"--bogus"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if code := run(tc.args); code != 2 {
				t.Fatalf("run(%v) exit = %d, want 2", tc.args, code)
			}
		})
	}
}

func TestRunLoadAndValidationExitCodes(t *testing.T) {
	root := t.TempDir()
	badJSON := filepath.Join(root, "trailing.json")
	if err := os.WriteFile(badJSON, []byte(`{"schema":"x"} {"schema":"y"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	badSchema := filepath.Join(root, "wrong-schema.json")
	if err := os.WriteFile(badSchema, []byte(`{"schema":"bogus"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name           string
		args           []string
		wantCode       int
		stderrContains string
	}{
		{
			name:           "missing source file",
			args:           []string{"--root", root, "--source", "absent.json"},
			wantCode:       1,
			stderrContains: "managedinventory:",
		},
		{
			name:           "trailing JSON value is rejected",
			args:           []string{"--root", root, "--source", "trailing.json"},
			wantCode:       1,
			stderrContains: "decode managed inventory",
		},
		{
			name:           "wrong schema fails validation",
			args:           []string{"--root", root, "--source", "wrong-schema.json"},
			wantCode:       1,
			stderrContains: "SCHEMA",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var code int
			stderr := captureStderr(t, func() { code = run(tc.args) })
			if code != tc.wantCode {
				t.Fatalf("run(%v) exit = %d, want %d (stderr: %s)", tc.args, code, tc.wantCode, stderr)
			}
			if !strings.Contains(stderr, tc.stderrContains) {
				t.Fatalf("stderr %q does not contain %q", stderr, tc.stderrContains)
			}
		})
	}
}

func TestVerifyDiscoveryReplay(t *testing.T) {
	root, rev := initTempRepo(t)
	if err := verifyDiscovery(root, seedCatalog(rev)); err != nil {
		t.Fatalf("verifyDiscovery rejected a catalog grounded at %s: %v", rev[:12], err)
	}
}

func TestVerifyDiscoveryDriftDetection(t *testing.T) {
	root, rev := initTempRepo(t)
	c := seedCatalog(rev)
	c.Discovery.Queries[0].ExpectedLines = 3
	err := verifyDiscovery(root, c)
	if err == nil {
		t.Fatal("verifyDiscovery accepted drifted expected line counts")
	}
	if !strings.Contains(err.Error(), "drifted") {
		t.Fatalf("drift error does not name the drift: %v", err)
	}
}

func TestVerifyDiscoveryFailurePaths(t *testing.T) {
	root, rev := initTempRepo(t)
	cases := []struct {
		name           string
		mutate         func(c *managedinventory.Catalog)
		stderrContains string
	}{
		{
			name:           "unavailable revision",
			mutate:         func(c *managedinventory.Catalog) { c.Discovery.Revision = "0000000000000000000000000000000000000000" },
			stderrContains: "unavailable",
		},
		{
			name: "absent evidence path",
			mutate: func(c *managedinventory.Catalog) {
				c.Objects[0].Evidence = []string{"docs/absent.md"}
			},
			stderrContains: "grounding",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := seedCatalog(rev)
			tc.mutate(&c)
			err := verifyDiscovery(root, c)
			if err == nil {
				t.Fatalf("verifyDiscovery accepted a catalog with %s", tc.name)
			}
			if !strings.Contains(err.Error(), tc.stderrContains) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.stderrContains)
			}
		})
	}
}

func TestGitOutputFormatsGitFailures(t *testing.T) {
	_, err := gitOutput(t.TempDir(), "rev-parse", "--show-toplevel")
	if err == nil {
		t.Fatal("gitOutput succeeded outside a git repository")
	}
	if !strings.HasPrefix(err.Error(), "git rev-parse --show-toplevel:") {
		t.Fatalf("git failure error lost its command prefix: %v", err)
	}
}
