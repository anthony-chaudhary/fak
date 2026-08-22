package architest

// dos_cli_prose_test.go keeps the executable DOS refusal lookup spelling aligned with
// the installed CLI (#3869). The MCP/API seam deliberately remains dos_check_reason;
// only command prose moved to `dos man wedge <TOKEN> --explain`.

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestDOSCLIProseUsesManWedgeAndKeepsMCPIdentifier(t *testing.T) {
	root := filepath.Dir(internalDir(t))
	staleCLI := strings.Join([]string{"dos ", "check-reason"}, "")
	const currentCLI = "dos man wedge <TOKEN> --explain"

	out, found, err := scanTrackedCLIProse(root, staleCLI)
	if err != nil {
		t.Fatalf("scan tracked live CLI prose: %v", err)
	}
	if found {
		t.Fatalf("tracked live prose still names the removed DOS CLI spelling; use `%s` for commands and keep dos_check_reason only for MCP/API identifiers:\n%s",
			currentCLI, out)
	}

	agents, err := os.ReadFile(filepath.Join(root, "AGENTS.md"))
	if err != nil {
		t.Fatalf("read AGENTS.md: %v", err)
	}
	if !strings.Contains(string(agents), "`"+currentCLI+"`") {
		t.Errorf("AGENTS.md recovery instructions do not name the supported `%s` command", currentCLI)
	}

	policy, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "guard-default-policy.json"))
	if err != nil {
		t.Fatalf("read guard-default-policy.json: %v", err)
	}
	if !strings.Contains(string(policy), "mcp__dos__dos_check_reason") {
		t.Error("guard-default-policy.json lost the intentional dos_check_reason MCP identifier")
	}
}

// scanTrackedCLIProse uses git in a live checkout so peer-untracked files cannot red the
// architecture law. fak validate intentionally runs from a metadata-free archive; there,
// every file is tracked by construction, so a filesystem scan is the equivalent contract.
func scanTrackedCLIProse(root, staleCLI string) ([]byte, bool, error) {
	if _, err := os.Stat(filepath.Join(root, ".git")); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return scanArchivedCLIProse(root, staleCLI)
		}
		return nil, false, fmt.Errorf("stat .git: %w", err)
	}

	runGrep := func(binary string) ([]byte, error) {
		cmd := exec.Command(binary, "grep", "-n", "-I", "-F", "-e", staleCLI, "--", ".",
			":(exclude)experiments/**",
			":(exclude)docs/archive/**",
			":(exclude)docs/releases/**",
			":(exclude)**/testdata/**",
		)
		cmd.Dir = root
		return cmd.CombinedOutput()
	}

	out, grepErr := runGrep("git")
	var exitErr *exec.ExitError
	if errors.As(grepErr, &exitErr) && exitErr.ExitCode() != 1 {
		// A Windows detached worktree stores a Windows gitdir path in its .git file.
		// Linux git under WSL cannot resolve that path, while Git for Windows can.
		if _, lookErr := exec.LookPath("git.exe"); lookErr == nil {
			out, grepErr = runGrep("git.exe")
		}
	}
	if grepErr == nil {
		return out, true, nil
	}
	exitErr = nil
	if !errors.As(grepErr, &exitErr) || exitErr.ExitCode() != 1 {
		return nil, false, fmt.Errorf("git grep: %w: %s", grepErr, out)
	}
	return out, false, nil
}

func scanArchivedCLIProse(root, staleCLI string) ([]byte, bool, error) {
	var matches strings.Builder
	needle := []byte(staleCLI)
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if archiveProseExcludedDir(rel) {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if bytes.IndexByte(data, 0) >= 0 {
			return nil
		}
		for offset := 0; offset < len(data); {
			at := bytes.Index(data[offset:], needle)
			if at < 0 {
				break
			}
			at += offset
			line := 1 + bytes.Count(data[:at], []byte{'\n'})
			fmt.Fprintf(&matches, "%s:%d\n", filepath.ToSlash(rel), line)
			offset = at + len(needle)
		}
		return nil
	})
	if err != nil {
		return nil, false, err
	}
	out := []byte(matches.String())
	return out, len(out) > 0, nil
}

func archiveProseExcludedDir(rel string) bool {
	rel = filepath.ToSlash(rel)
	if rel == "experiments" || rel == "docs/archive" || rel == "docs/releases" {
		return true
	}
	for _, part := range strings.Split(rel, "/") {
		if part == "testdata" {
			return true
		}
	}
	return false
}

func TestScanTrackedCLIProseWithoutGitMetadata(t *testing.T) {
	root := t.TempDir()
	staleCLI := strings.Join([]string{"dos ", "check-reason"}, "")
	for _, dir := range []string{"cmd", "experiments", "docs/archive", "docs/releases", "pkg/testdata"} {
		if err := os.MkdirAll(filepath.Join(root, filepath.FromSlash(dir)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{"experiments/frozen.md", "docs/archive/old.md", "docs/releases/old.md", "pkg/testdata/fixture.txt"} {
		if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(path)), []byte(staleCLI), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "cmd", "binary.bin"), append([]byte(staleCLI), 0), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, found, err := scanTrackedCLIProse(root, staleCLI); err != nil || found {
		t.Fatalf("excluded archive prose scan = found %v, err %v, out %q", found, err, out)
	}

	const livePath = "cmd/live.md"
	if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(livePath)), []byte("run "+staleCLI+" now\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out, found, err := scanTrackedCLIProse(root, staleCLI)
	if err != nil || !found || !strings.Contains(string(out), livePath+":1") {
		t.Fatalf("live archive prose scan = found %v, err %v, out %q", found, err, out)
	}
}
