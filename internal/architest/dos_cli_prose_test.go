package architest

// dos_cli_prose_test.go keeps the executable DOS refusal lookup spelling aligned with
// the installed CLI (#3869). The MCP/API seam deliberately remains dos_check_reason;
// only command prose moved to `dos man wedge <TOKEN> --explain`.

import (
	"errors"
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

	// Search tracked, live prose rather than walking the peer-dirty working tree. Frozen
	// external corpora and historical archives retain the bytes they captured; they are
	// evidence, not commands an agent should execute now.
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
		t.Fatalf("tracked live prose still names the removed DOS CLI spelling; use `%s` for commands and keep dos_check_reason only for MCP/API identifiers:\n%s",
			currentCLI, out)
	}
	exitErr = nil
	if !errors.As(grepErr, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("scan tracked live CLI prose: %v: %s", grepErr, out)
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
