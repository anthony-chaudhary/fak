package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestMacOSInstallScriptAndFormulaValidation witnesses issue #12250:
// Homebrew tap formula in Formula/fak.rb, macOS Gatekeeper-clean install.sh,
// embedded shell completions (bash, zsh, fish), and universal binary workflow.
func TestMacOSInstallScriptAndFormulaValidation(t *testing.T) {
	repoRoot := findRepoRootForTest(t)

	// -------------------------------------------------------------------------
	// 1. Homebrew Formula (Formula/fak.rb)
	// -------------------------------------------------------------------------
	formulaPath := filepath.Join(repoRoot, "Formula", "fak.rb")
	formulaBytes, err := os.ReadFile(formulaPath)
	if err != nil {
		t.Fatalf("Formula/fak.rb unreadable: %v", err)
	}
	formula := string(formulaBytes)

	// Verify Ruby syntax via `ruby -c` if ruby is installed on host
	if rubyPath, err := exec.LookPath("ruby"); err == nil {
		cmd := exec.Command(rubyPath, "-c", formulaPath)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("ruby -c Formula/fak.rb failed: %v\n%s", err, string(out))
		}
		if !strings.Contains(string(out), "Syntax OK") {
			t.Errorf("ruby -c Formula/fak.rb expected 'Syntax OK', got: %s", string(out))
		}
	}

	requiredFormulaTokens := []string{
		"class Fak < Formula",
		"desc ",
		"homepage ",
		`license "Apache-2.0"`,
		"on_macos do",
		"Hardware::CPU.arm?",
		`bin.install "fak"`,
		`generate_completions_from_executable(bin/"fak", "completion")`,
		"test do",
	}
	for _, tok := range requiredFormulaTokens {
		if !strings.Contains(formula, tok) {
			t.Errorf("Formula/fak.rb missing required token %q", tok)
		}
	}

	// -------------------------------------------------------------------------
	// 2. install.sh macOS and Architecture Detection
	// -------------------------------------------------------------------------
	installScriptPath := filepath.Join(repoRoot, "install.sh")
	installBytes, err := os.ReadFile(installScriptPath)
	if err != nil {
		t.Fatalf("install.sh unreadable: %v", err)
	}
	installScript := string(installBytes)

	// Verify POSIX sh syntax via `sh -n`
	if shPath, err := exec.LookPath("sh"); err == nil {
		cmd := exec.Command(shPath, "-n", installScriptPath)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("sh -n install.sh failed: %v\n%s", err, string(out))
		}
	}

	requiredInstallTokens := []string{
		"Darwin) GOOS=darwin ;;",
		"arm64|aarch64) GOARCH=arm64 ;;",
		"have brew",
		"xattr -d com.apple.quarantine",
		"sha256sum",
	}
	for _, tok := range requiredInstallTokens {
		if !strings.Contains(installScript, tok) {
			t.Errorf("install.sh missing required token %q", tok)
		}
	}

	// -------------------------------------------------------------------------
	// 3. GitHub Actions Release Workflow (.github/workflows/release-macos.yml)
	// -------------------------------------------------------------------------
	wfPath := filepath.Join(repoRoot, ".github", "workflows", "release-macos.yml")
	wfBytes, err := os.ReadFile(wfPath)
	if err != nil {
		t.Fatalf(".github/workflows/release-macos.yml unreadable: %v", err)
	}
	wf := string(wfBytes)

	requiredWfTokens := []string{
		"runs-on: macos-latest",
		"GOOS=darwin GOARCH=arm64",
		"GOOS=darwin GOARCH=amd64",
		"lipo -create",
		"codesign",
		"notarytool",
		"stapler",
		"release-macos",
	}
	for _, tok := range requiredWfTokens {
		if !strings.Contains(wf, tok) {
			t.Errorf(".github/workflows/release-macos.yml missing required token %q", tok)
		}
	}

	// -------------------------------------------------------------------------
	// 4. Shell Completion Generation (bash, zsh, fish)
	// -------------------------------------------------------------------------
	var bashBuf, zshBuf, fishBuf bytes.Buffer
	writeBashCompletion(&bashBuf)
	writeZshCompletion(&zshBuf)
	writeFishCompletion(&fishBuf)

	bashOutput := bashBuf.String()
	if !strings.Contains(bashOutput, "_fak()") || !strings.Contains(bashOutput, "complete -o default -F _fak fak") {
		t.Errorf("writeBashCompletion output invalid:\n%s", bashOutput)
	}

	zshOutput := zshBuf.String()
	if !strings.Contains(zshOutput, "#compdef fak") || !strings.Contains(zshOutput, "_describe") {
		t.Errorf("writeZshCompletion output invalid:\n%s", zshOutput)
	}

	fishOutput := fishBuf.String()
	if !strings.Contains(fishOutput, "complete -c fak") || !strings.Contains(fishOutput, "__fak_needs_command") {
		t.Errorf("writeFishCompletion output invalid:\n%s", fishOutput)
	}

	// Verify command dispatching handles --shell flag and positional args
	var runOut, runErr bytes.Buffer
	if code := runCompletion(&runOut, &runErr, []string{"-h"}); code != 0 {
		t.Errorf("runCompletion(-h) = %d, want 0", code)
	}
	runOut.Reset()
	if code := runCompletion(&runOut, &runErr, []string{"--shell=zsh"}); code != 0 {
		t.Errorf("runCompletion(--shell=zsh) = %d, want 0", code)
	}
	runOut.Reset()
	if code := runCompletion(&runOut, &runErr, []string{"bash"}); code != 0 {
		t.Errorf("runCompletion(bash) = %d, want 0", code)
	}
	runOut.Reset()
	runErr.Reset()
	if code := runCompletion(&runOut, &runErr, []string{"unsupported"}); code != 2 {
		t.Errorf("runCompletion(unsupported) = %d, want 2", code)
	}
}

func findRepoRootForTest(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("could not find repo root with go.mod starting from %s", dir)
		}
		dir = parent
	}
}
