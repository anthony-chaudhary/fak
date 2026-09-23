package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type cliValidationReceipt struct {
	Schema string   `json:"schema"`
	Mode   string   `json:"mode"`
	Mine   []string `json:"mine"`
	Tested []string `json:"tested"`
	OK     bool     `json:"ok"`
	Phases []struct {
		Name   string `json:"name"`
		Status string `json:"status"`
	} `json:"phases"`
}

func TestStandaloneCommandMatchesFakValidate(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo := seedCLIValidationFixture(t)
	args := []string{
		"--root", repo,
		"--mine", filepath.Join("p", "p.go"),
		"--wsl-tests=false",
		"--progress=false",
		"--json",
	}

	standalone := runValidationCommand(t, ".", args)
	existing := runValidationCommand(t, filepath.Join("..", "fak"), append([]string{"validate"}, args...))

	if standalone.Schema != existing.Schema || standalone.Mode != existing.Mode || standalone.OK != existing.OK {
		t.Fatalf("standalone summary=%+v, fak validate summary=%+v", standalone, existing)
	}
	if !reflect.DeepEqual(standalone.Mine, existing.Mine) || !reflect.DeepEqual(standalone.Tested, existing.Tested) {
		t.Fatalf("standalone mine/tested=%v/%v, fak validate=%v/%v", standalone.Mine, standalone.Tested, existing.Mine, existing.Tested)
	}
	if !reflect.DeepEqual(cliPhaseOutcomes(standalone), cliPhaseOutcomes(existing)) {
		t.Fatalf("standalone phases=%v, fak validate phases=%v", cliPhaseOutcomes(standalone), cliPhaseOutcomes(existing))
	}
}

func seedCLIValidationFixture(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	writeCLIValidationFile(t, filepath.Join(repo, "go.mod"), "module fixture.test/cli\n\ngo 1.26\n")
	writeCLIValidationFile(t, filepath.Join(repo, "p", "p.go"), "package p\n\nfunc Value() int { return 1 }\n")
	writeCLIValidationFile(t, filepath.Join(repo, "p", "p_test.go"), `package p

import "testing"

func TestValue(t *testing.T) {
	if Value() != 1 {
		t.Fatal("unexpected value")
	}
}
`)
	runCLIValidationGit(t, repo, "init", "-q")
	runCLIValidationGit(t, repo, "add", ".")
	runCLIValidationGit(t, repo, "-c", "user.name=Validator Test", "-c", "user.email=validator@example.invalid", "commit", "-q", "-m", "fixture")
	writeCLIValidationFile(t, filepath.Join(repo, "p", "p.go"), "package p\n\nfunc Value() int { return 1 + 0 }\n")
	return repo
}

func runValidationCommand(t *testing.T, packagePath string, args []string) cliValidationReceipt {
	t.Helper()
	cmdArgs := append([]string{"run", packagePath}, args...)
	cmd := exec.Command("go", cmdArgs...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("go %s: %v\nstderr: %s\nstdout: %s", strings.Join(cmdArgs, " "), err, stderr.String(), stdout.String())
	}
	var receipt cliValidationReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode go %s output: %v\n%s", strings.Join(cmdArgs, " "), err, stdout.String())
	}
	if !receipt.OK {
		t.Fatalf("go %s returned non-green receipt: %+v", strings.Join(cmdArgs, " "), receipt)
	}
	return receipt
}

func writeCLIValidationFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func runCLIValidationGit(t *testing.T, root string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", "-C", root)
	cmd.Args = append(cmd.Args, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
	}
}

func cliPhaseOutcomes(receipt cliValidationReceipt) map[string]string {
	outcomes := make(map[string]string, len(receipt.Phases))
	for _, phase := range receipt.Phases {
		outcomes[phase.Name] = phase.Status
	}
	return outcomes
}
