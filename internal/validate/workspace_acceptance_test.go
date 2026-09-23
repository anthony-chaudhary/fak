package validate_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/validate"
)

type validationReceipt struct {
	Mine   []string `json:"mine"`
	Tested []string `json:"tested"`
	OK     bool     `json:"ok"`
	Phases []struct {
		Name   string `json:"name"`
		Status string `json:"status"`
	} `json:"phases"`
	Overlays struct {
		Checked []string `json:"checked"`
	} `json:"overlays"`
}

func TestRunValidatesPrivateModuleAgainstSiblingPublicWorkspace(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	_, privateRoot := seedDualModuleFixture(t)

	owned := filepath.Join(privateRoot, "app", "app.go")
	writeFixtureFile(t, owned, `package app

import "github.com/anthony-chaudhary/fak/pkg/value"

func Got() int { return value.Number() }

func OwnedOverlay() string { return "owned" }
`)
	writeFixtureFile(t, filepath.Join(privateRoot, "peer", "peer.go"), "package peer\n\nfunc Broken( {\n")

	var stdout, stderr bytes.Buffer
	code := validate.Run(&stdout, &stderr, []string{
		"--root", privateRoot,
		"--mine", filepath.Join("app", "app.go"),
		"--wsl-tests=false",
		"--progress=false",
		"--json",
	})
	if code != 0 {
		t.Fatalf("Run code=%d stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}

	var got validationReceipt
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode receipt: %v\n%s", err, stdout.String())
	}
	if !got.OK {
		t.Fatalf("receipt is not green: %+v", got)
	}
	if fmt.Sprint(got.Mine) != fmt.Sprint([]string{"app/app.go"}) {
		t.Fatalf("mine=%v, want [app/app.go]", got.Mine)
	}
	if !containsString(got.Tested, "fixture.test/private/app") {
		t.Fatalf("tested=%v, want private app package", got.Tested)
	}
	if !containsString(got.Overlays.Checked, "app/app.go") {
		t.Fatalf("checked overlays=%v, want owned app source", got.Overlays.Checked)
	}
	for _, name := range []string{"build", "vet", "test"} {
		status, ok := phaseStatus(got, name)
		if !ok || status != "ok" {
			t.Fatalf("phase %q status=%q present=%v; phases=%+v", name, status, ok, got.Phases)
		}
	}
}

func TestRunRefusesMinePathOutsidePrivateRoot(t *testing.T) {
	_, privateRoot := seedDualModuleFixture(t)
	writeFixtureFile(t, filepath.Join(filepath.Dir(privateRoot), "escape.go"), "package escape\n")

	var stdout, stderr bytes.Buffer
	code := validate.Run(&stdout, &stderr, []string{
		"--root", privateRoot,
		"--mine", filepath.Join("..", "escape.go"),
		"--wsl-tests=false",
		"--json",
	})
	if code == 0 {
		t.Fatalf("escaping --mine path accepted: stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
	msg := strings.ToLower(stderr.String() + stdout.String())
	if !strings.Contains(msg, "outside") && !strings.Contains(msg, "escape") {
		t.Fatalf("refusal did not explain containment: code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
}

func seedDualModuleFixture(t *testing.T) (string, string) {
	t.Helper()
	parent := t.TempDir()
	publicRoot := filepath.Join(parent, "fak")
	privateRoot := filepath.Join(parent, "fak-private-worktree")

	commitFixture(t, publicRoot, map[string]string{
		"go.mod":             "module github.com/anthony-chaudhary/fak\n\ngo 1.26\n",
		"pkg/value/value.go": "package value\n\nfunc Number() int { return 42 }\n",
	})
	commitFixture(t, privateRoot, map[string]string{
		"go.mod":  "module fixture.test/private\n\ngo 1.26\n\nrequire github.com/anthony-chaudhary/fak v0.0.0\n",
		"go.work": "go 1.26\n\nuse (\n\t.\n\t../fak\n)\n",
		"app/app.go": `package app

import "github.com/anthony-chaudhary/fak/pkg/value"

func Got() int { return value.Number() }
`,
		"app/app_test.go": `package app

import "testing"

func TestGot(t *testing.T) {
	if Got() != 42 {
		t.Fatalf("Got() = %d, want 42", Got())
	}
}
`,
		"peer/peer.go": "package peer\n\nfunc Clean() {}\n",
	})
	return publicRoot, privateRoot
}

func commitFixture(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for path, body := range files {
		writeFixtureFile(t, filepath.Join(root, filepath.FromSlash(path)), body)
	}
	runFixtureGit(t, root, "init", "-q")
	runFixtureGit(t, root, "add", ".")
	runFixtureGit(t, root, "-c", "user.name=Validator Test", "-c", "user.email=validator@example.invalid", "commit", "-q", "-m", "fixture")
}

func writeFixtureFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func runFixtureGit(t *testing.T, root string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", "-C", root)
	cmd.Args = append(cmd.Args, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
	}
}

func phaseStatus(receipt validationReceipt, name string) (string, bool) {
	for _, phase := range receipt.Phases {
		if phase.Name == name {
			return phase.Status, true
		}
	}
	return "", false
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
