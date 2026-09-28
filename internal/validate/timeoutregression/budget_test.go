// Package timeoutregression witnesses #13558 through the production validate.Run
// path. It sits outside internal/validate so it compiles against a parent that
// lacks the fix, as the fix(*) red-then-green symptom witness requires.
package timeoutregression

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/validate"
)

// TestRunGivesGoTestTheValidateBudget fails while the isolated go test runs under
// Go's 10m default: the fixture's own test reads the -timeout it was handed and
// requires one derived from the 30m validate budget.
func TestRunGivesGoTestTheValidateBudget(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not on PATH")
	}
	root := t.TempDir()
	for path, body := range map[string]string{
		"go.mod":           "module fixture.test/budget\n\ngo 1.26\n",
		"budget/budget.go": "package budget\n",
		"budget/budget_test.go": `package budget

import (
	"flag"
	"testing"
	"time"
)

func TestTimeoutFollowsValidateBudget(t *testing.T) {
	got, err := time.ParseDuration(flag.Lookup("test.timeout").Value.String())
	if err != nil {
		t.Fatal(err)
	}
	if got <= 20*time.Minute || got >= 30*time.Minute {
		t.Fatalf("go test -timeout = %s, want one derived from the 30m validate budget", got)
	}
}
`,
	} {
		writeFile(t, filepath.Join(root, filepath.FromSlash(path)), body)
	}
	for _, args := range [][]string{
		{"init", "-q"},
		{"add", "."},
		{"-c", "user.name=Validator Test", "-c", "user.email=validator@example.invalid", "commit", "-q", "-m", "fixture"},
	} {
		if out, err := exec.Command("git", append([]string{"-C", root}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
	}
	writeFile(t, filepath.Join(root, "budget", "budget.go"), "package budget\n\nfunc Owned() {}\n")

	var stdout, stderr bytes.Buffer
	code := validate.Run(&stdout, &stderr, []string{
		"--root", root,
		"--mine", "budget/budget.go",
		"--test-only",
		"--wsl-tests=false",
		"--progress=false",
		"--timeout", "30m",
	})
	if code != 0 {
		t.Fatalf("validate.Run code=%d; want the fixture's timeout check green\nstdout:\n%s\nstderr:\n%s", code, stdout.String(), stderr.String())
	}
}

func writeFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
