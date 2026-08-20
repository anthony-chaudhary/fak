package selfinstall

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRealRunnerKeepsGoWorkOutsideRepoScratch(t *testing.T) {
	repoTmp := filepath.Join(t.TempDir(), "_scratch", "go-tmp")
	if err := os.MkdirAll(repoTmp, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOTMPDIR", repoTmp)

	out, ok := RealRunner(context.Background(), "", "go", "env", "GOTMPDIR")
	if !ok {
		t.Fatalf("go env GOTMPDIR failed: %s", out)
	}
	if got, want := strings.TrimSpace(out), filepath.Clean(os.TempDir()); !strings.EqualFold(got, want) {
		t.Fatalf("child GOTMPDIR = %q, want OS temp %q outside sweepable repo scratch", got, want)
	}
}

func TestGoRunnerEnvLeavesNonGoChildrenUnchanged(t *testing.T) {
	if got := goRunnerEnv("git", []string{"GOTMPDIR=repo-scratch"}, "os-temp"); got != nil {
		t.Fatalf("non-Go child env = %v, want nil (inherit unchanged)", got)
	}
}
