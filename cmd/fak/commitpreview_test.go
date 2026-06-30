package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// runCommit --preview is a lint-only dry run: it must never reach commitFn, so these exercise the
// exit-code contract (0 clean / 1 issues / 2 usage) without a git or a commit seam. --dir points
// at a tmp dir with no dos.toml, so lane recognition is skipped and path lanes come from the
// directory convention (internal/<leaf> -> <leaf>).

func TestRunCommitPreview_cleanIsExit0(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--dir", tmp,
		"-m", "feat(gateway): add the reclaim path (fak gateway)",
		"--path", "internal/gateway/server.go",
	})
	if code != 0 {
		t.Fatalf("want exit 0 for a clean preview, got %d (out=%q)", code, out.String())
	}
	if !strings.Contains(out.String(), "commit-preview OK") {
		t.Errorf("want OK line, got %q", out.String())
	}
}

func TestRunCommitPreview_issuesIsExit1(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--dir", tmp,
		"-m", "gateway reclaim improvements", // noun-led, no trailer
		"--path", "internal/gateway/server.go",
	})
	if code != 1 {
		t.Fatalf("want exit 1 for a defective preview, got %d (out=%q)", code, out.String())
	}
}

func TestRunCommitPreview_noMessageIsExit2(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--preview", "--dir", tmp, "--path", "internal/gateway/server.go"})
	if code != 2 {
		t.Fatalf("want exit 2 for a missing message, got %d", code)
	}
}

func TestRunCommitPreview_jsonShape(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--json", "--dir", tmp,
		"-m", "feat(gateway): add x (fak gateway)\n\nGeneration: gen/next",
		"--path", "internal/gateway/server.go",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (err=%q)", code, errb.String())
	}
	var got map[string]any
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("preview --json did not emit valid JSON: %v\n%s", err, out.String())
	}
	if got["ok"] != true {
		t.Errorf("want ok=true in JSON, got %v", got["ok"])
	}
	if got["score"] == nil || got["grade"] == nil {
		t.Errorf("preview JSON should carry score and grade, got %v", got)
	}
	if got["generation"] != "gen/next" {
		t.Errorf("preview JSON should preserve generation sidecar, got %v", got["generation"])
	}
	if got["expected_branch"] != "main" {
		t.Errorf("preview JSON should report expected branch, got %v", got["expected_branch"])
	}
}

func TestRunCommitPreview_rendersGenerationSidecar(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--dir", tmp,
		"-m", "feat(gateway): add x (fak gateway)\n\nGeneration: future",
		"--path", "internal/gateway/server.go",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (out=%q err=%q)", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "generation: gen/future") {
		t.Fatalf("preview should render normalized generation sidecar, got %q", out.String())
	}
}

func TestRunCommitPreview_doesNotCommit(t *testing.T) {
	// A preview must never invoke the commit seam. Swap in a fatal commitFn to prove it.
	withCommitFn(t, func(_ context.Context, _ safecommit.Options) (safecommit.Result, error) {
		t.Fatalf("commitFn must not be called in --preview")
		return safecommit.Result{}, nil
	})
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	_ = runCommit(&out, &errb, []string{
		"--preview", "--dir", tmp,
		"-m", "feat(gateway): add x (fak gateway)",
		"--path", "internal/gateway/server.go",
	})
}

func TestRunCommitPreview_reportsConfiguredDevelopmentBranch(t *testing.T) {
	tmp := t.TempDir()
	dos := `[branch_roles]
development_branch = "dev"

[lanes]
concurrent = ["gateway"]

[lanes.trees]
gateway = ["internal/gateway/**"]
`
	if err := os.WriteFile(filepath.Join(tmp, "dos.toml"), []byte(dos), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--dir", tmp,
		"-m", "feat(gateway): add x (fak gateway)",
		"--path", "internal/gateway/server.go",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (out=%q err=%q)", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "expected branch: dev") {
		t.Fatalf("preview should render configured expected branch, got %q", out.String())
	}
}
