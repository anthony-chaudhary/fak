package armbench

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestPonytailPromptfooInventoryAndNoSimulation(t *testing.T) {
	root := t.TempDir()
	for p := range promptfooInputs {
		q := filepath.Join(root, filepath.FromSlash(p))
		if err := os.MkdirAll(filepath.Dir(q), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(q, []byte(p), 0644); err != nil {
			t.Fatal(err)
		}
	}
	r, err := runPonytailPromptfoo(root, filepath.Join(root, "out"), false)
	if err != nil {
		t.Fatal(err)
	}
	if r.Complete {
		t.Fatal("dry inventory must not claim attempts")
	}
	if len(r.Cells) != 10 {
		t.Fatalf("cells=%d", len(r.Cells))
	}
	if len(r.Inputs) != 12 {
		t.Fatalf("inputs=%d", len(r.Inputs))
	}
	if r.ValueClaim != "none" {
		t.Fatal("must make no fak claim")
	}
	for _, c := range r.Cells {
		if c.Status != "not-attempted" || c.Attempts != 0 {
			t.Fatalf("simulated cell: %+v", c)
		}
	}
	seen := map[string]int{}
	for _, c := range r.Cells {
		seen[c.Config+"|"+c.Provider]++
		wantArms := 2
		if c.Config == "promptfooconfig.yaml" {
			wantArms = 3
		}
		if len(c.Arms) != wantArms {
			t.Fatalf("%s arms=%v", c.Config, c.Arms)
		}
	}
	if seen["promptfooconfig.gpt.yaml|openai:gpt-5.4-mini"] != 1 || seen["promptfooconfig.gpt-newest.yaml|openai:gpt-5.4-mini"] != 1 {
		t.Fatal("duplicate model declarations must remain distinct config cells")
	}
}

func TestVerifyPonytailRevisionFailsClosed(t *testing.T) {
	root := t.TempDir()
	cmd := exec.Command("git", "init")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	if err := verifyPonytailRevision(root); err == nil {
		t.Fatal("uncommitted or wrong checkout must not be accepted as pinned")
	}
}

func TestPromptfooResultRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "result.json")
	if err := os.WriteFile(path, []byte(`{"results":{"results":[{},{}]}}`), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := promptfooResultRows(path)
	if err != nil || got != 2 {
		t.Fatalf("rows=%d err=%v", got, err)
	}
}
