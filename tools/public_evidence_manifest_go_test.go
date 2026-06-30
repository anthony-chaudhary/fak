package tools_test

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPublicEvidenceManifestCitationCore(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	toolsDir, err := filepath.Abs(wd)
	if err != nil {
		t.Fatal(err)
	}
	quotedToolsDir, err := json.Marshal(toolsDir)
	if err != nil {
		t.Fatal(err)
	}
	code := fmt.Sprintf(`
import sys
sys.path.insert(0, %s)
import public_evidence_manifest as pem

assert pem._norm("a\\b\\c") == "a/b/c"
assert pem._norm("./experiments/x.json") == "experiments/x.json"

exp, res = pem._cited_in("see [data](experiments/qwen/run.json) for detail")
assert "experiments/qwen/run.json" in exp

exp, res = pem._cited_in("provenance in 8B-RESULTS.md and more")
assert "8B-RESULTS.md" in res

exp, _ = pem._cited_in("link ./fak/experiments/a/b.csv here")
assert "experiments/a/b.csv" in exp

exp, _ = pem._cited_in("run x --output experiments/tmp/out.json")
assert "experiments/tmp/out.json" not in exp

exp, _ = pem._cited_in("results saved to experiments/tmp/out.json earlier")
assert "experiments/tmp/out.json" in exp

exp, res = pem._cited_in("just prose, no artifacts here")
assert exp == set() and res == set()
`, string(quotedToolsDir))
	python := strings.TrimSpace(os.Getenv("FAK_PYTHON"))
	if python == "" {
		python = "python"
	}
	cmd := exec.Command(python, "-c", code)
	cmd.Dir = toolsDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("public evidence manifest citation core failed: %v\n%s", err, out)
	}
}
