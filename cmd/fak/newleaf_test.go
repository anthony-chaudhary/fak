package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewLeafCommandDryRun(t *testing.T) {
	root := commandNewLeafWorkspace(t)
	var stdout, stderr bytes.Buffer
	rc := runNewLeaf(&stdout, &stderr, []string{
		"--workspace", root,
		"--tier", "primitive",
		"--dry-run",
		"fedtrust",
	})
	if rc != 0 {
		t.Fatalf("runNewLeaf rc=%d stderr=%q", rc, stderr.String())
	}
	var report struct {
		Name   string `json:"name"`
		Tier   string `json:"tier"`
		DryRun bool   `json:"dry_run"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal report: %v", err)
	}
	if report.Name != "fedtrust" || report.Tier != "primitive" || !report.DryRun {
		t.Fatalf("unexpected report: %+v", report)
	}
	if _, err := os.Stat(filepath.Join(root, "internal", "fedtrust")); !os.IsNotExist(err) {
		t.Fatalf("leaf dir exists after dry-run, err=%v", err)
	}
}

func TestNewLeafCommandCreatesRegisteredLeaf(t *testing.T) {
	root := commandNewLeafWorkspace(t)
	var stdout, stderr bytes.Buffer
	rc := runNewLeaf(&stdout, &stderr, []string{
		"--workspace", root,
		"--tier", "composer",
		"--register",
		"--summary", "a federated trust gate",
		"fedtrust",
	})
	if rc != 0 {
		t.Fatalf("runNewLeaf rc=%d stderr=%q", rc, stderr.String())
	}
	if _, err := os.Stat(filepath.Join(root, "internal", "fedtrust", "fedtrust.go")); err != nil {
		t.Fatalf("leaf implementation was not written: %v", err)
	}
	arch := commandReadNewLeafFile(t, root, "internal/architest/architest_test.go")
	if !strings.Contains(arch, `"fedtrust": 3,`) {
		t.Fatalf("architest missing fedtrust tier: %s", arch)
	}
	reg := commandReadNewLeafFile(t, root, "internal/registrations/registrations.go")
	if !strings.Contains(reg, `_ "github.com/anthony-chaudhary/fak/internal/fedtrust"`) {
		t.Fatalf("registrations missing fedtrust: %s", reg)
	}
	dos := commandReadNewLeafFile(t, root, "dos.toml")
	if !strings.Contains(dos, `fedtrust = ["internal/fedtrust/**"]`) {
		t.Fatalf("dos missing fedtrust tree: %s", dos)
	}
	var report struct {
		Edits []string `json:"edits"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal report: %v", err)
	}
	if len(report.Edits) == 0 {
		t.Fatalf("report missing edits: %s", stdout.String())
	}
}

func TestNewLeafCommandValidatesArgs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runNewLeaf(&stdout, &stderr, []string{"--tier", "primitive", "Fed_Trust"})
	if rc != 2 {
		t.Fatalf("rc=%d, want 2", rc)
	}
	if !strings.Contains(stderr.String(), "valid lowercase Go package name") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestNewLeafSuggestTierFromImports(t *testing.T) {
	root := commandNewLeafSuggestWorkspace(t)

	// (1) no internal deps -> primitive.
	if s := commandRunSuggest(t, root, "alpha", ""); s.SuggestedTier != "primitive" {
		t.Fatalf("alpha suggested_tier = %q, want primitive (%+v)", s.SuggestedTier, s)
	}

	// (2) imports a mechanism -> suggests >= mechanism, naming the governing dep.
	beta := commandRunSuggest(t, root, "beta", "")
	if beta.SuggestedTier != "mechanism" || beta.GoverningDep != "engine" {
		t.Fatalf("beta suggestion = %+v, want mechanism governed by engine", beta)
	}

	// (3) over-declared tier -> advisory fires (never blocks; rc already 0 above).
	over := commandRunSuggest(t, root, "alpha", "composer")
	if over.Advisory == "" {
		t.Fatalf("over-declared alpha should surface an advisory: %+v", over)
	}
}

func TestNewLeafScaffoldSurfacesTierAdvisory(t *testing.T) {
	root := commandNewLeafWorkspace(t)
	var stdout, stderr bytes.Buffer
	rc := runNewLeaf(&stdout, &stderr, []string{
		"--workspace", root, "--tier", "composer", "--dry-run", "overshoot",
	})
	if rc != 0 {
		t.Fatalf("runNewLeaf rc=%d stderr=%q", rc, stderr.String())
	}
	var report struct {
		TierAdvisory string `json:"tier_advisory"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal report: %v", err)
	}
	if report.TierAdvisory == "" {
		t.Fatalf("scaffolding composer with no imports should surface a primitive advisory: %s", stdout.String())
	}
}

func commandRunSuggest(t *testing.T, root, leaf, tier string) struct {
	SuggestedTier string   `json:"suggested_tier"`
	GoverningDep  string   `json:"governing_dep"`
	InternalDeps  []string `json:"internal_deps"`
	Advisory      string   `json:"advisory"`
} {
	t.Helper()
	argv := []string{"--workspace", root, "--suggest-tier", leaf}
	if tier != "" {
		argv = append(argv, "--tier", tier)
	}
	var stdout, stderr bytes.Buffer
	if rc := runNewLeaf(&stdout, &stderr, argv); rc != 0 {
		t.Fatalf("runNewLeaf --suggest-tier %s rc=%d stderr=%q", leaf, rc, stderr.String())
	}
	var s struct {
		SuggestedTier string   `json:"suggested_tier"`
		GoverningDep  string   `json:"governing_dep"`
		InternalDeps  []string `json:"internal_deps"`
		Advisory      string   `json:"advisory"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &s); err != nil {
		t.Fatalf("unmarshal suggestion: %v (%s)", err, stdout.String())
	}
	return s
}

func commandNewLeafSuggestWorkspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	commandWriteNewLeafFile(t, root, "internal/architest/architest_test.go",
		"package architest\n\nvar tier = map[string]int{\n\t\"abi\": 0,\n\t\"engine\": 2,\n\t// new-leaf:tier\n}\n")
	commandWriteNewLeafFile(t, root, "internal/alpha/alpha.go",
		"package alpha\n\nimport \"fmt\"\n\nvar _ = fmt.Sprint\n")
	commandWriteNewLeafFile(t, root, "internal/beta/beta.go",
		"package beta\n\nimport _ \"github.com/anthony-chaudhary/fak/internal/engine\"\n")
	return root
}

func commandNewLeafWorkspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	commandWriteNewLeafFile(t, root, "internal/architest/architest_test.go", "package architest\n\nvar tier = map[string]int{\n\t\"abi\": 0,\n\t// new-leaf:tier\n}\n")
	commandWriteNewLeafFile(t, root, "internal/registrations/registrations.go", "package registrations\n\nimport (\n\t_ \"github.com/anthony-chaudhary/fak/internal/abi\"\n)\n")
	commandWriteNewLeafFile(t, root, "dos.toml", `[lanes]
concurrent = [
  "foo",
  # new-leaf:lane
]
autopick = [
  "foo",
  # new-leaf:lane
]
[lanes.trees]
foo = ["internal/foo/**"]
# new-leaf:tree
cmd = ["cmd/**"]
`)
	return root
}

func commandWriteNewLeafFile(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

func commandReadNewLeafFile(t *testing.T, root, rel string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatalf("read %s: %v", rel, err)
	}
	return string(raw)
}
