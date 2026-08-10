package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func TestHelpIdentifiesIndependentDevelopmentArtifact(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"help"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"fak-dev — repository-development tooling", "index ownership", "wiki <structure|verify|fresh|score>", "separately buildable 'fak' artifact"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("help missing %q:\n%s", want, out.String())
		}
	}
}

func TestOwnershipCommandUsesInventory(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"index", "ownership", "--json", "--root", devindex.FindRoot(".")}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got devindex.OwnershipReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, command := range got.Commands {
		if command.Name == "index" && command.Owner == devindex.OwnerDev && command.DispatchTarget == "fak-dev" {
			found = true
		}
	}
	if !found {
		t.Fatal("ownership inventory does not authorize index on fak-dev")
	}
}

func TestIndexLaneExecutesThroughDevelopmentArtifact(t *testing.T) {
	var out, errOut bytes.Buffer
	root := devindex.FindRoot(".")
	if code := run(&out, &errOut, []string{"index", "lane", "cmd/fak/main.go", "--root", root, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"lane"`) {
		t.Fatalf("index lane did not execute through fak-dev:\n%s", out.String())
	}
}

func TestWikiStructureExecutesThroughDevelopmentArtifact(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"dos.toml": `[lanes.trees]
gateway = ["internal/gateway/**"]
`,
		"README.md":                   "# fixture\n",
		"internal/gateway/gateway.go": "package gateway\n",
	}
	for rel, body := range files {
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"wiki", "structure", "--root", root, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got struct {
		Repo     string `json:"repo"`
		Sections []any  `json:"sections"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode output: %v\n%s", err, out.String())
	}
	if got.Repo == "" || len(got.Sections) == 0 {
		t.Fatalf("wiki structure did not execute: %+v", got)
	}
}

func TestCatchupExecutesThroughDevelopmentArtifact(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"catchup", "--workspace", t.TempDir(), "--no-index", "--intake-behind", "3", "--intake-total", "10", "--json"}); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"catchup_backlog": 3`) {
		t.Fatalf("catchup did not execute through fak-dev:\n%s", out.String())
	}
}

func TestBackendScaffoldExecutesThroughDevelopmentArtifact(t *testing.T) {
	dir := t.TempDir()
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"backend", "scaffold", "artifacttest", "--lane", "custom", "--dir", dir}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "artifacttest_backend.go")); err != nil {
		t.Fatalf("fak-dev backend did not write scaffold: %v", err)
	}
}

func TestOrientExecutesThroughDevelopmentArtifact(t *testing.T) {
	var out, errOut bytes.Buffer
	root := devindex.FindRoot(".")
	if code := run(&out, &errOut, []string{"orient", "--root", root, "--leases=false", "--json", "--paths", "cmd/fak/main.go"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"lane"`) {
		t.Fatalf("orient did not execute through fak-dev:\n%s", out.String())
	}
}

func TestRunDispatchesWhatsChangedUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"whats-changed", "--since", "HEAD"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "--paths is required") {
		t.Fatalf("stderr=%s", errOut.String())
	}
}
func TestRunDispatchesBoundaryUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"boundary", "extra"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}

func TestRunDispatchesCIPreflightUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"ci-preflight", "--repo", t.TempDir(), "--ref", "missing", "--json"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}

func TestRuntimeSourceDoesNotDispatchCIPreflight(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "fak", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(src, []byte(`case "ci-preflight":`)) {
		t.Fatal("runtime fak still dispatches dev-owned ci-preflight")
	}
}
func TestRuntimeSourceDoesNotDispatchBoundary(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "fak", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(src, []byte(`case "boundary":`)) {
		t.Fatal("runtime fak still dispatches dev-owned boundary")
	}
}
func TestRuntimeSourceDoesNotDispatchWhatsChanged(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "fak", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(src, []byte(`case "whats-changed":`)) {
		t.Fatal("runtime fak still dispatches dev-owned whats-changed")
	}
}
func TestRuntimeSourceDoesNotDispatchBackend(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "backend":`) || strings.Contains(string(body), "cmdBackend(") {
		t.Fatal("runtime fak still dispatches the dev-only backend command")
	}
}

func TestRuntimeSourceDoesNotDispatchOrient(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "orient":`) || strings.Contains(string(body), "cmdOrient(") {
		t.Fatal("runtime fak still dispatches the dev-only orient command")
	}
}

func TestRuntimeSourceDoesNotDispatchIndex(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "index"`) || strings.Contains(string(body), `case "index", "devindex"`) || strings.Contains(string(body), "cmdIndex(") {
		t.Fatal("runtime fak still dispatches the dev-only index command")
	}
}

func TestRuntimeSourceDoesNotDispatchWiki(t *testing.T) {
	mainPath := filepath.Join(devindex.FindRoot("."), "cmd", "fak", "main.go")
	body, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `case "wiki":`) || strings.Contains(string(body), "cmdWiki(") {
		t.Fatal("runtime fak still dispatches the dev-only wiki command")
	}
}
