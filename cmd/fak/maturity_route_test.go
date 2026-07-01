package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMaturityNextSubcommandAcceptsTrailingFlags(t *testing.T) {
	root := writeMaturityRouteWorkspace(t)

	var out, errb bytes.Buffer
	code := runMaturity(&out, &errb, []string{
		"next",
		"--workspace", root,
		"--json",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	var got struct {
		Schema  string `json:"schema"`
		Backlog []struct {
			Lane  string `json:"lane"`
			Title string `json:"title"`
		} `json:"backlog"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.Schema != "fak-maturity-scorecard/1" {
		t.Fatalf("schema = %q, want fak maturity scorecard", got.Schema)
	}
	if len(got.Backlog) == 0 || got.Backlog[0].Lane != "alpha" || !strings.Contains(got.Backlog[0].Title, "test alpha") {
		t.Fatalf("backlog[0] = %+v, want alpha test item", got.Backlog)
	}
}

func TestMaturityRouteDryRunPlansDedupedIssue(t *testing.T) {
	root := writeMaturityRouteWorkspace(t)
	existing := filepath.Join(t.TempDir(), "existing.json")
	if err := os.WriteFile(existing, []byte(`[{
		"number": 77,
		"state": "OPEN",
		"body": "<!-- fak-maturity-work-key: maturity/alpha/tested -->"
	}]`), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runMaturity(&out, &errb, []string{
		"route",
		"--workspace", root,
		"--limit", "1",
		"--existing-json", existing,
		"--json",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	var got struct {
		Schema  string `json:"schema"`
		Mode    string `json:"mode"`
		Planned []struct {
			Action string `json:"action"`
			Number *int   `json:"number"`
			Key    string `json:"key"`
			Lane   string `json:"lane"`
			Title  string `json:"title"`
		} `json:"planned"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got.Schema != "fak-maturity-issues/1" || got.Mode != "dry-run" {
		t.Fatalf("header = %q/%q, want maturity dry-run", got.Schema, got.Mode)
	}
	if len(got.Planned) != 1 {
		t.Fatalf("planned len = %d, want 1", len(got.Planned))
	}
	row := got.Planned[0]
	if row.Action != "update" || row.Number == nil || *row.Number != 77 {
		t.Fatalf("row = %+v, want update #77", row)
	}
	if row.Key != "maturity/alpha/tested" || row.Lane != "alpha" || row.Title != "maturity(alpha): add tests for the capability" {
		t.Fatalf("row identity = %+v", row)
	}
}

func TestMaturityRouteDryRunReportsPrivateBoundarySkips(t *testing.T) {
	root := t.TempDir()
	writeRouteFile(t, root, "dos.toml", `[lanes]
concurrent = ["dgxbridge", "alpha"]

[lanes.trees]
dgxbridge = ["internal/dgxbridge/**"]
alpha = ["internal/alpha/**"]
`)
	writeRouteFile(t, root, "internal/alpha/alpha.go", "package alpha\n\nfunc A() {}\n")
	writeRouteFile(t, root, "docs/cli-reference.md", "# verbs\n")

	var out, errb bytes.Buffer
	code := runMaturity(&out, &errb, []string{
		"route",
		"--workspace", root,
		"--limit", "1",
		"--json",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, errb.String())
	}
	var got struct {
		Planned []struct {
			Key  string `json:"key"`
			Lane string `json:"lane"`
		} `json:"planned"`
		Skipped []struct {
			Key    string `json:"key"`
			Lane   string `json:"lane"`
			Reason string `json:"reason"`
		} `json:"skipped"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if len(got.Skipped) != 1 || got.Skipped[0].Lane != "dgxbridge" {
		t.Fatalf("skipped = %+v, want dgxbridge", got.Skipped)
	}
	if len(got.Planned) != 1 || got.Planned[0].Lane != "alpha" || got.Planned[0].Key != "maturity/alpha/tested" {
		t.Fatalf("planned = %+v, want alpha tested", got.Planned)
	}
}

func writeMaturityRouteWorkspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeRouteFile(t, root, "dos.toml", `[lanes]
concurrent = ["alpha"]

[lanes.trees]
alpha = ["internal/alpha/**"]
`)
	writeRouteFile(t, root, "internal/alpha/alpha.go", "package alpha\n\nfunc A() {}\n")
	writeRouteFile(t, root, "docs/cli-reference.md", "# verbs\n")
	return root
}

func writeRouteFile(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
