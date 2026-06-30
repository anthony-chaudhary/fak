package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestProductScorecardCLIJSONChartAndMarkdown(t *testing.T) {
	root := makeProductScorecardWorkspace(t)
	var out, errb bytes.Buffer
	code := runProductScorecard(&out, &errb, []string{"--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("json exit=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("parse json: %v\n%s", err, out.String())
	}
	if payload["schema"] != "fak-product-scorecard/1" || payload["ok"] != true {
		t.Fatalf("payload = %#v", payload)
	}

	out.Reset()
	errb.Reset()
	code = runProductScorecard(&out, &errb, []string{"--workspace", root, "--chart"})
	if code != 0 || !strings.Contains(out.String(), "verdict ladder") {
		t.Fatalf("chart exit=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}

	docDir := filepath.Join(t.TempDir(), "docs")
	out.Reset()
	errb.Reset()
	code = runProductScorecard(&out, &errb, []string{"--workspace", root, "--markdown-dir", docDir, "--stamp", "test"})
	if code != 0 {
		t.Fatalf("markdown exit=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	readme, err := os.ReadFile(filepath.Join(docDir, "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(readme), "Product scorecard") || !strings.Contains(string(readme), "Standing at a glance") {
		t.Fatalf("unexpected markdown:\n%s", string(readme))
	}
}

func TestProductScorecardCLICompareCriticalGaps(t *testing.T) {
	root := makeProductScorecardWorkspace(t)
	base := map[string]any{
		"corpus": map[string]any{
			"product_debt": 3, "honesty_defects": 2, "coverage_debt": 1,
			"score": 70.0, "durable_products": 0,
			"debt_by_group": map[string]any{"well-formed": 1, "honesty": 1, "usefulness": 0, "durability": 1},
		},
	}
	basePath := filepath.Join(t.TempDir(), "base.json")
	b, _ := json.Marshal(base)
	if err := os.WriteFile(basePath, b, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		args []string
		want string
	}{
		{[]string{"--workspace", root, "--compare", basePath}, "product-debt:"},
		{[]string{"--workspace", root, "--critical"}, "product critical backlog"},
		{[]string{"--workspace", root, "--gaps"}, "product coverage backlog"},
	} {
		var out, errb bytes.Buffer
		code := runProductScorecard(&out, &errb, tc.args)
		if code != 0 || !strings.Contains(out.String(), tc.want) {
			t.Fatalf("%v exit=%d stderr=%s stdout=%s", tc.args, code, errb.String(), out.String())
		}
	}
}

func makeProductScorecardWorkspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	mkdirs := []string{
		filepath.Join(root, "cmd", "fak"),
		filepath.Join(root, "docs"),
		filepath.Join(root, "internal", "adjudicator"),
		filepath.Join(root, "tools", "product_scorecard.data"),
	}
	for _, d := range mkdirs {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	mustWrite(t, filepath.Join(root, "CLAIMS.md"), "## Adjudication\n- [SHIPPED] real\n")
	mustWrite(t, filepath.Join(root, "docs", "cli-reference.md"), "# CLI\n\npreflight\n")
	writeProductScorecardJSONFile(t, filepath.Join(root, "tools", "product_scorecard.data", "_meta.json"), map[string]any{
		"meta":       map[string]any{"as_of": "2026-06-24", "fak_version": "t"},
		"categories": []map[string]any{{"id": "security", "name": "Security"}},
	})
	writeProductScorecardJSONFile(t, filepath.Join(root, "tools", "product_scorecard.data", "rows-security.json"), map[string]any{"rows": []map[string]any{{
		"id": "r1", "concept": "C", "category": "security", "surface": "product",
		"what_you_get": "a thing", "audience": "developer", "maturity": "shipped",
		"claims_section": "Adjudication", "claims_tag": "SHIPPED",
		"first_command":      "go run ./cmd/fak preflight --tool t --args \"{}\"",
		"first_command_verb": "preflight", "needs_gpu": false, "needs_key": false,
		"witness_path": "internal/adjudicator", "witness": "TestX", "entry_doc": "docs/cli-reference.md",
		"verdict": "durable-product", "gaps": []string{}, "durability_note": "n",
	}}})
	return root
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeProductScorecardJSONFile(t *testing.T, path string, v any) {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}
