package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeLaneAuditRepo lays a tmp root with a dos.toml declaring one lane and an undeclared Go leaf.
func writeLaneAuditRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"),
		[]byte("[lanes]\nconcurrent = [\"gateway\"]\n[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, d := range []string{"internal/gateway", "internal/orphan"} {
		if err := os.MkdirAll(filepath.Join(root, filepath.FromSlash(d)), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(d), "x.go"), []byte("package x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestRunHooksLaneAudit_reportExit0(t *testing.T) {
	root := writeLaneAuditRepo(t)
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"lane-audit", "--root", root})
	if code != 0 {
		t.Fatalf("want exit 0 for a report, got %d (err=%q)", code, errb.String())
	}
}

func TestRunHooksLaneAudit_gateExit1(t *testing.T) {
	root := writeLaneAuditRepo(t)
	var out, errb bytes.Buffer
	// One undeclared leaf (orphan) > gate 0 -> exit 1.
	code := runHooks(&out, &errb, []string{"lane-audit", "--root", root, "--gate", "0"})
	if code != 1 {
		t.Fatalf("want exit 1 over gate, got %d", code)
	}
}

func TestRunHooksLaneAudit_jsonCount(t *testing.T) {
	root := writeLaneAuditRepo(t)
	var out, errb bytes.Buffer
	code := runHooks(&out, &errb, []string{"lane-audit", "--root", root, "--json"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d", code)
	}
	var got struct {
		Count      int `json:"count"`
		Undeclared []struct {
			Leaf string `json:"leaf"`
			Base string `json:"base"`
		} `json:"undeclared"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out.String())
	}
	if got.Count != 1 || len(got.Undeclared) != 1 || got.Undeclared[0].Leaf != "orphan" {
		t.Errorf("want exactly the orphan leaf, got %+v", got)
	}
}
