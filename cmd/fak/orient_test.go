package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeOrientCLIRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dosToml := `[lanes.trees]
gateway = ["internal/gateway/**"]
cmd     = ["cmd/**"]
`
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dosToml), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, dir := range []string{
		filepath.Join(root, "internal", "gateway"),
		filepath.Join(root, "internal", "architest"),
		filepath.Join(root, "cmd", "fak"),
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "gateway", "gateway.go"), []byte("package gateway\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	architest := `package architest

var tier = map[string]int{
	"gateway": 4,
}
`
	if err := os.WriteFile(filepath.Join(root, "internal", "architest", "architest_test.go"), []byte(architest), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestRunOrientPrintsPathConventions(t *testing.T) {
	root := writeOrientCLIRepo(t)
	var out, errb bytes.Buffer
	rc := runOrient(&out, &errb, []string{"--root", root, "--leases=false", "--paths", "internal/gateway/server.go"})
	if rc != 0 {
		t.Fatalf("runOrient rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{"internal/gateway/server.go", "lane=gateway", "stamp=(fak gateway)", "test=go test ./internal/gateway", "tier=4/integrator", "lease=none", "lane_tree=internal/gateway/**"} {
		if !strings.Contains(got, want) {
			t.Fatalf("orient output missing %q:\n%s", want, got)
		}
	}
}

func TestRunOrientJSON(t *testing.T) {
	root := writeOrientCLIRepo(t)
	var out, errb bytes.Buffer
	rc := runOrient(&out, &errb, []string{"--root", root, "--leases=false", "--json", "--paths", "cmd/fak/*.go"})
	if rc != 0 {
		t.Fatalf("runOrient --json rc=%d stderr=%s", rc, errb.String())
	}
	var rows []struct {
		Path       string `json:"path"`
		Lane       string `json:"lane"`
		Stamp      string `json:"stamp"`
		TestTarget string `json:"owning_test_target"`
	}
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil {
		t.Fatalf("orient JSON invalid: %v\n%s", err, out.String())
	}
	if len(rows) != 1 || rows[0].Lane != "cmd" || rows[0].Stamp != "(fak cmd)" || rows[0].TestTarget != "go test ./cmd/fak" {
		t.Fatalf("orient JSON = %+v, want cmd row with stamp and owning test", rows)
	}
}

func TestRunOrientNeedsPaths(t *testing.T) {
	root := writeOrientCLIRepo(t)
	var out, errb bytes.Buffer
	if rc := runOrient(&out, &errb, []string{"--root", root}); rc != 2 {
		t.Fatalf("runOrient without paths rc=%d, want 2", rc)
	}
	if !strings.Contains(errb.String(), "needs --paths") {
		t.Fatalf("usage error = %q", errb.String())
	}
}
