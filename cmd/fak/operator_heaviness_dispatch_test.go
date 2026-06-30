package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/heavinessscore"
)

func TestOperatorHeavinessGroupRunsHeavinessJSON(t *testing.T) {
	root := writeHeavinessDispatchWorkspace(t)

	var out, errb bytes.Buffer
	code := runOperatorHeavinessGroup(&out, &errb, []string{"heaviness", "--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("operator heaviness exit = %d, stderr=%s stdout=%s", code, errb.String(), out.String())
	}

	var payload struct {
		Schema string         `json:"schema"`
		OK     bool           `json:"ok"`
		Corpus map[string]any `json:"corpus"`
	}
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("operator heaviness JSON did not parse: %v\n%s", err, out.String())
	}
	if payload.Schema != heavinessscore.Schema || !payload.OK {
		t.Fatalf("operator heaviness payload = %+v", payload)
	}
	if got := payload.Corpus[heavinessscore.DebtKey]; got != float64(0) {
		t.Fatalf("heaviness debt = %v, want 0", got)
	}
}

func TestOperatorHeavinessGroupRejectsUnknownSubcommand(t *testing.T) {
	var out, errb bytes.Buffer
	code := runOperatorHeavinessGroup(&out, &errb, []string{"brief"})
	if code != 2 {
		t.Fatalf("unknown subcommand exit = %d, want 2; stdout=%s stderr=%s", code, out.String(), errb.String())
	}
}

func writeHeavinessDispatchWorkspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeHeavinessDispatchFile(t, root, "cmd/fak/main.go", `package main

func dispatch(name string) {
	switch name {
	case "complain":
	case "guard":
	case "operator":
	}
}
`)
	writeHeavinessDispatchFile(t, root, "cmd/fak/guard.go", `package main

func guardFlags(fs interface{
	Bool(string, bool, string) *bool
	String(string, string, string) *string
}) {
	fs.Bool("check", false, "")
	fs.String("policy", "", "")
}
`)
	writeHeavinessDispatchFile(t, root, "dos.toml", `[reasons.OFF_TRUNK]
summary = "stay on trunk"

[reasons.PUBLIC_LEAK]
summary = "scrub public copy"
`)
	writeHeavinessDispatchFile(t, root, "llms.txt", `- [Steerability scorecard](docs/STEERABILITY-SCORECARD.md): run fak steering.
- [Operator-heaviness scorecard](docs/OPERATOR-HEAVINESS.md): heaviness_pressure via fak operator heaviness.
`)
	return root
}

func writeHeavinessDispatchFile(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}
