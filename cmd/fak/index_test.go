package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeIndexRepo lays down a minimal dos.toml + CLAIMS.md so the index CLI is
// tested against known bytes via --root, not the live tree.
func writeIndexRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dosToml := "[lanes.trees]\n" +
		"gateway = [\"internal/gateway/**\"]\n" +
		"session = [\"internal/session/**\"]\n"
	claimsMd := "# CLAIMS.md\n" +
		"## Gateway\n" +
		"- [SHIPPED] internal/gateway speaks OpenAI at the front door.\n" +
		"- [STUB] internal/gateway streaming backpressure is deferred.\n" +
		"## Session\n" +
		"- [SIMULATED] internal/session cost ring uses stand-in data.\n"
	for name, body := range map[string]string{"dos.toml": dosToml, "CLAIMS.md": claimsMd} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestIndexLeafShowsStatusBadge(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runIndex(&out, &errb, []string{"leaf", "--root", root, "gateway"}); rc != 0 {
		t.Fatalf("runIndex leaf rc=%d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "1 shipped") || !strings.Contains(got, "1 stub") {
		t.Errorf("leaf row missing status rollup, got:\n%s", got)
	}
}

func TestIndexClaimsSearch(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runIndex(&out, &errb, []string{"claims", "--root", root, "gateway"}); rc != 0 {
		t.Fatalf("runIndex claims rc=%d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "SHIPPED") || !strings.Contains(got, "gateway") {
		t.Errorf("claims search missing the gateway SHIPPED claim, got:\n%s", got)
	}
}

func TestIndexClaimsNeedsQuery(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runIndex(&out, &errb, []string{"claims", "--root", root}); rc != 2 {
		t.Errorf("claims with no query rc=%d, want 2 (usage error)", rc)
	}
}

func TestIndexClaimsJSON(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runIndex(&out, &errb, []string{"claims", "--json", "--root", root, "session"}); rc != 0 {
		t.Fatalf("runIndex claims --json rc=%d, stderr=%s", rc, errb.String())
	}
	var claims []struct {
		Tag   string   `json:"tag"`
		Lanes []string `json:"lanes"`
		Text  string   `json:"text"`
	}
	if err := json.Unmarshal(out.Bytes(), &claims); err != nil {
		t.Fatalf("claims --json is not valid JSON: %v\n%s", err, out.String())
	}
	if len(claims) != 1 || claims[0].Tag != "SIMULATED" {
		t.Errorf("session claims = %+v, want exactly one SIMULATED", claims)
	}
}
