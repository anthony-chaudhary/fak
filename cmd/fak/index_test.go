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
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	generationMd := "# Generation Contract\n\n" +
		"| Stream | Label | Milestone | Meaning |\n" +
		"|---|---|---|---|\n" +
		"| now | `gen/now` | `Generation G0 - Now / Immediate` | Current product work. |\n" +
		"| next | `gen/next` | `Generation G1 - Next Gen` | Near-term foundation that needs a gate or dogfood proof. |\n" +
		"| second-next | `gen/second-next` | `Generation G2 - Second Next Gen` | Architectural option needing simulation. |\n" +
		"| future | `gen/future` | `Generation G3 - Future` | Long-horizon research. |\n"
	if err := os.WriteFile(filepath.Join(root, "docs", "generation.md"), []byte(generationMd), 0o644); err != nil {
		t.Fatal(err)
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

func TestIndexGenerationJSON(t *testing.T) {
	root := writeIndexRepo(t)
	var out, errb bytes.Buffer
	if rc := runIndex(&out, &errb, []string{"generation", "--json", "--root", root, "next"}); rc != 0 {
		t.Fatalf("runIndex generation --json rc=%d, stderr=%s", rc, errb.String())
	}
	var generations []struct {
		Stream                 string   `json:"stream"`
		Label                  string   `json:"label"`
		Milestone              string   `json:"milestone"`
		IssueBodySignals       []string `json:"issue_body_signals"`
		PromotionEvidence      string   `json:"promotion_evidence"`
		DemotionEvidence       string   `json:"demotion_evidence"`
		InvalidatingAssumption string   `json:"invalidating_assumption"`
	}
	if err := json.Unmarshal(out.Bytes(), &generations); err != nil {
		t.Fatalf("generation --json is not valid JSON: %v\n%s", err, out.String())
	}
	if len(generations) != 1 || generations[0].Stream != "next" || generations[0].Label != "gen/next" {
		t.Fatalf("generation query = %+v, want only gen/next", generations)
	}
	if !strings.Contains(generations[0].PromotionEvidence, "dogfood") ||
		!strings.Contains(strings.Join(generations[0].IssueBodySignals, " "), "milestone") ||
		!strings.Contains(generations[0].InvalidatingAssumption, "stream label") {
		t.Fatalf("generation row missing evidence/body contract: %+v", generations[0])
	}
}
