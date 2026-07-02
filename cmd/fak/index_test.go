package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/hooks"
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

// writeFreshnessCLIRepo lays down a tree with a known dead INDEX.md link and a
// known orphaned dated note so `fak index freshness` is tested against fixed bytes.
func writeFreshnessCLIRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeFile := func(rel, body string) {
		fp := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(fp, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	writeFile("dos.toml", "[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n")
	writeFile("INDEX.md", "# INDEX\n- [Gone](docs/gone.md) — dead local link.\n")
	writeFile("docs/notes/2026-05-05-lonely.md", "# lonely note, unlisted\n")
	return root
}

func TestIndexFreshness(t *testing.T) {
	root := writeFreshnessCLIRepo(t)

	// Table mode names both the dead-doc-link and the orphan-note drift.
	var out, errb bytes.Buffer
	if rc := runIndex(&out, &errb, []string{"freshness", "--root", root}); rc != 0 {
		t.Fatalf("runIndex freshness rc=%d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "dead-doc-link") || !strings.Contains(got, "docs/gone.md") {
		t.Errorf("freshness table missing dead-doc-link finding, got:\n%s", got)
	}
	if !strings.Contains(got, "orphan-note") || !strings.Contains(got, "2026-05-05-lonely.md") {
		t.Errorf("freshness table missing orphan-note finding, got:\n%s", got)
	}

	// JSON mode round-trips the typed findings.
	out.Reset()
	errb.Reset()
	if rc := runIndex(&out, &errb, []string{"freshness", "--json", "--root", root}); rc != 0 {
		t.Fatalf("runIndex freshness --json rc=%d, stderr=%s", rc, errb.String())
	}
	var drift []struct {
		Kind    string `json:"kind"`
		Subject string `json:"subject"`
		Reason  string `json:"reason"`
	}
	if err := json.Unmarshal(out.Bytes(), &drift); err != nil {
		t.Fatalf("freshness --json is not valid JSON: %v\n%s", err, out.String())
	}
	kinds := map[string]bool{}
	for _, d := range drift {
		kinds[d.Kind] = true
	}
	if !kinds["dead-doc-link"] || !kinds["orphan-note"] {
		t.Errorf("freshness --json kinds = %v, want dead-doc-link + orphan-note", kinds)
	}
}

// TestIndexFreshnessClean: a tree whose index agrees with the sources prints the
// reassuring no-drift line and still exits 0 (a query, not a gate).
func TestIndexFreshnessClean(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("# r\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "INDEX.md"), []byte("# INDEX\n- [R](README.md) — resolves.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runIndex(&out, &errb, []string{"freshness", "--root", root}); rc != 0 {
		t.Fatalf("runIndex freshness rc=%d, stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "no drift") {
		t.Errorf("clean tree freshness should report no drift, got: %q", out.String())
	}
}

// TestIndexFreshnessUndeclaredLeafParity dogfoods the freshness detector against the
// AUTHORITATIVE lane-audit gate on the LIVE tree: internal/devindex.UndeclaredLeaves
// (the tier-1 reimplementation behind `fak index freshness`) must return the SAME leaf
// set as internal/hooks.UndeclaredLeaves. They diverged once — devindex counted only
// [lanes.trees] keys as declared and ignored the flat [lanes] name list, so a lane
// declared in [lanes] with no explicit tree glob was falsely flagged — and this pins
// the two together against future drift.
func TestIndexFreshnessUndeclaredLeafParity(t *testing.T) {
	root := devindex.FindRoot(".")
	if _, err := os.Stat(filepath.Join(root, "dos.toml")); err != nil {
		t.Skipf("no repo root (%v); skipping live parity dogfood", err)
	}
	cat, err := devindex.Load(root)
	if err != nil {
		t.Fatalf("devindex.Load: %v", err)
	}
	gaps, err := hooks.UndeclaredLeaves(root)
	if err != nil {
		t.Fatalf("hooks.UndeclaredLeaves: %v", err)
	}
	want := make([]string, 0, len(gaps))
	for _, g := range gaps {
		want = append(want, strings.ToLower(g.Leaf))
	}
	sort.Strings(want)
	got := cat.UndeclaredLeaves()
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("undeclared-leaf parity broken:\n devindex=%v\n hooks   =%v\n(did devindex's declared-set fall behind dos.toml's [lanes]?)", got, want)
	}
}
