package devindex

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeSyntheticRepo lays down a tiny repo (dos.toml + INDEX.md) under a temp dir so
// the parser is tested against known, controlled bytes rather than the live tree.
func writeSyntheticRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dosToml := `[lanes]
concurrent = ["a", "b"]

[lanes.trees]
gateway = ["internal/gateway/**"]
session = ["internal/session/**"] # per-session DRIVE state + cost ring
cmd     = ["cmd/**"]
version = ["VERSION"]
docs    = ["docs/**", "README.md", "INDEX.md", "llms.txt", "llms-full.txt", "llms-updates.txt"]
dos     = ["dos.toml", ".gitignore"]
tools   = ["tools/**", "scripts/**"]
# new-leaf:tree -- generated leaf trees inserted above this line

[other]
ignored = ["internal/ignored/**"]
`
	indexMd := "# INDEX\n" +
		"- [README](README.md) — what fak is, in one read.\n" +
		"- [`fleet` console](tools/FLEET.md) — watch the agent fleet on a host.\n" +
		"- [Gateway](docs/gateway.md) — the OpenAI-compatible front door.\n" +
		"this is prose, not a link, and must be skipped\n" +
		"- [Issue tracker](https://github.com/x/y/issues) — the live open-issue count.\n"
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dosToml), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "INDEX.md"), []byte(indexMd), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	generationMd := "# Generation Contract\n\n" +
		"| Stream | Label | Milestone | Meaning |\n" +
		"|---|---|---|---|\n" +
		"| now | `gen/now` | `Generation G0 - Now / Immediate` | Current product work with a clear witness. |\n" +
		"| next | `gen/next` | `Generation G1 - Next Gen` | Near-term foundation that needs a gate or dogfood proof. |\n" +
		"| second-next | `gen/second-next` | `Generation G2 - Second Next Gen` | Architectural option needing simulation or compatibility policy. |\n" +
		"| future | `gen/future` | `Generation G3 - Future` | Long-horizon research or option value. |\n"
	if err := os.WriteFile(filepath.Join(root, "docs", "generation.md"), []byte(generationMd), 0o644); err != nil {
		t.Fatal(err)
	}
	// A synthetic CLAIMS.md exercising the claim/status join (C2 #1289): the legend
	// line writes its tag in backticks and MUST be excluded; real claims bind to a
	// lane via their internal/<pkg> reference; a product claim names no package and
	// must stay searchable with no rollup.
	claimsMd := "# CLAIMS.md — synthetic honesty ledger\n" +
		"- `[SHIPPED]` — legend line; backticked tag, must be EXCLUDED.\n" +
		"\n## Gateway\n" +
		"- [SHIPPED] The `internal/gateway` front door speaks OpenAI. Witness: gateway tests.\n" +
		"- [SHIPPED] Admission control in internal/gateway sheds at 429.\n" +
		"- [STUB] internal/gateway streaming backpressure is deferred.\n" +
		"\n## Session\n" +
		"- [SIMULATED] internal/session cost ring uses labeled stand-in data.\n" +
		"\n## Product\n" +
		"- [SHIPPED] One statically-linked Go binary runs the loop (no package ref).\n"
	if err := os.WriteFile(filepath.Join(root, "CLAIMS.md"), []byte(claimsMd), 0o644); err != nil {
		t.Fatal(err)
	}
	// Make internal/gateway real so Exists is exercised in both polarities.
	if err := os.MkdirAll(filepath.Join(root, "internal", "gateway"), 0o755); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestParseLanes(t *testing.T) {
	c, err := Load(writeSyntheticRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	// Only [lanes.trees] entries — never the [other] section's "ignored".
	if _, ok := c.LeafByName("ignored"); ok {
		t.Error("leaf from a non-[lanes.trees] section leaked into the catalog")
	}
	wantNames := []string{"cmd", "docs", "dos", "gateway", "session", "tools", "version"}
	if len(c.Leaves) != len(wantNames) {
		t.Fatalf("got %d leaves %v, want %d %v", len(c.Leaves), names(c.Leaves), len(wantNames), wantNames)
	}
	for i, n := range wantNames {
		if c.Leaves[i].Name != n { // Load sorts by name
			t.Errorf("leaf[%d] = %q, want %q (sorted)", i, c.Leaves[i].Name, n)
		}
	}

	sess, ok := c.LeafByName("session")
	if !ok {
		t.Fatal("session leaf missing")
	}
	if sess.Desc != "per-session DRIVE state + cost ring" {
		t.Errorf("session desc = %q, want the inline dos.toml comment", sess.Desc)
	}
	if sess.Dir != "internal/session" {
		t.Errorf("session dir = %q, want internal/session", sess.Dir)
	}
	if sess.Exists {
		t.Error("session dir should NOT exist in the synthetic repo")
	}

	gw, _ := c.LeafByName("gateway")
	if !gw.Exists {
		t.Error("gateway dir was created and should report Exists=true")
	}
}

func TestLaneForPath(t *testing.T) {
	c, err := Load(writeSyntheticRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	cases := []struct{ path, want string }{
		{"internal/gateway/foo.go", "gateway"},       // subtree prefix
		{"internal\\gateway\\foo.go", "gateway"},     // windows separators
		{"./internal/session/x.go", "session"},       // leading ./ trimmed
		{"cmd/fak/index.go", "cmd"},                  // cmd/** tree
		{"VERSION", "version"},                       // exact-file entry
		{"dos.toml", "dos"},                          // exact-file entry
		{".gitignore", "dos"},                        // exact-file entry
		{"scripts/gcp-fleet-janitor.sh", "tools"},    // subtree prefix
		{"internal/unknownleaf/x.go", "unknownleaf"}, // dir convention fallback
		{"docs/notes/x.md", "docs"},                  // top-level lane dir
		{"README.md", "docs"},                        // exact root-doc entry
		{"llms-updates.txt", "docs"},                 // exact generated root-doc entry
	}
	for _, tc := range cases {
		if got := c.LaneForPath(tc.path); got != tc.want {
			t.Errorf("LaneForPath(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}

func TestSuggestStamp(t *testing.T) {
	c, _ := Load(writeSyntheticRepo(t))
	if got := c.SuggestStamp("internal/session/x.go"); got != "(fak session)" {
		t.Errorf("SuggestStamp = %q, want (fak session)", got)
	}
	if got := c.SuggestStamp("dos.toml"); got != "(fak dos)" {
		t.Errorf("SuggestStamp(dos.toml) = %q, want (fak dos)", got)
	}
	if got := c.SuggestStamp("README.md"); got != "(fak docs)" {
		t.Errorf("SuggestStamp(README.md) = %q, want (fak docs)", got)
	}
}

func TestParseDocs(t *testing.T) {
	c, _ := Load(writeSyntheticRepo(t))
	if len(c.Docs) != 4 {
		t.Fatalf("got %d docs, want 4: %+v", len(c.Docs), c.Docs)
	}
	var fleet *Doc
	for i := range c.Docs {
		if c.Docs[i].Path == "tools/FLEET.md" {
			fleet = &c.Docs[i]
		}
	}
	if fleet == nil {
		t.Fatal("FLEET.md doc missing")
	}
	if fleet.Title != "fleet console" { // surrounding backticks stripped
		t.Errorf("title = %q, want backtick-stripped 'fleet console'", fleet.Title)
	}
	if fleet.Blurb != "watch the agent fleet on a host." {
		t.Errorf("blurb = %q", fleet.Blurb)
	}
}

func TestSearchDocsRanking(t *testing.T) {
	c, _ := Load(writeSyntheticRepo(t))
	hits := c.SearchDocs("gateway front door")
	if len(hits) == 0 {
		t.Fatal("expected a gateway doc hit")
	}
	if hits[0].Path != "docs/gateway.md" {
		t.Errorf("top hit = %q, want docs/gateway.md (title+blurb match)", hits[0].Path)
	}
	if got := c.SearchDocs(""); got != nil {
		t.Errorf("empty query should return nil, got %v", got)
	}
}

func TestGenerationIndexSearch(t *testing.T) {
	c, _ := Load(writeSyntheticRepo(t))
	if len(c.Generations) != 4 {
		t.Fatalf("generations = %d, want 4: %+v", len(c.Generations), c.Generations)
	}
	next, ok := c.GenerationByStream("gen/next")
	if !ok {
		t.Fatal("gen/next row missing")
	}
	if next.Label != "gen/next" || next.Milestone != "Generation G1 - Next Gen" {
		t.Fatalf("next row = %+v, want label and milestone from docs/generation.md", next)
	}
	signals := strings.Join(next.IssueBodySignals, " ")
	if !strings.Contains(signals, "Generation stream") || !strings.Contains(signals, "promotion evidence") {
		t.Fatalf("next issue-body signals = %q, want generation stream + promotion evidence", signals)
	}
	if !strings.Contains(next.PromotionEvidence, "dogfood") || !strings.Contains(next.DemotionEvidence, "stale") {
		t.Fatalf("next evidence rules = promote %q demote %q", next.PromotionEvidence, next.DemotionEvidence)
	}
	hits := c.SearchGenerations("gen/next gate")
	if len(hits) == 0 || hits[0].Stream != "next" {
		t.Fatalf("SearchGenerations(gen/next gate) = %+v, want next first", hits)
	}
	if got := c.SearchGenerations(""); len(got) != 4 || got[0].Stream != "now" || got[3].Stream != "future" {
		t.Fatalf("empty generation search = %+v, want all in now->future order", got)
	}
}

func TestSearchLeaves(t *testing.T) {
	c, _ := Load(writeSyntheticRepo(t))
	// Description token match.
	hits := c.SearchLeaves("DRIVE")
	if len(hits) != 1 || hits[0].Name != "session" {
		t.Errorf("SearchLeaves(DRIVE) = %v, want [session]", names(hits))
	}
	// Name match ranks above all; empty query returns the full set in order.
	if got := c.SearchLeaves(""); len(got) != len(c.Leaves) {
		t.Errorf("empty query returned %d leaves, want all %d", len(got), len(c.Leaves))
	}
	// A token nothing matches yields no hits.
	if got := c.SearchLeaves("nonexistenttoken"); len(got) != 0 {
		t.Errorf("expected no hits, got %v", names(got))
	}
}

func TestParseClaimsAndRollup(t *testing.T) {
	c, err := Load(writeSyntheticRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	// 5 real claims; the backticked legend line is excluded by claimTagRE.
	if len(c.Claims) != 5 {
		t.Fatalf("got %d claims, want 5 (legend excluded): %+v", len(c.Claims), c.Claims)
	}
	for _, cl := range c.Claims {
		if strings.HasPrefix(cl.Text, "—") || strings.Contains(cl.Text, "legend line") {
			t.Errorf("legend line leaked into the ledger as a claim: %q", cl.Text)
		}
	}

	gw, ok := c.LeafByName("gateway")
	if !ok {
		t.Fatal("gateway leaf missing")
	}
	if gw.Status.Shipped != 2 || gw.Status.Stub != 1 || gw.Status.Simulated != 0 {
		t.Errorf("gateway status = %+v, want {Shipped:2 Simulated:0 Stub:1}", gw.Status)
	}
	if gw.Status.Total() != 3 {
		t.Errorf("gateway Total() = %d, want 3", gw.Status.Total())
	}

	sess, _ := c.LeafByName("session")
	if sess.Status.Simulated != 1 || sess.Status.Total() != 1 {
		t.Errorf("session status = %+v, want exactly 1 SIMULATED", sess.Status)
	}

	// The product claim names no package -> it stays in the ledger but binds to no
	// lane (no rollup), so it never inflates a leaf's status.
	var product *Claim
	for i := range c.Claims {
		if strings.Contains(c.Claims[i].Text, "statically-linked") {
			product = &c.Claims[i]
		}
	}
	if product == nil {
		t.Fatal("product claim missing from the ledger")
	}
	if len(product.Lanes) != 0 {
		t.Errorf("product claim bound to %v, want no lane", product.Lanes)
	}
}

func TestSearchAndClaimsForLeaf(t *testing.T) {
	c, _ := Load(writeSyntheticRepo(t))

	// An empty query is a usage error the caller surfaces -> nil.
	if got := c.SearchClaims(""); got != nil {
		t.Errorf("empty query should return nil, got %v", got)
	}
	// A lane token outranks a bare text hit (lane weight 3 > text weight 1).
	hits := c.SearchClaims("gateway")
	if len(hits) == 0 {
		t.Fatal("expected gateway claim hits")
	}
	for _, h := range hits[:1] {
		hasLane := false
		for _, l := range h.Lanes {
			if l == "gateway" {
				hasLane = true
			}
		}
		if !hasLane {
			t.Errorf("top gateway hit does not bind to the gateway lane: %+v", h)
		}
	}
	// ClaimsForLeaf is the strict bound set: exactly the 3 gateway-bound claims.
	if got := c.ClaimsForLeaf("GateWay"); len(got) != 3 { // case-insensitive
		t.Errorf("ClaimsForLeaf(gateway) = %d claims, want 3", len(got))
	}
	if got := c.ClaimsForLeaf("nonexistent"); len(got) != 0 {
		t.Errorf("ClaimsForLeaf(nonexistent) = %d, want 0", len(got))
	}
}

func TestLoadMissingClaimsDegrades(t *testing.T) {
	// A repo with dos.toml but no CLAIMS.md loads cleanly with an empty rollup.
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"),
		[]byte("[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load without CLAIMS.md should not error: %v", err)
	}
	if len(c.Claims) != 0 {
		t.Errorf("no CLAIMS.md should mean no claims, got %d", len(c.Claims))
	}
	if gw, _ := c.LeafByName("gateway"); gw.Status.Total() != 0 {
		t.Errorf("missing ledger should leave an empty rollup, got %+v", gw.Status)
	}
}

func TestLoadMissingDosToml(t *testing.T) {
	if _, err := Load(t.TempDir()); err == nil {
		t.Error("Load with no dos.toml should error (no taxonomy to serve)")
	}
}

// TestRealRepoDogfood loads the live repo via FindRoot and asserts the catalog
// reflects reality — including this very leaf — so the index can't silently lose
// touch with the tree it indexes.
func TestRealRepoDogfood(t *testing.T) {
	root := FindRoot(".")
	c, err := Load(root)
	if err != nil {
		t.Skipf("no repo root found from test cwd (%v); skipping live dogfood", err)
	}
	for _, want := range []string{"devindex", "gateway", "cmd", "docs", "dos"} {
		l, ok := c.LeafByName(want)
		if !ok {
			t.Errorf("live catalog is missing the %q lane", want)
			continue
		}
		if want == "devindex" && !l.Exists {
			t.Errorf("devindex leaf should exist on disk (it is this package)")
		}
	}
	if c.LaneForPath("internal/gateway/gateway.go") != "gateway" {
		t.Error("live LaneForPath disagrees with the gateway tree")
	}
	// The claim/status join must bind to the live CLAIMS.md, not silently no-op:
	// the gateway leaf carries shipped claims in the real ledger.
	if len(c.Claims) == 0 {
		t.Error("live catalog parsed no CLAIMS.md claims (the C2 join is dead)")
	}
	if gw, _ := c.LeafByName("gateway"); gw.Status.Shipped == 0 {
		t.Error("live gateway leaf has no SHIPPED claims bound (join broken or regex drift)")
	}
}

func names(ls []Leaf) []string {
	out := make([]string, len(ls))
	for i, l := range ls {
		out[i] = l.Name
	}
	return out
}
