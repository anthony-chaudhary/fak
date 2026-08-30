package devindex

import (
	"os"
	"path/filepath"
	"reflect"
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

func TestSearchDocsRanksMultiTermCoverageBeforeFieldWeight(t *testing.T) {
	c := &Catalog{Docs: []Doc{
		{Title: "Dogfood archive", Path: "docs/notes/dogfood.md"},
		{Title: "Developer tooling", Path: "docs/dev-tooling.md", Blurb: "documentation dogfood lookup"},
	}}

	hits := c.SearchDocs("documentation dogfood")
	if len(hits) != 2 {
		t.Fatalf("SearchDocs(documentation dogfood) = %+v, want two hits", hits)
	}
	if hits[0].Path != "docs/dev-tooling.md" {
		t.Fatalf("top multi-term hit = %q, want the two-token owner before the higher-weight one-token note", hits[0].Path)
	}

	// Multi-term ranking must not change the established single-term field weights.
	single := c.SearchDocs("dogfood")
	if len(single) != 2 || single[0].Path != "docs/notes/dogfood.md" {
		t.Fatalf("SearchDocs(dogfood) = %+v, want title/path-weighted note first", single)
	}
}

func TestSearchDocsRepositoryMultiTermCanaries(t *testing.T) {
	c, err := Load(filepath.Clean(filepath.Join("..", "..")))
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		query     string
		limit     int
		unrelated string
	}{
		{
			query:     "documentation self-index dogfood",
			limit:     10,
			unrelated: "docs/notes/DOS-FRESH-INSTALL-VALUE-DOGFOOD-LAPTOP-2026-06-25.md",
		},
		{
			query:     "shared-tree commit",
			limit:     5,
			unrelated: "docs/notes/SHARED-CLONE-INTEGRATION-PASS-2026-06-29.md",
		},
	}
	for _, tc := range tests {
		t.Run(tc.query, func(t *testing.T) {
			hits := c.SearchDocs(tc.query)
			ownerRank := docPathRank(hits, "docs/dev-tooling.md")
			if ownerRank < 0 || ownerRank >= tc.limit {
				t.Fatalf("docs/dev-tooling.md rank = %d for %q, want within top %d; top hits = %+v", ownerRank+1, tc.query, tc.limit, hits[:min(tc.limit, len(hits))])
			}
			unrelatedRank := docPathRank(hits, tc.unrelated)
			if unrelatedRank < 0 {
				t.Fatalf("unrelated canary %q missing for %q", tc.unrelated, tc.query)
			}
			if ownerRank >= unrelatedRank {
				t.Fatalf("docs/dev-tooling.md rank %d for %q, want ahead of unrelated note rank %d", ownerRank+1, tc.query, unrelatedRank+1)
			}
		})
	}
}

func docPathRank(docs []Doc, path string) int {
	for i, d := range docs {
		if d.Path == path {
			return i
		}
	}
	return -1
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

func TestClaimDocumentSetBindsDetailPagePackages(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs", "claims"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(
		"[lanes.trees]\n"+
			"docs = [\"docs/**\"]\n"+
			"gateway = [\"internal/gateway/**\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "CLAIMS.md"), []byte(
		"## Claims\n- [SHIPPED] [Gateway](docs/claims/gateway.md) [exposure: default-on]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "claims", "gateway.md"), []byte(
		"- [SHIPPED] served by `internal/gateway` with a focused package witness.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if got := c.ClaimsForLeaf("gateway"); len(got) != 1 {
		t.Fatalf("ClaimsForLeaf(gateway) = %v, want the linked detail-page claim", got)
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

// TestFuzzyFallback pins #3925: when exact substring scoring returns nothing, the
// Search* verbs fall back to a trigram near-miss pass and return the correct
// capability flagged Approx — instead of a false-ABSENT — while exact queries stay
// exact-only (Approx never set) and a genuinely absent query still returns empty.
func TestFuzzyFallback(t *testing.T) {
	c := &Catalog{
		Leaves: []Leaf{
			{Name: "arbitrate", Tree: "internal/arbitrate/", Desc: "stop two agents colliding on the same files"},
			{Name: "gateway", Tree: "internal/gateway/", Desc: "the upstream front door"},
		},
		Docs: []Doc{
			{Title: "gateway", Path: "docs/gateway.md", Blurb: "the upstream front door"},
		},
		Claims: []Claim{
			{Tag: "SHIPPED", Section: "kernel", Lanes: []string{"arbitrate"}, Text: "stop two agents colliding"},
		},
	}

	// Exact query is unchanged: it returns the exact hit, ranks it first, and never
	// sets Approx (the fallback must not engage when exact scoring found something).
	exact := c.SearchLeaves("arbitrate")
	if len(exact) != 1 || exact[0].Name != "arbitrate" {
		t.Fatalf("exact SearchLeaves(arbitrate) = %v, want [arbitrate]", names(exact))
	}
	if exact[0].Approx {
		t.Errorf("exact hit flagged Approx=true, want false")
	}

	// Near-miss SPELLING of a real leaf name that shares NO substring ("arbitration"
	// does not contain "arbitrate") returns empty today; the fuzzy fallback must now
	// surface the arbitrate leaf, flagged approximate.
	if got := leafContains(c.SearchLeaves("arbitration"), "arbitrate"); !got {
		t.Fatalf("near-miss SearchLeaves(arbitration) = %v, want it to include arbitrate", names(c.SearchLeaves("arbitration")))
	}
	near := c.SearchLeaves("arbitration")
	if near[0].Name != "arbitrate" || !near[0].Approx {
		t.Errorf("near-miss top hit = %+v, want arbitrate with Approx=true", near[0])
	}

	// SYNONYM on the description ("collision" vs the desc word "colliding") — no exact
	// substring — also falls back to the arbitrate leaf.
	if syn := c.SearchLeaves("collision"); len(syn) == 0 || syn[0].Name != "arbitrate" || !syn[0].Approx {
		t.Errorf("synonym SearchLeaves(collision) = %v, want approximate arbitrate", names(syn))
	}

	// A genuinely absent capability still returns nothing — the fallback must not
	// manufacture noise from an unrelated query.
	if got := c.SearchLeaves("quantumfluxcapacitor"); len(got) != 0 {
		t.Errorf("absent SearchLeaves = %v, want empty", names(got))
	}

	// Docs: a typo'd title ("gatway") near-matches the gateway doc, flagged Approx.
	docs := c.SearchDocs("gatway")
	if len(docs) != 1 || docs[0].Path != "docs/gateway.md" || !docs[0].Approx {
		t.Errorf("near-miss SearchDocs(gatway) = %+v, want approximate gateway doc", docs)
	}
	// And an exact doc query stays exact-only.
	if ex := c.SearchDocs("gateway"); len(ex) != 1 || ex[0].Approx {
		t.Errorf("exact SearchDocs(gateway) = %+v, want exact (Approx=false)", ex)
	}

	// Claims: near-miss lane "arbitration" (section/text hold no substring) falls back
	// to the arbitrate claim, flagged Approx.
	claims := c.SearchClaims("arbitration")
	if len(claims) != 1 || !claims[0].Approx {
		t.Errorf("near-miss SearchClaims(arbitration) = %+v, want one approximate claim", claims)
	}
}

// leafContains reports whether any leaf in ls has the given name.
func leafContains(ls []Leaf, name string) bool {
	for _, l := range ls {
		if l.Name == name {
			return true
		}
	}
	return false
}

func TestLoadComposesCuratedFrontDoorsWithProvenance(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[lanes.trees]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"INDEX.md":  "- [Index owner](docs/owner.md) — canonical blurb\n",
		"llms.txt":  "- [LLM owner](docs/llm.md) — llm blurb\n- [Duplicate](docs/owner.md)\n",
		"README.md": "- [Runtime guide](docs/runtime.md) — runtime blurb\n",
		"AGENTS.md": "- [Agent guide](docs/agent.md) — agent blurb\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	c, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	wantSources := map[string][]string{
		"docs/owner.md": {"INDEX.md", "llms.txt"}, "docs/llm.md": {"llms.txt"},
		"docs/runtime.md": {"README.md"}, "docs/agent.md": {"AGENTS.md"},
	}
	if len(c.Docs) != len(wantSources) {
		t.Fatalf("docs=%+v", c.Docs)
	}
	for _, d := range c.Docs {
		want, ok := wantSources[d.Path]
		if !ok {
			t.Errorf("unexpected doc %+v", d)
			continue
		}
		if !reflect.DeepEqual(d.Sources, want) {
			t.Errorf("%s sources=%v want %v", d.Path, d.Sources, want)
		}
	}
}

func TestLoadIndexesInlineFrontDoorLinks(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[lanes.trees]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	body := "Native execution: [`fak-native`](docs/native-inference-goal.md) is the product path.\n"
	if err := os.WriteFile(filepath.Join(root, "llms.txt"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	hits := c.SearchDocs("native inference product")
	if len(hits) == 0 || hits[0].Path != "docs/native-inference-goal.md" {
		t.Fatalf("hits=%+v", hits)
	}
	if !reflect.DeepEqual(hits[0].Sources, []string{"llms.txt"}) {
		t.Fatalf("sources=%v", hits[0].Sources)
	}
}

func TestSearchDocsCanonicalMultiTermTieBreakerPreservesSingleTermWeights(t *testing.T) {
	c := &Catalog{Docs: []Doc{
		{Title: "Shared validation incident", Path: "docs/notes/SHARED-INCIDENT.md", Blurb: "shared tree validation"},
		{Title: "Developer tooling", Path: "docs/dev-tooling.md", Blurb: "shared tree validation", Sources: []string{"INDEX.md", "AGENTS.md"}},
	}}
	multi := c.SearchDocs("shared tree validation")
	if len(multi) != 2 || multi[0].Path != "docs/dev-tooling.md" {
		t.Fatalf("multi=%+v", multi)
	}
	single := c.SearchDocs("shared")
	if len(single) != 2 || single[0].Path != "docs/notes/SHARED-INCIDENT.md" {
		t.Fatalf("single=%+v", single)
	}
}
