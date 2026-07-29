package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// docFixture is a small markdown doc shaped like the real hazard: a fenced shell
// block whose comment sits at column 0, and a nested subsection that carries most
// of the parent's weight.
const docFixture = "lede before any heading\n" +
	"# Top\n" +
	"intro\n" +
	"## Alpha\n" +
	"```bash\n" +
	"# this is a shell comment, NOT a heading\n" +
	"## neither is this\n" +
	"```\n" +
	"alpha body\n" +
	"### Alpha detail\n" +
	"the bulk of alpha lives here and it is deliberately long enough to dominate\n" +
	"## Beta\n" +
	"beta body\n"

// TestDocFootprintPartitionsFile proves the inventory is a faithful partition, the
// property that makes every percentage in docs/context-budget/agents-md-floor.md
// mean something: the preamble plus every section's OwnBytes reconstruct the file
// exactly, and a parent's INCLUSIVE Bytes equal its own bytes plus its
// descendants'. A per-section table that does not partition is decoration, not a
// measurement (#5445, epic #3229).
func TestDocFootprintPartitionsFile(t *testing.T) {
	fp := computeDocFootprint("fixture.md", docFixture)

	if fp.Bytes != len(docFixture) {
		t.Fatalf("Bytes=%d, want %d", fp.Bytes, len(docFixture))
	}
	sum := fp.PreambleBytes
	for _, s := range fp.Sections {
		sum += s.OwnBytes
	}
	if sum != fp.Bytes {
		t.Fatalf("preamble+own bytes = %d, want file bytes %d (not a partition)", sum, fp.Bytes)
	}

	// "# Top" is level 1 and every other heading nests under it, so its inclusive
	// extent must run to the end of the file.
	top := fp.Sections[0]
	if top.Level != 1 || top.Title != "Top" {
		t.Fatalf("first section = L%d %q, want L1 Top", top.Level, top.Title)
	}
	if got, want := top.Bytes, fp.Bytes-fp.PreambleBytes; got != want {
		t.Fatalf("L1 inclusive bytes=%d, want %d (everything after the preamble)", got, want)
	}

	// Alpha's inclusive extent must swallow "### Alpha detail" and stop at "## Beta".
	alpha := docSectionByTitle(t, fp, "Alpha")
	detail := docSectionByTitle(t, fp, "Alpha detail")
	if alpha.Bytes != alpha.OwnBytes+detail.OwnBytes {
		t.Fatalf("Alpha inclusive=%d, want own %d + child own %d", alpha.Bytes, alpha.OwnBytes, detail.OwnBytes)
	}
	if alpha.Bytes <= alpha.OwnBytes {
		t.Fatalf("Alpha inclusive %d must exceed its own %d (the child is non-empty)", alpha.Bytes, alpha.OwnBytes)
	}
}

// TestDocFootprintIgnoresHeadingsInsideFences is the mutation witness for the fence
// tracker. AGENTS.md is command-dense; a naive `^#+ ` scan invents sections out of
// shell comments at column 0 and every byte percentage below them shifts. This
// fixture has two such lines inside one ``` block, so an implementation that drops
// the fence state reports 6 sections instead of 4 and fails here.
func TestDocFootprintIgnoresHeadingsInsideFences(t *testing.T) {
	fp := computeDocFootprint("fixture.md", docFixture)

	want := []string{"Top", "Alpha", "Alpha detail", "Beta"}
	got := make([]string, 0, len(fp.Sections))
	for _, s := range fp.Sections {
		got = append(got, s.Title)
	}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("sections = %v, want %v (a fenced shell comment was read as a heading)", got, want)
	}
	for _, s := range fp.Sections {
		if strings.Contains(s.Title, "shell comment") || strings.Contains(s.Title, "neither") {
			t.Fatalf("fenced line became section %q", s.Title)
		}
	}
}

// TestDocFootprintTokensMatchEstimator proves the token column is the house
// estimator's own answer, not a private divide-by-four that can drift. Every doc
// number in the epic's scorecards has to come out of EstimateAnthropicTokens or the
// resident floor (#3230) and the instruction-pulled floor (#5445) are not
// comparable quantities.
func TestDocFootprintTokensMatchEstimator(t *testing.T) {
	fp := computeDocFootprint("fixture.md", docFixture)
	want := agent.EstimateAnthropicTokens(&agent.AnthropicMessagesRequest{System: docFixture})
	if fp.Tokens != want {
		t.Fatalf("Tokens=%d, EstimateAnthropicTokens=%d (the doc verb grew its own estimator)", fp.Tokens, want)
	}
	for _, s := range fp.Sections {
		if s.Tokens < s.OwnTokens {
			t.Fatalf("section %q: inclusive tokens %d < own tokens %d", s.Title, s.Tokens, s.OwnTokens)
		}
	}
}

// TestDocFootprintRanksHeaviestFirst pins the ordering the paging-out lever reads:
// a cut line is chosen off this ranking, so an unstable or mis-sorted list would
// send a trim at the wrong section.
func TestDocFootprintRanksHeaviestFirst(t *testing.T) {
	ranked := docSectionsByWeight(computeDocFootprint("fixture.md", docFixture))
	for i := 1; i < len(ranked); i++ {
		if ranked[i].Bytes > ranked[i-1].Bytes {
			t.Fatalf("not heaviest-first at %d: %d > %d", i, ranked[i].Bytes, ranked[i-1].Bytes)
		}
	}
}

// TestDocFootprintVerbJSON witnesses the CLI contract end to end against the REAL
// AGENTS.md — the file docs/context-budget/agents-md-floor.md prices. It asserts
// the shape and the partition, deliberately NOT a pinned byte total: #5445 puts a
// ratchet on AGENTS.md explicitly out of scope until after the trim, because
// pinning a ceiling at today's size banks the bloat (the FLOOR_BUDGET_STALE lesson
// in internal/mcpfootprint/floorgate.go).
func TestDocFootprintVerbJSON(t *testing.T) {
	path := filepath.Join(repoRoot(), "AGENTS.md")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("AGENTS.md not readable from repoRoot(): %v", err)
	}

	var out bytes.Buffer
	if code := runMCPFootprint(&out, io.Discard, []string{"--doc", "AGENTS.md", "--json"}); code != 0 {
		t.Fatalf("fak footprint --doc AGENTS.md --json exit=%d", code)
	}
	m := decodeJSON(t, out.Bytes())

	if m["schema"] != "fak-doc-footprint/1" {
		t.Fatalf("schema=%v, want fak-doc-footprint/1", m["schema"])
	}
	if m["kind"] != "instruction-pulled" {
		t.Fatalf("kind=%v, want instruction-pulled", m["kind"])
	}
	if m["provenance"] != agent.FootprintProvenance {
		t.Fatalf("provenance=%v, want %v", m["provenance"], agent.FootprintProvenance)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read AGENTS.md: %v", err)
	}
	if got, want := int(m["bytes"].(float64)), len(raw); got != want {
		t.Fatalf("bytes=%d, want %d (the verb read a different file)", got, want)
	}

	sections := m["sections"].([]any)
	if len(sections) != int(m["section_count"].(float64)) {
		t.Fatalf("shown %d sections but section_count=%v (default --top 0 shows all)", len(sections), m["section_count"])
	}
	if len(sections) < 5 {
		t.Fatalf("AGENTS.md priced as only %d sections; the heading scan is broken", len(sections))
	}

	sum := m["preamble_bytes"].(float64)
	for _, e := range sections {
		sum += e.(map[string]any)["own_bytes"].(float64)
	}
	if int(sum) != len(raw) {
		t.Fatalf("preamble+own bytes = %d, want %d (the real-file inventory does not partition)", int(sum), len(raw))
	}
}

func docSectionByTitle(t *testing.T, fp docFootprint, title string) docSectionEntry {
	t.Helper()
	for _, s := range fp.Sections {
		if s.Title == title {
			return s
		}
	}
	t.Fatalf("section %q not found", title)
	return docSectionEntry{}
}
