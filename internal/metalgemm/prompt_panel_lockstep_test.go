package metalgemm

import (
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// TestPromptPanelMaxTokensLockstep pins the Go/MSL panel-ceiling lockstep that issue
// #13041's wider-panel path depends on. Widening the admitted prompt panel required
// raising three independently-declared constants together:
//
//   - PromptPanelMaxTokens (graph.go, Go-side admission in PromptPanelWitnessed)
//   - QG_MAX_ROWS         (qwen35_graph.m, native qg_ordered_rows row guard)
//   - MG_GDN_GRAPH_MAX_TOKENS (gdn.m, fused GDN encoder tokens guard)
//
// MSL cannot import a Go constant, so the three are hand-mirrored. A mismatch does not
// corrupt state -- the native guard returns NULL and the Go caller declines fail-open --
// but it silently *disables* the wider panel, turning the #13041 collapse back into the
// serial 32-token walk with no test failure. This source witness runs on every host and
// makes drift loud: change one ceiling and the others must move with it.
//
// The portable (non-darwin) graph_stub.go also declares PromptPanelMaxTokens for callers
// that size panels without linking the Metal graph; it must stay byte-identical to
// graph.go's value, so this witness pins graph.go == graph_stub.go too.
func TestPromptPanelMaxTokensLockstep(t *testing.T) {
	goSource, err := os.ReadFile("graph.go")
	if err != nil {
		t.Fatal(err)
	}
	stubSource, err := os.ReadFile("graph_stub.go")
	if err != nil {
		t.Fatal(err)
	}
	graphSource, err := os.ReadFile("qwen35_graph.m")
	if err != nil {
		t.Fatal(err)
	}
	gdnSource, err := os.ReadFile("gdn.m")
	if err != nil {
		t.Fatal(err)
	}

	// Normalize CRLF -> LF before matching. Go's regexp `$` matches before `\n` but not
	// before `\r\n`, so a CRLF checkout (Windows) would otherwise fail every `(?m)^...$`
	// anchor here even though the pinned line is present. This keeps the witness portable
	// across line-ending policies without weakening the pinned values.
	goMax, ok := firstIntMatch(normalizeLF(goSource), `(?m)^const PromptPanelMaxTokens = (\d+)$`)
	if !ok {
		t.Fatal("graph.go: could not find `const PromptPanelMaxTokens = <n>`")
	}
	stubMax, ok := firstIntMatch(normalizeLF(stubSource), `(?m)^const PromptPanelMaxTokens = (\d+)$`)
	if !ok {
		t.Fatal("graph_stub.go: could not find `const PromptPanelMaxTokens = <n>`")
	}
	graphMax, ok := firstIntMatch(normalizeLF(graphSource), `(?m)^#define QG_MAX_ROWS (\d+)$`)
	if !ok {
		t.Fatal("qwen35_graph.m: could not find `#define QG_MAX_ROWS <n>`")
	}
	gdnMax, ok := firstIntMatch(normalizeLF(gdnSource), `(?m)^enum \{ MG_GDN_GRAPH_MAX_TOKENS = (\d+) \};$`)
	if !ok {
		t.Fatal("gdn.m: could not find `enum { MG_GDN_GRAPH_MAX_TOKENS = <n> };`")
	}

	if stubMax != goMax {
		t.Fatalf("panel-ceiling drift: PromptPanelMaxTokens (graph.go)=%d but the portable graph_stub.go=%d; the two Go declarations must stay byte-identical", goMax, stubMax)
	}
	if graphMax != goMax {
		t.Fatalf("panel-ceiling drift: PromptPanelMaxTokens (graph.go)=%d but QG_MAX_ROWS (qwen35_graph.m)=%d; the native qg_ordered_rows guard will decline every wider panel fail-open", goMax, graphMax)
	}
	if gdnMax != goMax {
		t.Fatalf("panel-ceiling drift: PromptPanelMaxTokens (graph.go)=%d but MG_GDN_GRAPH_MAX_TOKENS (gdn.m)=%d; the fused GDN encoder will decline every wider panel fail-open", goMax, gdnMax)
	}
	if goMax < 32 {
		t.Fatalf("PromptPanelMaxTokens=%d is narrower than the historical 32-token panel; the #13041 collapse cannot hold", goMax)
	}
}

// normalizeLF converts CRLF line endings to LF so `(?m)^...$` anchors match on every
// platform. A lone `\r` is also stripped for robustness against mixed endings.
func normalizeLF(src []byte) string {
	return strings.ReplaceAll(strings.ReplaceAll(string(src), "\r\n", "\n"), "\r", "")
}

func firstIntMatch(source, pattern string) (int, bool) {
	re := regexp.MustCompile(pattern)
	m := re.FindStringSubmatch(source)
	if m == nil {
		return 0, false
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		return 0, false
	}
	return n, true
}
