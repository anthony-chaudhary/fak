package metalgemm

import (
	"os"
	"regexp"
	"strconv"
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
// corrupt state — the native guard returns NULL and the Go caller declines fail-open —
// but it silently *disables* the wider panel, turning the #13041 collapse back into the
// serial 32-token walk with no test failure. This source witness runs on every host and
// makes drift loud: change one ceiling and the others must move with it.
func TestPromptPanelMaxTokensLockstep(t *testing.T) {
	goSource, err := os.ReadFile("graph.go")
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

	goMax, ok := firstIntMatch(string(goSource), `(?m)^const PromptPanelMaxTokens = (\d+)$`)
	if !ok {
		t.Fatal("graph.go: could not find `const PromptPanelMaxTokens = <n>`")
	}
	graphMax, ok := firstIntMatch(string(graphSource), `(?m)^#define QG_MAX_ROWS (\d+)$`)
	if !ok {
		t.Fatal("qwen35_graph.m: could not find `#define QG_MAX_ROWS <n>`")
	}
	gdnMax, ok := firstIntMatch(string(gdnSource), `(?m)^enum \{ MG_GDN_GRAPH_MAX_TOKENS = (\d+) \};$`)
	if !ok {
		t.Fatal("gdn.m: could not find `enum { MG_GDN_GRAPH_MAX_TOKENS = <n> };`")
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
