package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/launchvisual"
)

func TestLaunchNumbersTriggerScope(t *testing.T) {
	cases := map[string]bool{
		"docs/launch/inkling2-launch-2026-11-24.md":    true,
		"docs/launch/assets/inkling2-decode-climb.svg": true,
		"docs/launch/assets/inkling2-charts.data.json": true,
		"docs/benchmarks/published/HEADLINES.json":     true,
		"docs/benchmarks/published/INDEX.md":           false,
		"docs/launchpad.md":                            false,
		"internal/launchvisual/headlines.go":           false,
	}
	for p, want := range cases {
		if got := launchNumbersTriggered(p); got != want {
			t.Errorf("launchNumbersTriggered(%q)=%v want %v", p, got, want)
		}
	}
}

// launchFixture serves the committed fixture data, index, doc and charts, with the charts
// rendered into the map at the data file's real location so the gate sees a consistent tree.
func launchFixture(t *testing.T) map[string][]byte {
	t.Helper()
	read := func(rel string) ([]byte, error) {
		return os.ReadFile(filepath.Join("..", "launchvisual", "testdata", filepath.Base(rel)))
	}
	data, err := read("charts.data.json")
	if err != nil {
		t.Fatal(err)
	}
	index, err := read("headlines.golden.json")
	if err != nil {
		t.Fatal(err)
	}
	doc, err := read("doc.md")
	if err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{
		launchNumbersData:                       data,
		launchNumbersIndex:                      index,
		"internal/launchvisual/testdata/doc.md": doc,
	}
	source := func(rel string) ([]byte, error) {
		if b, ok := files[rel]; ok {
			return b, nil
		}
		return nil, os.ErrNotExist
	}
	rendered, _, _, err := launchvisual.Render(source, launchNumbersData, launchNumbersIndex)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range rendered {
		files[r.Rel] = r.SVG
	}
	return files
}

func launchReader(files map[string][]byte) launchvisual.ReadFunc {
	return func(rel string) ([]byte, error) {
		if b, ok := files[rel]; ok {
			return b, nil
		}
		return nil, os.ErrNotExist
	}
}

func TestLaunchNumbersPassesWhenTreeIsCurrent(t *testing.T) {
	if got := launchNumbersFindings(launchReader(launchFixture(t)), "tree"); len(got) != 0 {
		t.Fatalf("current tree refused: %+v", got)
	}
}

func TestLaunchNumbersBlocksStaleSVG(t *testing.T) {
	files := launchFixture(t)
	files["docs/launch/assets/inkling2-decode-climb.svg"] = []byte("<svg>stale</svg>")
	got := launchNumbersFindings(launchReader(files), "tree")
	if len(got) != 1 || got[0].Gate != "LAUNCH_NUMBERS_FRESHNESS" || got[0].File != "docs/launch/assets/inkling2-decode-climb.svg" {
		t.Fatalf("stale SVG not refused as expected: %+v", got)
	}
	if !strings.Contains(got[0].Detail, "tree tree") {
		t.Errorf("detail does not name the scored tree: %s", got[0].Detail)
	}
}

func TestLaunchNumbersBlocksDocWithUntracedNumber(t *testing.T) {
	files := launchFixture(t)
	files["internal/launchvisual/testdata/doc.md"] = []byte("Decode reaches 999.9 tok/s.\n")
	got := launchNumbersFindings(launchReader(files), "tree")
	found := false
	for _, f := range got {
		if strings.Contains(f.Detail, "999.9") && f.Line == 1 {
			found = true
		}
	}
	if !found {
		t.Fatalf("superseded number not refused: %+v", got)
	}
}
