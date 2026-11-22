package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hooks/alphafence"
)

const (
	fencedLineA = "Rank every candidate by realized slippage before sizing the order."
	fencedLineB = "Hand off to the overnight desk only after the spread inverts twice."
)

// writeAlphaFence fences body under a fresh root and returns the root.
func writeAlphaFence(t *testing.T, opts alphafence.Options, body string) string {
	t.Helper()
	root := t.TempDir()
	f, err := alphafence.Build(opts, []alphafence.Source{{Label: "alpha-1", Body: []byte(body)}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := f.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	writeAlphaFenceBytes(t, root, raw)
	return root
}

func writeAlphaFenceBytes(t *testing.T, root string, raw []byte) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(alphaFenceRel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

func assertFenceFinding(t *testing.T, got []Finding, file string, line int) {
	t.Helper()
	if len(got) != 1 {
		t.Fatalf("want one fence finding, got %+v", got)
	}
	f := got[0]
	if f.Gate != "PUBLIC_LEAK" || f.File != file || f.Line != line || !strings.Contains(f.Detail, "alpha fence: alpha-1") {
		t.Fatalf("unexpected finding: %+v", f)
	}
	if strings.Contains(strings.ToLower(f.Detail), "slippage") || strings.Contains(strings.ToLower(f.Detail), "overnight") {
		t.Fatalf("finding repeats the private text: %q", f.Detail)
	}
}

func TestAlphaFenceRefusesOutboundPayload(t *testing.T) {
	root := writeAlphaFence(t, alphafence.Options{}, fencedLineA+"\n")
	payload := "status update\n> " + strings.ToUpper(fencedLineA) + "\nthanks"
	assertFenceFinding(t, ScanOutboundText(payload, root), "", 2)
	if got := ScanOutboundText("status update\nnothing private here at all, honestly", root); len(got) != 0 {
		t.Fatalf("clean payload refused: %+v", got)
	}
	if got := ScanOutboundText(payload, t.TempDir()); len(got) != 0 {
		t.Fatalf("a root with no fence must not refuse: %+v", got)
	}
}

func TestAlphaFenceRefusesCommitMessage(t *testing.T) {
	root := writeAlphaFence(t, alphafence.Options{}, fencedLineA+"\n")
	msg := "docs: note\n\n" + fencedLineA + "\n# " + fencedLineA + "\n"
	assertFenceFinding(t, ScanMessageNeedles(msg, root), "", 3)
}

func TestAlphaFenceRefusesStagedAddedLines(t *testing.T) {
	root := writeAlphaFence(t, alphafence.Options{}, fencedLineA+"\n"+fencedLineB+"\n")
	d := &StagedDiff{
		Root:      root,
		fileCache: map[string]fileEntry{},
		AddedByFile: map[string][]AddedLine{
			"docs/playbook.md": {{New: 7, Text: "Hand off to the overnight desk whenever the spread looks odd."}, {New: 8, Text: "an ordinary public sentence of similar length"}},
			"docs/clean.md":    {{New: 1, Text: "nothing here was ever declared private by anyone"}},
		},
	}
	got, err := gatePublicLeak(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("a sentence that only resembles the fenced line must not match: %+v", got)
	}
	d.AddedByFile["docs/playbook.md"][0].Text = "  // - " + fencedLineB
	got, err = gatePublicLeak(d)
	if err != nil {
		t.Fatal(err)
	}
	assertFenceFinding(t, got, "docs/playbook.md", 7)
}

func TestAlphaFenceRefusesTrackedTreeAndHonorsMinHits(t *testing.T) {
	root := writeAlphaFence(t, alphafence.Options{MinHits: 2}, fencedLineA+"\n"+fencedLineB+"\n")
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(alphaFenceRel)))
	if err != nil {
		t.Fatal(err)
	}
	files := mapReader{
		alphaFenceRel: string(raw),
		"one.md":      "intro\n" + fencedLineA + "\n",
		"both.md":     fencedLineA + "\nfiller\n" + fencedLineB + "\n",
	}
	got, _ := scanPublicLeakTreeFiles([]string{"both.md", "one.md"}, files.FileBytes)
	if len(got) != 2 || got[0].File != "both.md" || got[0].Line != 1 || got[1].File != "both.md" || got[1].Line != 3 {
		t.Fatalf("want both.md lines 1 and 3 only (one.md is under min_hits), got %+v", got)
	}
}

func TestAlphaFenceUnusableFenceIsAFindingNotAPass(t *testing.T) {
	root := t.TempDir()
	writeAlphaFenceBytes(t, root, []byte(`{"schema":"fak.alpha-fence/0"}`))
	got := ScanOutboundText("anything", root)
	if len(got) != 1 || got[0].File != alphaFenceRel || !strings.Contains(got[0].Detail, "alpha fence unusable") {
		t.Fatalf("defective fence must surface as a finding: %+v", got)
	}
}

func TestAlphaFenceSeesLinesEmbeddedInJSONAndTables(t *testing.T) {
	root := writeAlphaFence(t, alphafence.Options{}, fencedLineA+"\n")
	blocks := `[{"type":"section","text":{"type":"mrkdwn","text":"heads up\n` + fencedLineA + `\nthanks"}}]`
	assertFenceFinding(t, ScanOutboundText("update\n"+blocks, root), "", 2)
	assertFenceFinding(t, ScanOutboundText("| rule | "+fencedLineA+" | owner |", root), "", 1)
	zeroWidth := strings.Replace(fencedLineA, "candidate", "candi\u200bdate", 1)
	assertFenceFinding(t, ScanOutboundText(zeroWidth, root), "", 1)
}

func TestAlphaFenceIgnoresACommittedFenceAndReportsItPublished(t *testing.T) {
	root := writeAlphaFence(t, alphafence.Options{}, fencedLineA+"\n")
	d := &StagedDiff{
		Root:      root,
		fileCache: map[string]fileEntry{alphaFenceRel: {data: []byte(`{"schema":"fak.alpha-fence/1"}`), exists: true}},
		AddedByFile: map[string][]AddedLine{
			alphaFenceRel:   {{New: 1, Text: "{}"}},
			"docs/howto.md": {{New: 4, Text: fencedLineA}},
		},
	}
	got, err := gatePublicLeak(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("want the published-fence finding and the leak found by the local fence, got %+v", got)
	}
	if got[0].File != alphaFenceRel || !strings.Contains(got[0].Detail, "alpha fence published") {
		t.Fatalf("staged fence not reported: %+v", got[0])
	}
	assertFenceFinding(t, got[1:], "docs/howto.md", 4)
}

func TestAlphaFenceEmptyOrUnreadableFenceIsLoud(t *testing.T) {
	root := t.TempDir()
	writeAlphaFenceBytes(t, root, []byte(`{"schema":"fak.alpha-fence/1","salt":"aa","min_line_len":24,"min_hits":1,"entries":[]}`))
	if got := ScanOutboundText("anything", root); len(got) != 1 || !strings.Contains(got[0].Detail, "alpha fence unusable") {
		t.Fatalf("a fence with no hashes must be a finding: %+v", got)
	}
	blocked := t.TempDir()
	if err := os.MkdirAll(filepath.Join(blocked, filepath.FromSlash(alphaFenceRel)), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := ScanOutboundText("anything", blocked); len(got) != 1 || !strings.Contains(got[0].Detail, "cannot be read") {
		t.Fatalf("a fence that exists but cannot be read must be a finding: %+v", got)
	}
}
