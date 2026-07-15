package negframe

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestClassifyMechanicalReframes pins the high-precision idiom rules: each fixed negative idiom
// classifies to the right category AND yields its unambiguous positive rewrite.
func TestClassifyMechanicalReframes(t *testing.T) {
	cases := []struct {
		line    string
		wantCat Category
		wantSug string
	}{
		{"Don't forget to stamp the commit.", Prohibition, "remember to stamp"},
		{"Do not forget to close the lease.", Prohibition, "remember to close"},
		{"Don't hesitate to ask for a review.", Prohibition, "feel free to ask"},
		{"No need to rebuild the whole tree.", Absence, "you can skip rebuild"},
		{"The output is not unreadable.", Hedge, "readable"},
	}
	for _, c := range cases {
		fs := Classify("t.md", c.line)
		if len(fs) == 0 {
			t.Fatalf("%q: no finding", c.line)
		}
		f := fs[0]
		if f.Category != c.wantCat {
			t.Errorf("%q: category = %q, want %q", c.line, f.Category, c.wantCat)
		}
		if !f.Mechanical() {
			t.Errorf("%q: expected mechanical (a suggestion), got judgement", c.line)
		}
		if !strings.Contains(f.Suggest, c.wantSug) {
			t.Errorf("%q: suggest = %q, want to contain %q", c.line, f.Suggest, c.wantSug)
		}
	}
}

// TestClassifyJudgementTier pins that broad negatives are detected and classified but carry a
// category hint rather than a mechanical rewrite (SOFT, never gating).
func TestClassifyJudgementTier(t *testing.T) {
	cases := []struct {
		line    string
		wantCat Category
	}{
		{"Never merge without a green build.", Prohibition},
		{"Avoid touching the generated files.", Prohibition},
		{"This path is forbidden for workers.", Refusal},
		{"You may not skip the lease.", Refusal},
		{"It fails to record the SHA.", Absence},
	}
	for _, c := range cases {
		fs := Classify("t.md", c.line)
		if len(fs) == 0 {
			t.Fatalf("%q: no finding", c.line)
		}
		// The first finding left-to-right should be the expected category.
		f := fs[0]
		if f.Category != c.wantCat {
			t.Errorf("%q: category = %q, want %q (findings=%+v)", c.line, f.Category, c.wantCat, fs)
		}
		if f.Mechanical() {
			t.Errorf("%q: expected judgement tier, got mechanical suggestion %q", c.line, f.Suggest)
		}
		if f.Hint == "" {
			t.Errorf("%q: judgement finding must carry a category hint", c.line)
		}
	}
}

// TestSpecificRuleBeatsGeneric proves "don't forget to X" is reported once as the mechanical
// reframe, not also double-counted as a bare "don't" prohibition.
func TestSpecificRuleBeatsGeneric(t *testing.T) {
	fs := Classify("t.md", "Don't forget to push the branch.")
	if len(fs) != 1 {
		t.Fatalf("want exactly one finding (no double-count), got %d: %+v", len(fs), fs)
	}
	if !fs[0].Mechanical() {
		t.Fatalf("want the mechanical reframe to win, got judgement")
	}
}

// TestCodeFenceSkipped proves fenced code blocks are not gardened -- a "don't" inside a sample
// is not a prose finding.
func TestCodeFenceSkipped(t *testing.T) {
	text := "Prose here is clean.\n\n```\nif !ok { // don't do this\n}\n```\n\nMore clean prose."
	fs := Classify("t.md", text)
	if len(fs) != 0 {
		t.Fatalf("code fence should be skipped, got findings: %+v", fs)
	}
}

// TestClassifyLineNumbers pins that findings carry the 1-based line they were found on.
func TestClassifyLineNumbers(t *testing.T) {
	text := "line one clean\nDon't forget to sign.\nline three clean"
	fs := Classify("t.md", text)
	if len(fs) != 1 {
		t.Fatalf("want one finding, got %d", len(fs))
	}
	if fs[0].Line != 2 {
		t.Errorf("line = %d, want 2", fs[0].Line)
	}
	if fs[0].Path != "t.md" {
		t.Errorf("path = %q, want t.md", fs[0].Path)
	}
}

// TestScoreDocTiers pins the per-doc mechanical/judgement split.
func TestScoreDocTiers(t *testing.T) {
	text := "Don't forget to stamp.\nNever merge broken code.\nAvoid the shortcut."
	d := ScoreDoc("t.md", text)
	if d.Mechanical != 1 {
		t.Errorf("mechanical = %d, want 1", d.Mechanical)
	}
	if d.Judgement != 2 {
		t.Errorf("judgement = %d, want 2", d.Judgement)
	}
	if d.Negatives() != 3 {
		t.Errorf("negatives = %d, want 3", d.Negatives())
	}
}

// TestNewFindingsRatchet pins the diff primitive: only NEWLY introduced mechanical negatives are
// returned; a pre-existing one (even moved to a new line) is not.
func TestNewFindingsRatchet(t *testing.T) {
	before := "Intro line.\nDon't forget to stamp.\n"
	// after: the pre-existing "don't forget to stamp" moved down a line, plus a brand-new one.
	after := "Intro line.\nA new middle line.\nDon't forget to stamp.\nNo need to poll the queue.\n"
	nf := NewFindings("t.md", before, after)
	if len(nf) != 1 {
		t.Fatalf("want exactly the one NEW mechanical negative, got %d: %+v", len(nf), nf)
	}
	if !strings.Contains(strings.ToLower(nf[0].Text), "no need to poll") {
		t.Errorf("new finding = %q, want the 'no need to poll' one", nf[0].Text)
	}
}

// TestNewFindingsIgnoresJudgement proves the ratchet gates on mechanical wins only: a newly
// added judgement-tier negative is not returned (it is advisory, never a gate).
func TestNewFindingsIgnoresJudgement(t *testing.T) {
	before := "Intro.\n"
	after := "Intro.\nNever do the risky thing.\n"
	if nf := NewFindings("t.md", before, after); len(nf) != 0 {
		t.Fatalf("judgement negatives must not ratchet, got %+v", nf)
	}
}

// TestHintCoversEveryCategory guards against a category with no reframe hint (an empty hint
// would render a bare "[category: ]" in the soft work-list).
func TestHintCoversEveryCategory(t *testing.T) {
	for _, c := range Categories {
		if Hint(c) == "" {
			t.Errorf("category %q has no reframe hint", c)
		}
	}
}

// TestHedgeNotUnFalsePositives is the regression guard for the "un"-is-not-always-negating bug
// that a real-corpus scan surfaced: the double-negative mechanical rule must fire ONLY on genuine
// un- antonyms ("not unreadable" -> "readable"), and must stay silent on words where "un" is part
// of the root ("not unique", "not unless", "does not unlock", "not universal"). A false mechanical
// hit here is a GATING defect -- it would red the ratchet on innocent prose.
func TestHedgeNotUnFalsePositives(t *testing.T) {
	// These carry a real un- antonym: a mechanical hedge suggestion is correct.
	for _, line := range []string{
		"The output is not unreadable.",
		"This failure mode is not uncommon.",
		"The result is not unexpected here.",
	} {
		fs := Classify("t.md", line)
		if len(fs) == 0 || !fs[0].Mechanical() {
			t.Errorf("%q: want a mechanical hedge reframe, got %+v", line, fs)
		}
	}
	// In these, "un" is part of the root -- a mechanical suggestion would be nonsense. The rule
	// must not produce ANY mechanical finding for them.
	for _, line := range []string{
		"This lane is not unique to the worker.",
		"The interface is not universal across hosts.",
		"A retry is not allowed unless you opt in.",
		"Safety takes precedence and does not unlock a retry.",
		"The plan runs not until the lease drains.",
	} {
		for _, f := range Classify("t.md", line) {
			if f.Category == Hedge && f.Mechanical() {
				t.Errorf("%q: rule emitted a bogus mechanical hedge %q", line, f.Suggest)
			}
		}
	}
}

func TestWeightHotFindingOutranksColdFinding(t *testing.T) {
	hot := Classify("cmd/fak/guard_runtime.go", "do not forget to recover")
	cold := Classify(".claude/skills/cold/SKILL.md", "do not forget to recover")
	if len(hot) != 1 || len(cold) != 1 {
		t.Fatalf("hot=%v cold=%v", hot, cold)
	}
	if hot[0].Tier != TierPerTurn || cold[0].Tier != TierCold || hot[0].Weight <= cold[0].Weight {
		t.Fatalf("tier/weight mismatch hot=%+v cold=%+v", hot[0], cold[0])
	}

	root := t.TempDir()
	for path, text := range map[string]string{
		"cmd/fak/guard_runtime.go":     "do not forget to recover",
		".claude/skills/cold/SKILL.md": "do not forget to recover",
	} {
		full := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(text), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	got := AllFindings(root, []string{".claude/skills/cold/SKILL.md", "cmd/fak/guard_runtime.go"})
	if len(got) != 2 || got[0].Path != "cmd/fak/guard_runtime.go" {
		t.Fatalf("weighted order=%+v", got)
	}
}

func TestWeightBuildExposesWeightedDebtWithoutChangingFlatDebt(t *testing.T) {
	root := t.TempDir()
	paths := []string{"AGENTS.md", ".claude/skills/cold/SKILL.md"}
	for _, path := range paths {
		full := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte("do not forget to recover"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	payload := Build(root, paths)
	if got := payload.Corpus["mechanical_debt"]; got != 2 {
		t.Fatalf("flat debt=%v want 2", got)
	}
	if got := payload.Corpus["weighted_debt"]; got != TierPerSession.Weight()+TierCold.Weight() {
		t.Fatalf("weighted debt=%v want %d", got, TierPerSession.Weight()+TierCold.Weight())
	}
}
