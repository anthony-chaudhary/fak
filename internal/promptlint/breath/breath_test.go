package breath

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// What this suite has to prove, and why each part exists.
//
// A lint's own tests have one job ordinary tests do not: proving the check FIRES.
// `fak breath` printing `clean` over the real tree is evidence of nothing on its own — it
// is the same output a scan that examined zero pages produces. So every rule below is
// driven from a body that breaks it AND a body that does not, the floor is tested by
// starving the corpus rather than by reading the code and agreeing with it, and the SCOPE
// boundary is tested too: the judgement half must stay un-gated by construction.

// page renders a minimal page around a block, so a rule's input is the line under test.
func page(block, tail string) []byte {
	return []byte("# Title\n\nlede.\n\n> **In one breath:** " + block + "\n\n" + tail + "\n\n## Body\n\nprose.\n")
}

const goodTail = "**One line:** the precise version, where nuance is allowed."

func TestRulesFire(t *testing.T) {
	// Each case names the rule and the Kind that proves the RIGHT rule fired — a count
	// of findings alone would pass if a different rule fired for a different reason.
	cases := []struct {
		name  string
		block string
		tail  string
		want  Kind // "" means the page must be clean
	}{{
		name:  "compliant four-sentence block is clean",
		block: "A label has three parts. This engine makes two of them by itself. It draws its own\n> outlines and classes. Only existence comes from another detector.",
		tail:  goodTail,
	}, {
		name:  "compliant two-sentence block is clean",
		block: "A cache keeps an answer twice paid for. It is stored inside the kernel.",
		tail:  goodTail,
	}, {
		name:  "one sentence is below the floor",
		block: "This is a single sentence and so it cannot be a block.",
		tail:  goodTail,
		want:  BreathSentenceCount,
	}, {
		name:  "five sentences is over the ceiling",
		block: "One idea. Two ideas. Three ideas. Four ideas. Five ideas.",
		tail:  goodTail,
		want:  BreathSentenceCount,
	}, {
		// The two cases either side of the ceiling, counted by hand and pinned: an
		// off-by-one in Words would otherwise show up as a whole tree of pages being
		// told to shorten sentences that already comply.
		name:  "a sixteen-word sentence is one word over",
		block: "This sentence has exactly sixteen words which is one more than the contract can allow here. And a second.",
		tail:  goodTail,
		want:  BreathSentenceLength,
	}, {
		name:  "a fifteen-word sentence is exactly at the ceiling and passes",
		block: "This sentence has exactly fifteen words in it which is just what the contract allows. And a second.",
		tail:  goodTail,
	}, {
		name:  "em-dash is refused",
		block: "A label has three parts. The engine makes two of them — the other is borrowed.",
		tail:  goodTail,
		want:  BreathEmDash,
	}, {
		name:  "the same clause as a comma, not a parenthetical, is clean",
		block: "A label has three parts. The engine makes two of them, the third is borrowed. It also stores a value.",
		tail:  goodTail,
	}, {
		name:  "a parenthetical is refused",
		block: "A label has three parts. The engine makes two of them (the third is borrowed).",
		tail:  goodTail,
		want:  BreathParentheses,
	}, {
		name:  "an unexpanded acronym is refused",
		block: "A GPU runs the model. It is faster than the alternative.",
		tail:  goodTail,
		want:  BreathUnexpandedAcronym,
	}, {
		name:  "an acronym spelled out on the spot is clean",
		block: "A GPU is a graphics processing unit. It runs many sums at once.",
		tail:  goodTail,
	}, {
		name:  "a filename in caps is not an acronym",
		block: "The rules live in AGENTS.md. Read that page before you write code.",
		tail:  goodTail,
	}, {
		name:  "a block with no One line paragraph is refused",
		block: "A label has three parts. The engine makes two of them itself.",
		tail:  "Some other paragraph entirely.",
		want:  BreathMissingOneLine,
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := DefaultContract().Check(Doc{Path: "docs/explainers/x.md", Body: page(c.block, c.tail)})
			if c.want == "" {
				if len(got) != 0 {
					t.Fatalf("want clean, got %d finding(s): %s: %s", len(got), got[0].Kind, got[0].Detail)
				}
				return
			}
			if len(got) == 0 {
				t.Fatalf("want a %s finding, got clean — the rule did not fire", c.want)
			}
			for _, f := range got {
				if f.Kind == c.want {
					if f.Path != "docs/explainers/x.md" {
						t.Errorf("finding path = %q, want the page path", f.Path)
					}
					if f.Line <= 0 {
						t.Errorf("finding line = %d, want the block's 1-based line", f.Line)
					}
					return
				}
			}
			t.Fatalf("fired on the wrong rule: want %s, got %s (%s)", c.want, got[0].Kind, got[0].Detail)
		})
	}
}

func TestMissingBlockFires(t *testing.T) {
	body := []byte("# Title\n\n**One line:** the precise version.\n\n## Body\n\nprose.\n")
	got := DefaultContract().Check(Doc{Path: "docs/explainers/x.md", Body: body})
	if len(got) != 1 || got[0].Kind != BreathMissing {
		t.Fatalf("a page with no block must report exactly BREATH_MISSING, got %+v", got)
	}
	if !strings.Contains(got[0].Detail, ContractDoc) {
		t.Errorf("the finding must route the author to %s; got %q", ContractDoc, got[0].Detail)
	}
}

// TestMisplacedBlockFires pins the POSITION half of the contract: a block below the first
// section heading is unreachable to a loader that serves only the lede, which is the
// consumer the block exists for (#3229 / #3535).
func TestMisplacedBlockFires(t *testing.T) {
	body := []byte("# Title\n\n## Body\n\nprose.\n\n> **In one breath:** A short one. And a second one.\n\n" + goodTail + "\n")
	got := DefaultContract().Check(Doc{Path: "docs/explainers/x.md", Body: body})
	if !hasKind(got, BreathMisplaced) {
		t.Fatalf("a block after the first `##` must report BREATH_MISPLACED, got %+v", got)
	}
}

// TestExactMarkerIsRequired pins the deliberate intolerance: a near-miss marker is
// reported as a MISSING block, not quietly parsed, because the marker is also read by eye
// and keyed on by the budget-aware loader.
func TestExactMarkerIsRequired(t *testing.T) {
	for _, near := range []string{
		"> **In One Breath:** A short one. And a second.",
		"> *In one breath:* A short one. And a second.",
		"**In one breath:** A short one. And a second.",
	} {
		body := []byte("# Title\n\n" + near + "\n\n" + goodTail + "\n")
		got := DefaultContract().Check(Doc{Path: "docs/explainers/x.md", Body: body})
		if !hasKind(got, BreathMissing) {
			t.Errorf("near-miss marker %q must report BREATH_MISSING, got %+v", near, got)
		}
	}
}

// TestBlockIsRewrapIndependent: the counts must not move when an author hard-wraps
// differently. A rewrap is not a content change.
func TestBlockIsRewrapIndependent(t *testing.T) {
	one := page("A label has three parts. The engine makes two of them itself.", goodTail)
	wrapped := page("A label has three parts. The engine\n> makes two of them itself.", goodTail)
	a, okA := Extract(one)
	b, okB := Extract(wrapped)
	if !okA || !okB {
		t.Fatalf("both bodies must yield a block (%v, %v)", okA, okB)
	}
	if a.Text != b.Text {
		t.Fatalf("rewrapping changed the block text:\n one = %q\n wrapped = %q", a.Text, b.Text)
	}
}

func TestOneLineMustImmediatelyFollowItsBreathBlock(t *testing.T) {
	for _, body := range []string{
		"# Title\n\n**One line:** an unrelated earlier summary.\n\n> **In one breath:** A short one. And a second one.\n\n## Body\n",
		"# Title\n\n> **In one breath:** A short one. And a second one.\n\nAn intervening paragraph.\n\n**One line:** too late.\n",
	} {
		got := DefaultContract().Check(Doc{Path: "docs/explainers/x.md", Body: []byte(body)})
		if !hasKind(got, BreathMissingOneLine) {
			t.Fatalf("a non-sibling `One line` paragraph must not satisfy the pairing, got %+v", got)
		}
	}
}

// TestElaborativeBlockStaysClean is the PlainQAFact regression pin. This block says things
// its page body does not — which is what a GOOD plain-language block does — and it must
// stay green. If anyone later adds an entailment or faithfulness rule, this test is the
// one that goes red first, before the doc corpus does.
func TestElaborativeBlockStaysClean(t *testing.T) {
	body := []byte("# Title\n\n> **In one breath:** A graphics card is a chip doing many sums at once.\n" +
		"> Our models need one to run fast.\n\n" + goodTail + "\n\n## Body\n\n" +
		"The scheduler assigns work to devices by capability class.\n")
	if got := DefaultContract().Check(Doc{Path: "docs/explainers/x.md", Body: body}); len(got) != 0 {
		t.Fatalf("an elaborative block (adds background absent from the body) must stay clean; got %+v", got)
	}
}

// TestNoJudgementHalfKind is the scope boundary, mechanized. The closed vocabulary may not
// grow a finding that judges whether the block is TRUE. Adding one means deleting this
// test, which means arguing past the package doc and docs/ONE-BREATH-CONTRACT.md first —
// which is exactly the review chokepoint the scope decision needs.
func TestNoJudgementHalfKind(t *testing.T) {
	banned := []string{"FAITHFUL", "ENTAIL", "ACCURA", "CORRECT", "COMPLETE", "CONSIST", "HALLUCIN", "SUPPORT"}
	for _, k := range Kinds() {
		for _, b := range banned {
			if strings.Contains(string(k), b) {
				t.Errorf("Kind %q names the JUDGEMENT half of the contract (%q). breath gates the "+
					"COUNTABLE half only; an entailment-style check scores the best-written blocks as "+
					"the least faithful (PlainQAFact, arXiv 2503.08890). Read the package doc and "+
					"%s before re-adding this.", k, b, ContractDoc)
			}
		}
	}
}

// TestEveryReportCarriesTheScopeNotice: a consumer reading only the tool's output must
// still learn which half was not judged. A gate that silently implies it measured
// faithfulness is the failure mode this package exists to avoid.
func TestEveryReportCarriesTheScopeNotice(t *testing.T) {
	cen, _ := DefaultContract().Scan(nil)
	if cen.Notice != ScopeNotice {
		t.Errorf("Census.Notice must carry ScopeNotice verbatim, got %q", cen.Notice)
	}
	if !strings.Contains(BaselineHeader, ScopeNotice) {
		t.Error("the regenerated baseline header must carry ScopeNotice, so the artefact itself says " +
			"which half produced it")
	}
	for _, want := range []string{"does NOT judge", "accuracy", "completeness", "faithfulness", ContractDoc} {
		if !strings.Contains(ScopeNotice, want) {
			t.Errorf("ScopeNotice must name %q; got %q", want, ScopeNotice)
		}
	}
}

func TestSentenceSplitterHandlesTheTraps(t *testing.T) {
	// Both traps fail LENIENT — a block at the ceiling reads as one over or under, and
	// nobody looks twice at a count.
	for _, c := range []struct {
		text string
		want int
	}{
		{"A score of 1.0 means the boxes match. A score of 0 means they do not.", 2},
		{"It is fast, e.g. 30 frames per second. That is enough.", 2},
		{"One sentence with no full stop", 1},
		{"Is it here? Yes. No!", 3},
	} {
		if got := len(Sentences(c.text)); got != c.want {
			t.Errorf("Sentences(%q) = %d, want %d", c.text, got, c.want)
		}
	}
}

func TestWordCountIgnoresMarkdownEmphasis(t *testing.T) {
	if got, want := Words("**Exactly one** thing is `borrowed` here."), 6; got != want {
		t.Errorf("Words = %d, want %d — emphasis and code fences are markup, not words", got, want)
	}
}

func TestUnexpandedAcronyms(t *testing.T) {
	c := DefaultContract()
	for _, tc := range []struct {
		text string
		want []string
	}{
		{"A GPU is a graphics processing unit. It runs sums.", nil},
		{"A GPU runs the model fast.", []string{"GPU"}},
		{"Two GPUs run the model. A graphics processing unit is a chip.", nil},
		{"A KV cache stores key value pairs.", nil},
		{"The rules live in AGENTS.md today.", nil},
		{"TTL means time to live here.", nil},
		{"AI is here.", []string{"AI"}}, // the window may not contain the acronym itself
		{"No acronyms at all in this one.", nil},
	} {
		got := c.UnexpandedAcronyms(tc.text)
		if strings.Join(got, ",") != strings.Join(tc.want, ",") {
			t.Errorf("UnexpandedAcronyms(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
	// The allowlist escape hatch, for a name in capitals that is not an acronym at all.
	c.AllowAcronyms = []string{"REST"}
	if got := c.UnexpandedAcronyms("A REST endpoint answers the call."); len(got) != 0 {
		t.Errorf("AllowAcronyms did not exempt REST: %v", got)
	}
}

// TestScanFloorFiresOnAStarvedCorpus proves the gate refuses to report clean when it
// examined too little to have an opinion. This is the test that would catch the roots
// being renamed out from under the scan.
func TestScanFloorFiresOnAStarvedCorpus(t *testing.T) {
	c := DefaultContract()
	cen, got := c.Scan([]Doc{{Path: "docs/explainers/x.md", Body: page("A short one. And a second one.", goodTail)}})
	if len(got) == 0 || got[0].Kind != BreathScanFloor {
		t.Fatalf("one page must trip the floor as the FIRST finding, got %+v", got)
	}
	if got[0].Path != "internal/promptlint/breath" {
		t.Errorf("the floor finding must point at the gate, not a page; got %q", got[0].Path)
	}
	if cen.Conforming != 1 || cen.Pages != 1 {
		t.Errorf("census = %+v, want 1 page / 1 conforming", cen)
	}
	// A run at or above the floor must NOT carry the floor finding.
	var many []Doc
	for i := 0; i < c.Floor; i++ {
		many = append(many, Doc{Path: "docs/explainers/p" + strconv.Itoa(i) + ".md",
			Body: page("A short one. And a second one.", goodTail)})
	}
	if _, got := c.Scan(many); hasKind(got, BreathScanFloor) {
		t.Errorf("a corpus at the floor must not trip it, got %+v", got)
	}
}

// TestCensusCountsThreeWaysWithADenominator pins the measurement shape the contract asks
// for: conforming + failing + missing must exhaust the denominator.
func TestCensusCountsThreeWaysWithADenominator(t *testing.T) {
	c := DefaultContract()
	c.Floor = 0
	docs := []Doc{
		{Path: "docs/explainers/a.md", Body: page("A short one. And a second one.", goodTail)},
		{Path: "docs/explainers/b.md", Body: page("One idea. Two ideas. Three ideas. Four ideas. Five ideas.", goodTail)},
		{Path: "docs/explainers/c.md", Body: []byte("# Title\n\nno block here.\n")},
		{Path: ContractDoc, Body: []byte("# Contract\n\n> **In one breath:** exempt.\n")},
	}
	cen, _ := c.Scan(docs)
	if cen.Pages != 3 || cen.Conforming != 1 || cen.Failing != 1 || cen.Missing != 1 || cen.Exempt != 1 {
		t.Fatalf("census = %+v, want pages=3 conforming=1 failing=1 missing=1 exempt=1", cen)
	}
	if cen.Conforming+cen.Failing+cen.Missing != cen.Pages {
		t.Errorf("the three numbers must exhaust the denominator: %+v", cen)
	}
}

// TestContractDocIsExemptFromItsOwnRule pins the exemption that would otherwise make the
// page DEFINING the contract the first page to break it: it quotes the marker in prose.
func TestContractDocIsExemptFromItsOwnRule(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), ContractDoc))
	if err != nil {
		t.Fatalf("read %s: %v", ContractDoc, err)
	}
	if got := DefaultContract().Check(Doc{Path: ContractDoc, Body: raw}); len(got) != 0 {
		t.Fatalf("the contract page must be exempt from its own rule, got %+v", got)
	}
	// The exemption has to be load-bearing, not decorative: judged as an ordinary page,
	// the contract page trips its own rules (it quotes the marker inside a fenced example
	// and discusses em-dashes and parentheses in prose beneath it).
	bare := DefaultContract()
	bare.Exempt = nil
	if got := bare.Check(Doc{Path: ContractDoc, Body: raw}); len(got) == 0 {
		t.Error("the exemption is decorative: the contract page passes even when judged as a page, " +
			"so this test would not notice if the exemption silently stopped applying")
	}
}

// TestContractNumbersMatchThisPage is the drift guard. The numbers this gate ENFORCES must
// be the numbers docs/ONE-BREATH-CONTRACT.md PROMISES, because the page states them in
// prose a human retypes. Without this the contract could be relaxed on the page and kept
// strict by the gate, or — worse — the reverse, where the page goes on telling authors a
// rule nothing enforces.
func TestContractNumbersMatchThisPage(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), ContractDoc))
	if err != nil {
		t.Fatalf("read %s: %v", ContractDoc, err)
	}
	body := string(raw)
	c := DefaultContract()
	spelled := map[int]string{2: "two", 4: "four", 15: "fifteen"}
	for _, n := range []int{c.MinSentences, c.MaxSentences, c.MaxWords} {
		digits := regexp.MustCompile(`\b` + strconv.Itoa(n) + `\b`).MatchString(body)
		if !digits && !strings.Contains(strings.ToLower(body), spelled[n]) {
			t.Errorf("%s states neither %d nor %q, but the gate enforces it — the contract the page "+
				"promises and the contract the gate applies have drifted", ContractDoc, n, spelled[n])
		}
	}
	for _, want := range []string{"em-dash", "parenthes", "One line", "In one breath", "#3229", "#3535"} {
		if !strings.Contains(body, want) {
			t.Errorf("%s no longer mentions %q, which the gate enforces or cites as its consumer", ContractDoc, want)
		}
	}
	// Every closed-vocabulary Kind must be documented on the page a reader is routed to.
	for _, k := range Kinds() {
		if k == BreathScanFloor {
			continue // named in the ratchet section as a tool-integrity refusal
		}
		if !strings.Contains(body, string(k)) {
			t.Errorf("%s does not name the refusal token %s, so an author who hits it has nowhere to read", ContractDoc, k)
		}
	}
	// And the scope refusal must be on the page in writing, not only in Go.
	for _, want := range []string{"2507.14096", "2503.08890", "elaborative"} {
		if !strings.Contains(strings.ToLower(body), strings.ToLower(want)) {
			t.Errorf("%s must carry the scope-refusal citation %q in writing — a future contributor "+
				"must not be able to add an entailment check without arguing past it", ContractDoc, want)
		}
	}
}

func hasKind(fs []Finding, k Kind) bool {
	for _, f := range fs {
		if f.Kind == k {
			return true
		}
	}
	return false
}

// repoRoot walks up from the test's working directory to the module root.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatal("could not find the module root above the test working directory")
	return ""
}
