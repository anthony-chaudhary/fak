package heavinessscore

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These exercise each KPI on a DEFECT fixture (the friction it must catch) and a CLEAN fixture
// (the healthy surface it must pass), plus the parsers that turn source text into a Surface. The
// KPIs are pure functions of a Surface, so the fixtures are Surfaces built by hand -- no tree.
// A live-tree smoke (TestBuild_LiveTree) pins the real headline against the actual repo.

func TestParseVerbs_OnlyTheDispatchBlock(t *testing.T) {
	// Mirrors cmd/fak/main.go: a top-level `switch os.Args[1]` verb table, closed by its own
	// 1-tab `default:`, followed by an UNRELATED inner switch whose string cases must NOT be
	// counted as CLI verbs (the M2 over-count bug). Duplicated verb collapses to one.
	src := "func main() {\n" +
		"\tswitch os.Args[1] {\n" +
		"\tcase \"run\":\n\t\tcmdRun(rest)\n" +
		"\tcase \"commit\":\n\t\tcmdCommit(rest)\n" +
		"\tcase \"commit\":\n\t\tcmdCommit(rest)\n" + // alias/dup -> one verb
		"\tcase \"operator\":\n\t\tcmdOperator(rest)\n" +
		"\tdefault:\n\t\tusage()\n\t}\n}\n" +
		"func decideSession() {\n" +
		"\tswitch verb {\n" +
		"\tcase \"budget\":\n\t\treturn\n" + // inner-switch case: MUST be excluded
		"\tcase \"pace\":\n\t\treturn\n\t}\n}\n"
	got := ParseVerbs(src)
	want := []string{"commit", "operator", "run"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("ParseVerbs = %v, want %v (inner-switch cases budget/pace must be excluded)", got, want)
	}
}

func TestIsMetaVerb(t *testing.T) {
	for _, v := range []string{"conflation-scorecard", "dojo-rsi", "loop-score", "scorecard"} {
		if !isMetaVerb(v) {
			t.Errorf("%q should be a meta verb", v)
		}
	}
	for _, v := range []string{"commit", "run", "operator", "guard"} {
		if isMetaVerb(v) {
			t.Errorf("%q should NOT be a meta verb", v)
		}
	}
}

func TestGuardFlagRegex_CountsEveryFlagFormButNoMethods(t *testing.T) {
	// M1: the *Var binder idiom must count; FlagSet methods (Parse/Args/Usage/...) must not.
	guard := `fs.String("a","","")
		fs.Bool("b",false,"")
		fs.Int("c",0,"")
		fs.Int64("d",0,"")
		fs.Duration("e",0,"")
		fs.StringVar(&x,"f","","")
		fs.BoolVar(&y,"g",false,"")
		fs.IntVar(&z,"h",0,"")
		fs.DurationVar(&w,"i",0,"")
		fs.Float64Var(&q,"j",0,"")
		fs.Var(&v,"k","")
		fs.Parse(args)
		fs.Args()
		fs.Usage()
		fs.PrintDefaults()
		fs.Visit(nil)`
	got := len(reGuardFlag.FindAllString(guard, -1))
	if got != 11 {
		t.Errorf("flag count = %d, want 11 (5 plain + 5 *Var + 1 Var; methods excluded)", got)
	}
}

func TestReasonsRegex_CountsBlocksNotDecoys(t *testing.T) {
	dos := "[reasons.OFF_TRUNK]\nx=1\n[reasons.PUBLIC_LEAK]\n[other.THING]\n[reasons.A_B_C]\n# [reasons.NOT_REAL] in a comment\n"
	if got := len(reReasonsBlock.FindAllString(dos, -1)); got != 3 {
		t.Errorf("refusal-reason count = %d, want 3", got)
	}
}

func TestDocMapCoversSteering_UnlinkedMentionIsStillDebt(t *testing.T) {
	// M3: a bare prose mention of the tokens (no markdown link) must NOT satisfy the HARD gate.
	s := Surface{DocMap: strings.ToLower(
		"the steerability index and operator-heaviness are great ideas we should write up someday")}
	k := kpiDocMapCoversSteering(s)
	if len(k.Defects) != 2 {
		t.Fatalf("an unlinked prose mention must not cover either surface, got %d defects: %v", len(k.Defects), k.Defects)
	}
	if k.Score != 0 {
		t.Errorf("score=%v want 0 when nothing is covered by a linked entry", k.Score)
	}
}

func TestDocMapCoversSteering_LinkedEntryIsClean(t *testing.T) {
	s := Surface{DocMap: strings.ToLower(
		"- [Steerability scorecard](docs/STEERABILITY-SCORECARD.md): run `fak steering`.\n" +
			"- [Operator-heaviness](docs/OPERATOR-HEAVINESS.md): the heaviness_pressure gauge.")}
	k := kpiDocMapCoversSteering(s)
	if len(k.Defects) != 0 {
		t.Fatalf("two linked doc-map entries are clean, got defects %v", k.Defects)
	}
	if k.Score != 100 {
		t.Errorf("clean score=%v want 100", k.Score)
	}
}

func TestAppealChannelWired(t *testing.T) {
	if d := kpiAppealChannelWired(Surface{AppealWired: false}).Defects; len(d) != 1 {
		t.Errorf("an unwired appeal channel must be 1 defect, got %v", d)
	}
	if d := kpiAppealChannelWired(Surface{AppealWired: true}).Defects; len(d) != 0 {
		t.Errorf("a wired appeal channel is clean, got %v", d)
	}
}

func TestCLIVerbCount_SoftBelowCeilingHardAbove(t *testing.T) {
	below := kpiCLIVerbCount(Surface{Verbs: make([]string, 136)})
	if len(below.Defects) != 0 {
		t.Errorf("136 verbs (under the hard ceiling) must NOT be hard debt, got %v", below.Defects)
	}
	if len(below.Soft) != 1 {
		t.Errorf("136 verbs (over the soft line) must be a soft signal, got %v", below.Soft)
	}
	above := kpiCLIVerbCount(Surface{Verbs: make([]string, verbHardCeiling+1)})
	if len(above.Defects) != 1 {
		t.Errorf("a verb count past the hard ceiling must be hard debt, got %v", above.Defects)
	}
	clean := kpiCLIVerbCount(Surface{Verbs: make([]string, verbSoftLine-1)})
	if len(clean.Soft) != 0 || clean.Score != 100 {
		t.Errorf("under the soft line must be clean score 100, got soft=%v score=%v", clean.Soft, clean.Score)
	}
}

func TestFrontDoorFlagBurden_HardPastCeiling(t *testing.T) {
	if d := kpiFrontDoorFlagBurden(Surface{FrontDoorFlags: flagHardCeiling + 1}).Defects; len(d) != 1 {
		t.Errorf("a flag count past the ceiling must be hard debt, got %v", d)
	}
	if k := kpiFrontDoorFlagBurden(Surface{FrontDoorFlags: 49}); len(k.Defects) != 0 || len(k.Soft) != 1 {
		t.Errorf("49 flags must be soft, not hard, got defects=%v soft=%v", k.Defects, k.Soft)
	}
}

func TestRefusalVocabSize_SoftOnly(t *testing.T) {
	k := kpiRefusalVocabSize(Surface{RefusalReasons: 20})
	if len(k.Defects) != 0 {
		t.Errorf("refusal vocab is SOFT and must never emit hard debt, got %v", k.Defects)
	}
	if len(k.Soft) != 1 {
		t.Errorf("20 reasons (over soft line) must be a soft signal, got %v", k.Soft)
	}
}

func TestPressure_NormalizedHeadroomSum(t *testing.T) {
	// 136 verbs (headroom 42), 49 flags (48), 20 reasons (44), 15/136 meta = 11% (18) => 152.
	s := Surface{
		Verbs:          make([]string, 136),
		MetaVerbs:      make([]string, 15),
		FrontDoorFlags: 49,
		RefusalReasons: 20,
	}
	if got := Pressure(s); got != 152 {
		t.Errorf("Pressure = %d, want 152 (normalized headroom-consumed sum)", got)
	}
	// Per-term breakdown must be commensurable (each 0-100), none dominating by raw unit.
	bt := pressureByTerm(s)
	for k, v := range bt {
		if v < 0 || v > 100 {
			t.Errorf("term %s = %d out of [0,100]", k, v)
		}
	}
	// A surface at/under every soft line has zero pressure.
	light := Surface{Verbs: make([]string, verbSoftLine), FrontDoorFlags: flagSoftLine, RefusalReasons: reasonSoftLine}
	if got := Pressure(light); got != 0 {
		t.Errorf("a surface at the soft lines must have zero pressure, got %d", got)
	}
}

func TestHeadroomConsumed_Bounds(t *testing.T) {
	if headroomConsumed(10, 20, 80) != 0 {
		t.Error("at/below soft line must be 0")
	}
	if headroomConsumed(80, 20, 80) != 100 {
		t.Error("at/above ceiling must be 100")
	}
	if got := headroomConsumed(50, 20, 80); got <= 0 || got >= 100 {
		t.Errorf("between lines must be strictly 0<x<100, got %d", got)
	}
}

func TestBuild_DebtCountsHardDefectsOnly(t *testing.T) {
	// Build over a throwaway empty dir: no source files -> 0 verbs/flags/reasons, unwired appeal,
	// undiscoverable doc map. Debt must be exactly the HARD defects (2 docmap + 1 appeal = 3), and
	// the SOFT magnitude KPIs (all zero/empty) must add no debt.
	p := Build(t.TempDir())
	debt, ok := p.Corpus[DebtKey].(int)
	if !ok {
		t.Fatalf("corpus[%s] is not an int: %T", DebtKey, p.Corpus[DebtKey])
	}
	if debt != 3 {
		t.Errorf("empty-tree debt = %d, want 3 (2 docmap + 1 appeal)", debt)
	}
	if p.Schema != Schema {
		t.Errorf("schema = %q, want %q", p.Schema, Schema)
	}
	if _, ok := p.Corpus["heaviness_pressure"]; !ok {
		t.Error("corpus must carry heaviness_pressure")
	}
	if _, ok := p.Corpus["pressure_by_term"]; !ok {
		t.Error("corpus must carry the pressure_by_term breakdown")
	}
}

// TestBuild_LiveTree is the live-tree smoke (H3): it folds the REAL repo surface and pins the HARD
// gate (debt must stay 0 -- the steering surfaces stay discoverable + the appeal channel stays
// wired) and that the headline is computed and non-trivial. The pressure MAGNITUDE ratchet lives
// in the scorecard control pane (a pinned baseline re-derived from this Build), not a hardcoded
// assertion here, because the live verb/flag surface legitimately drifts as the repo grows.
func TestBuild_LiveTree(t *testing.T) {
	root := repoRoot(t)
	if root == "" {
		t.Skip("repo root not found (go.mod); skipping live-tree smoke")
	}
	p := Build(root)
	debt, _ := p.Corpus[DebtKey].(int)
	if debt != CleanFloor {
		t.Errorf("live-tree heaviness_debt = %d, want %d -- a steering surface lost its linked doc-map entry or the appeal channel was un-wired; reason: %s",
			debt, CleanFloor, p.Reason)
	}
	pressure, _ := p.Corpus["heaviness_pressure"].(int)
	if pressure <= 0 {
		t.Errorf("live-tree heaviness_pressure = %d, want > 0 (the operator surface is non-trivial); did the source files fail to read?", pressure)
	}
	t.Logf("live-tree operator-heaviness: debt=%d pressure=%d verbs=%v flags=%v reasons=%v",
		debt, pressure, p.Corpus["verbs"], p.Corpus["front_door_flags"], p.Corpus["refusal_reasons"])
}

// repoRoot walks up from the test's working directory to the directory holding go.mod.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}
