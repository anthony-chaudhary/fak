package skilleffectiveness

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeSkill materializes .claude/skills/<name>/SKILL.md under root.
func writeSkill(t *testing.T, root, name, body string) {
	t.Helper()
	dir := filepath.Join(root, ".claude", "skills", name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(body), 0o644); err != nil {
		t.Fatalf("write SKILL.md: %v", err)
	}
}

// frontmatter builds a SKILL.md with the given description and body words.
func frontmatter(name, description, body string) string {
	return "---\nname: " + name + "\ndescription: " + description + "\n---\n\n# " + name + "\n\n" + body + "\n"
}

// TestScanProbesEveryAffordanceFromDisk pins the four per-skill probes against a real tree:
// a skill missing its frontmatter description, a skill whose only trigger phrasing is the
// "use to" variant (which MUST count as a trigger), a skill one word over the metadata
// budget, and a skill one word over the body budget. Each is a distinct axis -- collapsing
// any two of them, or dropping either trigger alternative, reds this test.
func TestScanProbesEveryAffordanceFromDisk(t *testing.T) {
	root := t.TempDir()
	writeSkill(t, root, "good", frontmatter("good", "Use when you need the good path", "a short body"))
	writeSkill(t, root, "usetovariant", frontmatter("usetovariant", "Use to bake bread", "a short body"))
	writeSkill(t, root, "nodesc", "---\nname: nodesc\n---\n\n# nodesc\n\nUse when you need nodesc\n")
	writeSkill(t, root, "notrigger", frontmatter("notrigger", "A skill with no trigger phrasing at all", "body without the phrase"))
	writeSkill(t, root, "fatmeta", frontmatter("fatmeta", "Use when "+strings.TrimSpace(strings.Repeat("w ", MetadataWordBudget)), "short"))
	writeSkill(t, root, "fatbody", frontmatter("fatbody", "Use when you need fatbody", strings.Repeat("word ", BodyWordBudget+1)))

	got := map[string]Scanned{}
	for _, s := range Scan(root) {
		got[s.Name] = s
	}
	// Assert the relation the count stood for -- Scan returns exactly the skills that are
	// on disk -- by deriving the expectation from the tree rather than freezing today's
	// total. A probe added above then cannot pass while silently going unscanned, and no
	// magic number needs hand-updating when one is (CHANGE_DETECTOR_TEST).
	entries, err := os.ReadDir(filepath.Join(root, ".claude", "skills"))
	if err != nil {
		t.Fatalf("read skills dir: %v", err)
	}
	for _, e := range entries {
		if _, ok := got[e.Name()]; !ok {
			t.Fatalf("Scan skipped the %q skill on disk; got %v", e.Name(), got)
		}
	}
	if len(got) != len(entries) {
		t.Fatalf("Scan found %d skills, want one per dir on disk (%d): %v", len(got), len(entries), got)
	}

	if s := got["good"]; !s.Readable || !s.HasDescription || !s.HasTrigger || s.OverMetadataBudget() || s.OverBodyBudget() {
		t.Errorf("good = %+v, want readable+described+triggered and inside both budgets", s)
	}
	// "use to" is a first-class trigger alternative; dropping it would red here.
	if s := got["usetovariant"]; !s.HasTrigger {
		t.Errorf("usetovariant = %+v, want HasTrigger (the \"use to\" alternative must count)", s)
	}
	if s := got["nodesc"]; s.HasDescription {
		t.Errorf("nodesc = %+v, want HasDescription=false", s)
	} else if !s.HasTrigger {
		t.Errorf("nodesc = %+v: the description axis must be independent of the trigger axis", s)
	}
	if s := got["notrigger"]; s.HasTrigger {
		t.Errorf("notrigger = %+v, want HasTrigger=false", s)
	} else if !s.HasDescription {
		t.Errorf("notrigger = %+v: the trigger axis must be independent of the description axis", s)
	}
	// MetadataWordBudget+1 words on `description:` -> over budget; the body stays tiny.
	if s := got["fatmeta"]; !s.OverMetadataBudget() {
		t.Errorf("fatmeta = %+v (MetaWords=%d), want over the %d-word metadata budget", s, s.MetaWords, MetadataWordBudget)
	} else if s.OverBodyBudget() {
		t.Errorf("fatmeta = %+v: a fat description must not spill into the body tier", s)
	}
	if s := got["fatbody"]; !s.OverBodyBudget() {
		t.Errorf("fatbody = %+v (BodyWords=%d), want over the %d-word body budget", s, s.BodyWords, BodyWordBudget)
	} else if s.OverMetadataBudget() {
		t.Errorf("fatbody = %+v: a fat body must not spill into the metadata tier", s)
	}
}

// TestTierWordCountsSplitsTheTwoLoadTiers pins the split itself: the always-resident tier is
// EXACTLY the frontmatter description value (not the whole frontmatter, not the whole file),
// and the fault-on-demand tier is everything after the closing fence. A file with no fence is
// all body, and CRLF must not change either count.
func TestTierWordCountsSplitsTheTwoLoadTiers(t *testing.T) {
	doc := "---\nname: x\ndescription: one two three\ntags: [a, b, c, d, e]\n---\n\nalpha beta\n"
	meta, body := TierWordCounts(doc)
	if meta != 3 {
		t.Errorf("meta words = %d, want 3 (the description value only -- not name/tags)", meta)
	}
	if body != 2 {
		t.Errorf("body words = %d, want 2 (everything after the closing fence)", body)
	}

	crlfMeta, crlfBody := TierWordCounts(strings.ReplaceAll(doc, "\n", "\r\n"))
	if crlfMeta != meta || crlfBody != body {
		t.Errorf("CRLF counts = (%d,%d), want (%d,%d) -- line endings must not move the tiers", crlfMeta, crlfBody, meta, body)
	}

	noFence := "# plain\n\nfive words in this body\n" // 7 whitespace-separated fields incl. the "#"
	if m, b := TierWordCounts(noFence); m != 0 || b != 7 {
		t.Errorf("un-fenced file = (%d,%d), want (0,7) -- no frontmatter means all body", m, b)
	}
}

// TestFoldPinsCorpusCountersAndDebtSplit is the live-floor sentinel: it folds a synthetic
// corpus with a KNOWN gap in every dimension and pins the exact published integers.
//
//	skill_debt   == the AFFORDANCE dimension only (unreadable + no-description + no-trigger)
//	total_debt   == Σ len(kpi.Defects) across every KPI
//	loader_debt  == queryable + pages + in_sync
//
// Conflating the affordance headline with the total, dropping a loader component from the
// fold, or letting a budget violation leak into skill_debt all red this test.
func TestFoldPinsCorpusCountersAndDebtSplit(t *testing.T) {
	skills := []Scanned{
		{Name: "clean", Readable: true, HasDescription: true, HasTrigger: true, MetaWords: 10, BodyWords: 10},
		{Name: "unreadable"}, // +1 affordance
		{Name: "nodesc", Readable: true, HasTrigger: true, MetaWords: 10, BodyWords: 10},                                       // +1 affordance
		{Name: "notrigger", Readable: true, HasDescription: true, MetaWords: 10, BodyWords: 10},                                // +1 affordance
		{Name: "fatmeta", Readable: true, HasDescription: true, HasTrigger: true, MetaWords: MetadataWordBudget + 1},           // +1 metadata_budget
		{Name: "fatbody", Readable: true, HasDescription: true, HasTrigger: true, MetaWords: 1, BodyWords: BodyWordBudget + 1}, // +1 body_budget
	}
	loader := Loader{Skills: 6, Queryable: 2, Pages: 1, InSync: 3}

	p := Fold("/repo", skills, loader)
	c := p.Corpus

	want := map[string]int{
		DebtKey:            3, // affordance only
		"metadata_budget":  1,
		"body_budget":      1,
		"loader_queryable": 2,
		"loader_pages":     1,
		"loader_in_sync":   3,
		"loader_debt":      6, // 2+1+3
		"skills":           6,
		TotalDebtKey:       11, // 3 affordance + 2 budget + 6 loader
	}
	for k, v := range want {
		if got, ok := c[k].(int); !ok || got != v {
			t.Errorf("corpus[%q] = %v, want int %d", k, c[k], v)
		}
	}

	// total_debt must be the kernel's own Σ-defects, not a separately-maintained tally.
	sum := 0
	for _, k := range p.KPIs {
		sum += len(k.Defects)
	}
	if sum != want[TotalDebtKey] {
		t.Errorf("Σ kpi defects = %d, want %d -- every debt unit must be a nameable defect", sum, want[TotalDebtKey])
	}

	if p.OK || p.Verdict != "ACTION" || p.Finding != "skill_debt" {
		t.Errorf("dirty fold = ok:%v verdict:%q finding:%q, want false/ACTION/skill_debt", p.OK, p.Verdict, p.Finding)
	}
	// The reason line is the operator-facing split; it names the affordance and loader
	// dimensions separately, plus the budget tail only when a budget actually blew.
	wantReason := "3 skill affordance + 6 loader debt unit(s) + 2 word-budget violation(s)"
	if p.Reason != wantReason {
		t.Errorf("reason = %q, want %q", p.Reason, wantReason)
	}
	if p.Workspace != "/repo" {
		t.Errorf("workspace = %q, want /repo", p.Workspace)
	}
}

// TestFoldCleanCorpusIsOKAndPerfect pins the zero-defect floor: a clean tree grades A at a
// value of 1.0, reports OK, and carries zero in every debt counter. A KPI that scored an
// empty corpus at 0 instead of 100 (the classic divide-by-zero regression) reds here.
func TestFoldCleanCorpusIsOKAndPerfect(t *testing.T) {
	skills := []Scanned{
		{Name: "a", Readable: true, HasDescription: true, HasTrigger: true, MetaWords: 5, BodyWords: 50},
		{Name: "b", Readable: true, HasDescription: true, HasTrigger: true, MetaWords: 5, BodyWords: 50},
	}
	p := Fold("/repo", skills, Loader{Skills: 2})
	if !p.OK || p.Verdict != "OK" || p.Finding != "skills_effective" {
		t.Fatalf("clean fold = ok:%v verdict:%q finding:%q, want true/OK/skills_effective", p.OK, p.Verdict, p.Finding)
	}
	if g := p.Corpus["grade"]; g != "A" {
		t.Errorf("clean grade = %v, want A", g)
	}
	if v := p.Corpus["value"]; v != 1.0 {
		t.Errorf("clean value = %v, want 1", v)
	}
	for _, k := range []string{DebtKey, TotalDebtKey, "loader_debt", "metadata_budget", "body_budget"} {
		if n, _ := p.Corpus[k].(int); n != 0 {
			t.Errorf("clean corpus[%q] = %v, want 0", k, p.Corpus[k])
		}
	}
	// An empty tree must also read clean, not catastrophically broken.
	empty := Fold("/repo", nil, Loader{})
	if !empty.OK || empty.Corpus["grade"] != "A" {
		t.Errorf("empty tree = ok:%v grade:%v, want true/A", empty.OK, empty.Corpus["grade"])
	}
}

// TestBuildIsDeterministicOverARealTree pins the end-to-end read: Build over the same tree
// twice is byte-identical (no map-iteration order, no clock, no network), and the counters
// it derives from disk match the ones Fold derives from the same probe.
func TestBuildIsDeterministicOverARealTree(t *testing.T) {
	root := t.TempDir()
	writeSkill(t, root, "alpha", frontmatter("alpha", "Use when you need alpha", "alpha body text"))
	writeSkill(t, root, "beta", frontmatter("beta", "A beta skill with no trigger phrasing", "beta body text"))

	a, b := Build(root), Build(root)
	ja, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Fatalf("Build is not deterministic:\n%s\n%s", ja, jb)
	}
	if a.Schema != Schema {
		t.Errorf("schema = %q, want %q", a.Schema, Schema)
	}
	if n, _ := a.Corpus["skills"].(int); n != 2 {
		t.Errorf("skills = %v, want 2", a.Corpus["skills"])
	}
	// beta carries a description but no "use when"/"use to" -> exactly one affordance unit.
	if n, _ := a.Corpus[DebtKey].(int); n != 1 {
		t.Errorf("%s = %v, want 1 (beta has no trigger)", DebtKey, a.Corpus[DebtKey])
	}

	// The same probe folded by hand must agree with Build -- the shell and the core cannot drift.
	manual := Fold(root, Scan(root), ScanLoader(root))
	jm, _ := json.Marshal(manual)
	if string(jm) != string(ja) {
		t.Fatalf("Build != Fold(Scan, ScanLoader):\n%s\n%s", ja, jm)
	}
}

// TestAffordanceDebtCountsEachAxisOnce pins the counting rule the headline depends on: an
// unreadable skill is ONE unit (its other axes are unprobeable, not extra debt), while a
// readable skill missing both a description and a trigger is TWO.
func TestAffordanceDebtCountsEachAxisOnce(t *testing.T) {
	if n := AffordanceDebt([]Scanned{{Name: "gone"}}); n != 1 {
		t.Errorf("unreadable skill = %d affordance unit(s), want 1", n)
	}
	if n := AffordanceDebt([]Scanned{{Name: "bare", Readable: true}}); n != 2 {
		t.Errorf("readable skill missing description+trigger = %d unit(s), want 2", n)
	}
	// A budget violation is NOT affordance debt -- it has its own dimension.
	fat := Scanned{Name: "fat", Readable: true, HasDescription: true, HasTrigger: true,
		MetaWords: MetadataWordBudget + 1, BodyWords: BodyWordBudget + 1}
	if n := AffordanceDebt([]Scanned{fat}); n != 0 {
		t.Errorf("budget-only violation = %d affordance unit(s), want 0", n)
	}
	if m, b := BudgetDebt([]Scanned{fat}); m != 1 || b != 1 {
		t.Errorf("BudgetDebt = (%d,%d), want (1,1)", m, b)
	}
}

func TestHasTriggerAcceptsNaturalDiscoveryPhrases(t *testing.T) {
	for _, description := range []string{
		"Use for recurring dispatch work.",
		"Use after a release changes the front door.",
		"Use before launching workers.",
		"Run when the queue drains.",
		"Invoke when an external implementation is relevant.",
		"Triggers when the operator asks for a score.",
	} {
		if !hasTrigger(strings.ToLower(description)) {
			t.Errorf("hasTrigger(%q) = false", description)
		}
	}
	if hasTrigger("A workflow with no invocation cue.") {
		t.Fatal("hasTrigger accepted a description without a trigger")
	}
}
