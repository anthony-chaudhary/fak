package quality

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"testing/fstest"
)

// TestSplitCorpusWitnessesPlantedTierDefect is the captured #4574 witness, replayed
// from bytes on disk rather than structs assembled in memory: the committed corpus
// under testdata/suite/defect plants one representative defect — a corpora replay
// case (3000s wall) mis-declared as a per-PR check — and the split REFUSES it,
// naming the first budget it broke. testdata/suite/fixed is the same corpus with
// that one case routed to the nightly tier it can afford, and it splits clean.
//
// Fail-then-pass, from committed bytes, with no ambient state: the two directories
// are the before/after of one fix, so a clean checkout replays this independently.
func TestSplitCorpusWitnessesPlantedTierDefect(t *testing.T) {
	fsys := os.DirFS("testdata/suite")

	planted, err := SplitCorpus(fsys, "defect", nil)
	if err != nil {
		t.Fatalf("defect corpus unreadable (infrastructure failure, not a verdict): %v", err)
	}
	ok, why := planted.Green()
	if ok {
		t.Fatalf("planted defect corpus reported GREEN — an expensive case rode the PR lane:\n%s", ExplainPlan(planted))
	}
	// The refusal must localize the FIRST actionable divergence: which case, which
	// budget it broke, and where it belongs — not a bare "too expensive".
	for _, want := range []string{"corpora-replay", "3000s", "tier pr budget of 120s", "route to a slower tier"} {
		if !strings.Contains(why, want) {
			t.Fatalf("refusal did not localize the defect (missing %q): %q", want, why)
		}
	}
	// Refused means NOT PLACED. A mislabeled case must never appear in any suite.
	for _, s := range planted.Suites {
		for _, sc := range s.Cases {
			if sc.CaseID == "corpora-replay" {
				t.Fatalf("refused case was still placed in the %s suite", s.Tier)
			}
		}
	}
	// One bad case does not sink the corpus: the other four families still route.
	if placed := placedCount(planted); placed != 4 {
		t.Fatalf("defect corpus placed %d clean cases, want 4:\n%s", placed, ExplainPlan(planted))
	}

	fixed, err := SplitCorpus(fsys, "fixed", nil)
	if err != nil {
		t.Fatalf("fixed corpus unreadable: %v", err)
	}
	if ok, why := fixed.Green(); !ok {
		t.Fatalf("corpus is not green after the fix: %s\n%s", why, ExplainPlan(fixed))
	}

	// After the fix every evidence family lands in the tier its cost can afford:
	// cheap deterministic evidence gates every push, sampling and corpora replay
	// go nightly, and accelerator/review qualification waits for release.
	want := map[string]struct {
		tier   Tier
		family EvidenceFamily
	}{
		"det-greedy-cpu": {TierPR, FamilyDeterministic},
		"stats-topp":     {TierNightly, FamilyStatistics},
		"corpora-replay": {TierNightly, FamilyCorpora},
		"gpu-parity-f16": {TierRelease, FamilyGPUParity},
		"review-rubric":  {TierRelease, FamilyReview},
	}
	got := map[string]SuiteCase{}
	for _, s := range fixed.Suites {
		for _, sc := range s.Cases {
			if sc.Tier != s.Tier {
				t.Fatalf("case %s carries tier %s but sits in the %s suite", sc.CaseID, sc.Tier, s.Tier)
			}
			got[sc.CaseID] = sc
		}
	}
	if len(got) != len(want) {
		t.Fatalf("fixed corpus placed %d cases, want %d:\n%s", len(got), len(want), ExplainPlan(fixed))
	}
	for id, w := range want {
		sc, placed := got[id]
		if !placed {
			t.Fatalf("case %s was not placed after the fix", id)
		}
		if sc.Tier != w.tier || sc.Family != w.family {
			t.Fatalf("case %s routed to tier=%s family=%s, want tier=%s family=%s",
				id, sc.Tier, sc.Family, w.tier, w.family)
		}
		// Every placed case documents its runtime and resource cost and names an owner.
		if sc.Owner == "" || sc.Cost.RuntimeSeconds <= 0 || sc.Cost.TimeoutSeconds <= 0 ||
			sc.Cost.CPU <= 0 || sc.Cost.MemoryMiB <= 0 {
			t.Fatalf("case %s placed without a complete owner/cost header: %+v", id, sc)
		}
	}
	// Only the release tier may buy accelerators; the PR and nightly lanes stay CPU-only.
	for _, s := range fixed.Suites {
		if s.Tier != TierRelease && s.MaxAccelerators != 0 {
			t.Fatalf("tier %s admitted %d accelerator(s) — cheap lanes are CPU-only", s.Tier, s.MaxAccelerators)
		}
	}
	if suiteFor(fixed, TierRelease).MaxAccelerators != 8 {
		t.Fatalf("release suite lost the accelerator requirement: %+v", suiteFor(fixed, TierRelease))
	}
}

// TestSplitCorpusPlanIsScrubbed proves the emitted replay artifact is scrubbed by
// construction: a SuitePlan carries only routing headers — case id, family, owner,
// tier, cost — so it can be published to an operator or a CI log without leaking
// the prompts, reference completions, or rubric phrases the corpus decodes. It
// stays replay-complete: the case id resolves back to the committed fixture.
func TestSplitCorpusPlanIsScrubbed(t *testing.T) {
	plan, err := SplitCorpus(os.DirFS("testdata/suite"), "fixed", nil)
	if err != nil {
		t.Fatalf("fixed corpus unreadable: %v", err)
	}
	blob, err := json.Marshal(plan)
	if err != nil {
		t.Fatalf("suite plan does not marshal: %v", err)
	}
	artifact := string(blob)
	// Payload text that must NOT ride along in the artifact.
	for _, secret := range []string{
		"Answer every question in the replay corpus shard.",
		"Return the canonical answer.",
		"Decode the parity probe under f16",
		"Sample one color from the fixture palette.",
		"The run shipped with one unresolved divergence.",
		"no issues found",
	} {
		if strings.Contains(artifact, secret) {
			t.Fatalf("suite plan leaked case payload %q", secret)
		}
	}
	// Routing evidence that MUST ride along, or the artifact cannot be replayed.
	for _, want := range []string{SuitePlanSchema, "corpora-replay", "corpora-team", "timeout_seconds"} {
		if !strings.Contains(artifact, want) {
			t.Fatalf("suite plan dropped replay evidence %q", want)
		}
	}
}

// TestSplitCorpusRefusesBrokenAndAmbiguousFiles proves the corpus loader is
// fail-closed in the two ways a directory of files can quietly lose evidence: a
// file that does not load is REFUSED by name rather than skipped (a silently
// skipped case stops being checked with nobody noticing), and two files claiming
// one case id are refused as ambiguous rather than racing on directory order.
func TestSplitCorpusRefusesBrokenAndAmbiguousFiles(t *testing.T) {
	good := mustMarshal(t, validCase("det-1", "pr", "deterministic", cost(1, 20, 1, 32, 0)))
	dup := mustMarshal(t, validCase("det-1", "nightly", "statistics", cost(2, 40, 1, 64, 0)))
	fsys := fstest.MapFS{
		"corpus/a-det.json":     {Data: good},
		"corpus/b-dup.json":     {Data: dup},
		"corpus/c-broken.json":  {Data: []byte("{not json")},
		"corpus/d-notacase.txt": {Data: []byte("ignored: not a .json case")},
	}

	plan, err := SplitCorpus(fsys, "corpus", nil)
	if err != nil {
		t.Fatalf("corpus directory readable but SplitCorpus errored: %v", err)
	}
	if rej := rejectFor(plan, "c-broken.json"); rej == nil || !strings.Contains(rej.Reason, "unloadable") {
		t.Fatalf("malformed case file was not refused by name: %+v", plan.Rejected)
	}
	if rej := rejectFor(plan, "det-1"); rej == nil || !strings.Contains(rej.Reason, "duplicate case id") {
		t.Fatalf("duplicate case id was not refused: %+v", plan.Rejected)
	}
	if ok, _ := plan.Green(); ok {
		t.Fatal("a corpus with a refused case reported GREEN")
	}
	// The first-declared case still routes — refusal is per case, not per corpus.
	if pr := suiteFor(plan, TierPR).Cases; len(pr) != 1 || pr[0].CaseID != "det-1" {
		t.Fatalf("clean case did not survive its broken neighbours: %+v", pr)
	}
}

// TestSplitCorpusEmptyIsNotGreen proves a corpus that qualifies nothing is not a
// pass. An empty directory rejects nothing, so the only thing standing between it
// and a green verdict is Green's placed-evidence floor — missing evidence is never
// a pass. An unreadable DIRECTORY is an infrastructure error, never a verdict.
func TestSplitCorpusEmptyIsNotGreen(t *testing.T) {
	fsys := fstest.MapFS{"corpus/README.md": {Data: []byte("no cases here yet")}}
	plan, err := SplitCorpus(fsys, "corpus", nil)
	if err != nil {
		t.Fatalf("readable empty corpus errored: %v", err)
	}
	if len(plan.Rejected) != 0 {
		t.Fatalf("empty corpus invented rejections: %+v", plan.Rejected)
	}
	ok, why := plan.Green()
	if ok {
		t.Fatal("empty corpus reported GREEN — a split that qualifies nothing is not evidence")
	}
	if !strings.Contains(why, "no case placed") {
		t.Fatalf("empty-corpus refusal reason not actionable: %q", why)
	}

	if _, err := SplitCorpus(fsys, "absent", nil); err == nil {
		t.Fatal("missing corpus directory returned a verdict instead of an infrastructure error")
	}
}

func placedCount(p SuitePlan) int {
	n := 0
	for _, s := range p.Suites {
		n += len(s.Cases)
	}
	return n
}

func mustMarshal(t *testing.T, c QualityCase) []byte {
	t.Helper()
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal case %s: %v", c.ID, err)
	}
	return b
}
