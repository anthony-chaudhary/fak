package armbench

import (
	"context"
	"strings"
	"testing"
)

// TestSelfcheckPasses is the spine witness: the deterministic fake-provider run
// plus every fail-closed proof #6676 names, all green in one artifact.
func TestSelfcheckPasses(t *testing.T) {
	res, err := Selfcheck()
	if err != nil {
		t.Fatalf("Selfcheck: %v", err)
	}
	for _, c := range res.Checks {
		if !c.OK {
			t.Errorf("check %s failed: %s (%s)", c.Name, c.Detail, c.Evidence)
		}
	}
	if !res.OK {
		t.Fatalf("selfcheck reported not ok")
	}
	if len(res.Checks) < 12 {
		t.Fatalf("selfcheck ran only %d checks; the acceptance list is longer than that", len(res.Checks))
	}
	if !strings.Contains(res.Report, "IN_TOK") || !strings.Contains(res.Report, "OUT_TOK") {
		t.Errorf("captured report does not keep input and output tokens in separate columns:\n%s", res.Report)
	}
}

func TestSelfcheckMissingEvidenceWitnessIsStable(t *testing.T) {
	const want = "arm fak-ctxmmu task task-charlie trial 0: MISSING_RAW_EVIDENCE"
	for i := 0; i < 5; i++ {
		res, err := Selfcheck()
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, check := range res.Checks {
			if check.Name == "failclosed.missing_raw_evidence" {
				found = true
				if !strings.Contains(check.Evidence, want) {
					t.Fatalf("run %d evidence = %q, want stable %q", i, check.Evidence, want)
				}
			}
		}
		if !found {
			t.Fatal("missing failclosed.missing_raw_evidence check")
		}
	}
}

// TestSpineRunsAllFourArmKinds covers baseline + upstream treatment + fak
// passthrough + one isolated fak capability end to end on the fake provider.
func TestSpineRunsAllFourArmKinds(t *testing.T) {
	m, corpus := DemoManifest(), DemoCorpus()
	run, err := Execute(context.Background(), m, corpus, &FakeProvider{SetupWallMS: 1000}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	rep, err := Summarize(run)
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	want := map[ArmKind]bool{ArmBaseline: false, ArmUpstreamTreatment: false, ArmFakPassthrough: false, ArmFakCapability: false}
	for _, a := range rep.Arms {
		if a.Trials != len(corpus)*m.Trials.Count {
			t.Errorf("arm %s ran %d trials, want %d", a.ArmID, a.Trials, len(corpus)*m.Trials.Count)
		}
		if a.InputTokens == 0 || a.OutputTokens == 0 {
			t.Errorf("arm %s captured no usage: in=%d out=%d", a.ArmID, a.InputTokens, a.OutputTokens)
		}
		if a.TotalTokens != a.InputTokens+a.OutputTokens {
			t.Errorf("arm %s total %d != in+out %d", a.ArmID, a.TotalTokens, a.InputTokens+a.OutputTokens)
		}
		want[a.Kind] = true
	}
	for kind, seen := range want {
		if !seen {
			t.Errorf("arm kind %s never ran", kind)
		}
	}
	if rep.FailureCount != 0 {
		t.Errorf("spine reported %d failures, want 0", rep.FailureCount)
	}
	for _, row := range run.Trials {
		if row.Response.RawRequest == "" || row.Response.RawResponse == "" || row.Judgment.RawJudgment == "" {
			t.Errorf("row %s is missing raw request/response/judgment evidence", row.Key())
		}
		if row.Response.Latency.WallMS <= 0 || !row.Response.Latency.TTFTAvailable || !row.Response.Latency.InterTokenAvailable {
			t.Errorf("row %s did not capture available wall/TTFT/inter-token latency: %+v", row.Key(), row.Response.Latency)
		}
		if row.Response.Cache.Hits+row.Response.Cache.Misses == 0 {
			t.Errorf("row %s did not capture a cache counter", row.Key())
		}
	}
}

// TestRunIsDeterministic pins the reproducibility the whole design rests on: the
// same manifest and corpus produce the same ledger, whatever order the
// concurrent units happened to finish in.
func TestRunIsDeterministic(t *testing.T) {
	a, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("run a: %v", err)
	}
	b, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("run b: %v", err)
	}
	ja, err := MarshalRun(a)
	if err != nil {
		t.Fatalf("marshal a: %v", err)
	}
	jb, err := MarshalRun(b)
	if err != nil {
		t.Fatalf("marshal b: %v", err)
	}
	if string(ja) != string(jb) {
		t.Fatalf("two runs of the same manifest produced different ledgers")
	}
}

// TestMissingRawEvidenceFailsClosed proves a usage row with nothing behind it
// aborts the run rather than being reported with a gap.
func TestMissingRawEvidenceFailsClosed(t *testing.T) {
	_, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{OmitRawResponse: "fak-ctxmmu"}, &FakeGrader{}, Options{})
	requireReason(t, err, ReasonMissingRawEvidence)
}

// TestMissingRawJudgmentFailsClosed does the same for the grader half: a
// pass/fail with no judge evidence is not evidence either.
func TestMissingRawJudgmentFailsClosed(t *testing.T) {
	_, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{}, &blankGrader{}, Options{})
	requireReason(t, err, ReasonMissingRawEvidence)
}

type blankGrader struct{}

func (blankGrader) Grade(context.Context, Request, Response) (Judgment, error) {
	return Judgment{Pass: true, Score: 1}, nil // no RawJudgment
}

// TestSummarizeRefusesUnevidencedLedger proves the fence is re-asked at publish
// time, so a hand-edited run artifact cannot be summarized into a report.
func TestSummarizeRefusesUnevidencedLedger(t *testing.T) {
	run, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	run.Trials[0].Response.RawResponse = ""
	if _, err := Summarize(run); !isReason(err, ReasonMissingRawEvidence) {
		t.Fatalf("Summarize accepted an unevidenced row, got err=%v", err)
	}
}

// TestSummarizeRefusesEditedManifest proves a manifest edited after the run
// cannot be published under the run's old identity.
func TestSummarizeRefusesEditedManifest(t *testing.T) {
	run, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	run.Manifest.Model.Snapshot = "fake-deterministic-2026-12-01"
	if _, err := Summarize(run); !isReason(err, ReasonIncomparableManifest) {
		t.Fatalf("Summarize accepted a post-hoc manifest edit, got err=%v", err)
	}
}

// TestBundledCapabilityFailsClosed is the "bundle only after isolated arms" law
// from the epic's experiment law, enforced at validation.
func TestBundledCapabilityFailsClosed(t *testing.T) {
	m := DemoManifest()
	for i := range m.Arms {
		if m.Arms[i].Kind == ArmFakCapability {
			m.Arms[i].Capabilities = []string{"ctxmmu-paging", "toon-encoding"}
		}
	}
	requireReason(t, m.Validate(), ReasonArmCapabilityBundled)
}

// TestUnnamedCapabilityFailsClosed covers both halves of the same fence: a
// capability arm naming nothing, and a non-capability arm naming something.
func TestUnnamedCapabilityFailsClosed(t *testing.T) {
	empty := DemoManifest()
	for i := range empty.Arms {
		if empty.Arms[i].Kind == ArmFakCapability {
			empty.Arms[i].Capabilities = nil
		}
	}
	requireReason(t, empty.Validate(), ReasonArmCapabilityUnnamed)

	smuggled := DemoManifest()
	for i := range smuggled.Arms {
		if smuggled.Arms[i].Kind == ArmFakPassthrough {
			smuggled.Arms[i].Capabilities = []string{"ctxmmu-paging"}
		}
	}
	requireReason(t, smuggled.Validate(), ReasonArmCapabilityUnnamed)
}

// TestValidateRequiresEveryPin walks the provenance fields one at a time. Each
// row is a term that, left unpinned, makes a published number unreproducible.
func TestValidateRequiresEveryPin(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Manifest)
		reason string
	}{
		{"schema", func(m *Manifest) { m.Schema = "fak.armbench.manifest/0" }, ReasonManifestInvalid},
		{"model snapshot", func(m *Manifest) { m.Model.Snapshot = "" }, ReasonManifestInvalid},
		{"model region", func(m *Manifest) { m.Model.Region = "" }, ReasonManifestInvalid},
		{"max tokens", func(m *Manifest) { m.Model.MaxTokens = 0 }, ReasonManifestInvalid},
		{"corpus hash", func(m *Manifest) { m.Corpus.Hash = "" }, ReasonManifestInvalid},
		{"judge hash", func(m *Manifest) { m.Judge.Hash = "" }, ReasonManifestInvalid},
		{"trial count", func(m *Manifest) { m.Trials.Count = 0 }, ReasonManifestInvalid},
		{"order strategy", func(m *Manifest) { m.Trials.Order = "whatever" }, ReasonManifestInvalid},
		{"unseeded randomization", func(m *Manifest) { m.Trials.Order = OrderRandomized; m.Trials.Seed = 0 }, ReasonManifestInvalid},
		{"concurrency", func(m *Manifest) { m.Trials.Concurrency = 0 }, ReasonManifestInvalid},
		{"pricing date", func(m *Manifest) { m.Environment.PricingDate = "" }, ReasonManifestInvalid},
		{"source sha", func(m *Manifest) { m.Sources[0].SHA = "" }, ReasonManifestInvalid},
		{"source content hash", func(m *Manifest) { m.Sources[0].ContentHash = "" }, ReasonManifestInvalid},
		{"arm prompt hash", func(m *Manifest) { m.Arms[0].PromptHash = "" }, ReasonManifestInvalid},
		{"unbound upstream arm", func(m *Manifest) { m.Arms[1].SourceName = "" }, ReasonManifestInvalid},
		{"dangling source name", func(m *Manifest) { m.Arms[1].SourceName = "nope" }, ReasonManifestInvalid},
		{"missing control", func(m *Manifest) { m.Arms[0].Kind = ArmFakPassthrough }, ReasonManifestInvalid},
		{"single arm", func(m *Manifest) { m.Arms = m.Arms[:1] }, ReasonManifestInvalid},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := DemoManifest()
			tc.mutate(m)
			requireReason(t, m.Validate(), tc.reason)
		})
	}
	if err := DemoManifest().Validate(); err != nil {
		t.Fatalf("the unmutated demo manifest must validate, got %v", err)
	}
}

// TestIdentityMovesOnEveryPinnedTerm is the acceptance criterion "a changed
// model, prompt, judge, corpus, or capability changes the manifest identity".
func TestIdentityMovesOnEveryPinnedTerm(t *testing.T) {
	base := DemoManifest().Identity()
	for _, mut := range identityMutations() {
		t.Run(mut.name, func(t *testing.T) {
			m := DemoManifest()
			mut.apply(m)
			if got := m.Identity(); got == base {
				t.Fatalf("mutating the %s left the identity at %s", mut.name, got)
			}
		})
	}
	if DemoManifest().Identity() != base {
		t.Fatal("identity is not stable across two constructions of the same manifest")
	}
}

// TestIdentityIncludesSchedulingAndEnvironment proves the manifest identity
// covers every recorded execution term, not just the five semantic mutations
// called out explicitly by #6676.
func TestIdentityIncludesSchedulingAndEnvironment(t *testing.T) {
	base := DemoManifest().Identity()
	m := DemoManifest()
	m.Trials.Concurrency = 16
	if m.Identity() == base {
		t.Fatal("concurrency did not change the immutable-manifest identity")
	}
	m = DemoManifest()
	m.Environment.HostClass = "ci"
	if m.Identity() == base {
		t.Fatal("environment did not change the immutable-manifest identity")
	}
}

// TestIdentityCannotBeForgedByFieldContent proves a value cannot smuggle a field
// boundary into the canonical stream and collide two different manifests.
func TestIdentityCannotBeForgedByFieldContent(t *testing.T) {
	a := DemoManifest()
	a.Corpus.ID = "x\x1fcorpus.hash\x1fsha256:c0rpu5"
	a.Corpus.Hash = ""
	b := DemoManifest()
	if a.Identity() == b.Identity() {
		t.Fatal("a separator-bearing value forged the same identity as the real manifest")
	}
}

// TestResumeDoesNotDuplicateTrials is the "resume without silently duplicating
// trials" criterion.
func TestResumeDoesNotDuplicateTrials(t *testing.T) {
	ctx := context.Background()
	first, err := Execute(ctx, DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	second, err := Execute(ctx, DemoManifest(), DemoCorpus(), &FakeProvider{}, &countingGrader{}, Options{Resume: first})
	if err != nil {
		t.Fatalf("resume run: %v", err)
	}
	if second.Executed != 0 {
		t.Errorf("resume re-executed %d trial(s), want 0", second.Executed)
	}
	if len(second.Trials) != len(first.Trials) {
		t.Errorf("resume produced %d rows, want the original %d", len(second.Trials), len(first.Trials))
	}
	if dupes := duplicateKeys(second.Trials); len(dupes) != 0 {
		t.Errorf("resume duplicated %d trial(s): %v", len(dupes), dupes)
	}
	for _, tr := range second.Trials {
		if !tr.Resumed {
			t.Errorf("row %s is not marked resumed, so a resumed report could be read as a fresh run", tr.Key())
		}
	}
}

// TestResumeRerunsFailedTrials proves a transient provider failure is not frozen
// into the published result by the resume path.
func TestResumeRerunsFailedTrials(t *testing.T) {
	ctx := context.Background()
	first, err := Execute(ctx, DemoManifest(), DemoCorpus(), &failingProvider{arm: "fak-ctxmmu"}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	failed := 0
	for _, tr := range first.Trials {
		if tr.Response.Failure != "" {
			failed++
		}
	}
	if failed == 0 {
		t.Fatal("fixture produced no failed trials")
	}
	second, err := Execute(ctx, DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{Resume: first})
	if err != nil {
		t.Fatalf("resume run: %v", err)
	}
	if second.Executed != failed {
		t.Errorf("resume re-executed %d trial(s), want exactly the %d failed one(s)", second.Executed, failed)
	}
	for _, tr := range second.Trials {
		if tr.Response.Failure != "" {
			t.Errorf("row %s is still failed after a clean resume", tr.Key())
		}
	}
}

// TestResumeRefusesForeignLedger proves a ledger from a different experiment
// cannot be silently mixed in.
func TestResumeRefusesForeignLedger(t *testing.T) {
	ctx := context.Background()
	first, err := Execute(ctx, DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	drift := DemoManifest()
	drift.Model.Snapshot = "fake-deterministic-2026-09-01"
	_, err = Execute(ctx, drift, DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{Resume: first})
	requireReason(t, err, ReasonResumeIdentityMismatch)
}

// TestResumeRefusesEditedPairPosition closes a subtler resume hole: position is
// not part of the duplicate key, but it is part of the randomized/counterbalanced
// plan. A ledger cannot claim an arm ran in another position and be carried.
func TestResumeRefusesEditedPairPosition(t *testing.T) {
	ctx := context.Background()
	first, err := Execute(ctx, DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	first.Trials[0].Position = (first.Trials[0].Position + 1) % len(first.Manifest.Arms)
	_, err = Execute(ctx, DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{Resume: first})
	requireReason(t, err, ReasonResumeIdentityMismatch)
}

// TestCheckComparableRejectsDrift covers the "reject incomparable manifests"
// criterion, including the point that differing ARMS are the legitimate case.
func TestCheckComparableRejectsDrift(t *testing.T) {
	if _, err := CheckComparable(DemoManifest(), DemoManifest()); err != nil {
		t.Fatalf("two identical manifests must be comparable, got %v", err)
	}
	differentArms := DemoManifest()
	differentArms.Arms = differentArms.Arms[:2]
	if fields, err := CheckComparable(DemoManifest(), differentArms); !isReason(err, ReasonIncomparableManifest) || len(fields) == 0 {
		t.Fatalf("a different arm contract must be incomparable, fields=%v err=%v", fields, err)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*Manifest)
		field  string
	}{
		{"model", func(m *Manifest) { m.Model.Snapshot = "other" }, "model.snapshot"},
		{"sampling", func(m *Manifest) { m.Model.Sampling.Temperature = 0.7 }, "model.sampling.temperature"},
		{"corpus", func(m *Manifest) { m.Corpus.Hash = fixtureDigest("other corpus") }, "corpus.hash"},
		{"judge", func(m *Manifest) { m.Judge.Hash = fixtureDigest("other judge") }, "judge.hash"},
		{"trials", func(m *Manifest) { m.Trials.Count = 9 }, "trials.count"},
		{"order seed", func(m *Manifest) { m.Trials.Seed = 99 }, "trials.seed"},
		{"region", func(m *Manifest) { m.Model.Region = "other-region" }, "model.region"},
		{"environment", func(m *Manifest) { m.Environment.HostClass = "other-host" }, "environment.host_class"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := DemoManifest()
			tc.mutate(b)
			fields, err := CheckComparable(DemoManifest(), b)
			requireReason(t, err, ReasonIncomparableManifest)
			found := false
			for _, f := range fields {
				if f.Field == tc.field {
					found = true
				}
			}
			if !found {
				t.Errorf("refusal did not name %s; got %v", tc.field, fields)
			}
		})
	}
}

// TestPairedOrderIsCounterbalanced proves no arm is systematically advantaged by
// always going first, under both order strategies.
func TestPairedOrderIsCounterbalanced(t *testing.T) {
	for _, order := range []OrderStrategy{OrderCounterbalanced, OrderRandomized} {
		t.Run(string(order), func(t *testing.T) {
			m := DemoManifest()
			m.Trials.Order = order
			m.Trials.Count = 8
			units := PlanUnits(m, DemoCorpus())
			firsts := map[string]int{}
			for _, u := range units {
				if len(u.ArmOrder) != len(m.Arms) {
					t.Fatalf("unit %s/%d planned %d arms, want %d", u.TaskID, u.Trial, len(u.ArmOrder), len(m.Arms))
				}
				seen := map[string]bool{}
				for _, id := range u.ArmOrder {
					if seen[id] {
						t.Fatalf("unit %s/%d plans arm %s twice", u.TaskID, u.Trial, id)
					}
					seen[id] = true
				}
				firsts[u.ArmOrder[0]]++
			}
			if len(firsts) != len(m.Arms) {
				t.Fatalf("only %d of %d arms ever went first: %v", len(firsts), len(m.Arms), firsts)
			}
		})
	}
}

// TestPlanUnitsIsReproducible proves the randomized order is seeded, not
// irreproducible, and that a different seed genuinely reorders.
func TestPlanUnitsIsReproducible(t *testing.T) {
	m := DemoManifest()
	m.Trials.Order = OrderRandomized
	a := PlanUnits(m, DemoCorpus())
	b := PlanUnits(m, DemoCorpus())
	for i := range a {
		if strings.Join(a[i].ArmOrder, ",") != strings.Join(b[i].ArmOrder, ",") {
			t.Fatalf("unit %d differs between two plans of the same seeded manifest", i)
		}
	}
	m.Trials.Seed = 12345
	c := PlanUnits(m, DemoCorpus())
	same := true
	for i := range a {
		if strings.Join(a[i].ArmOrder, ",") != strings.Join(c[i].ArmOrder, ",") {
			same = false
		}
	}
	if same {
		t.Fatal("changing trials.seed did not change any randomized arm order")
	}
}

// TestCorpusMustMatchThePinnedCount proves a truncated corpus is refused rather
// than quietly measured.
func TestCorpusMustMatchThePinnedCount(t *testing.T) {
	_, err := Execute(context.Background(), DemoManifest(), DemoCorpus()[:2], &FakeProvider{}, &FakeGrader{}, Options{})
	requireReason(t, err, ReasonIncomparableManifest)
}

// TestCorpusMustMatchThePinnedHash proves a same-sized edited corpus cannot run
// under the original manifest identity.
func TestCorpusMustMatchThePinnedHash(t *testing.T) {
	tasks := DemoCorpus()
	tasks[0].Input = "silently edited input"
	_, err := Execute(context.Background(), DemoManifest(), tasks, &FakeProvider{}, &FakeGrader{}, Options{})
	requireReason(t, err, ReasonIncomparableManifest)
}

// TestFailedTrialsAreCountedNotDropped proves a provider failure survives into
// the report instead of vanishing from the denominator.
func TestFailedTrialsAreCountedNotDropped(t *testing.T) {
	run, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &failingProvider{arm: "fak-ctxmmu"}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	rep, err := Summarize(run)
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	if rep.FailureCount == 0 {
		t.Fatal("report shows no failures for a provider that failed every fak-ctxmmu trial")
	}
	for _, a := range rep.Arms {
		if a.ArmID != "fak-ctxmmu" {
			continue
		}
		if a.Failures != a.Trials {
			t.Errorf("arm %s: %d failures over %d trials, want all", a.ArmID, a.Failures, a.Trials)
		}
		if a.Graded != 0 {
			t.Errorf("arm %s graded %d trials despite failing all of them", a.ArmID, a.Graded)
		}
		if a.PassRate != 0 {
			t.Errorf("arm %s reports pass rate %v with nothing graded", a.ArmID, a.PassRate)
		}
	}
}

// TestQualityLossIsVisibleNextToTokenSaving is the honesty fixture: an arm that
// cuts output tokens but loses correctness must show BOTH, so a token saving can
// never be quoted without the pass rate beside it.
func TestQualityLossIsVisibleNextToTokenSaving(t *testing.T) {
	run, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{FailArm: "caveman-upstream"}, Options{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	rep, err := Summarize(run)
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	var base, treat ArmSummary
	for _, a := range rep.Arms {
		switch a.ArmID {
		case "baseline":
			base = a
		case "caveman-upstream":
			treat = a
		}
	}
	if treat.OutputTokens >= base.OutputTokens {
		t.Fatalf("fixture broken: treatment output %d is not below baseline %d", treat.OutputTokens, base.OutputTokens)
	}
	if treat.PassRate >= base.PassRate {
		t.Fatalf("fixture broken: treatment pass rate %v is not below baseline %v", treat.PassRate, base.PassRate)
	}
	human := Human(rep)
	if !strings.Contains(human, "PASS%") {
		t.Errorf("human summary omits the pass-rate column:\n%s", human)
	}
}

// TestSetupCostIsChargedAndAmortized proves a one-time install cost is charged
// to its arm and spread over its trials rather than dropped.
func TestSetupCostIsChargedAndAmortized(t *testing.T) {
	run, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{SetupWallMS: 3000}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	rep, err := Summarize(run)
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	for _, a := range rep.Arms {
		switch a.Kind {
		case ArmBaseline:
			if a.Setup.WallMS != 0 {
				t.Errorf("baseline was charged %v ms of setup it never paid", a.Setup.WallMS)
			}
		default:
			if a.Setup.WallMS != 3000 {
				t.Errorf("arm %s setup wall %v, want 3000", a.ArmID, a.Setup.WallMS)
			}
			want := 3000 / float64(a.Graded)
			if a.SetupAmortizedWallMS != want {
				t.Errorf("arm %s amortized %v ms/trial, want %v", a.ArmID, a.SetupAmortizedWallMS, want)
			}
		}
	}
}

// TestUnavailableTimingsAreNotAveragedAsZero proves an unmeasurable TTFT is
// reported as unavailable rather than folded in as a zero, which would fabricate
// a speedup.
func TestUnavailableTimingsAreNotAveragedAsZero(t *testing.T) {
	run, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &noTimingProvider{}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	rep, err := Summarize(run)
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	for _, a := range rep.Arms {
		if a.TTFTAvailable {
			t.Errorf("arm %s claims TTFT is available when the provider never measured it", a.ArmID)
		}
		if a.TTFTSamples != 0 {
			t.Errorf("arm %s counted %d TTFT samples from a provider that reports none", a.ArmID, a.TTFTSamples)
		}
	}
	if !strings.Contains(Human(rep), "n/a") {
		t.Errorf("human summary does not mark the unavailable timing as n/a:\n%s", Human(rep))
	}
}

// TestRunRoundTripsThroughJSON proves the artifact a later ledger verb (#6680)
// will store reloads to the same report.
func TestRunRoundTripsThroughJSON(t *testing.T) {
	run, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{SetupWallMS: 500}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	blob, err := MarshalRun(run)
	if err != nil {
		t.Fatalf("MarshalRun: %v", err)
	}
	back, err := UnmarshalRun(blob)
	if err != nil {
		t.Fatalf("UnmarshalRun: %v", err)
	}
	if back.ManifestIdentity != run.ManifestIdentity {
		t.Fatalf("identity drifted across the round trip: %s vs %s", back.ManifestIdentity, run.ManifestIdentity)
	}
	a, err := Summarize(run)
	if err != nil {
		t.Fatalf("summarize original: %v", err)
	}
	b, err := Summarize(back)
	if err != nil {
		t.Fatalf("summarize reloaded: %v", err)
	}
	ja, err := MarshalReport(a)
	if err != nil {
		t.Fatalf("marshal a: %v", err)
	}
	jb, err := MarshalReport(b)
	if err != nil {
		t.Fatalf("marshal b: %v", err)
	}
	if string(ja) != string(jb) {
		t.Fatal("a run that round-tripped through JSON summarized differently")
	}
}

// TestUnmarshalRefusesUnknownFields proves a typo'd provenance key is a refusal,
// not a silently-unpinned term.
func TestUnmarshalRefusesUnknownFields(t *testing.T) {
	if _, err := UnmarshalManifest([]byte(`{"schema":"fak.armbench.manifest/1","modle":{}}`)); err == nil {
		t.Fatal("a misspelled manifest key was accepted")
	}
	if _, err := UnmarshalRun([]byte(`{"schema":"fak.armbench.run/0"}`)); err == nil {
		t.Fatal("a run with an unknown schema tag was accepted")
	}
	manifest, err := MarshalManifest(DemoManifest())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := UnmarshalManifest(append(manifest, []byte(`{"second":"value"}`)...)); err == nil {
		t.Fatal("a second trailing JSON value was accepted")
	}
}

// TestResumeRefusesDuplicatePriorRows proves resume never silently deduplicates
// a corrupt ledger before continuing.
func TestResumeRefusesDuplicatePriorRows(t *testing.T) {
	first, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	first.Trials = append(first.Trials, first.Trials[0])
	_, err = Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{Resume: first})
	requireReason(t, err, ReasonDuplicateTrial)
}

type failingProvider struct{ arm string }

func (p *failingProvider) Complete(ctx context.Context, req Request) (Response, error) {
	base := &FakeProvider{}
	resp, err := base.Complete(ctx, req)
	if err != nil || req.ArmID != p.arm {
		return resp, err
	}
	return Response{
		RawRequest: resp.RawRequest,
		Failure:    "synthetic provider outage",
		Retries:    2,
		Latency:    resp.Latency,
	}, nil
}

type noTimingProvider struct{}

func (noTimingProvider) Complete(ctx context.Context, req Request) (Response, error) {
	resp, err := (&FakeProvider{}).Complete(ctx, req)
	if err != nil {
		return resp, err
	}
	resp.Latency = Latency{WallMS: resp.Latency.WallMS}
	return resp, nil
}

type countingGrader struct{}

func (countingGrader) Grade(context.Context, Request, Response) (Judgment, error) {
	return Judgment{}, errNotExpected
}

var errNotExpected = &RefusalError{Reason: "GRADER_CALLED", Detail: "a fully resumed run must not re-grade anything"}

func requireReason(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected refusal %s, got no error", want)
	}
	if !isReason(err, want) {
		t.Fatalf("expected refusal %s, got %v", want, err)
	}
}
