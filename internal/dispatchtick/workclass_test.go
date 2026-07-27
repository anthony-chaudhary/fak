package dispatchtick

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestATierLabelIsTheOnlyThingThatNamesAWorkClass(t *testing.T) {
	for _, tc := range []struct {
		name   string
		labels []string
		class  modelroute.WorkClass
		why    ClassAttribution
	}{
		{"routine", []string{"tier/T2-optimal", "tier/T2-required"}, modelroute.ClassRoutine, ClassFromTierLabel},
		{"normal", []string{"tier/T1-optimal", "tier/T1-required"}, modelroute.ClassNormalImpl, ClassFromTierLabel},
		{"hard", []string{"tier/T0-optimal", "tier/T0-required"}, modelroute.ClassUltraHard, ClassFromTierLabel},
		{"ultra", []string{"tier/ultra"}, modelroute.ClassUltraHard, ClassFromTierLabel},
	} {
		t.Run(tc.name, func(t *testing.T) {
			class, why := WorkClassForIssue(tc.labels)
			if class != tc.class || why != tc.why {
				t.Errorf("class=%q why=%q, want %q/%q", class, why, tc.class, tc.why)
			}
		})
	}
}

func TestAnUntaggedIssueGradesNothingRatherThanGradingAsRoutine(t *testing.T) {
	// The tempting shortcut. Untagged work IS usually small, and calling it routine would
	// produce far more evidence far faster — which is exactly how a 4B laptop model earns
	// a rung from a backlog nobody triaged. The empty class is the refusal, and the reason
	// says the fix is triage.
	for _, labels := range [][]string{nil, {}, {"bug", "area/gateway"}, {"tier/nonsense"}} {
		class, why := WorkClassForIssue(labels)
		if class != "" {
			t.Errorf("labels %v graded as %q — nothing declared what kind of work this was", labels, class)
		}
		if why != ClassNoTierLabel {
			t.Errorf("labels %v: why = %q, want %q", labels, why, ClassNoTierLabel)
		}
	}
}

func TestCoordinationWorkIsNotEvidenceAboutImplementingAnything(t *testing.T) {
	// A PM slot's witnessed commit is a plan, a triage pass, a label sweep. Grading it as
	// routine would let a model earn the routine IMPLEMENTATION rung by labelling issues.
	class, why := WorkClassForIssue([]string{PMLabel})
	if class != "" {
		t.Errorf("a project-management slot graded as %q — coordination success is not implementation evidence", class)
	}
	if why != ClassCoordinationBucket {
		t.Errorf("why = %q, want %q — the refusal must name the vocabulary gap, not look like missing triage", why, ClassCoordinationBucket)
	}
	// The two empty answers must stay distinguishable: one is fixed by triaging the
	// backlog, the other by adding a class. Summing them hides which.
	if _, untagged := WorkClassForIssue(nil); untagged == why {
		t.Errorf("an untagged issue and a coordination slot both report %q", why)
	}
	// A PM issue that ALSO carries a real tier tag is a hard planning issue, and the
	// shipped bucket parser already escalates it. The class must follow that escalation
	// rather than being swallowed by the PM label.
	if class, why := WorkClassForIssue([]string{PMLabel, "tier/T0-optimal", "tier/T0-required"}); class != modelroute.ClassUltraHard || why != ClassFromTierLabel {
		t.Errorf("tier-tagged PM issue: class=%q why=%q, want ultra-hard from its tier label", class, why)
	}
}

func TestTheBucketToClassTableIsNotTransposed(t *testing.T) {
	// The failure this guards is a table whose ROWS are right and whose ORDER is inverted:
	// tier/T2 routine work mapped to ultra-hard would grade a cheap model at the frontier
	// tier from the easiest work on the board. Asserting the literal pairs would not catch
	// a consistent transposition, so this asserts the ORDER through the sanctioned
	// comparator — the same reason modelroute routes every tier comparison through
	// MeetsRequirement rather than a raw `>=`.
	floor := func(bucket LaunchBucket) modelroute.WorkTier {
		class, ok := bucketWorkClass[bucket]
		if !ok {
			t.Fatalf("bucket %q maps to no class", bucket)
		}
		return modelroute.PolicyFor(class).RequiredTier
	}
	routine, normal, hard := floor(BucketRoutine), floor(BucketNormal), floor(BucketHard)
	if !normal.MoreDemandingThan(routine) {
		t.Errorf("normal floor %v is not more demanding than routine floor %v", normal, routine)
	}
	if !hard.MoreDemandingThan(normal) {
		t.Errorf("hard floor %v is not more demanding than normal floor %v", hard, normal)
	}
	// Both T0 buckets are the same class: tier/ultra is a promotion within T0, not a
	// fifth class, and inventing a rung above the frontier would grade against a bar the
	// vocabulary cannot express.
	if bucketWorkClass[BucketUltra] != bucketWorkClass[BucketHard] {
		t.Errorf("ultra=%q hard=%q — the tier vocabulary has no level beyond T0",
			bucketWorkClass[BucketUltra], bucketWorkClass[BucketHard])
	}
}

func TestNoDispatchLabelCanMintTheSecurityReleaseClass(t *testing.T) {
	// ClassSecurityRelease's floor exists to stop a cheap model serving push/delete/
	// release work. No dispatch label declares that class, so no slot may be graded into
	// it — a grade there would be evidence nobody produced about work nobody declared.
	for bucket, class := range bucketWorkClass {
		if class == modelroute.ClassSecurityRelease {
			t.Errorf("bucket %q mints %q", bucket, class)
		}
	}
	for _, labels := range [][]string{
		{"tier/ultra"}, {PMLabel}, {"tier/T0-optimal", "tier/T0-required"},
		{"tier/T1-optimal", "tier/T1-required"}, {"tier/T2-optimal", "tier/T2-required"}, nil,
	} {
		if class, _ := WorkClassForIssue(labels); class == modelroute.ClassSecurityRelease {
			t.Errorf("labels %v graded as %q", labels, class)
		}
	}
}

func TestEveryClassTheResolverEmitsIsOneTheFoldAccepts(t *testing.T) {
	// Totality against the consumer. modelroute.PolicyFor maps an UNRECOGNIZED class to
	// the T0 floor, so a class this file emits that the capability grader does not know
	// would be silently read as frontier-tier evidence. The check is that the emitted
	// class survives a round trip through the grader's own class filter.
	for bucket, class := range bucketWorkClass {
		ev := []modelroute.ClassEvidence{{Class: class, Attempts: 100, Successes: 100, Verify: modelroute.VerifyWitness}}
		g := modelroute.GradeCapability("tiny", ev, modelroute.GradeFloor{})
		if !g.Measured {
			t.Errorf("bucket %q class %q: 100/100 witnessed attempts graded UNMEASURED (%s) — the class is not one the grader knows",
				bucket, class, g.Reason)
		}
		if g.Class != class {
			t.Errorf("bucket %q graded against class %q, want %q", bucket, g.Class, class)
		}
	}
}

func TestAResolverWithNoLabelSourceGradesNothingInsteadOfPanicking(t *testing.T) {
	// A caller with no label lookup has read no declaration, and the honest result of
	// that is zero evidence. The dangerous alternatives are a panic (which would take
	// down a sweep) and a fallback class (which would file every slot at the T0 floor).
	resolve := ClassResolver(nil)
	if got := resolve(WitnessRecord{Issue: 7}); got != "" {
		t.Errorf("class = %q, want empty", got)
	}
	// And the producer must then DROP those rows rather than emit them, counting each.
	records := []WitnessRecord{
		{Issue: 1, Log: "resolve-1.log", Model: "tiny", Claim: ClaimWitnessed, TestClaim: ClaimTestGreen},
		{Issue: 2, Log: "resolve-2.log", Model: "tiny", Claim: ClaimWitnessed, TestClaim: ClaimTestGreen},
	}
	out, stats := TurnOutcomesFromWitness(records, WitnessEvidenceOptions{Class: resolve})
	if len(out) != 0 || stats.Unclassified != 2 || stats.Produced != 0 {
		t.Errorf("out=%d stats=%+v — unlabelled slots must be dropped and counted, never filed at the unknown-class floor", len(out), stats)
	}
}

func TestTheResolverReadsEachSlotsOwnIssueRatherThanTheFirstOne(t *testing.T) {
	// The bug this catches is a resolver that closes over one issue's labels and reuses
	// them: a single tier/T0 issue in the sweep would then grade every model in it at the
	// frontier tier.
	labels := map[int][]string{
		1: {"tier/T0-optimal", "tier/T0-required"},
		2: {"tier/T2-optimal", "tier/T2-required"},
		3: nil,
	}
	resolve := ClassResolver(func(issue int) []string { return labels[issue] })
	for issue, want := range map[int]modelroute.WorkClass{
		1: modelroute.ClassUltraHard,
		2: modelroute.ClassRoutine,
		3: "",
	} {
		if got := resolve(WitnessRecord{Issue: issue}); got != want {
			t.Errorf("issue %d graded %q, want %q", issue, got, want)
		}
	}
}

func TestCoverageSeparatesWhatWasGradedFromWhatCouldNotBe(t *testing.T) {
	labels := map[int][]string{
		1: {"tier/T2-optimal", "tier/T2-required"},
		2: {"tier/T2-optimal", "tier/T2-required"},
		3: {"tier/T0-optimal", "tier/T0-required"},
		4: {PMLabel},
	}
	records := []WitnessRecord{
		{Issue: 1}, {Issue: 2}, {Issue: 3}, {Issue: 4}, {Issue: 5}, {Issue: 6}, {Issue: 7},
	}
	c := FoldClassTally(records, func(issue int) []string { return labels[issue] })
	if c.Total != 7 || c.Classified != 3 {
		t.Fatalf("coverage = %+v", c)
	}
	if c.ByClass[modelroute.ClassRoutine] != 2 || c.ByClass[modelroute.ClassUltraHard] != 1 {
		t.Errorf("by class = %v", c.ByClass)
	}
	if c.Unclassified[ClassNoTierLabel] != 3 || c.Unclassified[ClassCoordinationBucket] != 1 {
		t.Errorf("unclassified = %v — the untriaged backlog and the coordination slot need different fixes", c.Unclassified)
	}
	if c.Unclassified[ClassFromTierLabel] != 0 {
		t.Errorf("a classified record was counted as unclassified: %v", c.Unclassified)
	}
	if got := c.Reasons(); len(got) != 2 || got[0] != ClassCoordinationBucket || got[1] != ClassNoTierLabel {
		t.Errorf("reasons = %v, want a stable sorted pair", got)
	}
}

func TestASweepWithNoLabelSourceReportsNoCoverageRatherThanFullCoverage(t *testing.T) {
	c := FoldClassTally([]WitnessRecord{{Issue: 1}, {Issue: 2}}, nil)
	if c.Classified != 0 || c.Unclassified[ClassNoTierLabel] != 2 || c.Total != 2 {
		t.Errorf("coverage = %+v — a caller that supplied no labels saw no declaration", c)
	}
	if len(c.ByClass) != 0 {
		t.Errorf("by class = %v, want empty", c.ByClass)
	}
}

func TestCoverageAndTheResolverAgreeOnWhatCountsAsClassified(t *testing.T) {
	// Two paths answer "was this gradeable?" and a fleet reads the fold while the producer
	// obeys the resolver. If they ever disagree, the reported coverage describes a
	// different set of slots than the ones that actually produced evidence.
	labels := map[int][]string{
		1: {"tier/T1-optimal", "tier/T1-required"},
		2: {PMLabel},
		3: {"chore"},
		4: {"tier/ultra"},
	}
	lookup := func(issue int) []string { return labels[issue] }
	records := []WitnessRecord{{Issue: 1}, {Issue: 2}, {Issue: 3}, {Issue: 4}}
	c := FoldClassTally(records, lookup)
	resolve := ClassResolver(lookup)
	named := 0
	for _, r := range records {
		if resolve(r) != "" {
			named++
		}
	}
	if named != c.Classified {
		t.Errorf("resolver named %d, fold counted %d classified", named, c.Classified)
	}
}
