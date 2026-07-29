package conceptbench

import (
	"strings"
	"testing"
)

// smallArmAsk is the one ask every gate case perturbs exactly one field of: the
// admitted shape (opted in, weaker tier, the commit-stamp concept, a real leaf).
func smallArmAsk() AffordanceAsk {
	return AffordanceAsk{Concept: ConceptCommitStamp, Leaf: "gateway", Tier: TierSmall, Enabled: true}
}

// TestAffordanceHintCarriesTrailerTemplateAndCheckableStep is half (a) of #5380's
// done condition: the hint must carry the concept's EXACT `(fak <leaf>)` trailer
// template for the concept's own lane, and the witness rule.
//
// It asserts more than the issue's literal words on purpose. A hint that only
// restates "use a `(fak <leaf>)` trailer and do not claim an unproduced witness" is
// a paraphrase of the terse clause the small arm already failed to act on, so the
// test also pins the CHECKABLE STEP — the offline subject check the arm can run
// before it commits. If a later edit trims that line, the hint degrades back into
// prose and this test is what says so.
func TestAffordanceHintCarriesTrailerTemplateAndCheckableStep(t *testing.T) {
	hint, ok := smallArmAsk().Hint()
	if !ok {
		t.Fatalf("Hint() refused the admitted ask; want the hint")
	}
	for _, want := range []string{
		// The exact trailer template, for THIS concept's lane.
		"(fak gateway)",
		"type(scope): <verb> <what> (fak gateway)",
		// The checkable step: the offline verdict check, named as a runnable
		// command, plus what its exit codes mean.
		"tools/check_commit_msg.py",
		"--message",
		"Exit 0",
		// The witness rule, in the form that makes an unfinished episode honest.
		"CLAIM_UNWITNESSED",
		"not yet",
	} {
		if !strings.Contains(hint, want) {
			t.Errorf("hint is missing %q:\n%s", want, hint)
		}
	}
}

// TestAffordanceHintTemplatesTheAskedLeaf pins that the trailer template follows
// the ASK's lane rather than being frozen at whichever leaf the first fixture used.
// A hard-coded leaf would still pass the "carries a `(fak <leaf>)` template" test
// above while teaching every arm the wrong stamp.
func TestAffordanceHintTemplatesTheAskedLeaf(t *testing.T) {
	ask := smallArmAsk()
	ask.Leaf = "conceptbench"
	hint, ok := ask.Hint()
	if !ok {
		t.Fatalf("Hint() refused the admitted ask; want the hint")
	}
	if !strings.Contains(hint, "(fak conceptbench)") {
		t.Errorf("hint does not name the asked leaf:\n%s", hint)
	}
	if strings.Contains(hint, "(fak gateway)") {
		t.Errorf("hint names a leaf the ask did not ask for:\n%s", hint)
	}
}

// TestAffordanceHintFiresOnlyForTheWeakerTier is half (b) of #5380's done
// condition, and it is the half that keeps the benchmark honest: every arm that is
// NOT the opted-in weaker-tier commit-stamp arm must receive nothing at all.
//
// The frontier and unrated rows are the load-bearing ones. #5380's promotion
// evidence is a contrast ("the small arm moved, the frontier arm unchanged"), and
// that contrast is only readable if the control's frame is provably untouched.
func TestAffordanceHintFiresOnlyForTheWeakerTier(t *testing.T) {
	mutate := func(f func(*AffordanceAsk)) AffordanceAsk {
		a := smallArmAsk()
		f(&a)
		return a
	}
	refused := []struct {
		name string
		ask  AffordanceAsk
		why  string
	}{
		{"opted_out", mutate(func(a *AffordanceAsk) { a.Enabled = false }),
			"gen/next dogfoods before default: an un-opted-in run's frame must be byte-identical to today's"},
		{"frontier_tier", mutate(func(a *AffordanceAsk) { a.Tier = TierFrontier }),
			"the frontier arm is the control the promotion evidence reads against"},
		{"unrated_tier", mutate(func(a *AffordanceAsk) { a.Tier = TierUnrated }),
			"an id with no rating on record is not PROVEN weak; injecting on a guess rewrites an arm the report may read as a control"},
		{"zero_tier", mutate(func(a *AffordanceAsk) { a.Tier = "" }),
			"a zero-value tier is an absent rating, not a weak one"},
		{"other_concept", mutate(func(a *AffordanceAsk) { a.Concept = ConceptLane }),
			"this text is the commit-stamp contract; a hint that fires on every concept is one an arm learns to skip"},
		{"blank_leaf", mutate(func(a *AffordanceAsk) { a.Leaf = "   " }),
			"an underivable leaf must fail closed: a guessed stamp is worse than silence"},
		{"zero_ask", AffordanceAsk{}, "the zero value is a refusal"},
	}
	for _, tc := range refused {
		t.Run(tc.name, func(t *testing.T) {
			if hint, ok := tc.ask.Hint(); ok {
				t.Fatalf("Hint() fired for %s; want a refusal (%s)\n%s", tc.name, tc.why, hint)
			}
		})
	}

	if _, ok := smallArmAsk().Hint(); !ok {
		t.Fatalf("Hint() refused the admitted ask; the gate cases above would then be vacuous")
	}
}

// TestAffordanceFrameLeavesARefusedAskByteIdentical pins the control's guarantee at
// the frame level: a refused ask returns the prompt UNCHANGED, so a frontier arm's
// transport receives exactly the bytes an un-flagged run would have sent.
func TestAffordanceFrameLeavesARefusedAskByteIdentical(t *testing.T) {
	const prompt = "Make this one-line change and commit it the fak way.\n"
	frontier := smallArmAsk()
	frontier.Tier = TierFrontier
	if got := frontier.Frame(prompt); got != prompt {
		t.Errorf("frontier frame was rewritten:\n got %q\nwant %q", got, prompt)
	}
	small := smallArmAsk().Frame(prompt)
	if !strings.HasPrefix(small, strings.TrimRight(prompt, "\n")) {
		t.Errorf("the treated frame dropped the original task text:\n%s", small)
	}
	if len(small) <= len(prompt) {
		t.Errorf("the treated frame carries no hint:\n%s", small)
	}
}

// TestRegistryTierRatesTheMeasuredContrast pins the two ids the replay findings
// actually contrasted, plus the unknown-id refusal. Without this, the gate above
// could be correct while every arm resolved to the same band.
func TestRegistryTierRatesTheMeasuredContrast(t *testing.T) {
	r := NewRegistry()
	for _, tc := range []struct {
		model string
		want  ArmTier
	}{
		{"claude-3-5-haiku", TierSmall},   // the arm the findings recorded falling off
		{"CLAUDE-3-5-HAIKU", TierSmall},   // resolution is case-insensitive, like Resolve
		{"claude-opus-4-8", TierFrontier}, // the arm that passed the same frame
		{"claude-opus-5", TierFrontier},
		{"smollm2", TierSmall},
		{"no-such-model", TierUnrated},
	} {
		if got := r.TierOf(tc.model); got != tc.want {
			t.Errorf("TierOf(%q)=%q, want %q", tc.model, got, tc.want)
		}
	}
	// A raw endpoint carries no rating the registry can justify, so it must never
	// be treated as the weaker arm.
	r.RegisterRaw("some-raw-model", "http://endpoint.invalid/v1")
	if got := r.TierOf("some-raw-model"); got != TierUnrated {
		t.Errorf("TierOf(raw)=%q, want %q", got, TierUnrated)
	}
}

// TestLeafOfPaths pins the leaf derivation the spine feeds the template, including
// the empty answer that makes the injection fail closed.
func TestLeafOfPaths(t *testing.T) {
	for _, tc := range []struct {
		name  string
		paths []string
		want  string
	}{
		{"top_level_dir", []string{"gateway/tick.go"}, "gateway"},
		{"internal_leaf", []string{"internal/conceptbench/report.go"}, "conceptbench"},
		{"cmd_leaf", []string{"cmd/conceptbench/spine.go"}, "conceptbench"},
		{"windows_separators", []string{`internal\conceptbench\grade.go`}, "conceptbench"},
		{"sorted_not_insertion_order", []string{"zeta/x.go", "alpha/y.go"}, "alpha"},
		{"root_level_only", []string{"README.md"}, ""},
		{"empty", nil, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := LeafOfPaths(tc.paths); got != tc.want {
				t.Errorf("LeafOfPaths(%v)=%q, want %q", tc.paths, got, tc.want)
			}
		})
	}
}
