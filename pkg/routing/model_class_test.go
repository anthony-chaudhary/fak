package routing

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

// eligibleCand builds a candidate map in which every named model passes all
// admission verdicts, so a test can override one field at a time.
func eligibleCand(models ...string) map[string]ClassCandidate {
	out := make(map[string]ClassCandidate, len(models))
	for _, m := range models {
		out[m] = ClassCandidate{Model: m, Eligible: true, FitsContext: true, Capable: true}
	}
	return out
}

// setVerdict rewrites one candidate line after applying mutate. This
// toolchain refuses to assign into a struct field through a map index, so the
// line is loaded, mutated, and stored whole.
func setVerdict(cands map[string]ClassCandidate, model string, mutate func(*ClassCandidate)) {
	cd := cands[model]
	mutate(&cd)
	cands[model] = cd
}

func TestModelClassValidate(t *testing.T) {
	cases := []struct {
		name  string
		class ModelClass
		want  error // nil = valid
	}{
		{"happy", ModelClass{Name: "big", Models: []string{"big-a", "big-b"}}, nil},
		{"empty strategy resolves to priority", ModelClass{Name: "big", Models: []string{"big-a"}, Strategy: BalanceStrategyPriority}, nil},
		{"drained member alongside live one", ModelClass{Name: "big", Models: []string{"big-a", "big-b"}, Weights: map[string]int{"big-a": 0}}, nil},
		{"empty name", ModelClass{Name: "", Models: []string{"big-a"}}, errors.New("empty name")},
		{"no members", ModelClass{Name: "big"}, errors.New("no members")},
		{"empty member", ModelClass{Name: "big", Models: []string{"big-a", ""}}, errors.New("is empty")},
		{"duplicate member", ModelClass{Name: "big", Models: []string{"big-a", "big-a"}}, errors.New("twice")},
		{"unknown strategy", ModelClass{Name: "big", Models: []string{"big-a"}, Strategy: "quantum"}, errors.New("unknown strategy")},
		{"weight for undeclared member", ModelClass{Name: "big", Models: []string{"big-a"}, Weights: map[string]int{"ghost": 1}}, errors.New("undeclared member")},
		{"negative weight", ModelClass{Name: "big", Models: []string{"big-a"}, Weights: map[string]int{"big-a": -1}}, errors.New("negative weight")},
		{"all drained", ModelClass{Name: "big", Models: []string{"big-a", "big-b"}, Weights: map[string]int{"big-a": 0, "big-b": 0}}, errors.New("can never select")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.class.Validate()
			if tc.want == nil {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("Validate() = nil, want an error naming %q", tc.want.Error())
			}
			if !strings.Contains(err.Error(), tc.want.Error()) {
				t.Fatalf("Validate() error %q does not name %q", err, tc.want.Error())
			}
		})
	}
}

func TestModelClassWeightOf(t *testing.T) {
	c := ModelClass{Name: "big", Models: []string{"a", "b", "c"}, Weights: map[string]int{"a": 3, "c": 0}}
	if got := c.WeightOf("a"); got != 3 {
		t.Errorf("WeightOf(a) = %d, want 3", got)
	}
	if got := c.WeightOf("b"); got != DefaultBalanceWeight {
		t.Errorf("WeightOf(b) = %d, want DefaultBalanceWeight %d", got, DefaultBalanceWeight)
	}
	if got := c.WeightOf("c"); got != 0 {
		t.Errorf("WeightOf(c) = %d, want 0 (declared drain)", got)
	}
}

// ladderManifest is the shared happy fixture: two classes, three routes, every
// reference resolving, no cycle, depth within the bound.
func ladderManifest() ([]ModelClass, map[string]ClassLadder) {
	classes := []ModelClass{
		{Name: "big", Models: []string{"big-a", "big-b"}},
		{Name: "small", Models: []string{"small-a"}},
	}
	ladders := map[string]ClassLadder{
		"big-a":   {Preferred: "big", Fallbacks: []string{"small"}},
		"big-b":   {Preferred: "big", Fallbacks: []string{"small"}},
		"small-a": {Preferred: "small"},
	}
	return classes, ladders
}

func TestValidateModelClasses(t *testing.T) {
	classes, ladders := ladderManifest()
	if err := ValidateModelClasses(classes, ladders); err != nil {
		t.Fatalf("happy path: ValidateModelClasses = %v, want nil", err)
	}

	cases := []struct {
		name     string
		classes  []ModelClass
		ladders  map[string]ClassLadder
		sentinel error
	}{
		{
			"unknown preferred ref",
			[]ModelClass{{Name: "big", Models: []string{"big-a"}}},
			map[string]ClassLadder{"big-a": {Preferred: "ghost"}},
			ErrModelClassUnknownRef,
		},
		{
			"unknown fallback ref",
			[]ModelClass{{Name: "big", Models: []string{"big-a"}}},
			map[string]ClassLadder{"big-a": {Preferred: "big", Fallbacks: []string{"ghost"}}},
			ErrModelClassUnknownRef,
		},
		{
			"self ref in fallbacks",
			[]ModelClass{{Name: "big", Models: []string{"big-a"}}},
			map[string]ClassLadder{"big-a": {Preferred: "big", Fallbacks: []string{"big"}}},
			ErrModelClassCycle,
		},
		{
			"two class cycle",
			[]ModelClass{
				{Name: "big", Models: []string{"big-a"}},
				{Name: "small", Models: []string{"small-a"}},
			},
			map[string]ClassLadder{
				"big-a":   {Preferred: "big", Fallbacks: []string{"small"}},
				"small-a": {Preferred: "small", Fallbacks: []string{"big"}},
			},
			ErrModelClassCycle,
		},
		{
			"chain deeper than bound",
			[]ModelClass{
				{Name: "a", Models: []string{"ma"}},
				{Name: "b", Models: []string{"mb"}},
				{Name: "c", Models: []string{"mc"}},
				{Name: "d", Models: []string{"md"}},
			},
			map[string]ClassLadder{"ma": {Preferred: "a", Fallbacks: []string{"b", "c", "d"}}},
			ErrModelClassTooDeep,
		},
		{
			"duplicate model across two classes",
			[]ModelClass{
				{Name: "big", Models: []string{"shared"}},
				{Name: "small", Models: []string{"shared"}},
			},
			map[string]ClassLadder{
				"shared": {Preferred: "big", Fallbacks: []string{"small"}},
			},
			ErrModelClassDuplicateMember,
		},
		{
			"duplicate class name",
			[]ModelClass{
				{Name: "big", Models: []string{"big-a"}},
				{Name: "big", Models: []string{"big-b"}},
			},
			map[string]ClassLadder{"big-a": {Preferred: "big"}},
			ErrModelClassDuplicateMember,
		},
		{
			"unreferenced class",
			[]ModelClass{
				{Name: "big", Models: []string{"big-a"}},
				{Name: "tiny", Models: []string{"tiny-a"}},
			},
			map[string]ClassLadder{"big-a": {Preferred: "big"}},
			ErrModelClassUnreferenced,
		},
		{
			"ladder model not in preferred class",
			[]ModelClass{{Name: "big", Models: []string{"big-b"}}},
			map[string]ClassLadder{"big-a": {Preferred: "big"}},
			ErrModelClassUnknownRef,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateModelClasses(tc.classes, tc.ladders)
			if err == nil {
				t.Fatalf("ValidateModelClasses = nil, want %v", tc.sentinel)
			}
			if !errors.Is(err, tc.sentinel) {
				t.Fatalf("ValidateModelClasses error %q is not %v", err, tc.sentinel)
			}
		})
	}
}

func TestSelectClassSingleEligiblePreferred(t *testing.T) {
	classes := map[string]ModelClass{"big": {Name: "big", Models: []string{"big-a"}}}
	d := SelectClass("big-a", ClassLadder{Preferred: "big"}, classes, eligibleCand("big-a"), nil, "k1")
	want := ClassDecision{
		Schema:         ModelClassSchema,
		RequestedModel: "big-a",
		ServedModel:    "big-a",
		RequestedClass: "big",
		ServedClass:    "big",
		Chosen:         true,
		Order:          []string{"big-a"},
	}
	if d.Schema != want.Schema || d.RequestedModel != want.RequestedModel ||
		d.ServedModel != want.ServedModel || d.RequestedClass != want.RequestedClass ||
		d.ServedClass != want.ServedClass || !d.Chosen || d.Transition != "" {
		t.Fatalf("decision = %+v, want %+v", d, want)
	}
	if !reflect.DeepEqual(d.Order, want.Order) {
		t.Fatalf("Order = %v, want %v", d.Order, want.Order)
	}
	if len(d.Skipped) != 0 {
		t.Fatalf("Skipped = %v, want none", d.Skipped)
	}
}

// measuredIdle returns a snapshot in which every member of the class is
// measured and idle, so weighted-hash actually ranks instead of failing open
// to declared order.
func measuredIdle(cls ModelClass) BalanceSnapshot {
	members := make(map[string]MemberScore, len(cls.Models))
	for _, m := range cls.Models {
		members[m] = MemberScore{Load: 0, Latency: 0, Measured: true}
	}
	return BalanceSnapshot{Group: cls.Name, Members: members, Observed: true}
}

func TestSelectClassWeightedHashSpreadAndDeterminism(t *testing.T) {
	classes := map[string]ModelClass{
		"big": {Name: "big", Models: []string{"big-a", "big-b"}, Strategy: BalanceStrategyWeightedHash},
	}
	ladder := ClassLadder{Preferred: "big"}
	cands := eligibleCand("big-a", "big-b")
	snap := func(cls ModelClass) BalanceSnapshot { return measuredIdle(cls) }

	counts := map[string]int{}
	for i := 0; i < 32; i++ {
		key := "spread-" + itoa(i)
		d := SelectClass("big-a", ladder, classes, cands, snap, key)
		if !d.Chosen {
			t.Fatalf("key %s: Chosen = false", key)
		}
		counts[d.ServedModel]++
	}
	if counts["big-a"] == 0 || counts["big-b"] == 0 {
		t.Fatalf("spread over 32 keys = %v, want both models chosen at least once", counts)
	}

	// Determinism: the same inputs replay byte-identically.
	again := SelectClass("big-a", ladder, classes, cands, snap, "spread-7")
	replay := SelectClass("big-a", ladder, classes, cands, snap, "spread-7")
	if !reflect.DeepEqual(again, replay) {
		t.Fatalf("replayed decision differs:\n%+v\nvs\n%+v", again, replay)
	}
}

func TestSelectClassIneligiblePreferredMember(t *testing.T) {
	classes := map[string]ModelClass{"big": {Name: "big", Models: []string{"big-a", "big-b"}}}
	cands := eligibleCand("big-a", "big-b")
	setVerdict(cands, "big-a", func(c *ClassCandidate) { c.Eligible = false })
	d := SelectClass("big-a", ClassLadder{Preferred: "big"}, classes, cands, nil, "k")
	if !d.Chosen || d.ServedModel != "big-b" {
		t.Fatalf("ServedModel = %q chosen=%v, want big-b", d.ServedModel, d.Chosen)
	}
	if d.Transition != "same_class" {
		t.Fatalf("Transition = %q, want same_class (requested big-a served by big-b)", d.Transition)
	}
	wantSkip := []ClassSkipped{{Model: "big-a", Class: "big", Reason: ClassSkipNotEligible}}
	if !reflect.DeepEqual(d.Skipped, wantSkip) {
		t.Fatalf("Skipped = %v, want %v", d.Skipped, wantSkip)
	}
}

func TestSelectClassCheckPrecedence(t *testing.T) {
	classes := map[string]ModelClass{"big": {Name: "big", Models: []string{"big-a", "big-b", "big-c"}}}
	cands := map[string]ClassCandidate{
		// Fails all three: the FIRST check (Eligible) is the recorded reason.
		"big-a": {Model: "big-a", Eligible: false, FitsContext: false, Capable: false},
		// Passes admission, fails context only.
		"big-b": {Model: "big-b", Eligible: true, FitsContext: false, Capable: false},
		// Passes admission and context, fails capability only.
		"big-c": {Model: "big-c", Eligible: true, FitsContext: true, Capable: false},
	}
	d := SelectClass("big-a", ClassLadder{Preferred: "big"}, classes, cands, nil, "k")
	if d.Chosen {
		t.Fatalf("Chosen = true, want full exhaustion")
	}
	want := []ClassSkipped{
		{Model: "big-a", Class: "big", Reason: ClassSkipNotEligible},
		{Model: "big-b", Class: "big", Reason: ClassSkipContextTooSmall},
		{Model: "big-c", Class: "big", Reason: ClassSkipCapabilityMismatch},
	}
	if !reflect.DeepEqual(d.Skipped, want) {
		t.Fatalf("Skipped = %v, want %v", d.Skipped, want)
	}

	// With the last blocker cleared, the capable member wins and the other two
	// keep their fixed-order reasons.
	setVerdict(cands, "big-c", func(c *ClassCandidate) { c.Capable = true })
	d = SelectClass("big-a", ClassLadder{Preferred: "big"}, classes, cands, nil, "k")
	if !d.Chosen || d.ServedModel != "big-c" {
		t.Fatalf("ServedModel = %q chosen=%v, want big-c", d.ServedModel, d.Chosen)
	}
	want = []ClassSkipped{
		{Model: "big-a", Class: "big", Reason: ClassSkipNotEligible},
		{Model: "big-b", Class: "big", Reason: ClassSkipContextTooSmall},
	}
	if !reflect.DeepEqual(d.Skipped, want) {
		t.Fatalf("Skipped = %v, want %v", d.Skipped, want)
	}
}

func TestSelectClassDrainedOverridesVerdicts(t *testing.T) {
	classes := map[string]ModelClass{
		"big": {Name: "big", Models: []string{"big-a", "big-b"}, Weights: map[string]int{"big-a": 0}},
	}
	// big-a is fully eligible on paper but declared drained: the config
	// decision outruns the runtime verdicts.
	cands := eligibleCand("big-a", "big-b")
	d := SelectClass("big-a", ClassLadder{Preferred: "big"}, classes, cands, nil, "k")
	if !d.Chosen || d.ServedModel != "big-b" {
		t.Fatalf("ServedModel = %q chosen=%v, want big-b", d.ServedModel, d.Chosen)
	}
	want := []ClassSkipped{{Model: "big-a", Class: "big", Reason: ClassSkipDrained}}
	if !reflect.DeepEqual(d.Skipped, want) {
		t.Fatalf("Skipped = %v, want %v", d.Skipped, want)
	}

	// A drained member absent from the candidate map still reports drained,
	// not not_eligible: regardless means regardless.
	cands = eligibleCand("big-b")
	d = SelectClass("big-a", ClassLadder{Preferred: "big"}, classes, cands, nil, "k")
	if !d.Chosen || d.ServedModel != "big-b" {
		t.Fatalf("ServedModel = %q chosen=%v, want big-b", d.ServedModel, d.Chosen)
	}
	if !reflect.DeepEqual(d.Skipped, want) {
		t.Fatalf("Skipped = %v, want %v", d.Skipped, want)
	}
}

func TestSelectClassFullExhaustion(t *testing.T) {
	classes := map[string]ModelClass{
		"big":   {Name: "big", Models: []string{"big-a", "big-b"}},
		"small": {Name: "small", Models: []string{"small-a"}},
	}
	cands := eligibleCand("big-a", "big-b", "small-a")
	setVerdict(cands, "big-a", func(c *ClassCandidate) { c.Eligible = false })
	setVerdict(cands, "big-b", func(c *ClassCandidate) { c.FitsContext = false })
	setVerdict(cands, "small-a", func(c *ClassCandidate) { c.Capable = false })
	d := SelectClass("big-a", ClassLadder{Preferred: "big", Fallbacks: []string{"small"}}, classes, cands, nil, "k")
	if d.Chosen || d.ServedModel != "" || d.ServedClass != "" || d.Transition != "" {
		t.Fatalf("decision = %+v, want no choice", d)
	}
	if len(d.Order) != 0 {
		t.Fatalf("Order = %v, want empty on exhaustion", d.Order)
	}
	want := []ClassSkipped{
		{Model: "big-a", Class: "big", Reason: ClassSkipNotEligible},
		{Model: "big-b", Class: "big", Reason: ClassSkipContextTooSmall},
		{Model: "small-a", Class: "small", Reason: ClassSkipCapabilityMismatch},
	}
	if !reflect.DeepEqual(d.Skipped, want) {
		t.Fatalf("Skipped = %v, want %v", d.Skipped, want)
	}
}

func TestSelectClassStepDown(t *testing.T) {
	classes := map[string]ModelClass{
		"big":   {Name: "big", Models: []string{"big-a", "big-b"}},
		"small": {Name: "small", Models: []string{"small-a", "small-b"}},
	}
	cands := eligibleCand("big-a", "big-b", "small-a", "small-b")
	setVerdict(cands, "big-a", func(c *ClassCandidate) { c.Eligible = false })
	setVerdict(cands, "big-b", func(c *ClassCandidate) { c.Eligible = false })
	d := SelectClass("big-a", ClassLadder{Preferred: "big", Fallbacks: []string{"small"}}, classes, cands, nil, "k")
	if !d.Chosen || d.ServedClass != "small" || d.ServedModel != "small-a" {
		t.Fatalf("ServedModel/Class = %q/%q, want small/small-a (priority keeps declared head)", d.ServedModel, d.ServedClass)
	}
	if d.Transition != "class_step_down" {
		t.Fatalf("Transition = %q, want class_step_down", d.Transition)
	}
	wantOrder := []string{"small-a", "small-b"}
	if !reflect.DeepEqual(d.Order, wantOrder) {
		t.Fatalf("Order = %v, want %v (fallback winner leads, then its siblings)", d.Order, wantOrder)
	}
	wantSkip := []ClassSkipped{
		{Model: "big-a", Class: "big", Reason: ClassSkipNotEligible},
		{Model: "big-b", Class: "big", Reason: ClassSkipNotEligible},
	}
	if !reflect.DeepEqual(d.Skipped, wantSkip) {
		t.Fatalf("Skipped = %v, want %v", d.Skipped, wantSkip)
	}
}

func TestSelectClassDisallowedDegradation(t *testing.T) {
	// "tiny" exists in the manifest but the ladder names no fallbacks: the
	// walk may never invent a lower class, however idle tiny is.
	classes := map[string]ModelClass{
		"big":  {Name: "big", Models: []string{"big-a"}},
		"tiny": {Name: "tiny", Models: []string{"tiny-a"}},
	}
	cands := eligibleCand("big-a", "tiny-a")
	setVerdict(cands, "big-a", func(c *ClassCandidate) { c.Eligible = false })
	d := SelectClass("big-a", ClassLadder{Preferred: "big"}, classes, cands, nil, "k")
	if d.Chosen {
		t.Fatalf("decision = %+v, want no choice: degradation was not declared", d)
	}
	if d.ServedModel != "" || d.ServedClass != "" || d.Transition != "" {
		t.Fatalf("exhausted decision carries a served model/class/transition: %+v", d)
	}
}

func TestSelectClassFallbackOrdering(t *testing.T) {
	classes := map[string]ModelClass{
		"big": {Name: "big", Models: []string{"big-a"}},
		"f1":  {Name: "f1", Models: []string{"f1-a"}},
		"f2":  {Name: "f2", Models: []string{"f2-a"}},
	}
	ladder := ClassLadder{Preferred: "big", Fallbacks: []string{"f1", "f2"}}
	cands := eligibleCand("big-a", "f1-a", "f2-a")
	setVerdict(cands, "big-a", func(c *ClassCandidate) { c.Eligible = false })

	// First fallback with an eligible member wins; the second is never chosen
	// but its selectable members still appear in Order, behind the first's.
	d := SelectClass("big-a", ladder, classes, cands, nil, "k")
	if !d.Chosen || d.ServedModel != "f1-a" || d.ServedClass != "f1" {
		t.Fatalf("ServedModel/Class = %q/%q, want f1/f1-a", d.ServedModel, d.ServedClass)
	}
	if d.Transition != "class_step_down" {
		t.Fatalf("Transition = %q, want class_step_down", d.Transition)
	}
	wantOrder := []string{"f1-a", "f2-a"}
	if !reflect.DeepEqual(d.Order, wantOrder) {
		t.Fatalf("Order = %v, want %v (first class's members before second's)", d.Order, wantOrder)
	}

	// First fallback fully ineligible: the second takes the request.
	setVerdict(cands, "f1-a", func(c *ClassCandidate) { c.Eligible = false })
	d = SelectClass("big-a", ladder, classes, cands, nil, "k")
	if !d.Chosen || d.ServedModel != "f2-a" || d.ServedClass != "f2" {
		t.Fatalf("ServedModel/Class = %q/%q, want f2/f2-a", d.ServedModel, d.ServedClass)
	}
	if !reflect.DeepEqual(d.Order, []string{"f2-a"}) {
		t.Fatalf("Order = %v, want [f2-a]", d.Order)
	}
	wantSkip := []ClassSkipped{
		{Model: "big-a", Class: "big", Reason: ClassSkipNotEligible},
		{Model: "f1-a", Class: "f1", Reason: ClassSkipNotEligible},
	}
	if !reflect.DeepEqual(d.Skipped, wantSkip) {
		t.Fatalf("Skipped = %v, want %v", d.Skipped, wantSkip)
	}
}

func TestSelectClassStrategyHonored(t *testing.T) {
	// Least-load with a measured snapshot picks the low-load member even
	// though it is not the declared head.
	classes := map[string]ModelClass{
		"ll": {Name: "ll", Models: []string{"ll-a", "ll-b"}, Strategy: BalanceStrategyLeastLoad},
	}
	snap := func(cls ModelClass) BalanceSnapshot {
		return BalanceSnapshot{
			Group: cls.Name,
			Members: map[string]MemberScore{
				"ll-a": {Load: 0.9, Measured: true},
				"ll-b": {Load: 0.1, Measured: true},
			},
			Observed: true,
		}
	}
	d := SelectClass("ll-a", ClassLadder{Preferred: "ll"}, classes, eligibleCand("ll-a", "ll-b"), snap, "k")
	if !d.Chosen || d.ServedModel != "ll-b" {
		t.Fatalf("least-load ServedModel = %q chosen=%v, want ll-b (load 0.1 beats 0.9)", d.ServedModel, d.Chosen)
	}

	// Priority ignores the same scores entirely: the declared head serves.
	ll := classes["ll"]
	ll.Strategy = BalanceStrategyPriority
	classes["ll"] = ll
	d = SelectClass("ll-a", ClassLadder{Preferred: "ll"}, classes, eligibleCand("ll-a", "ll-b"), snap, "k")
	if !d.Chosen || d.ServedModel != "ll-a" {
		t.Fatalf("priority ServedModel = %q chosen=%v, want ll-a (scores ignored)", d.ServedModel, d.Chosen)
	}
}

func TestSelectClassMissingCandsEntry(t *testing.T) {
	classes := map[string]ModelClass{"big": {Name: "big", Models: []string{"big-a", "big-b"}}}
	// big-a has no candidate line at all: absence is a verdict, not an error.
	d := SelectClass("big-b", ClassLadder{Preferred: "big"}, classes, eligibleCand("big-b"), nil, "k")
	if !d.Chosen || d.ServedModel != "big-b" {
		t.Fatalf("ServedModel = %q chosen=%v, want big-b", d.ServedModel, d.Chosen)
	}
	want := []ClassSkipped{{Model: "big-a", Class: "big", Reason: ClassSkipNotEligible}}
	if !reflect.DeepEqual(d.Skipped, want) {
		t.Fatalf("Skipped = %v, want %v", d.Skipped, want)
	}
}

func TestSelectClassMissingClassIsZeroCandidates(t *testing.T) {
	// The manifest lacks "ghost" entirely (an unvalidated manifest): the walk
	// treats it as zero candidates and steps down rather than panicking.
	classes := map[string]ModelClass{"real": {Name: "real", Models: []string{"real-a"}}}
	d := SelectClass("real-a", ClassLadder{Preferred: "ghost", Fallbacks: []string{"real"}}, classes, eligibleCand("real-a"), nil, "k")
	if !d.Chosen || d.ServedModel != "real-a" || d.ServedClass != "real" {
		t.Fatalf("decision = %+v, want step-down to real/real-a", d)
	}
	if d.Transition != "class_step_down" {
		t.Fatalf("Transition = %q, want class_step_down", d.Transition)
	}
}

func TestSelectClassAuditNeverEchoesKey(t *testing.T) {
	classes := map[string]ModelClass{
		"big":   {Name: "big", Models: []string{"big-a", "big-b"}},
		"small": {Name: "small", Models: []string{"small-a"}},
	}
	cands := eligibleCand("big-a", "big-b", "small-a")
	setVerdict(cands, "big-a", func(c *ClassCandidate) { c.Eligible = false })
	setVerdict(cands, "big-b", func(c *ClassCandidate) { c.Capable = false })
	// Long and distinctive so any accidental echo is unmistakable.
	key := "session-key-abcdef-1234567890-super-distinctive-and-long-tail"
	d := SelectClass("big-a", ClassLadder{Preferred: "big", Fallbacks: []string{"small"}}, classes, cands, nil, key)
	for _, line := range d.Audit {
		if strings.Contains(line, key) {
			t.Fatalf("audit line echoes the balance key: %q", line)
		}
	}
	// The whole receipt, serialized, must be equally clean: a receipt is what
	// an operator pastes into an issue.
	blob, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("json.Marshal(decision) = %v", err)
	}
	if strings.Contains(string(blob), key) {
		t.Fatalf("serialized receipt echoes the balance key: %s", blob)
	}
	// Pin the audit shapes: one request line, one per class visited, one per
	// skip, one verdict line.
	joined := strings.Join(d.Audit, "\n")
	for _, shape := range []string{
		"model_class: requested=big-a class=big fallbacks=1",
		"model_class: class=big candidates=2 selectable=0",
		"model_class: class=small candidates=1 selectable=1",
		"model_class: skip=big-a class=big reason=not_eligible",
		"model_class: skip=big-b class=big reason=capability_mismatch",
		"model_class: requested=big-a class=big transition=class_step_down chosen=small-a",
	} {
		if !strings.Contains(joined, shape) {
			t.Fatalf("audit missing line %q;\naudit:\n%s", shape, joined)
		}
	}
}

func TestClassSkipReasonsClosedSet(t *testing.T) {
	got := ClassSkipReasons()
	want := []string{"not_eligible", "drained", "context_too_small", "capability_mismatch"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ClassSkipReasons() = %v, want %v (declaration order)", got, want)
	}
}
