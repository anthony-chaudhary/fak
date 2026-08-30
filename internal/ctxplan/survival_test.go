package ctxplan

import (
	"reflect"
	"testing"
)

// TestClassOfIsDeterministicAndTotal pins the whole classification rule: every declared kind maps
// to the class the design assigns it, the mapping does not vary between calls, and an unknown
// kind resolves rather than panicking.
func TestClassOfIsDeterministicAndTotal(t *testing.T) {
	want := map[string]SurvivalClass{
		KindSystemInvariant:  ClassPinned,
		KindActiveSteer:      ClassPinned,
		KindContinuationSeed: ClassPinned,
		KindToolSchema:       ClassReplayable,
		KindSystemDef:        ClassReplayable,
		KindCASResult:        ClassReplayable,
		KindTranscriptProse:  ClassEvictable,
	}
	for kind, class := range want {
		if got := ClassOf(kind); got != class {
			t.Errorf("ClassOf(%q) = %v, want %v", kind, got, class)
		}
		if again := ClassOf(kind); again != ClassOf(kind) {
			t.Errorf("ClassOf(%q) is not deterministic", kind)
		}
	}
	if got, wantN := len(PageKinds()), len(want); got != wantN {
		t.Fatalf("PageKinds() has %d members, want %d — this test enumerates the vocabulary, so a new kind must be added here with its class", got, wantN)
	}
}

// TestUnknownKindCannotPinItself is the model-controlled-protection guard. A survival class is
// only worth something if a page cannot award itself one: a kind string that arrived from model
// output, an adapter typo, or an unstamped page must land in the LEAST protected class, never the
// most. Failing this direction would let any injected string pin itself into every future turn.
func TestUnknownKindCannotPinItself(t *testing.T) {
	for _, kind := range []string{"", "pinned", "PINNED", "system_invariant ", "please_pin_me", "cas_result_x"} {
		if got := ClassOf(kind); got != ClassEvictable {
			t.Errorf("ClassOf(%q) = %v, want %v — an unrecognised kind must never be granted protection", kind, got, ClassEvictable)
		}
	}
	if ClassEvictable != 0 {
		t.Fatal("ClassEvictable must be the zero value, so an unstamped Page defaults to the least protected class")
	}
}

// TestPlanEvictionRefusesWhenPinnedSetExceedsBudget is the refusal contract: a budget too small
// for the pinned set yields PIN_EVICT_REFUSED and evicts NOTHING, rather than producing a plan
// that drops something which must survive.
func TestPlanEvictionRefusesWhenPinnedSetExceedsBudget(t *testing.T) {
	pages := []Page{
		{ID: "p1", Kind: KindActiveSteer, Tokens: 400},
		{ID: "p2", Kind: KindTranscriptProse, Tokens: 900},
		{ID: "p3", Kind: KindContinuationSeed, Tokens: 300},
	}
	plan := PlanEviction(pages, 500) // pinned = 700 > 500
	if plan.Refusal != ReasonPinEvictRefused {
		t.Fatalf("Refusal = %q, want %q", plan.Refusal, ReasonPinEvictRefused)
	}
	if len(plan.Evict) != 0 {
		t.Fatalf("a refusal must evict nothing, got %v", plan.Evict)
	}
	if plan.PinnedTokens != 700 {
		t.Fatalf("PinnedTokens = %d, want 700", plan.PinnedTokens)
	}
	if !reflect.DeepEqual(plan.Keep, []string{"p1", "p3"}) {
		t.Fatalf("Keep = %v, want the pinned set [p1 p3]", plan.Keep)
	}
}

// TestPlanEvictionShedsEvictableBeforeReplayable is the ORDER the two droppable classes exist to
// express: an evicted REPLAYABLE page costs a page fault, an evicted EVICTABLE page costs
// nothing recoverable — so the cheap loss goes first, and a REPLAYABLE page is only touched once
// the evictable set is exhausted.
func TestPlanEvictionShedsEvictableBeforeReplayable(t *testing.T) {
	pages := []Page{
		{ID: "steer", Kind: KindActiveSteer, Tokens: 100},
		{ID: "tool", Kind: KindCASResult, Tokens: 200},
		{ID: "prose1", Kind: KindTranscriptProse, Tokens: 200},
		{ID: "prose2", Kind: KindTranscriptProse, Tokens: 200},
		{ID: "seed", Kind: KindContinuationSeed, Tokens: 100},
	}
	// total 800, pinned 200. A 500 budget must shed 300: both prose pages (400) clear it before
	// the replayable tool result is ever considered.
	plan := PlanEviction(pages, 500)
	if plan.Refusal != "" {
		t.Fatalf("Refusal = %q, want none (the pinned set fits)", plan.Refusal)
	}
	if !reflect.DeepEqual(plan.Evict, []string{"prose1", "prose2"}) {
		t.Fatalf("Evict = %v, want the evictable prose only [prose1 prose2]", plan.Evict)
	}
	if !reflect.DeepEqual(plan.Keep, []string{"steer", "tool", "seed"}) {
		t.Fatalf("Keep = %v, want [steer tool seed]", plan.Keep)
	}
	if plan.KeptTokens != 400 {
		t.Fatalf("KeptTokens = %d, want 400", plan.KeptTokens)
	}

	// Tighten past what the evictable set alone can pay for and the replayable page joins the
	// shed — but the pinned pages still never do.
	tight := PlanEviction(pages, 250)
	if tight.Refusal != "" {
		t.Fatalf("Refusal = %q, want none (pinned=200 still fits 250)", tight.Refusal)
	}
	if !reflect.DeepEqual(tight.Evict, []string{"tool", "prose1", "prose2"}) {
		t.Fatalf("Evict = %v, want every non-pinned page", tight.Evict)
	}
	if !reflect.DeepEqual(tight.Keep, []string{"steer", "seed"}) {
		t.Fatalf("Keep = %v, want the pinned set [steer seed]", tight.Keep)
	}
}

// TestPlanEvictionUnderBudgetEvictsNothing: a context that already fits is left alone.
func TestPlanEvictionUnderBudgetEvictsNothing(t *testing.T) {
	pages := []Page{
		{ID: "a", Kind: KindTranscriptProse, Tokens: 10},
		{ID: "b", Kind: KindActiveSteer, Tokens: 10},
	}
	plan := PlanEviction(pages, 1000)
	if plan.Refusal != "" || len(plan.Evict) != 0 {
		t.Fatalf("under-budget plan must be inert, got refusal=%q evict=%v", plan.Refusal, plan.Evict)
	}
	if plan.KeptTokens != 20 {
		t.Fatalf("KeptTokens = %d, want 20", plan.KeptTokens)
	}
	if empty := PlanEviction(nil, 0); empty.Refusal != "" || len(empty.Evict) != 0 || len(empty.Keep) != 0 {
		t.Fatalf("the empty page set must plan nothing, got %+v", empty)
	}
}

// TestPlanEvictionIgnoresNegativeTokenCost: a sloppy adapter reporting a negative cost must not
// be able to buy budget headroom for the pages around it.
func TestPlanEvictionIgnoresNegativeTokenCost(t *testing.T) {
	pages := []Page{
		{ID: "bad", Kind: KindTranscriptProse, Tokens: -1000},
		{ID: "pin", Kind: KindActiveSteer, Tokens: 300},
	}
	plan := PlanEviction(pages, 200)
	if plan.PinnedTokens != 300 {
		t.Fatalf("PinnedTokens = %d, want 300 (a negative sibling must not offset it)", plan.PinnedTokens)
	}
	if plan.Refusal != ReasonPinEvictRefused {
		t.Fatalf("Refusal = %q, want %q", plan.Refusal, ReasonPinEvictRefused)
	}
}

// TestCheckEvictionRefusesAPinnedDrop is the verification half: a plan produced elsewhere (a byte
// splicer on a wire body, say) still has to answer to the classes.
func TestCheckEvictionRefusesAPinnedDrop(t *testing.T) {
	pages := []Page{
		{ID: "steer", Kind: KindActiveSteer, Tokens: 10},
		{ID: "prose", Kind: KindTranscriptProse, Tokens: 10},
		{ID: "tool", Kind: KindCASResult, Tokens: 10},
	}
	if got := CheckEviction(pages, []string{"prose", "tool"}); got != "" {
		t.Fatalf("dropping only droppable pages must pass, got %q", got)
	}
	if got := CheckEviction(pages, []string{"prose", "steer"}); got != ReasonPinEvictRefused {
		t.Fatalf("CheckEviction = %q, want %q when the drop includes a PINNED page", got, ReasonPinEvictRefused)
	}
	if got := CheckEviction(pages, nil); got != "" {
		t.Fatalf("an empty drop cannot evict anything, got %q", got)
	}
	if got := CheckEviction(pages, []string{"not-in-this-page-set"}); got != "" {
		t.Fatalf("an ID outside the page set is not this set's business, got %q", got)
	}
}

// TestSurvivalClassStringsAreTheRefusalVocabulary pins the operator-facing tokens: they appear in
// refusals and readouts, so renaming one is a wire change, not a cosmetic edit.
func TestSurvivalClassStringsAreTheRefusalVocabulary(t *testing.T) {
	for class, want := range map[SurvivalClass]string{
		ClassPinned:     "PINNED",
		ClassReplayable: "REPLAYABLE",
		ClassEvictable:  "EVICTABLE",
	} {
		if got := class.String(); got != want {
			t.Errorf("SurvivalClass(%d).String() = %q, want %q", int(class), got, want)
		}
	}
	if ReasonPinEvictRefused != "PIN_EVICT_REFUSED" {
		t.Fatalf("ReasonPinEvictRefused = %q — the token is registered in dos.toml and returned on the wire; it cannot drift", ReasonPinEvictRefused)
	}
}

func TestPlanEvictionOrdersRetentionWithinClass(t *testing.T) {
	t.Run("drop before neutral and keep", func(t *testing.T) {
		pages := []Page{
			{ID: "old-keep", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionKeep, Source: "deterministic:needle"}}},
			{ID: "old-neutral", Kind: KindTranscriptProse, Tokens: 100},
			{ID: "new-drop", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionDrop, Source: "agent:trash-filter"}}},
		}
		plan := PlanEviction(pages, 200)
		if !reflect.DeepEqual(plan.Evict, []string{"new-drop"}) || !reflect.DeepEqual(plan.Keep, []string{"old-keep", "old-neutral"}) {
			t.Fatalf("plan = keep %v evict %v, want the newer drop shed before older neutral/keep", plan.Keep, plan.Evict)
		}
	})
	t.Run("older keep survives over newer neutral", func(t *testing.T) {
		pages := []Page{
			{ID: "old-keep", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionKeep, Source: "deterministic:needle"}}},
			{ID: "new-neutral", Kind: KindTranscriptProse, Tokens: 100},
		}
		plan := PlanEviction(pages, 100)
		if !reflect.DeepEqual(plan.Evict, []string{"new-neutral"}) || !reflect.DeepEqual(plan.Keep, []string{"old-keep"}) {
			t.Fatalf("plan = keep %v evict %v, want older keep to survive over newer neutral", plan.Keep, plan.Evict)
		}
	})
	t.Run("stable input order within equal intent", func(t *testing.T) {
		pages := []Page{
			{ID: "drop-1", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionDrop, Source: "agent:ranker"}}},
			{ID: "drop-2", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionDrop, Source: "agent:ranker"}}},
			{ID: "drop-3", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionDrop, Source: "agent:ranker"}}},
		}
		plan := PlanEviction(pages, 100)
		if !reflect.DeepEqual(plan.Evict, []string{"drop-1", "drop-2"}) || !reflect.DeepEqual(plan.Keep, []string{"drop-3"}) {
			t.Fatalf("plan = keep %v evict %v, want equal-intent pages shed oldest-first", plan.Keep, plan.Evict)
		}
	})
}

func TestPlanEvictionKeepRemainsEvictableAndDropCannotOverridePin(t *testing.T) {
	pages := []Page{
		{ID: "pinned-drop", Kind: KindActiveSteer, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionDrop, Source: "agent:bad-advice"}}},
		{ID: "keep-a", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionKeep, Source: "deterministic:needle"}}},
		{ID: "keep-b", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionKeep, Source: "deterministic:needle"}}},
	}
	plan := PlanEviction(pages, 100)
	if plan.Refusal != "" {
		t.Fatalf("Refusal = %q, want none", plan.Refusal)
	}
	if !reflect.DeepEqual(plan.Evict, []string{"keep-a", "keep-b"}) || !reflect.DeepEqual(plan.Keep, []string{"pinned-drop"}) {
		t.Fatalf("plan = keep %v evict %v, want both keep preferences eventually shed and pinned drop retained", plan.Keep, plan.Evict)
	}
}

func TestPlanEvictionRejectsInvalidRetentionAtomically(t *testing.T) {
	valid := Page{ID: "valid", Kind: KindTranscriptProse, Tokens: 100}
	cases := map[string]Page{
		"unknown intent":      {ID: "bad", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: "forever", Source: "agent:ranker"}}},
		"missing source":      {ID: "bad", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionKeep}}},
		"invalid source kind": {ID: "bad", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionDrop, Source: "human:alice"}}},
		"free form reason":    {ID: "bad", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{{Intent: RetentionKeep, Source: "agent:ranker", ReasonCode: "this is prose"}}},
		"conflict": {ID: "bad", Kind: KindTranscriptProse, Tokens: 100, Retention: []RetentionAnnotation{
			{Intent: RetentionKeep, Source: "agent:ranker"},
			{Intent: RetentionDrop, Source: "deterministic:cleanup"},
		}},
	}
	for name, bad := range cases {
		t.Run(name, func(t *testing.T) {
			plan := PlanEviction([]Page{valid, bad}, 50)
			if plan.Refusal != ReasonRetentionAnnotationInvalid {
				t.Fatalf("Refusal = %q, want %q", plan.Refusal, ReasonRetentionAnnotationInvalid)
			}
			if len(plan.Evict) != 0 || len(plan.Keep) != 0 || plan.KeptTokens != 0 || plan.PinnedTokens != 0 {
				t.Fatalf("invalid annotation must refuse atomically, got %+v", plan)
			}
		})
	}
}
