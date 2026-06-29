package syspromptmmu

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

var update = flag.Bool("update", false, "regenerate the golden base-context plan")

const goldenPath = "testdata/base-context-plan.golden"

// render produces the canonical textual form of the plan the golden pins: every
// segment's tier/kind/evictability/tokens/witness header followed by its exact content,
// then the plan-level digest.
func render() []byte {
	var b bytes.Buffer
	for _, s := range BaseContext() {
		fmt.Fprintf(&b, "tier=%s kind=%s evictable=%t tokens=%d witness=%s\n",
			s.Tier, s.Kind, !NonEvictable(s.Tier), s.Tokens, s.Witness)
		b.Write(s.Content)
		b.WriteString("\n----\n")
	}
	fmt.Fprintf(&b, "digest=%s\n", PlanDigest())
	return b.Bytes()
}

// TestBaseContextPlanGolden pins the emitted plan byte-for-byte. A change to the spine
// or policy content, the tier order, or the witness derivation fails here — the
// deterministic authorship contract (invariant 1) made visible. Regenerate
// intentionally with:
//
//	go test ./internal/syspromptmmu -run TestBaseContextPlanGolden -update
func TestBaseContextPlanGolden(t *testing.T) {
	got := render()
	if *update {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote golden %s (%d bytes)", goldenPath, len(got))
		return
	}
	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden (run with -update to create): %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("base-context plan drifted from golden %s.\n--- got ---\n%s\n--- want ---\n%s",
			goldenPath, got, want)
	}
}

// TestDeterministic asserts BaseContextPlan is byte-identical across calls (invariant 1:
// same inputs → byte-identical segment contents).
func TestDeterministic(t *testing.T) {
	a := BaseContextPlan()
	b := BaseContextPlan()
	if !reflect.DeepEqual(a, b) {
		t.Fatal("BaseContextPlan is not deterministic across calls")
	}
	if len(a) == 0 {
		t.Fatal("BaseContextPlan is empty")
	}
}

// TestEverySegmentHasWitness asserts every emitted segment is a SegStable segment with
// non-empty content and a non-empty, content-derived Witness that re-derives to its own
// content hash (so a later turn can prove the spine is unchanged).
func TestEverySegmentHasWitness(t *testing.T) {
	segs := BaseContext()
	if len(segs) == 0 {
		t.Fatal("base context is empty")
	}
	for i, s := range segs {
		if s.Witness == "" {
			t.Errorf("segment %d (%s) has an empty Witness", i, s.Tier)
		}
		if s.Kind != cachemeta.SegStable {
			t.Errorf("segment %d (%s) kind = %q, want SegStable", i, s.Tier, s.Kind)
		}
		if len(s.Content) == 0 {
			t.Errorf("segment %d (%s) has empty content", i, s.Tier)
		}
		if got := WitnessFor(s.Content); got != s.Witness {
			t.Errorf("segment %d (%s) witness %q != recomputed %q", i, s.Tier, s.Witness, got)
		}
	}
}

// TestTierLayout asserts the plan is the spine tier followed by the policy tier, with no
// overlay segments (this rung emits a spine + policy floor only) — the head→tail order
// the attention geometry forces.
func TestTierLayout(t *testing.T) {
	var sawPolicy bool
	var spineN, policyN int
	for i, s := range BaseContext() {
		switch s.Tier {
		case TierSpine:
			if sawPolicy {
				t.Errorf("segment %d: a spine segment appears after a policy segment (order violated)", i)
			}
			spineN++
		case TierPolicy:
			sawPolicy = true
			policyN++
		case TierOverlay:
			t.Errorf("segment %d: this rung must not emit an overlay segment", i)
		default:
			t.Errorf("segment %d: unknown tier %d", i, s.Tier)
		}
	}
	if spineN == 0 {
		t.Error("no spine segments emitted")
	}
	if policyN == 0 {
		t.Error("no policy-floor segments emitted")
	}
}

// TestNonEvictable asserts the spine and policy tiers are flagged non-evictable
// (invariant 3 substrate) and the overlay tier is evictable.
func TestNonEvictable(t *testing.T) {
	if !NonEvictable(TierSpine) {
		t.Error("spine tier must be non-evictable")
	}
	if !NonEvictable(TierPolicy) {
		t.Error("policy tier must be non-evictable")
	}
	if NonEvictable(TierOverlay) {
		t.Error("overlay tier must be evictable")
	}
}

// TestPlanProjectionMatches asserts the flat BaseContextPlan is exactly the embedded
// PromptSegment of each tier-tagged BaseContext segment, in order.
func TestPlanProjectionMatches(t *testing.T) {
	segs := BaseContext()
	plan := BaseContextPlan()
	if len(plan) != len(segs) {
		t.Fatalf("plan len %d != tagged len %d", len(plan), len(segs))
	}
	for i := range segs {
		if !reflect.DeepEqual(plan[i], segs[i].PromptSegment) {
			t.Errorf("segment %d: plan projection differs from tagged segment", i)
		}
	}
}
