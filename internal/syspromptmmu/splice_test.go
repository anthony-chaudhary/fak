package syspromptmmu

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/promptmmu"
)

// overlaySeg builds a non-resident overlay segment (the queried harness layer Rung 3
// fills; here it stands in for any after-breakpoint content).
func overlaySeg(text string) cachemeta.PromptSegment {
	return cachemeta.PromptSegment{
		Kind:    cachemeta.SegMessage,
		Content: []byte(text),
		Tokens:  estTokens([]byte(text)),
		Witness: WitnessFor([]byte(text)),
	}
}

// bodyWith builds a minimal Anthropic /v1/messages request body carrying the given
// `system` value verbatim, plus optional extra top-level keys (e.g. tools[]).
func bodyWith(t *testing.T, sysValue []byte, extra map[string]json.RawMessage) []byte {
	t.Helper()
	obj := map[string]json.RawMessage{
		"model":    json.RawMessage(`"claude-x"`),
		"system":   json.RawMessage(sysValue),
		"messages": json.RawMessage(`[{"role":"user","content":"hi"}]`),
	}
	for k, v := range extra {
		obj[k] = v
	}
	raw, err := json.Marshal(obj)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	return raw
}

// decodeOK is a permissive re-decode callback: the spliced body must remain a JSON
// object carrying system + messages.
func decodeOK(b []byte) error {
	var obj struct {
		System   json.RawMessage `json:"system"`
		Messages json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(b, &obj); err != nil {
		return err
	}
	if len(obj.System) == 0 || len(obj.Messages) == 0 {
		return errors.New("missing system or messages")
	}
	return nil
}

// TestBuildSystemValueDeterministic asserts the realize step is byte-deterministic
// (invariant 1) and places exactly one cache_control breakpoint, on the LAST resident
// block — never on a spine block before it and never on an overlay block.
func TestBuildSystemValueDeterministic(t *testing.T) {
	plan := BaseContextPlan()
	overlay := []cachemeta.PromptSegment{overlaySeg("skill: read_file"), overlaySeg("skill: grep")}

	a := BuildSystemValue(plan, overlay)
	b := BuildSystemValue(plan, overlay)
	if !bytes.Equal(a, b) {
		t.Fatal("BuildSystemValue is not byte-deterministic")
	}

	var blocks []textBlock
	if err := json.Unmarshal(a, &blocks); err != nil {
		t.Fatalf("system value is not a block array: %v", err)
	}
	if len(blocks) != len(plan)+len(overlay) {
		t.Fatalf("block count = %d, want %d", len(blocks), len(plan)+len(overlay))
	}
	for i, blk := range blocks {
		hasCC := len(blk.CacheControl) > 0
		wantCC := i == len(plan)-1
		if hasCC != wantCC {
			t.Errorf("block %d cache_control = %v, want %v", i, hasCC, wantCC)
		}
	}
}

// TestSplicePreservesPrefixAcrossTurns is the headline e2e proof (invariants 1 + 2):
// across N turns where ONLY the overlay changes, the resident spine+policy prefix bytes
// are byte-identical, and the overlay region is the only thing that moves.
func TestSplicePreservesPrefixAcrossTurns(t *testing.T) {
	plan := BaseContextPlan()
	body0 := bodyWith(t, BuildSystemValue(plan, nil), nil)

	_, prefixEnd0, _, ok := promptmmu.ArraySplicePoints(body0, "system")
	if !ok {
		t.Fatal("could not anchor the cached prefix on the freshly built body")
	}
	prefix := append([]byte(nil), body0[:prefixEnd0]...) // snapshot the resident prefix

	overlays := [][]cachemeta.PromptSegment{
		{overlaySeg("turn-1 overlay: skill A")},
		{overlaySeg("turn-2 overlay: skill A"), overlaySeg("skill B")},
		nil, // turn 3 evicts the whole overlay
		{overlaySeg("turn-4 overlay: a much longer card body that changes the tail length")},
	}

	cur := body0
	var prevTail []byte
	for turn, ov := range overlays {
		res := SpliceSystemOverlay(cur, plan, ov, decodeOK)
		if !res.Changed {
			t.Fatalf("turn %d: expected a splice, got identity (%s)", turn, res.SkipReason)
		}
		if res.OverlayLen != len(ov) {
			t.Errorf("turn %d: OverlayLen = %d, want %d", turn, res.OverlayLen, len(ov))
		}
		// invariant 1: the resident prefix is byte-identical to the original.
		if len(res.Body) < len(prefix) || !bytes.Equal(res.Body[:len(prefix)], prefix) {
			t.Fatalf("turn %d: resident prefix bytes diverged (invariant 1 violated)", turn)
		}
		// invariant 2: the overlay tail is the only span that changed turn-to-turn.
		tail := res.Body[len(prefix):]
		if prevTail != nil && bytes.Equal(tail, prevTail) {
			t.Errorf("turn %d: overlay tail did not change between distinct overlays", turn)
		}
		prevTail = append([]byte(nil), tail...)
		cur = res.Body
	}
}

// TestSpliceMutatedSpineFailsSafe asserts a body whose resident head is NOT fak's
// authored spine triggers fail-safe identity with the closed SkipSpineMismatch reason —
// fak never splices a head it did not author (no silent corruption).
func TestSpliceMutatedSpineFailsSafe(t *testing.T) {
	plan := BaseContextPlan()

	// A body with the SAME structure (cache_control on the last resident block) but one
	// spine block's text altered — the mutated-spine fixture.
	mutated := make([]cachemeta.PromptSegment, len(plan))
	copy(mutated, plan)
	mutated[0].Content = append(append([]byte(nil), plan[0].Content...), " TAMPERED"...)
	body := bodyWith(t, BuildSystemValue(mutated, nil), nil)

	res := SpliceSystemOverlay(body, plan, []cachemeta.PromptSegment{overlaySeg("x")}, decodeOK)
	if res.Changed {
		t.Fatal("expected fail-safe identity on a mutated spine, got a splice")
	}
	if res.SkipReason != SkipSpineMismatch {
		t.Errorf("SkipReason = %q, want %q", res.SkipReason, SkipSpineMismatch)
	}
	if &res.Body[0] != &body[0] {
		t.Error("identity must return the input slice unchanged")
	}
}

// TestSpliceBreakpointMisplacedFailsSafe asserts that a body whose breakpoint sits
// somewhere other than the last resident block (e.g. a foreign extra cache_control in
// the overlay region) is refused — the breakpoint must be exactly on the policy floor.
func TestSpliceBreakpointMisplacedFailsSafe(t *testing.T) {
	plan := BaseContextPlan()
	// Build a system array where an OVERLAY block also carries cache_control, so the
	// LAST breakpoint lands past the policy floor.
	var blocks []json.RawMessage
	for i, seg := range plan {
		blocks = append(blocks, marshalBlock(seg.Content, i == len(plan)-1))
	}
	blocks = append(blocks, marshalBlock([]byte("overlay with a stray breakpoint"), true))
	sys, err := json.Marshal(blocks)
	if err != nil {
		t.Fatal(err)
	}
	body := bodyWith(t, sys, nil)

	res := SpliceSystemOverlay(body, plan, nil, decodeOK)
	if res.Changed || res.SkipReason != SkipSpineMismatch {
		t.Fatalf("misplaced breakpoint: Changed=%v reason=%q, want identity/%s", res.Changed, res.SkipReason, SkipSpineMismatch)
	}
}

// TestSpliceEmptyOverlayDropsTail asserts splicing a nil overlay evicts the whole
// overlay while preserving the resident prefix and leaving a valid block array.
func TestSpliceEmptyOverlayDropsTail(t *testing.T) {
	plan := BaseContextPlan()
	body := bodyWith(t, BuildSystemValue(plan, []cachemeta.PromptSegment{overlaySeg("A"), overlaySeg("B")}), nil)

	res := SpliceSystemOverlay(body, plan, nil, decodeOK)
	if !res.Changed {
		t.Fatalf("expected a splice, got identity (%s)", res.SkipReason)
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(res.Body, &obj); err != nil {
		t.Fatalf("spliced body is not a JSON object: %v", err)
	}
	var blocks []textBlock
	if err := json.Unmarshal(obj["system"], &blocks); err != nil {
		t.Fatalf("spliced system is not a block array: %v", err)
	}
	if len(blocks) != len(plan) {
		t.Errorf("after emptying overlay: %d blocks, want %d (resident only)", len(blocks), len(plan))
	}
}

// TestSpliceTrustBoundaryUnchanged asserts the splice touches ONLY system text: the
// tools[] block the kernel polices is byte-identical before and after.
func TestSpliceTrustBoundaryUnchanged(t *testing.T) {
	plan := BaseContextPlan()
	tools := json.RawMessage(`[{"name":"read_file","description":"d","input_schema":{"type":"object"}}]`)
	body := bodyWith(t, BuildSystemValue(plan, nil), map[string]json.RawMessage{"tools": tools})

	res := SpliceSystemOverlay(body, plan, []cachemeta.PromptSegment{overlaySeg("new overlay")}, decodeOK)
	if !res.Changed {
		t.Fatalf("expected a splice, got identity (%s)", res.SkipReason)
	}
	var before, after map[string]json.RawMessage
	if err := json.Unmarshal(body, &before); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(res.Body, &after); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before["tools"], after["tools"]) {
		t.Errorf("tools[] changed across the splice:\n before=%s\n after =%s", before["tools"], after["tools"])
	}
}

// TestSpliceReDecodeFailureIsSafe asserts a decode callback that rejects the spliced
// body forces fail-safe identity rather than shipping an unproven body.
func TestSpliceReDecodeFailureIsSafe(t *testing.T) {
	plan := BaseContextPlan()
	body := bodyWith(t, BuildSystemValue(plan, nil), nil)

	res := SpliceSystemOverlay(body, plan, []cachemeta.PromptSegment{overlaySeg("x")},
		func([]byte) error { return errors.New("nope") })
	if res.Changed {
		t.Fatal("expected fail-safe identity when re-decode fails")
	}
	if res.SkipReason != SkipSpliceUnproven {
		t.Errorf("SkipReason = %q, want %q", res.SkipReason, SkipSpliceUnproven)
	}
}

// TestSpliceDegenerateInputs covers every closed-set skip reason for malformed input.
func TestSpliceDegenerateInputs(t *testing.T) {
	plan := BaseContextPlan()
	ov := []cachemeta.PromptSegment{overlaySeg("x")}

	cases := []struct {
		name string
		raw  []byte
		plan []cachemeta.PromptSegment
		want string
	}{
		{"empty-input", nil, plan, SkipEmptyInput},
		{"nil-plan", bodyWith(t, BuildSystemValue(plan, nil), nil), nil, SkipSpineMismatch},
		{"not-json", []byte("not json at all"), plan, SkipNotJSONObject},
		{"no-system", []byte(`{"model":"x","messages":[]}`), plan, SkipNoSystemArray},
		{"system-string", bodyWith(t, []byte(`"just a string system"`), nil), plan, SkipNoSystemArray},
		{"no-breakpoint", bodyWith(t, []byte(`[{"type":"text","text":"foo"}]`), nil), plan, SkipNoBreakpoint},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res := SpliceSystemOverlay(tc.raw, tc.plan, ov, decodeOK)
			if res.Changed {
				t.Fatalf("expected identity, got a splice")
			}
			if res.SkipReason != tc.want {
				t.Errorf("SkipReason = %q, want %q", res.SkipReason, tc.want)
			}
		})
	}
}
