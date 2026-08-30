package gateway

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// pinSurvivalGoalMarker is the wire sentinel a pinned steer carries. It is spelled out here (the
// compactor's own constant is unexported) but is NOT the thing under test: the classifier calls
// agent.IsGoalPinnedMessage, so TestPinnedPageSurvivesCompaction fails loudly if this literal ever
// stops being what the compactor honours — the goal turn would simply not be hoisted.
const pinSurvivalGoalMarker = "[fak:goal]"

// pinSurvivalWireBody builds a real-Claude-Code-shaped Anthropic body: a stable system head that
// marks its OWN cache_control (so head re-anchoring has an anchor), nMsgs alternating turns, a
// recent message breakpoint two turns from the end (which therefore sits inside the kept window,
// not the dropped middle), and a [fak:goal]-marked steer at goalIdx in the compactible middle.
//
// goalPad scales the steer's size, which is what lets one builder produce both the ordinary case
// and the case where the pinned floor cannot fit the budget.
func pinSurvivalWireBody(t *testing.T, nMsgs, goalIdx, goalPad int) []byte {
	t.Helper()
	type block map[string]any
	recentBpIdx := nMsgs - 3
	msgs := make([]map[string]any, 0, nMsgs)
	for i := 0; i < nMsgs; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		text := strings.Repeat("conversation turn body words. ", 12)
		if i == goalIdx {
			role = "user"
			text = pinSurvivalGoalMarker + " standing instruction: never rewrite the vendored tree. " +
				strings.Repeat("steer detail. ", goalPad)
		}
		blk := block{"type": "text", "text": text}
		if i == recentBpIdx {
			blk["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		msgs = append(msgs, map[string]any{"role": role, "content": []block{blk}})
	}
	ordered := struct {
		Model     string           `json:"model"`
		MaxTokens int              `json:"max_tokens"`
		System    []block          `json:"system"`
		Messages  []map[string]any `json:"messages"`
	}{
		Model:     "claude-sonnet-4-6",
		MaxTokens: 1024,
		System: []block{
			{"type": "text", "text": "You are a coding agent."},
			{"type": "text", "text": strings.Repeat("policy text. ", 40), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		Messages: msgs,
	}
	raw, err := json.Marshal(ordered)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}

// pinSurvivalPinnedBytes returns the verbatim messages[] element bytes of every page the gate
// classes PINNED — the exact byte strings a compacted body must still contain.
func pinSurvivalPinnedBytes(t *testing.T, raw []byte) [][]byte {
	t.Helper()
	_, pinned, ok := anthropicSurvivalPages(raw)
	if !ok {
		t.Fatal("fixture sanity: the body has no classifiable eviction domain")
	}
	if len(pinned) < 2 {
		t.Fatalf("fixture sanity: want at least the steer + the continuation seed pinned, got %d", len(pinned))
	}
	return pinned
}

// TestPinnedPageSurvivesCompaction is the acceptance witness for #2421: across 100 forced
// compactions of a synthetic long session — one per turn as the session GROWS, which is the shape
// that actually strands state upstream — every PINNED page's bytes come out of the compactor
// byte-identical, and the body still decodes.
//
// This is stronger than "the compactor happened to keep them": the classification is recomputed
// from the pre-compaction body each round (so the continuation seed moves as the session grows),
// and the check is a verbatim byte-substring match against the compacted output.
func TestPinnedPageSurvivesCompaction(t *testing.T) {
	const rounds = 100
	fired := 0
	for round := 0; round < rounds; round++ {
		raw := pinSurvivalWireBody(t, 120+2*round, 10, 3)
		req, err := agent.DecodeAnthropicMessagesRequest(raw)
		if err != nil {
			t.Fatalf("round %d: decode: %v", round, err)
		}
		pinned := pinSurvivalPinnedBytes(t, raw)

		s := anthropicPassthroughServer(1200)
		s.compactAnchorHead = true
		ok, reason := s.compactAnthropicRawWithReason(req, 1000, "")
		if reason == agent.CompactReasonPinEvictRefused {
			t.Fatalf("round %d: a budget that clears the pinned floor must not refuse", round)
		}
		if ok {
			fired++
			if len(req.Raw) >= len(raw) {
				t.Fatalf("round %d: a fire must shrink the body, got %d (in %d)", round, len(req.Raw), len(raw))
			}
		}
		for i, p := range pinned {
			if !bytes.Contains(req.Raw, p) {
				t.Fatalf("round %d: PINNED page %d did not survive compaction byte-identical (fired=%v reason=%q)\npage: %s",
					round, i, ok, reason, truncateForLog(p))
			}
		}
		if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
			t.Fatalf("round %d: compacted body failed to decode: %v", round, err)
		}
	}
	if fired != rounds {
		t.Fatalf("only %d/%d rounds actually compacted — a survival witness over bodies that were never compacted proves nothing", fired, rounds)
	}
}

// TestPinEvictRefused is the second acceptance witness: a budget too small for the pinned set
// yields PIN_EVICT_REFUSED and forwards the body UNCHANGED, rather than compacting lossily.
//
// The counterfactual runs the IDENTICAL body with only the budget raised above the pinned floor
// and gets a clean fire, so the refusal is attributable to the budget-vs-pinned-floor relation and
// not to any of the compactor's other bails.
func TestPinEvictRefused(t *testing.T) {
	// A steer large enough that it alone overruns the small budget below.
	raw := pinSurvivalWireBody(t, 120, 10, 1200)

	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	pinned := pinSurvivalPinnedBytes(t, raw)

	tight := anthropicPassthroughServer(1200)
	tight.compactAnchorHead = true
	fired, reason := tight.compactAnthropicRawWithReason(req, 1000, "")
	if fired {
		t.Fatalf("a budget below the pinned floor must NOT compact, got fired=true reason=%q", reason)
	}
	if reason != agent.CompactReasonPinEvictRefused {
		t.Fatalf("reason = %q, want %q", reason, agent.CompactReasonPinEvictRefused)
	}
	if !bytes.Equal(req.Raw, raw) {
		t.Fatal("a PIN_EVICT_REFUSED must forward the body byte-identical — a refusal that still rewrote the body would be the lossy compaction it exists to prevent")
	}
	for i, p := range pinned {
		if !bytes.Contains(req.Raw, p) {
			t.Fatalf("PINNED page %d missing from a refused body", i)
		}
	}

	// Counterfactual: same body, a budget that clears the pinned floor — fires cleanly.
	reqOK, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode counterfactual: %v", err)
	}
	roomy := anthropicPassthroughServer(6000)
	roomy.compactAnchorHead = true
	firedOK, reasonOK := roomy.compactAnthropicRawWithReason(reqOK, 1000, "")
	if !firedOK || reasonOK != agent.CompactReasonNone {
		t.Fatalf("the same body under a budget above the pinned floor must FIRE, got fired=%v reason=%q — without this the refusal above is not attributable to the budget", firedOK, reasonOK)
	}
	for i, p := range pinned {
		if !bytes.Contains(reqOK.Raw, p) {
			t.Fatalf("PINNED page %d did not survive the counterfactual fire", i)
		}
	}
}

// TestPinEvictRefusedIsARegisteredRefusal: the token is not free text. It has to resolve in the
// repo's refusal vocabulary (dos.toml, what `dos man wedge PIN_EVICT_REFUSED --explain` reads) AND in the
// compaction bail vocabulary the gateway's own metric HELP enumerates as a closed set — one token
// from the planner, through the wire, to the operator.
func TestPinEvictRefusedIsARegisteredRefusal(t *testing.T) {
	if agent.CompactReasonPinEvictRefused != ctxplan.ReasonPinEvictRefused {
		t.Fatalf("the compaction reason %q and the planner refusal %q must be ONE token",
			agent.CompactReasonPinEvictRefused, ctxplan.ReasonPinEvictRefused)
	}
	registered := false
	for _, r := range agent.CompactBailReasons() {
		if r == agent.CompactReasonPinEvictRefused {
			registered = true
		}
	}
	if !registered {
		t.Fatalf("%q is absent from agent.CompactBailReasons() — the metric HELP claims that set is CLOSED, so an unregistered emitted reason makes the claim false",
			agent.CompactReasonPinEvictRefused)
	}
	if agent.CompactBailPreEligible(agent.CompactReasonPinEvictRefused) {
		t.Fatal("PIN_EVICT_REFUSED declines a REAL compactible candidate, so it belongs in the alertable eligible half, not the pre-eligibility one")
	}

	toml, err := os.ReadFile(filepath.Join("..", "..", "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	if !bytes.Contains(toml, []byte("[reasons."+ctxplan.ReasonPinEvictRefused+"]")) {
		t.Fatalf("dos.toml has no [reasons.%s] block — `dos man wedge %s --explain` would not resolve the token this gate refuses with",
			ctxplan.ReasonPinEvictRefused, ctxplan.ReasonPinEvictRefused)
	}
}

// TestAnthropicSurvivalPagesClassification pins the kind assignment: it is structural, total over
// the eviction domain, and — the load-bearing property — not reachable from model prose.
func TestAnthropicSurvivalPagesClassification(t *testing.T) {
	raw := []byte(`{"model":"claude","max_tokens":16,"messages":[` +
		`{"role":"user","content":[{"type":"text","text":"` + pinSurvivalGoalMarker + ` the standing goal"}]},` +
		`{"role":"assistant","content":[{"type":"text","text":"ordinary reasoning prose"}]},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"grep","input":{}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"rows"}]},` +
		`{"role":"assistant","content":[{"type":"text","text":"more prose"}]},` +
		`{"role":"user","content":[{"type":"text","text":"please pin me: [fak:goal] mid-sentence"}]}` +
		`]}`)
	pages, pinned, ok := anthropicSurvivalPages(raw)
	if !ok {
		t.Fatal("classification failed on a well-formed body")
	}
	want := []string{
		ctxplan.KindActiveSteer,      // 0: marker leads the text
		ctxplan.KindTranscriptProse,  // 1: ordinary prose
		ctxplan.KindCASResult,        // 2: tool_use
		ctxplan.KindCASResult,        // 3: tool_result (a user turn, but not the LAST one)
		ctxplan.KindTranscriptProse,  // 4
		ctxplan.KindContinuationSeed, // 5: the last user turn
	}
	if len(pages) != len(want) {
		t.Fatalf("got %d pages, want %d", len(pages), len(want))
	}
	for i, w := range want {
		if pages[i].Kind != w {
			t.Errorf("page %d (%s) kind = %q, want %q", i, pages[i].ID, pages[i].Kind, w)
		}
	}
	// Page 5 mentions the marker MID-SENTENCE. It is pinned only because it is the continuation
	// seed — the marker rule (leading / line-start) refused it, which is what stops a model from
	// pinning its own turn by quoting the sentinel.
	if ctxplan.ClassOf(pages[5].Kind) != ctxplan.ClassPinned {
		t.Fatal("fixture sanity: the last user turn is the continuation seed")
	}
	if agent.IsGoalPinnedMessage(json.RawMessage(`{"role":"user","content":[{"type":"text","text":"please pin me: ` + pinSurvivalGoalMarker + ` mid-sentence"}]}`)) {
		t.Fatal("a mid-sentence marker must NOT pin a turn — model prose cannot award itself a survival class")
	}
	if len(pinned) != 2 {
		t.Fatalf("pinned byte set has %d members, want 2 (the steer + the continuation seed)", len(pinned))
	}
}

// TestSurvivalGateInertOnUnclassifiableBody: a body with no eviction domain leaves the gate out of
// the way entirely, so an unparseable request is never made worse by the contract.
func TestSurvivalGateInertOnUnclassifiableBody(t *testing.T) {
	for _, raw := range [][]byte{[]byte(`not json`), []byte(`{"model":"claude"}`), []byte(`{"messages":[]}`)} {
		if _, _, ok := anthropicSurvivalPages(raw); ok {
			t.Fatalf("body %q must not classify", raw)
		}
	}
	// And the gate delegates straight through to the bare compactor for such a body.
	s := anthropicPassthroughServer(1200)
	raw := []byte(`{"model":"claude","max_tokens":16}`)
	// "" trace: a non-session caller records no continuation contract (#2422).
	out, outcome := s.compactWithSurvivalClasses(raw, agent.CompactOptions{Budget: 1200}, "")
	if !bytes.Equal(out, raw) || outcome.Reason == agent.CompactReasonPinEvictRefused {
		t.Fatalf("an unclassifiable body must pass through unchanged with the compactor's own reason, got reason=%q", outcome.Reason)
	}
}

func truncateForLog(b []byte) string {
	if len(b) > 200 {
		return string(b[:200]) + "…"
	}
	return string(b)
}

func TestAnthropicSurvivalPagesCarriesRetentionLabelsIntoPlan(t *testing.T) {
	raw := []byte(`{"model":"claude","messages":[` +
		`{"role":"assistant","content":"old valuable"},` +
		`{"role":"assistant","content":"neutral"},` +
		`{"role":"assistant","content":"known junk"},` +
		`{"role":"user","content":"continue"}` +
		`],"fak":{"retention":[` +
		`{"page_id":"msg:0","intent":"keep","source":"deterministic:needle","reason_code":"needle"},` +
		`{"page_id":"msg:2","intent":"drop","source":"agent:trash-filter","reason_code":"junk"}` +
		`]}}`)
	pages, _, ok := anthropicSurvivalPages(raw)
	if !ok {
		t.Fatal("anthropicSurvivalPages did not classify a valid body")
	}
	if got := pages[0].Retention; len(got) != 1 || got[0].Intent != ctxplan.RetentionKeep || got[0].Source != "deterministic:needle" {
		t.Fatalf("msg:0 retention = %+v, want keep from deterministic:needle", got)
	}
	if got := pages[2].Retention; len(got) != 1 || got[0].Intent != ctxplan.RetentionDrop || got[0].Source != "agent:trash-filter" {
		t.Fatalf("msg:2 retention = %+v, want drop from agent:trash-filter", got)
	}

	budget := pages[0].Tokens + pages[3].Tokens
	plan := ctxplan.PlanEviction(pages, budget)
	if plan.Refusal != "" {
		t.Fatalf("Refusal = %q, want none", plan.Refusal)
	}
	if !reflect.DeepEqual(plan.Evict, []string{"msg:1", "msg:2"}) || !reflect.DeepEqual(plan.Keep, []string{"msg:0", "msg:3"}) {
		t.Fatalf("plan = keep %v evict %v, want drop label first and older keep retained over neutral", plan.Keep, plan.Evict)
	}
}

func TestAnthropicSurvivalPagesUnknownRetentionAddressFailsClosed(t *testing.T) {
	raw := []byte(`{"messages":[{"role":"assistant","content":"old"},{"role":"user","content":"continue"}],` +
		`"fak":{"retention":[{"page_id":"msg:99","intent":"drop","source":"agent:ranker"}]}}`)
	pages, _, ok := anthropicSurvivalPages(raw)
	if !ok {
		t.Fatal("valid message envelope must remain classifiable")
	}
	plan := ctxplan.PlanEviction(pages, 0)
	if plan.Refusal != ctxplan.ReasonRetentionAnnotationInvalid || len(plan.Evict) != 0 {
		t.Fatalf("plan = %+v, want atomic invalid-annotation refusal", plan)
	}
}

func TestCompactWithSurvivalClassesAppliesAnnotatedPlan(t *testing.T) {
	raw := []byte(`{"model":"claude","messages":[` +
		`{"role":"assistant","content":"valuable needle"},` +
		`{"role":"assistant","content":"ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle ordinary middle"},` +
		`{"role":"assistant","content":"known junk known junk known junk known junk known junk known junk known junk known junk known junk known junk known junk known junk known junk known junk known junk known junk"},` +
		`{"role":"user","content":"continue task"}` +
		`],"fak":{"retention":[` +
		`{"page_id":"msg:0","intent":"keep","source":"deterministic:needle"},` +
		`{"page_id":"msg:2","intent":"drop","source":"agent:ranker"}` +
		`]}}`)
	if _, _, ok := anthropicSurvivalPages(raw); !ok {
		t.Fatal("fixture did not classify")
	}
	// Leave room for the two tombstone message frames while requiring both large middle pages
	// to be evicted. The postcondition checks the actual serialized resident cost.
	s := anthropicPassthroughServer(60)
	out, outcome := s.compactWithSurvivalClasses(raw, agent.CompactOptions{Budget: 1}, "")
	if outcome.Reason != agent.CompactReasonNone || outcome.Dropped != 2 {
		t.Fatalf("outcome = %+v, want a two-page annotated compaction", outcome)
	}
	if !bytes.Contains(out, []byte("valuable needle")) || !bytes.Contains(out, []byte("continue task")) {
		t.Fatalf("planned keep pages did not survive: %s", out)
	}
	if bytes.Contains(out, []byte("ordinary middle ordinary")) || bytes.Contains(out, []byte("known junk known")) {
		t.Fatalf("planned evictions remained in provider body: %s", out)
	}
	if bytes.Contains(out, []byte(`"retention"`)) || bytes.Contains(out, []byte(`"fak"`)) {
		t.Fatalf("kernel-only retention transport leaked to provider body: %s", out)
	}
}

func TestCompactWithSurvivalClassesMalformedRetentionFailsClosedAndStripsExtension(t *testing.T) {
	tests := []struct {
		name      string
		retention string
	}{
		{name: "object instead of array", retention: `{}`},
		{name: "non-object label", retention: `["drop"]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := []byte(`{"model":"claude","messages":[` +
				`{"role":"assistant","content":"old context"},` +
				`{"role":"user","content":"continue"}` +
				`],"fak":{"retention":` + tt.retention + `}}`)
			s := anthropicPassthroughServer(1)
			out, outcome := s.compactWithSurvivalClasses(raw, agent.CompactOptions{Budget: 1}, "")
			if outcome.Reason != ctxplan.ReasonRetentionAnnotationInvalid || outcome.Dropped != 0 {
				t.Fatalf("outcome = %+v, want atomic %s refusal", outcome, ctxplan.ReasonRetentionAnnotationInvalid)
			}
			if bytes.Contains(out, []byte(`"retention"`)) || bytes.Contains(out, []byte(`"fak"`)) {
				t.Fatalf("malformed kernel-only retention transport leaked: %s", out)
			}
			if !bytes.Contains(out, []byte("old context")) || !bytes.Contains(out, []byte("continue")) {
				t.Fatalf("refusal changed message history: %s", out)
			}
		})
	}
}

func TestCompactWithSurvivalClassesRejectsAnnotatedOutputOverActualBudget(t *testing.T) {
	raw := []byte(`{"model":"claude","messages":[` +
		`{"role":"assistant","content":"x"},` +
		`{"role":"user","content":"continue"}` +
		`],"fak":{"retention":[` +
		`{"page_id":"msg:0","intent":"drop","source":"agent:ranker"}` +
		`]}}`)
	s := anthropicPassthroughServer(1)
	out, outcome := s.compactWithSurvivalClasses(raw, agent.CompactOptions{Budget: 1}, "")
	if outcome.Reason != agent.CompactReasonPinEvictRefused || outcome.Dropped != 0 {
		t.Fatalf("outcome = %+v, want refusal when actual tombstoned output exceeds budget", outcome)
	}
	if !bytes.Contains(out, []byte(`"content":"x"`)) {
		t.Fatalf("budget refusal must preserve the original message: %s", out)
	}
	if bytes.Contains(out, []byte(`"retention"`)) || bytes.Contains(out, []byte(`"fak"`)) {
		t.Fatalf("kernel-only retention transport leaked on budget refusal: %s", out)
	}
}

func TestCompactWithSurvivalClassesRefusesIndependentToolHistoryTombstone(t *testing.T) {
	raw := []byte(`{"model":"claude","messages":[` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"grep","input":{}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"rows"}]},` +
		`{"role":"user","content":"continue"}` +
		`],"fak":{"retention":[` +
		`{"page_id":"msg:0","intent":"drop","source":"agent:ranker"}` +
		`]}}`)
	s := anthropicPassthroughServer(1)
	out, outcome := s.compactWithSurvivalClasses(raw, agent.CompactOptions{Budget: 1}, "")
	if outcome.Reason != agent.CompactReasonPinEvictRefused || outcome.Dropped != 0 {
		t.Fatalf("outcome = %+v, want atomic refusal for independently selected tool history", outcome)
	}
	if !bytes.Contains(out, []byte(`"type":"tool_use"`)) || !bytes.Contains(out, []byte(`"type":"tool_result"`)) {
		t.Fatalf("tool history was partially rewritten: %s", out)
	}
	if bytes.Contains(out, []byte(`context page msg:0 evicted`)) {
		t.Fatalf("tool history was tombstoned independently: %s", out)
	}
	if bytes.Contains(out, []byte(`"retention"`)) || bytes.Contains(out, []byte(`"fak"`)) {
		t.Fatalf("kernel-only retention transport leaked on tool-history refusal: %s", out)
	}
}

func TestCompactWithSurvivalClassesStripsEmptyRetentionWithoutChangingLegacyHistory(t *testing.T) {
	raw := []byte(`{"model":"claude","messages":[` +
		`{"role":"assistant","content":"old context"},` +
		`{"role":"user","content":"continue"}` +
		`],"fak":{"retention":[]}}`)
	s := anthropicPassthroughServer(100)
	out, outcome := s.compactWithSurvivalClasses(raw, agent.CompactOptions{Budget: 0}, "")
	if outcome.Reason != agent.CompactReasonUnderBudget || outcome.Dropped != 0 {
		t.Fatalf("outcome = %+v, want legacy identity behavior", outcome)
	}
	if !bytes.Contains(out, []byte("old context")) || !bytes.Contains(out, []byte("continue")) {
		t.Fatalf("empty retention changed legacy message history: %s", out)
	}
	if bytes.Contains(out, []byte(`"retention"`)) || bytes.Contains(out, []byte(`"fak"`)) {
		t.Fatalf("empty kernel-only retention transport leaked: %s", out)
	}
}
