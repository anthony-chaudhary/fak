package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// responses_prompt_cache_key_test.go pins the Responses-wire cross-shard cache-routing
// lever (the codex half of the managed-cache posture). On /responses the provider prompt
// cache is automatic and prefix-keyed, so fak pins a stable prompt_cache_key derived from
// the cacheable HEAD (model + system instructions + tools). The contract these arms hold:
//
//   - the key is PRESENT and well-formed (32 hex chars) on every Responses request;
//   - it is STABLE across turns of a session (a follow-up that extends the transcript
//     yields the SAME key), so chained turns keep routing to the one warm node;
//   - it is SHARED across sessions with the same fixed harness prefix (the codex win:
//     a constant system prompt stays warm across sessions), yet DISTINCT when the
//     head — system instructions or tools — differs;
//   - it rides BEHIND the model+input head, so it never perturbs the stable leading
//     bytes the automatic prefix cache keys on.

// responsesCacheKeyOf marshals a Responses request and returns its prompt_cache_key.
func responsesCacheKeyOf(t *testing.T, model string, msgs []Message, tools []ToolDef) (string, []byte) {
	t.Helper()
	body, err := openAIResponsesAdapter{}.MarshalRequest(adapterRequest{
		Model: model, Messages: msgs, Tools: tools, MaxTokens: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		PromptCacheKey string `json:"prompt_cache_key"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("decode responses body: %v", err)
	}
	return req.PromptCacheKey, body
}

// TestResponsesPromptCacheKeyPresentAndStable is the core arm: the key is emitted, is a
// 32-char hex routing hint, and is identical for turn N and the turn N+1 extension.
func TestResponsesPromptCacheKeyPresentAndStable(t *testing.T) {
	turn1, turn2 := prefixStabilityTurns(`{"fare":"$420"}`, `{"fare":"$1980"}`)
	tools := adapterTestTools()

	k1, _ := responsesCacheKeyOf(t, "gpt-test", turn1, tools)
	k2, _ := responsesCacheKeyOf(t, "gpt-test", turn2, tools)

	if k1 == "" {
		t.Fatal("prompt_cache_key absent — codex loses the cross-shard cache-routing hint")
	}
	if len(k1) != 32 { //boundarylint:ignore CHANGE_DETECTOR_TEST 32-hex-char (128-bit) routing-hint width is a fixed invariant
		t.Errorf("prompt_cache_key = %q (len %d), want a 32-hex-char routing hint", k1, len(k1))
	}
	if strings.TrimLeft(k1, "0123456789abcdef") != "" {
		t.Errorf("prompt_cache_key = %q, want lowercase hex only", k1)
	}
	if k1 != k2 {
		t.Errorf("prompt_cache_key changed across turns of one session:\n turn1 %s\n turn2 %s\n(a per-turn key defeats the prefix cache it exists to pin)", k1, k2)
	}
}

// TestResponsesPromptCacheKeyTracksHead pins that the key is a pure function of the
// cacheable head: the SAME model+system+tools yields the SAME key even when the
// conversation suffix differs (cross-session warmth sharing), while a different system
// prompt OR a different tool set yields a DIFFERENT key.
func TestResponsesPromptCacheKeyTracksHead(t *testing.T) {
	tools := adapterTestTools()
	base := adapterTestMessages(`{"fare":"$420"}`)
	baseKey, _ := responsesCacheKeyOf(t, "gpt-test", base, tools)

	// Same head, divergent suffix -> shared key (two sessions, one warm prefix node).
	divergentSuffix := append(append([]Message(nil), base...),
		Message{Role: RoleUser, Content: "a completely unrelated follow-up question"})
	if k, _ := responsesCacheKeyOf(t, "gpt-test", divergentSuffix, tools); k != baseKey {
		t.Errorf("key must depend only on the head, not the suffix:\n base      %s\n divergent %s", baseKey, k)
	}

	// Different system instructions -> different key.
	altSystem := append([]Message(nil), base...)
	altSystem[0] = Message{Role: RoleSystem, Content: "DIFFERENT system rules"}
	if k, _ := responsesCacheKeyOf(t, "gpt-test", altSystem, tools); k == baseKey {
		t.Error("a different system prompt must yield a different prompt_cache_key")
	}

	// Different tool set -> different key.
	altTools := []ToolDef{{Type: "function", Function: ToolDefFunction{
		Name: "other", Description: "Other.", Parameters: rawSchema(`{"type":"object"}`),
	}}}
	if k, _ := responsesCacheKeyOf(t, "gpt-test", base, altTools); k == baseKey {
		t.Error("a different tool set must yield a different prompt_cache_key")
	}

	// Different model -> different key (a distinct prefix on a distinct backend).
	if k, _ := responsesCacheKeyOf(t, "gpt-other", base, tools); k == baseKey {
		t.Error("a different model must yield a different prompt_cache_key")
	}
}

// TestResponsesPromptCacheKeyIgnoresLateSystemMessages pins the head anchor (#5186):
// only the LEADING contiguous run of system messages feeds the key. A RoleSystem item
// spliced AFTER a user turn (mid-conversation steering, a spliceSeed carryover) is
// conversation suffix and must NOT churn the key — otherwise every steered turn
// re-routes away from the warm prefix node. A change to the leading head itself must
// still change the key.
func TestResponsesPromptCacheKeyIgnoresLateSystemMessages(t *testing.T) {
	tools := adapterTestTools()
	base := adapterTestMessages(`{"fare":"$420"}`)
	baseKey, _ := responsesCacheKeyOf(t, "gpt-test", base, tools)

	// A system message AFTER a user turn is suffix, not head -> same key.
	lateSystem := append(append([]Message(nil), base...),
		Message{Role: RoleSystem, Content: "mid-conversation steering item"})
	if k, _ := responsesCacheKeyOf(t, "gpt-test", lateSystem, tools); k != baseKey {
		t.Errorf("a RoleSystem message after a user turn must not change the key:\n base %s\n late %s\n(a churning key defeats the prefix cache it exists to pin)", baseKey, k)
	}

	// Same steering item folded INTO the leading head -> different key.
	headGrown := append([]Message{
		base[0],
		{Role: RoleSystem, Content: "mid-conversation steering item"},
	}, base[1:]...)
	if k, _ := responsesCacheKeyOf(t, "gpt-test", headGrown, tools); k == baseKey {
		t.Error("growing the leading instruction head must yield a different prompt_cache_key")
	}
}

// TestResponsesPromptCacheKeyRidesBehindHead pins byte-safety: prompt_cache_key appears
// AFTER the model+input head in the marshaled body, so it never enters the leading bytes
// the provider's automatic prefix cache keys on (the cache-default[22] contract, Responses
// side). Since the key is constant per head it would not bust within-session reuse even if
// it led, but keeping it behind the head matches how every other non-prefix field rides.
func TestResponsesPromptCacheKeyRidesBehindHead(t *testing.T) {
	_, body := responsesCacheKeyOf(t, "gpt-test", adapterTestMessages(`{"fare":"$420"}`), adapterTestTools())
	s := string(body)
	inputAt := strings.Index(s, `"input"`)
	keyAt := strings.Index(s, `"prompt_cache_key"`)
	if inputAt < 0 || keyAt < 0 {
		t.Fatalf("body missing input or prompt_cache_key:\n%s", s)
	}
	if keyAt < inputAt {
		t.Errorf("prompt_cache_key (at %d) precedes the input head (at %d) — it would perturb the stable prefix bytes:\n%s", keyAt, inputAt, s)
	}
	if !strings.HasPrefix(s, `{"model":`) {
		t.Errorf("body no longer starts with the model head:\n%s", s)
	}
}
