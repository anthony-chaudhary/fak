package agent

import (
	"bytes"
	"encoding/json"
	"testing"
)

// Deterministic regressions for #3773. The cachebp head transforms located the `system`/`tools`
// array with bytes.Index(raw, value), which returns the FIRST byte-identical occurrence. When a
// user message's content array is byte-identical to the head array AND appears earlier in the body
// (messages precedes system in these fixtures), the splice landed in the MESSAGE copy — silently
// caching the volatile conversation tail and, for the upgrade, breaking idempotence. The fix
// anchors by top-level KEY (decodeTopLevelArray / objectValueSpanLastWins). Both fixtures below are
// hand-built so the first byte match is the wrong array; they FAIL on the pre-fix locator and pass
// on the key walk.

// TestPlaceBreakpointNotMisroutedToByteIdenticalMessage is the placement repro: the breakpoint must
// physically land in the system array the outcome names, not messages[0].content.
func TestPlaceBreakpointNotMisroutedToByteIdenticalMessage(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"dup"}]}],` +
		`"system":[{"type":"text","text":"dup"}]}`)

	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone {
		t.Fatalf("want a placement, got reason=%q (out=%q)", oc.Reason, out)
	}
	if oc.Target != "system" {
		t.Fatalf("want target=system, got %q", oc.Target)
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("placed body does not re-decode: %v\nout: %q", err, out)
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(out, &obj); err != nil {
		t.Fatalf("output does not unmarshal: %v\nout: %q", err, out)
	}
	if !bytes.Contains(obj["system"], []byte(cacheControlBreakpoint)) {
		t.Fatalf("breakpoint did NOT land in the system array (misrouted)\nsystem: %q\nout: %q", obj["system"], out)
	}
	if bytes.Contains(obj["messages"], []byte(cacheControlBreakpoint)) {
		t.Fatalf("breakpoint misrouted INTO the messages array — caches the volatile tail\nout: %q", out)
	}
}

// TestUpgradeTTLNotMisroutedToByteIdenticalMessage is the 1h-upgrade repro: the ttl must be spliced
// into the real system breakpoint, not a byte-identical message copy, and the upgrade must stay
// idempotent (before the fix, a second pass located the real system and re-spliced, mutating bytes).
func TestUpgradeTTLNotMisroutedToByteIdenticalMessage(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":1,` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"dup","cache_control":{"type":"ephemeral"}}]}],` +
		`"system":[{"type":"text","text":"dup","cache_control":{"type":"ephemeral"}}]}`)

	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	if oc.Reason != TTLUpgradeReasonNone {
		t.Fatalf("want an upgrade, got reason=%q (out=%q)", oc.Reason, out)
	}
	if oc.Target != "system" {
		t.Fatalf("want target=system, got %q", oc.Target)
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(out, &obj); err != nil {
		t.Fatalf("output does not unmarshal: %v\nout: %q", err, out)
	}
	if !bytes.Contains(obj["system"], []byte(`"ttl":"1h"`)) {
		t.Fatalf("ttl:1h did NOT land in the system array (misrouted)\nsystem: %q\nout: %q", obj["system"], out)
	}
	if bytes.Contains(obj["messages"], []byte(`"ttl":"1h"`)) {
		t.Fatalf("ttl:1h misrouted INTO the messages array\nout: %q", out)
	}
	// Idempotence: a second pass must be identity now that the target is anchored by key.
	again, oc2 := UpgradeAnthropicStableCacheTTL1h(out)
	if !bytes.Equal(again, out) {
		t.Fatalf("upgrade not idempotent: second pass (reason=%q) changed bytes\nfirst:  %q\nsecond: %q", oc2.Reason, out, again)
	}
}
