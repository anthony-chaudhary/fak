package agent

// zz_redact_probe_testadequacy_test.go — TEMPORARY probe for a test-adequacy review of #2191.
// DELETE BEFORE RETURNING. Exercises untested branches of the redaction retry.

import (
	"bytes"
	"testing"
)

const probeUUID = "123e4567-e89b-12d3-a456-426614174000"

// P1: volatile TOOLS head with NO system at all — retry should convert and place on tools.
func TestProbeToolsOnlyVolatileConverts(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":10,` +
		`"tools":[{"name":"probe","description":"session ` + probeUUID + `","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	t.Logf("default: reason=%q redactReason=%q identity=%v", oc.Reason, oc.RedactReason, bytes.Equal(out, raw))

	t.Setenv(CacheBPRedactEnvVar, "1")
	out, oc = PlaceAnthropicCacheBreakpointWithOutcome(raw)
	t.Logf("lever-on: reason=%q target=%q redacted=%v uuid=%d out=%s", oc.Reason, oc.Target, oc.Redacted, oc.RedactedUUID, out)
	if oc.Reason != BreakpointReasonNone || oc.Target != "tools" {
		t.Errorf("want conversion placed on tools, got %+v", oc)
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Errorf("redecode: %v", err)
	}
}

// P2: redact_unconverted on the TTL path — volatile head whose breakpoint already carries a
// foreign ttl. Once refuses volatile_head; redaction succeeds; retried upgrade refuses
// ttl_already_set -> wrapper must return ORIGINAL bytes + original reason + redact_unconverted.
func TestProbeTTLUnconverted(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "1")
	raw := []byte(`{"model":"m","max_tokens":10,` +
		`"system":[{"type":"text","text":"req ` + probeUUID + `","cache_control":{"type":"ephemeral","ttl":"30m"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	t.Logf("reason=%q redactReason=%q redacted=%v identity=%v", oc.Reason, oc.RedactReason, oc.Redacted, bytes.Equal(out, raw))
	if oc.Reason != TTLUpgradeReasonVolatileHead || oc.RedactReason != RedactReasonUnconverted || !bytes.Equal(out, raw) {
		t.Errorf("want original volatile_head + redact_unconverted identity, got %+v", oc)
	}
}

// P3: redact_unconverted on the PLACEMENT path — system array of volatile STRING elements:
// redaction stabilizes the head but the splice needs an object, so the retry refuses.
func TestProbePlacementUnconverted(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "1")
	raw := []byte(`{"model":"m","max_tokens":10,` +
		`"system":["req ` + probeUUID + `"],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := placeAnthropicCacheBreakpointOnce(raw)
	t.Logf("once: reason=%q", oc.Reason)
	out, oc = PlaceAnthropicCacheBreakpointWithOutcome(raw)
	t.Logf("wrapped: reason=%q redactReason=%q identity=%v", oc.Reason, oc.RedactReason, bytes.Equal(out, raw))
}

// P4: duplicate top-level "system" keys, LAST volatile — engine redacts the last-wins span
// only; what does the converted body still carry?
func TestProbeDuplicateSystemKeys(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "1")
	raw := []byte(`{"model":"m","max_tokens":10,` +
		`"system":[{"type":"text","text":"first dup ` + probeUUID + `"}],` +
		`"system":[{"type":"text","text":"second dup ` + probeUUID + `"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	t.Logf("reason=%q redacted=%v uuid=%d\nout=%s", oc.Reason, oc.Redacted, oc.RedactedUUID, out)
	t.Logf("raw uuid still present in converted body: %v", bytes.Contains(out, []byte(probeUUID)))
}

// P5: seconds-only (no zone) timestamp + MULTIPLE volatile tokens in one block — residue and counts.
func TestProbeSecondsOnlyAndMultiTokens(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "1")
	raw := []byte(`{"model":"m","max_tokens":10,` +
		`"system":[{"type":"text","text":"a ` + probeUUID + ` b ffffffff-ffff-4fff-8fff-ffffffffffff at 2026-07-13T09:15:02 and 2026-07-13 23:59:58"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	t.Logf("reason=%q uuid=%d ts=%d\nout=%s", oc.Reason, oc.RedactedUUID, oc.RedactedTimestamp, out)
	for _, residue := range []string{":02", ":58", "09:15", "23:59"} {
		if bytes.Contains(out, []byte(residue)) {
			t.Errorf("sub-day residue %q survived", residue)
		}
	}
	if oc.RedactedUUID != 2 || oc.RedactedTimestamp != 2 {
		t.Errorf("counts = %d uuid / %d ts, want 2/2", oc.RedactedUUID, oc.RedactedTimestamp)
	}
}

// P6: system is a plain volatile STRING (Anthropic allows a string system) + volatile tools —
// the retry triggers on tools volatility; does the engine handle the string system?
func TestProbeSystemStringVolatile(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "1")
	raw := []byte(`{"model":"m","max_tokens":10,` +
		`"system":"req ` + probeUUID + `",` +
		`"tools":[{"name":"probe","description":"session ` + probeUUID + `","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	t.Logf("reason=%q target=%q uuid=%d\nout=%s", oc.Reason, oc.Target, oc.RedactedUUID, out)
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Errorf("redecode: %v", err)
	}

	// And system-as-string volatile ALONE (no tools): which reason does Once give? (no retry expected)
	raw2 := []byte(`{"model":"m","max_tokens":10,"system":"req ` + probeUUID + `","messages":[{"role":"user","content":"hi"}]}`)
	_, oc2 := PlaceAnthropicCacheBreakpointWithOutcome(raw2)
	t.Logf("string-system-alone: reason=%q redactReason=%q", oc2.Reason, oc2.RedactReason)
}

// P7: TTL upgrade, TOOLS target — client breakpoint on a tools entry, uuid in an earlier tool.
func TestProbeTTLToolsTargetVolatile(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":10,` +
		`"tools":[{"name":"a","description":"session ` + probeUUID + `","input_schema":{"type":"object"}},` +
		`{"name":"b","description":"stable","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	_, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	t.Logf("default: reason=%q target=%q redactReason=%q", oc.Reason, oc.Target, oc.RedactReason)

	t.Setenv(CacheBPRedactEnvVar, "1")
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	t.Logf("lever-on: reason=%q target=%q redacted=%v uuid=%d\nout=%s", oc.Reason, oc.Target, oc.Redacted, oc.RedactedUUID, out)
	if oc.Reason != TTLUpgradeReasonNone || oc.Target != "tools" {
		t.Errorf("want converted tools upgrade, got %+v", oc)
	}
}

// P8: TTL upgrade, SYSTEM breakpoint with volatile TOOLS (inheritedVolatile path).
func TestProbeTTLInheritedToolsVolatile(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "1")
	raw := []byte(`{"model":"m","max_tokens":10,` +
		`"tools":[{"name":"a","description":"session ` + probeUUID + `","input_schema":{"type":"object"}}],` +
		`"system":[{"type":"text","text":"stable","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	t.Logf("reason=%q target=%q redacted=%v uuid=%d\nout=%s", oc.Reason, oc.Target, oc.Redacted, oc.RedactedUUID, out)
	if oc.Reason != TTLUpgradeReasonNone {
		t.Errorf("want converted, got %+v", oc)
	}
	if bytes.Contains(out, []byte(probeUUID)) {
		t.Errorf("volatile tools uuid survived")
	}
}

// P9: converted body then CompactAnthropicHistory — cross-transform sanity.
func TestProbeCompactAfterRedactedPlace(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "1")
	raw := []byte(`{"model":"m","max_tokens":10,` +
		`"system":[{"type":"text","text":"boot ` + probeUUID + `"}],` +
		`"messages":[{"role":"user","content":"` + strings0(400) + `"},{"role":"assistant","content":"y"},{"role":"user","content":"z"}]}`)
	placed, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	t.Logf("place: reason=%q redacted=%v", oc.Reason, oc.Redacted)
	compacted, coc := CompactAnthropicHistoryWithOutcome(placed, 100)
	t.Logf("compact: reason=%q identity=%v", coc.Reason, bytes.Equal(compacted, placed))
	if _, err := DecodeAnthropicMessagesRequest(compacted); err != nil {
		t.Errorf("compacted redecode: %v", err)
	}
	if !bytes.Contains(compacted, []byte(redactedUUIDPlaceholder)) {
		t.Errorf("placeholder head lost through compaction:\n%s", compacted)
	}
}

func strings0(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = 'a'
	}
	return string(b)
}
