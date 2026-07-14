package agent

import (
	"bytes"
	"strings"
	"testing"
)

// anthropic_cachebp_redact_test.go — witnesses for the volatile-head redaction spec (#2191):
// the repro of the permanent volatile_head dead-end, its opt-in conversion on both refusal
// sites (TTL upgrade, placement), the byte-safety proofs (identity off-lever, untouched
// messages tail, idempotence), the cross-turn stability that makes the provider prefix
// cacheable again, and the prefer-hoist-over-strip boundary (#2181).

const redactTestUUID = "123e4567-e89b-12d3-a456-426614174000"

// ttlUpgradeBodyWithUUID is a client-managed layout: a breakpoint already on the system head,
// whose covered span carries a per-request UUID — the exact shape that permanently refused
// the managed-cache TTL upgrade with volatile_head before #2191.
func ttlUpgradeBodyWithUUID(uuid string) []byte {
	return []byte(`{"model":"claude-x","max_tokens":100,` +
		`"system":[{"type":"text","text":"policy for request ` + uuid + `","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
}

// TestVolatileHeadTTLUpgradeDeadEndRepro captures the #2191 defect: with the lever off (the
// default) a volatile head refuses the TTL upgrade with volatile_head and identity bytes —
// permanently. With FAK_CACHEBP_REDACT=1 the SAME body converts: the head is normalized to
// the spec placeholder and the 1h upgrade lands, witnessed per attempt.
func TestVolatileHeadTTLUpgradeDeadEndRepro(t *testing.T) {
	raw := ttlUpgradeBodyWithUUID(redactTestUUID)

	// The dead-end (default posture): refused, identity, and the refusal names the lever gap.
	out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
	if oc.Reason != TTLUpgradeReasonVolatileHead {
		t.Fatalf("default reason = %q, want volatile_head", oc.Reason)
	}
	if !bytes.Equal(out, raw) {
		t.Fatal("default volatile_head refusal must be identity")
	}
	if oc.Redacted || oc.RedactReason != RedactReasonDisabled {
		t.Fatalf("default refusal witness = {Redacted:%v RedactReason:%q}, want the disabled-lever label", oc.Redacted, oc.RedactReason)
	}

	// The conversion (lever on): upgraded on a normalized head.
	t.Setenv(CacheBPRedactEnvVar, "1")
	out, oc = UpgradeAnthropicStableCacheTTL1h(raw)
	if oc.Reason != TTLUpgradeReasonNone {
		t.Fatalf("lever-on reason = %q, want upgraded (none)", oc.Reason)
	}
	if !oc.Redacted || oc.RedactedUUID != 1 || oc.RedactedTimestamp != 0 {
		t.Fatalf("redaction witness = %+v, want Redacted with exactly 1 uuid", oc)
	}
	if !bytes.Contains(out, []byte(`"cache_control":{"type":"ephemeral","ttl":"1h"}`)) {
		t.Fatalf("no 1h ttl on the converted head:\n%s", out)
	}
	if bytes.Contains(out, []byte(redactTestUUID)) {
		t.Fatalf("volatile uuid survived in the converted body:\n%s", out)
	}
	if !bytes.Contains(out, []byte(redactedUUIDPlaceholder)) {
		t.Fatalf("spec placeholder missing from the converted head:\n%s", out)
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("converted body does not re-decode: %v", err)
	}
}

// TestVolatileHeadRedactionStableAcrossTurns is the point of the whole conversion: two turns
// whose heads differ ONLY in their volatile token values (a fresh UUID, a later time-of-day on
// the same date) must normalize to byte-IDENTICAL upgraded bodies — the provider prefix is
// cacheable again.
func TestVolatileHeadRedactionStableAcrossTurns(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "1")
	body := func(uuid, ts string) []byte {
		return []byte(`{"model":"claude-x","max_tokens":100,` +
			`"system":[{"type":"text","text":"req ` + uuid + ` at ` + ts + `","cache_control":{"type":"ephemeral"}}],` +
			`"messages":[{"role":"user","content":"hi"}]}`)
	}
	turn1, oc1 := UpgradeAnthropicStableCacheTTL1h(body(redactTestUUID, "2026-07-13T09:15:02Z"))
	turn2, oc2 := UpgradeAnthropicStableCacheTTL1h(body("ffffffff-ffff-4fff-8fff-ffffffffffff", "2026-07-13T17:48:59.221+05:30"))
	if oc1.Reason != TTLUpgradeReasonNone || oc2.Reason != TTLUpgradeReasonNone {
		t.Fatalf("upgrade reasons = %q, %q, want both converted", oc1.Reason, oc2.Reason)
	}
	if oc1.RedactedTimestamp != 1 || oc2.RedactedTimestamp != 1 {
		t.Fatalf("timestamp witness = %d, %d, want 1 each", oc1.RedactedTimestamp, oc2.RedactedTimestamp)
	}
	if !bytes.Equal(turn1, turn2) {
		t.Fatalf("normalized turns differ — the prefix still busts:\n%s\n%s", turn1, turn2)
	}
	// The date survives at day resolution; every sub-day byte (minute, second, fraction, zone)
	// is consumed into the placeholder, not left behind as residual per-request noise.
	if !bytes.Contains(turn1, []byte("2026-07-13Thh:mm")) {
		t.Fatalf("date-preserving placeholder missing:\n%s", turn1)
	}
	for _, residue := range []string{"09:15", ":02", "17:48", ":59", ".221", "+05:30", "Z"} {
		if bytes.Contains(turn1, []byte(residue)) || bytes.Contains(turn2, []byte(residue)) {
			t.Fatalf("sub-day residue %q survived normalization", residue)
		}
	}
}

// TestVolatileHeadPlacementConverts covers the placement refusal sites: a tools head carrying
// a UUID (which no hoist can help — tools sit ahead of every system anchor), and a system head
// with no stable block at all. Off-lever both stay the labeled volatile_head identity; on, both
// place a breakpoint on the normalized maximal head, and the messages tail is untouched.
func TestVolatileHeadPlacementConverts(t *testing.T) {
	toolsVolatile := []byte(`{"model":"claude-x","max_tokens":100,` +
		`"tools":[{"name":"probe","description":"session ` + redactTestUUID + `","input_schema":{"type":"object"}}],` +
		`"system":[{"type":"text","text":"stable policy"}],` +
		`"messages":[{"role":"user","content":"uuid in tail ` + redactTestUUID + ` stays"}]}`)
	allSystemVolatile := []byte(`{"model":"claude-x","max_tokens":100,` +
		`"system":[{"type":"text","text":"boot ` + redactTestUUID + `"},{"type":"text","text":"seen 2026-07-13 14:32:11"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)

	for name, raw := range map[string][]byte{"tools-volatile": toolsVolatile, "all-system-volatile": allSystemVolatile} {
		out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
		if oc.Reason != BreakpointReasonVolatileHead || !bytes.Equal(out, raw) {
			t.Fatalf("%s default: reason=%q, want volatile_head identity", name, oc.Reason)
		}
	}

	t.Setenv(CacheBPRedactEnvVar, "1")

	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(toolsVolatile)
	if oc.Reason != BreakpointReasonNone || oc.Target != "system" {
		t.Fatalf("tools-volatile conversion: outcome=%+v, want placement on the maximal system head", oc)
	}
	if !oc.Redacted || oc.RedactedUUID != 1 {
		t.Fatalf("tools-volatile witness = %+v, want 1 redacted uuid", oc)
	}
	// Redaction touches HEAD spans only: the identical UUID in the messages tail is preserved
	// verbatim, and the tools description carries the placeholder.
	if !bytes.Contains(out, []byte(`uuid in tail `+redactTestUUID+` stays`)) {
		t.Fatalf("messages tail was rewritten:\n%s", out)
	}
	if !bytes.Contains(out, []byte(`"description":"session `+redactedUUIDPlaceholder+`"`)) {
		t.Fatalf("tools head not normalized:\n%s", out)
	}

	out, oc = PlaceAnthropicCacheBreakpointWithOutcome(allSystemVolatile)
	if oc.Reason != BreakpointReasonNone || oc.Target != "system" {
		t.Fatalf("all-system-volatile conversion: outcome=%+v, want system placement", oc)
	}
	if !oc.Redacted || oc.RedactedUUID != 1 || oc.RedactedTimestamp != 1 {
		t.Fatalf("all-system-volatile witness = %+v, want 1 uuid + 1 timestamp", oc)
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("converted body does not re-decode: %v", err)
	}
}

// TestVolatileHeadRedactionPrefersHoist draws the #2181 boundary: when the M2 hoist can already
// earn a stable anchor (one stable system block among volatile siblings), placement must hoist —
// the volatile token survives VERBATIM behind the anchor, and no redaction runs even with the
// lever armed. Strip only what hoist cannot help.
func TestVolatileHeadRedactionPrefersHoist(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "1")
	raw := []byte(`{"model":"claude-x","max_tokens":100,` +
		`"system":[{"type":"text","text":"req ` + redactTestUUID + `"},{"type":"text","text":"stable policy"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone || !oc.Rewritten || oc.MovedVolatile != 1 {
		t.Fatalf("outcome = %+v, want an M2 hoist placement", oc)
	}
	if oc.Redacted || oc.RedactedUUID != 0 {
		t.Fatalf("hoistable head was stripped: %+v", oc)
	}
	if !bytes.Contains(out, []byte(redactTestUUID)) {
		t.Fatalf("hoisted volatile token must survive verbatim:\n%s", out)
	}
}

// TestVolatileHeadRedactionOffLeverSpellings: only the documented spellings arm the lever;
// everything else keeps the pre-#2191 refusal exactly.
func TestVolatileHeadRedactionOffLeverSpellings(t *testing.T) {
	raw := ttlUpgradeBodyWithUUID(redactTestUUID)
	for _, v := range []string{"", "0", "off", "false", "banana"} {
		t.Setenv(CacheBPRedactEnvVar, v)
		out, oc := UpgradeAnthropicStableCacheTTL1h(raw)
		if oc.Reason != TTLUpgradeReasonVolatileHead || !bytes.Equal(out, raw) {
			t.Fatalf("env=%q: reason=%q, want the unconverted volatile_head identity", v, oc.Reason)
		}
	}
	for _, v := range []string{"1", "on", "true", " ON "} {
		t.Setenv(CacheBPRedactEnvVar, v)
		_, oc := UpgradeAnthropicStableCacheTTL1h(raw)
		if oc.Reason != TTLUpgradeReasonNone {
			t.Fatalf("env=%q: reason=%q, want converted", v, oc.Reason)
		}
	}
}

// TestRedactVolatileHeadEngine white-boxes the engine's own contract: idempotence (a normalized
// head re-normalizes to identity), the stable-head no-op, date-only tokens left alone, identity
// on a non-object body, and byte-identity outside the head spans.
func TestRedactVolatileHeadEngine(t *testing.T) {
	raw := ttlUpgradeBodyWithUUID(redactTestUUID)
	once, oc := redactVolatileHead(raw)
	if oc.Reason != RedactReasonNone || oc.RedactedUUID != 1 {
		t.Fatalf("first pass outcome = %+v, want 1 uuid normalized", oc)
	}
	twice, oc2 := redactVolatileHead(once)
	if oc2.Reason != RedactReasonStableHead || !bytes.Equal(twice, once) {
		t.Fatalf("second pass = %+v, want stable_head identity (idempotence)", oc2)
	}

	// Byte-identity outside the head: the prefix before the system value and the suffix after
	// it are verbatim (the engine only rewrites the flagged head-value spans).
	start, end, ok := objectValueSpanLastWins(raw, "system")
	rs, re, rok := objectValueSpanLastWins(once, "system")
	if !ok || !rok {
		t.Fatal("system span lookup failed")
	}
	if !bytes.Equal(raw[:start], once[:rs]) || !bytes.Equal(raw[end:], once[re:]) {
		t.Fatal("bytes outside the redacted head span changed")
	}

	stable := []byte(`{"model":"m","system":[{"type":"text","text":"Today's date is 2026-07-13"}],"messages":[]}`)
	out, oc3 := redactVolatileHead(stable)
	if oc3.Reason != RedactReasonStableHead || !bytes.Equal(out, stable) {
		t.Fatalf("date-only head outcome = %+v, want stable_head identity", oc3)
	}

	if _, oc4 := redactVolatileHead([]byte(`not json`)); oc4.Reason != RedactReasonNoHead {
		t.Fatalf("non-object outcome = %+v, want no_head", oc4)
	}
}

// TestVolatileHeadRedactionRespectsAlreadySet: placement's already-set guard outranks the
// redaction retry — a client-managed layout with a volatile head is never rewritten by the
// PLACEMENT path even with the lever armed (the TTL-upgrade path owns client-placed heads).
func TestVolatileHeadRedactionRespectsAlreadySet(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "1")
	raw := ttlUpgradeBodyWithUUID(redactTestUUID) // carries the client's own cache_control
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonAlreadySet || !bytes.Equal(out, raw) {
		t.Fatalf("outcome = %+v, want already_set identity", oc)
	}
}

// TestVolatileHeadRedactionGoldenDefault pins the default-path bytes: with the lever off, the
// exported transforms are byte-for-byte the pre-#2191 behavior on a representative corpus —
// placed, already_set, no_stable_head, volatile_head — so the retry seam provably costs the
// default posture nothing.
func TestVolatileHeadRedactionGoldenDefault(t *testing.T) {
	t.Setenv(CacheBPRedactEnvVar, "")
	bodies := map[string]string{
		"placed":         `{"model":"m","system":[{"type":"text","text":"stable"}],"messages":[{"role":"user","content":"hi"}]}`,
		"already_set":    `{"model":"m","system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral"}}],"messages":[]}`,
		"no_stable_head": `{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		"volatile_head":  `{"model":"m","system":[{"type":"text","text":"req ` + redactTestUUID + `"}],"messages":[]}`,
	}
	for want, body := range bodies {
		out, oc := PlaceAnthropicCacheBreakpointWithOutcome([]byte(body))
		once, ocOnce := placeAnthropicCacheBreakpointOnce([]byte(body))
		if !bytes.Equal(out, once) {
			t.Fatalf("%s: wrapper bytes diverge from the un-retried pass", want)
		}
		if oc.Reason != ocOnce.Reason || oc.Target != ocOnce.Target || oc.Rewritten != ocOnce.Rewritten {
			t.Fatalf("%s: wrapper outcome %+v diverges from un-retried %+v", want, oc, ocOnce)
		}
		if reason := oc.Reason; want != "placed" && !strings.Contains(want, reason) {
			t.Fatalf("corpus label %q got reason %q", want, reason)
		}
	}
}
