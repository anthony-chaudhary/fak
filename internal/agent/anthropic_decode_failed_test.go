package agent

import (
	"bytes"
	"testing"
)

// The bail-reason vocabularies of the request-side byte transforms are CLOSED SETS, and until
// #5387 each one folded a STRUCTURAL messages[] failure into a BENIGN, high-volume bucket:
// compaction and both elision passes reported a failed decodeArrayElements as too_few_msgs (the
// short-request idle), and the two toolref-file sanitizers reported it as no_messages_a (the
// empty-array idle). Every OTHER structural failure in those same functions already had its own
// reason — non_json for a non-object body, no_messages_key for an absent key — so the undecodable
// `messages` value was the one structural event counted where a rising number is expected and
// therefore raises no suspicion. Downstream (observeCompaction → the bail-reason counter/gauge and
// `fak cachevalue compaction`) inherits the merged label, and no query can recover the split.
//
// These tests pin the SPLIT, which is the whole point — a test that only proved the new reason
// fires would prove nothing about a bug that was two events sharing one label. Each subsystem is
// driven through its REAL exported entry point (never anchorCompactablePrefixMode directly) with
// two bodies that differ ONLY in the `messages` value:
//
//	structural — `messages` present but NOT a JSON array (null, or an object). decodeArrayElements
//	             returns ok=false (decodeArrayElementsAt's leading token is not '['), so the
//	             transform must now report decode_failed.
//	benign     — `messages` a well-formed array that is merely SHORTER than the subsystem's
//	             minimum. It must STILL report the original short-request reason.
//
// Both halves must also remain byte-identity returns: the split changes only the label, never the
// wire. Severity note, kept honest: on well-formed traffic the structural half is close to
// unreachable by construction (msgsRaw is a json.RawMessage carved out of the same raw whose
// Unmarshal already succeeded), so this is attribution and closed-vocabulary hygiene over
// defensive code — not the closing of a live fault.

// Bodies whose `messages` key is present but whose VALUE is not a JSON array. Each is valid JSON
// at the top level, so json.Unmarshal into map[string]json.RawMessage succeeds and the non_json /
// no_messages_key bails are both passed — the only way out is the decodeArrayElements branch.
// The extra keys are carried so the two toolref-file fast-path pre-scans (a literal
// "tool_reference", and containsRepairableEmptyContent's `[]`/`""` signature) are satisfied and
// those functions actually REACH the decode; without them they would short-circuit earlier and the
// test would be vacuous.
const (
	decodeFailNullMsgs = `{"model":"m","messages":null,"system":"tool_reference","tools":[]}`
	decodeFailObjMsgs  = `{"model":"m","messages":{"role":"user","content":"hi"},"system":"tool_reference","tools":[]}`
)

// wellFormedMsgs renders a valid /v1/messages body carrying exactly n alternating messages, so a
// "benign" case is genuinely short rather than malformed. It carries the same auxiliary keys as the
// structural bodies above so the ONLY difference between a structural and a benign case is the
// shape of the `messages` value itself.
func wellFormedMsgs(n int) []byte {
	var b bytes.Buffer
	b.WriteString(`{"model":"m","system":"tool_reference","tools":[],"messages":[`)
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		b.WriteString(`{"role":"` + role + `","content":[{"type":"text","text":"turn ` + string(rune('a'+i)) + `"}]}`)
	}
	b.WriteString(`]}`)
	return b.Bytes()
}

// TestCompactionSplitsDecodeFailedFromTooFewMsgs drives CompactAnthropicHistoryWithOutcome — the
// exported entry point the gateway calls, which reaches anchorCompactablePrefixMode with minElems=3
// — and proves the two outcomes that used to share too_few_msgs now carry different labels.
func TestCompactionSplitsDecodeFailedFromTooFewMsgs(t *testing.T) {
	for _, body := range []string{decodeFailNullMsgs, decodeFailObjMsgs} {
		raw := []byte(body)
		out, oc := CompactAnthropicHistoryWithOutcome(raw, 100)
		if oc.Reason != CompactReasonDecodeFailed {
			t.Errorf("undecodable messages[] %s: Reason=%q, want %q (a structural decode failure must not be filed under the benign short-request bucket)", body, oc.Reason, CompactReasonDecodeFailed)
		}
		if !bytes.Equal(out, raw) {
			t.Errorf("undecodable messages[] %s: body was rewritten; every bail must be byte-identity", body)
		}
	}
	// The discrimination: a genuinely SHORT but perfectly well-formed request (2 < minElems 3)
	// must still report too_few_msgs. Without this half the test would not witness the bug.
	short := wellFormedMsgs(2)
	out, oc := CompactAnthropicHistoryWithOutcome(short, 100)
	if oc.Reason != CompactReasonTooFewMsgs {
		t.Errorf("2 well-formed messages: Reason=%q, want %q (the benign short-request bail must keep its own label)", oc.Reason, CompactReasonTooFewMsgs)
	}
	if !bytes.Equal(out, short) {
		t.Error("2 well-formed messages: body was rewritten; the bail must be byte-identity")
	}
}

// TestCompactionDecodeFailedAcrossAnchorsAndView covers the other two ways into the same shared
// front half: the head-anchored option set (#1408) and the ctxview twin, which calls
// anchorCompactablePrefix with minElems=1. The ctxview twin is where the split is sharpest — with
// minElems=1 the ONLY body that can still report too_few_msgs is a well-formed EMPTY array, so the
// two labels partition cleanly.
func TestCompactionDecodeFailedAcrossAnchorsAndView(t *testing.T) {
	raw := []byte(decodeFailNullMsgs)

	if out, oc := CompactAnthropicHistoryWithOptions(raw, CompactOptions{Budget: 100, Anchor: CompactAnchorHead}); oc.Reason != CompactReasonDecodeFailed || !bytes.Equal(out, raw) {
		t.Errorf("head anchor: Reason=%q changed=%v, want %q and identity", oc.Reason, !bytes.Equal(out, raw), CompactReasonDecodeFailed)
	}

	planned := []Message{{Role: "user", Content: "turn a"}}
	if out, oc := CompactAnthropicHistoryToView(raw, planned); oc.Reason != CompactReasonDecodeFailed || !bytes.Equal(out, raw) {
		t.Errorf("ctxview twin: Reason=%q changed=%v, want %q and identity", oc.Reason, !bytes.Equal(out, raw), CompactReasonDecodeFailed)
	}
	// Discrimination for the minElems=1 caller: a well-formed but EMPTY array is the benign case.
	empty := []byte(`{"model":"m","system":"tool_reference","tools":[],"messages":[]}`)
	if out, oc := CompactAnthropicHistoryToView(empty, planned); oc.Reason != CompactReasonTooFewMsgs || !bytes.Equal(out, empty) {
		t.Errorf("ctxview twin, well-formed empty array: Reason=%q changed=%v, want %q and identity", oc.Reason, !bytes.Equal(out, empty), CompactReasonTooFewMsgs)
	}
}

// TestElisionSplitsDecodeFailedFromTooFewMsgs is the same discrimination for the oversized
// tool_result shrinker, whose vocabulary comment previously documented the merge outright.
func TestElisionSplitsDecodeFailedFromTooFewMsgs(t *testing.T) {
	for _, body := range []string{decodeFailNullMsgs, decodeFailObjMsgs} {
		raw := []byte(body)
		out, oc := ElideAnthropicResultsWithOutcome(raw, 64)
		if oc.Reason != ElideReasonDecodeFailed || !bytes.Equal(out, raw) {
			t.Errorf("undecodable messages[] %s: Reason=%q changed=%v, want %q and identity", body, oc.Reason, !bytes.Equal(out, raw), ElideReasonDecodeFailed)
		}
	}
	short := wellFormedMsgs(1) // 1 < 2, the elision minimum
	if out, oc := ElideAnthropicResultsWithOutcome(short, 64); oc.Reason != ElideReasonTooFewMsgs || !bytes.Equal(out, short) {
		t.Errorf("1 well-formed message: Reason=%q changed=%v, want %q and identity", oc.Reason, !bytes.Equal(out, short), ElideReasonTooFewMsgs)
	}
}

// TestStaleElisionSplitsDecodeFailedFromTooFewMsgs is the same discrimination for the stale-read
// pass — the third subsystem sharing decodeArrayElements and the merged label.
func TestStaleElisionSplitsDecodeFailedFromTooFewMsgs(t *testing.T) {
	for _, body := range []string{decodeFailNullMsgs, decodeFailObjMsgs} {
		raw := []byte(body)
		out, oc := ElideStaleReadsWithOutcome(raw)
		if oc.Reason != StaleReasonDecodeFailed || !bytes.Equal(out, raw) {
			t.Errorf("undecodable messages[] %s: Reason=%q changed=%v, want %q and identity", body, oc.Reason, !bytes.Equal(out, raw), StaleReasonDecodeFailed)
		}
	}
	short := wellFormedMsgs(1)
	if out, oc := ElideStaleReadsWithOutcome(short); oc.Reason != StaleReasonTooFewMsgs || !bytes.Equal(out, short) {
		t.Errorf("1 well-formed message: Reason=%q changed=%v, want %q and identity", oc.Reason, !bytes.Equal(out, short), StaleReasonTooFewMsgs)
	}
}

// TestSanitizersSplitDecodeFailedFromEmptyMsgs covers the two correctness sanitizers in
// anthropic_toolref.go. Their merged bucket is spelled no_messages_a rather than too_few_msgs, but
// it is the same defect: "could not be decoded / is empty" put a structural failure and a benign
// empty array under one label. Both bodies below satisfy the relevant fast-path pre-scan, so the
// decode branch is genuinely reached rather than short-circuited.
func TestSanitizersSplitDecodeFailedFromEmptyMsgs(t *testing.T) {
	emptyArr := []byte(`{"model":"m","system":"tool_reference","messages":[]}`)

	for _, body := range []string{decodeFailNullMsgs, decodeFailObjMsgs} {
		raw := []byte(body)
		if out, oc := SanitizeAnthropicToolReferences(raw); oc.Reason != ToolRefReasonDecodeFailed || !bytes.Equal(out, raw) {
			t.Errorf("toolref, undecodable messages[] %s: Reason=%q changed=%v, want %q and identity", body, oc.Reason, !bytes.Equal(out, raw), ToolRefReasonDecodeFailed)
		}
		if out, oc := RepairEmptyToolResultContent(raw); oc.Reason != EmptyContentReasonDecodeFailed || !bytes.Equal(out, raw) {
			t.Errorf("empty-content, undecodable messages[] %s: Reason=%q changed=%v, want %q and identity", body, oc.Reason, !bytes.Equal(out, raw), EmptyContentReasonDecodeFailed)
		}
	}
	// Discrimination: a well-formed EMPTY array still reports the benign no_messages_a.
	if out, oc := SanitizeAnthropicToolReferences(emptyArr); oc.Reason != ToolRefReasonNoMsgs || !bytes.Equal(out, emptyArr) {
		t.Errorf("toolref, well-formed empty array: Reason=%q changed=%v, want %q and identity", oc.Reason, !bytes.Equal(out, emptyArr), ToolRefReasonNoMsgs)
	}
	if out, oc := RepairEmptyToolResultContent(emptyArr); oc.Reason != EmptyContentReasonNoMsgs || !bytes.Equal(out, emptyArr) {
		t.Errorf("empty-content, well-formed empty array: Reason=%q changed=%v, want %q and identity", oc.Reason, !bytes.Equal(out, emptyArr), EmptyContentReasonNoMsgs)
	}
}

// TestDecodeFailedTokenIsOneWordAcrossSubsystems pins the cross-subsystem agreement the issue asked
// for: whichever transform bails, the ledger sees the SAME wire token, so `decode_failed` means one
// thing everywhere. It also pins that the new token is distinct from the benign buckets it was
// carved out of — the property a merged label cannot have.
func TestDecodeFailedTokenIsOneWordAcrossSubsystems(t *testing.T) {
	const want = "decode_failed"
	for name, got := range map[string]string{
		"CompactReasonDecodeFailed":      CompactReasonDecodeFailed,
		"ElideReasonDecodeFailed":        ElideReasonDecodeFailed,
		"StaleReasonDecodeFailed":        StaleReasonDecodeFailed,
		"ToolRefReasonDecodeFailed":      ToolRefReasonDecodeFailed,
		"EmptyContentReasonDecodeFailed": EmptyContentReasonDecodeFailed,
	} {
		if got != want {
			t.Errorf("%s = %q, want %q (one token, or the merged label just moves)", name, got, want)
		}
	}
	for _, benign := range []string{CompactReasonTooFewMsgs, ElideReasonTooFewMsgs, StaleReasonTooFewMsgs, ToolRefReasonNoMsgs, EmptyContentReasonNoMsgs} {
		if benign == want {
			t.Errorf("benign bucket %q collides with the structural token %q", benign, want)
		}
	}
	// It must also not collide with the neighbouring structural reasons it sits beside.
	for _, other := range []string{CompactReasonNonJSON, CompactReasonNoMsgsKey, CompactReasonRedecodeFail} {
		if other == want {
			t.Errorf("reason %q collides with the structural token %q", other, want)
		}
	}
}
