package agent

// Property/fuzz coverage for the byte-safety contract of the offensive cache stages
// (#3692): PlaceAnthropicCacheBreakpoint (including the M2 star-anchor hoist) and
// UpgradeAnthropicStableCacheTTL1h. The fixed-fixture tests pin known shapes; this file
// asserts the contract over ARBITRARY bodies:
//
//  1. Prefix safety — a placement/upgrade differs from its input by exactly ONE
//     contiguous replacement confined to the splice site: the added bytes are the
//     breakpoint / ttl key, and the only bytes a splice may remove are the interior
//     whitespace of a degenerate empty `{ }` target (found by the fuzzer; splicing into
//     an empty object cannot keep its whitespace). Every byte outside that span, prefix
//     AND suffix, is verbatim. The star-anchor hoist is the one sanctioned mover: there the
//     bytes ahead of the system array (the tools prefix the provider caches first) and
//     the whole suffix stay verbatim, the system elements are preserved byte-verbatim as
//     a permutation with no volatile element at or ahead of the anchor, and exactly one
//     element gains the breakpoint.
//  2. Fail-safe identity — every bail (non-JSON, volatile head, already-set, splice or
//     redecode failure, ...) returns the input bytes unchanged.
//  3. Idempotence — re-applying either transform to its own output is identity.
//
// checkCacheControlByteSafety is the shared oracle. FuzzCacheControlByteSafety drives it
// from the native fuzzer (seed corpus in testdata/fuzz/FuzzCacheControlByteSafety);
// TestCacheControlByteSafetyPropertyCorpus drives it over a deterministic generated
// corpus so plain `go test` witnesses the property, with coverage floors so the corpus
// cannot silently stop exercising the interesting paths.

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"strconv"
	"strings"
	"testing"
)

// singleContiguousInsertion reports whether after == before[:pos] + inserted + before[pos:]
// for some pos — i.e. the only change is one contiguous insertion. Any byte mutated
// outside the inserted span makes the decomposition impossible.
func singleContiguousInsertion(before, after []byte) (pos int, inserted []byte, ok bool) {
	d := len(after) - len(before)
	if d <= 0 {
		return 0, nil, false
	}
	p := 0
	for p < len(before) && before[p] == after[p] {
		p++
	}
	if !bytes.Equal(after[p+d:], before[p:]) {
		return 0, nil, false
	}
	return p, after[p : p+d], true
}

// confinedContiguousDelta decomposes after as before with ONE contiguous span replaced:
// after == before[:p] + added + before[len(before)-s:], removed = before[p:len(before)-s].
// Callers assert what a lawful splice may remove (whitespace only) and add (the key).
func confinedContiguousDelta(before, after []byte) (removed, added []byte, ok bool) {
	p := 0
	for p < len(before) && p < len(after) && before[p] == after[p] {
		p++
	}
	s := 0
	for s < len(before)-p && s < len(after)-p && before[len(before)-1-s] == after[len(after)-1-s] {
		s++
	}
	removed = before[p : len(before)-s]
	added = after[p : len(after)-s]
	if len(added) == 0 {
		return nil, nil, false
	}
	return removed, added, true
}

func isJSONWhitespaceOnly(b []byte) bool {
	for _, c := range b {
		if !isJSONSpace(c) {
			return false
		}
	}
	return true
}

type byteSafetyStats struct {
	placed   bool
	hoisted  bool
	upgraded bool
}

// checkCacheControlByteSafety is the invariant oracle both the fuzz target and the
// property corpus run every body through.
func checkCacheControlByteSafety(t testing.TB, raw []byte) byteSafetyStats {
	t.Helper()
	var stats byteSafetyStats
	placed := checkPlaceByteSafety(t, raw, &stats)
	stats.upgraded = checkUpgradeByteSafety(t, placed)
	if !bytes.Equal(placed, raw) {
		// The upgrade contract must hold on the raw input too, not only on placed output.
		checkUpgradeByteSafety(t, raw)
	}
	return stats
}

func checkPlaceByteSafety(t testing.TB, raw []byte, stats *byteSafetyStats) []byte {
	t.Helper()
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone {
		if !bytes.Equal(out, raw) {
			t.Fatalf("place bailed (%q) but did not return identity\nraw: %q\nout: %q", oc.Reason, raw, out)
		}
		return out
	}
	stats.placed = true
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("placed body does not re-decode: %v\nraw: %q\nout: %q", err, raw, out)
	}
	// Placement only proceeds when the input had no cache_control at all, so the output
	// must carry exactly the one spliced literal.
	if got := bytes.Count(out, []byte(cacheControlBreakpoint)); got != 1 {
		t.Fatalf("placement wrote %d breakpoint literals, want exactly 1\nraw: %q\nout: %q", got, raw, out)
	}
	if oc.Rewritten {
		stats.hoisted = true
		checkStarAnchorRewrite(t, raw, out)
	} else {
		removed, added, ok := confinedContiguousDelta(raw, out)
		if !ok || !isJSONWhitespaceOnly(removed) {
			t.Fatalf("placement changed bytes outside the breakpoint splice (removed %q)\nraw: %q\nout: %q", removed, raw, out)
		}
		if d := len(added); d != len(cacheControlBreakpoint) && d != len(cacheControlBreakpoint)+1 {
			t.Fatalf("placement added %d bytes, want the breakpoint key (%d, or +1 with comma)\nadded: %q", d, len(cacheControlBreakpoint), added)
		}
	}
	again, oc2 := PlaceAnthropicCacheBreakpointWithOutcome(out)
	if oc2.Reason != BreakpointReasonAlreadySet || !bytes.Equal(again, out) {
		t.Fatalf("place is not idempotent: second pass reason=%q\nfirst: %q\nsecond: %q", oc2.Reason, out, again)
	}
	return out
}

// checkStarAnchorRewrite verifies the one transform allowed to MOVE bytes: everything
// outside the system array value is verbatim, and inside it the elements are the original
// byte-verbatim elements permuted, exactly one carrying the spliced breakpoint, with no
// volatile element at or ahead of the anchor.
func checkStarAnchorRewrite(t testing.TB, raw, out []byte) {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("star-anchor: input no longer unmarshals: %v\nraw: %q", err, raw)
	}
	sysRaw := obj["system"]
	start := bytes.Index(raw, sysRaw)
	if start < 0 {
		t.Fatalf("star-anchor: cannot locate the system value in the input\nraw: %q", raw)
	}
	tail := raw[start+len(sysRaw):]
	if len(out) < start+len(tail) {
		t.Fatalf("star-anchor: output shorter than the invariant spans\nraw: %q\nout: %q", raw, out)
	}
	if !bytes.Equal(out[:start], raw[:start]) {
		t.Fatalf("star-anchor: bytes AHEAD of the system array changed — the tools prefix is no longer cache-stable\nraw: %q\nout: %q", raw, out)
	}
	if !bytes.Equal(out[len(out)-len(tail):], tail) {
		t.Fatalf("star-anchor: bytes AFTER the system array changed\nraw: %q\nout: %q", raw, out)
	}
	var origElems, newElems []json.RawMessage
	if err := json.Unmarshal(sysRaw, &origElems); err != nil {
		t.Fatalf("star-anchor: original system value is not a JSON array: %v\nraw: %q", err, raw)
	}
	newSys := out[start : len(out)-len(tail)]
	if err := json.Unmarshal(newSys, &newElems); err != nil {
		t.Fatalf("star-anchor: rewritten system span is not a JSON array: %v\nspan: %q", err, newSys)
	}
	if len(newElems) != len(origElems) {
		t.Fatalf("star-anchor: system element count changed: %d -> %d\nraw: %q\nout: %q", len(origElems), len(newElems), raw, out)
	}
	counts := map[string]int{}
	for _, el := range origElems {
		counts[string(el)]++
	}
	anchor := -1
	for i, el := range newElems {
		// Detect the anchor by the exact spliced BYTE literal, matching the transform's own
		// byte-level detector. A semantic decode here is wrong: a _-escaped
		// cache_control key evades the transform's bytes.Contains guard (so placement
		// lawfully proceeds) yet decodes to the same key, and would mis-identify the
		// verbatim escaped element as the anchor (fuzzer-found false failure).
		hasCC := bytes.Contains(el, []byte(cacheControlBreakpoint))
		key := string(el)
		if hasCC {
			if anchor >= 0 {
				t.Fatalf("star-anchor: more than one breakpoint in the rewritten system array\nspan: %q", newSys)
			}
			anchor = i
			key = string(unspliceBreakpoint(t, el))
		}
		if counts[key] == 0 {
			// The splice of an empty object cannot preserve interior whitespace; accept
			// any not-yet-matched whitespace-only original for a recovered "{}".
			if key != "{}" || !consumeEmptyObject(counts) {
				t.Fatalf("star-anchor: rewritten element %d is not byte-verbatim from the input\nelement: %q\nraw: %q\nout: %q", i, el, raw, out)
			}
		} else {
			counts[key]--
		}
		if anchor < 0 || i <= anchor {
			if headValueIsVolatile(json.RawMessage(key)) {
				t.Fatalf("star-anchor: volatile block at index %d sits at/ahead of the anchor — the cached span is not stable\nelement: %q\nout: %q", i, key, out)
			}
		}
	}
	if anchor < 0 {
		t.Fatalf("star-anchor: rewritten system array carries no breakpoint\nspan: %q", newSys)
	}
}

// unspliceBreakpoint recovers the pre-splice element bytes from the anchored element.
func unspliceBreakpoint(t testing.TB, el json.RawMessage) json.RawMessage {
	t.Helper()
	s := string(el)
	if suffix := "," + cacheControlBreakpoint + "}"; strings.HasSuffix(s, suffix) {
		return json.RawMessage(s[:len(s)-len(suffix)] + "}")
	}
	if s == "{"+cacheControlBreakpoint+"}" {
		return json.RawMessage("{}")
	}
	t.Fatalf("star-anchor: anchored element does not end with the spliced breakpoint key\nelement: %q", el)
	return nil
}

func consumeEmptyObject(counts map[string]int) bool {
	for k, n := range counts {
		if n <= 0 {
			continue
		}
		inner := strings.TrimSpace(k)
		if len(inner) >= 2 && inner[0] == '{' && inner[len(inner)-1] == '}' &&
			strings.TrimSpace(inner[1:len(inner)-1]) == "" {
			counts[k]--
			return true
		}
	}
	return false
}

// checkUpgradeByteSafety returns whether an upgrade actually fired.
func checkUpgradeByteSafety(t testing.TB, in []byte) bool {
	t.Helper()
	out, oc := UpgradeAnthropicStableCacheTTL1h(in)
	if oc.Reason != TTLUpgradeReasonNone {
		if !bytes.Equal(out, in) {
			t.Fatalf("ttl upgrade bailed (%q) but did not return identity\nin: %q\nout: %q", oc.Reason, in, out)
		}
		return false
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("upgraded body does not re-decode: %v\nin: %q\nout: %q", err, in, out)
	}
	removed, added, ok := confinedContiguousDelta(in, out)
	if !ok || !isJSONWhitespaceOnly(removed) {
		t.Fatalf("ttl upgrade changed bytes outside the cache_control splice (removed %q)\nin: %q\nout: %q", removed, in, out)
	}
	if d := len(added); d != len(cacheControlTTL1h) && d != len(cacheControlTTL1h)+1 {
		t.Fatalf("ttl upgrade added %d bytes, want the ttl key (%d, or +1 with comma)\nadded: %q", d, len(cacheControlTTL1h), added)
	}
	if got, want := bytes.Count(out, []byte(`"ttl":"1h"`)), bytes.Count(in, []byte(`"ttl":"1h"`))+1; got != want {
		t.Fatalf("ttl upgrade: %d 1h ttl literals, want %d\nin: %q\nout: %q", got, want, in, out)
	}
	again, oc2 := UpgradeAnthropicStableCacheTTL1h(out)
	if !bytes.Equal(again, out) {
		t.Fatalf("ttl upgrade is not idempotent: second pass (reason=%q) changed bytes\nfirst: %q\nsecond: %q", oc2.Reason, out, again)
	}
	return true
}

func FuzzCacheControlByteSafety(f *testing.F) {
	for _, seed := range [][]byte{
		// Plain system-array placement.
		[]byte(`{"model":"claude-x","max_tokens":64,"system":[{"type":"text","text":"stable head"}],"messages":[{"role":"user","content":"hi"}]}`),
		// Tools-only head.
		[]byte(`{"model":"claude-x","max_tokens":64,"tools":[{"name":"search","description":"stable","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"}]}`),
		// Volatile head (uuid) — must be identity.
		[]byte(`{"model":"claude-x","max_tokens":64,"system":[{"type":"text","text":"session 123e4567-e89b-12d3-a456-426614174000"}],"messages":[{"role":"user","content":"hi"}]}`),
		// Volatile head (datetime with time-of-day) — must be identity.
		[]byte(`{"model":"claude-x","max_tokens":64,"system":[{"type":"text","text":"generated at 2026-07-09T14:03:22Z"}],"messages":[{"role":"user","content":"hi"}]}`),
		// Star-anchor hoist: volatile block ahead of a stable one.
		[]byte(`{"model":"claude-x","max_tokens":64,"tools":[{"name":"t","description":"stable","input_schema":{"type":"object"}}],"system":[{"type":"text","text":"run 123e4567-e89b-12d3-a456-426614174000"},{"type":"text","text":"stable policy"}],"messages":[{"role":"user","content":"hi"}]}`),
		// Client breakpoint already set, 5m -> upgrade path.
		[]byte(`{"model":"claude-x","max_tokens":64,"system":[{"type":"text","text":"head","cache_control":{"type":"ephemeral","ttl":"5m"}}],"messages":[{"role":"user","content":[{"type":"text","text":"tail","cache_control":{"type":"ephemeral"}}]}]}`),
		// Breakpoint with ttl already 1h — everything is identity.
		[]byte(`{"model":"claude-x","max_tokens":64,"system":[{"type":"text","text":"head","cache_control":{"type":"ephemeral","ttl":"1h"}}],"messages":[{"role":"user","content":"hi"}]}`),
		// String system + tools (placement lands on tools).
		[]byte(`{"model":"claude-x","max_tokens":64,"system":"plain string system","tools":[{"name":"t","description":"d","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"}]}`),
		// Date-only text is stable by design.
		[]byte(`{"model":"claude-x","max_tokens":64,"system":[{"type":"text","text":"knowledge cutoff 2026-07-09"}],"messages":[{"role":"user","content":"hi"}]}`),
		// Message text that NAMES cache_control — conservative already-set identity.
		[]byte(`{"model":"claude-x","max_tokens":64,"system":[{"type":"text","text":"we discuss cache_control here"}],"messages":[{"role":"user","content":"hi"}]}`),
		// No head at all / empty arrays / non-JSON.
		[]byte(`{"model":"claude-x","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`),
		[]byte(`{"model":"claude-x","max_tokens":64,"system":[],"tools":[],"messages":[{"role":"user","content":"hi"}]}`),
		// Fuzzer-found (2026-07): splicing into an empty `{ }` anchor drops its interior
		// whitespace — the one sanctioned byte removal.
		[]byte(`{"system":[{ }]}`),
		// Review-found (2026-07): a unicode-escaped cache_control key (the underscore as
		// backslash-u005f) evades the transform's byte-literal already-set guard, so the
		// star-anchor hoist lawfully proceeds around the escaped element; the oracle must
		// not mistake it for the spliced anchor.
		[]byte(`{"model":"m","max_tokens":1,"system":[{"type":"text","text":"run 123e4567-e89b-12d3-a456-426614174000"},{"type":"text","text":"a","cache\` + `u005fcontrol":{"type":"ephemeral"}},{"type":"text","text":"b"}],"messages":[{"role":"user","content":"hi"}]}`),
		[]byte(`not json at all`),
		[]byte(`[]`),
		[]byte(``),
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		checkCacheControlByteSafety(t, raw)
	})
}

// TestCacheControlByteSafetyPropertyCorpus runs the oracle over a deterministic corpus of
// generated bodies (varied key order, whitespace, head shapes, volatile tokens, existing
// breakpoints, plus byte-level mutations) so plain `go test` exercises the property. The
// coverage floors keep the corpus honest: if the generator drifts and the interesting
// paths stop firing, the test fails rather than passing vacuously.
func TestCacheControlByteSafetyPropertyCorpus(t *testing.T) {
	r := rand.New(rand.NewSource(0x3692))
	const corpusSize = 4000
	var placed, hoisted, upgraded int
	for i := 0; i < corpusSize; i++ {
		body := genAnthropicBody(r)
		if r.Intn(4) == 0 {
			body = mutateBodyBytes(r, body)
		}
		stats := checkCacheControlByteSafety(t, body)
		if stats.placed {
			placed++
		}
		if stats.hoisted {
			hoisted++
		}
		if stats.upgraded {
			upgraded++
		}
	}
	t.Logf("corpus coverage: placed=%d hoisted=%d upgraded=%d of %d", placed, hoisted, upgraded, corpusSize)
	if placed < corpusSize/20 {
		t.Fatalf("corpus coverage collapsed: only %d/%d bodies took the placement path", placed, corpusSize)
	}
	if hoisted < 5 {
		t.Fatalf("corpus coverage collapsed: only %d bodies took the star-anchor hoist path", hoisted)
	}
	if upgraded < corpusSize/100 {
		t.Fatalf("corpus coverage collapsed: only %d/%d bodies took the ttl upgrade path", upgraded, corpusSize)
	}
}

// TestByteSafetyOracleRejectsPrefixByteMutation witnesses that the oracle actually FAILS
// on a prefix-byte mutation: for every byte ahead of (and after) the insertion, a
// corrupted output is rejected by the single-insertion decomposition. This is what makes
// the corpus test non-vacuous as a regression tripwire.
func TestByteSafetyOracleRejectsPrefixByteMutation(t *testing.T) {
	raw := []byte(`{"model":"claude-x","max_tokens":64,` +
		`"tools":[{"name":"search","description":"stable tool","input_schema":{"type":"object"}}],` +
		`"system":[{"type":"text","text":"head A"},{"type":"text","text":"head B"}],` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	out, oc := PlaceAnthropicCacheBreakpointWithOutcome(raw)
	if oc.Reason != BreakpointReasonNone || oc.Rewritten {
		t.Fatalf("fixture sanity: want a plain placement, got reason=%q rewritten=%v", oc.Reason, oc.Rewritten)
	}
	pos, inserted, ok := singleContiguousInsertion(raw, out)
	if !ok {
		t.Fatalf("fixture sanity: placement must be a single contiguous insertion")
	}
	for i := 0; i < len(out); i++ {
		if i >= pos && i < pos+len(inserted) {
			continue // inside the inserted span: not a prefix/suffix violation
		}
		mutated := append([]byte(nil), out...)
		mutated[i] ^= 0x01
		if _, ins, ok := singleContiguousInsertion(raw, mutated); ok &&
			(len(ins) == len(cacheControlBreakpoint) || len(ins) == len(cacheControlBreakpoint)+1) {
			t.Fatalf("byte %d outside the breakpoint mutated, yet the oracle still accepts the output — the invariant check is vacuous", i)
		}
	}
}

// --- deterministic structured-body generator ---

var genStableTexts = []string{
	"",
	"You are a terse code reviewer.",
	"Answer in English. Prefer short replies.",
	"a text that mentions cache_control by name",
	`braces { and } and a "quoted" fragment`,
	"unicode héllo — and an emoji 🌐",
	"Knowledge cutoff 2026-07-09.",
}

var genVolatileTexts = []string{
	"request id 123e4567-e89b-12d3-a456-426614174000",
	"generated at 2026-07-09T14:03:22Z",
	"logged 2026-07-09 14:03 UTC",
}

func jq(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func genText(r *rand.Rand) string {
	s := genStableTexts[r.Intn(len(genStableTexts))]
	if r.Intn(6) == 0 {
		s += strings.Repeat(" stable head padding", 1+r.Intn(30))
	}
	if r.Intn(4) == 0 {
		s += " " + genVolatileTexts[r.Intn(len(genVolatileTexts))]
	}
	return s
}

func genCacheControl(r *rand.Rand) string {
	switch r.Intn(3) {
	case 0:
		return `{"type":"ephemeral"}`
	case 1:
		return `{"type":"ephemeral","ttl":"1h"}`
	default:
		return `{"type":"ephemeral","ttl":"5m"}`
	}
}

func genSystemBlock(r *rand.Rand, allowCC bool) string {
	if r.Intn(40) == 0 {
		return jq("bare string system entry") // non-object element: pokes splice-failure handling
	}
	s := `{"type":"text","text":` + jq(genText(r))
	if allowCC && r.Intn(5) == 0 {
		s += `,"cache_control":` + genCacheControl(r)
	}
	return s + "}"
}

func genTool(r *rand.Rand, i int, allowCC bool) string {
	s := `{"name":"tool_` + strconv.Itoa(i) + `","description":` + jq(genText(r)) +
		`,"input_schema":{"type":"object","properties":{"q":{"type":"string"}}}`
	if allowCC && r.Intn(6) == 0 {
		s += `,"cache_control":` + genCacheControl(r)
	}
	return s + "}"
}

func genMessage(r *rand.Rand, i int, allowCC bool) string {
	role := "user"
	if i%2 == 1 {
		role = "assistant"
	}
	if r.Intn(2) == 0 {
		return `{"role":"` + role + `","content":` + jq(genText(r)) + `}`
	}
	n := 1 + r.Intn(2)
	blocks := make([]string, 0, n)
	for j := 0; j < n; j++ {
		b := `{"type":"text","text":` + jq(genText(r))
		if allowCC && j == n-1 && r.Intn(4) == 0 {
			b += `,"cache_control":` + genCacheControl(r)
		}
		blocks = append(blocks, b+"}")
	}
	return `{"role":"` + role + `","content":[` + strings.Join(blocks, ",") + `]}`
}

func genWS(r *rand.Rand) string {
	switch r.Intn(8) {
	case 0:
		return " "
	case 1:
		return "\n  "
	case 2:
		return "\t"
	default:
		return ""
	}
}

func genAnthropicBody(r *rand.Rand) []byte {
	// A third of the corpus is guaranteed cache_control-free so placement actually fires;
	// the rest may carry client breakpoints to exercise already-set and upgrade paths.
	allowCC := r.Intn(3) != 0
	parts := []string{
		`"model":"claude-x"`,
		`"max_tokens":` + strconv.Itoa(1+r.Intn(2048)),
	}
	if r.Intn(3) == 0 {
		parts = append(parts, `"stream":true`)
	}
	if r.Intn(4) == 0 {
		parts = append(parts, `"metadata":{"user_id":`+jq(genVolatileTexts[r.Intn(len(genVolatileTexts))])+`}`)
	}
	switch p := r.Intn(6); {
	case p == 0: // no tools
	case p == 1:
		parts = append(parts, `"tools":[]`)
	default:
		n := 1 + r.Intn(3)
		ts := make([]string, 0, n)
		for j := 0; j < n; j++ {
			ts = append(ts, genTool(r, j, allowCC))
		}
		parts = append(parts, `"tools":[`+strings.Join(ts, ","+genWS(r))+`]`)
	}
	switch p := r.Intn(8); {
	case p == 0: // no system
	case p == 1:
		parts = append(parts, `"system":`+jq(genText(r)))
	case p == 2:
		parts = append(parts, `"system":[]`)
	default:
		n := 1 + r.Intn(4)
		ss := make([]string, 0, n)
		for j := 0; j < n; j++ {
			ss = append(ss, genSystemBlock(r, allowCC))
		}
		parts = append(parts, `"system":[`+strings.Join(ss, ","+genWS(r))+`]`)
	}
	n := 1 + r.Intn(4)
	ms := make([]string, 0, n)
	for j := 0; j < n; j++ {
		ms = append(ms, genMessage(r, j, allowCC))
	}
	parts = append(parts, `"messages":[`+strings.Join(ms, ",")+`]`)
	r.Shuffle(len(parts), func(i, j int) { parts[i], parts[j] = parts[j], parts[i] })
	return []byte("{" + genWS(r) + strings.Join(parts, ","+genWS(r)) + genWS(r) + "}")
}

// mutateBodyBytes applies 1-3 random byte-level edits so the corpus also covers
// malformed and shape-shifted inputs (mostly exercising the identity paths).
func mutateBodyBytes(r *rand.Rand, raw []byte) []byte {
	out := append([]byte(nil), raw...)
	for n := 1 + r.Intn(3); n > 0 && len(out) > 0; n-- {
		switch r.Intn(4) {
		case 0:
			out[r.Intn(len(out))] ^= 1 << uint(r.Intn(8))
		case 1:
			i := r.Intn(len(out))
			out = append(out[:i], out[i+1:]...)
		case 2:
			i := r.Intn(len(out) + 1)
			rest := append([]byte{byte(r.Intn(256))}, out[i:]...)
			out = append(out[:i], rest...)
		default:
			out = out[:r.Intn(len(out)+1)]
		}
	}
	return out
}
