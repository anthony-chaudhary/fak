package vdso

// neardup_test.go — the opt-in near-dup key + the temporal-cache negative-result guard.
// Default-OFF behaviour is byte-identical to the exact key (asserted), so this never
// touches the soundness witness or the existing tests.

import (
	"encoding/json"
	"testing"
)

// OFF by default: formatting-variant args do NOT collide — the exact canonical key holds,
// preserving byte-soundness and the Part B novelty posture.
func TestNearDup_OffByDefault_VariantsDoNotCollide(t *testing.T) {
	v := New(8)
	a := roCall("convert_currency", `{"from":"USD","to":"EUR"}`)
	b := roCall("convert_currency", `{"from":" usd ","to":"eur"}`)
	fillAndExpectHit(t, v, a, `{"rate":1.1}`)
	if hits(t, v, b) {
		t.Errorf("formatting variant collided with near-dup OFF — the default must be the exact key")
	}
}

// ON: args differing only in case + whitespace of their string VALUES collapse to one
// entry, so the second phrasing is served from the first.
func TestNearDup_On_VariantsCollide(t *testing.T) {
	v := New(8)
	v.SetNearDup(true)
	a := roCall("convert_currency", `{"from":"USD","to":"EUR","note":"Trip  Budget"}`)
	b := roCall("convert_currency", `{"from":" usd ","to":"eur","note":"trip budget"}`)
	fillAndExpectHit(t, v, a, `{"rate":1.1}`)
	if !hits(t, v, b) {
		t.Errorf("formatting variant did NOT collide with near-dup ON")
	}
	// A genuinely different value must still MISS (near-dup collapses formatting, not meaning).
	if hits(t, v, roCall("convert_currency", `{"from":"gbp","to":"eur"}`)) {
		t.Errorf("a different currency collided — near-dup must not alias distinct values")
	}
}

// The temporal-cache negative-result guard: in near-dup mode a negative answer is never
// stored (so it can never be served stale to a variant), while a positive answer IS
// near-dup-shared.
func TestNearDup_NegativeResultGuard(t *testing.T) {
	v := New(8)
	v.SetNearDup(true)

	// Negative ("no flights") is NOT cached — even the exact repeat misses.
	neg := roCall("search_direct_flight", `{"origin":"SFO","destination":"XYZ"}`)
	v.Emit(completeEvent(neg, `{"flights":[]}`))
	if hits(t, v, neg) {
		t.Errorf("a negative result was cached in near-dup mode — the temporal guard failed")
	}

	// Positive IS cached and shared across a formatting variant.
	pos := roCall("search_direct_flight", `{"origin":"SFO","destination":"JFK"}`)
	v.Emit(completeEvent(pos, `{"flights":["AA1"]}`))
	if !hits(t, v, roCall("search_direct_flight", `{"origin":" sfo ","destination":"jfk"}`)) {
		t.Errorf("a positive result was not near-dup-shared with its formatting variant")
	}
}

func TestNegativeResult(t *testing.T) {
	cases := []struct {
		body string
		neg  bool
	}{
		{`null`, true},
		{`{}`, true},
		{`[]`, true},
		{`   `, true},
		{`{"flights":[]}`, true},
		{`{"results":[],"count":0}`, true},
		{`{"found":false}`, true},
		{`{"error":"nope"}`, true},
		{`{"ok":false}`, true},
		{`{"flights":["AA1"]}`, false},
		{`{"rate":1.1}`, false},
		{`{"found":true,"data":{"x":1}}`, false},
		{`"a string body"`, false},
		{`42`, false},
		{`not json at all`, false},
	}
	for _, c := range cases {
		if got := negativeResult([]byte(c.body)); got != c.neg {
			t.Errorf("negativeResult(%q) = %v, want %v", c.body, got, c.neg)
		}
	}
}

func TestNormalizeStr(t *testing.T) {
	cases := []struct{ in, want string }{
		{"USD", "usd"},
		{"  sfo  ", "sfo"},
		{"Hello  World", "hello world"},
		{"", ""},
		{"   ", ""},
		{"MixED\tCase\n", "mixed case"},
	}
	for _, c := range cases {
		if got := normalizeStr(c.in); got != c.want {
			t.Errorf("normalizeStr(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// Object KEYS are structural and must NOT be folded: two objects whose VALUES match after
// normalization but whose KEYS differ in case are still distinct.
func TestNearDup_KeysAreStructural(t *testing.T) {
	v := New(8)
	v.SetNearDup(true)
	a := roCall("lookup", `{"Code":"USD"}`)
	b := roCall("lookup", `{"code":"USD"}`)
	fillAndExpectHit(t, v, a, `{"x":1}`)
	if hits(t, v, b) {
		t.Errorf("a differing object KEY collided — near-dup must fold values, not keys")
	}
}

// legacyNearDupArgHash reproduces the pre-optimization decode/re-encode behavior
// as an oracle to prove bit-for-bit hash equivalence.
func legacyNearDupArgHash(b []byte) string {
	var v any
	if json.Unmarshal(b, &v) != nil {
		return argHash(b)
	}
	normalizeStrings(&v)
	out, err := json.Marshal(v)
	if err != nil {
		return argHash(b)
	}
	return argHash(out)
}

func TestNearDupArgHash_ExactHashEquivalence(t *testing.T) {
	cases := [][]byte{
		[]byte(`{}`),
		[]byte(`{"a":1}`),
		[]byte(`{"b":2,"a":1}`),
		[]byte(`{"from":"USD","to":"EUR"}`),
		[]byte(`{"from":" usd ","to":"eur"}`),
		[]byte(`{"from":" USD ","to":"eur","note":"Trip   Budget","passengers":2}`),
		[]byte(`{"nested":{"b":" BAR ","a":" FOO "},"items":[" ONE ",2,true,null]}`),
		[]byte(`[{"z":1,"a":2},{"k":" V "}]`),
		[]byte(`" Just  A  String "`),
		[]byte(`42.5`),
		[]byte(`true`),
		[]byte(`null`),
		[]byte(`not json at all`),
		[]byte(`{"invalid":`),
		[]byte(`{"open": "quote`),
		[]byte(``),
		[]byte(`    `),
		[]byte("\x00\x01\x02\xff"),
	}

	for _, c := range cases {
		got := nearDupArgHash(c)
		want := legacyNearDupArgHash(c)
		if got != want {
			t.Errorf("nearDupArgHash(%q) = %q, want legacy %q", string(c), got, want)
		}
	}
}

func BenchmarkNearDupArgHash(b *testing.B) {
	input := []byte(`{"from":" usd ","to":"eur","note":"Trip   Budget","passengers":2}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = nearDupArgHash(input)
	}
}

func BenchmarkArgHashFor_NearDup(b *testing.B) {
	v := New(8)
	v.SetNearDup(true)
	input := []byte(`{"from":" usd ","to":"eur","note":"Trip   Budget","passengers":2}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v.argHashFor("convert_currency", input)
	}
}

func BenchmarkArgHashFor_Exact(b *testing.B) {
	v := New(8)
	input := []byte(`{"from":"usd","to":"eur","note":"Trip Budget","passengers":2}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v.argHashFor("convert_currency", input)
	}
}
