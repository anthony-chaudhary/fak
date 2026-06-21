package vdso

// neardup_test.go — the opt-in near-dup key + the temporal-cache negative-result guard.
// Default-OFF behaviour is byte-identical to the exact key (asserted), so this never
// touches the soundness witness or the existing tests.

import (
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
