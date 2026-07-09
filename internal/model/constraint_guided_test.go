package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guideddecode"
)

// The guided-decode adapter test drives GuidedByteMask over a tiny synthetic token
// vocabulary (id -> exact decoded bytes). Using a hand-built vocab lets each test
// state be reached by a single history token whose bytes ARE the desired prefix, so
// the (elsewhere-tested) real tokenizer is not needed here — the adapter's contract
// is purely "given TokenBytes + a schema, mask the illegal tokens".

// guidedVocab maps a synthetic token id to the exact bytes it decodes to. The multi-
// byte "prefix" tokens (P*) exist only to seed a decode state via a 1-token history.
var guidedVocab = map[int][]byte{
	0:  []byte("{"),                          // legal first byte at start
	1:  []byte("x"),                          // illegal first byte at start
	2:  []byte(`{"name":"`),                  // whole PRE skeleton in one token (legal from start)
	3:  []byte(`{"nXme`),                     // diverges inside PRE (illegal from start)
	4:  []byte("g"),                          // legal after prefix {"name":" (get/get_weather)
	5:  []byte("e"),                          // legal after {"name":"g
	6:  []byte("et"),                         // legal after {"name":"g -> "get"
	7:  []byte("l"),                          // legal after {"name":" (list)
	8:  []byte("z"),                          // illegal name first byte
	9:  []byte("t"),                          // legal after {"name":"ge
	10: []byte(`"`),                          // close-quote
	11: []byte("_"),                          // continue get_weather
	12: []byte("ex"),                         // diverges after 'e' ({"name":"gex..)
	13: []byte(`{"name":"g`),                 // P: seed mid-name state
	14: []byte(`{"name":"get","arguments":`), // P: seed the UNCONSTRAINED (skeleton complete) state
	15: []byte("anything at all }{}"),        // free bytes, only legal once unconstrained
	16: nil,                                  // undecodable id -> must be masked
	17: []byte(`{"name":"z`),                 // P: seed a DEAD-END state (no name starts 'z')
}

const guidedVocabSize = 18

// tokenBytesFrom returns a TokenBytes-shaped closure over the synthetic vocab.
func tokenBytesFrom(v map[int][]byte) func(int) []byte {
	return func(id int) []byte { return v[id] }
}

// legalContinuation is the INDEPENDENT soundness oracle: it replays the byte-level
// guideddecode FSM over prefix+tb directly (no reference to the adapter's code) and
// reports whether the whole byte sequence stays on a valid path. The adapter must
// mask a token iff this returns false — that equivalence is the soundness property.
func legalContinuation(prefix, tb []byte, schema guideddecode.ToolSchema) bool {
	if len(tb) == 0 {
		return false
	}
	cur := append([]byte(nil), prefix...)
	for _, b := range tb {
		allowed := guideddecode.AllowedNextBytes(cur, schema)
		if allowed == nil {
			return true // unconstrained: the rest is free
		}
		if !allowed[b] {
			return false
		}
		cur = append(cur, b)
	}
	return true
}

func isNegInf(f float32) bool { return math.IsInf(float64(f), -1) }

func TestGuidedByteMaskAdapter(t *testing.T) {
	schema := guideddecode.ToolSchema{Names: []string{"get", "get_weather", "list"}}
	mask := &GuidedByteMask{Schema: schema, TokenBytes: tokenBytesFrom(guidedVocab)}

	cases := []struct {
		name    string
		history []int // decodes (via guidedVocab) to the envelope prefix under test
		admit   []int // ids that MUST be admitted (logit left unchanged)
		reject  []int // ids that MUST be masked (logit -> -inf)
		allFree bool  // true when the prefix is UNCONSTRAINED: NO token may be masked
	}{
		{
			name:    "start: empty prefix",
			history: nil,
			// '{' opens; the full PRE token walks legally to the enum boundary.
			admit:  []int{0, 2},
			reject: []int{1, 3, 4, 5, 8, 16},
		},
		{
			name:    "enum boundary: prefix == PRE",
			history: []int{2}, // {"name":"
			// first name bytes: 'g' (get/get_weather), 'l' (list); "et"/"ex" don't start a name.
			admit:  []int{4, 7},
			reject: []int{1, 5, 6, 8, 9, 12, 16},
		},
		{
			name:    "mid-name: prefix == {\"name\":\"g",
			history: []int{13}, // {"name":"g
			// after 'g' only 'e' (get*) is legal. "e" admits; "et" = e then t => "get", legal.
			// "t" alone diverges (prefix has only 'g', not 'ge'); "ex" = e then x diverges.
			admit:  []int{5, 6},
			reject: []int{1, 4, 8, 9, 11, 12, 16},
		},
		{
			name:    "unconstrained: full skeleton consumed",
			history: []int{14}, // {"name":"get","arguments":
			allFree: true,
			admit:   []int{0, 1, 2, 3, 4, 5, 15}, // everything is free now
		},
		{
			name:    "dead end: prefix left every envelope",
			history: []int{17}, // {"name":"z  -> no name starts 'z'
			// documented no-op: nothing is masked (the adapter declines to zero the whole vector).
			allFree: true,
			admit:   []int{0, 1, 4, 8, 15},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			logits := make([]float32, guidedVocabSize) // all zero
			mask.MaskLogits(c.history, logits)

			for _, id := range c.admit {
				if isNegInf(logits[id]) {
					t.Errorf("id %d (%q) was MASKED but must be admitted", id, guidedVocab[id])
				}
			}
			for _, id := range c.reject {
				if !isNegInf(logits[id]) {
					t.Errorf("id %d (%q) was ADMITTED (logit %v) but must be masked", id, guidedVocab[id], logits[id])
				}
			}

			// allFree: assert NO token was masked at all (whole distribution intact).
			if c.allFree {
				for id := 0; id < guidedVocabSize; id++ {
					if isNegInf(logits[id]) {
						t.Errorf("unconstrained/dead-end state masked id %d (%q); must be a no-op", id, guidedVocab[id])
					}
				}
			}

			// Soundness sweep (acceptance #3): for EVERY id, the adapter's verdict must
			// exactly match the independent byte-level FSM oracle — never mask a legal
			// continuation (no false reject), always mask an illegal one. On an
			// unconstrained/dead-end prefix the adapter is a deliberate no-op, so the
			// oracle-equivalence is only asserted on the CONSTRAINED states.
			if !c.allFree {
				prefix := decodePrefixFor(guidedVocab, c.history)
				for id := 0; id < guidedVocabSize; id++ {
					wantLegal := legalContinuation(prefix, guidedVocab[id], schema)
					gotMasked := isNegInf(logits[id])
					if wantLegal && gotMasked {
						t.Errorf("SOUNDNESS: id %d (%q) is a legal continuation of %q but was masked",
							id, guidedVocab[id], prefix)
					}
					if !wantLegal && !gotMasked {
						t.Errorf("id %d (%q) is NOT a legal continuation of %q but was admitted",
							id, guidedVocab[id], prefix)
					}
				}
			}
		})
	}
}

// decodePrefixFor mirrors the adapter's prefix reconstruction for the oracle.
func decodePrefixFor(v map[int][]byte, history []int) []byte {
	var prefix []byte
	for _, id := range history {
		prefix = append(prefix, v[id]...)
	}
	return prefix
}

// TestGuidedByteMaskNilSafe pins the identity contracts: a nil receiver, a nil
// TokenBytes, and an empty-logits call are all no-ops that never panic.
func TestGuidedByteMaskNilSafe(t *testing.T) {
	var nilMask *GuidedByteMask
	nilMask.MaskLogits([]int{1, 2}, make([]float32, 4)) // nil receiver: no panic

	noFn := &GuidedByteMask{Schema: guideddecode.ToolSchema{Names: []string{"get"}}}
	logits := []float32{1, 2, 3}
	noFn.MaskLogits(nil, logits)
	for i, v := range logits {
		if isNegInf(v) {
			t.Fatalf("nil TokenBytes masked id %d", i)
		}
	}
}

// TestGuidedByteMaskViaSeam proves the adapter plugs into the existing DecodeConstraint
// seam as a LogitMask, and stays dormant unless FAK_NATIVE_GUIDED_DECODE=1 — the
// default-off contract acceptance #4 requires. With the flag off the constraint is
// inert (Active()==false) even though a real Mask is attached.
func TestGuidedByteMaskViaSeam(t *testing.T) {
	var _ LogitMask = (*GuidedByteMask)(nil) // compile-time: implements the seam

	c := &DecodeConstraint{Mask: &GuidedByteMask{
		Schema:     guideddecode.ToolSchema{Names: []string{"get"}},
		TokenBytes: tokenBytesFrom(guidedVocab),
	}}
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "")
	if c.Active() {
		t.Fatal("mask attached but flag off: constraint must be inert (Active()==false)")
	}
	t.Setenv("FAK_NATIVE_GUIDED_DECODE", "1")
	if !c.maskActive() {
		t.Fatal("flag on with a mask: maskActive() must be true")
	}
}
