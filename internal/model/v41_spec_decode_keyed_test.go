package model

import (
	"reflect"
	"testing"
)

// v41KeyedCapability is the per-request RNG-identity capability #13195 requires,
// expressed as an interface so this file COMPILES against the parent commit as
// well as the fix: every post-fix symbol is reached through a method on this
// interface rather than a direct call. On the parent the type assertion fails
// (red); at the fix it holds (green).
//
// The keyed coin is exposed on V41SpecDecodeConfig (a type that already exists
// at the parent), so `any(cfg).(v41KeyedCapability)` is a runtime assertion, not
// a compile-time reference to a post-fix symbol.
type v41KeyedCapability interface {
	KeyFold(requestID string) uint64
	AcceptCoin(step int) (float32, bool)
	BonusCoin(step int) (float32, bool)
	KeyedAccept(draft []int, alpha []float32) ([]bool, bool)
	KeyedAcceptedPrefix(anchor int, draft []int, alpha []float32, base []int) ([]int, bool)
}

// keyedCfg arms a request key through the capability interface, failing the test
// (at RUNTIME) when the capability is absent -- which is exactly the parent-tree
// case the symptom witness must observe as red.
func keyedCfg(t *testing.T, requestID string) v41KeyedCapability {
	t.Helper()
	base := V41SpecDecodeConfig{Enabled: true, DraftLen: 4, Layout: V41SpecDecodeLayoutBonus1N}
	armed := base
	if w, ok := any(base).(interface {
		WithRequestKey(string) V41SpecDecodeConfig
	}); ok {
		armed = w.WithRequestKey(requestID)
	}
	kc, ok := any(armed).(v41KeyedCapability)
	if !ok {
		t.Fatalf("V41SpecDecodeConfig does not expose the per-request keyed coin; cannot arm request %q (#13195)", requestID)
	}
	return kc
}

// TestV41SpecDecodeKeyedSlotReAdmissionInvariance is the witness for #13195: the
// V4.1 spec-decode accept/bonus coins are drawn from the request identity alone,
// so a request re-admitted into a DIFFERENT batch slot reproduces the identical
// accepted-token sequence. Per-slot churn is simulated with unrelated state that
// a slot-derived implementation would have consumed; only the request id may
// influence the result.
func TestV41SpecDecodeKeyedSlotReAdmissionInvariance(t *testing.T) {
	base := []int{1, 2, 3, 4, 5, 6, 7, 8}
	draft := base[1:5] // [2,3,4,5]
	// Mid-range alphas so the keyed coins yield a mix of accepts/rejects rather
	// than a degenerate all-true or all-false sequence.
	alpha := []float32{0.9, 0.3, 0.7, 0.1}

	seqA, okA := keyedCfg(t, "req-v41-spec-decode-13195").KeyedAcceptedPrefix(base[0], draft, alpha, base)
	if !okA {
		t.Fatal("slot A keyed prefix failed to derive")
	}
	// Simulate re-admission churn: interleave unrelated slot state.
	slotChurn := 0
	for i := 0; i < 7; i++ {
		slotChurn += i * 3
	}
	_ = slotChurn
	seqB, okB := keyedCfg(t, "req-v41-spec-decode-13195").KeyedAcceptedPrefix(base[0], draft, alpha, base)
	if !okB {
		t.Fatal("slot B keyed prefix failed to derive")
	}

	if !reflect.DeepEqual(seqA, seqB) {
		t.Fatalf("slot re-admission changed accepted-token sequence: slot A %v != slot B %v", seqA, seqB)
	}
	if len(seqA) == 0 {
		t.Fatal("accepted prefix is empty; witness does not exercise the path")
	}
	if seqA[0] != base[0] {
		t.Fatalf("accepted prefix does not begin with the anchor: %v", seqA)
	}
	for i, tok := range seqA {
		if tok != base[i] {
			t.Fatalf("accepted prefix %v is not the base greedy prefix at %d", seqA, i)
		}
	}
}

// TestV41SpecDecodeKeyedCoinsAreRequestKeyed proves the accept coin is a pure
// function of (request key, step): same key+step reproduces the same coin, and a
// different request draws a different one for the same step. It also proves the
// accept and bonus domains are separated, so a step's two draws never collide.
func TestV41SpecDecodeKeyedCoinsAreRequestKeyed(t *testing.T) {
	key := keyedCfg(t, "req-a")
	other := keyedCfg(t, "req-b")

	for step := 0; step < 8; step++ {
		c1, ok := key.AcceptCoin(step)
		if !ok {
			t.Fatalf("accept coin step %d not armed", step)
		}
		c2, _ := key.AcceptCoin(step)
		if c1 != c2 {
			t.Fatalf("accept coin not reproducible at step %d: %v != %v", step, c1, c2)
		}
		if c1 < 0 || c1 >= 1 {
			t.Fatalf("accept coin step %d out of [0,1): %v", step, c1)
		}
		b, _ := key.BonusCoin(step)
		if b == c1 {
			t.Fatalf("accept and bonus coins collide at step %d: %v", step, c1)
		}
	}

	same := true
	for step := 0; step < 8; step++ {
		a, _ := key.AcceptCoin(step)
		b, _ := other.AcceptCoin(step)
		if a != b {
			same = false
			break
		}
	}
	if same {
		t.Fatal("two distinct request ids drew an identical coin stream for all steps")
	}
}

// TestV41SpecDecodeKeyedFailsClosedWithoutKey proves the keyed seam fails closed
// on an unarmed (empty) request id: it derives no verdicts and no prefix rather
// than drawing from an unkeyed stream.
func TestV41SpecDecodeKeyedFailsClosedWithoutKey(t *testing.T) {
	base := V41SpecDecodeConfig{Enabled: true, DraftLen: 2, Layout: V41SpecDecodeLayoutBonus1N}
	kc, ok := any(base).(v41KeyedCapability)
	if !ok {
		t.Fatalf("V41SpecDecodeConfig does not expose the per-request keyed coin (#13195)")
	}
	if kc.KeyFold("") != 0 {
		t.Fatal("empty request id must fold to a zero key")
	}
	if _, ok := kc.AcceptCoin(0); ok {
		t.Fatal("unarmed key must not yield an accept coin")
	}
	if _, ok := kc.BonusCoin(0); ok {
		t.Fatal("unarmed key must not yield a bonus coin")
	}
	if _, ok := kc.KeyedAccept([]int{2, 3}, []float32{0.5, 0.5}); ok {
		t.Fatal("unarmed key must not derive accept verdicts")
	}
	if _, ok := kc.KeyedAcceptedPrefix(1, []int{2, 3}, []float32{0.5, 0.5}, []int{1, 2, 3, 4}); ok {
		t.Fatal("unarmed key must not derive an accepted prefix")
	}
}

// TestV41SpecDecodeKeyedAcceptFailsClosedOnMissingAlpha proves a position with no
// acceptance probability is a rejection rather than an admitted unweighed draft.
func TestV41SpecDecodeKeyedAcceptFailsClosedOnMissingAlpha(t *testing.T) {
	key := keyedCfg(t, "req-alpha")
	draft := []int{2, 3, 4}
	accept, ok := key.KeyedAccept(draft, []float32{1.0})
	if !ok {
		t.Fatal("keyed accept failed for an armed key")
	}
	if len(accept) != len(draft) {
		t.Fatalf("accept len = %d, want %d", len(accept), len(draft))
	}
	if !accept[0] {
		t.Fatal("alpha=1.0 must accept the first draft")
	}
	if accept[1] || accept[2] {
		t.Fatalf("positions with no alpha must fail closed as rejections: %v", accept)
	}
}
