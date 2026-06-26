package l3referee

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// reasonToken pulls the precise L3 sub-reason token off a verdict.
func reasonToken(v abi.Verdict) string { return v.Meta[metaReason] }

// TestAdmitSet_DefaultFloor proves G6: with the default floor (Bounded),
// turn/session pages are refused (stay local) and bounded/durable are admitted.
// An unknown class default-expires (refuse). Every refusal cites a core reason
// code AND the precise L3 token.
func TestAdmitSet_DefaultFloor(t *testing.T) {
	r := New("") // empty floor => Bounded

	cases := []struct {
		class     string
		wantKind  abi.VerdictKind
		wantToken string // "" for an Allow
	}{
		{DurabilityTurn, abi.VerdictDeny, ReasonDurabilityFloor},
		{DurabilitySession, abi.VerdictDeny, ReasonDurabilityFloor},
		{DurabilityBounded, abi.VerdictAllow, ""},
		{DurabilityDurable, abi.VerdictAllow, ""},
		{"wat", abi.VerdictDeny, ReasonDurabilityUnknown},
		{"", abi.VerdictDeny, ReasonDurabilityUnknown},
	}
	for _, c := range cases {
		v := r.AdmitSet(abi.Ref{Kind: abi.RefBlob, Digest: "deadbeef"}, c.class)
		if v.Kind != c.wantKind {
			t.Errorf("AdmitSet(%q): kind = %d, want %d", c.class, v.Kind, c.wantKind)
		}
		if v.By != refereeBy {
			t.Errorf("AdmitSet(%q): By = %q, want %q", c.class, v.By, refereeBy)
		}
		if c.wantKind == abi.VerdictDeny {
			if got := reasonToken(v); got != c.wantToken {
				t.Errorf("AdmitSet(%q): reason token = %q, want %q", c.class, got, c.wantToken)
			}
			if v.Reason == abi.ReasonNone {
				t.Errorf("AdmitSet(%q): a refusal must cite a non-NONE core reason code", c.class)
			}
		}
	}
}

// TestAdmitSet_ConfigurableFloor proves the floor is a knob: raise it to Durable
// and a bounded page is now refused while durable still passes.
func TestAdmitSet_ConfigurableFloor(t *testing.T) {
	r := New(DurabilityDurable)
	if v := r.AdmitSet(abi.Ref{Digest: "x"}, DurabilityBounded); v.Kind != abi.VerdictDeny {
		t.Fatalf("floor=durable: bounded should be refused, got kind %d", v.Kind)
	}
	if v := r.AdmitSet(abi.Ref{Digest: "x"}, DurabilityDurable); v.Kind != abi.VerdictAllow {
		t.Fatalf("floor=durable: durable should be admitted, got kind %d", v.Kind)
	}
}

// TestVerifyGet_Bite is the max|Δ|=0-style bite: a correct (byte-exact) page
// passes and a tampered/wrong page is refused with L3_DIGEST_MISMATCH — driven
// by the fake in-memory L3 backend, no RDMA/CAMA dependency.
func TestVerifyGet_Bite(t *testing.T) {
	store := NewMemL3()
	r := New("")

	page := []byte("kv-page: prompt prefix block 0..19, byte-exact")
	ref := store.Set(page)

	// Correct path: the bytes that come back hash to the recorded digest (Δ=0).
	_, pd, ok := store.Get(ref)
	if !ok {
		t.Fatal("Get of a just-Set page should succeed")
	}
	if v := r.VerifyGet(ref, pd); v.Kind != abi.VerdictAllow {
		t.Fatalf("byte-exact page: want Allow, got kind %d token %q", v.Kind, reasonToken(v))
	}

	// Bite: corrupt the stored bytes under the SAME key (the hash-collision /
	// mis-computed-prefix failure). The returned page now hashes differently.
	store.Corrupt(ref, []byte("kv-page: TAMPERED — wrong KV, same address"))
	_, pdBad, ok := store.Get(ref)
	if !ok {
		t.Fatal("Get after Corrupt should still resolve the key")
	}
	if pdBad == ref.Digest {
		t.Fatal("test setup: corrupted page must hash differently from the recorded digest")
	}
	v := r.VerifyGet(ref, pdBad)
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("tampered page: want Deny, got kind %d", v.Kind)
	}
	if got := reasonToken(v); got != ReasonDigestMismatch {
		t.Fatalf("tampered page: reason token = %q, want %q", got, ReasonDigestMismatch)
	}
	if v.Reason != abi.ReasonTrustViolation {
		t.Fatalf("tampered page: core reason = %d, want ReasonTrustViolation (%d)", v.Reason, abi.ReasonTrustViolation)
	}
	// Forensics: the verdict records both digests so a consumer can see the Δ.
	if v.Meta["recorded"] != ref.Digest || v.Meta["returned"] != pdBad {
		t.Errorf("mismatch verdict should record both digests, got %v", v.Meta)
	}
}

// TestVerifyGet_NoRecordedDigest proves the fail-closed middle: a Ref fak never
// recorded a digest for cannot be PROVEN correct, so it is quarantined (held
// out), not allowed.
func TestVerifyGet_NoRecordedDigest(t *testing.T) {
	r := New("")
	v := r.VerifyGet(abi.Ref{Kind: abi.RefBlob, Digest: ""}, "anything")
	if v.Kind != abi.VerdictQuarantine {
		t.Fatalf("no recorded digest: want Quarantine, got kind %d", v.Kind)
	}
	if got := reasonToken(v); got != ReasonNoRecordedDigest {
		t.Fatalf("no recorded digest: reason token = %q, want %q", got, ReasonNoRecordedDigest)
	}
	if p, ok := v.Payload.(abi.QuarantinePayload); !ok || !p.PageOut {
		t.Fatalf("quarantine should carry a PageOut payload, got %#v", v.Payload)
	}
}

// TestNoDataPathHook proves the epic's sharpest constraint STRUCTURALLY: neither
// referee method may take the page payload ([]byte). The referee adjudicates on
// the control path only — Ref + a caller-computed digest — so verified bytes
// still flow client-direct over the 1–5 µs RDMA path fak never sits on.
func TestNoDataPathHook(t *testing.T) {
	typ := reflect.TypeOf((*L3Referee)(nil)).Elem()
	byteSlice := reflect.TypeOf([]byte(nil))
	for i := 0; i < typ.NumMethod(); i++ {
		m := typ.Method(i)
		ft := m.Type // interface method type: In(0) is the first real arg, no receiver
		for a := 0; a < ft.NumIn(); a++ {
			if ft.In(a) == byteSlice {
				t.Fatalf("L3Referee.%s takes a []byte parameter — that is a DATA-PATH hook; "+
					"the referee must adjudicate on the control path only (Ref + digest)", m.Name)
			}
		}
	}
}
