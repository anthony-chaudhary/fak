package gateway

// l3deletioncert_test.go — the witness for the L3 page-key rung (#56): a per-span
// DeletionCertificate rung for a shared L3 pool. Each test maps to one acceptance bullet
// on the issue, all FAK-SIDE against MockL3Backend with NO external-store wire:
//
//   - AC#1: a minted rung NAMES the exact (non-empty) set of L3 page-keys that backed the
//     span, resolved via the Ref.Digest -> page-key seam (ResolveL3PageKeys).
//   - AC#2: the rung carries a post-delete ALL-MISS witness folded into the chain;
//     tampering any recorded key state fails the self-check.
//   - AC#3: Scope is the honest "l3-working-set" verbatim, with no over-claim string.
//   - AC#5: control-path only — no page payload byte appears in the rung; the witness is
//     keys + presence bits.
//   - AC#7: determinism — the same span + same backend state mints a byte-identical rung.

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// l3DelSpan builds a span's backing L3 pages, sets them into a fresh pool keyed by their
// content-address page-keys, and returns the pool, the resolved key set, and a subject
// digest standing in for the box cert's Subject. It is the FAK-side stand-in for "a span
// was backed by a set of external L3 page-keys".
func l3DelSpan(t *testing.T, contents ...string) (pool *MockL3Backend, keys []string, subject string) {
	t.Helper()
	pool = NewMockL3Backend()
	refs := make([]abi.Ref, 0, len(contents))
	var joined strings.Builder
	for _, c := range contents {
		b := []byte(c)
		refs = append(refs, abi.Ref{Kind: abi.RefRegion, Digest: l3Digest(b), Len: int64(len(b)), Scope: abi.ScopeAgent})
		joined.WriteString(c)
	}
	keys = ResolveL3PageKeys(refs) // the Ref.Digest -> page-key seam
	for i, k := range keys {
		pool.Set(k, []byte(contents[i]), L3PageMeta{Digest: k, Scope: abi.ScopeAgent, OwnerTag: "tenant-A"})
	}
	subject = "sha256:" + l3Digest([]byte(joined.String()))
	return pool, keys, subject
}

// TestL3DeletionRung_MintNamesKeysAllMissAndVerifies is AC#1 + AC#2 (+ AC#6 binding): set
// a span's pages, OP_DELETE them, mint the rung. It must name exactly the page-key set
// (non-empty), be all-miss, bind the box cert subject, and verify clean.
func TestL3DeletionRung_MintNamesKeysAllMissAndVerifies(t *testing.T) {
	pool, keys, subject := l3DelSpan(t, "alice page 0", "alice page 1", "alice page 2")
	if len(keys) == 0 {
		t.Fatalf("resolver returned an empty page-key set")
	}
	for _, k := range keys {
		pool.Delete(k) // the store's fire-and-forget OP_DELETE
	}

	rung, err := MintL3DeletionRung(subject, keys, pool)
	if err != nil {
		t.Fatalf("MintL3DeletionRung: %v", err)
	}
	if len(rung.PageKeys) != len(keys) {
		t.Fatalf("rung names %d page-keys, want %d", len(rung.PageKeys), len(keys))
	}
	// The named set must equal the resolved set (order-insensitive).
	gotSet := map[string]bool{}
	for _, k := range rung.PageKeys {
		gotSet[k] = true
	}
	for _, k := range keys {
		if !gotSet[k] {
			t.Fatalf("rung omits resolved page-key %q", k)
		}
	}
	if !rung.AllMiss {
		t.Fatalf("post-delete witness must be all-miss, got AllMiss=false: %+v", rung.KeyStates)
	}
	if rung.Subject != subject {
		t.Fatalf("rung subject = %q, want bound %q", rung.Subject, subject)
	}
	if err := VerifyL3DeletionRung(rung); err != nil {
		t.Fatalf("freshly minted rung did not verify: %v", err)
	}
}

// TestL3DeletionRung_NotAllMissRejected is AC#2/AC#4 fail-closed: if even one backing
// page still resolves after the delete, the witness is not all-miss and Verify refuses
// the rung (the deletion was not witnessed).
func TestL3DeletionRung_NotAllMissRejected(t *testing.T) {
	pool, keys, subject := l3DelSpan(t, "page 0", "page 1", "still-resident page 2")
	// Delete all but the LAST page — it still resolves in the pool.
	for _, k := range keys[:len(keys)-1] {
		pool.Delete(k)
	}
	rung, err := MintL3DeletionRung(subject, keys, pool)
	if err != nil {
		t.Fatalf("MintL3DeletionRung: %v", err)
	}
	if rung.AllMiss {
		t.Fatalf("a still-resident page must make AllMiss=false")
	}
	if err := VerifyL3DeletionRung(rung); err == nil {
		t.Fatalf("a not-all-miss rung must fail verification")
	}
}

// TestL3DeletionRung_TamperFailsClosed is AC#2: any post-mint edit to the recorded key
// states or chain breaks the self-check. Covers a flipped presence bit, a relabeled key,
// a dropped page-key, and a forged chain digest.
func TestL3DeletionRung_TamperFailsClosed(t *testing.T) {
	pool, keys, subject := l3DelSpan(t, "p0", "p1", "p2")
	for _, k := range keys {
		pool.Delete(k)
	}
	good, err := MintL3DeletionRung(subject, keys, pool)
	if err != nil {
		t.Fatalf("MintL3DeletionRung: %v", err)
	}
	if err := VerifyL3DeletionRung(good); err != nil {
		t.Fatalf("baseline rung must verify: %v", err)
	}

	tampers := map[string]func(r *L3DeletionRung){
		"flip-presence-bit": func(r *L3DeletionRung) { r.KeyStates[0].Present = true },
		"relabel-key":       func(r *L3DeletionRung) { r.KeyStates[0].Key = "forged-key" },
		"drop-a-page-key": func(r *L3DeletionRung) {
			r.PageKeys = r.PageKeys[:len(r.PageKeys)-1]
			r.KeyStates = r.KeyStates[:len(r.KeyStates)-1]
		},
		"forge-chain-digest": func(r *L3DeletionRung) {
			r.ChainDigest = strings.Repeat("0", len(r.ChainDigest))
		},
		"forge-all-miss-bit": func(r *L3DeletionRung) { r.AllMiss = false },
	}
	for name, mut := range tampers {
		t.Run(name, func(t *testing.T) {
			forged := deepCopyRung(good)
			mut(&forged)
			if err := VerifyL3DeletionRung(forged); err == nil {
				t.Fatalf("tamper %q passed verification", name)
			}
		})
	}
}

// TestL3DeletionRung_ScopeIsHonestWorkingSet is AC#3: the scope is the honest
// l3-working-set token verbatim, the marshaled rung carries no over-claim string, and an
// over-claim scope is refused.
func TestL3DeletionRung_ScopeIsHonestWorkingSet(t *testing.T) {
	pool, keys, subject := l3DelSpan(t, "a", "b")
	for _, k := range keys {
		pool.Delete(k)
	}
	rung, err := MintL3DeletionRung(subject, keys, pool)
	if err != nil {
		t.Fatalf("MintL3DeletionRung: %v", err)
	}
	if rung.Scope != L3DeletionScopeWorkingSet {
		t.Fatalf("scope = %q, want %q", rung.Scope, L3DeletionScopeWorkingSet)
	}
	b, _ := json.Marshal(rung)
	for _, tok := range bannedL3DeletionScopeTokens {
		if strings.Contains(string(b), tok) {
			t.Fatalf("marshaled rung carries over-claim token %q", tok)
		}
	}
	// An over-claim scope (e.g. someone widening it to "deleted-everywhere") must fail.
	forged := deepCopyRung(rung)
	forged.Scope = "deleted-everywhere"
	if err := VerifyL3DeletionRung(forged); err == nil {
		t.Fatalf("an over-claim scope must be refused")
	}
}

// TestL3DeletionRung_ControlPathNoPayload is AC#5: no page payload byte from the span
// surfaces in the rung — only content-address page-keys, presence bits, scope, subject.
func TestL3DeletionRung_ControlPathNoPayload(t *testing.T) {
	contents := []string{"SECRET-api-key-0", "SECRET-api-key-1"}
	pool, keys, subject := l3DelSpan(t, contents...)
	for _, k := range keys {
		pool.Delete(k)
	}
	rung, err := MintL3DeletionRung(subject, keys, pool)
	if err != nil {
		t.Fatalf("MintL3DeletionRung: %v", err)
	}
	b, _ := json.Marshal(rung)
	for _, c := range contents {
		if strings.Contains(string(b), c) {
			t.Fatalf("rung leaked page payload %q (control-path-only violated)", c)
		}
	}
}

// TestL3DeletionRung_Deterministic is AC#7: the same span + same backend state mints a
// byte-identical rung across runs (the keyless chain stays reproducible).
func TestL3DeletionRung_Deterministic(t *testing.T) {
	pool, keys, subject := l3DelSpan(t, "x0", "x1", "x2", "x3")
	for _, k := range keys {
		pool.Delete(k)
	}
	a, err := MintL3DeletionRung(subject, keys, pool)
	if err != nil {
		t.Fatalf("mint a: %v", err)
	}
	// Re-mint over the SAME post-delete pool, and with the keys supplied in a DIFFERENT
	// order — Mint sorts them, so the rung must be byte-identical either way.
	shuffled := []string{keys[2], keys[0], keys[3], keys[1]}
	c, err := MintL3DeletionRung(subject, shuffled, pool)
	if err != nil {
		t.Fatalf("mint c: %v", err)
	}
	ja, _ := json.Marshal(a)
	jc, _ := json.Marshal(c)
	if string(ja) != string(jc) {
		t.Fatalf("rung is not deterministic:\n a=%s\n c=%s", ja, jc)
	}
}

// TestL3DeletionRung_StructuralRefusals pins Mint's fail-closed guards: no page-keys and a
// nil pool are refused (nothing to witness gone / no witness seam).
func TestL3DeletionRung_StructuralRefusals(t *testing.T) {
	if _, err := MintL3DeletionRung("subj", nil, NewMockL3Backend()); err == nil {
		t.Fatalf("empty page-key set must be refused")
	}
	if _, err := MintL3DeletionRung("subj", []string{"k0"}, nil); err == nil {
		t.Fatalf("nil pool witness must be refused")
	}
}

// TestMockL3Backend_ExistsAndDelete pins the witness primitives: Exists returns one
// presence bit per key (never bytes), and Delete is idempotent.
func TestMockL3Backend_ExistsAndDelete(t *testing.T) {
	pool := NewMockL3Backend()
	k := l3Digest([]byte("page"))
	pool.Set(k, []byte("page"), L3PageMeta{Digest: k, Scope: abi.ScopeAgent, OwnerTag: "tenant-A"})

	if got := pool.Exists([]string{k, "absent"}); len(got) != 2 || !got[0] || got[1] {
		t.Fatalf("Exists = %v, want [true false]", got)
	}
	if !pool.Delete(k) {
		t.Fatalf("Delete of a resident key should report true")
	}
	if pool.Delete(k) {
		t.Fatalf("Delete is idempotent: a second delete reports false")
	}
	if got := pool.Exists([]string{k}); got[0] {
		t.Fatalf("key must be absent after delete")
	}
}

// deepCopyRung clones a rung (including its slices) so a tamper mutation in one subtest
// does not bleed into the shared baseline.
func deepCopyRung(r L3DeletionRung) L3DeletionRung {
	out := r
	out.PageKeys = append([]string(nil), r.PageKeys...)
	out.KeyStates = append([]L3KeyState(nil), r.KeyStates...)
	return out
}
