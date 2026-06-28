package gateway

// l3deletioncert.go — child E of the L3 disaggregated-cache epic (#56; study
// docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md §5 deletion-cert headline + §6.5
// control-path constraint): the L3 PAGE-KEY RUNG (G3 — prove a deletion happened),
// projected onto the shared, multi-tenant L3 tier.
//
// THE GAP. The shipped box DeletionCertificate (internal/deletioncert.Mint) attests a
// bit-exact eviction from the LOCAL working set; it carries no notion of the external L3
// page-keys that backed the span, and no post-delete witness that those keys stopped
// resolving in the shared pool. The reference L3 store's OP_DELETE / LRU eviction is
// fire-and-forget — no receipt, no witness the page is actually gone. A tenant who asks
// "prove my span was forgotten from the shared pool" gets nothing back. This rung is the
// receipt the semantics-free store structurally omits, riding ABOVE its delete path.
//
// WHAT THE RUNG ADDS — purely additive; the box Mint rung is untouched.
//   - the SET of external L3 page-keys that backed the evicted span (resolved via the
//     Ref.Digest -> page-key seam — l3region.L3RegionBackend.PageKeys in production,
//     content-addressed page digests here via ResolveL3PageKeys); the set is non-empty.
//   - a POST-DELETE ALL-MISS WITNESS: after the delete, an exists/mget over EXACTLY those
//     page-keys against the pool returns all-miss, folded into a keyless, reproducible
//     hash chain. Tamper any recorded key state — flip a presence bit, drop or reorder a
//     key — and the self-check (VerifyL3DeletionRung) fails closed.
//   - an honest Scope stamped verbatim "l3-working-set" — never "deleted-everywhere".
//     The rung proves the KV working-set pages are gone from the L3 tier, nothing more
//     (no claim over weights, embeddings, backups, or replicas).
//
// CONTROL PATH ONLY (§6.5, load-bearing). The witness is page-keys + presence BITS only;
// the mint path reads NO page payload byte (L3PoolWitness.Exists returns booleans, never
// bytes). The external store's NO-REPLICATION property HELPS here: one copy to prove
// gone, not N — so an all-miss over the page-keys is a COMPLETE pool witness, not a
// sample.
//
// HONEST LIMITS — inherited from the box cert, so a caller cannot over-read this rung.
// The lane journal records DECISIONS not EFFECTS, so this does NOT upgrade the box-side
// evicted_count self-report; it is the L3 EFFECT read-back rung only (the box-side
// effect-witness is a separate v2 rung). The keyless chain proves CONSISTENCY (the
// records bind to each other), not ORIGINALITY (an external identity).

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// L3DeletionScopeWorkingSet is the ONLY honest scope token an L3 deletion rung carries:
// it proves the KV working-set pages stopped resolving in the shared L3 tier, NOT
// deletion from weights, embeddings, backups, or replicas. Mint stamps it verbatim and
// Verify refuses anything else (mirrors the box cert's Scope-honesty discipline).
const L3DeletionScopeWorkingSet = "l3-working-set"

// l3DeletionRungSchema is the rung schema id folded into the keyless chain, so an old
// verifier never silently accepts a new-shaped rung (the bump rule the box cert uses).
const l3DeletionRungSchema = "fak.l3-deletion-rung/v1"

// bannedL3DeletionScopeTokens are the over-claim strings the honest l3-working-set scope
// must never contain — defense in depth over the exact-match scope check, mirroring the
// isolation-bench artifact ban-list so a future edit that smuggles an over-claim into the
// scope fails closed.
var bannedL3DeletionScopeTokens = []string{"deleted-everywhere", "weights", "backups", "replicas", "embeddings"}

// L3PoolWitness is the minimal CONTROL-PATH probe the deletion rung folds: an exists/mget
// over an ordered page-key set returning ONLY a presence bit per key — never the page
// bytes. It is the seam the rung reads the post-delete all-miss witness through; the
// in-process MockL3Backend satisfies it, and a real CAMA-complete connector supplies the
// same exists/mget over the RDMA pool. Returning bits (not bytes) is what keeps the mint
// path off the data path (§6.5).
type L3PoolWitness interface {
	// Exists reports, for each requested page-key in order, whether the pool still holds
	// it — presence bits only, no payload. len(result) == len(keys).
	Exists(keys []string) []bool
}

// L3KeyState records one page-key's post-delete presence bit: the key (a content
// address, never a payload byte) and whether it still resolved in the pool. For an
// all-miss witness every Present is false.
type L3KeyState struct {
	Key     string `json:"key"`
	Present bool   `json:"present"`
}

// L3DeletionRung is the additive L3 page-key rung that extends the box DeletionCertificate
// fold to the shared L3 tier. It names the external page-keys that backed an evicted span
// and carries the post-delete all-miss witness, bound to the box cert by Subject and made
// tamper-evident by a keyless ChainDigest. Marshaled as JSON it is the portable receipt
// the reference store could not produce; VerifyL3DeletionRung re-checks it fail-closed.
type L3DeletionRung struct {
	Schema      string       `json:"schema"`       // l3DeletionRungSchema
	Scope       string       `json:"scope"`        // verbatim "l3-working-set" — never an over-claim
	Subject     string       `json:"subject"`      // the evicted span's digest — binds the rung to the box cert
	PageKeys    []string     `json:"page_keys"`    // the external L3 page-keys that backed the span (non-empty, sorted)
	KeyStates   []L3KeyState `json:"key_states"`   // post-delete presence bit per page-key (1:1 with PageKeys, key order)
	AllMiss     bool         `json:"all_miss"`     // true iff every KeyStates.Present == false (the all-miss witness)
	ChainDigest string       `json:"chain_digest"` // keyless sha256 fold over the rung — reproducible + tamper-evident
}

// ResolveL3PageKeys maps a set of L3-resident page Refs to their content-address page
// keys — the Ref.Digest -> page-key seam in its simplest content-addressed form (a page
// key IS sha256(page), byte-identical to the address internal/l3region and the blob tier
// mint pages by). In production l3region.L3RegionBackend.PageKeys returns the ordered set
// from its manifest; this helper is the seam the FAK-side gate resolves through with no
// external wire. The returned set is in the caller's order; Mint sorts it for a
// reproducible chain.
func ResolveL3PageKeys(pages []abi.Ref) []string {
	keys := make([]string, 0, len(pages))
	for _, p := range pages {
		keys = append(keys, p.Digest)
	}
	return keys
}

// MintL3DeletionRung resolves the post-delete state of a span's backing page-keys into an
// L3 deletion rung. It runs the exists/mget witness over EXACTLY the supplied page-keys
// (presence bits only — never page bytes, §6.5), records each key's state, and folds the
// result into a keyless, reproducible hash chain bound to subject (the box cert's
// Subject). It does NOT delete: the caller fires the store's OP_DELETE first, then mints
// this rung to witness the effect — mirroring how the box Mint is a pure fold over facts
// the caller supplies. It refuses a structurally empty witness (no page-keys, nil pool),
// but records an honest non-all-miss state if a key still resolves — Verify, not Mint, is
// the gate that rejects a deletion that did not actually happen.
func MintL3DeletionRung(subject string, pageKeys []string, pool L3PoolWitness) (L3DeletionRung, error) {
	if len(pageKeys) == 0 {
		return L3DeletionRung{}, fmt.Errorf("gateway: refusing to mint an L3 deletion rung with no page-keys (nothing to witness gone)")
	}
	if pool == nil {
		return L3DeletionRung{}, fmt.Errorf("gateway: L3 deletion rung needs a pool witness")
	}
	keys := append([]string(nil), pageKeys...)
	sort.Strings(keys) // deterministic key order => reproducible chain (AC #7)

	present := pool.Exists(keys) // CONTROL PATH: presence bits only, no payload read (§6.5)
	if len(present) != len(keys) {
		return L3DeletionRung{}, fmt.Errorf("gateway: pool witness returned %d presence bits for %d page-keys", len(present), len(keys))
	}

	states := make([]L3KeyState, len(keys))
	allMiss := true
	for i, k := range keys {
		states[i] = L3KeyState{Key: k, Present: present[i]}
		if present[i] {
			allMiss = false
		}
	}

	rung := L3DeletionRung{
		Schema:    l3DeletionRungSchema,
		Scope:     L3DeletionScopeWorkingSet,
		Subject:   subject,
		PageKeys:  keys,
		KeyStates: states,
		AllMiss:   allMiss,
	}
	rung.ChainDigest = l3DeletionChainDigest(rung)
	return rung, nil
}

// VerifyL3DeletionRung re-checks an L3 deletion rung and returns nil iff every rung holds.
// It fails CLOSED — any of these is an error:
//   - the schema is unknown, or the scope is not the honest l3-working-set token (or
//     carries an over-claim string);
//   - the page-key set is empty (nothing witnessed gone);
//   - the all-miss witness is absent or incomplete — a named page-key has no recorded
//     presence bit, or the witness names a different / reordered key (disagrees with the
//     page-key set);
//   - the witness is not ALL-MISS — a page-key still resolves in the pool, so the deletion
//     was not witnessed — or the recorded AllMiss disagrees with the states;
//   - the keyless chain digest does not re-derive (a recorded key state was tampered after
//     mint).
func VerifyL3DeletionRung(r L3DeletionRung) error {
	if r.Schema != l3DeletionRungSchema {
		return fmt.Errorf("l3 deletion rung: unknown schema %q (want %q)", r.Schema, l3DeletionRungSchema)
	}
	if r.Scope != L3DeletionScopeWorkingSet {
		return fmt.Errorf("l3 deletion rung: scope %q is not the honest %q token", r.Scope, L3DeletionScopeWorkingSet)
	}
	for _, tok := range bannedL3DeletionScopeTokens {
		if strings.Contains(r.Scope, tok) {
			return fmt.Errorf("l3 deletion rung: scope carries over-claim token %q (scope must stay %q)", tok, L3DeletionScopeWorkingSet)
		}
	}
	if len(r.PageKeys) == 0 {
		return fmt.Errorf("l3 deletion rung: empty page-key set (nothing witnessed gone)")
	}
	if len(r.KeyStates) != len(r.PageKeys) {
		return fmt.Errorf("l3 deletion rung: all-miss witness absent or incomplete (%d states for %d page-keys)", len(r.KeyStates), len(r.PageKeys))
	}
	allMiss := true
	for i, s := range r.KeyStates {
		if s.Key != r.PageKeys[i] {
			return fmt.Errorf("l3 deletion rung: witness key %q does not match named page-key %q at index %d (witness disagrees with the page-key set)", s.Key, r.PageKeys[i], i)
		}
		if s.Present {
			allMiss = false
		}
	}
	if !allMiss {
		return fmt.Errorf("l3 deletion rung: a page-key still resolves in the pool — deletion NOT witnessed (not all-miss)")
	}
	if r.AllMiss != allMiss {
		return fmt.Errorf("l3 deletion rung: recorded all_miss=%v disagrees with the witness states", r.AllMiss)
	}
	if got := l3DeletionChainDigest(r); got != r.ChainDigest {
		return fmt.Errorf("l3 deletion rung: chain digest mismatch (rung altered after mint)")
	}
	return nil
}

// l3DeletionChainDigest is the KEYLESS, reproducible fold over the rung's load-bearing
// fields — the schema tag, the honest scope, the bound subject, and each (page-key,
// presence-bit) pair in sorted key order. It is the tamper-evident anchor: flip any
// recorded key state, drop / add / reorder a key, or alter the scope or subject and the
// recomputed digest no longer matches, so Verify fails closed. No key, no clock, no
// randomness — the same inputs always fold to the same bytes (the determinism AC #7
// needs). ChainDigest is NOT part of its own pre-image. AllMiss is DERIVED from the
// states, so it is not folded directly; Verify recomputes it and rejects a recorded
// AllMiss that disagrees.
func l3DeletionChainDigest(r L3DeletionRung) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\n", l3DeletionRungSchema)
	fmt.Fprintf(h, "scope=%s\n", r.Scope)
	fmt.Fprintf(h, "subject=%s\n", r.Subject)
	for _, s := range r.KeyStates {
		fmt.Fprintf(h, "key=%s present=%t\n", s.Key, s.Present)
	}
	return hex.EncodeToString(h.Sum(nil))
}
