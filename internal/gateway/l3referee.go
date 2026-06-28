package gateway

// l3referee.go — child A of the L3 disaggregated-cache epic (#58; study
// docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md §4 Option A): the REFEREE SIDECAR
// seam that sits in front of an external L3 tier's get/set and adjudicates the CONTROL
// path only, exactly the shape `fak serve --provider anthropic` already runs in front
// of the real Claude API — a byte-exact passthrough hop where fak rules on the metadata
// and never mutates the forwarded bytes (memory fak-api-passthrough-first-class).
//
// THE COMPOSITION. The external L3 tier (CAMA-complete is the reference target) is a
// world-class build of the three layers fak does NOT change — routing, addressing,
// fusion — and is SEMANTICS-FREE by its own design: it "does NOT verify content;
// assumes the client hash is the source of truth; no checksums, no re-validation"
// (DISAGGREGATED-AGENT-MEMORY.md §2.5). The store is the "Kubernetes" (it places opaque
// cells); fak is the "Docker" (it defines what the cell IS). This sidecar is the seam
// where they compose: fak rides ABOVE the store's get/set and supplies the four control-
// path guarantees the store threw away at the syscall boundary —
//
//   - G1 (verify, don't trust): record the page's content digest (Ref.Digest) on set;
//     on get, the bytes the tier returned MUST hash to that recorded digest or the page
//     is REFUSED — a collision or mis-tag is served to no one.
//   - G4 (scope gate): the reader's tenant must be permitted by the page's ShareScope;
//     a cross-tenant read of a private/tenant-bound page is REFUSED before any byte.
//   - G5 (provenance): stamp the page's IFC taint label (abi.TaintLabel — the lattice
//     internal/ifc adjudicates over) on set so the matching get can read it back; the
//     label survives the round-trip.
//   - G8 (typed refusal): every refusal carries a token from the closed gateway L3
//     vocabulary (L3_PAGE_DIGEST_MISMATCH / L3_CROSS_TENANT_SCOPE_DENIED), never an
//     UNCLASSIFIED free-text reason.
//
// CONTROL PATH ONLY (§6.5, load-bearing). The sidecar touches only key/meta/digest. The
// page payload is forwarded UNMODIFIED — on an admit, the exact byte slice the backend
// handed back is returned identity-equal, never re-copied or re-scanned byte-by-byte at
// line rate, so the ~1–5µs L3 read budget is preserved. This is the sidecar split, not
// an inline byte-scanner: the G1 digest check is ONE admission decision per get, not a
// per-byte gate on the streamed data path (the client-direct RDMA read the sidecar is
// NOT on). l3referee_test.go's identity-equality test is the witness.
//
// WHAT THIS DOES NOT DO (honest scope, carried from the issue). It does not rebuild the
// external store (routing/addressing/fusion stay the store's job). It does not touch the
// data path. It does not do G2 (middle-of-sequence evict) or G3 (deletion certificate)
// — those need ownership of the KV math and are Option B (internal/l3region's later
// stages behind the Ref/Resolver/RegionBackend seam), not the sidecar. And it does not
// DECIDE what deserves to persist (the S7 truth-duration judgment): on set it CALLS the
// injected durability verdict (#496) to TIER the set and records the class it returns —
// it does not duplicate the S7 gate.

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
)

// L3DurabilityDeferred is the durability class the sidecar records when no #496 S7
// write-time durability verdict is wired in. It is an explicit, honest "not classified
// here" sentinel — NOT a durability judgment this issue invented. #496 replaces it by
// injecting the real S7 classifier (NewRefereeSidecar's durability argument); this seam
// only CALLS that verdict, it does not re-implement the truth-duration decision.
const L3DurabilityDeferred = "deferred-to-s7-durability-gate"

// L3DurabilityVerdict classifies how long a page written to the L3 tier deserves to be
// believed — the S7 write-time durability judgment (#496 / S7 epic #82). This file does
// NOT make that judgment; the sidecar CALLS this verdict on set to TIER the page and
// records the class it returns in the page's control-path meta. The default
// (deferDurability) returns L3DurabilityDeferred so a set is honestly stamped "not yet
// classified" until #496 injects the real classifier.
type L3DurabilityVerdict func(page abi.Ref, payload []byte) string

// deferDurability is the default verdict: it defers to #496 rather than guessing a
// durability class, returning the explicit deferred sentinel for every page.
func deferDurability(abi.Ref, []byte) string { return L3DurabilityDeferred }

// L3PageMeta is the CONTROL-PATH record the sidecar stamps for a page on set and reads
// back on get — the metadata the semantics-free store threw away, kept beside the opaque
// bytes so a later get can adjudicate without ever scanning the payload:
//
//   - Digest: the content identity (hex(sha256), Ref.Digest) recorded at set and
//     verified at get (G1) — byte-identical to the address internal/l3region and the
//     blob tier mint pages by, so the identity fak checks is the identity the tier uses.
//   - Scope: the share capability the producer stamped (G4).
//   - Taint: the IFC provenance label (abi.TaintLabel) carried from the producer (G5);
//     the zero value (TaintTainted) is the fail-closed "untrusted unless proven" floor.
//   - OwnerTag: the tenant that wrote the page (the G4 boundary the reader is checked
//     against).
//   - Durability: the class the injected #496 verdict assigned this set (informational;
//     L3DurabilityDeferred until #496 is wired).
type L3PageMeta struct {
	Digest     string
	Scope      abi.ShareScope
	Taint      abi.TaintLabel
	OwnerTag   string
	Durability string
}

// L3Backend is the minimal external L3 tier the sidecar rides: a content-addressed
// get/set keyed by the page's content digest. The CAMA-complete store is the named real
// integration target (talking to it over its wire protocol is follow-on); MockL3Backend
// is the local exemplar the acceptance gate runs against with no network dependency. The
// backend moves bytes and holds opaque meta; it makes NO trust decision — that is the
// sidecar's job on the control path.
type L3Backend interface {
	// Get returns the stored bytes and control-path meta for a page key, and ok=false on
	// a miss (a key the tier does not hold). The returned slice is the tier's own copy;
	// a sidecar that admits forwards it UNMODIFIED (§6.5).
	Get(key string) (payload []byte, meta L3PageMeta, ok bool)
	// Set stores bytes under a page key with its control-path meta. Bytes are opaque to
	// the backend; the sidecar passes them through unmodified.
	Set(key string, payload []byte, meta L3PageMeta)
}

// RefereeSidecar is the control-path referee in front of an L3Backend's get/set. It is
// the gateway-in-front-of-Claude passthrough shape applied to an L3 tier: page bytes
// flow through byte-exact while fak adjudicates G1/G4/G5/G8 on the metadata. Construct
// with NewRefereeSidecar.
type RefereeSidecar struct {
	backend    L3Backend
	durability L3DurabilityVerdict
}

// NewRefereeSidecar wraps an L3Backend with the control-path referee. The durability
// argument is the #496 S7 verdict the set defers to; pass nil to defer (every set is
// stamped L3DurabilityDeferred until the real classifier is wired).
func NewRefereeSidecar(backend L3Backend, durability L3DurabilityVerdict) *RefereeSidecar {
	if durability == nil {
		durability = deferDurability
	}
	return &RefereeSidecar{backend: backend, durability: durability}
}

// Set admits a page to the L3 tier on the CONTROL path and forwards its bytes unmodified
// to the backend. It records the three things the semantics-free store discards — the
// content digest (G1, so a later get can verify), the share scope (G4), and the IFC
// provenance taint label (G5, read back on get) — and CALLS the injected #496 durability
// verdict to TIER the set (it does not duplicate the S7 judgment). The page is keyed by
// its content digest, the same address the tier itself uses. The returned meta is the
// exact record the matching get will adjudicate against. Bytes are never mutated.
func (s *RefereeSidecar) Set(page abi.Ref, ownerTag string, payload []byte) L3PageMeta {
	meta := L3PageMeta{
		Digest:     page.Digest, // G1: the content identity a later get verifies the bytes against
		Scope:      page.Scope,  // G4: the capability the producer stamped
		Taint:      page.Taint,  // G5: the IFC provenance label (fail-closed TaintTainted if unstamped)
		OwnerTag:   ownerTag,
		Durability: s.durability(page, payload), // #496: call the S7 verdict to tier the set
	}
	// CONTROL PATH ONLY: the bytes are forwarded unmodified; the sidecar touches only meta.
	s.backend.Set(page.Digest, payload, meta)
	return meta
}

// Get adjudicates a read of an L3 page on the CONTROL path and, on an admit, forwards the
// tier's bytes identity-equal (no defensive copy — §6.5). It runs the SAME G1+G4+G8
// adjudication child D ships (AdmitL3SharedPage): G1 verifies the returned bytes hash to
// the recorded digest, G4 checks the reader's tenant against the page's ShareScope, and
// any failure is REFUSED with a typed token from the closed vocabulary. A key the tier
// does not hold is a MISS (found=false), NOT a refusal — the caller takes its cold path;
// only an adjudication failure carries a refusal reason. On a refusal the bytes are never
// returned (fail-closed). The returned meta lets the caller read back the G5 provenance
// label even on an admit.
func (s *RefereeSidecar) Get(pageKey, readerTag string) (payload []byte, meta L3PageMeta, found bool, v L3ShareVerdict) {
	fetched, meta, ok := s.backend.Get(pageKey)
	if !ok {
		// A miss is not a refusal: the tier simply does not hold this page.
		return nil, L3PageMeta{}, false, L3ShareVerdict{}
	}

	// Reconstruct the control-path Ref from the recorded meta and run the shipped
	// G1+G4+G8 adjudication — no new crypto, no new refusal class invented here.
	page := abi.Ref{
		Kind:   abi.RefRegion,
		Digest: meta.Digest, // G1: the identity the bytes must hash to
		Scope:  meta.Scope,  // G4: the capability the reader is checked against
		Taint:  meta.Taint,  // G5: provenance carried through the round-trip
		Len:    int64(len(fetched)),
	}
	v = AdmitL3SharedPage(L3SharedGet{
		Page:      page,
		OwnerTag:  meta.OwnerTag,
		ReaderTag: readerTag,
		Fetched:   fetched,
	})
	if !v.Admitted {
		// REFUSED (G1 mismatch or G4 scope) — fail-closed, the bytes are never served.
		return nil, meta, true, v
	}
	// ADMITTED: forward the EXACT slice the backend returned — identity-equal, no copy.
	return fetched, meta, true, v
}

// MockL3Backend is the LOCAL exemplar L3 tier the acceptance gate runs against: an
// in-memory map keyed by the page's content digest — the same content-addressing the
// real CAMA-complete store uses — standing in for the external pool reached over RDMA in
// production (no CAMA, no network). On Set it keeps its OWN copy of the bytes, as a real
// L3 pool holds the bytes independently of the caller's buffer; on Get it returns that
// SAME stored slice, so a sidecar that forwards it unmodified hands the caller a slice
// identity-equal to the tier's — the §6.5 control-path-only witness. Not concurrency-safe
// (the acceptance gate is single-goroutine); a real backend adds its own locking.
type MockL3Backend struct {
	pages      map[string]mockL3Page
	sets, gets int
}

// mockL3Page is one resident page: the tier's own copy of the bytes and its meta.
type mockL3Page struct {
	payload []byte
	meta    L3PageMeta
}

// NewMockL3Backend allocates an empty in-memory L3 exemplar.
func NewMockL3Backend() *MockL3Backend {
	return &MockL3Backend{pages: map[string]mockL3Page{}}
}

// Set stores the tier's own copy of the bytes under the page key (content-addressed:
// re-setting an identical key overwrites with the latest copy). Bytes are opaque.
func (m *MockL3Backend) Set(key string, payload []byte, meta L3PageMeta) {
	m.sets++
	m.pages[key] = mockL3Page{payload: append([]byte(nil), payload...), meta: meta}
}

// Get returns the SAME stored slice and meta for a key, or ok=false on a miss. Returning
// the stored slice (not a fresh copy) is what lets the gate witness that the sidecar
// forwards it identity-equal on the hot read path.
func (m *MockL3Backend) Get(key string) ([]byte, L3PageMeta, bool) {
	m.gets++
	p, ok := m.pages[key]
	if !ok {
		return nil, L3PageMeta{}, false
	}
	return p.payload, p.meta, true
}

// Stats reports set/get op counts for tests and KPI taps.
func (m *MockL3Backend) Stats() (sets, gets int) { return m.sets, m.gets }

// Exists is the CONTROL-PATH all-miss witness the L3 deletion rung (#56) folds: for each
// requested page-key in order it returns ONLY a presence bit — never the page bytes
// (§6.5). It is the exists/mget the rung runs AFTER a delete to witness that a span's
// backing pages stopped resolving in the shared pool. Reading only the map's key set, it
// touches no payload, so the mint path stays off the data path. len(result) == len(keys).
func (m *MockL3Backend) Exists(keys []string) []bool {
	m.gets += len(keys) // an exists probe is an mget over the key set
	out := make([]bool, len(keys))
	for i, k := range keys {
		_, ok := m.pages[k]
		out[i] = ok
	}
	return out
}

// Delete removes a page-key from the pool — the external store's fire-and-forget
// OP_DELETE / LRU eviction the deletion rung rides ABOVE (the rung adds the receipt the
// store omits, it does not reimplement the delete). It is idempotent (deleting an absent
// key is a no-op) and reports whether a page was actually resident.
func (m *MockL3Backend) Delete(key string) bool {
	_, ok := m.pages[key]
	delete(m.pages, key)
	return ok
}

// Compile-time proof the mock satisfies the backend seam the sidecar rides and the
// control-path witness seam the L3 deletion rung folds.
var (
	_ L3Backend     = (*MockL3Backend)(nil)
	_ L3PoolWitness = (*MockL3Backend)(nil)
)
