// Package normgate is a write-time ResultAdmitter that closes the context-MMU's
// measured DETECTION gap: the v0.1 ctxmmu matches injection markers and secret
// shapes as raw ASCII regex/substring, so any obfuscation (char-spacing, base64,
// homoglyph, zero-width, fullwidth, bidi, format-variant secrets) walks straight
// through (~100% evasion, measured on a private transcript-derived corpus). normgate
// is a NORMALIZE-AND-RESCAN driver: it canonicalizes a COPY of the result bytes
// (strip invisibles, fold homoglyph/fullwidth, reverse bidi, decode base64/hex,
// de-separate single-letter runs), broadens the secret vocabulary to real formats
// the regex never enumerated, and runs the detectors over that canonical view.
//
// It registers at rank 5 — BEFORE ctxmmu (rank 10) — so it composes in FRONT of
// the existing gate: a hit pages the bytes out and stubs the payload in place
// (same convention as ctxmmu), and the rank-10 gate then sees a benign stub. This
// is the "plug a SOTA-informed driver into the fused core" demonstration — the
// kernel walks the registry; enabling normgate is one blank-import line.
//
// PROVENANCE (the ABI Ref.Taint seam, unused by ctxmmu v0.1): the fold is
// monotonic (most-restrictive wins), so a chained driver can only ADD restriction,
// never remove a false positive a later driver imposes. normgate therefore treats
// provenance as a FIRST-CLASS input to its OWN verdict: an injection-shaped result
// whose source is a TRUSTED LOCAL read (the agent reading its own files) is paged
// out as a RETRIEVABLE Transform, not sealed as a Quarantine — closing the
// "quarantined my own source code" false positives. Untrusted egress still
// quarantines. Secrets quarantine regardless of source (a leaked credential is
// worth holding even from a local read).
//
// The de-obfuscating canonicalization + lexical detection lives in internal/canon
// (one primitive, tested once, reused by the read-time recall re-screen too).
// normgate keeps only the POLICY: what to DO with a canon finding given its
// provenance.
package normgate

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/canon"
	"github.com/anthony-chaudhary/fak/internal/provenance"
)

// enabled is the runtime toggle (FAK_NORMGATE=off makes Admit a no-op Defer) so
// the before/after A/B can be measured against the SAME binary.
var enabled = os.Getenv("FAK_NORMGATE") != "off"

// trustedLocal reports whether a result came from the agent reading its OWN local
// environment (a trusted, agent-initiated read) rather than untrusted egress. It
// delegates to the kernel-authored internal/provenance classifier so normgate and
// ifc share ONE definition of trust, and so a model can never decide its own
// content is trusted-local: provenance keys on the kernel-stamped result state and
// the host-registered tool source class, and ignores the model-forgeable
// ToolCall.Meta entirely (the old Meta["provenance"]="trusted_local" self-tag,
// which let a poisoned read page out retrievably instead of sealing, is gone).
func trustedLocal(c *abi.ToolCall, r *abi.Result) bool {
	return provenance.Trusted(c, r)
}

// DefaultMaxHeld bounds the quarantine-handle ledger so a long-lived gate on a
// poison/obfuscation-heavy stream cannot grow `held` without bound. Like ctxmmu's
// twin ledger (and ifc/ratelimit), this is a process-lifetime cap: every quarantine
// minted a permanent held["ng-q<n>"] entry with no removal path, so the registered
// rank-5 Default gate leaked one entry per caught threat for the life of the server.
// Oldest handles drop first (FIFO); the bytes live in the shared CAS keyed by digest.
const DefaultMaxHeld = 8192

// Gate is the normalize-and-rescan ResultAdmitter.
type Gate struct {
	total      int64
	quarantine int64
	transform  int64
	evicted    int64 // held entries dropped by the maxHeld bound (observability)
	mu         sync.Mutex
	held       map[string]abi.Ref
	cleared    map[string]bool // ids that passed a witness Clear() — the #76 page-in gate
	order      []string        // FIFO insertion order of held ids, for bounded eviction
	orderHead  int             // consumed-prefix index into order (compacted in place)
	maxHeld    int             // cap on len(held)
}

// New builds the registered-default-shaped gate with the standard ledger bound, overridable
// at process start via FAK_NORMGATE_MAX_HELD (the FAK_WORKERS/FAK_BUDGET/FAK_NORMGATE idiom:
// a sensible default with an env escape hatch).
func New() *Gate { return NewWithLimit(envPositiveInt("FAK_NORMGATE_MAX_HELD", DefaultMaxHeld)) }

// envPositiveInt returns the positive integer in env var key, or def if unset/empty/
// non-numeric/<= 0 (a bad value fails safe to the default, never to 0).
func envPositiveInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// NewWithLimit builds a gate whose held ledger holds at most maxHeld handles (oldest
// dropped first). A non-positive maxHeld falls back to DefaultMaxHeld. Used by the
// leak-regression test to exercise eviction with a small bound.
func NewWithLimit(maxHeld int) *Gate {
	if maxHeld < 1 {
		maxHeld = DefaultMaxHeld
	}
	return &Gate{held: map[string]abi.Ref{}, cleared: map[string]bool{}, maxHeld: maxHeld}
}

// Caps reports the gate negotiates no optional capabilities (always nil).
func (g *Gate) Caps() []abi.Capability { return nil }

func (g *Gate) bytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

// Admit normalizes-and-rescans. On a detected (and provenance-relevant) threat it
// performs the page-out side-effect in place (ctxmmu's convention) and returns the
// verdict; the rank-10 gate then sees a benign stub.
func (g *Gate) Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	if !enabled {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "normgate(off)"}
	}
	atomic.AddInt64(&g.total, 1)
	body := g.bytes(ctx, r.Payload)
	if len(body) == 0 {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "normgate"}
	}
	f := canon.Scan(body)
	secret, injection := f.Secret, f.Injection
	if !secret && !injection {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "normgate"} // let ctxmmu handle oversize/verbatim
	}

	// Secret (any provenance) -> quarantine. Injection from UNTRUSTED egress ->
	// quarantine. Injection from a TRUSTED LOCAL read -> retrievable Transform.
	if secret {
		return g.quarantineOut(ctx, r, abi.ReasonSecretExfil, body)
	}
	if !trustedLocal(c, r) {
		return g.quarantineOut(ctx, r, abi.ReasonTrustViolation, body)
	}
	return g.transformOut(ctx, r, body) // trusted-local injection-shaped: page out, retrievable
}

func (g *Gate) quarantineOut(ctx context.Context, r *abi.Result, reason abi.ReasonCode, body []byte) abi.Verdict {
	atomic.AddInt64(&g.quarantine, 1)
	id := fmt.Sprintf("ng-q%d", atomic.LoadInt64(&g.quarantine))
	handle := g.pageOut(ctx, body)
	g.mu.Lock()
	g.held[id] = handle
	g.order = append(g.order, id)
	// Pin the held bytes UNDER g.mu so the bounded CAS cannot reclaim them before the
	// gated PageIn (#76) and its re-screen resolve them; the FIFO bound unpins on
	// eviction. Mirrors ctxmmu.quarantineResult.
	abi.PinResolved(handle)
	g.evictExcessLocked()
	g.mu.Unlock()
	stub := map[string]any{"_quarantined": true, "id": id, "reason": abi.ReasonName(reason),
		"by": "normgate", "len": len(body), "_note": "obfuscated threat caught on normalized view"}
	if ref, ok := putJSON(ctx, stub); ok {
		r.Payload = ref
	} else {
		r.Payload = abi.Ref{Kind: abi.RefInline, Taint: abi.TaintQuarantined}
	}
	if r.Meta == nil {
		r.Meta = map[string]string{}
	}
	r.Meta["quarantine_id"] = id
	r.Meta["normgate"] = "quarantined"
	return abi.Verdict{Kind: abi.VerdictQuarantine, Reason: reason, By: "normgate",
		Payload: abi.QuarantinePayload{PageOut: true}}
}

// Clear records a witness clearance for a quarantined id (the page-in gate). It is
// NECESSARY but NOT SUFFICIENT: PageIn still re-screens the bytes, so clearing a held
// credential does not launder it back into context. Clearing an unknown/evicted id is
// a harmless no-op — PageIn refuses it anyway.
func (g *Gate) Clear(id string) {
	g.mu.Lock()
	g.cleared[id] = true
	g.mu.Unlock()
}

// PageIn is the gated read of the held quarantine map (#76): the bytes paged out at
// quarantineOut are returned ONLY through this gate, never by a raw map read. It
// enforces two gates, mirroring recall's read-time re-screen:
//
//  1. a witness Clear(id) must have run — an uncleared (or unknown/evicted) id is
//     refused fail-closed, exactly like ctxmmu.PageIn;
//  2. the retrieved bytes are RE-SCREENED through canon.Scan — the same de-obfuscating
//     primitive normgate uses at write time. A SECRET re-screen hit refuses release
//     even after a witness clear: a leaked credential does not launder back into
//     context (normgate's secrets-are-absolute policy, normgate.go top-of-file, joined
//     with recall's "clearance does not launder poison"). An injection-only hit is
//     released so a witness can retrieve the caught payload for forensics/inspection.
//
// The fresh Findings ride back with the bytes so a retrieval is never blind to what it
// is handling. An evicted id's CAS bytes were unpinned and may be gone — that surfaces
// as a resolve error, the safe direction.
func (g *Gate) PageIn(ctx context.Context, id string) ([]byte, canon.Findings, error) {
	g.mu.Lock()
	handle, ok := g.held[id]
	cleared := g.cleared[id]
	g.mu.Unlock()
	if !ok {
		return nil, canon.Findings{}, fmt.Errorf("normgate: no quarantined result %s", id)
	}
	if !cleared {
		return nil, canon.Findings{}, fmt.Errorf("normgate: page-in of %s refused — no witness Clear()", id)
	}
	b, has := abi.PageOut("blob")
	if !has {
		return nil, canon.Findings{}, fmt.Errorf("normgate: no page-out backend")
	}
	ref, err := b.PageIn(ctx, handle)
	if err != nil {
		return nil, canon.Findings{}, fmt.Errorf("normgate: page-in of %s: %w", id, err)
	}
	body := ref.Inline
	f := canon.Scan(body)
	if f.Secret {
		return nil, f, fmt.Errorf("normgate: page-in of %s refused — re-screen found a secret; a cleared credential does not launder back into context", id)
	}
	return body, f, nil
}

func (g *Gate) transformOut(ctx context.Context, r *abi.Result, body []byte) abi.Verdict {
	atomic.AddInt64(&g.transform, 1)
	handle := g.pageOut(ctx, body)
	stub := map[string]any{"_paged": true, "ref": handle.Digest, "len": len(body),
		"by": "normgate", "hint": "trusted-local injection-shaped content (retrievable, not sealed)"}
	ref, ok := putJSON(ctx, stub)
	if !ok {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "normgate"}
	}
	r.Payload = ref
	if r.Meta == nil {
		r.Meta = map[string]string{}
	}
	r.Meta["normgate"] = "paged-trusted-local"
	return abi.Verdict{Kind: abi.VerdictTransform, By: "normgate", Payload: abi.TransformPayload{NewArgs: ref}}
}

func (g *Gate) pageOut(ctx context.Context, body []byte) abi.Ref {
	if b, ok := abi.PageOut("blob"); ok {
		inline := abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))}
		if h, err := b.PageOut(ctx, inline); err == nil {
			return h
		}
	}
	return abi.Ref{}
}

func putJSON(ctx context.Context, v any) (abi.Ref, bool) {
	b, err := json.Marshal(v)
	if err != nil {
		return abi.Ref{}, false
	}
	if res := abi.ActiveResolver(); res != nil {
		if ref, err := res.Put(ctx, b); err == nil {
			return ref, true
		}
	}
	return abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b))}, true
}

// evictExcessLocked drops the oldest held handles (FIFO) until len(held) is within
// maxHeld. The consumed prefix of order is compacted in place once it reaches half the
// slice, so order's backing array stays bounded to ≈2·maxHeld. Caller holds g.mu.
func (g *Gate) evictExcessLocked() {
	for len(g.held) > g.maxHeld && g.orderHead < len(g.order) {
		old := g.order[g.orderHead]
		g.orderHead++
		if h, ok := g.held[old]; ok {
			abi.UnpinResolved(h) // release the CAS pin taken in quarantineOut (#76)
			delete(g.held, old)
			delete(g.cleared, old)
			atomic.AddInt64(&g.evicted, 1)
		}
	}
	if g.orderHead > 0 && g.orderHead*2 >= len(g.order) {
		n := copy(g.order, g.order[g.orderHead:])
		g.order = g.order[:n]
		g.orderHead = 0
	}
}

// HeldLen reports the current number of quarantine handles in the bounded ledger
// (≤ maxHeld); Evicted reports how many were dropped by the bound over the gate's
// lifetime. Together they make the bound observable (HeldLen plateaus while Evicted
// climbs) — and give `held` a reader so the ledger is not write-only dead state.
func (g *Gate) HeldLen() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.held)
}

// Evicted reports the lifetime count of held entries dropped by the maxHeld bound.
func (g *Gate) Evicted() int64 { return atomic.LoadInt64(&g.evicted) }

// Stats reports normgate's tallies.
func (g *Gate) Stats() (total, quarantined, transformed int64) {
	return atomic.LoadInt64(&g.total), atomic.LoadInt64(&g.quarantine), atomic.LoadInt64(&g.transform)
}

// Default is the registered gate.
var Default = New()

func init() {
	abi.RegisterResultAdmitter(5, Default) // rank 5: BEFORE ctxmmu (rank 10)
	abi.RegisterCapability("normgate.v1")
}
