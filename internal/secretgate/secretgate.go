// Package secretgate is the execution-time, on-discovery SECRET rung (epic #880,
// pillar [B], issue #884). It is a thin write-time abi.ResultAdmitter sibling of
// internal/normgate that does ONE thing: when a tool result bears a credential,
// classify it as a CREDENTIAL DISCOVERY event (distinct from a poisoned/injected
// result), quarantine the bytes under a stable handle with the same CAS-pin +
// PageIn-after-Clear witness normgate/ctxmmu already mint, and record a structured
// Finding (+ an optional witness.Decision) an operator can act on.
//
// It DELEGATES detection to internal/canon (canon.Scan over the de-obfuscating
// canonical views) and locates spans with the EXPORTED canon.SecretPatterns — it
// never forks the pattern list. This is a discovery/record layer over the one
// detector, not a second detector.
//
// DEFAULT-INERT, OPT-IN: unlike normgate (default-on), secretgate is OFF unless
// FAK_SECRETGATE is opted in (on/1/true/yes). When off, Admit returns a no-op
// Defer, so the default behavior is exactly today's normgate-only secret path —
// this rung is purely additive. When on it registers at rank 4 (BEFORE normgate's
// rank 5), so the secret-discovery classification runs in front and normgate then
// sees a benign stub.
//
// SECRET vs EXFIL (the #152 anti-pattern retirement): the verdict reason here is
// abi.ReasonSecretDiscovered ("RESULT_SECRET_DISCOVERED"), the ON-DISCOVERY event.
// abi.ReasonSecretExfil ("SECRET_EXFIL") stays the EGRESS verdict. Detection
// semantics live in this rung (and #885's posture), not in a static deny grammar.
package secretgate

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/canon"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

// enabled is the opt-in runtime toggle. Default OFF so the registered rung is a
// no-op Defer and today's normgate-only secret path is preserved byte-for-byte;
// set FAK_SECRETGATE=on (or 1/true/yes) to activate the discovery rung. It is a
// package var (not read per-call) so tests can flip it deterministically, the
// internal/normgate idiom.
var enabled = parseEnabled(os.Getenv("FAK_SECRETGATE"))

func parseEnabled(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "on", "1", "true", "yes":
		return true
	default:
		return false
	}
}

// DefaultMaxHeld bounds the quarantine-handle ledger so a long-lived gate on a
// secret-heavy stream cannot grow `held` without bound (the normgate idiom: a
// process-lifetime FIFO cap, oldest handle unpinned first). It also bounds the
// Finding ledger.
const DefaultMaxHeld = 8192

// Finding is the structured, durable record of one on-discovery secret event. It
// is digest-only by construction — it never holds the cleartext (that stays in the
// CAS pin, reachable only via the gated PageIn). #886 escalates on a repeat Digest.
type Finding struct {
	Kind       string `json:"kind"`                // "secret" (a raw canon.SecretPatterns match) | "obfuscated" (canon.Scan caught it on a canonical view)
	Location   string `json:"location"`            // a human handle for where in the body it sat ("offset N..M" | "obfuscated")
	Confidence string `json:"confidence"`          // "medium" | "high" | "critical" (escalated on a repeat sighting)
	Digest     string `json:"digest"`              // sha256 hex (truncated) of the matched span — never the secret itself
	Handle     string `json:"handle"`              // the stable quarantine handle (refs/fak/secrets/<nonce>)
	Len        int    `json:"len"`                 // length of the quarantined result body
	Sightings  int    `json:"sightings"`           // how many times this digest has been seen this session (1 = first)
	Escalated  bool   `json:"escalated,omitempty"` // true once a repeat sighting raised confidence (#886)
}

// Gate is the on-discovery secret ResultAdmitter.
type Gate struct {
	total      int64
	discovered int64
	evicted    int64

	mu        sync.Mutex
	held      map[string]abi.Ref // handle -> CAS pin of the quarantined body
	cleared   map[string]bool    // handles that passed a witness Clear() (the #76 page-in gate)
	findings  []Finding          // bounded ledger of discovery records (observability)
	order     []string           // FIFO insertion order of held handles, for bounded eviction
	orderHead int
	maxHeld   int

	digests *digestTracker // bounded digest->sighting-count map for reuse escalation (#886)

	rec atomic.Pointer[witness.Recorder] // optional durable sink (nil = in-memory Finding only)
}

// digestTracker is the bounded set of discovered secret-span digests with their
// sighting counts (#886). It holds DIGESTS only — never the cleartext, which stays
// in the CAS pin. A secret seen once is ambiguous (placeholder/example/fixture); the
// SAME digest seen again is strong evidence of a real, live credential handled
// repeatedly, so the second sighting escalates. Bounded FIFO so a long, secret-heavy
// session cannot grow the map without bound.
type digestTracker struct {
	mu    sync.Mutex
	count map[string]int
	order []string
	head  int
	cap   int
}

// sight records one sighting of dig and returns its new cumulative count (1 on the
// first sighting). Oldest digests are dropped first once the cap is exceeded.
func (d *digestTracker) sight(dig string) int {
	d.mu.Lock()
	defer d.mu.Unlock()
	if _, existed := d.count[dig]; !existed {
		d.order = append(d.order, dig)
	}
	d.count[dig]++
	n := d.count[dig]
	for len(d.count) > d.cap && d.head < len(d.order) {
		old := d.order[d.head]
		d.head++
		delete(d.count, old)
	}
	if d.head > 0 && d.head*2 >= len(d.order) {
		m := copy(d.order, d.order[d.head:])
		d.order = d.order[:m]
		d.head = 0
	}
	return n
}

// size reports the current number of tracked digests (≤ cap) — for the bounded-set
// witness (leakcheck.BoundedSize).
func (d *digestTracker) size() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.count)
}

// New builds the registered-default-shaped gate with the standard ledger bound,
// overridable via FAK_SECRETGATE_MAX_HELD.
func New() *Gate { return NewWithLimit(envPositiveInt("FAK_SECRETGATE_MAX_HELD", DefaultMaxHeld)) }

// NewWithLimit builds a gate whose held ledger holds at most maxHeld handles
// (oldest dropped first). A non-positive maxHeld falls back to DefaultMaxHeld.
func NewWithLimit(maxHeld int) *Gate {
	if maxHeld < 1 {
		maxHeld = DefaultMaxHeld
	}
	return &Gate{held: map[string]abi.Ref{}, cleared: map[string]bool{}, maxHeld: maxHeld,
		digests: &digestTracker{count: map[string]int{}, cap: maxHeld}}
}

func envPositiveInt(key string, def int) int {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		var n int
		if _, err := fmt.Sscan(v, &n); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// SetRecorder wires an optional durable witness sink. When set, a discovery also
// appends a witness.Decision (Op=result-admit, Verdict=refuse,
// ReasonClass=RESULT_SECRET_DISCOVERED) to refs/notes/fak/decisions, best-effort —
// a recorder failure never blocks admission. Pass nil to detach. Default nil so
// the hot serve path takes no git dependency unless an operator opts in.
func (g *Gate) SetRecorder(r *witness.Recorder) { g.rec.Store(r) }

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

// Admit runs the on-discovery secret classification. With the rung off (default)
// or no secret found it returns Defer (a no-op in the fold). On a canon-confirmed
// secret it pages the bytes out under a stable handle, records the Finding (+
// optional witness.Decision), stubs the payload in place, and returns a Quarantine
// verdict carrying ReasonSecretDiscovered.
func (g *Gate) Admit(ctx context.Context, _ *abi.ToolCall, r *abi.Result) abi.Verdict {
	if !enabled {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "secretgate(off)"}
	}
	atomic.AddInt64(&g.total, 1)
	body := g.bytes(ctx, r.Payload)
	if len(body) == 0 {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "secretgate"}
	}
	if !canon.Scan(body).Secret {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "secretgate"} // clean (of secrets) -> let the next link decide
	}
	return g.quarantineSecret(ctx, r, body)
}

func (g *Gate) quarantineSecret(ctx context.Context, r *abi.Result, body []byte) abi.Verdict {
	atomic.AddInt64(&g.discovered, 1)
	handle := newHandle()
	pin := g.pageOut(ctx, body)
	f := classifySpans(body, handle, pin)
	g.escalate(f) // #886: a repeat digest raises confidence/severity before the record is filed

	g.mu.Lock()
	g.held[handle] = pin
	g.order = append(g.order, handle)
	abi.PinResolved(pin) // hold the bytes under g.mu so the bounded CAS can't reclaim them before a gated PageIn
	g.findings = append(g.findings, f...)
	g.evictExcessLocked()
	g.mu.Unlock()

	g.record(ctx, f)

	stub := map[string]any{"_quarantined": true, "handle": handle, "reason": abi.ReasonName(abi.ReasonSecretDiscovered),
		"by": "secretgate", "len": len(body), "findings": len(f), "_note": "secret discovered in tool result; held out of context"}
	if ref, ok := putJSON(ctx, stub); ok {
		r.Payload = ref
	} else {
		r.Payload = abi.Ref{Kind: abi.RefInline, Taint: abi.TaintQuarantined}
	}
	if r.Meta == nil {
		r.Meta = map[string]string{}
	}
	r.Meta["secret_handle"] = handle
	r.Meta["secretgate"] = "quarantined"
	return abi.Verdict{Kind: abi.VerdictQuarantine, Reason: abi.ReasonSecretDiscovered, By: "secretgate",
		Payload: abi.QuarantinePayload{PageOut: true}}
}

// classifySpans builds the Finding(s) for a discovered-secret body. It locates raw
// matches with the EXPORTED canon.SecretPatterns (delegation, not a fork) and
// digests each matched span; if canon.Scan caught a secret only on a canonical
// view (obfuscated) so no raw span is locatable, it records one "obfuscated"
// Finding whose digest is over the whole body. Digest-only — never the cleartext.
func classifySpans(body []byte, handle string, _ abi.Ref) []Finding {
	var out []Finding
	for _, re := range canon.SecretPatterns {
		for _, loc := range re.FindAllIndex(body, -1) {
			out = append(out, Finding{
				Kind:       "secret",
				Location:   fmt.Sprintf("offset %d..%d", loc[0], loc[1]),
				Confidence: "high",
				Digest:     digest(body[loc[0]:loc[1]]),
				Handle:     handle,
				Len:        len(body),
			})
		}
	}
	if len(out) == 0 {
		// canon.Scan flagged it on a de-obfuscated view but the raw bytes carry no
		// locatable span — record the event at medium confidence over the body digest.
		out = append(out, Finding{
			Kind:       "obfuscated",
			Location:   "obfuscated",
			Confidence: "medium",
			Digest:     digest(body),
			Handle:     handle,
			Len:        len(body),
		})
	}
	return out
}

// escalate records a sighting of each finding's digest and, on a REPEAT (the same
// secret span seen again this session), raises the finding's confidence and marks
// it Escalated. A token seen once is ambiguous; the same token seen again is strong
// evidence it is a real, live credential being handled repeatedly (#886). The
// optional verdict tighten on a repeat (quarantine -> fail_closed) is gated on the
// #885 secret-detection posture; the Escalated flag is the seam that reader uses.
func (g *Gate) escalate(findings []Finding) {
	if g.digests == nil {
		return
	}
	for i := range findings {
		n := g.digests.sight(findings[i].Digest)
		findings[i].Sightings = n
		if n >= 2 {
			findings[i].Confidence = escalatedConfidence(findings[i].Confidence)
			findings[i].Escalated = true
		}
	}
}

// escalatedConfidence bumps a confidence one rung on a repeat sighting.
func escalatedConfidence(c string) string {
	switch c {
	case "high":
		return "critical"
	case "medium":
		return "high"
	default:
		return c
	}
}

// record appends a witness.Decision per discovery to the durable sink when one is
// wired. Best-effort: a recorder error is swallowed so a git hiccup never blocks
// admission. The note anchors to the empty-tree sentinel (no commit to bind to at
// result-admit time).
func (g *Gate) record(ctx context.Context, findings []Finding) {
	rec := g.rec.Load()
	if rec == nil || len(findings) == 0 {
		return
	}
	d := witness.Decision{
		Op:          "result-admit",
		Verdict:     witness.VerdictRefuse,
		ReasonClass: abi.ReasonName(abi.ReasonSecretDiscovered),
		Tree:        []string{findings[0].Handle},
	}
	_ = rec.AppendDecision(ctx, "", d)
}

// Clear records a witness clearance for a held handle (the page-in gate). Necessary
// but not sufficient: PageIn still re-screens, so clearing a held credential does
// not launder it back into context. Clearing an unknown/evicted handle is a no-op.
func (g *Gate) Clear(handle string) {
	g.mu.Lock()
	g.cleared[handle] = true
	g.mu.Unlock()
}

// PageIn is the gated read of a held quarantine (the #76 page-in gate). The bytes
// are returned ONLY through this gate and ONLY when (1) a witness Clear(handle) has
// run and (2) a fresh canon.Scan re-screen of the retrieved bytes does NOT find a
// secret. A secret re-screen hit refuses release even after a clear: a leaked
// credential does not launder back into context (secrets-are-absolute). An
// uncleared/unknown/evicted handle is refused fail-closed.
func (g *Gate) PageIn(ctx context.Context, handle string) ([]byte, error) {
	g.mu.Lock()
	pin, ok := g.held[handle]
	cleared := g.cleared[handle]
	g.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("secretgate: no quarantined result %s", handle)
	}
	if !cleared {
		return nil, fmt.Errorf("secretgate: page-in of %s refused — no witness Clear()", handle)
	}
	b, has := abi.PageOut("blob")
	if !has {
		return nil, fmt.Errorf("secretgate: no page-out backend")
	}
	ref, err := b.PageIn(ctx, pin)
	if err != nil {
		return nil, fmt.Errorf("secretgate: page-in of %s: %w", handle, err)
	}
	body := ref.Inline
	if canon.Scan(body).Secret {
		return nil, fmt.Errorf("secretgate: page-in of %s refused — re-screen still finds a secret; a cleared credential does not launder back into context", handle)
	}
	return body, nil
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
// maxHeld, unpinning each dropped handle's CAS bytes. It also caps the Finding
// ledger to maxHeld (newest kept). Caller holds g.mu.
func (g *Gate) evictExcessLocked() {
	for len(g.held) > g.maxHeld && g.orderHead < len(g.order) {
		old := g.order[g.orderHead]
		g.orderHead++
		if h, ok := g.held[old]; ok {
			abi.UnpinResolved(h)
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
	if len(g.findings) > g.maxHeld {
		g.findings = append(g.findings[:0], g.findings[len(g.findings)-g.maxHeld:]...)
	}
}

// Findings returns a copy of the current discovery ledger (observability).
func (g *Gate) Findings() []Finding {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make([]Finding, len(g.findings))
	copy(out, g.findings)
	return out
}

// HeldLen reports the current number of quarantine handles in the bounded ledger.
func (g *Gate) HeldLen() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.held)
}

// Stats reports the gate's lifetime tallies.
func (g *Gate) Stats() (total, discovered, evicted int64) {
	return atomic.LoadInt64(&g.total), atomic.LoadInt64(&g.discovered), atomic.LoadInt64(&g.evicted)
}

// digest is the non-reversible span fingerprint: sha256 truncated to 16 hex chars.
// It identifies a repeat sighting (#886) without ever persisting the cleartext.
func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:8])
}

// newHandle mints a stable, unguessable quarantine handle under the secrets ref
// namespace. The handle is the ledger key + the operator-facing reference; the
// bytes live in the CAS keyed by the pin's digest.
func newHandle() string {
	var n [8]byte
	if _, err := rand.Read(n[:]); err != nil {
		// crypto/rand failure is fatal-rare; fall back to a discovery-counter-free
		// constant-shaped handle so the caller still gets a valid namespaced key.
		return "refs/fak/secrets/0000000000000000"
	}
	return "refs/fak/secrets/" + hex.EncodeToString(n[:])
}

// Default is the registered gate.
var Default = New()

func init() {
	abi.RegisterResultAdmitter(4, Default) // rank 4: BEFORE normgate (rank 5) so secret discovery runs in front
	abi.RegisterCapability("secretgate.v1")
}
