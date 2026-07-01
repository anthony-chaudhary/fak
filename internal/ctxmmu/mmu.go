// Package ctxmmu is the context-MMU: a write-time (post-tool) gate on tool
// RESULTS, the dual of the call-side adjudicator. It is the memory-management
// unit for the agent's context — it decides, at the moment a result would be
// written into the conversation, whether the bytes may enter as-is, must be
// QUARANTINED (held out — prompt-injection / secret / pollution), or PAGED OUT to
// a <2KB pointer (oversize but benign). The cold/quarantined bytes live in the
// shared content-addressed blob store (the same CAS the vDSO tier-2 uses), so a
// 100KB result costs the context a pointer, not 100KB (unit 65).
//
// It registers as a ResultAdmitter. Quarantined bytes are NEVER in context; they
// page in only on an explicit retrieve AFTER a witness clear() (unit 67).
package ctxmmu

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/numfmt"
)

const (
	// OversizeBytes: a result larger than this benign-pages-out to a pointer.
	OversizeBytes = 4096
	// PointerMax: the injected pointer must be smaller than this (unit 65).
	PointerMax = 2048
	// DefaultMaxHeld bounds the quarantine ledger so a long-lived gate driven by
	// a poison-heavy stream cannot grow held/cleared without bound (every
	// quarantine minted a permanent held["q<n>"] entry with no removal path —
	// the process-lifetime leak this cap closes). It matches the repo's other
	// process-lifetime ledgers (ifc.DefaultLedgerLimit, ratelimit.defaultMaxKeys
	// = 8192). When the cap is reached the OLDEST handles are dropped first; the
	// quarantined bytes themselves live in the shared CAS keyed by digest, so a
	// page-in of a dropped id simply refuses ("no quarantined result") — the
	// correct fail-closed degradation, never a bytes-into-context leak.
	DefaultMaxHeld = 8192
)

// secretPattern matches common secret shapes (unit 70).
var secretPattern = regexp.MustCompile(`(?i)(sk-[a-z0-9]{16,}|AKIA[0-9A-Z]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})`)

// injectionMarkers are prompt-injection tells (unit 68 poison fixture).
var injectionMarkers = []string{
	"ignore previous instructions", "ignore all previous", "disregard the above",
	"you are now", "system override", "new instructions:", "###system",
	"reveal your system prompt", "exfiltrate",
}

// MMU is the context memory-management unit. Construct with New; Default is
// registered.
type MMU struct {
	total      int64
	quarantine int64
	paged      int64
	evicted    int64 // held entries dropped by the maxHeld bound (observability)
	screened   int64 // results additively quarantined by a registered SemanticScreen
	digested   int64 // oversize results paged out to a digest-bearing stub (rung 3, ScreenDigest)
	capPaged   int64 // capability bodies paged out via PageOutBody (C3, issue #1106)

	mu        sync.Mutex
	held      map[string]abi.Ref // id -> paged-out handle (quarantined bytes)
	cleared   map[string]bool    // ids that passed a witness clear() (⊆ held)
	order     []string           // FIFO insertion order of held ids (bounded eviction)
	orderHead int                // consumed-prefix index into order (compacted in place)
	maxHeld   int                // cap on len(held); 0 in zero-value, set by constructors
	pageOutID string             // keyed page-out codec id (default "blob"; FAK_PAGEOUT_BACKEND)
}

// New builds the registered-default-shaped gate with the standard quarantine-ledger
// bound. The default (DefaultMaxHeld) is overridable at process start via the env var
// FAK_CTXMMU_MAX_HELD — the repo's "sensible default + escape hatch" idiom (FAK_WORKERS,
// FAK_BUDGET, FAK_NORMGATE): a tight box can shrink it, a high-memory server can raise it,
// neither needs a recompile.
func New() *MMU {
	return NewWithLimit(numfmt.EnvPositiveInt("FAK_CTXMMU_MAX_HELD", DefaultMaxHeld))
}

// NewWithLimit builds a gate whose quarantine ledger holds at most maxHeld entries
// (oldest dropped first). A non-positive maxHeld falls back to DefaultMaxHeld. This
// is the seam the leak-regression test uses to exercise eviction with a small bound.
func NewWithLimit(maxHeld int) *MMU {
	if maxHeld < 1 {
		maxHeld = DefaultMaxHeld
	}
	return &MMU{held: map[string]abi.Ref{}, cleared: map[string]bool{}, maxHeld: maxHeld, pageOutID: pageOutBackendID()}
}

// pageOutBackendID is the keyed page-out codec id the MMU pages cold/quarantined
// bytes through. It defaults to "blob" (the in-memory v0.1 codec) and is
// overridable at process start via FAK_PAGEOUT_BACKEND — the seam an operator uses
// to spill page-out to a DURABLE codec (e.g. "blobfs", or the storedrv router's
// id) so quarantined/cold bytes survive a process restart. Read once at
// construction, never on the hot path; an empty value falls back to "blob".
func pageOutBackendID() string {
	if id := os.Getenv("FAK_PAGEOUT_BACKEND"); id != "" {
		return id
	}
	return "blob"
}

// codecID returns the configured page-out codec id, falling back to "blob" for a
// zero-value MMU (a struct literal that bypassed the constructors).
func (m *MMU) codecID() string {
	if m.pageOutID == "" {
		return "blob"
	}
	return m.pageOutID
}

// NewWithHeldLimit is a back-compat alias for NewWithLimit (the pre-merge constructor
// name still used by ctxmmu_test.go). limit <= 0 falls back to DefaultMaxHeld.
func NewWithHeldLimit(limit int) *MMU { return NewWithLimit(limit) }

func (m *MMU) Caps() []abi.Capability { return nil }

// Admit is the write-time gate. It inspects the produced result and returns:
//
//	Allow      — bytes enter context unchanged
//	Quarantine — bytes held out (secret / injection / pollution); payload becomes
//	             a stub pointer in-place so the caller cannot accidentally use them
//	Transform  — oversize benign result paged out to a <2KB pointer
func (m *MMU) Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	atomic.AddInt64(&m.total, 1)
	body := m.bytes(ctx, r.Payload)

	// 1-3. unsafe result bytes -> quarantine (unit 70/68/62).
	if reason, ok := ScreenBytes(body); ok {
		return m.quarantineResult(ctx, r, reason, body)
	}
	// 3b. SEMANTIC screen — the local-model-on-the-wire seam (RESEARCH-local-model-on
	// -the-wire-2026-06-23.md). A registered abi.SemanticScreen may ADDITIVELY quarantine
	// a result the regex floor admitted: the injection-shaped payload with no literal
	// marker. It runs ONLY after the floor cleared the bytes, is strictly one-sided
	// (Allow, or a stricter Quarantine — never weaker), and routes a hit through the SAME
	// quarantineResult path so the held bytes keep the CAS-pin + PageIn-after-Clear
	// witness. With no screen registered this ranges a nil slice — the inert v0.1 default.
	//
	// A screen may instead return ScreenDigest (rung 3, issue #570): it authors a lossy
	// ~200-token summary that the oversize page-out stub below carries INSTEAD of an
	// opaque {_paged,ref,len} pointer, so the model reads the gist without a demand-page
	// fault. The first non-empty ScreenDigest advisory is captured here and applied in
	// the oversize branch; a later screen may still return ScreenQuarantine (strictly
	// stronger), which returns immediately. The original stays pinned in CAS and a gated
	// PageIn (after a witness Clear) restores it byte-exact — the digest is lossy display,
	// never the witness.
	var digestAdv abi.ScreenAdvice
	for _, sc := range abi.SemanticScreens() {
		adv := sc.ScreenResult(ctx, c, body)
		switch adv.Disposition {
		case abi.ScreenQuarantine:
			reason := adv.Reason
			if reason == abi.ReasonNone {
				reason = abi.ReasonTrustViolation
			}
			atomic.AddInt64(&m.screened, 1)
			return m.quarantineResult(ctx, r, reason, body)
		case abi.ScreenDigest:
			// Keep the first authored digest; continue scanning so a later screen can
			// still escalate to a Quarantine (a held-out body is never digested).
			if digestAdv.Digest == "" && adv.Digest != "" {
				digestAdv = adv
			}
		}
	}
	// The result survived the screen — classify its WRITE-TIME DURABILITY (S7 rung 1,
	// issue #82) and carry the class on the OPEN Verdict.Meta map so the durable
	// boundary downstream (recall's promotion gate) can refuse non-durable facts. The
	// tag is orthogonal to the trust Kind and additive over the frozen ABI, exactly
	// like the quarantine_id stamp above (mmu.go:107). A Quarantine verdict (returned
	// above) needs no class — sealed bytes never promote.
	class := classifyDurability(c, body)

	// 4. oversize-but-benign -> page out to a pointer (unit 65, Transform). If a
	// screen authored a digest (rung 3, issue #570), the stub carries that summary
	// instead of an opaque pointer and the original is pinned in CAS under the held
	// ledger so a witness Clear + PageIn restores it byte-exact.
	if len(body) > OversizeBytes {
		if digestAdv.Digest != "" {
			ptr, id, ok := m.digestToPointer(ctx, body, digestAdv.Digest, digestAdv.By)
			if ok {
				atomic.AddInt64(&m.paged, 1)
				if r.Meta == nil {
					r.Meta = map[string]string{}
				}
				r.Meta["pageout_id"] = id
				return abi.Verdict{Kind: abi.VerdictTransform, By: "ctxmmu",
					Payload: abi.TransformPayload{NewArgs: ptr},
					Meta:    map[string]string{DurabilityKey: class, "digest_by": digestAdv.By}}
			}
			// A digest page-out refused (no CAS backend / stub too big) falls through
			// to the opaque pointer — still a useful, reversible page-out.
		}
		ptr, ok := m.pageToPointer(ctx, r.Payload, body, "oversize")
		if ok {
			atomic.AddInt64(&m.paged, 1)
			return abi.Verdict{Kind: abi.VerdictTransform, By: "ctxmmu",
				Payload: abi.TransformPayload{NewArgs: ptr},
				Meta:    map[string]string{DurabilityKey: class}}
		}
	}
	// 5. admit as-is.
	return abi.Verdict{Kind: abi.VerdictAllow, By: "ctxmmu",
		Meta: map[string]string{DurabilityKey: class}}
}

// quarantineResult pages the offending bytes out, replaces the result payload
// in-place with a tiny stub so they are absent from context, records the held
// handle for a gated page-in, and returns the Quarantine verdict.
func (m *MMU) quarantineResult(ctx context.Context, r *abi.Result, reason abi.ReasonCode, body []byte) abi.Verdict {
	atomic.AddInt64(&m.quarantine, 1)
	id := fmt.Sprintf("q%d", atomic.LoadInt64(&m.quarantine))
	handle := m.pageOut(ctx, body)
	m.mu.Lock()
	m.held[id] = handle
	m.order = append(m.order, id)
	// Pin the held bytes UNDER m.mu so the bounded CAS cannot reclaim them before the
	// gated PageIn resolves them later; the FIFO bound unpins on eviction.
	abi.PinResolved(handle)
	m.evictExcessLocked()
	m.mu.Unlock()
	stub := map[string]any{"_quarantined": true, "id": id, "reason": abi.ReasonName(reason), "len": len(body)}
	if ref, ok := putJSON(ctx, stub); ok {
		r.Payload = ref // bytes now ABSENT from context
	} else {
		r.Payload = abi.Ref{Kind: abi.RefInline, Taint: abi.TaintQuarantined}
	}
	if r.Meta == nil {
		r.Meta = map[string]string{}
	}
	r.Meta["quarantine_id"] = id
	return abi.Verdict{Kind: abi.VerdictQuarantine, Reason: reason, By: "ctxmmu",
		Payload: abi.QuarantinePayload{PageOut: true}}
}

func (m *MMU) pageToPointer(ctx context.Context, orig abi.Ref, body []byte, hint string) (abi.Ref, bool) {
	handle := m.pageOut(ctx, body)
	stub := map[string]any{"_paged": true, "ref": handle.Digest, "len": len(body), "hint": hint}
	ref, ok := putJSON(ctx, stub)
	if !ok || ref.Len >= PointerMax {
		return abi.Ref{}, false
	}
	return ref, true
}

// digestToPointer is the rung-3 useful-page-out peer of pageToPointer (issue #570):
// it pages body out to CAS, pins the original under the held lock (the witness),
// records a held id so a gated PageIn (after a witness Clear) restores the original
// byte-exact, and builds a stub that carries the model-authored DIGEST instead of an
// opaque {_paged,ref,len} pointer. The model reads the gist from the digest without a
// demand-page fault; the original — never the lossy digest — is the witness.
//
// It is strictly additive over pageToPointer: it only fires when a screen authored a
// digest, and on failure (no CAS backend / stub too big) the caller falls back to the
// opaque pointer. ok is false (and the returns zero) when the witness cannot be
// established, so the page-out refuses rather than dropping bytes it cannot recover.
func (m *MMU) digestToPointer(ctx context.Context, body []byte, digest, by string) (ptr abi.Ref, id string, ok bool) {
	handle := m.pageOut(ctx, body)
	if handle.Digest == "" {
		return abi.Ref{}, "", false // no CAS backend -> refuse (never drop unwitnessed bytes)
	}
	// Build the stub first; if it overflows PointerMax we have not yet touched the
	// ledger or the counter, so there is nothing to roll back.
	stub := map[string]any{"_paged": true, "ref": handle.Digest, "len": len(body), "digest": digest}
	if by != "" {
		stub["digest_by"] = by
	}
	ref, putOK := putJSON(ctx, stub)
	if !putOK || ref.Len >= PointerMax {
		return abi.Ref{}, "", false
	}
	// Success: take the id sequence + lifetime count from the digested counter, then
	// pin the original under the held lock so the bounded ledger's eviction cannot
	// reclaim it before the gated PageIn resolves it — the same witness as quarantine.
	id = fmt.Sprintf("d%d", atomic.AddInt64(&m.digested, 1))
	m.mu.Lock()
	m.held[id] = handle
	m.order = append(m.order, id)
	abi.PinResolved(handle)
	m.evictExcessLocked()
	m.mu.Unlock()
	return ref, id, true
}

func (m *MMU) pageOut(ctx context.Context, body []byte) abi.Ref {
	if b, ok := abi.PageOut(m.codecID()); ok {
		inline := abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))}
		if h, err := b.PageOut(ctx, inline); err == nil {
			return h
		}
	}
	return abi.Ref{}
}

// Clear records a witness clearance for a quarantined id (unit 67): only after
// this may the bytes page back in. Clearing an id that is not currently held
// (never quarantined, or already aged out of the bounded ledger) is a no-op —
// PageIn refuses an unheld id regardless — which keeps cleared ⊆ held and so
// bounded by maxHeld (it cannot grow independently of the held ledger).
func (m *MMU) Clear(id string) {
	m.mu.Lock()
	if _, ok := m.held[id]; ok {
		m.cleared[id] = true
	}
	m.mu.Unlock()
}

// evictExcessLocked drops the oldest held quarantines (FIFO) until len(held) is
// within maxHeld, removing each id's cleared flag with it so cleared stays ⊆ held.
// The consumed prefix of order is compacted in place once it reaches half the
// slice, so order's backing array is bounded to ≈2·maxHeld and never leaks. The
// caller holds m.mu. Dropping a handle releases its CAS pin (taken in
// quarantineResult) and makes that id un-page-in-able; an evicted id's bytes were
// never in model-visible context, so a later PageIn(id) is refused like an unknown id
// (fail-closed — the safe direction), which is why releasing the pin here loses
// nothing still resolvable through the gate.
func (m *MMU) evictExcessLocked() {
	for len(m.held) > m.maxHeld && m.orderHead < len(m.order) {
		old := m.order[m.orderHead]
		m.orderHead++
		if h, ok := m.held[old]; ok {
			abi.UnpinResolved(h) // release the CAS pin taken in quarantineResult
			delete(m.held, old)
			delete(m.cleared, old)
			atomic.AddInt64(&m.evicted, 1)
		}
	}
	if m.orderHead > 0 && m.orderHead*2 >= len(m.order) {
		n := copy(m.order, m.order[m.orderHead:])
		m.order = m.order[:n]
		m.orderHead = 0
	}
}

// PageIn returns the quarantined bytes for id, but ONLY if it was cleared first
// (unit 67). An uncleared page-in is refused.
func (m *MMU) PageIn(ctx context.Context, id string) ([]byte, error) {
	m.mu.Lock()
	handle, ok := m.held[id]
	cleared := m.cleared[id]
	m.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("ctxmmu: no quarantined result %s", id)
	}
	if !cleared {
		return nil, fmt.Errorf("ctxmmu: page-in of %s refused — no witness clear()", id)
	}
	if b, has := abi.PageOut(m.codecID()); has {
		ref, err := b.PageIn(ctx, handle)
		if err != nil {
			return nil, err
		}
		return ref.Inline, nil
	}
	return nil, fmt.Errorf("ctxmmu: no page-out backend")
}

// Held returns a copy of the id->handle map for every quarantined result still
// held out of context. It is the read side the recall leaf needs to PERSIST the
// quarantine state across the process boundary (the bytes themselves already live
// in the shared content-addressed blob store, addressed by the handle Digest).
// Read-only: it copies, so a caller can never mutate the live gate. Additive — it
// changes no Admit/PageIn/Clear behaviour.
func (m *MMU) Held() map[string]abi.Ref {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]abi.Ref, len(m.held))
	for id, h := range m.held {
		out[id] = h
	}
	return out
}

// Cleared returns a copy of the set of ids that passed a witness Clear(). The
// recall leaf persists this alongside Held so a dead session reloads with its
// exact clearance state (an id cleared in the live session stays cleared; an
// uncleared injection stays sealed). Read-only copy; additive.
func (m *MMU) Cleared() map[string]bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]bool, len(m.cleared))
	for id, ok := range m.cleared {
		out[id] = ok
	}
	return out
}

// HeldLen reports the current number of quarantine handles in the bounded ledger
// (≤ maxHeld). Evicted reports how many were dropped by the bound over the gate's
// lifetime. Together they make the leak fix observable: HeldLen plateaus at the cap
// while Evicted climbs, instead of HeldLen growing without bound.
func (m *MMU) HeldLen() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.held)
}

// Evicted reports the lifetime count of held entries dropped by the maxHeld bound.
func (m *MMU) Evicted() int64 { return atomic.LoadInt64(&m.evicted) }

// Screened reports the lifetime count of results additively quarantined by a
// registered SemanticScreen (the local-model-on-the-wire seam) — i.e. results the
// regex floor admitted but a semantic screen held out. Zero on the inert default.
func (m *MMU) Screened() int64 { return atomic.LoadInt64(&m.screened) }

// Digested reports the lifetime count of oversize-benign results paged out to a
// digest-bearing stub via a ScreenDigest advisory (rung 3, issue #570) — the
// useful-page-out peer of Screened(). Zero on the inert default (no screen registered
// or none authored a digest).
func (m *MMU) Digested() int64 { return atomic.LoadInt64(&m.digested) }

// PollutionRate is quarantined/total (unit 66).
func (m *MMU) PollutionRate() (quarantined, total int64, rate float64) {
	q := atomic.LoadInt64(&m.quarantine)
	t := atomic.LoadInt64(&m.total)
	if t > 0 {
		rate = float64(q) / float64(t)
	}
	return q, t, rate
}

// Quarantined reports whether a result was held out of context.
func Quarantined(r *abi.Result) bool {
	return r != nil && r.Meta != nil && r.Meta["quarantine_id"] != ""
}

func (m *MMU) bytes(ctx context.Context, r abi.Ref) []byte {
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

func hasInjection(b []byte) bool {
	s := strings.ToLower(string(b))
	for _, m := range injectionMarkers {
		if strings.Contains(s, m) {
			return true
		}
	}
	return false
}

// repeats flags a result with a long run of one repeated chunk (a degenerate /
// pollution payload), independent of posttool stream-loop signals.
func repeats(b []byte) bool {
	if len(b) < 512 {
		return false
	}
	const chunk = 16
	first := b[:chunk]
	reps := 0
	for i := 0; i+chunk <= len(b); i += chunk {
		if string(b[i:i+chunk]) == string(first) {
			reps++
			if reps > 50 {
				return true
			}
		} else {
			reps = 0
		}
	}
	return false
}

// ScreenBytes reports whether bytes must be held out of model-visible transcript.
// It is the reusable predicate behind both the context-MMU's post-tool admission
// gate and closed-API clients' pre-send transcript quarantine.
func ScreenBytes(body []byte) (abi.ReasonCode, bool) {
	if secretPattern.Match(body) {
		return abi.ReasonSecretExfil, true
	}
	if hasInjection(body) {
		return abi.ReasonTrustViolation, true
	}
	if repeats(body) {
		return abi.ReasonOversize, true
	}
	return abi.ReasonNone, false
}

// ---------------------------------------------------------------------------
// Write-time durability classification (S7 rung 1 — issue #82).
//
// The write gate decides WHETHER a result may enter context; it never decided HOW
// LONG it should be believed. classifyDurability assigns a result a durability class
// from a cheap lexical/tense prior, and MMU.Admit stamps it on Verdict.Meta. The
// downstream durable boundary (recall's promotion gate) reads it to enforce the
// headline inversion: expire by default, promotion is the earned exception.
// ---------------------------------------------------------------------------

// Durability classes ride the OPEN Verdict.Meta map under DurabilityKey — orthogonal
// to the trust Kind, additive over the frozen ABI (TestABIGoldenFreeze does not move).
// v1 emits {turn, session, durable}; `bounded` is accepted as the explicit-expiry
// class even though the lexical prior does not infer it on its own. Readers degrade
// unknown values fail-closed to turn.
const (
	DurabilityKey     = "durability"
	DurabilityTurn    = "turn"    // true only this turn; the fail-closed default
	DurabilitySession = "session" // true for this session
	DurabilityBounded = "bounded" // true until a caller-supplied expiry
	DurabilityDurable = "durable" // true across sessions until revised — the only promotable class
)

const (
	ExpiryPolicyTurn     = "turn_end"
	ExpiryPolicySession  = "session_end"
	ExpiryPolicyRequired = "requires_expiry"
	ExpiryPolicyNone     = "none"
)

// DurabilityPolicy is the renderable context-ledger policy for a fact's truth duration.
type DurabilityPolicy struct {
	Class          string `json:"class"`
	ExpiryPolicy   string `json:"expiry_policy"`
	RequiresExpiry bool   `json:"requires_expiry,omitempty"`
}

// NormalizeDurabilityClass maps unknown durability tags to the fail-closed turn class.
func NormalizeDurabilityClass(class string) string {
	switch class {
	case DurabilityTurn, DurabilitySession, DurabilityBounded, DurabilityDurable:
		return class
	default:
		return DurabilityTurn
	}
}

// PolicyForDurability maps a durability class to its deterministic expiry policy.
func PolicyForDurability(class string) DurabilityPolicy {
	class = NormalizeDurabilityClass(class)
	switch class {
	case DurabilitySession:
		return DurabilityPolicy{Class: class, ExpiryPolicy: ExpiryPolicySession}
	case DurabilityBounded:
		return DurabilityPolicy{Class: class, ExpiryPolicy: ExpiryPolicyRequired, RequiresExpiry: true}
	case DurabilityDurable:
		return DurabilityPolicy{Class: class, ExpiryPolicy: ExpiryPolicyNone}
	default:
		return DurabilityPolicy{Class: DurabilityTurn, ExpiryPolicy: ExpiryPolicyTurn}
	}
}

// DurabilityLabel renders the compact ledger label a context fact can carry.
func DurabilityLabel(class string) string {
	p := PolicyForDurability(class)
	return "durability=" + p.Class + " expiry=" + p.ExpiryPolicy
}

var (
	// durableFrame: habitual/stative frames => durable (stated preferences, identity).
	// Deliberately NARROW — a false-positive promotion (a transient fact recalled as
	// current truth) is "strictly worse than absence" (CONTEXT-IS-NOT-MEMORY.md §4), so
	// weak copular/imperative alternations are excluded: `my <noun> is` is whitelisted
	// to identity/disposition nouns (NOT the generic `my \w+ is`, which fired on
	// `my build is failing right now`), and bare `i am a` / `call me` / `we work` are
	// dropped (they fire on `I am a bit busy`, `call me back later`, `we work until 5pm`).
	// RE2 has no negative lookahead, so the safe shape is a noun whitelist, not exclusion.
	durableFrame = regexp.MustCompile(`(?i)\b(prefers?|preferred|preference|i always|i usually|i normally|we (?:use|prefer)|my (?:name|role|title|pronouns?|timezone|tz|email|handle|username|nickname|birthday|address|favou?rite \w+) is)\b`)
	// sessionFrame: explicit session-scoped frames => session.
	sessionFrame = regexp.MustCompile(`(?i)(\bthis session\b|\bthis branch\b|\bworking on\b|\btoday'?s task\b|\bcurrent task\b|\bfor now\b)`)
	// turnFrame: punctual/progressive deictics + bare clock times => turn.
	turnFrame = regexp.MustCompile(`(?i)(\bright now\b|\bcurrently\b|\btoday\b|\bat the moment\b|\bas of now\b|\bit is now\b|\b\d{1,2}\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b|o'?clock)`)
)

// classifyDurability assigns a rung-1 write-time durability class to a produced result
// from a cheap lexical/tense prior over the bytes — NO model call, and explicitly NOT
// the Zhang-Choi fact-duration estimator (CONTEXT-IS-NOT-MEMORY.md §5), which has no
// callsite and is deferred. It leans on bytes (and may consult the tool) only; it does
// NOT take a turn index / session id / principal / as-of clock — threading those into
// the hot ResultAdmitter signature is a named follow-on, not this rung.
//
// Precedence is most-durable-first so a stated preference is durable even if it also
// mentions "today"; a clearly session-scoped frame ("today's task") beats the bare
// "today" deictic; everything unmatched fails closed to turn, because a false-positive
// promotion (a poltergeist fact recalled as current) is the expensive error direction.
func classifyDurability(c *abi.ToolCall, body []byte) string {
	_ = c // reserved: a future tool prior (a clock/now source is inherently turn-class)
	switch {
	case durableFrame.Match(body):
		return DurabilityDurable
	case sessionFrame.Match(body):
		return DurabilitySession
	case turnFrame.Match(body):
		return DurabilityTurn
	default:
		return DurabilityTurn
	}
}

// ClassifyText is the exported, chat-message-shaped entry to the SAME rung-1
// durability prior classifyDurability runs over tool-result bytes — it lets a caller
// outside the admit path (e.g. a budget-reset carryover builder that must decide which
// transcript lines a fresh session keeps) reuse the shipped tense/deixis classifier
// instead of reinventing it. It runs the identical durableFrame/sessionFrame/turnFrame
// priors over the message text and fails closed to turn, so "it's 3pm" => turn and "I
// prefer afternoons" => durable, exactly as the admit path classifies the same words.
//
// It takes the message text only (no tool call): a chat line has no producing tool,
// and classifyDurability's tool argument is reserved/unused at rung 1. role is accepted
// for forward compatibility (a future prior may weight assistant vs user vs system text)
// but does not change the rung-1 verdict.
func ClassifyText(role, content string) string {
	_ = role // reserved: a future prior may weight by author; rung 1 is text-only
	return classifyDurability(nil, []byte(content))
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

// Default is the registered MMU.
var Default = New()

func init() {
	abi.RegisterResultAdmitter(10, Default)
	abi.RegisterCapability("ctxmmu.v1")
}
