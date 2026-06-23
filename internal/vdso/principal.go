package vdso

// principal.go — MULTI-TENANT ISOLATION for the tier-2 content cache.
//
// The problem. The tier-2 key (vdso.go keyLocked) is tool:argHash:epoch — it carries
// NO caller identity ("agent-blind", by design, so one agent's warmed read is served
// to the next). That is the right default for PUBLIC knowledge (a shared policy doc),
// but it silently assumes a tool's result is a pure function of (tool, args). For an
// IDENTITY-DEPENDENT tool — one whose result depends on WHO is asking but whose args do
// not encode them: whoami, read_inbox{}, get_my_account{}, list_my_files{} — two
// different principals issuing the SAME (tool,args) are served ONE cached result, so
// principal B reads principal A's private bytes: a cross-user data leak. Even for a
// fully arg-keyed tool, the shared key exposes a hit/miss TIMING ORACLE across
// principals ("did someone else recently query this exact (tool,args)?").
//
// The fix, in the same shape as every other vDSO soundness gate ("a change can only
// turn a hit into a miss -> engine, fresh"): bind an optional PRINCIPAL — a tenant /
// user / auth subject the host names on the call — into the cache key, so a different
// principal computes a different key and can neither be served nor fill the other's
// entry. It is folded into the HASH COMPONENT (not added as a new key segment), so the
// tool:hash:epoch grammar every downstream parser relies on (cachemeta.FromVDSOKey) is
// unchanged AND the raw principal id never lands in an observability record.
//
// Two deliberate escape hatches keep the existing wins intact:
//
//   - EMPTY principal (the default — a single-tenant gateway, or any caller that names
//     no principal) leaves the hash untouched, so the key is BYTE-IDENTICAL to the v0.1
//     key: every caller still shares, and not one existing entry or test regresses. You
//     cannot isolate callers you cannot tell apart, so isolation engages only once the
//     host TELLS the vDSO who the principal is.
//   - A tool declared Shareable (RegisterShareable) is identity-INDEPENDENT public
//     knowledge: its principal is dropped, so it is shared ACROSS principals on purpose.
//     Cross-tenant sharing becomes a DECLARED capability, not an accident.
//
// So per-principal isolation is the default for a NAMED principal, and cross-tenant
// sharing is opt-in per tool. The gateway lowers the principal from an X-Fak-Principal
// header / request field onto ToolCall.Meta[MetaPrincipal]; the same ToolCall flows to
// both Lookup and the Emit fill, so the two keys are computed identically and a fill
// under principal A is only ever reachable by a later Lookup under principal A.

import (
	"crypto/sha256"
	"encoding/hex"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// MetaPrincipal is the OPTIONAL, OPEN ToolCall.Meta key naming the isolation principal
// (tenant / user / auth subject) a call is made on behalf of. Set it to scope this
// call's tier-2 cache entries to that principal; leave it unset for single-tenant
// sharing. It joins the MetaAgentID/MetaTurn/MetaSession family of open identity keys,
// but unlike those (which only ATTRIBUTE a hit) this one KEYS the entry.
const MetaPrincipal = "principal"

// principalOf returns the isolation principal a call carries, or "" when it names none.
func principalOf(c *abi.ToolCall) string {
	if c == nil || c.Meta == nil {
		return ""
	}
	return c.Meta[MetaPrincipal]
}

// scopeHash folds a principal into an args-hash component, yielding a fixed-width hex
// digest distinct per (principal, args) yet structurally identical to argHash — so the
// tool:hash:epoch key grammar is preserved and a different principal can never collide
// onto the same entry. The NUL separator cannot appear in the (hex) base, so the
// (principal, base) -> input mapping is injective and two distinct pairs never alias
// pre-hash. Empty principal callers never reach here (keyLocked skips the scoping).
func scopeHash(principal, base string) string {
	sum := sha256.Sum256([]byte(principal + "\x00" + base))
	return hex.EncodeToString(sum[:])[:24]
}

// RegisterShareable marks a tool as identity-independent PUBLIC knowledge: its tier-2
// entries drop the principal dimension, so a warmed read is shared ACROSS principals
// (the deliberate cross-tenant cache win — e.g. a shared policy or reference doc). The
// default for an unregistered tool is per-principal isolation. Idempotent and safe to
// call from a host's init/registration; written under v.mu like RegisterPure/Static.
func (v *VDSO) RegisterShareable(tool string) {
	v.mu.Lock()
	if v.shareable == nil {
		v.shareable = map[string]bool{}
	}
	v.shareable[tool] = true
	v.mu.Unlock()
}

// Shareable reports whether a tool has been declared cross-principal shareable.
func (v *VDSO) Shareable(tool string) bool {
	v.mu.Lock()
	ok := v.shareable[tool]
	v.mu.Unlock()
	return ok
}
