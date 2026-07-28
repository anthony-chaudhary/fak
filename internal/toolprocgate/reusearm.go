package toolprocgate

// reusearm.go — the ARMING rung for internal/toolproc's reuse verdict
// vocabulary (#5407, the rung repeatreason.go defers to; parent #5119).
//
// repeatreason.go allocates the six REUSE_* codes at 1090–1095 and ships them
// as pure data with no consumer, on the argument that allocating a vocabulary
// and wiring a live seam are different risks. This file is the consumer half:
// it REGISTERS the pairs and it CITES the code on a repeat it actually serves.
//
// WHY HERE. toolprocgate is already the in-kernel consumer of internal/toolproc
// (it owns the process-table vocabulary's name registration, see
// toolprocgate.go's init), and it is the leaf an embedder drives on the live
// wire. The reuse pairs register in this file's own init: one site, run once by
// construction, so the "registered twice under different names" failure the
// registry cannot undo is unreachable. The offline `fak toolproc repeats` fold
// keeps registering nothing, exactly as repeatreason.go requires.
//
// FAIL-CLOSED AT THE MAPPING SITE. ReuseVerdictFor is the single place a leaf
// token becomes a registered code. A Receipt.Reason outside toolproc's closed
// set — an empty reason from a zero-valued receipt that never ran the fold, or
// an invented token — renders NOTHING: Named stays false, no code is carried,
// and Serve refuses to answer the call locally even if the armed cache had the
// bytes on hand. A verdict is never fabricated for a decision that was never
// made.
//
// Tier: integrator (4), unchanged — this file imports abi (0) and toolproc (2).

import (
	"context"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// The result-metadata keys a locally answered repeat carries. They sit beside
// kill_reason on abi.Result.Meta — the same surface an operator already reads
// for a quarantined completion — so a served repeat is distinguishable from a
// fresh fetch without a second reporting channel.
const (
	// ReuseReasonMetaKey carries the stable REUSE_* token of the verdict that
	// blessed the local answer (never the REASON_<n> fallback: an unregistered
	// code never reaches a Result, because an unnamed verdict never serves).
	ReuseReasonMetaKey = "reuse_reason"
	// ReuseSourceMetaKey carries the receipt's provenance (immutable_reuse /
	// freshness_window) so a stale-window answer is not read as a live one.
	ReuseSourceMetaKey = "reuse_source"
	// ReuseServedMetaValue is the toolprocgate marker a locally answered repeat
	// stamps, the reuse twin of the "quarantined" marker the rank-2 Gate sets.
	ReuseServedMetaValue = "reuse_served"
)

// init registers the reuse verdict vocabulary. Separate from toolprocgate.go's
// init on purpose: the process-table pairs are registered by every consumer
// including the offline CLI, while these are only meaningful where a repeat is
// actually served. Keeping the arming in one file also keeps the negative
// witness a single-file swap.
func init() {
	for _, pr := range toolproc.ReuseReasonPairs() {
		abi.RegisterReason(pr.Code, pr.Name)
	}
}

// ReuseVerdict is one reuse decision rendered in the kernel's vocabulary.
// Receipt is the leaf's own evidence (identity, provenance, saved bytes — never
// a body); Reason is the registered code, valid ONLY when Named is true; Result
// is non-nil only when the repeat was actually answered locally.
type ReuseVerdict struct {
	Receipt toolproc.Receipt
	Reason  abi.ReasonCode
	Named   bool
	Result  *abi.Result
}

// Name renders the stable verdict token, or "" when the receipt cited no
// registered verdict. It never returns the REASON_<n> forward-compat spelling
// for an unmapped receipt: an unmapped receipt has no code to render at all.
func (v ReuseVerdict) Name() string {
	if !v.Named {
		return ""
	}
	return abi.ReasonName(v.Reason)
}

// Served reports the repeat was answered from the armed store, so the caller
// may skip the real fetch. A named verdict with Served false is the ordinary
// miss: the verdict says why, and the call proceeds.
func (v ReuseVerdict) Served() bool { return v.Result != nil }

// ReuseVerdictFor maps one decision Receipt onto its registered verdict. This
// is the consumer's whole half of repeatreason.go's bridge, and it is
// fail-closed: a reason outside the closed set yields Named=false with no code,
// and callers must render nothing rather than invent one.
func ReuseVerdictFor(r toolproc.Receipt) ReuseVerdict {
	v := ReuseVerdict{Receipt: r}
	code, ok := toolproc.ReuseReasonCode(r.Reason)
	if !ok {
		return v
	}
	v.Reason, v.Named = code, true
	return v
}

// ReuseArm is the live side of the armed reuse store, the counterpart to
// Supervisor for the reuse seam: an embedder (the gateway proxy, `fak guard`,
// the agent loop) hands it each normalized tool call before dispatch and
// deposits the payload of every real fetch. It serializes admission with its
// own lock, because toolproc.ArmedCache is deliberately not safe for concurrent
// use. Build one with NewReuseArm; the zero value has no store.
type ReuseArm struct {
	mu    sync.Mutex
	armed *toolproc.ArmedCache
}

// NewReuseArm builds an arm over a fresh armed store. A zero ArmedConfig takes
// the leaf's defaults (8 MiB budget, 1 MiB per entry, query coalescing off).
func NewReuseArm(cfg toolproc.ArmedConfig) *ReuseArm {
	return &ReuseArm{armed: toolproc.NewArmedCache(cfg)}
}

// Serve runs the reuse decision for one live tool call and, when the store
// holds the bytes for a verdict it can name, returns the answer as a Result
// stamped with the verdict's stable token. A miss (or a hit whose bytes are
// absent) returns a named verdict with a nil Result: the caller runs the real
// call and deposits the payload through Offer so the NEXT repeat serves.
//
// c may be nil (it is only carried back on the Result for the caller's own
// correlation); ctx is used to store the served payload through the active
// resolver, falling back to an inline ref exactly as the rank-2 Gate does.
func (a *ReuseArm) Serve(ctx context.Context, c *abi.ToolCall, rec toolproc.CallRecord) ReuseVerdict {
	a.mu.Lock()
	out := a.armed.Admit(rec)
	a.mu.Unlock()

	v := ReuseVerdictFor(out.Receipt)
	if !v.Named || !out.BodyServed {
		// Unnamed is the fail-closed branch: no registered verdict means no
		// verdict to cite, and bytes that no verdict blessed are not served.
		return v
	}
	v.Result = &abi.Result{
		Call:    c,
		Payload: putBody(ctx, out.Body),
		Status:  abi.StatusOK,
		Meta: map[string]string{
			"toolprocgate":     ReuseServedMetaValue,
			ReuseReasonMetaKey: v.Name(),
			ReuseSourceMetaKey: string(out.Receipt.Source),
		},
	}
	return v
}

// Offer deposits the payload of a REAL fetch under the identity the decision
// blesses, so a later repeat can be answered locally. The store's own
// fail-closed admission rules apply (writes and unknowns refused, mutable
// queries only when coalescing is opted in, over-cap payloads not retained);
// the bool reports whether the body was retained.
func (a *ReuseArm) Offer(rec toolproc.CallRecord, body []byte) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.armed.Offer(rec, body)
}

// putBody stores a served payload through the active resolver when one is
// wired, falling back to an inline ref. The default Taint/Scope (tainted,
// agent-private) is intentional: a repeat carries exactly the trust its
// original fetch did, and reuse never widens it.
func putBody(ctx context.Context, b []byte) abi.Ref {
	if res := abi.ActiveResolver(); res != nil {
		if ref, err := res.Put(ctx, b); err == nil {
			return ref
		}
	}
	return abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b))}
}
