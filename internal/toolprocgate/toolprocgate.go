package toolprocgate

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// AdmitRank places the revocation check in FRONT of the content screens
// (secretgate rank 4, normgate rank 5, ctxmmu rank 10, ifc rank 20): a revoked
// call's raw payload is stubbed before any content gate runs, so the later
// rungs screen the benign stub — the original bytes of a dead call never even
// reach the detectors, let alone context.
const AdmitRank = 2

// maxKills bounds the revocation table; the oldest entries evict FIFO. A
// supervisor that kills more than this many calls in one process lifetime is
// itself the anomaly, but the gate must not become the memory leak.
const maxKills = 4096

type table struct {
	mu    sync.Mutex
	kills map[string]string // ToolCall.TraceID -> closed reason token for the kill
	order []string          // FIFO eviction order
	head  int
}

var tbl = &table{kills: map[string]string{}}

// Kill marks a call id revoked, citing a closed reason token (normally one of
// the toolproc verdict names, e.g. TOOL_DEADLINE_EXCEEDED, or an operator
// token). Idempotent: the first reason for a call id wins. This is the
// in-process control plane the supervisor seams call the moment they act on a
// toolproc `kill` advice; any completion for that id arriving afterwards is
// quarantined by the Gate below.
func Kill(callID, reason string) {
	if callID == "" {
		return
	}
	if reason == "" {
		reason = toolproc.ReasonToolResultAfterKillName
	}
	tbl.mu.Lock()
	defer tbl.mu.Unlock()
	if _, dup := tbl.kills[callID]; dup {
		return
	}
	tbl.kills[callID] = reason
	tbl.order = append(tbl.order, callID)
	for len(tbl.kills) > maxKills && tbl.head < len(tbl.order) {
		delete(tbl.kills, tbl.order[tbl.head])
		tbl.head++
	}
	if tbl.head > 0 && tbl.head*2 >= len(tbl.order) {
		n := copy(tbl.order, tbl.order[tbl.head:])
		tbl.order = tbl.order[:n]
		tbl.head = 0
	}
}

// KilledReason reports whether callID was revoked, and the reason token cited.
func KilledReason(callID string) (string, bool) {
	tbl.mu.Lock()
	defer tbl.mu.Unlock()
	r, ok := tbl.kills[callID]
	return r, ok
}

// Reset clears the revocation table (tests).
func Reset() {
	tbl.mu.Lock()
	defer tbl.mu.Unlock()
	tbl.kills = map[string]string{}
	tbl.order = nil
	tbl.head = 0
}

// Gate is the rank-2 ResultAdmitter: the write-time enforcement of toolproc's
// TOOL_RESULT_AFTER_KILL verdict. With an empty revocation table it Defers on
// every result — registered-but-inert, byte-identical to the pre-gate chain —
// so shipping it enabled changes nothing until something actually kills a call.
type Gate struct{}

// Caps implements abi.ResultAdmitter.
func (Gate) Caps() []abi.Capability { return nil }

// Admit quarantines a result whose call the kernel already revoked. The
// payload is replaced in place with a structured stub; the original bytes are
// DROPPED, not held for a gated page-in — unlike a quarantined secret (held
// for witnessed release), a post-kill payload has no legitimate re-entry path,
// so fail-closed here means fail-forgotten. Everything else Defers.
func (Gate) Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	if c == nil || r == nil || c.TraceID == "" {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "toolprocgate"}
	}
	killReason, killed := KilledReason(c.TraceID)
	if !killed {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "toolprocgate"}
	}

	stub := map[string]any{
		"_quarantined": true,
		"reason":       toolproc.ReasonToolResultAfterKillName,
		"kill_reason":  killReason,
		"by":           "toolprocgate",
		"len":          payloadLen(ctx, r.Payload),
		"_note":        "completion arrived after the kernel revoked this call; payload held out of context",
	}
	if ref, ok := putJSON(ctx, stub); ok {
		ref.Taint = abi.TaintQuarantined
		r.Payload = ref
	} else {
		r.Payload = abi.Ref{Kind: abi.RefInline, Taint: abi.TaintQuarantined}
	}
	if r.Meta == nil {
		r.Meta = map[string]string{}
	}
	r.Meta["toolprocgate"] = "quarantined"
	r.Meta["kill_reason"] = killReason

	return abi.Verdict{
		Kind:    abi.VerdictQuarantine,
		Payload: abi.QuarantinePayload{PageOut: true},
		Reason:  toolproc.ReasonToolResultAfterKill,
		By:      "toolprocgate",
		Meta:    map[string]string{"kill_reason": killReason, "call": c.TraceID},
	}
}

// payloadLen reports the byte length of the payload being held out, resolving
// a non-inline Ref when a resolver is wired (forensic size only, never bytes).
func payloadLen(ctx context.Context, ref abi.Ref) int {
	if ref.Kind == abi.RefInline {
		return len(ref.Inline)
	}
	if ref.Len > 0 {
		return int(ref.Len)
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, ref); err == nil {
			return len(b)
		}
	}
	return 0
}

// putJSON stores the stub through the active resolver when one is wired,
// falling back to an inline ref (the secretgate idiom).
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

// init registers the enforcement rung and the toolproc verdict vocabulary.
// toolprocgate is the in-kernel CONSUMER of internal/toolproc's out-of-tree
// reason codes, so it owns their name registration (the adjudicator-owns-
// egressfloor pattern); `fak toolproc` registers the same pairs for the
// offline CLI path — RegisterReason is an idempotent map write, same values.
func init() {
	for _, pr := range toolproc.ReasonPairs() {
		abi.RegisterReason(pr.Code, pr.Name)
	}
	abi.RegisterResultAdmitter(AdmitRank, Gate{})
}
