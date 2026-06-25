package adjudicator

// promote.go — the promotion ledger (epic #669): a thread-safe per-tool counter folded
// from the kernel's live adjudication stream. It mirrors rulesynth's near-miss
// harvester (internal/rulesynth/stream.go: a sync.Mutex append-log fed by an abi.Emitter
// keyed on EvDecide), but counts COMPLAIN-MODE ADMITS — the would-have-denied calls a
// Policy.Complain trial admitted-and-logged (#670/#671) — so an operator can see which
// complained tool has earned promotion to the Allow list.
//
// It changes no verdict and lands nothing: like the harvester it is OPT-IN (attach it
// via abi.RegisterEmitter), it only OBSERVES, and its sole output is a per-tool count an
// operator reviews. A hard-refusal would-deny for a tool RESETS its clean run to zero: a
// tool that ever provokes a self-modify / policy / exfil refusal is not a candidate.

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Ledger is the thread-safe per-tool promotion counter. It implements abi.Emitter:
// attach it with abi.RegisterEmitter and it folds every decided call. The zero value is
// a ready empty ledger (its maps are lazily allocated under the mutex).
type Ledger struct {
	mu    sync.Mutex
	clean map[string]int // tool -> consecutive clean complain-mode admits
	hard  map[string]int // tool -> hard-refusal would-deny count (a single one disqualifies)
}

// NewLedger returns an empty promotion ledger.
func NewLedger() *Ledger { return &Ledger{} }

// Emit folds one adjudication event into the ledger. It keys ONLY on EvDecide (the
// verdict-resolved event, emitted exactly once per decided call) so a later
// EvDispatch/EvComplete for the same call cannot double-count it — the same discipline
// rulesynth.Harvester.Emit uses. A complain-mode admit (an Allow carrying
// posture=admit_and_log) increments the tool's clean run; a hard-refusal Deny resets
// that run to zero and records the hard event.
func (l *Ledger) Emit(ev abi.Event) {
	if ev.Kind != abi.EvDecide || ev.Verdict == nil || ev.Call == nil {
		return
	}
	tool := ev.Call.Tool
	if tool == "" {
		return
	}
	v := ev.Verdict
	switch {
	case v.Kind == abi.VerdictAllow && v.Meta["posture"] == "admit_and_log":
		l.mu.Lock()
		if l.clean == nil {
			l.clean = make(map[string]int)
		}
		l.clean[tool]++
		l.mu.Unlock()
	case v.Kind == abi.VerdictDeny && hardRefusal(v.Reason):
		l.mu.Lock()
		if l.hard == nil {
			l.hard = make(map[string]int)
		}
		l.hard[tool]++
		delete(l.clean, tool) // a hard refusal resets the clean run to zero
		l.mu.Unlock()
	}
}

// hardRefusal reports whether a refusal reason is a HARD-refusal class: a provable
// policy/security refusal that disqualifies a tool from promotion. It deliberately
// EXCLUDES the soft DEFAULT_DENY (the reason complain mode admits-and-logs), the
// model-fixable MISROUTE, and the transient RATE_LIMITED / OVERSIZE / LEASE_HELD /
// UNKNOWN_TOOL — none of which proves the tool is unsafe to promote.
func hardRefusal(r abi.ReasonCode) bool {
	switch r {
	case abi.ReasonPolicyBlock, abi.ReasonSelfModify, abi.ReasonSecretExfil,
		abi.ReasonTrustViolation, abi.ReasonMalformed, abi.ReasonUnwitnessed:
		return true
	}
	return false
}

// Clean returns a tool's current clean complain-mode admit count.
func (l *Ledger) Clean(tool string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.clean[tool]
}

// HardEvents returns a tool's recorded hard-refusal count.
func (l *Ledger) HardEvents(tool string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.hard[tool]
}
