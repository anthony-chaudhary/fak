package main

// dispatch_worker_heartbeat.go bridges dispatchaudit's structural liveness evidence
// (a live PID or real log bytes — never a worker's self-report) into a loop-ledger
// heartbeat event, so the ledger can distinguish a worker that reached its prompt
// and actually STARTED from one that NEVER did (issue #1782). It is opt-in
// (`fak dispatch audit --heartbeat`) so a routine read-only audit never writes to
// the shared hot ledger file.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// dispatchWorkerHeartbeatLoopID is the shared loop identity every worker-heartbeat
// event rides under, keyed by backend so `fak loop health` can show one lane per
// backend rather than one per worker.
func dispatchWorkerHeartbeatLoopID(c dispatchaudit.Classification) string {
	return "dispatch-worker-heartbeat/" + string(c.Backend)
}

// dispatchWorkerHeartbeatRunID derives a stable per-worker run id. The log base name
// is already the stable per-worker anchor dispatchaudit fingerprints against, so
// reuse it rather than minting a new identifier.
func dispatchWorkerHeartbeatRunID(c dispatchaudit.Classification) string {
	if c.Log != "" {
		return "resolve-" + c.Log
	}
	return fmt.Sprintf("resolve-issue-%s-%s", firstString(c.Issue, "unknown"), string(c.Backend))
}

// workerHeartbeatEvent builds the STARTED/NEVER_STARTED heartbeat for one classified
// worker. STARTED covers every outcome that required a live process or actual log
// bytes to reach (Classification.Started()); NEVER_STARTED is the sole outcome
// nothing proves ever ran.
func workerHeartbeatEvent(c dispatchaudit.Classification) loopmgr.Event {
	status := loopmgr.StatusRunning
	reason := "STARTED"
	if !c.Started() {
		status = loopmgr.StatusFailed
		reason = "NEVER_STARTED"
	}
	ev := loopmgr.Event{
		LoopID:    dispatchWorkerHeartbeatLoopID(c),
		RunID:     dispatchWorkerHeartbeatRunID(c),
		Kind:      loopmgr.EventHeartbeat,
		Source:    "fak dispatch audit",
		Principal: string(c.Backend),
		Status:    status,
		Reason:    reason,
		Summary:   truncateString(c.Reason, 200),
	}
	if c.Log != "" {
		ev.EvidenceRefs = append(ev.EvidenceRefs, loopmgr.EvidenceRef{Kind: "log", Ref: c.Log})
	}
	if c.Issue != "" {
		ev.EvidenceRefs = append(ev.EvidenceRefs, loopmgr.EvidenceRef{Kind: "issue", Ref: c.Issue})
	}
	return ev
}

// appendWorkerHeartbeats appends one heartbeat event per classification to the loop
// ledger at path, in order, returning the count successfully appended. It stops and
// returns the first error — a straight loop over the already lock-serialized
// loopmgr.Append, so a mid-run failure leaves a partial-but-consistent ledger tail.
func appendWorkerHeartbeats(path string, classifications []dispatchaudit.Classification) (int, error) {
	n := 0
	for _, c := range classifications {
		if _, err := loopmgr.Append(path, workerHeartbeatEvent(c)); err != nil {
			return n, fmt.Errorf("append heartbeat for %s: %w", c.Log, err)
		}
		n++
	}
	return n, nil
}
