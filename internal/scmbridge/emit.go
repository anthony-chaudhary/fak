package scmbridge

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/servicelease"
	"github.com/anthony-chaudhary/fak/internal/serviceledger"
	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// Ledger emission (#4756): every bridge action lands in the portable
// fak.service.events.v1 ledger carrying the identifiers the issue names —
// native event IDs, service exit codes, task instance IDs, request/receipt
// IDs, and the resumed session identity. These constructors return validated
// rows; the caller appends them (serviceledger dedupes on source_uid).

// tokenDigestInput renders the fencing token for HashLeaseToken — the raw
// token never reaches the ledger, only its digest.
func tokenDigestInput(t servicelease.FencingToken) string {
	return fmt.Sprintf("g%d.s%d", t.Generation, t.LeaseSeq)
}

// ReceiptFor derives the durable receipt ID that closes a launch request.
func ReceiptFor(requestID string) string {
	return "receipt-" + strings.TrimPrefix(requestID, "launch-")
}

// LaunchEvent records an admitted launch: the manager (re)started the
// workload under one fenced grant. Request ID and token digest bind the row
// to the ONE generation/lease that made the launch valid.
func LaunchEvent(id servicespec.Identity, g Grant, pid int, atMS int64) serviceledger.Event {
	return serviceledger.Event{
		Type:      serviceledger.EventManagerRestart,
		AtUnixMS:  atMS,
		Source:    serviceledger.SourceFak,
		SourceUID: g.RequestID,
		Identity:  id,
		Correlation: serviceledger.Correlation{
			Request:        g.RequestID,
			Generation:     int64(g.Token.Generation),
			LeaseTokenHash: serviceledger.HashLeaseToken(tokenDigestInput(g.Token)),
			PID:            pid,
		},
		Detail: fmt.Sprintf("bridge launch admitted role=%s", g.Role),
	}
}

// RefusalEvent records a refused duplicate launch as a lease-fence row: the
// fence held, so exactly one launcher owns the workload.
func RefusalEvent(id servicespec.Identity, workload string, refused Role, holder Role, atMS int64) serviceledger.Event {
	return serviceledger.Event{
		Type:      serviceledger.EventLeaseFence,
		AtUnixMS:  atMS,
		Source:    serviceledger.SourceFak,
		SourceUID: fmt.Sprintf("fence-%s-%s-%d", workload, refused, atMS),
		Identity:  id,
		Detail:    fmt.Sprintf("duplicate launch refused: %s held by %s", refused, holder),
	}
}

// StopEvent classifies native stop evidence and records it: the native event
// ID and provider ride in the detail and source UID, the service exit code in
// the exit record, the task instance ID and session identity in the
// correlation. It returns the classified cause alongside the row.
func StopEvent(id servicespec.Identity, r StopReport, atMS int64) (serviceledger.Event, StopCause) {
	cause, class := Classify(r)
	ev := serviceledger.Event{
		Type:      serviceledger.EventProcessExit,
		AtUnixMS:  atMS,
		Source:    serviceledger.SourceFak,
		SourceUID: fmt.Sprintf("stop-%s-%d-%d", id.Workload, r.NativeEventID, atMS),
		Identity:  id,
		Exit:      &servicespec.ExitRecord{Class: class, Code: r.ExitCode, AtUnixMS: atMS},
		Correlation: serviceledger.Correlation{
			ManagerInvocation: r.TaskInstance,
			Session:           r.Session,
		},
		Detail: fmt.Sprintf("native event %d (%s): cause=%s exit_code=%d", r.NativeEventID, r.Provider, cause, r.ExitCode),
	}
	return ev, cause
}

// ResumeEvent records that the workload re-entered its logical work: the
// receipt closes the launch request, and the session identity proves WHICH
// interactive session the broker resumed into.
func ResumeEvent(id servicespec.Identity, receiptID, session string, atMS int64) serviceledger.Event {
	return serviceledger.Event{
		Type:      serviceledger.EventResume,
		AtUnixMS:  atMS,
		Source:    serviceledger.SourceFak,
		SourceUID: receiptID,
		Identity:  id,
		Correlation: serviceledger.Correlation{
			Receipt: receiptID,
			Session: session,
		},
		Detail: "bridge resume",
	}
}
