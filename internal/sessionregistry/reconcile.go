package sessionregistry

import (
	"fmt"
	"sort"
	"time"
)

// Reconciliation is a bounded classification for a persisted registration that
// no longer has a matching independently observed process identity.
type Reconciliation struct {
	RegistrationID string `json:"registration_id"`
	From           State  `json:"from"`
	To             State  `json:"to"`
	Reason         string `json:"reason"`
}

// ReconcileStale classifies stale registered/active rows using PID plus process
// start identity. It never treats PID presence alone as liveness and does not
// mutate the supplied rows or their store.
func ReconcileStale(rows []Record, observed []ObservedProcess, now time.Time, staleAfter time.Duration) []Reconciliation {
	if staleAfter <= 0 {
		staleAfter = 10 * time.Minute
	}
	live := make(map[string]bool, len(observed))
	for _, p := range observed {
		if p.PID > 0 && !p.ProcessStartedAt.IsZero() {
			live[processIdentityKey(p.PID, p.ProcessStartedAt)] = true
		}
	}
	var out []Reconciliation
	for _, r := range rows {
		if r.State != StateRegistered && r.State != StateActive {
			continue
		}
		last := r.HeartbeatAt
		if last.IsZero() {
			last = r.StartedAt
		}
		if last.IsZero() {
			last = r.CreatedAt
		}
		if last.IsZero() || now.UTC().Sub(last.UTC()) <= staleAfter {
			continue
		}
		if r.Identity.PID > 0 && !r.Identity.ProcessStartedAt.IsZero() {
			if live[processIdentityKey(r.Identity.PID, r.Identity.ProcessStartedAt)] {
				continue
			}
			out = append(out, Reconciliation{RegistrationID: r.RegistrationID, From: r.State, To: StateLost, Reason: "stale registration has no matching pid plus process_start identity"})
			continue
		}
		out = append(out, Reconciliation{RegistrationID: r.RegistrationID, From: r.State, To: StateUnknown, Reason: "stale registration lacks process identity for liveness read-back"})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].RegistrationID < out[j].RegistrationID })
	return out
}

func processIdentityKey(pid int, started time.Time) string {
	return fmt.Sprintf("%d@%s", pid, started.UTC().Format(time.RFC3339Nano))
}
