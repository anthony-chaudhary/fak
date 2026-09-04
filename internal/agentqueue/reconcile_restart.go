package agentqueue

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

const RestartReconcileSchema = "fak.agentqueue.restart-reconcile.v1"

type AttemptAction string

const (
	AttemptActionAdopt   AttemptAction = "ADOPT"
	AttemptActionReplace AttemptAction = "REPLACE"
	AttemptActionHold    AttemptAction = "HOLD"

	ActionAdopt   = AttemptActionAdopt
	ActionReplace = AttemptActionReplace
	ActionHold    = AttemptActionHold
)

type AttemptDisposition struct {
	IntentID string        `json:"intent_id"`
	Action   AttemptAction `json:"action"`
	PID      int           `json:"pid"`
	Reason   string        `json:"reason"`
}

type RestartReconciliation struct {
	Schema     string               `json:"schema"`
	Generation string               `json:"generation"`
	Adopted    []AttemptDisposition `json:"adopted"`
	Replaced   []AttemptDisposition `json:"replaced"`
	Held       []AttemptDisposition `json:"held"`
}

type ProcessLivenessChecker func(pid int) bool

type RestartOptions struct {
	Now           time.Time                                  `json:"now,omitempty"`
	LeaseTimeout  time.Duration                              `json:"lease_timeout,omitempty"`
	IsExpired     func(intent Intent, attempt *Attempt) bool `json:"-"`
	Indeterminate func(intent Intent, attempt *Attempt) bool `json:"-"`
}

func ReconcileRestart(snapshot Snapshot, liveness ProcessLivenessChecker, opts RestartOptions) (RestartReconciliation, Snapshot, error) {
	if snapshot.Schema != "" && snapshot.Schema != Schema {
		return RestartReconciliation{}, snapshot, fmt.Errorf("unsupported schema %q", snapshot.Schema)
	}
	if snapshot.Generation == "" {
		return RestartReconciliation{}, snapshot, errors.New("generation is required")
	}
	if err := snapshot.Pool.Validate(); err != nil {
		return RestartReconciliation{}, snapshot, err
	}

	intentByID := make(map[string]*Intent, len(snapshot.Intents))
	for i := range snapshot.Intents {
		intent := &snapshot.Intents[i]
		if intent.ID == "" {
			return RestartReconciliation{}, snapshot, errors.New("intent id required")
		}
		if _, exists := intentByID[intent.ID]; exists {
			return RestartReconciliation{}, snapshot, fmt.Errorf("duplicate intent %q", intent.ID)
		}
		intentByID[intent.ID] = intent
	}

	attemptsByIntent := make(map[string][]*Attempt, len(snapshot.Attempts))
	for i := range snapshot.Attempts {
		att := &snapshot.Attempts[i]
		if _, ok := intentByID[att.IntentID]; !ok {
			return RestartReconciliation{}, snapshot, fmt.Errorf("attempt references unknown intent %q", att.IntentID)
		}
		attemptsByIntent[att.IntentID] = append(attemptsByIntent[att.IntentID], att)
	}

	now := opts.Now
	if now.IsZero() {
		now = time.Now()
	}

	if liveness == nil {
		liveness = processalive.Check
	}

	rec := RestartReconciliation{
		Schema:   RestartReconcileSchema,
		Adopted:  []AttemptDisposition{},
		Replaced: []AttemptDisposition{},
		Held:     []AttemptDisposition{},
	}

	for i := range snapshot.Intents {
		intent := &snapshot.Intents[i]
		attempts := attemptsByIntent[intent.ID]

		var activeAttempts []*Attempt
		for _, att := range attempts {
			if att.State == AttemptReserved || att.State == AttemptRunning {
				activeAttempts = append(activeAttempts, att)
			}
		}

		if intent.State != IntentRunning && len(activeAttempts) == 0 {
			continue
		}

		var latestActive *Attempt
		if len(activeAttempts) > 0 {
			latestActive = activeAttempts[len(activeAttempts)-1]
		}

		pid := 0
		if latestActive != nil && latestActive.PID > 0 {
			pid = latestActive.PID
		} else if intent.PID > 0 {
			pid = intent.PID
		}

		isExpired := false
		isIndeterminate := false
		var holdReason string

		if opts.IsExpired != nil && opts.IsExpired(*intent, latestActive) {
			isExpired = true
			holdReason = "lease expired"
		} else if !intent.ExpiresAt.IsZero() && !now.Before(intent.ExpiresAt) {
			isExpired = true
			holdReason = "lease expired"
		} else if !intent.LeaseExpires.IsZero() && !now.Before(intent.LeaseExpires) {
			isExpired = true
			holdReason = "lease expired"
		} else if latestActive != nil && !latestActive.ExpiresAt.IsZero() && !now.Before(latestActive.ExpiresAt) {
			isExpired = true
			holdReason = "lease expired"
		} else if latestActive != nil && !latestActive.LeaseExpires.IsZero() && !now.Before(latestActive.LeaseExpires) {
			isExpired = true
			holdReason = "lease expired"
		}

		if !isExpired {
			if opts.Indeterminate != nil && opts.Indeterminate(*intent, latestActive) {
				isIndeterminate = true
				holdReason = "indeterminate state"
			} else if intent.State == IntentHeld {
				isIndeterminate = true
				holdReason = "intent held"
			} else if latestActive != nil && (latestActive.State == AttemptState("held") || latestActive.State == AttemptState("indeterminate")) {
				isIndeterminate = true
				holdReason = "indeterminate attempt state"
			}
		}

		if isExpired || isIndeterminate {
			intent.State = IntentHeld
			for _, att := range activeAttempts {
				att.State = AttemptFailed
			}
			rec.Held = append(rec.Held, AttemptDisposition{
				IntentID: intent.ID,
				Action:   AttemptActionHold,
				PID:      pid,
				Reason:   holdReason,
			})
			continue
		}

		if pid > 0 && liveness(pid) {
			intent.State = IntentRunning
			for _, att := range activeAttempts {
				att.State = AttemptRunning
				if att.PID == 0 {
					att.PID = pid
				}
			}
			rec.Adopted = append(rec.Adopted, AttemptDisposition{
				IntentID: intent.ID,
				Action:   AttemptActionAdopt,
				PID:      pid,
				Reason:   "process live",
			})
		} else {
			intent.State = IntentQueued
			for _, att := range activeAttempts {
				att.State = AttemptFailed
			}
			reason := "process dead"
			if pid <= 0 {
				reason = "no process pid"
			}
			rec.Replaced = append(rec.Replaced, AttemptDisposition{
				IntentID: intent.ID,
				Action:   AttemptActionReplace,
				PID:      pid,
				Reason:   reason,
			})
		}
	}

	sort.Slice(rec.Adopted, func(i, j int) bool { return rec.Adopted[i].IntentID < rec.Adopted[j].IntentID })
	sort.Slice(rec.Replaced, func(i, j int) bool { return rec.Replaced[i].IntentID < rec.Replaced[j].IntentID })
	sort.Slice(rec.Held, func(i, j int) bool { return rec.Held[i].IntentID < rec.Held[j].IntentID })

	newGen := nextRestartGeneration(snapshot.Generation, rec)
	snapshot.Generation = newGen
	rec.Generation = newGen
	if snapshot.Schema == "" {
		snapshot.Schema = Schema
	}

	return rec, snapshot, nil
}

func nextRestartGeneration(current string, rec RestartReconciliation) string {
	h := sha256.New()
	_, _ = h.Write([]byte("restart\x00"))
	_, _ = h.Write([]byte(current))
	for _, a := range rec.Adopted {
		_, _ = h.Write([]byte("\x00adopt:" + a.IntentID + ":" + strconv.Itoa(a.PID)))
	}
	for _, r := range rec.Replaced {
		_, _ = h.Write([]byte("\x00replace:" + r.IntentID + ":" + strconv.Itoa(r.PID)))
	}
	for _, hld := range rec.Held {
		_, _ = h.Write([]byte("\x00hold:" + hld.IntentID + ":" + strconv.Itoa(hld.PID)))
	}
	return "gen:" + hex.EncodeToString(h.Sum(nil)[:16])
}
