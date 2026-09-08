package main

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/session"
)

type guardDeadlineConfig struct {
	MaxDuration       time.Duration
	SoftDeadlineLead  time.Duration
	CommitGracePeriod time.Duration
	ChildStopGrace    time.Duration
}

func normalizeGuardDeadlineSettings(maxDuration, softLead, commitGrace, stopGrace time.Duration) guardDeadlineConfig {
	if stopGrace < 1*time.Second {
		stopGrace = 3 * time.Second
	}
	if commitGrace < 0 {
		commitGrace = 0
	}
	if maxDuration <= 0 {
		return guardDeadlineConfig{
			MaxDuration:       0,
			SoftDeadlineLead:  0,
			CommitGracePeriod: 0,
			ChildStopGrace:    stopGrace,
		}
	}
	if softLead < 0 {
		softLead = 0
	} else if softLead >= maxDuration {
		softLead = maxDuration / 2
	}
	return guardDeadlineConfig{
		MaxDuration:       maxDuration,
		SoftDeadlineLead:  softLead,
		CommitGracePeriod: commitGrace,
		ChildStopGrace:    stopGrace,
	}
}

func guardCheckSoftDeadline(sessions *session.Table, traceID string, lead time.Duration, now time.Time) (warn bool, remaining time.Duration) {
	if sessions == nil || strings.TrimSpace(traceID) == "" || lead <= 0 {
		return false, 0
	}
	v := sessions.QueryTimeBudget(traceID, now)
	if v.Bounded && !v.Exceeded && v.Remaining > 0 && v.Remaining <= lead {
		return true, v.Remaining
	}
	return false, 0
}

func guardEmitSoftDeadlineWarning(w1, w2 io.Writer, auditJournal *journal.Journal, srv *gateway.Server, traceID string, remaining, limit time.Duration) {
	msg := fmt.Sprintf("fak guard: WARNING — soft deadline reached (%s remaining of %s max-duration); please flush and commit all in-flight work\n", remaining.Round(time.Second), limit.Round(time.Second))
	if w1 != nil {
		fmt.Fprint(w1, msg)
	}
	if w2 != nil && w2 != w1 {
		fmt.Fprint(w2, msg)
	}
	if auditJournal != nil {
		auditJournal.AppendAgentEvent("SOFT_DEADLINE_WARNING", traceID, fmt.Sprintf("remaining=%s limit=%s", remaining.Round(time.Second), limit.Round(time.Second)))
	}
}
