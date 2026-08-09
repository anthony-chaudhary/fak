package gitdaily

import (
	"errors"
	"strings"
)

// RefusalCode is the stable operator-facing reason a daily tick did not complete.
type RefusalCode string

const (
	RefusalInvalidOptions RefusalCode = "INVALID_OPTIONS"
	RefusalTickBusy       RefusalCode = "TICK_BUSY"
	RefusalTickLock       RefusalCode = "TICK_LOCK_ERROR"
	RefusalLockCleanup    RefusalCode = "LOCK_CLEANUP_FAILED"
	RefusalMaintenance    RefusalCode = "MAINTENANCE_INCIDENT"
	RefusalLedgerWrite    RefusalCode = "LEDGER_WRITE_FAILED"
)

type Refusal struct {
	Code    RefusalCode `json:"code"`
	Message string      `json:"message"`
	Retry   bool        `json:"retry"`
}

// Refusal translates structural outcomes into concise operator guidance.
func (r Result) Refusal() *Refusal {
	switch {
	case r.ConfigErr != "":
		return &Refusal{Code: RefusalInvalidOptions, Message: "git-daily is missing its repository identity; resolve the repository root and Git common directory before retrying"}
	case r.TickLockErr != "":
		return &Refusal{Code: RefusalTickLock, Message: "git-daily could not open its serializer; check permissions on the Git common directory, then retry", Retry: true}
	case r.Skipped == SkipTickBusy:
		return &Refusal{Code: RefusalTickBusy, Message: "git-daily is already running for this clone; wait for the active tick, then read fak git-daily status", Retry: true}
	case r.Locks.Failed():
		return &Refusal{Code: RefusalLockCleanup, Message: "git-daily left an evidence-gated lock unresolved; inspect the lock sweep in fak git-daily --json before retrying"}
	case r.Maint.Incident:
		return &Refusal{Code: RefusalMaintenance, Message: "git-daily stopped at a maintenance safety gate; inspect the maintenance refusal in fak git-daily --json before retrying"}
	case r.LedgerErr != "":
		return &Refusal{Code: RefusalLedgerWrite, Message: "git-daily finished maintenance but could not record its witness; repair the ledger path before the next tick"}
	default:
		return nil
	}
}

func (o Options) Validate() error {
	var missing []string
	if strings.TrimSpace(o.RepoRoot) == "" {
		missing = append(missing, "RepoRoot")
	}
	if strings.TrimSpace(o.GitCommonDir) == "" {
		missing = append(missing, "GitCommonDir")
	}
	if len(missing) > 0 {
		return errors.New("gitdaily: missing " + strings.Join(missing, " and "))
	}
	return nil
}
