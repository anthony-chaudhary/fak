package selfupdate

import (
	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

// CheckStatus is the stable receipt-facing posture of self-update --check.
type CheckStatus string

const (
	StatusCurrent   CheckStatus = "current"
	StatusStale     CheckStatus = "stale"
	StatusDivergent CheckStatus = "divergent"
	// StatusAttention means only audit-only roles are unsafe. The updater must keep that state
	// visible, but cannot honestly advertise an automatic repair for a role it never swaps.
	StatusAttention CheckStatus = "attention"
)

// CheckPosture is the automation decision emitted by self-update --check receipts.
type CheckPosture struct {
	Status      CheckStatus
	NextCommand string
}

// ClassifyCheck maps target freshness and the deployed-copy audit to one actionable posture.
// Revision staleness wins because a normal update also repairs convergeable sibling copies.
func ClassifyCheck(freshness binstamp.Freshness, audit selfinstall.AuditPartition) CheckPosture {
	switch {
	case freshness != binstamp.Fresh:
		return CheckPosture{Status: StatusStale, NextCommand: "fak self-update"}
	case audit.Convergeable.Present():
		return CheckPosture{Status: StatusDivergent, NextCommand: "fak self-update"}
	case audit.AuditOnly.Present():
		return CheckPosture{Status: StatusAttention, NextCommand: "fak self-update --check"}
	default:
		return CheckPosture{Status: StatusCurrent, NextCommand: "fak version"}
	}
}

// InstallPosture classifies the post-activation audit. Completed says every role the updater is
// allowed to repair is current. AuditOnlyAttention separately preserves unsafe manual state.
type InstallPosture struct {
	Completed          bool
	AuditOnlyAttention bool
}

// ClassifyInstall prevents an audit-only role from relabeling a successful automatic
// convergence as a failed update while keeping strict admission evidence available in Audit.
func ClassifyInstall(audit selfinstall.AuditPartition) InstallPosture {
	return InstallPosture{
		Completed:          !audit.Convergeable.Present(),
		AuditOnlyAttention: audit.AuditOnly.Present(),
	}
}
