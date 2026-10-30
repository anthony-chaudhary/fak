package gateway

import (
	"context"
	"math"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/dosdecision"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

// maintenanceWorkspaceAdmission reads both canonical ownership planes at the
// host-selected grant root. It deliberately has no injectable ALLOW provider,
// caller-supplied workspace, or holder-text ownership exemption.
func maintenanceWorkspaceAdmission(ctx context.Context, grant remotemaint.Grant) (WireVerdict, bool) {
	if grant.Reservation == nil {
		return WireVerdict{}, false
	}
	root := grant.Reservation.WorkspaceRoot
	canonical, err := canonicalExistingWorkspacePath(root)
	if err != nil || canonical != filepath.Clean(root) {
		return maintenanceWorkspaceReadRefusal()
	}
	readCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	// The live-plane subprocess lives in dosdecision so this Adjudicator
	// package stays os/exec-free; any read fault maps to the same fail-closed
	// refusal.
	liveRows, err := dosdecision.ReadLiveRows(readCtx, canonical)
	if err != nil {
		return maintenanceWorkspaceReadRefusal()
	}
	leases := make([]adjudicateClassifiedLease, 0, len(liveRows))
	for _, row := range liveRows {
		trees := row.Tree
		if len(trees) == 0 {
			return maintenanceWorkspaceReadRefusal()
		}
		for _, tree := range trees {
			if strings.TrimSpace(tree) == "" {
				return maintenanceWorkspaceReadRefusal()
			}
		}
		leases = append(leases, adjudicateClassifiedLease{TreeGlobs: trees})
	}
	// StrictSnapshot rejects missing/corrupt blobs rather than dropping unknown
	// peers. Validate row semantics before expiry filtering: an invalid clock or
	// empty tree cannot disguise unknown ownership as an expired lease.
	gitLeases, err := leaseref.NewInDir(canonical).StrictSnapshot(readCtx)
	if err != nil {
		return maintenanceWorkspaceReadRefusal()
	}
	for _, row := range gitLeases {
		if row.Holder == "" || row.AcquiredAt <= 0 || row.RenewedAt < 0 ||
			(row.RenewedAt != 0 && row.RenewedAt < row.AcquiredAt) || row.TTLSeconds < 0 || len(row.TreeGlobs) == 0 {
			return maintenanceWorkspaceReadRefusal()
		}
		activeAt := max(row.AcquiredAt, row.RenewedAt)
		if row.TTLSeconds > math.MaxInt64-activeAt {
			// Expired adds activeAt+TTL. A corrupt clock must never wrap into
			// the past and silently remove an overlapping peer from admission.
			return maintenanceWorkspaceReadRefusal()
		}
		if row.Expired(time.Now()) {
			continue
		}
		leases = append(leases, adjudicateClassifiedLease{TreeGlobs: row.TreeGlobs, SessionID: row.SessionID})
	}
	targets, failure := adjudicateNormalizeWriteTargetsDetailed(nil, append([]string(nil), grant.Paths...), canonical)
	if failure != "" {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, failure, "maintenance write set cannot be canonicalized inside trusted workspace"), true
	}
	return adjudicateWorkspaceLeaseOverlap(canonical, targets, leases)
}

func maintenanceWorkspaceReadRefusal() (WireVerdict, bool) {
	return leaseAdmissionRefusal(abi.ReasonDefaultDeny, leaseFailureWorkspaceAuthorityRead, "canonical maintenance workspace authority read failed"), true
}
