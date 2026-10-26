package gateway

import "github.com/anthony-chaudhary/fak/internal/abi"

// leaseAdmissionFailure is the closed machine-readable classification for the
// lease-admission boundary. It is selected at the decision site; diagnostics
// never infer a failure class from the human claim.
type leaseAdmissionFailure string

const (
	leaseFailureUnknownFootprint              leaseAdmissionFailure = "unknown_write_footprint"
	leaseFailureOutsideWorkspace              leaseAdmissionFailure = "outside_trusted_workspace"
	leaseFailureWriteTargetUnresolvable       leaseAdmissionFailure = "write_target_unresolvable"
	leaseFailureWorkspaceAuthorityUnavailable leaseAdmissionFailure = "workspace_authority_unavailable"
	leaseFailureWorkspaceAuthorityRead        leaseAdmissionFailure = "workspace_authority_read_failed"
	leaseFailureInvalidWorkspaceRoot          leaseAdmissionFailure = "invalid_workspace_root"
	leaseFailureWorkspaceRootCanonicalization leaseAdmissionFailure = "workspace_root_canonicalization_failed"
	leaseFailureLeaseAuthorityUnavailable     leaseAdmissionFailure = "lease_authority_unavailable"
	leaseFailureLeaseAuthorityRead            leaseAdmissionFailure = "lease_authority_read_failed"
	leaseFailureInvalidLeaseView              leaseAdmissionFailure = "invalid_lease_view"
	leaseFailureInvalidWorkspaceLeaseView     leaseAdmissionFailure = "invalid_workspace_lease_view"
	leaseFailureInvalidLeaseTree              leaseAdmissionFailure = "invalid_lease_tree"
	leaseFailurePeerHeld                      leaseAdmissionFailure = "peer_lease_held"
	leaseFailureAtomicGuardUnavailable        leaseAdmissionFailure = "atomic_guard_unavailable"
)

const (
	leaseRecoveryUseFileTools       = "USE_FILE_TOOLS"
	leaseRecoveryCheckWorkspace     = "CHECK_WORKSPACE_BINDING"
	leaseRecoveryCheckWriteTarget   = "CHECK_WRITE_TARGET"
	leaseRecoveryRestoreAuthority   = "RESTORE_LEASE_AUTHORITY"
	leaseRecoveryRespectPeer        = "RESPECT_PEER_LEASE"
	leaseRecoveryUseGuardedExecutor = "USE_GUARDED_EXECUTOR"
	leaseRecoveryInspectAdmission   = "INSPECT_LEASE_ADMISSION"
)

type leaseAdmissionDiagnostic struct {
	recovery string
	remedy   string
}

func diagnosticForLeaseAdmissionFailure(failure leaseAdmissionFailure) (leaseAdmissionDiagnostic, bool) {
	switch failure {
	case leaseFailureUnknownFootprint:
		return leaseAdmissionDiagnostic{
			recovery: leaseRecoveryUseFileTools,
			remedy:   "Use structured file tools with explicit paths. Native shell commands have no blanket fak-run exemption: use an actually registered typed operation; generic opaque effects remain denied.",
		}, true
	case leaseFailureOutsideWorkspace:
		return leaseAdmissionDiagnostic{
			recovery: leaseRecoveryCheckWorkspace,
			remedy:   "Check the host-selected trusted workspace binding and propose only a path inside that canonical workspace.",
		}, true
	case leaseFailureWriteTargetUnresolvable:
		return leaseAdmissionDiagnostic{
			recovery: leaseRecoveryCheckWriteTarget,
			remedy:   "Check the literal working directory and write target, and ensure the target has an existing canonical parent.",
		}, true
	case leaseFailureWorkspaceAuthorityUnavailable, leaseFailureWorkspaceAuthorityRead,
		leaseFailureInvalidWorkspaceRoot, leaseFailureWorkspaceRootCanonicalization,
		leaseFailureLeaseAuthorityUnavailable, leaseFailureLeaseAuthorityRead,
		leaseFailureInvalidLeaseView, leaseFailureInvalidWorkspaceLeaseView,
		leaseFailureInvalidLeaseTree:
		return leaseAdmissionDiagnostic{
			recovery: leaseRecoveryRestoreAuthority,
			remedy:   "Restore a readable, valid canonical workspace and lease authority view before retrying the write.",
		}, true
	case leaseFailurePeerHeld:
		return leaseAdmissionDiagnostic{
			recovery: leaseRecoveryRespectPeer,
			remedy:   "Respect the peer lease: wait for its release or move the work to a disjoint leased tree.",
		}, true
	case leaseFailureAtomicGuardUnavailable:
		return leaseAdmissionDiagnostic{
			recovery: leaseRecoveryUseGuardedExecutor,
			remedy:   "Use the native atomic workspace executor, or emit a structured ABSTAIN record; never split admission from a client-side write.",
		}, true
	default:
		return leaseAdmissionDiagnostic{}, false
	}
}

func leaseAdmissionRefusal(reason abi.ReasonCode, failure leaseAdmissionFailure, claim string) WireVerdict {
	verdict := renderVerdict(abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  reason,
		By:      leaseAdmissionBy,
		Payload: abi.WitnessPayload{Claim: claim},
	}, nil)
	diagnostic, known := diagnosticForLeaseAdmissionFailure(failure)
	if !known {
		diagnostic = leaseAdmissionDiagnostic{
			recovery: leaseRecoveryInspectAdmission,
			remedy:   "Inspect the actual lease-admission claim and authority state before choosing a recovery; do not infer a failure class from unclassified metadata.",
		}
	}
	if verdict.Detail == nil {
		verdict.Detail = make(map[string]string)
	}
	verdict.Detail["admission_failure"] = string(failure)
	verdict.Detail["recovery"] = diagnostic.recovery
	verdict.Detail["remedy"] = diagnostic.remedy
	return verdict
}
