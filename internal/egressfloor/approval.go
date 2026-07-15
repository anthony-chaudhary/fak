package egressfloor

import "github.com/anthony-chaudhary/fak/internal/abi"

// ApprovalClass is the closed input vocabulary for the deterministic replacement
// of a model-based "smart approve" judge. The classifier that proves a class is
// separate; this fold only lowers a proved class to a stable kernel verdict.
type ApprovalClass string

const (
	ApprovalExplicitAllow  ApprovalClass = "explicit_allow"
	ApprovalMetadataEgress ApprovalClass = "metadata_egress"
	ApprovalPolicyBlocked  ApprovalClass = "policy_blocked"
	ApprovalSelfModify     ApprovalClass = "self_modify"
	ApprovalSecretExfil    ApprovalClass = "secret_exfil"
	ApprovalMalformed      ApprovalClass = "malformed"
	ApprovalUnknownTool    ApprovalClass = "unknown_tool"
	ApprovalNeedsJudgment  ApprovalClass = "needs_human_judgment"
)

const approvalAdjudicator = "egressfloor/smart-approval"

// ApprovalClasses returns the complete stable label space in declaration order.
func ApprovalClasses() []ApprovalClass {
	return []ApprovalClass{
		ApprovalExplicitAllow, ApprovalMetadataEgress, ApprovalPolicyBlocked,
		ApprovalSelfModify, ApprovalSecretExfil, ApprovalMalformed,
		ApprovalUnknownTool, ApprovalNeedsJudgment,
	}
}

// AdjudicateApproval deterministically maps a proved decision class onto fak's
// closed verdict/reason vocabulary. There is no prose parser and no model call.
// A genuine judgment call is held for an independent human witness rather than
// being auto-approved; an unknown class fails closed as MALFORMED.
func AdjudicateApproval(class ApprovalClass) abi.Verdict {
	switch class {
	case ApprovalExplicitAllow:
		return abi.Verdict{Kind: abi.VerdictAllow, By: approvalAdjudicator}
	case ApprovalMetadataEgress:
		return denyApproval(ReasonEgressBlock)
	case ApprovalPolicyBlocked:
		return denyApproval(abi.ReasonPolicyBlock)
	case ApprovalSelfModify:
		return denyApproval(abi.ReasonSelfModify)
	case ApprovalSecretExfil:
		return denyApproval(abi.ReasonSecretExfil)
	case ApprovalMalformed:
		return denyApproval(abi.ReasonMalformed)
	case ApprovalUnknownTool:
		return denyApproval(abi.ReasonUnknownTool)
	case ApprovalNeedsJudgment:
		return abi.Verdict{
			Kind:    abi.VerdictRequireWitness,
			Reason:  abi.ReasonUnwitnessed,
			By:      approvalAdjudicator,
			Payload: abi.WitnessPayload{Claim: "human approval required for ambiguous tool call"},
		}
	default:
		return denyApproval(abi.ReasonMalformed)
	}
}

func denyApproval(reason abi.ReasonCode) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: reason, By: approvalAdjudicator}
}
