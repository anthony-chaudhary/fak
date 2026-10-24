package gateway

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// operatorApprovedLeaseOverride admits a call the operator approved (the floor
// verdict carries posture=operator-approved) past a lease-admission DEFAULT_DENY,
// recording the would-deny the same way the permissive workspace setting does.
// A lease refusal with any other reason (a peer owns the tree) still stands.
func operatorApprovedLeaseOverride(kernel abi.Verdict, wv *WireVerdict, lease WireVerdict) bool {
	if kernel.Meta["posture"] != adjudicator.PostureOperatorApproved || lease.Reason != abi.ReasonName(abi.ReasonDefaultDeny) {
		return false
	}
	if wv.Detail == nil {
		wv.Detail = make(map[string]string)
	}
	wv.Detail["kernel_by"] = wv.By
	wv.By = "lease-admission-operator-approved"
	wv.Detail["workspace_admission"] = adjudicator.PostureOperatorApproved
	wv.Detail["would_deny"] = lease.Reason
	wv.Detail["would_deny_by"] = lease.By
	wv.Detail["would_deny_claim"] = lease.Detail["claim"]
	return true
}

// offerOperatorApproval stamps a lease-admission DEFAULT_DENY with the approval
// id an operator could grant for this exact call.
func offerOperatorApproval(lease *WireVerdict, tool, rawArgs string) {
	if lease.Reason != abi.ReasonName(abi.ReasonDefaultDeny) {
		return
	}
	var args map[string]any
	if rawArgs != "" {
		_ = json.Unmarshal([]byte(rawArgs), &args)
	}
	if lease.Detail == nil {
		lease.Detail = make(map[string]string)
	}
	lease.Detail[detailOperatorApprovalID] = adjudicator.OperatorApprovalID(tool, args)
}
