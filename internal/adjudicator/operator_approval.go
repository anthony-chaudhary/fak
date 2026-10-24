package adjudicator

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Operator approval is the per-call human override of the capability floor: the
// operator-escalation amendment channel (internal/policy ChannelOperatorEscalation).
// See docs/notes/GUARD-OPERATOR-APPROVAL-2026-10-23.md.
//
// It is deliberately NOT a policy widening. An approval names ONE exact call (the
// tool plus its effect-bearing arguments, hashed the same way the reversibility
// confirm token is), so it can lift a POLICY_BLOCK arg rule or a DEFAULT_DENY for
// that call and nothing else. The floor's rules stay as they are for every other
// call. Every admitted call records the approval in Verdict.Meta.
//
// An approval also answers the reversibility preview-confirm hold for that call:
// the hold still runs, and a human approval of the exact call is accepted as its
// confirmation, in place of the agent echoing the confirm token.
//
// What an approval can never lift: the credential/host-config block, the
// self-modify rungs, SECRET_EXFIL and the hardwired egress floor. Those return
// before the lifted rungs, so an approved call still has to pass them.

// Meta keys an operator approval writes.
const (
	// MetaOperatorApprovalID names the approval id an operator would grant to
	// admit this exact call. It is stamped on every refusal an approval can lift.
	MetaOperatorApprovalID = "operator_approval_id"
	// PostureOperatorApproved is the Meta["posture"] value on a call admitted
	// because an operator approved it.
	PostureOperatorApproved = "operator-approved"
)

// OperatorApprovalID derives the stable id of one call. It reuses the confirm
// token's canonical argument view (argsForToken), so a byte-identical
// re-proposal keeps its id even when the free-text description or cosmetic
// command spacing changes.
func OperatorApprovalID(tool string, args map[string]any) string {
	canon, _ := json.Marshal(argsForToken(args))
	sum := sha256.Sum256([]byte("operator-approval\x00" + strings.ToLower(strings.TrimSpace(tool)) + "\x00" + string(canon)))
	return "ap-" + hex.EncodeToString(sum[:6])
}

// operatorApprovalLiftable reports whether a refusal reason may be lifted by a
// per-call operator approval. Only the operator-authored rule families qualify.
func operatorApprovalLiftable(r abi.ReasonCode) bool {
	return r == abi.ReasonPolicyBlock || r == abi.ReasonDefaultDeny
}

// operatorApprovalLiftableBy names the rungs whose POLICY_BLOCK / DEFAULT_DENY
// an approval may lift: the base monitor (arg rules, name deny, default deny)
// and the dangerous-gotcha floor, which re-checks the same danger classes under
// a default-open posture. Sub-rungs with their own authority (research egress,
// credential block) are not liftable.
func operatorApprovalLiftableBy(by string) bool {
	return by == "monitor" || by == gotchasBy
}

// operatorApprovalState tracks one Adjudicate call's approval. The id is
// derived lazily, so a call that is never refused and a session with no
// approvals pay no hashing cost.
type operatorApprovalState struct {
	approvals map[string]bool
	tool      string
	args      map[string]any
	id        string
	lifted    []string // reason names the approval lifted, in order
}

func newOperatorApprovalState(p Policy, tool string, args map[string]any) *operatorApprovalState {
	return &operatorApprovalState{approvals: p.OperatorApprovals, tool: tool, args: args}
}

func (s *operatorApprovalState) approvalID() string {
	if s.id == "" {
		s.id = OperatorApprovalID(s.tool, s.args)
	}
	return s.id
}

// lift resolves one liftable refusal site. When the operator approved this
// call it records the lift and returns lifted=true, so the caller continues to
// the later rungs. Otherwise it returns the refusal stamped with the approval
// id an operator could grant. Any other verdict passes through untouched.
func (s *operatorApprovalState) lift(v abi.Verdict) (abi.Verdict, bool) {
	if v.Kind != abi.VerdictDeny || !operatorApprovalLiftableBy(v.By) || !operatorApprovalLiftable(v.Reason) {
		return v, false
	}
	if len(s.approvals) > 0 && s.approvals[s.approvalID()] {
		s.lifted = append(s.lifted, abi.ReasonName(v.Reason))
		return v, true
	}
	v.Meta = cloneMeta(v.Meta)
	v.Meta[MetaOperatorApprovalID] = s.approvalID()
	return v, false
}

// confirmHold resolves the reversibility preview-confirm hold. An approved call
// counts as confirmed (confirmed=true, the caller continues to admit); otherwise
// the hold is returned stamped with the approval id an operator could grant.
func (s *operatorApprovalState) confirmHold(v abi.Verdict) (abi.Verdict, bool) {
	if v.Kind != abi.VerdictRequireWitness || v.By != "monitor/reversibility" {
		return v, false
	}
	if len(s.approvals) > 0 && s.approvals[s.approvalID()] {
		s.lifted = append(s.lifted, holdLiftName)
		return v, true
	}
	v.Meta = cloneMeta(v.Meta)
	v.Meta[MetaOperatorApprovalID] = s.approvalID()
	return v, false
}

// holdLiftName is the would_deny entry recorded when an approval answered the
// reversibility hold.
const holdLiftName = "REVERSIBILITY_HOLD"

// finish stamps an admitted call with the approval record when the operator
// approved it, whether or not an approval had to lift a floor refusal. The
// gateway's later workspace lease admission reads the record (posture), so an
// approved call is not refused there for an unprovable write footprint.
func (s *operatorApprovalState) finish(v abi.Verdict) abi.Verdict {
	if v.Kind != abi.VerdictAllow && v.Kind != abi.VerdictTransform {
		return v
	}
	if len(s.lifted) == 0 && (len(s.approvals) == 0 || !s.approvals[s.approvalID()]) {
		return v
	}
	v.Meta = cloneMeta(v.Meta)
	v.Meta["posture"] = PostureOperatorApproved
	if len(s.lifted) > 0 {
		v.Meta["would_deny"] = strings.Join(s.lifted, ",")
	}
	v.Meta[MetaOperatorApprovalID] = s.approvalID()
	return v
}

func cloneMeta(m map[string]string) map[string]string {
	out := make(map[string]string, len(m)+3)
	for k, val := range m {
		out[k] = val
	}
	return out
}
