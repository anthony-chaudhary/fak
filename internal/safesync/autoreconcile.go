package safesync

import (
	"context"
	"strings"
	"time"
)

// AutoReconcile reason codes. They explain why a disjoint push refusal was NOT
// healed; a successful heal carries no reason.
const (
	AutoReconcileReasonNotDisjoint   = "NOT_DISJOINT"
	AutoReconcileReasonMergeActive   = "MERGE_ACTIVE"
	AutoReconcileReasonPacketError   = "PACKET_ERROR"
	AutoReconcileReasonNotDispatched = "PACKET_NOT_SAFE_DISJOINT"
	AutoReconcileReasonSourceMoved   = "SOURCE_MOVED"
	AutoReconcileReasonExecuteFailed = "EXECUTE_FAILED"
	AutoReconcileReasonExecNotPushed = "EXECUTE_NOT_PUSHED"
	AutoReconcileReasonReadback      = "READBACK_FAILED"
)

// AutoReconcileOptions configures AutoReconcileDisjoint: the one shared,
// verified heal that both `fak commit --push` (safecommit) and `fak sync push`
// route a DIVERGED_DISJOINT push refusal through, so the two verbs cannot drift.
type AutoReconcileOptions struct {
	Repo    string
	Remote  string
	Branch  string
	Session string
	// SourceSHA, when set, is the exact object the caller tried to publish. The
	// heal refuses (SOURCE_MOVED) when the packet's local HEAD is not that object,
	// so a peer commit that landed on HEAD after capture is never swept into the
	// push (#4221's pinned-source guarantee).
	SourceSHA          string
	MaxPushRetries     int
	PushVelocityBudget time.Duration
	Runner             Runner
	// Build and Execute are seams; nil means BuildReconciliationPacket / ExecutePacket.
	Build   func(context.Context, PacketOptions) (*ReconciliationPacket, error)
	Execute func(context.Context, *ReconciliationPacket, ExecuteOptions) (*ExecutionReceipt, error)
}

// AutoReconcileOutcome is the evidence of one AutoReconcileDisjoint attempt.
type AutoReconcileOutcome struct {
	Attempted   bool                  `json:"attempted"`
	Pushed      bool                  `json:"pushed"`
	Disposition Disposition           `json:"disposition,omitempty"`
	Reason      string                `json:"reason,omitempty"`
	Detail      string                `json:"detail,omitempty"`
	Receipt     *ExecutionReceipt     `json:"receipt,omitempty"`
	Packet      *ReconciliationPacket `json:"-"`
}

// IsDisjointPushRefusal reports whether a SafePush result is the one refusal
// the auto-reconcile heal is allowed to act on: an unpublished push whose
// divergence classified as disjoint paths. Overlapping divergence, behind,
// errors, and network refusals are never eligible.
func IsDisjointPushRefusal(res PushResult) bool {
	return !res.Pushed && res.Reason == ReasonDivergedDisjoint
}

// AutoReconcileDisjoint heals a DIVERGED_DISJOINT push refusal by building an
// owner-aware reconciliation packet and executing it only when the packet is
// dispatchable with disposition safe-disjoint. The executor performs the merge,
// the re-push through SafePush, and the independent graph + peer-byte readback.
// Anything else (an active merge, a non-disjoint packet, colliding dirty paths,
// a moved source, an executor refusal) leaves the caller's original refusal in
// place: the outcome is a value, never a force-push.
func AutoReconcileDisjoint(ctx context.Context, pushed PushResult, opts AutoReconcileOptions) AutoReconcileOutcome {
	out := AutoReconcileOutcome{}
	if !IsDisjointPushRefusal(pushed) {
		out.Reason = AutoReconcileReasonNotDisjoint
		out.Detail = "push refusal is not a disjoint divergence; nothing to auto-reconcile"
		return out
	}
	run := opts.Runner
	if run == nil {
		run = RealRunner
	}
	repo := opts.Repo
	if strings.TrimSpace(repo) == "" {
		repo = "."
	}
	remote := opts.Remote
	if strings.TrimSpace(remote) == "" {
		remote = "origin"
	}
	build := opts.Build
	if build == nil {
		build = BuildReconciliationPacket
	}
	execute := opts.Execute
	if execute == nil {
		execute = ExecutePacket
	}
	session := strings.TrimSpace(opts.Session)

	if isMergeActive(ctx, run, repo) {
		out.Reason = AutoReconcileReasonMergeActive
		out.Detail = "a merge is in progress (MERGE_HEAD present); not auto-reconciling"
		return out
	}

	pkt, err := build(ctx, PacketOptions{
		Repo:    repo,
		Remote:  remote,
		Branch:  opts.Branch,
		Runner:  run,
		Session: session,
	})
	if err != nil || pkt == nil {
		out.Reason = AutoReconcileReasonPacketError
		if err != nil {
			out.Detail = "build reconciliation packet: " + err.Error()
		} else {
			out.Detail = "build reconciliation packet returned no packet"
		}
		return out
	}
	out.Packet = pkt
	out.Disposition = pkt.Disposition
	if !pkt.Dispatchable || pkt.Disposition != DispositionSafeDisjoint {
		out.Reason = AutoReconcileReasonNotDispatched
		out.Detail = "reconciliation packet is " + string(pkt.Disposition) + ", not a dispatchable safe-disjoint packet"
		return out
	}
	if src := strings.TrimSpace(opts.SourceSHA); src != "" && strings.TrimSpace(pkt.LocalHead) != src {
		out.Reason = AutoReconcileReasonSourceMoved
		out.Detail = "local HEAD " + pkt.LocalHead + " is not the pinned push source " + src + "; not auto-reconciling"
		return out
	}

	out.Attempted = true
	receipt, execErr := execute(ctx, pkt, ExecuteOptions{
		Repo:               repo,
		Remote:             remote,
		Branch:             opts.Branch,
		Runner:             run,
		WriterLeaseTTL:     DefaultWriterLeaseTTL,
		MaxPushRetries:     opts.MaxPushRetries,
		PushVelocityBudget: opts.PushVelocityBudget,
		Session:            session,
	})
	out.Receipt = receipt
	if receipt != nil && receipt.Pushed {
		// The merged source is on the remote. A failed post-push readback is
		// reported, but it cannot un-publish the push, so Pushed stays true.
		out.Pushed = true
		if execErr != nil {
			out.Reason = AutoReconcileReasonReadback
			out.Detail = execErr.Error()
		}
		return out
	}
	if execErr != nil {
		out.Reason = AutoReconcileReasonExecuteFailed
		out.Detail = execErr.Error()
		return out
	}
	out.Reason = AutoReconcileReasonExecNotPushed
	out.Detail = "reconciliation executed but did not publish"
	if receipt != nil && receipt.Detail != "" {
		out.Detail = receipt.Detail
	}
	return out
}
