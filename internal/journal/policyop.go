package journal

// Policy-op telemetry — the durable witness for an adjudicated POLICY MUTATION.
//
// #2406 makes policy mutation a kernel SYSCALL, not a config edit: the gateway's
// PolicyRegime adjudicates each add_rules / remove_rules / set_regime op (a
// tighten-only op applies immediately; a durable WIDENING without a witness is
// refused UNWITNESSED) and records every APPLIED — or witness-gated — op here as a
// first-class chained row, so `fak audit verify` covers a regime pivot exactly like
// a per-call tool decision.
//
// Like AppendCrash, the row is written DIRECTLY through the chain (AppendPolicyOp →
// appendLocked), NOT through Emit(abi.Event): a policy mutation is not a per-call
// kernel decision, and routing a synthetic event through the frozen ABI would fan a
// non-decision out to every decision-stream folder (harvest, trajectory, rungobs)
// that assumes an event IS an adjudication. Its chained forensic identity — Kind, the
// op verb (Tool), the session it is scoped to (TraceID), the applied/refused
// disposition (Verdict) and the closed reason (Reason) — rides the frozen decision
// fields, so it verifies end-to-end with every existing row and needs NO format
// migration (the confusion-risk the issue flagged): the rule EPOCH the op minted is
// carried on the existing CallSeq correlation field, not a new chained column.

// KindPolicyOp marks an adjudicated policy-mutation row. It is a genuine chained row
// (it consumes the next Seq and chains onto the prior head) that carries no per-call
// tool verdict, so a decision-folding consumer that keys on the closed verdict set
// skips its verdict accounting while the chain verifier covers it like every row.
const KindPolicyOp = "POLICY_OP"

// AppendPolicyOp records one adjudicated policy mutation as a durable, chained
// POLICY_OP row and returns the committed row (with its stamped Seq/hash). verb is
// the op kind (add_rules|remove_rules|set_regime, recorded on Tool), traceID is the
// session the op is scoped to (recorded on TraceID; "" for a durable op), verdict is
// the applied disposition ("ALLOW" for a tighten/session op, "WITNESS" for a
// witness-corroborated durable promotion, "DENY" for a refused unwitnessed widen),
// reason is a closed-vocabulary code ("" when none), and epoch is the rule epoch the
// op minted so an admission can cite which epoch allowed it (recorded on CallSeq, the
// same join-key field a call's DECIDE/QUARANTINE share). It is a no-op returning the
// zero Row on a nil receiver, so a caller that guarded the journal on may call it
// unconditionally.
func (j *Journal) AppendPolicyOp(verb, traceID, verdict, reason string, epoch uint64) Row {
	if j == nil {
		return Row{}
	}
	row := Row{
		Kind:    KindPolicyOp,
		Tool:    verb,
		TraceID: traceID,
		Verdict: verdict,
		Reason:  reason,
		By:      "policy-regime",
		CallSeq: epoch,
	}
	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()
	return committed
}
