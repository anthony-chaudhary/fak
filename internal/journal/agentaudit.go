package journal

// In-process microagent lifecycle audit — the durable witness for a HOST that
// runs 1000s of agent loops in ONE process (#2011, epic #2000 M11).
//
// Production today wraps each agent in its own `fak guard` process, each opening
// its OWN per-process hash-chained decision journal
// (.dispatch-runs/guard-audit/<lane>-<backend>-<pid>-<uuid>.jsonl) because the
// sha256 chain has no cross-process lock. In-process that inverts: one host holds
// ONE mutex-guarded journal and every hosted agent appends to the SAME chain, so
// N agents leave O(1) files per host instead of one JSONL each. This is the write
// path an in-process AuditSink (internal/microagent) targets.
//
// A lifecycle row is written DIRECTLY through the chain (AppendAgentEvent →
// append), NOT through Emit(abi.Event): a spawn/retire is not a kernel decision,
// and routing a synthetic one through the frozen ABI would fan it out to every
// decision-stream folder (harvest, trajectory, rungobs) that assumes an event IS
// an adjudication. This mirrors AppendCrash and Cut's boundary anchor: a genuine
// chained row that carries no verdict. Its forensic identity — Kind, the agent id
// (Tool + TraceID), and the terminal reason (Reason) — rides the frozen decision
// fields, so it chains and verifies end-to-end with every existing row. Crucially,
// because the host's journal is ALSO the kernel's registered ABI emitter, a
// QUARANTINE the shared gateway raises lands in the SAME chain with its witness +
// call_seq intact (see Row.Witness / Row.CallSeq): mixing lifecycle rows into the
// audit stream drops no adjudication forensics.

// Agent-lifecycle kinds — the CLOSED vocabulary for the Kind field of an
// in-process microagent audit row, one per host lifecycle transition
// (internal/microagent.EventKind). A closed set (mirroring the Crash* and the
// adjudicator's closed reason vocabularies) keeps Kind-keyed consumers stable
// instead of exploding into free-text. These are additive to the decision Kind
// set; a decision-folding consumer that keys on the closed verdict set skips them.
const (
	KindAgentSpawn  = "AGENT_SPAWN"  // agent accepted and enqueued
	KindAgentReject = "AGENT_REJECT" // spawn refused (bounded queue full)
	KindAgentDone   = "AGENT_DONE"   // Step reported done
	KindAgentCancel = "AGENT_CANCEL" // retired by cancel/close before done
	KindAgentError  = "AGENT_ERROR"  // Step returned a non-cancel error
)

// AppendAgentEvent records one in-process microagent lifecycle transition as a
// durable, chained row and returns the committed row (with its stamped Seq/hash).
// kind is one of the KindAgent* closed-vocabulary constants; agentID is the hosted
// agent's id (recorded on Tool AND TraceID — the host uses the agent id as its
// session TraceID, so the row is tagged with the agent/trace id the acceptance
// asks for); reason carries the terminal reason (the Step error for an
// AGENT_ERROR row, empty otherwise). It is a no-op returning the zero Row on a nil
// receiver, so a caller that guarded the host journal on may call it
// unconditionally.
//
// Like AppendCrash, the row is written directly through the chain (not the ABI
// fan-out): a lifecycle transition is not a kernel decision. The write is flushed
// per row by append, so the row survives a crash of the host that produced it.
func (j *Journal) AppendAgentEvent(kind, agentID, reason string) Row {
	if j == nil {
		return Row{}
	}
	row := Row{
		Kind:    kind,
		Tool:    agentID,
		TraceID: agentID,
		Reason:  reason,
		By:      "microagent-host",
	}
	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()
	return committed
}
