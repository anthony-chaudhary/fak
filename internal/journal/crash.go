package journal

// Crash telemetry — the durable witness for a supervised-child ABNORMAL EXIT.
//
// The decision journal records what the KERNEL decided over each tool call
// (DECIDE/DENY/QUARANTINE/…). A guard child CRASH is a different beast entirely:
// the wrapped agent (or guard itself) dies with a non-zero exit, an OS signal, or
// a Go panic — an event that happens OUTSIDE the adjudication path and so never
// flows through the ABI emitter that drives every decision row. Before this rung,
// a crash left no trace in the journal, which made it invisible to the guard-RSI
// loop (that loop folds ONLY this journal). CHILD_CRASH closes that hole: a crash
// lands as a first-class chained row, so the RSI fold can see, rank, and route it
// exactly like a verdict-quality hole.
//
// A CHILD_CRASH row is written DIRECTLY through the chain (AppendCrash → append),
// NOT through Emit(abi.Event): a crash is not a kernel event, and routing a
// synthetic one through the frozen ABI would fan it out to every decision-stream
// folder (harvest, trajectory, rungobs) that assumes an event IS an adjudication.
// This mirrors Cut's KindCut anchor, which is likewise a genuine chained row that
// carries no verdict. Its chained forensic identity — Kind, agent (Tool), session
// (TraceID), and the closed-vocabulary class (Reason) — rides the frozen decision
// fields, so it verifies end-to-end with every existing row.

// KindChildCrash marks a supervised-child abnormal-exit row. It is a genuine
// chained row (it consumes the next Seq and chains onto the prior head) that
// carries no verdict, so a decision-folding consumer that keys on the closed
// verdict set skips its verdict accounting — but the guard-RSI fold, which keys on
// Kind, counts it as the worst honesty hole a session can carry.
const KindChildCrash = "CHILD_CRASH"

// Crash classes — the CLOSED vocabulary for the Reason field of a CHILD_CRASH row.
// A closed set (mirroring the adjudicator's closed reason vocabulary) keeps the
// RSI recovery rung's per-class recurrence buckets stable instead of exploding
// into free-text: every abnormal exit maps to exactly one of these.
const (
	// CrashSignal is a child killed by an OS signal (SIGSEGV/SIGABRT/SIGKILL/…): a
	// genuine crash or an external kill. On a signaled exit os.ProcessState reports
	// ExitCode() == -1.
	CrashSignal = "SIGNAL_CRASH"
	// CrashOOM is the out-of-memory kill (SIGKILL from the OOM killer, or the
	// conventional 137 = 128+SIGKILL exit code). Split from SIGNAL_CRASH so the loop
	// can tell a resource exhaustion apart from a logic fault.
	CrashOOM = "OOM"
	// CrashNonzeroExit is a child that exited on its own with a non-zero code (a Go
	// panic the runtime turned into exit 2, an unrecovered error, a terminal upstream
	// failure) — abnormal, but a clean process teardown rather than a signal.
	CrashNonzeroExit = "NONZERO_EXIT"
)

// AppendCrash records one supervised-child abnormal exit as a durable, chained
// CHILD_CRASH row and returns the committed row (with its stamped Seq/hash). agent
// is the wrapped agent name (recorded on Tool), traceID is the guard session id
// (recorded on TraceID) so the recovery rung can attribute a crash to a session,
// reasonClass is one of the Crash* closed-vocabulary constants (recorded on
// Reason), and exitCode is the child's exit code (-1 when signaled). It is a no-op
// returning the zero Row on a nil receiver, so a caller that guarded the journal
// on may call it unconditionally.
//
// Like Cut, the row is written directly through the chain (not the ABI fan-out):
// a crash is not a kernel decision. The write is flushed per row by append, so a
// crash row survives the very exit that produced it.
func (j *Journal) AppendCrash(agent, traceID, reasonClass string, exitCode int) Row {
	if j == nil {
		return Row{}
	}
	row := Row{
		Kind:     KindChildCrash,
		Tool:     agent,
		TraceID:  traceID,
		Reason:   reasonClass,
		By:       "guard-supervisor",
		ExitCode: exitCode,
	}
	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()
	return committed
}
