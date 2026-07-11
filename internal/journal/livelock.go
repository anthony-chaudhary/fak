package journal

// Livelock telemetry — the durable witness for a result-side repeat loop.
//
// The decision journal records what the KERNEL decided over each tool call
// (DECIDE/DENY/QUARANTINE/…). A LIVELOCK is a different beast: the gateway's
// result-side detector observed the SAME failing tool call re-issued past its
// repeat threshold in one session — a loop the model drove, not a verdict the
// kernel rendered. Before this rung a trip left only an in-band advisory note
// that the client's transcript replay could not preserve, so a loop that burned
// tokens for many turns vanished from the durable record the guard-RSI fold and
// the fleet correlator both read. LIVELOCK closes that hole: a trip lands as a
// first-class chained row, so the fold can see, rank, and route it — and the
// fleet-observation feed can correlate it across traces — exactly like a crash.
//
// A LIVELOCK row is written DIRECTLY through the chain (AppendLivelock → append),
// NOT through Emit(abi.Event): a livelock trip is not a kernel event, and routing
// a synthetic one through the frozen ABI would fan it out to every decision-stream
// folder that assumes an event IS an adjudication. This mirrors AppendCrash and
// Cut's boundary anchor, which are likewise genuine chained rows that carry no
// verdict. Its chained forensic identity — Kind, tool (Tool), session (TraceID),
// and the failure class (Reason) — rides the frozen decision fields, so it
// verifies end-to-end with every existing row; the content-free repeat detail
// rides the non-chained Livelock carrier.

// KindLivelock marks a result-side repeat-loop trip row. It is a genuine chained
// row (it consumes the next Seq and chains onto the prior head) that carries no
// verdict, so a decision-folding consumer that keys on the closed verdict set skips
// its verdict accounting — but a Kind-keyed consumer (the guard-RSI fold, the fleet
// correlator) counts it as a first-class honesty signal.
const KindLivelock = "LIVELOCK"

// LivelockRow is the content-free repeat detail a LIVELOCK row carries. It is
// journal-local (rather than a re-export of the gateway/guardrsi envelope) so the
// on-disk schema stays self-contained — the same choice RestartHop makes — and it
// records only the failure's identity and shape, never raw tool arguments.
type LivelockRow struct {
	Tool        string `json:"tool,omitempty"`
	ArgsDigest  string `json:"args_digest,omitempty"`
	FailureHash string `json:"failure_hash,omitempty"`
	Verdict     string `json:"verdict,omitempty"`
	Reason      string `json:"reason,omitempty"`
	RepeatCount int    `json:"repeat_count,omitempty"`
	Fuse        bool   `json:"fuse,omitempty"`
	Escalate    bool   `json:"escalate,omitempty"`
}

// AppendLivelock records one result-side repeat-loop trip as a durable, chained
// LIVELOCK row and returns the committed row (with its stamped Seq/hash). traceID is
// the guard session id (recorded on TraceID) so the fold can attribute the loop to a
// session, and lr carries the content-free repeat detail (its Tool and Reason are
// mirrored onto the chained Tool/Reason fields so the row's forensic identity rides
// the frozen decision fields). It is a no-op returning the zero Row on a nil receiver
// or a nil payload, so a caller that guarded the journal on may call it
// unconditionally.
//
// Like a crash, the row is written directly through the chain (not the ABI fan-out):
// a livelock trip is not a kernel decision. The write is flushed per row by append.
func (j *Journal) AppendLivelock(traceID string, lr *LivelockRow) Row {
	if j == nil || lr == nil {
		return Row{}
	}
	payload := *lr
	row := Row{
		Kind:     KindLivelock,
		Tool:     lr.Tool,
		TraceID:  traceID,
		Reason:   lr.Reason,
		By:       "guard-gateway",
		Livelock: &payload,
	}
	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()
	return committed
}
