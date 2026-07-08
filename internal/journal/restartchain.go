package journal

// Restart-chain telemetry — the durable witness for a supervised-child BUDGET
// RESTART (#3057, the follow-up to #3055's --continue reattach).
//
// When `fak guard --restart-on-budget` relaunches its wrapped child under a
// continuation trace, the continuity of the whole session hangs on that hop:
// which trace handed off to which, where the carryover seed landed, how large it
// was, and whether the relaunched child actually resumed the conversation or
// booted cold. Before this rung the evidence was two uncorrelated stderr lines
// plus a seed JSON in a temp dir — nothing an operator could query after the
// fact, which is how a fleet accumulated over a thousand orphaned seeds with no
// record of whether continuity survived any of the restarts that wrote them.
//
// A RESTART_HOP row closes that hole the same way CHILD_CRASH closed the crash
// hole: one first-class chained row per restart, written DIRECTLY through the
// chain (AppendRestartHop → append), NOT through Emit(abi.Event) — a restart is
// supervision, not a kernel decision, and routing a synthetic event through the
// frozen ABI would fan it out to every decision-stream folder that assumes an
// event IS an adjudication. The chained forensic identity — Kind, agent (Tool),
// guard session (TraceID), and the closed continuity class (Reason) — rides the
// frozen decision fields, so it verifies end-to-end with every existing row; the
// full correlated record (schema fak.guard.restart_chain.v1) rides the
// non-chained Restart payload field, the same layering ExitCode uses for crashes.

// KindRestartHop marks a supervised-child budget-restart row. It is a genuine
// chained row (it consumes the next Seq and chains onto the prior head) that
// carries no verdict, so decision-folding consumers skip it; readers that key on
// Kind (fak guard restart-audit, fak session status) fold it into the session's
// restart chain.
const KindRestartHop = "RESTART_HOP"

// RestartChainSchema names the correlated per-restart record carried on a
// RESTART_HOP row's Restart field. Versioned like every fak wire schema:
// additive-only; never edit a shipped /vN in place.
const RestartChainSchema = "fak.guard.restart_chain.v1"

// Restart-hop continuity classes — the CLOSED vocabulary for the Reason field of
// a RESTART_HOP row (and the Status field of its payload). A closed set keeps
// audit buckets stable instead of exploding into free-text.
const (
	// RestartHopOK is a hop whose continuity handback ENGAGED: the relaunched
	// child resumes the captured conversation (handback "continue", or the future
	// #3056 "seed-prompt").
	RestartHopOK = "ok"
	// RestartHopInert is a hop that carried continuity data nothing consumed: the
	// agent was unrecognized, so the child relaunched cold while the seed sat
	// unread on disk (handback "ORPHANED"). The session keeps running, but the
	// task context did not ride along.
	RestartHopInert = "inert"
	// RestartHopBreak is a hop whose handover itself failed: no continuation
	// trace was minted or no seed survived the write, so there was nothing to
	// hand the relaunched child at all (the emit-time analogue of the
	// reset-limit status line's continuity=blocked).
	RestartHopBreak = "break"
	// RestartHopLoss is the AUDIT-TIME backfill class, never emitted live: a seed
	// file exists on disk with no recorded hop, so the outcome is unknowable and
	// presumed lost. `fak guard restart-audit` stamps it when it backfills the
	// orphans that predate this rung.
	RestartHopLoss = "loss"
)

// RestartHop is the correlated per-restart record (RestartChainSchema): every
// axis of one supervised-child budget restart tied together in a single value,
// instead of scattered across stderr lines, env vars, and a seed file.
type RestartHop struct {
	Schema     string `json:"schema"`                     // RestartChainSchema
	Hop        int    `json:"hop"`                        // 1-based restart ordinal within the guard session
	FromTrace  string `json:"from_trace_id"`              // the exhausted trace the child was serving under
	ToTrace    string `json:"to_trace_id,omitempty"`      // the continuation trace it was relaunched under
	SeedFile   string `json:"seed_file,omitempty"`        // where the carryover seed JSON landed ("" = write failed)
	SeedTokens int    `json:"seed_tokens,omitempty"`      // approximate token count of the seed text
	Handback   string `json:"handback"`                   // "continue" | "seed-prompt" | "ORPHANED"
	Child      string `json:"child_session_id,omitempty"` // the session id the relaunched child serves under (FAK_SESSION_ID)
	Status     string `json:"status"`                     // RestartHopOK | RestartHopInert | RestartHopBreak | RestartHopLoss
}

// AppendRestartHop records one supervised-child budget restart as a durable,
// chained RESTART_HOP row and returns the committed row (with its stamped
// Seq/hash). agent is the wrapped agent name (recorded on Tool), guardTraceID is
// the guard session id (recorded on TraceID) so the chain can be attributed to a
// session, and hop is the correlated record (its Status is mirrored onto the
// chained Reason field; its Schema defaults to RestartChainSchema when unset).
// It is a no-op returning the zero Row on a nil receiver, so a caller that
// guarded the journal on may call it unconditionally.
//
// Like AppendCrash, the row is written directly through the chain (not the ABI
// fan-out): a restart is not a kernel decision. The write is flushed per row by
// append, so the hop survives the child teardown that follows it.
func (j *Journal) AppendRestartHop(agent, guardTraceID string, hop RestartHop) Row {
	if j == nil {
		return Row{}
	}
	if hop.Schema == "" {
		hop.Schema = RestartChainSchema
	}
	row := Row{
		Kind:    KindRestartHop,
		Tool:    agent,
		TraceID: guardTraceID,
		Reason:  hop.Status,
		By:      "guard-supervisor",
		Restart: &hop,
	}
	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()
	return committed
}
