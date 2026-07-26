package agent

// loop_midflight.go — structured MID-FLIGHT session verbs (#5158, the write half of
// #2403, epic #2388): interrupt / drop-pending-call / set-budget, each landing at the
// loop's next CLEAN turn boundary (never mid-tool, never mid-adjudication) with a
// tamper-evident journal witness.
//
// Now that the loop's state is legible over the read half's typed progress SSE
// (#5148), an operator can act on it: MidflightVerbs is the owned per-run mailbox a
// transport (route / CLI) enqueues verbs onto, and the loop consumes at its boundary:
//
//   interrupt          — stop the loop at the next boundary with the closed
//                        session.ReasonInterrupted witness on ArmMetrics.
//                        StoppedBySession: the boundary-clean sibling of cancel
//                        (which drives the drive-state table) and terminate (which
//                        cuts mid-turn). Registered in the #2754 vocabulary spine as
//                        sessionctl.OpInterrupt with WitnessBoundaryStop.
//   drop-pending-call  — skip exactly one queued tool call, named by call_id, BEFORE
//                        it is dispatched. The net-new per-call skip consult: the
//                        model sees a typed status=skipped receipt (#2414), never a
//                        feigned success.
//   set-budget         — stage a live token/turn budget the loop WRITES THROUGH to
//                        the wired session table at its next boundary, so the same
//                        boundary's gate adjudicates the fresh allotment (an
//                        exhausted one stops the arm with its closed exhaustion
//                        reason — the existing OpBudget witness).
//
// The mailbox follows the additive-RunOption shape of WithSpeculator /
// WithToolTerminalWake: nil-safe everywhere, so with no verb issued (or no mailbox
// wired) the loop is byte-for-byte the historical loop. The journal is hash-chained
// (each record folds the previous record's sum into its own), so a dropped or edited
// record breaks the chain — VerifyMidflightJournal recomputes it.

import (
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// Mid-flight verb tokens — the closed verb vocabulary of this mailbox. Interrupt is
// additionally registered in the sessionctl #2754 spine (OpInterrupt); set-budget is
// the mid-flight setter for the spine's existing OpBudget; drop-pending-call is the
// net-new per-call verb #5158 introduces.
const (
	MidflightInterrupt       = "interrupt"
	MidflightDropPendingCall = "drop-pending-call"
	MidflightSetBudget       = "set-budget"
)

// Mid-flight journal statuses — the closed lifecycle a journaled verb moves through.
const (
	// MidflightQueued — the verb was accepted onto the mailbox (enqueue is NOT
	// applied; the loop consuming it at a boundary is).
	MidflightQueued = "QUEUED"
	// MidflightApplied — the loop consumed the verb at the recorded boundary.
	MidflightApplied = "APPLIED"
	// MidflightRefused — the verb was refused (sealed run at the enqueue edge, or no
	// budget sink at the boundary); the closed token / cause rides Detail.
	MidflightRefused = "REFUSED"
)

// MidflightRecord is one tamper-evident journal row: the verb, its arrival, its
// lifecycle status, and — once consumed — the 1-based turn boundary it landed at.
// Sum chains over PrevSum and this record's fields, so the journal is append-only
// evidence: editing or dropping a row breaks every later sum.
type MidflightRecord struct {
	Verb              string `json:"verb"`
	CallID            string `json:"call_id,omitempty"`
	Detail            string `json:"detail,omitempty"`
	ArrivedAtUnixNano int64  `json:"arrived_at_unix_nano"`
	Status            string `json:"status"`
	BoundaryTurn      int    `json:"boundary_turn,omitempty"`
	PrevSum           string `json:"prev_sum,omitempty"`
	Sum               string `json:"sum"`
}

// MidflightVerbs is the owned per-run mid-flight verb mailbox + journal. A transport
// enqueues (Interrupt / DropPendingCall / SetBudget); the loop consumes at its next
// clean turn boundary and seals the mailbox when the arm returns, after which every
// enqueue refuses with the closed CONTROL_SESSION_TERMINAL token — a finished run is
// not interruptible, exactly like every other control write against a terminal
// session. All methods are safe for concurrent use.
type MidflightVerbs struct {
	mu               sync.Mutex
	sealed           bool
	interruptArmed   bool
	interruptArrived int64
	drops            map[string]int64 // pending call_id -> arrival unix-nanos
	budget           *session.Budget
	budgetArrived    int64
	journal          []MidflightRecord
}

// NewMidflightVerbs constructs an empty mailbox for one run.
func NewMidflightVerbs() *MidflightVerbs { return &MidflightVerbs{} }

// WithMidflightVerbs wires the mid-flight verb mailbox into RunArm. A nil mailbox is
// accepted and degrades to the historical loop, so a caller may pass the option
// unconditionally.
func WithMidflightVerbs(v *MidflightVerbs) RunOption {
	return func(c *runConfig) { c.midflight = v }
}

// refuseSealed is the shared sealed-run refusal every enqueue edge returns once the
// arm has finished: the same closed terminal-session token the drive-state control
// verbs refuse with.
func (v *MidflightVerbs) refuseSealed(verb string) *session.ControlRefusal {
	r := &session.ControlRefusal{
		Op:     verb,
		Reason: session.ReasonControlSessionTerminal,
		Detail: "the run this mailbox belongs to has finished (sealed); a finished run cannot take a mid-flight verb",
	}
	v.append(MidflightRecord{Verb: verb, Detail: r.Reason, ArrivedAtUnixNano: time.Now().UnixNano(), Status: MidflightRefused})
	return r
}

// Interrupt arms a boundary-clean stop: the running arm completes its in-flight
// turn's admitted results, then stops at the next clean turn boundary with the
// closed session.ReasonInterrupted witness on StoppedBySession. Refuses with
// CONTROL_SESSION_TERMINAL once the run is sealed. Idempotent while armed.
func (v *MidflightVerbs) Interrupt() *session.ControlRefusal {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.sealed {
		return v.refuseSealed(MidflightInterrupt)
	}
	if v.interruptArmed {
		return nil
	}
	v.interruptArmed = true
	v.interruptArrived = time.Now().UnixNano()
	v.append(MidflightRecord{Verb: MidflightInterrupt, ArrivedAtUnixNano: v.interruptArrived, Status: MidflightQueued})
	return nil
}

// DropPendingCall names one queued tool call, by call_id, to be skipped BEFORE it is
// dispatched — exactly that call and nothing else. An empty call id names nothing
// and is ignored. Refuses with CONTROL_SESSION_TERMINAL once the run is sealed.
func (v *MidflightVerbs) DropPendingCall(callID string) *session.ControlRefusal {
	if callID == "" {
		return nil
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.sealed {
		return v.refuseSealed(MidflightDropPendingCall)
	}
	if v.drops == nil {
		v.drops = map[string]int64{}
	}
	if _, dup := v.drops[callID]; dup {
		return nil
	}
	arrived := time.Now().UnixNano()
	v.drops[callID] = arrived
	v.append(MidflightRecord{Verb: MidflightDropPendingCall, CallID: callID, ArrivedAtUnixNano: arrived, Status: MidflightQueued})
	return nil
}

// SetBudget stages a live budget the loop writes through to the wired session table
// at its next clean turn boundary (last staged write wins), so the SAME boundary's
// gate reads the fresh allotment. Refuses with CONTROL_SESSION_TERMINAL once the run
// is sealed.
func (v *MidflightVerbs) SetBudget(b session.Budget) *session.ControlRefusal {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.sealed {
		return v.refuseSealed(MidflightSetBudget)
	}
	v.budget = &b
	v.budgetArrived = time.Now().UnixNano()
	v.append(MidflightRecord{
		Verb:              MidflightSetBudget,
		Detail:            budgetDetail(b),
		ArrivedAtUnixNano: v.budgetArrived,
		Status:            MidflightQueued,
	})
	return nil
}

// Journal returns a stable copy of the tamper-evident verb journal.
func (v *MidflightVerbs) Journal() []MidflightRecord {
	v.mu.Lock()
	defer v.mu.Unlock()
	out := make([]MidflightRecord, len(v.journal))
	copy(out, v.journal)
	return out
}

// takeInterrupt consumes an armed interrupt at the given 1-based turn boundary,
// journaling the boundary it landed at. False when none is armed.
func (v *MidflightVerbs) takeInterrupt(boundaryTurn int) bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.interruptArmed {
		return false
	}
	v.interruptArmed = false
	v.append(MidflightRecord{Verb: MidflightInterrupt, ArrivedAtUnixNano: v.interruptArrived, Status: MidflightApplied, BoundaryTurn: boundaryTurn})
	return true
}

// takeDrop consumes a pending drop for callID at the given boundary, journaling it.
// False (dispatch normally) when the call was never named.
func (v *MidflightVerbs) takeDrop(callID string, boundaryTurn int) bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	arrived, ok := v.drops[callID]
	if !ok {
		return false
	}
	delete(v.drops, callID)
	v.append(MidflightRecord{Verb: MidflightDropPendingCall, CallID: callID, ArrivedAtUnixNano: arrived, Status: MidflightApplied, BoundaryTurn: boundaryTurn})
	return true
}

// applyBudget drains the staged budget (if any) into sink at the given boundary.
// The mailbox drains either way (a refused write must not retry forever, the same
// discipline as the redirect mailbox): sink=false — no table wired, or the table
// refused the write — journals REFUSED with the cause.
func (v *MidflightVerbs) applyBudget(boundaryTurn int, sink func(session.Budget) (applied bool, cause string)) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.budget == nil {
		return
	}
	b, arrived := *v.budget, v.budgetArrived
	v.budget = nil
	rec := MidflightRecord{Verb: MidflightSetBudget, Detail: budgetDetail(b), ArrivedAtUnixNano: arrived, BoundaryTurn: boundaryTurn}
	if applied, cause := sink(b); applied {
		rec.Status = MidflightApplied
	} else {
		rec.Status = MidflightRefused
		rec.Detail = cause + "; " + rec.Detail
	}
	v.append(rec)
}

// seal closes the mailbox when the arm returns: every later enqueue refuses with the
// closed terminal-session token. Idempotent.
func (v *MidflightVerbs) seal() {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.sealed = true
}

// append chains and appends one journal record. Callers hold v.mu.
func (v *MidflightVerbs) append(r MidflightRecord) {
	if n := len(v.journal); n > 0 {
		r.PrevSum = v.journal[n-1].Sum
	}
	r.Sum = midflightSum(r)
	v.journal = append(v.journal, r)
}

// midflightSum folds a record's fields (including PrevSum) into its chain sum.
func midflightSum(r MidflightRecord) string {
	payload := r.PrevSum + "|" + r.Verb + "|" + r.CallID + "|" + r.Detail + "|" + r.Status + "|" +
		strconv.FormatInt(r.ArrivedAtUnixNano, 10) + "|" + strconv.Itoa(r.BoundaryTurn)
	return fmt.Sprintf("%016x", fnv1a([]byte(payload)))
}

// VerifyMidflightJournal recomputes the journal's hash chain and reports whether it
// is intact — the tamper-evidence check: an edited, dropped, or reordered record
// breaks every later sum.
func VerifyMidflightJournal(records []MidflightRecord) bool {
	prev := ""
	for _, r := range records {
		if r.PrevSum != prev {
			return false
		}
		if midflightSum(r) != r.Sum {
			return false
		}
		prev = r.Sum
	}
	return true
}

// budgetDetail renders the bounded human note for a staged budget write.
func budgetDetail(b session.Budget) string {
	return fmt.Sprintf("turns_left=%d tokens_left=%d context_tokens_left=%d", b.TurnsLeft, b.TokensLeft, b.ContextTokensLeft)
}

// --- runConfig consults (all nil-safe: no mailbox => the historical loop) ---

// takeMidflightInterrupt consumes an armed mid-flight interrupt at a clean turn
// boundary, returning the closed stop reason to record on StoppedBySession.
func (c runConfig) takeMidflightInterrupt(boundaryTurn int) (string, bool) {
	if c.midflight == nil || !c.midflight.takeInterrupt(boundaryTurn) {
		return "", false
	}
	return session.ReasonInterrupted, true
}

// applyMidflightBudget writes a staged set-budget through to the wired session
// table at a clean turn boundary, so the same boundary's gate reads it. With no
// table (function-shaped gates carry no budget setter) the staged write drains as
// REFUSED — never silently applied.
func (c runConfig) applyMidflightBudget(boundaryTurn int) {
	if c.midflight == nil {
		return
	}
	c.midflight.applyBudget(boundaryTurn, func(b session.Budget) (bool, string) {
		if c.table == nil || c.trace == "" {
			return false, "no budget sink (no session table wired)"
		}
		if _, ok := c.table.SetBudget(c.trace, b); !ok {
			return false, "session table refused the budget write"
		}
		return true, ""
	})
}

// dropMidflightCall reports whether the operator named this queued call to be
// skipped before dispatch, consuming and journaling the drop.
func (c runConfig) dropMidflightCall(callID string, boundaryTurn int) bool {
	return c.midflight != nil && c.midflight.takeDrop(callID, boundaryTurn)
}

// sealMidflightVerbs seals the mailbox when the arm returns.
func (c runConfig) sealMidflightVerbs() {
	if c.midflight != nil {
		c.midflight.seal()
	}
}
