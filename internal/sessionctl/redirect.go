package sessionctl

// redirect.go — the typed OUT-OF-BAND redirect / set-objective op of the operator
// control epic (#2755), the semantic twin of the add-constraint op (constraint.go,
// #2756): where add-constraint TIGHTENS a running session's floor, redirect changes
// WHAT the session is pursuing — it replaces the injection-shaped freeform `steer`
// prose for a GOAL change with a first-class, structured objective.
//
// The distinction this op exists to draw: a steer is spliced into the turn as a
// user MESSAGE (interpreted prose); a redirect is APPLIED as a first-class objective
// CHANGE. The loop carries the redirected objective as a standing directive, not as
// another operator turn — so the next turn pursues the new objective, and the change
// is witnessed as consumed (the mailbox drains, CurrentObjective reflects it), not
// merely enqueued.
//
// Like constraint.go this package owns ONLY the op payload, its validation, and the
// per-session next-boundary mailbox + live objective store. The per-op control
// ENVELOPE (who may send, the boundary declaration, the generalized witness-of-
// applied contract) and transport (the gateway route + `fak signal redirect` /
// `fak session objective` verbs) bind at the vocabulary spine (#2754), which
// registers this op into the one closed control vocabulary. Freeform steer stays as
// the escape hatch.

import (
	"fmt"
	"strings"
	"sync"
)

// ObjectiveStatus is the closed lifecycle vocabulary for a session's live objective.
// It mirrors internal/trajctl's objective status set so a redirect and the durable
// trajctl ledger agree on which states are steerable. active|paused are REDIRECTABLE
// (live); met|abandoned are terminal — a finished or abandoned objective cannot be
// redirected (that is a restart, by the epic's design).
type ObjectiveStatus string

const (
	// ObjectiveActive is the live, being-pursued objective a redirect sets.
	ObjectiveActive ObjectiveStatus = "active"
	// ObjectivePaused is a held-but-still-steerable objective; a redirect may
	// update it.
	ObjectivePaused ObjectiveStatus = "paused"
	// ObjectiveMet is a completed objective — terminal, not redirectable.
	ObjectiveMet ObjectiveStatus = "met"
	// ObjectiveAbandoned is a dropped objective — terminal, not redirectable.
	ObjectiveAbandoned ObjectiveStatus = "abandoned"
)

// Objective is the live per-session objective a redirect op sets or updates: the
// first-class thing the loop pursues. It is deliberately NOT a message — the loop
// renders it as a standing objective directive each turn (Directive), distinct from
// the user-message steer splice.
type Objective struct {
	// ID is the objective's stable id (carried through to the trajctl ledger). Empty
	// is allowed — a redirect that names no id still sets the goal.
	ID string `json:"id,omitempty"`
	// Goal is the objective's goal text — WHAT the session should now pursue.
	Goal string `json:"goal"`
	// Witness is the optional acceptance witness (how the objective is proven met).
	Witness string `json:"witness,omitempty"`
	// Status is the objective's lifecycle state. A redirect always lands an active
	// objective; the field lets the store hold a terminal (non-redirectable) one.
	Status ObjectiveStatus `json:"status,omitempty"`
}

// Redirectable reports whether an objective may be redirected. A live (active/paused
// or unset) objective is; a terminal (met/abandoned) one is not — that is the "no
// redirectable state" case. It mirrors trajctl's objectiveOpen predicate.
func (o Objective) Redirectable() bool {
	switch o.Status {
	case ObjectiveMet, ObjectiveAbandoned:
		return false
	default:
		return true
	}
}

// Directive renders the model-facing standing objective directive the loop carries
// into a turn as a first-class objective change (a system directive), NOT a spliced
// user message. Empty goal renders "" so the historical loop is unchanged.
func (o Objective) Directive() string {
	if strings.TrimSpace(o.Goal) == "" {
		return ""
	}
	var b strings.Builder
	b.WriteString("OBJECTIVE (redirected out of band): ")
	b.WriteString(strings.TrimSpace(o.Goal))
	if w := strings.TrimSpace(o.Witness); w != "" {
		b.WriteString("\nAcceptance witness: ")
		b.WriteString(w)
	}
	b.WriteString("\nPursue this objective from the next turn; it replaces the prior goal.")
	return b.String()
}

// Redirect is the typed out-of-band redirect / set-objective op payload. JSON tags
// are the wire shape the (spine-bound, #2754) route will carry.
type Redirect struct {
	// ObjectiveID optionally names the objective being set/updated.
	ObjectiveID string `json:"objective_id,omitempty"`
	// Goal is the new goal text — required; a redirect with no goal has nothing to
	// redirect TO and is refused malformed at the edge.
	Goal string `json:"goal"`
	// Witness is the optional acceptance witness carried onto the objective.
	Witness string `json:"witness,omitempty"`
}

// RedirectRefuseReason is the closed refusal vocabulary for a redirect op — the same
// closed-reason discipline as constraint.go's RefuseReason and every other fak
// refusal.
type RedirectRefuseReason string

const (
	// RedirectMalformed refuses a redirect whose shape cannot be applied — today,
	// an empty goal (nothing to redirect to).
	RedirectMalformed RedirectRefuseReason = "REDIRECT_MALFORMED"
	// RedirectNoRedirectableState refuses a redirect for a session whose current
	// objective is terminal (met/abandoned) — there is no live objective to
	// redirect; reopening a finished objective is a restart, not an out-of-band op.
	RedirectNoRedirectableState RedirectRefuseReason = "REDIRECT_NO_REDIRECTABLE_STATE"
)

// RedirectRefusal is a structured, closed-reason refusal of one redirect op. It
// implements error so plumbing can thread it, but callers should switch on Reason,
// never parse Detail.
type RedirectRefusal struct {
	Reason RedirectRefuseReason `json:"reason"`
	Detail string               `json:"detail,omitempty"`
}

func (r *RedirectRefusal) Error() string { return refusalString(string(r.Reason), r.Detail) }

// redirectRefuse builds a RedirectRefusal in one line.
func redirectRefuse(reason RedirectRefuseReason, format string, args ...any) *RedirectRefusal {
	return &RedirectRefusal{Reason: reason, Detail: fmt.Sprintf(format, args...)}
}

// Validate checks the op's SHAPE — parse, don't validate later: the producer edge
// (route/CLI) refuses a malformed op synchronously. Live-state legality (a terminal
// current objective) is necessarily decided against the session's live objective and
// is handled by EnqueueRedirect / ApplyPendingRedirect, not here.
func (r Redirect) Validate() *RedirectRefusal {
	if strings.TrimSpace(r.Goal) == "" {
		return redirectRefuse(RedirectMalformed, "redirect needs a non-empty goal")
	}
	return nil
}

// asObjective folds a validated redirect into the live objective it lands: an active
// objective carrying the new goal (and, when named, its id/witness). A redirect
// always lands an ACTIVE objective — it makes the goal the live pursued one.
func (r Redirect) asObjective() Objective {
	return Objective{
		ID:      strings.TrimSpace(r.ObjectiveID),
		Goal:    strings.TrimSpace(r.Goal),
		Witness: strings.TrimSpace(r.Witness),
		Status:  ObjectiveActive,
	}
}

// redirect state is per-session, keyed by the run's trace id — the same keying as
// constraint.go's mailbox and the a2achan steer bus. pending holds redirects accepted
// out of band until the loop drains them at its next turn boundary; objective holds
// the live current objective the loop reads and carries. Process-global, like
// a2achan.Default and constraint.go's pending.
var (
	redirectMu        sync.Mutex
	redirectPending   = map[string][]Redirect{}
	redirectNext      = map[string][]NextRecord{}
	redirectObjective = map[string]Objective{}
)

// SetObjective seeds/overwrites the live objective for trace directly (not via the
// out-of-band op path). It is how a session declares its initial objective and how a
// test arranges a terminal (non-redirectable) objective for the refusal case. An
// empty trace is ignored.
func SetObjective(trace string, o Objective) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return
	}
	redirectMu.Lock()
	defer redirectMu.Unlock()
	redirectObjective[trace] = o
}

// CurrentObjective returns the live objective for trace and whether one is declared.
// The loop reads it at the turn boundary to carry the standing objective directive.
func CurrentObjective(trace string) (Objective, bool) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return Objective{}, false
	}
	redirectMu.Lock()
	defer redirectMu.Unlock()
	o, ok := redirectObjective[trace]
	return o, ok
}

// ClearObjective drops the live objective and any pending redirects for trace — the
// session teardown hook so a finished run leaks no per-trace state. Idempotent.
func ClearObjective(trace string) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return
	}
	redirectMu.Lock()
	defer redirectMu.Unlock()
	delete(redirectObjective, trace)
	delete(redirectPending, trace)
	delete(redirectNext, trace)
}

// EnqueueRedirect accepts one redirect for the session keyed by trace, to be applied
// at its next turn boundary. The op is validated HERE, at the edge: a malformed op
// (empty goal) is refused synchronously, and — because the live objective is queryable
// now — a redirect against a TERMINAL current objective is refused at the edge too
// with its closed "no redirectable state" reason, so an illegal op never enters the
// mailbox.
func EnqueueRedirect(trace string, r Redirect) *RedirectRefusal {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return redirectRefuse(RedirectMalformed, "empty session trace")
	}
	if ref := r.Validate(); ref != nil {
		return ref
	}
	redirectMu.Lock()
	defer redirectMu.Unlock()
	if cur, ok := redirectObjective[trace]; ok && !cur.Redirectable() {
		return redirectRefuse(RedirectNoRedirectableState,
			"current objective status %q is terminal; a finished/abandoned objective cannot be redirected (restart instead)", cur.Status)
	}
	redirectPending[trace] = append(redirectPending[trace], r)
	return nil
}

// RedirectPendingLen reports how many redirects are queued for trace (observability +
// tests).
func RedirectPendingLen(trace string) int {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return 0
	}
	redirectMu.Lock()
	defer redirectMu.Unlock()
	return len(redirectPending[trace])
}

// ApplyPendingRedirect drains every redirect queued for trace and folds each into the
// live objective, in arrival order — the consumer half the session loop calls at its
// turn boundary. A redirect that finds a terminal current objective (a race: the
// objective completed between enqueue and apply) is refused, not silently applied; the
// mailbox is drained either way (a refused op must not retry forever). It returns the
// objectives it applied and any refusals. Idempotent on an empty mailbox.
func ApplyPendingRedirect(trace string) (applied []Objective, refused []RedirectRefusal) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil, nil
	}
	redirectMu.Lock()
	defer redirectMu.Unlock()
	queued := redirectPending[trace]
	delete(redirectPending, trace)
	appendNext := func(r Redirect, result ApplyResult) {
		move := Move{
			Kind: MoveRedirect, Render: RenderSystemDirective,
			Session: SessionAutonomous, Gate: "sessionctl-redirect",
			Source: "agent-turn-boundary", Payload: r.asObjective().Directive(),
		}
		// Constants above satisfy the shared closed contract; a future vocabulary
		// change that violates it is pinned by the redirect Next tests.
		if record, err := WitnessMove(move, result); err == nil {
			redirectNext[trace] = append(redirectNext[trace], record)
		}
	}
	for _, r := range queued {
		if cur, ok := redirectObjective[trace]; ok && !cur.Redirectable() {
			refusal := RedirectRefusal{
				Reason: RedirectNoRedirectableState,
				Detail: fmt.Sprintf("current objective status %q is terminal", cur.Status),
			}
			refused = append(refused, refusal)
			appendNext(r, ApplyResult{Refusal: refusal.Error()})
			continue
		}
		obj := r.asObjective()
		redirectObjective[trace] = obj
		applied = append(applied, obj)
		appendNext(r, ApplyResult{Applied: true})
	}
	return applied, refused
}

// ReadRedirectNextRecords returns and clears the independently re-readable Next
// witnesses emitted while the trace's redirect mailbox was applied. Empty/no-op
// drains produce no records.
func ReadRedirectNextRecords(trace string) []NextRecord {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	redirectMu.Lock()
	defer redirectMu.Unlock()
	records := append([]NextRecord(nil), redirectNext[trace]...)
	delete(redirectNext, trace)
	return records
}
