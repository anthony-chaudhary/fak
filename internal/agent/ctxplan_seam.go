package agent

// ctxplan_seam.go — the guarded seam between the agent turn loop and the ctxplan
// context PLANNER (issue #546, rung b). When Enabled, each turn the loop refreshes a
// heuristic Forecast from the running message list, runs ctxplan.Materialize under a
// window Budget, and renders the planned View as the next turn's history INSTEAD of
// appending+compacting — the live-loop integration the baseline spine left unbuilt
// (docs/notes/O1-TURN-CONTEXT-PLANNER-2026-06-23.md §6 "Not yet on the live loop").
//
// It is OFF BY DEFAULT behind config (CtxViewPlanner.Enabled == false, or set
// FAK_CTXPLAN_SEAM=on). Disabled, PlanTurn/RenderHistory are inert and the existing
// append+compact loop is byte-for-byte unchanged — the guard a production deploy needs
// before an in-flight rewrite of turn history ships.
//
// The seam is a thin adapter: it lowers agent Message into ctxplan Span (the lossless
// store the planner views), authors the heuristic Forecast (pin system prompt + active
// goal + last user turn; intents from the last user message), and renders the planned
// View back to []Message by paging each resident span's bytes through the gate. A
// mid-turn MISS routes through ctxplan.DemandPage (rung a) and feeds back into the next
// forecast via Forecast.Learn — "a forecast MISS costs one demand-page, never a lost
// fact."

import (
	"context"
	"errors"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// CtxViewPlanner is the guarded ctxplan seam for the agent turn loop. The zero value is
// DISABLED (Enabled == false); construct it with NewCtxViewPlanner to honor the
// FAK_CTXPLAN_SEAM config and a window Budget.
type CtxViewPlanner struct {
	// Enabled gates the seam. When false (the default), PlanTurn returns
	// ErrCtxSeamDisabled and RenderHistory returns its input unchanged, so the loop's
	// existing append+compact path is untouched. Flip it to integrate ctxplan.
	Enabled bool
	// Budget is the O(1) resident-token window the planner materializes each turn.
	Budget int
	// Layout optionally enables ctxplan's four-area profile (base/current/recent/deep).
	// nil preserves the original ProbeOptions path; a non-nil layout lets a caller tune
	// each area's N and precision while keeping the same global resident-token Budget.
	Layout *ctxplan.Layout
}

// ErrCtxSeamDisabled is returned by PlanTurn when the seam is OFF — the caller falls
// back to the append+compact loop. It is a sentinel, not an error to surface: a disabled
// seam is the documented default.
var ErrCtxSeamDisabled = errors.New("agent: ctxplan seam disabled (set CtxViewPlanner.Enabled or FAK_CTXPLAN_SEAM=on)")

// DefaultCtxViewBudget is the O(1) resident window the seam plans under when no explicit
// Budget is set. It is a conservative seed (a few thousand tokens), not a tuned constant
// — the same posture ctxplan.DefaultWeights takes.
const DefaultCtxViewBudget = 4096

// NewCtxViewPlanner builds a seam gated by the FAK_CTXPLAN_SEAM config. On ("on"/"1"/
// "true") enables it; anything else (including unset, the default) leaves it disabled.
// The Budget defaults to DefaultCtxViewBudget when budget <= 0.
func NewCtxViewPlanner(budget int) *CtxViewPlanner {
	p := &CtxViewPlanner{Budget: DefaultCtxViewBudget}
	if budget > 0 {
		p.Budget = budget
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_CTXPLAN_SEAM"))) {
	case "on", "1", "true":
		p.Enabled = true
	}
	return p
}

// PlanTurn is the per-turn integration point: lower the running messages into a lossless
// ctxplan store, author a heuristic Forecast, and run Materialize under the window budget
// — returning the planned O(1) View that RenderHistory turns into the next turn's
// history. When the seam is disabled it returns ErrCtxSeamDisabled so the loop falls
// back to append+compact unchanged.
func (p *CtxViewPlanner) PlanTurn(ctx context.Context, messages []Message) (ctxplan.View, error) {
	if !p.Enabled {
		return ctxplan.View{}, ErrCtxSeamDisabled
	}
	store, pinned := messagesToStore(messages)
	forecast := heuristicForecast(messages, pinned)
	if p.Layout != nil {
		return ctxplan.MaterializeLayout(ctx, store, forecast, ctxplan.Budget{Tokens: p.Budget}, nil, *p.Layout)
	}
	return ctxplan.Materialize(ctx, store, forecast, ctxplan.Budget{Tokens: p.Budget}, nil)
}

// RenderTurn is the one-step gateway entry point: lower the running messages into a
// lossless ctxplan store, author the heuristic Forecast, Materialize the O(1) view under
// the window Budget, and render it as the next turn's message history — the full "replace
// append+compact with a planned view" pass in one call. It is what the gateway serve/guard
// loop calls each turn to substitute a planned view for the forwarded history (issue #555).
//
// When the seam is disabled it returns its input UNCHANGED: the caller's existing history
// is byte-for-byte identical, so a deploy that leaves the flag off sees no behavior change
// at all — the guard a production deploy needs before an in-flight rewrite of turn history
// ships. On a planner error the caller (the gateway's maybePlanMessages) falls back to the
// full lossless history, so an experimental rewrite can never break a turn.
func (p *CtxViewPlanner) RenderTurn(ctx context.Context, messages []Message) ([]Message, error) {
	if !p.Enabled {
		return messages, nil
	}
	store, pinned := messagesToStore(messages)
	forecast := heuristicForecast(messages, pinned)
	var (
		view ctxplan.View
		err  error
	)
	if p.Layout != nil {
		view, err = ctxplan.MaterializeLayout(ctx, store, forecast, ctxplan.Budget{Tokens: p.Budget}, nil, *p.Layout)
	} else {
		view, err = ctxplan.Materialize(ctx, store, forecast, ctxplan.Budget{Tokens: p.Budget}, nil)
	}
	if err != nil {
		return nil, err
	}
	return renderPlanned(ctx, store, view), nil
}

// RenderHistory renders a planned View as the agent message list — the "renders a ctxplan
// View as turn history" half of the seam. It pages each resident span's bytes in through
// the store's trust gate (poison never enters context) and emits one Message per span in
// step order. When the seam is disabled it returns ErrCtxSeamDisabled so the caller falls
// back to append+compact unchanged.
func (p *CtxViewPlanner) RenderHistory(ctx context.Context, store ctxplan.Store, v ctxplan.View) ([]Message, error) {
	if !p.Enabled {
		return nil, ErrCtxSeamDisabled
	}
	return renderPlanned(ctx, store, v), nil
}

// renderPlanned pages each resident span of a planned View in through the store's trust
// gate and emits one Message per span in step order — the shared render loop behind both
// RenderHistory (the two-step agent-loop API) and RenderTurn (the one-step gateway API).
// A rendered span the gate declines mid-render stays out of context (the view's Refused
// set already accounts for it); it is skipped rather than emit poison.
func renderPlanned(ctx context.Context, store ctxplan.Store, v ctxplan.View) []Message {
	out := make([]Message, 0, len(v.Rendered))
	for _, r := range v.Rendered {
		body, err := store.Materialize(ctx, r.ID)
		if err != nil {
			continue
		}
		out = append(out, Message{Role: spanRoleToMessage(r.Role), Name: r.Role, Content: string(body)})
	}
	return out
}

// DemandPage is the mid-turn MISS handler — a thin pass-through to ctxplan.DemandPage
// (rung a) so the loop can fault an elided span back into the resident View without
// reaching across the seam boundary into the planner package directly.
func (p *CtxViewPlanner) DemandPage(ctx context.Context, store ctxplan.Store, v ctxplan.View, spanID string) (ctxplan.View, ctxplan.Fault, error) {
	return ctxplan.DemandPage(ctx, store, v, spanID)
}

// messagesToStore lowers the running agent messages into a fresh lossless ctxplan store
// (one Span per message, in turn order) and returns the store plus the ids of the spans a
// Forecast should pin (the active goal + system prompt + first user turn + last user turn).
// The store is the "core dump" the planner views; the lowering is faithful (the bytes are
// preserved verbatim, never summarized) so recall stays exact. The MemStore assigns span:<i>
// by insertion order == message index, so the pin ids address the spans directly.
//
// The active GOAL (a RoleGoal message, #845) is pinned FIRST — it is the intentional GC root
// of the heap, charged against the resident budget ahead of the structural pins so a session
// pursuing one goal never elides the span that goal depends on. It is a DISTINCT root from
// the first user turn, which the planner previously used as a goal proxy. Absent a RoleGoal
// message the pin set is byte-identical to before, so a caller that injects no goal sees no
// behavior change.
func messagesToStore(messages []Message) (*ctxplan.MemStore, []string) {
	store := ctxplan.NewMemStore()
	var pinned []string
	goal := -1
	firstUser := -1
	lastUser := -1
	systemSeen := false
	for i, msg := range messages {
		role := msg.Role
		if role == "" {
			role = RoleUser
		}
		store.Add(role, messageDurability(role), []byte(msg.Content), false)
		switch {
		case role == RoleGoal && goal < 0:
			goal = i
		case role == RoleSystem && !systemSeen:
			systemSeen = true
			pinned = append(pinned, ctxplanSpanID(i))
		case role == RoleUser && firstUser < 0:
			firstUser = i
		case role == RoleUser:
			lastUser = i
		}
	}
	// The goal root is charged first (prepended), ahead of the structural pins.
	if goal >= 0 {
		pinned = append([]string{ctxplanSpanID(goal)}, pinned...)
	}
	if firstUser >= 0 {
		pinned = append(pinned, ctxplanSpanID(firstUser))
	}
	if lastUser >= 0 && lastUser != firstUser {
		pinned = append(pinned, ctxplanSpanID(lastUser))
	}
	return store, pinned
}

// heuristicForecast authors the Forecast the planner optimizes under: intents are the
// content words of the LAST user message (what the upcoming turns are expected to ask
// about), and Pins are the active goal (a RoleGoal message, when present) + the system
// prompt + the first and last user turns (the spans a turn cannot proceed without). The
// pins are supplied by messagesToStore, which charges the goal root first. The weights
// stay at the ctxplan default seed — a sensible prior, tunable later via the learning loop.
func heuristicForecast(messages []Message, pins []string) ctxplan.Forecast {
	var intents []string
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == RoleUser {
			intents = contentIntents(messages[i].Content)
			break
		}
	}
	return ctxplan.Forecast{
		Intents: intents,
		Horizon: 1,
		Pins:    pins,
	}
}

// contentIntents extracts the content-word intents from one user message: lowercased
// tokens longer than two characters, the same extractive tokenization ctxplan's relevance
// ranker uses. Empty content yields no intents (selection falls to the priors + pins).
func contentIntents(s string) []string {
	seen := map[string]bool{}
	var out []string
	for _, t := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	}) {
		if len(t) > 2 && !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	return out
}

// messageDurability maps a chat role to the ctxplan durability class: the system prompt
// and durable user preferences are durable; tool results are session-scoped (reusable
// within the task); assistant turns are turn-scoped (a transient reasoning step).
func messageDurability(role string) string {
	switch role {
	case RoleSystem:
		return ctxplan.DurabilityDurable
	case RoleGoal:
		// A goal is durable within the session: it outlives any single turn (the
		// whole point of pinning it as a root) but is not a permanent fact like the
		// system prompt — it is discharged when the session's goal is met.
		return ctxplan.DurabilitySession
	case RoleTool:
		return ctxplan.DurabilitySession
	case RoleAssistant:
		return ctxplan.DurabilityTurn
	default:
		return ctxplan.DurabilitySession
	}
}

// spanRoleToMessage maps a ctxplan span role back to a chat role for rendering. A role
// that is already a chat role passes through; a tool name (e.g. "WebSearch") renders as
// role=tool with Name set (the caller distinguishes tool results by Name).
func spanRoleToMessage(role string) string {
	switch role {
	case RoleSystem, RoleUser, RoleAssistant, RoleTool, RoleGoal:
		return role
	default:
		return RoleTool // a span whose role was a tool name
	}
}

// ctxplanSpanID is the id the MemStore assigns the i-th span added (span:<i>), used to
// address pinned spans before the store is built. It reuses the package-level itoa.
func ctxplanSpanID(i int) string {
	return "span:" + itoa(i)
}
