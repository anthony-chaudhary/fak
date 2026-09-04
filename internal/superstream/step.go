package superstream

import (
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// StepAction represents the exact next state transition for the super workstream.
type StepAction string

const (
	ActionAcquireLease     StepAction = "ACQUIRE_LEASE"
	ActionYieldContended   StepAction = "YIELD_CONTENDED"
	ActionExecuteItem      StepAction = "EXECUTE_ITEM"
	ActionWitnessAndCommit StepAction = "WITNESS_AND_COMMIT"
	ActionReleaseLease     StepAction = "RELEASE_LEASE"
	ActionResetContext     StepAction = "RESET_CONTEXT"
	ActionAdvanceQueue     StepAction = "ADVANCE_QUEUE"
	ActionStreamComplete   StepAction = "STREAM_COMPLETE"
	ActionStreamHalted     StepAction = "STREAM_HALTED"
)

// StepDecision is the pure verdict produced by DecideStep naming what must happen next.
type StepDecision struct {
	Action        StepAction           `json:"action"`
	ItemIndex     int                  `json:"item_index"`
	Item          *WorkItem            `json:"item,omitempty"`
	LeaseRequest  *laneadmit.Request   `json:"lease_request,omitempty"`
	Safety        ContextSafetyVerdict `json:"safety"`
	Carryover     *StreamCarryoverSeed `json:"carryover,omitempty"`
	DisjointIndex int                  `json:"disjoint_index,omitempty"`
	Reason        string               `json:"reason"`
	NextSafeStep  string               `json:"next_safe_step"`
}

// StepResult captures the witnessed outcome of the action taken by the executor shell.
type StepResult struct {
	LeaseAcquired    bool   `json:"lease_acquired"`
	LeaseError       string `json:"lease_error,omitempty"`
	FencingToken     string `json:"fencing_token,omitempty"`
	ExecutedTurns    int    `json:"executed_turns"`
	ExecutedTokens   int    `json:"executed_tokens"`
	WitnessSuccess   bool   `json:"witness_success"`
	WitnessOutput    string `json:"witness_output,omitempty"`
	CommitSHA        string `json:"commit_sha,omitempty"`
	ExecutionError   string `json:"execution_error,omitempty"`
	LeaseReleased    bool   `json:"lease_released"`
	ContextResetDone bool   `json:"context_reset_done"`
}

// DecideStep computes the single next action to progress the workstream.
// It is pure and deterministic over spec, state, live leases, and taxonomy.
func DecideStep(spec StreamSpec, state StreamState, holder string, liveLeases []laneadmit.Lease, tax laneadmit.Taxonomy) StepDecision {
	norm := spec.NormalizedSpec()
	safety := EvaluateContextSafety(norm, state)

	if state.Closed {
		return StepDecision{
			Action:       ActionStreamHalted,
			ItemIndex:    state.ActiveIndex,
			Safety:       safety,
			Reason:       fmt.Sprintf("stream is already closed: %s", state.CloseReason),
			NextSafeStep: "no action; stream terminated",
		}
	}

	if safety.Status == StatusContextExhausted {
		return StepDecision{
			Action:       ActionStreamHalted,
			ItemIndex:    state.ActiveIndex,
			Safety:       safety,
			Reason:       safety.Reason,
			NextSafeStep: "close stream and report budget exhaustion to operator",
		}
	}

	// If all items in queue are already terminal:
	if state.AllTerminal() {
		if state.CurrentLease != nil {
			return StepDecision{
				Action:       ActionReleaseLease,
				ItemIndex:    state.ActiveIndex,
				Safety:       safety,
				Reason:       "all items terminal; release trailing held lane lease",
				NextSafeStep: fmt.Sprintf("release lease %q before final completion", state.CurrentLease.LeaseID),
			}
		}
		seed := BuildCarryoverSeed(norm, state)
		return StepDecision{
			Action:       ActionStreamComplete,
			ItemIndex:    state.ActiveIndex,
			Safety:       safety,
			Carryover:    &seed,
			Reason:       fmt.Sprintf("all %d queue items completed or settled", len(state.Queue)),
			NextSafeStep: "archive stream run record and report completion",
		}
	}

	// Ensure ActiveIndex points to a valid pending or in-flight item.
	if state.ActiveIndex >= len(state.Queue) {
		// Find first non-terminal item
		found := -1
		for i, it := range state.Queue {
			if !it.Status.Terminal() {
				found = i
				break
			}
		}
		if found < 0 {
			if state.CurrentLease != nil {
				return StepDecision{
					Action:       ActionReleaseLease,
					ItemIndex:    state.ActiveIndex,
					Safety:       safety,
					Reason:       "no pending items; release held lease",
					NextSafeStep: "release lease",
				}
			}
			return StepDecision{
				Action:       ActionStreamComplete,
				ItemIndex:    state.ActiveIndex,
				Safety:       safety,
				Reason:       "queue fully processed",
				NextSafeStep: "archive stream run",
			}
		}
		state.ActiveIndex = found
	}

	active := state.ActiveItem()
	if active == nil {
		return StepDecision{
			Action:       ActionStreamComplete,
			Safety:       safety,
			Reason:       "empty queue",
			NextSafeStep: "complete",
		}
	}

	// Case 1: Active item is terminal but lease is still held -> Release Lease first.
	if active.Status.Terminal() && state.CurrentLease != nil {
		return StepDecision{
			Action:       ActionReleaseLease,
			ItemIndex:    state.ActiveIndex,
			Item:         active,
			Safety:       safety,
			Reason:       fmt.Sprintf("item %q is terminal (%s) but lease %q is still held", active.ID, active.Status, state.CurrentLease.LeaseID),
			NextSafeStep: fmt.Sprintf("release lane lease for %s", active.Lane),
		}
	}

	// Case 2: Active item is terminal and lease is released -> Advance queue or Reset Context.
	if active.Status.Terminal() && state.CurrentLease == nil {
		seed := BuildCarryoverSeed(norm, state)
		return StepDecision{
			Action:       ActionAdvanceQueue,
			ItemIndex:    state.ActiveIndex,
			Item:         active,
			Safety:       safety,
			Carryover:    &seed,
			Reason:       fmt.Sprintf("item %q reached terminal status %s; advance to next queue item", active.ID, active.Status),
			NextSafeStep: "advance ActiveIndex to next pending item with fresh carryover context",
		}
	}

	// Case 3: Active item is Pending -> check lease admission.
	if active.Status == ItemPending {
		verdict, req := EvaluateLeaseAdmission(state.StreamID, *active, holder, liveLeases, tax)
		if verdict.Admit {
			return StepDecision{
				Action:       ActionAcquireLease,
				ItemIndex:    state.ActiveIndex,
				Item:         active,
				LeaseRequest: &req,
				Safety:       safety,
				Reason:       fmt.Sprintf("lane %q is free; acquire lease %s", active.Lane, req.LeaseID),
				NextSafeStep: fmt.Sprintf("acquire lane lease %s over %v", req.LeaseID, req.Tree),
			}
		}

		// Contended: search for a downstream disjoint item in queue.
		disjointIdx, ok := FindNextDisjointItem(state.StreamID, state.Queue, state.ActiveIndex+1, holder, liveLeases, tax)
		if ok {
			return StepDecision{
				Action:        ActionYieldContended,
				ItemIndex:     state.ActiveIndex,
				Item:          active,
				Safety:        safety,
				DisjointIndex: disjointIdx,
				Reason: fmt.Sprintf("item %q lane %q is contended (%s); disjoint pending item %q (index %d) is available to advance",
					active.ID, active.Lane, verdict.Reason, state.Queue[disjointIdx].ID, disjointIdx),
				NextSafeStep: fmt.Sprintf("skip to disjoint item index %d (%s) or yield current item", disjointIdx, state.Queue[disjointIdx].ID),
			}
		}

		return StepDecision{
			Action:        ActionYieldContended,
			ItemIndex:     state.ActiveIndex,
			Item:          active,
			Safety:        safety,
			DisjointIndex: -1,
			Reason: fmt.Sprintf("item %q lane %q is contended by live peer lease (%s) with no disjoint candidates",
				active.ID, active.Lane, verdict.Reason),
			NextSafeStep: "yield item or pause until peer lease clears",
		}
	}

	// Case 4: Lease acquired -> execute item.
	if active.Status == ItemLeaseAcquired {
		return StepDecision{
			Action:       ActionExecuteItem,
			ItemIndex:    state.ActiveIndex,
			Item:         active,
			Safety:       safety,
			Reason:       fmt.Sprintf("lease held for item %q; execute work under bounded budget (%d turns)", active.ID, active.MaxTurns),
			NextSafeStep: fmt.Sprintf("execute task %s in lane %s", active.ID, active.Lane),
		}
	}

	// Case 5: Item Executing -> check context safety or proceed to witness & commit.
	if active.Status == ItemExecuting {
		// If context ceiling is hit mid-execution, demand context reset/checkpoint.
		if safety.Status == StatusContextResetRequired {
			seed := BuildCarryoverSeed(norm, state)
			return StepDecision{
				Action:       ActionResetContext,
				ItemIndex:    state.ActiveIndex,
				Item:         active,
				Safety:       safety,
				Carryover:    &seed,
				Reason:       safety.Reason,
				NextSafeStep: "flush raw context transcript, re-arm with StreamCarryoverSeed, and resume item",
			}
		}

		return StepDecision{
			Action:       ActionWitnessAndCommit,
			ItemIndex:    state.ActiveIndex,
			Item:         active,
			Safety:       safety,
			Reason:       fmt.Sprintf("item %q execution complete; verify witness %q and commit", active.ID, active.Witness),
			NextSafeStep: fmt.Sprintf("run witness %s and commit explicit paths %v", active.Witness, active.Tree),
		}
	}

	// Case 6: Item Witnessed or Committed -> release lease.
	if active.Status == ItemWitnessed || active.Status == ItemCommitted {
		if state.CurrentLease != nil {
			return StepDecision{
				Action:       ActionReleaseLease,
				ItemIndex:    state.ActiveIndex,
				Item:         active,
				Safety:       safety,
				Reason:       fmt.Sprintf("item %q committed (%s); release lane lease %s", active.ID, active.CommitSHA, state.CurrentLease.LeaseID),
				NextSafeStep: fmt.Sprintf("release lane lease %s", state.CurrentLease.LeaseID),
			}
		}
		// If lease already released, advance.
		return StepDecision{
			Action:       ActionAdvanceQueue,
			ItemIndex:    state.ActiveIndex,
			Item:         active,
			Safety:       safety,
			Reason:       fmt.Sprintf("item %q committed and lease released; advance queue", active.ID),
			NextSafeStep: "advance to next queue item",
		}
	}

	return StepDecision{
		Action:       ActionStreamHalted,
		ItemIndex:    state.ActiveIndex,
		Item:         active,
		Safety:       safety,
		Reason:       fmt.Sprintf("unhandled state for item %q (status=%s)", active.ID, active.Status),
		NextSafeStep: "inspect stream state",
	}
}

// ApplyStep transitions the stream state forward given the decision taken and witnessed result.
func ApplyStep(spec StreamSpec, state *StreamState, decision StepDecision, result StepResult, now time.Time) {
	norm := spec.NormalizedSpec()
	active := state.ActiveItem()

	switch decision.Action {
	case ActionAcquireLease:
		if result.LeaseAcquired && active != nil {
			held := MakeHeldLease(state.StreamID, *active, "stream-worker", now, result.FencingToken)
			state.CurrentLease = &held
			active.Status = ItemLeaseAcquired
		} else if active != nil {
			active.Error = result.LeaseError
		}

	case ActionYieldContended:
		if decision.DisjointIndex >= 0 && decision.DisjointIndex < len(state.Queue) {
			// Skip to disjoint item: switch ActiveIndex to the disjoint item
			state.ActiveIndex = decision.DisjointIndex
		} else if active != nil {
			active.Status = ItemYielded
			state.YieldedCount++
			active.Error = decision.Reason
		}

	case ActionExecuteItem:
		if active != nil {
			active.Status = ItemExecuting
			active.TurnsSpent += result.ExecutedTurns
			active.TokensSpent += result.ExecutedTokens
			state.CurrentItemTurns += result.ExecutedTurns
			state.CurrentItemTokens += result.ExecutedTokens
			state.TotalTurnsSpent += result.ExecutedTurns
			state.TotalTokensSpent += result.ExecutedTokens
			if result.ExecutionError != "" {
				active.Error = result.ExecutionError
			}
		}

	case ActionWitnessAndCommit:
		if active != nil {
			active.WitnessResult = result.WitnessOutput
			if result.WitnessSuccess && result.CommitSHA != "" {
				active.CommitSHA = result.CommitSHA
				active.Status = ItemCommitted
			} else if !result.WitnessSuccess {
				active.Status = ItemFailed
				active.Error = fmt.Sprintf("witness verification failed: %s", result.ExecutionError)
				state.FailedCount++
			} else {
				active.Status = ItemWitnessed
			}
		}

	case ActionReleaseLease:
		state.CurrentLease = nil
		if active != nil {
			if active.Status == ItemCommitted {
				active.Status = ItemCompleted
				state.CompletedCount++
			}
		}

	case ActionResetContext:
		state.CurrentItemTurns = 0
		state.CurrentItemTokens = 0

	case ActionAdvanceQueue:
		state.CurrentItemTurns = 0
		state.CurrentItemTokens = 0
		// Advance to the next non-terminal item
		advanced := false
		for i := state.ActiveIndex + 1; i < len(state.Queue); i++ {
			if !state.Queue[i].Status.Terminal() {
				state.ActiveIndex = i
				advanced = true
				break
			}
		}
		if !advanced {
			state.ActiveIndex = len(state.Queue)
		}

	case ActionStreamComplete:
		state.Closed = true
		state.CloseReason = "stream queue completed"

	case ActionStreamHalted:
		state.Closed = true
		state.CloseReason = decision.Reason
	}

	// Update safety check after mutations
	_ = norm
}
