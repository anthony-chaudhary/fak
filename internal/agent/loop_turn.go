package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/attemptbudget"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// armRunner owns the mutable state shared by the phases of one arm's turn loop.
// Keeping it named avoids passing the same large parameter set through every phase
// while leaving runArm responsible only for arm-level setup and teardown.
type armRunner struct {
	cfg              *runConfig
	metrics          *ArmMetrics
	fak              bool
	kernel           *kernel.Kernel
	speculation      *specState
	messages         []Message
	tools            []ToolDef
	model            string
	stream           bool
	sink             StreamSink
	complete         armCompleteFunc
	log              *[]traceEvent
	stopTerminated   func() bool
	repeatedFailures attemptbudget.RepeatedFailureTracker
}

type armTurnAction uint8

const (
	armTurnDispatchTools armTurnAction = iota
	armTurnContinue
	armTurnStop
)

func (r *armRunner) run(ctx context.Context, maxTurns int) error {
	for turn := 0; turn < maxTurns; turn++ {
		stop, err := r.runTurn(ctx, turn)
		if err != nil || stop {
			return err
		}
	}
	r.metrics.HitTurnCap = true
	// The loop hit the turn cap with a speculation still pending: squash it (it was never
	// confirmed by an authoritative call), so no provisional effect leaks past the run.
	r.speculation.resolve(ctx, nil, r.metrics)
	if r.cfg != nil && r.cfg.gracefulDrain && r.metrics.FinalAnswer == "" {
		if err := r.runSynthesisTurn(ctx, maxTurns); err != nil {
			return err
		}
	}
	r.finalizeFak()
	return nil
}

func (r *armRunner) runSynthesisTurn(ctx context.Context, turn int) error {
	r.metrics.GracefulDrained = true
	turnSink := r.sink
	comp, err := r.complete(ctx, r.messages, nil, turnSink)
	if err != nil {
		return err
	}
	if comp == nil {
		return fmt.Errorf("synthesis turn: nil completion")
	}
	r.metrics.Turns++
	r.metrics.PromptTokens += comp.Usage.PromptTokens
	r.metrics.CompletionTokens += comp.Usage.CompletionTokens
	if r.cfg != nil {
		r.cfg.debitTurn(comp.Usage)
	}
	asst := comp.Message
	asst.Role = RoleAssistant
	r.messages = append(r.messages, asst)
	r.metrics.FinalAnswer = asst.Content
	r.metrics.SynthesizedFinalTurn = true
	if r.cfg != nil {
		r.cfg.emitProgress(ProgressEvent{Kind: ProgressTurnDone, Turn: turn + 1})
	}
	return nil
}

func (r *armRunner) runTurn(ctx context.Context, turn int) (bool, error) {
	perTurnCap, stop, err := r.beginTurn(ctx, turn)
	if err != nil || stop {
		return stop, err
	}

	asst, action, err := r.requestModel(ctx, turn, perTurnCap)
	if err != nil {
		return false, err
	}
	switch action {
	case armTurnContinue:
		return false, nil
	case armTurnStop:
		return true, nil
	default:
		return r.dispatchToolCalls(ctx, turn, asst)
	}
}

// beginTurn applies the clean-boundary controls before a model request is admitted.
func (r *armRunner) beginTurn(ctx context.Context, turn int) (int, bool, error) {
	var toolTerminalPayload string
	if turn > 0 && r.cfg.toolTerminalWake != nil {
		select {
		case <-r.cfg.toolTerminalWake.signal:
			wake := r.cfg.toolTerminalWake.next()
			payload, _ := json.Marshal(wake)
			toolTerminalPayload = string(payload)
		case <-ctx.Done():
			return 0, false, ctx.Err()
		}
	}
	// Mid-flight set-budget (#5158): write staged budget before the gate reads it.
	r.cfg.applyMidflightBudget(turn + 1)
	// Mid-flight interrupt (#5158): take a boundary-clean stop only after the prior turn.
	if reason, stopped := r.cfg.takeMidflightInterrupt(turn + 1); stopped {
		if toolTerminalPayload != "" {
			r.cfg.toolTerminalWake.release()
		}
		r.metrics.StoppedBySession = reason
		r.finalizeFak()
		return 0, true, nil
	}
	// The session-control gate admits or stops the turn at the same clean boundary.
	perTurnCap, proceed, stopReason := r.cfg.gateTurn(ctx)
	if !proceed {
		if toolTerminalPayload != "" {
			r.cfg.toolTerminalWake.release()
		}
		r.metrics.StoppedBySession = stopReason
		r.finalizeFak()
		return 0, true, nil
	}
	if toolTerminalPayload != "" {
		r.cfg.toolTerminalWake.mark("DISPATCHED")
		r.messages = append(r.messages, Message{Role: RoleUser, Content: toolTerminalPayload})
		sessionctl.RecordToolTerminalWakeNext(r.cfg.trace, toolTerminalPayload)
	}
	r.cfg.applyPace(perTurnCap)
	// A turn is announced only after the session gate admits it.
	r.cfg.emitProgress(ProgressEvent{Kind: ProgressTurnStarted, Turn: turn + 1})
	return perTurnCap, false, nil
}

// requestModel assembles and receipts one model request, then admits its assistant
// response. It returns a closed action so runTurn preserves the original continue,
// final-answer, and tool-dispatch ordering.
func (r *armRunner) requestModel(ctx context.Context, turn, perTurnCap int) (Message, armTurnAction, error) {
	// Boundary directives are spliced in the established steer/redirect/constraint order.
	beforeDirectives := len(r.messages)
	r.messages = spliceTurnDirectives(*r.cfg, r.messages)
	injected := append([]Message(nil), r.messages[beforeDirectives:]...)
	inputClaim, err := r.cfg.claimTurnInputs(turn+1, injected)
	if err != nil {
		return Message{}, armTurnStop, fmt.Errorf("%s arm turn %d claim admitted input: %w", r.metrics.Arm, turn+1, err)
	}
	releaseClaim := func(reason string, cause error) error {
		if err := r.cfg.releaseInputClaim(inputClaim, reason); err != nil {
			return fmt.Errorf("%w; release input claim: %v", cause, err)
		}
		return cause
	}
	// Render exactly once because SessionPlanner.RenderTurn is stateful.
	planned, err := r.cfg.promptMessages(ctx, r.messages)
	if err != nil {
		return Message{}, armTurnStop, fmt.Errorf("%s arm turn %d prompt assembly: %w", r.metrics.Arm, turn+1, releaseClaim("PROMPT_ASSEMBLY_FAILED", err))
	}
	if inputClaim.ID != "" && !claimedInputsSurviveAssembly(injected, planned) {
		err := fmt.Errorf("prompt assembly dropped claimed input")
		return Message{}, armTurnStop, fmt.Errorf("%s arm turn %d prompt assembly: %w", r.metrics.Arm, turn+1, releaseClaim("PROMPT_ASSEMBLY_DROPPED_CLAIMED_INPUT", err))
	}
	if r.cfg.modelRequestObserver != nil {
		boundary := ModelRequestBoundary{
			Model: r.model, Turn: turn + 1, Stream: r.stream, MaxTokens: perTurnCap,
			Messages: append([]Message(nil), planned...),
			Tools:    append([]ToolDef(nil), r.tools...),
			Injected: injected,
		}
		if inputClaim.ID != "" {
			claimCopy := inputClaim
			boundary.InputClaim = &claimCopy
		}
		if err := r.cfg.modelRequestObserver(boundary); err != nil {
			return Message{}, armTurnStop, fmt.Errorf("%s arm turn %d model request receipt: %w", r.metrics.Arm, turn+1, releaseClaim("REQUEST_RECEIPT_FAILED", err))
		}
	} else if inputClaim.ID != "" {
		err := fmt.Errorf("durable input claim requires a model request observer")
		return Message{}, armTurnStop, fmt.Errorf("%s arm turn %d model request receipt: %w", r.metrics.Arm, turn+1, releaseClaim("REQUEST_RECEIPT_UNWIRED", err))
	}

	var streamedChunks []string
	turnSink := r.sink
	if r.stream {
		turnSink = func(chunk string) error {
			if r.sink != nil {
				if err := r.sink(chunk); err != nil {
					return err
				}
			}
			streamedChunks = append(streamedChunks, chunk)
			return nil
		}
	}
	sampleOpts := sampleOptsFor(perTurnCap)
	if r.cfg != nil && (r.cfg.reasoningEffort != "" || r.cfg.thinkingBudget != nil) {
		if r.cfg.reasoningEffort == EffortTierBalanced || r.cfg.reasoningEffort == EffortTierAdaptive {
			ta, _ := AssessTranscriptTurn(planned)
			b := ResolveEffortBudget(r.cfg.reasoningEffort, r.cfg.thinkingBudget, ta)
			sampleOpts = append(sampleOpts, WithThinkingBudget(b), WithReasoningEffort(r.cfg.reasoningEffort))
		} else {
			if r.cfg.reasoningEffort != "" {
				sampleOpts = append(sampleOpts, WithReasoningEffort(r.cfg.reasoningEffort))
			}
			if r.cfg.thinkingBudget != nil {
				sampleOpts = append(sampleOpts, WithThinkingBudget(*r.cfg.thinkingBudget))
			}
		}
	}
	comp, err := r.complete(ctx, planned, r.tools, turnSink, sampleOpts...)
	if err != nil {
		completionErr := err
		err = releaseClaim("MODEL_DISPATCH_FAILED", err)
		// A terminate-cancelled completion is a typed stop; other errors remain fail-loud.
		terminated := r.stopTerminated()
		if r.stream && r.cfg.interruptedTurnObserver != nil {
			observed := InterruptedTurn{
				Turn: turn + 1, Chunks: append([]string(nil), streamedChunks...),
				Reason: ClassifyTermination(completionErr),
			}
			if observeErr := r.cfg.interruptedTurnObserver(observed); observeErr != nil {
				return Message{}, armTurnStop, fmt.Errorf("%s arm turn %d interrupted output receipt: %v (completion: %w)", r.metrics.Arm, turn+1, observeErr, err)
			}
		}
		if terminated {
			return Message{}, armTurnStop, nil
		}
		return Message{}, armTurnStop, fmt.Errorf("%s arm turn %d: %w", r.metrics.Arm, turn+1, err)
	}
	r.metrics.Turns++
	r.metrics.PromptTokens += comp.Usage.PromptTokens
	r.metrics.CompletionTokens += comp.Usage.CompletionTokens
	r.cfg.debitTurn(comp.Usage)
	asst := comp.Message
	asst.Role = RoleAssistant
	if comp.ToolCallsDropped && len(asst.ToolCalls) == 0 {
		return Message{}, armTurnStop, fmt.Errorf("%s arm turn %d: upstream announced tool_calls but none parsed; refusing to skip adjudication", r.metrics.Arm, turn+1)
	}
	r.messages = append(r.messages, asst)
	if len(asst.ToolCalls) != 0 {
		return asst, armTurnDispatchTools, nil
	}
	if r.cfg.finalGate != nil {
		if satisfied, missing := r.cfg.finalGate(); !satisfied {
			continuation := "STOP_UNWITNESSED: missing declared witness: " + missing + ". Continue working until that witness exists."
			sessionctl.RecordStopWitnessNext(r.cfg.trace, continuation)
			r.messages = append(r.messages, Message{Role: RoleUser, Content: continuation})
			r.cfg.emitProgress(ProgressEvent{Kind: ProgressTurnDone, Turn: turn + 1})
			return Message{}, armTurnContinue, nil
		}
	}
	// A final answer cannot confirm a pending speculation, so squash it before returning.
	r.speculation.resolve(ctx, nil, r.metrics)
	r.metrics.FinalAnswer = asst.Content
	r.cfg.emitProgress(ProgressEvent{Kind: ProgressTurnDone, Turn: turn + 1})
	r.finalizeFak()
	return Message{}, armTurnStop, nil
}

// dispatchToolCalls adjudicates and admits every tool call from one assistant turn.
func (r *armRunner) dispatchToolCalls(ctx context.Context, turn int, asst Message) (bool, error) {
	// Resolve a suspended speculation against the first authoritative call in this turn.
	r.speculation.resolve(ctx, authoritativeCall(asst.ToolCalls[0]), r.metrics)
	var turnResults []*abi.Result
	for _, tc := range asst.ToolCalls {
		if r.stopTerminated() {
			return true, nil
		}
		if reason := r.cfg.debitToolCall(); reason != "" {
			r.metrics.StoppedBySession = reason
			r.finalizeFak()
			return true, nil
		}
		r.metrics.ToolCalls++
		tool := tc.Function.Name
		rawArgs := tc.Function.Arguments
		var content string
		ev := traceEvent{Turn: turn + 1, Arm: r.metrics.Arm, Tool: tool, RawArgs: rawArgs}
		r.cfg.emitProgress(ProgressEvent{Kind: ProgressToolStarted, Turn: turn + 1, CallID: tc.ID, Tool: tool})
		switch {
		case r.cfg.dropMidflightCall(tc.ID, turn+1):
			content = ToolReceipt{
				Status:      ToolResultSkipped,
				Reason:      "CALL_DROPPED_BY_OPERATOR",
				Disposition: "RETRYABLE",
				Fix:         "the operator dropped this call mid-flight; re-issue it if it is still needed",
				Detail:      "skipped before dispatch by a mid-flight drop-pending-call verb; never dispatched",
			}.JSON()
			ev.Verdict = "DROPPED"
			ev.By = "operator"
			ev.Note = "DROPPED by a mid-flight drop-pending-call verb (skipped before dispatch)"
		case r.cfg.constraintDenied(tool, &content, &ev):
		case r.speculation.barWrite(tool, r.metrics):
			content = ToolReceipt{
				Status:      ToolResultSkipped,
				Reason:      "WRITE_BARRED",
				Disposition: "RETRYABLE",
				Fix:         "re-issue this write after the authoritative read it depends on has actually run",
				Detail:      "held behind an unconfirmed speculative read (squashed); never dispatched",
			}.JSON()
			ev.Verdict = "BARRED"
			ev.By = "write-barrier"
			ev.Note = "BARRED by the before-consumption write barrier (dependent speculation squashed)"
		case r.fak:
			engine, routeErr := r.cfg.resolveCallEngine(tool, rawArgs, metaFor(tool))
			if routeErr != nil {
				detail, _ := json.Marshal(map[string]string{"error": routeErr.Error()})
				content = string(detail)
				ev.Verdict = "route-error"
				ev.By = "route-accounts"
				ev.Note = "ROUTE REFUSED (fail-loud): " + routeErr.Error()
			} else {
				content, ev = execViaKernel(ctx, r.kernel, tool, rawArgs, engine, ev, r.cfg.principal)
				content, ev = r.cfg.parkEscalatedDeny(ctx, r.kernel, tool, rawArgs, engine, content, ev)
			}
		default:
			content, ev = execNaive(tool, rawArgs, r.metrics, ev)
		}
		r.cfg.emitProgress(ProgressEvent{
			Kind: ProgressCallAdjudicated, Turn: turn + 1, CallID: tc.ID, Tool: tool,
			Verdict: ev.Verdict, Reason: ev.Reason,
		})
		if r.log != nil {
			*r.log = append(*r.log, ev)
		}
		if strings.Contains(strings.ToLower(content), "ignore previous instructions") {
			r.metrics.InjectionInContext = true
		}
		if tool == toolBook && strings.Contains(content, "confirmation") && !strings.Contains(content, `"error"`) {
			r.metrics.TaskCompleted = true
		}
		if recordRepeatedFailure(&r.repeatedFailures, tool, rawArgs, content) {
			return false, fmt.Errorf("REPEATED_IDENTICAL_TOOL_FAILURE: tool %s failed three consecutive times", tool)
		}
		r.messages = append(r.messages, Message{Role: RoleTool, ToolCallID: tc.ID, Name: tool, Content: content})
		r.cfg.emitProgress(ProgressEvent{
			Kind: ProgressResultAdmitted, Turn: turn + 1, CallID: tc.ID, Tool: tool,
			Taint: admittedTaint(ev, content), Summary: progressResultSummary(tool, content),
		})
		if r.speculation != nil {
			turnResults = append(turnResults, &abi.Result{
				Call:    &abi.ToolCall{Tool: tool},
				Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(content), Len: int64(len(content))},
				Status:  abi.StatusOK,
			})
		}
	}
	r.cfg.emitProgress(ProgressEvent{Kind: ProgressTurnDone, Turn: turn + 1})
	r.speculation.disarm()
	if r.speculation != nil && len(asst.ToolCalls) > 0 {
		r.speculation.speculate(ctx, turn, asst.ToolCalls[len(asst.ToolCalls)-1].Function.Name, turnResults, r.metrics)
	}
	return false, nil
}

func (r *armRunner) finalizeFak() {
	if r.fak {
		finalizeFak(r.kernel, r.metrics)
	}
}
