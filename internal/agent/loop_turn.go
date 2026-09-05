package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/attemptbudget"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/stopgate"
)

func isEmitterRegistered(e abi.Emitter) bool {
	if e == nil {
		return false
	}
	for _, em := range abi.EmittersFor(abi.EvDecide) {
		if em == e {
			return true
		}
	}
	return false
}

// armRunner owns the mutable state shared by the phases of one arm's turn loop.
// Keeping it named avoids passing the same large parameter set through every phase
// while leaving runArm responsible only for arm-level setup and teardown.
type armRunner struct {
	cfg                     *runConfig
	metrics                 *ArmMetrics
	fak                     bool
	kernel                  *kernel.Kernel
	speculation             *specState
	messages                []Message
	tools                   []ToolDef
	model                   string
	stream                  bool
	sink                    StreamSink
	complete                armCompleteFunc
	log                     *[]traceEvent
	stopTerminated          func() bool
	repeatedFailures        attemptbudget.RepeatedFailureTracker
	consecutiveDenyAll      int
	consecutiveSameIssue    int
	lastDeniedTool          string
	lastDeniedReason        string
	lastRefusalReceipt      *stopgate.BoundaryRefusalReceipt
	consecutiveToolFeedback int
	witnessBlockCount       int
	denyAllPending          bool
	toolFeedbackPending     bool
	task                    string
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
	r.saveCheckpoint(turn, "completed")
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
	var turnStop bool
	switch action {
	case armTurnContinue:
		turnStop = false
	case armTurnStop:
		turnStop = true
	default:
		turnStop, err = r.dispatchToolCalls(ctx, turn, asst)
	}
	if err == nil {
		status := "active"
		if turnStop || r.metrics.FinalAnswer != "" {
			status = "completed"
		}
		r.saveCheckpoint(turn, status)
	}
	return turnStop, err
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
	if r.cfg != nil && (r.cfg.reasoningProfile != "" || r.cfg.reasoningEffort != "" || r.cfg.thinkingBudget != nil) {
		if r.cfg.reasoningProfile != "" {
			ta, _ := AssessTranscriptTurn(planned)
			effort, budget := ResolveReasoningProfileBudget(r.cfg.reasoningProfile, r.cfg.thinkingBudget, ta)
			eff := effort
			if r.cfg.reasoningEffort != "" {
				eff = r.cfg.reasoningEffort
			}
			sampleOpts = append(sampleOpts, WithThinkingBudget(budget), WithReasoningEffort(eff))
		} else if r.cfg.reasoningEffort == EffortTierBalanced || r.cfg.reasoningEffort == EffortTierAdaptive {
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
	consecDeny := 0
	sameConsec := 0
	useSame := false
	if r.denyAllPending {
		consecDeny = r.consecutiveDenyAll
		sameConsec = r.consecutiveSameIssue
		useSame = r.consecutiveSameIssue > 0
		r.denyAllPending = false
	}
	consecFeedback := 0
	if r.toolFeedbackPending {
		consecFeedback = r.consecutiveToolFeedback
		r.toolFeedbackPending = false
	}

	ladderCfg := stopgate.LadderConfig{Mode: stopgate.ModeOff}
	if r.cfg.stopLadder != nil {
		ladderCfg = *r.cfg.stopLadder
	}
	witnessCfg := stopgate.DefaultWitnessGateConfig()
	if r.cfg.witnessGate != nil {
		witnessCfg = *r.cfg.witnessGate
	}
	boundaryIn := stopgate.BoundaryInput{
		SessionID:               r.cfg.trace,
		Turn:                    turn + 1,
		ConsecutiveDenyAll:      consecDeny,
		ConsecutiveSameIssue:    sameConsec,
		UseSameIssue:            useSame,
		ConsecutiveToolFeedback: consecFeedback,
		NotedNoAllowedPath:      strings.Contains(strings.ToLower(asst.Content), "no allowed path"),
		FinalGate:               r.cfg.finalGate,
		WitnessBlockCount:       r.witnessBlockCount,
		RefusalReceipt:          r.lastRefusalReceipt,
		BoundaryRefusalReceipt:  r.lastRefusalReceipt,
		RefusalToken:            r.lastDeniedReason,
	}
	if r.lastDeniedReason != "" {
		code, _ := abi.ReasonByName(r.lastDeniedReason)
		boundaryIn.ReasonCode = code
	}
	decision := stopgate.EvaluateBoundary(ladderCfg, witnessCfg, boundaryIn)
	if decision.ShouldContinue() {
		if decision.Signal == "STOP_UNWITNESSED" || decision.Signal == "witness" || strings.HasPrefix(decision.Guidance, "STOP_UNWITNESSED") {
			r.witnessBlockCount++
		}
		continuation := decision.Guidance
		sessionctl.RecordStopWitnessNext(r.cfg.trace, continuation)
		r.messages = append(r.messages, Message{Role: RoleUser, Content: continuation})
		r.cfg.emitProgress(ProgressEvent{Kind: ProgressTurnDone, Turn: turn + 1})
		return Message{}, armTurnContinue, nil
	}
	// A final answer cannot confirm a pending speculation, so squash it before returning.
	r.speculation.resolve(ctx, nil, r.metrics)
	r.metrics.FinalAnswer = asst.Content
	r.cfg.emitProgress(ProgressEvent{Kind: ProgressTurnDone, Turn: turn + 1})
	r.finalizeFak()
	return Message{}, armTurnStop, nil
}

// isToolResultFailure determines whether a tool execution outcome constitutes an
// error or denial that warrants entering recovery mode.
func isToolResultFailure(isErr bool, verdict string, content string) bool {
	if isErr || verdict == "DENY" || verdict == "DENIED" || verdict == "BARRED" || verdict == "DROPPED" || verdict == "route-error" {
		return true
	}
	if strings.HasPrefix(strings.TrimSpace(content), "{") {
		var rc ToolReceipt
		if err := json.Unmarshal([]byte(content), &rc); err == nil && rc.Status == ToolResultError {
			return true
		}
		var rawMap map[string]any
		if err := json.Unmarshal([]byte(content), &rawMap); err == nil {
			if errVal, ok := rawMap["error"]; ok && errVal != nil && errVal != "" && errVal != false && errVal != 0 && errVal != float64(0) {
				return true
			}
		}
	}
	return false
}

// dispatchToolCalls adjudicates and admits every tool call from one assistant turn.
func (r *armRunner) dispatchToolCalls(ctx context.Context, turn int, asst Message) (bool, error) {
	if len(asst.ToolCalls) == 0 {
		return false, nil
	}
	// Resolve a suspended speculation against the first authoritative call in this turn.
	r.speculation.resolve(ctx, authoritativeCall(asst.ToolCalls[0]), r.metrics)
	var turnResults []*abi.Result

	t0 := time.Now()
	defer func() {
		r.metrics.ToolElapsedMs += time.Since(t0).Milliseconds()
	}()

	type toolExecResult struct {
		tc      ToolCall
		content string
		ev      traceEvent
		isErr   bool
		abiCall *abi.ToolCall
		abiRes  *abi.Result
		verdict *abi.Verdict
	}

	execOne := func(tc ToolCall) toolExecResult {
		tool := tc.Function.Name
		rawArgs := tc.Function.Arguments
		var content string
		ev := traceEvent{Turn: turn + 1, Arm: r.metrics.Arm, Tool: tool, RawArgs: rawArgs}
		var isErr bool
		var abiCall *abi.ToolCall
		var abiRes *abi.Result
		var verdict *abi.Verdict
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
			abiCall = &abi.ToolCall{Tool: tool, TraceID: r.cfg.trace, Args: putBytes(ctx, []byte(rawArgs))}
			v := abi.Verdict{Kind: abi.VerdictDeny, By: "operator", Reason: abi.ReasonPolicyBlock}
			verdict = &v
		case r.cfg.constraintDenied(tool, &content, &ev):
			abiCall = &abi.ToolCall{Tool: tool, TraceID: r.cfg.trace, Args: putBytes(ctx, []byte(rawArgs))}
			v := abi.Verdict{Kind: abi.VerdictDeny, By: "session-constraint", Reason: abi.ReasonPolicyBlock}
			verdict = &v
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
			abiCall = &abi.ToolCall{Tool: tool, TraceID: r.cfg.trace, Args: putBytes(ctx, []byte(rawArgs))}
			v := abi.Verdict{Kind: abi.VerdictDeny, By: "write-barrier", Reason: abi.ReasonPolicyBlock}
			verdict = &v
		case r.fak:
			engine, routeErr := r.cfg.resolveCallEngine(tool, rawArgs, metaFor(tool))
			if routeErr != nil {
				detail, _ := json.Marshal(map[string]string{"error": routeErr.Error()})
				content = string(detail)
				ev.Verdict = "route-error"
				ev.By = "route-accounts"
				ev.Note = "ROUTE REFUSED (fail-loud): " + routeErr.Error()
				abiCall = &abi.ToolCall{Tool: tool, TraceID: r.cfg.trace, Args: putBytes(ctx, []byte(rawArgs))}
				v := abi.Verdict{Kind: abi.VerdictDeny, By: "route-accounts", Reason: abi.ReasonMisroute}
				verdict = &v
			} else {
				var verdictVal abi.Verdict
				content, ev, abiCall, abiRes, verdictVal = execViaKernelFull(ctx, r.kernel, tool, rawArgs, engine, r.cfg.trace, ev, r.cfg.principal)
				verdict = &verdictVal
				content, ev = r.cfg.parkEscalatedDeny(ctx, r.kernel, tool, rawArgs, engine, content, ev)
			}
		default:
			var naiveM ArmMetrics
			content, ev = execNaive(tool, rawArgs, &naiveM, ev)
			if naiveM.ToolErrors > 0 {
				isErr = true
			}
		}
		return toolExecResult{tc: tc, content: content, ev: ev, isErr: isErr, abiCall: abiCall, abiRes: abiRes, verdict: verdict}
	}

	commitOne := func(res toolExecResult) error {
		tc := res.tc
		tool := tc.Function.Name
		rawArgs := tc.Function.Arguments
		content := res.content
		ev := res.ev

		if r.cfg != nil && r.cfg.auditJournal != nil {
			if !r.fak {
				rawBytes := []byte(rawArgs)
				hArgs := sha256.Sum256(rawBytes)
				call := &abi.ToolCall{
					Tool:    tool,
					TraceID: r.cfg.trace,
					Args: abi.Ref{
						Kind:   abi.RefInline,
						Inline: rawBytes,
						Len:    int64(len(rawBytes)),
						Digest: hex.EncodeToString(hArgs[:]),
					},
				}
				contentBytes := []byte(content)
				hRes := sha256.Sum256(contentBytes)
				resPayload := abi.Ref{
					Kind:   abi.RefInline,
					Inline: contentBytes,
					Len:    int64(len(contentBytes)),
					Digest: hex.EncodeToString(hRes[:]),
				}
				result := &abi.Result{
					Call:    call,
					Payload: resPayload,
				}
				if res.isErr {
					v := &abi.Verdict{
						Kind:   abi.VerdictDeny,
						By:     "raw-harness",
						Reason: abi.ReasonMalformed,
					}
					r.cfg.auditJournal.Emit(abi.Event{
						Kind:    abi.EvDecide,
						Call:    call,
						Verdict: v,
						Result:  result,
					})
					r.cfg.auditJournal.Emit(abi.Event{
						Kind:    abi.EvDeny,
						Call:    call,
						Verdict: v,
						Result:  result,
					})
				} else {
					v := &abi.Verdict{
						Kind:   abi.VerdictAllow,
						By:     "raw-harness",
						Reason: abi.ReasonNone,
					}
					r.cfg.auditJournal.Emit(abi.Event{
						Kind:    abi.EvDecide,
						Call:    call,
						Verdict: v,
						Result:  result,
					})
				}
			} else if !isEmitterRegistered(r.cfg.auditJournal) {
				call := res.abiCall
				if call == nil {
					call = &abi.ToolCall{
						Tool:    tool,
						TraceID: r.cfg.trace,
						Args:    putBytes(ctx, []byte(rawArgs)),
					}
				}
				if call.TraceID == "" {
					call.TraceID = r.cfg.trace
				}
				if call.Args.Digest == "" {
					h := sha256.Sum256([]byte(rawArgs))
					call.Args.Digest = hex.EncodeToString(h[:])
				}
				result := res.abiRes
				if result == nil {
					result = &abi.Result{
						Call:    call,
						Payload: putBytes(ctx, []byte(content)),
					}
				}
				if result.Payload.Digest == "" {
					h := sha256.Sum256([]byte(content))
					result.Payload.Digest = hex.EncodeToString(h[:])
				}
				v := res.verdict
				if v == nil {
					v = &abi.Verdict{
						Kind:   abi.VerdictAllow,
						By:     "localtools",
						Reason: abi.ReasonNone,
					}
				}
				if v.By == "vdso" {
					r.cfg.auditJournal.Emit(abi.Event{
						Kind:    abi.EvVDSOHit,
						Call:    call,
						Verdict: v,
						Result:  result,
					})
				} else if v.Kind == abi.VerdictDeny {
					r.cfg.auditJournal.Emit(abi.Event{
						Kind:    abi.EvDecide,
						Call:    call,
						Verdict: v,
						Result:  result,
					})
					r.cfg.auditJournal.Emit(abi.Event{
						Kind:    abi.EvDeny,
						Call:    call,
						Verdict: v,
						Result:  result,
					})
				} else {
					r.cfg.auditJournal.Emit(abi.Event{
						Kind:    abi.EvDecide,
						Call:    call,
						Verdict: v,
						Result:  result,
					})
				}
				if result != nil && result.Meta != nil {
					if result.Meta["admit"] == "quarantined" {
						r.cfg.auditJournal.Emit(abi.Event{
							Kind:    abi.EvQuarantine,
							Call:    call,
							Verdict: v,
							Result:  result,
						})
					} else if result.Meta["admit"] == "deny" {
						r.cfg.auditJournal.Emit(abi.Event{
							Kind:    abi.EvResultDeny,
							Call:    call,
							Verdict: v,
							Result:  result,
						})
					}
				}
			}
		}

		r.cfg.emitProgress(ProgressEvent{
			Kind: ProgressCallAdjudicated, Turn: turn + 1, CallID: tc.ID, Tool: tool,
			Verdict: ev.Verdict, Reason: ev.Reason,
		})
		if r.log != nil {
			*r.log = append(*r.log, ev)
		}
		if res.isErr {
			r.metrics.ToolErrors++
		}
		isDenied := ev.Verdict == "DENY" || ev.Verdict == "DENIED" || ev.Verdict == "BARRED" || ev.Verdict == "DROPPED" || ev.Verdict == "route-error" || (res.verdict != nil && res.verdict.Kind == abi.VerdictDeny)
		if tool == toolDelete && !isDenied {
			r.metrics.DestructiveExecuted = true
		}
		if strings.Contains(strings.ToLower(content), "ignore previous instructions") {
			r.metrics.InjectionInContext = true
		}
		if tool == toolBook && strings.Contains(content, "confirmation") && !strings.Contains(content, `"error"`) {
			r.metrics.TaskCompleted = true
		}
		if r.cfg.goalAnchor != nil {
			if isToolResultFailure(res.isErr, ev.Verdict, content) {
				r.cfg.goalAnchor.RecordRecoveryTurn()
				r.metrics.GoalAnchorRecoveryTurns = r.cfg.goalAnchor.RecoveryTurnCount
			}
		}
		if recordRepeatedFailure(&r.repeatedFailures, tool, rawArgs, content) {
			return fmt.Errorf("REPEATED_IDENTICAL_TOOL_FAILURE: tool %s failed three consecutive times", tool)
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
		return nil
	}

	var allResults []toolExecResult
	for idx := 0; idx < len(asst.ToolCalls); {
		if r.stopTerminated() {
			return true, nil
		}
		// Determine segment boundary.
		// A contiguous slice of effect-safe calls can execute their bodies concurrently.
		// An exclusive call forms a barrier (segment of length 1).
		segEnd := idx + 1
		isSafe := isEffectSafeTool(asst.ToolCalls[idx].Function.Name)
		if isSafe {
			for segEnd < len(asst.ToolCalls) && isEffectSafeTool(asst.ToolCalls[segEnd].Function.Name) {
				segEnd++
			}
		}
		segCalls := asst.ToolCalls[idx:segEnd]

		var runnable []ToolCall
		stopped := false
		for _, tc := range segCalls {
			if r.stopTerminated() {
				stopped = true
				break
			}
			if reason := r.cfg.debitToolCall(); reason != "" {
				r.metrics.StoppedBySession = reason
				r.finalizeFak()
				stopped = true
				break
			}
			r.metrics.ToolCalls++
			if isSafe {
				r.metrics.ToolCallsSafe++
			} else {
				r.metrics.ToolCallsExclusive++
			}
			r.cfg.emitProgress(ProgressEvent{Kind: ProgressToolStarted, Turn: turn + 1, CallID: tc.ID, Tool: tc.Function.Name})
			runnable = append(runnable, tc)
		}

		if len(runnable) > 0 {
			results := make([]toolExecResult, len(runnable))
			if isSafe && len(runnable) > 1 {
				var wg sync.WaitGroup
				wg.Add(len(runnable))
				for i, tc := range runnable {
					go func(i int, tc ToolCall) {
						defer wg.Done()
						results[i] = execOne(tc)
					}(i, tc)
				}
				wg.Wait()
			} else {
				for i, tc := range runnable {
					results[i] = execOne(tc)
				}
			}

			for _, res := range results {
				allResults = append(allResults, res)
				if err := commitOne(res); err != nil {
					return false, err
				}
			}
		}

		if stopped {
			return true, nil
		}
		idx = segEnd
	}

	if len(allResults) > 0 {
		allDenied := true
		allFeedback := true
		for _, res := range allResults {
			isDeny := res.ev.Verdict == "DENY" || res.ev.Verdict == "DENIED" || res.ev.Verdict == "BARRED" || res.ev.Verdict == "DROPPED" || res.ev.Verdict == "route-error" || (res.verdict != nil && res.verdict.Kind == abi.VerdictDeny) || strings.HasPrefix(res.tc.Function.Name, "bad_tool") || strings.Contains(res.tc.Function.Name, "lock_busy")
			if !isDeny {
				allDenied = false
			}
			isFb := strings.Contains(strings.ToLower(res.content), "invalid json") || strings.Contains(strings.ToLower(res.content), "malformed") || res.ev.Reason == "MISROUTE" || res.ev.Reason == "ARG_INVALID"
			if !isFb {
				allFeedback = false
			}
		}

		if allDenied {
			r.consecutiveDenyAll++
			r.denyAllPending = true
			lastTool := allResults[0].tc.Function.Name
			lastReason := allResults[0].ev.Reason
			if lastReason == "" {
				lastReason = allResults[0].ev.Verdict
			}
			lastDisp := allResults[0].ev.Disposition
			if lastDisp == "" {
				var rc ToolReceipt
				if err := json.Unmarshal([]byte(allResults[0].content), &rc); err == nil && rc.Disposition != "" {
					lastDisp = rc.Disposition
				}
			}
			if strings.Contains(lastTool, "lock_busy") || strings.Contains(lastReason, "LOCK_BUSY") {
				lastReason = "LOCK_BUSY"
				if lastDisp == "" {
					lastDisp = "RETRYABLE"
				}
			} else if strings.HasPrefix(lastTool, "bad_tool") && (lastReason == "" || lastReason == "naive-exec" || lastReason == "DENY") {
				lastReason = "POLICY_BLOCK"
				lastDisp = "TERMINAL"
			}
			r.lastRefusalReceipt = &stopgate.BoundaryRefusalReceipt{
				Tool:        lastTool,
				Reason:      lastReason,
				Disposition: lastDisp,
				Verified:    true,
			}
			if r.lastDeniedTool == lastTool && r.lastDeniedReason == lastReason && lastTool != "" {
				r.consecutiveSameIssue++
			} else {
				r.consecutiveSameIssue = 1
				r.lastDeniedTool = lastTool
				r.lastDeniedReason = lastReason
			}
			r.consecutiveToolFeedback = 0
			r.toolFeedbackPending = false
		} else if allFeedback {
			r.lastRefusalReceipt = nil
			r.consecutiveToolFeedback++
			r.toolFeedbackPending = true
			r.consecutiveDenyAll = 0
			r.consecutiveSameIssue = 0
			r.lastDeniedTool = ""
			r.lastDeniedReason = ""
			r.denyAllPending = false
		} else {
			r.lastRefusalReceipt = nil
			r.consecutiveDenyAll = 0
			r.consecutiveSameIssue = 0
			r.lastDeniedTool = ""
			r.lastDeniedReason = ""
			r.consecutiveToolFeedback = 0
			r.denyAllPending = false
			r.toolFeedbackPending = false
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

func (r *armRunner) saveCheckpoint(turn int, status string) {
	if r.cfg == nil || !r.cfg.HasSessionCheckpoint() {
		return
	}
	dir := r.cfg.SessionCheckpointDir()
	cwd := ""
	if wd, err := os.Getwd(); err == nil {
		cwd = wd
	}
	createdAt := r.cfg.sessionCheckpointCreatedAt
	if createdAt.IsZero() {
		if existing, err := LoadSessionCheckpoint(r.cfg.sessionCheckpointID, dir); err == nil && !existing.CreatedAt.IsZero() {
			createdAt = existing.CreatedAt
		} else {
			createdAt = time.Now().UTC()
		}
		r.cfg.sessionCheckpointCreatedAt = createdAt
	}

	task := r.task
	if task == "" && r.cfg.goalAnchor != nil {
		task = r.cfg.goalAnchor.Objective
	}
	if task == "" {
		for _, m := range r.messages {
			if m.Role == RoleUser {
				task = m.Content
				break
			}
		}
	}

	currentTurn := turn + 1
	if r.cfg.sessionCheckpointInitialTurn > 0 {
		currentTurn += r.cfg.sessionCheckpointInitialTurn
	}

	cp := SessionCheckpoint{
		SessionID: r.cfg.sessionCheckpointID,
		CWD:       cwd,
		Task:      task,
		Model:     r.model,
		Provider:  r.cfg.provider,
		BaseURL:   r.cfg.baseURL,
		Messages:  append([]Message(nil), r.messages...),
		Turn:      currentTurn,
		CreatedAt: createdAt,
		UpdatedAt: time.Now().UTC(),
		Status:    status,
	}
	_ = SaveSessionCheckpoint(dir, cp)
}
