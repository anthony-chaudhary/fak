package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/codetools"
	"github.com/anthony-chaudhary/fak/internal/fusedturn"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// plannerKind classifies the /v1/chat/completions backend for the /healthz
// "planner" field, so an operator (or a liveness probe) can tell at a glance
// whether a served chat is a real model or the deterministic offline mock:
//
//   - "mock"     the scripted offline fallback (no --base-url, no --gguf) — the
//     same condition New warns about loudly at boot.
//   - "proxy"    one live upstream provider (fak serve --base-url).
//   - "replica"  a static round-robin live upstream fleet.
//   - "inkernel" the model fused into the kernel (fak serve --gguf).
//   - "dual"     the in-kernel model alongside a live upstream proxy
//     (--gguf AND --base-url); the local model id routes in-kernel, the rest proxy.
//
// A nil or unrecognized planner reports "unknown" rather than masquerading as a
// real backend.
func plannerKind(p agent.Planner) string {
	switch p.(type) {
	case *agent.MockPlanner:
		return "mock"
	case *agent.HTTPPlanner:
		return "proxy"
	case *ReplicaDispatch:
		return "replica"
	case *agent.InKernelPlanner:
		return "inkernel"
	case *DualPlanner:
		return "dual"
	default:
		return "unknown"
	}
}

func engineRegistered(id string) bool {
	for _, e := range abi.EngineIDs() {
		if e == id {
			return true
		}
	}
	return false
}

// adjudicate runs ONLY the adjudicator chain (k.Decide) over a (tool, rawArgs)
// pair and returns the pre-execution verdict — no engine dispatch, no pending
// submission state, nothing to leak. This is what a client-side executor asks for
// before it runs a tool. On a TRANSFORM (grammar repair) it resolves the rewritten
// args so the client can run the canonical form; that repaired-args string is the
// second return.
func (s *Server) adjudicate(ctx context.Context, tool, rawArgs string, readOnly bool, witness, traceID string) (wv WireVerdict, repaired string, err error) {
	return s.adjudicateWithSeq(ctx, tool, rawArgs, readOnly, witness, traceID, 0)
}

func adjudicateReceipt(wv WireVerdict, err error, duration time.Duration) AdjudicateReceipt {
	receipt := AdjudicateReceipt{
		Schema:     "fak-adjudicate-receipt/1",
		Outcome:    adjudicateOutcome(wv, err),
		DurationNS: duration.Nanoseconds(),
		Execution:  "not_executed",
		Provenance: "kernel_decide",
	}
	if err != nil {
		receipt.Error = &AdjudicateReceiptError{Code: adjudicateErrorCode(err), Source: "gateway"}
	}
	return receipt
}

func adjudicateOutcome(wv WireVerdict, err error) string {
	if err != nil {
		return "failed"
	}
	switch wv.Kind {
	case "ALLOW":
		return "allowed"
	case "DENY":
		return "denied"
	case "TRANSFORM":
		return "transformed"
	case "REQUIRE_WITNESS":
		return "witness_required"
	default:
		// The folded kernel normally lowers restrictive unknown/defer verdicts to
		// DENY. Fail closed if a future kind reaches this projection.
		return "denied"
	}
}

func adjudicateErrorCode(err error) string {
	if err != nil && strings.Contains(err.Error(), "missing tool name") {
		return "invalid_arguments"
	}
	return "adjudication_failed"
}

func (s *Server) adjudicateWithSeq(ctx context.Context, tool, rawArgs string, readOnly bool, witness, traceID string, callSeq uint64) (wv WireVerdict, repaired string, err error) {
	start := time.Now()
	opTrace, opTool := traceID, tool
	defer func() {
		dur := time.Since(start)
		s.metrics.observeOperation("adjudicate", wv, err, dur)
		s.logGatewayOperation("adjudicate", opTrace, opTool, wv, err, dur)
	}()
	if nativeToolName(tool) == remoteMaintenanceTool {
		readOnly = false
	}
	tc, err := s.buildCall(ctx, tool, rawArgs, readOnly, witness, traceID)
	if err != nil {
		return WireVerdict{}, "", err
	}
	if callSeq != 0 {
		tc.SeqNo = callSeq
	}
	opTrace, opTool = tc.TraceID, tc.Tool
	if nativeToolName(tc.Tool) == remoteMaintenanceTool {
		wv, err = s.adjudicateRemoteMaintenance(ctx, tc)
		return wv, "", err
	}
	v := s.k.Decide(ctx, tc)
	wv = renderVerdict(v, nil)
	effectiveArgs := rawArgs
	if v.Kind == abi.VerdictTransform {
		if tp, ok := v.Payload.(abi.TransformPayload); ok {
			repaired = string(resolveBytes(ctx, tp.NewArgs))
			effectiveArgs = repaired
		}
	}
	if v.Kind == abi.VerdictAllow || v.Kind == abi.VerdictTransform {
		if leaseVerdict, refused := s.adjudicateLeaseAdmission(ctx, tc.Tool, effectiveArgs, readOnly); refused {
			if operatorApprovedLeaseOverride(v, &wv, leaseVerdict) {
				return wv, repaired, nil
			}
			offerOperatorApproval(&leaseVerdict, tc.Tool, effectiveArgs)
			if s.workspaceAdmissionPermissive && leaseVerdict.Reason == abi.ReasonName(abi.ReasonDefaultDeny) {
				// Preserve TRANSFORM so clients still apply the repaired arguments.
				if wv.Detail == nil {
					wv.Detail = make(map[string]string)
				}
				wv.Detail["kernel_by"] = wv.By
				wv.By = "lease-admission-permissive"
				wv.Detail["workspace_admission"] = "permissive"
				wv.Detail["would_deny"] = leaseVerdict.Reason
				wv.Detail["would_deny_by"] = leaseVerdict.By
				wv.Detail["would_deny_claim"] = leaseVerdict.Detail["claim"]
			} else {
				wv = leaseVerdict
			}
		}
	}
	return wv, repaired, nil
}

// (k.Syscall: adjudicate -> vDSO -> dispatch to the registered engine ->
// context-MMU admit) and returns the rendered verdict plus the admitted result.
// This is the self-contained path: fak's registered engine produces the result,
// and a quarantined/poisoned result is already paged out before the gateway sees
// it.
func (s *Server) syscall(ctx context.Context, tool, rawArgs string, readOnly bool, witness, traceID string) (wv WireVerdict, env *ResultEnvelope, err error) {
	start := time.Now()
	opTrace, opTool := traceID, tool
	defer func() {
		dur := time.Since(start)
		s.metrics.observeOperation("syscall", wv, err, dur)
		s.logGatewayOperation("syscall", opTrace, opTool, wv, err, dur)
	}()
	if nativeToolName(tool) == remoteMaintenanceTool {
		readOnly = false
	}
	tc, err := s.buildCall(ctx, tool, rawArgs, readOnly, witness, traceID)
	if err != nil {
		return WireVerdict{}, nil, err
	}
	opTrace, opTool = tc.TraceID, tc.Tool
	if nativeToolName(tc.Tool) == remoteMaintenanceTool {
		// Remote maintenance proves a Prepared, trusted remote scope inside its
		// native handler before it omits the local-workspace footprint gate. No
		// other native or external tool takes this route.
		return s.syscallRemoteMaintenance(ctx, tc)
	}
	if refusal, refused := s.strictInitialWorkspaceExecution(ctx, tc, readOnly); refused {
		return refusal, nil, nil
	}

	if wv, env, handled, err := s.syscallNative(ctx, tc, readOnly); handled {
		return wv, env, err
	}

	// In-process read optimization (#11035): promote pure read-only shell commands (cat, head, tail)
	// directly without spawning subprocess shells.
	if promotedEnv, promotedWv, ok := s.PromoteShellReadToInProcess(ctx, tool, rawArgs, traceID); ok {
		return promotedWv, promotedEnv, nil
	}

	// Ensemble fan-out (issue #597): a multi-member routing Plan runs each member as
	// its OWN adjudicated kernel call and folds the outputs. buildCall left tc.Engine
	// unset for an ensemble (routeEngine returns "" rather than collapse to one member);
	// dispatchEnsemble re-reads the same routing decision and submits N independent
	// calls. The single-model PICK below is byte-for-byte the pre-#597 path.
	if plan, ok := s.ensemblePlan(tc.Tool, readOnly, tc.Meta); ok {
		wv, env, err = s.dispatchEnsembleWithReadOnly(ctx, tc, plan, readOnly)
		if err != nil {
			return wv, env, err
		}
		// The scripted-fold witness rides the SAME execute floor as a single-model call
		// (#2858): an ensemble is still a scripted RPC result folded into the parent.
		wv = witnessScriptedFold(tc.Tool, env, wv)
		return wv, env, nil
	}
	lookupCtx, lookupReceipt := vdso.WithLookupReceipt(ctx)
	r, v, admissionDetails, refusal := s.workspaceKernelSyscall(lookupCtx, tc, readOnly)
	if refusal != nil {
		return *refusal, nil, nil
	}
	defer func() { annotateWorkspaceExecution(&wv, admissionDetails) }()
	if lookupReceipt.Matches(r) && v.Kind == abi.VerdictAllow && r.Status == abi.StatusOK && resultMeta(r)["admit"] != "quarantined" {
		FeatureActivationTrackerFromContext(ctx).RecordActivation(FeatureVDSO, FeatureOutcomeUsed)
	}
	if v.Kind == abi.VerdictAllow && isCapabilitiesTool(tc.Tool) {
		var capReq CapabilitiesRequest
		argsBytes := resolveBytes(ctx, tc.Args)
		if len(argsBytes) > 0 {
			_ = json.Unmarshal(argsBytes, &capReq)
		}
		capResp, capErr := s.capabilities(capReq)
		if capErr == nil {
			capBytes, _ := json.Marshal(capResp)
			r = &abi.Result{
				Status: abi.StatusOK,
				Payload: abi.Ref{
					Kind:   abi.RefInline,
					Inline: capBytes,
				},
				Meta: map[string]string{
					"engine": "fak-mcp",
				},
			}
		} else {
			r = &abi.Result{
				Status: abi.StatusError,
				Payload: abi.Ref{
					Kind:   abi.RefInline,
					Inline: []byte(capErr.Error()),
				},
				Meta: map[string]string{
					"engine": "fak-mcp",
				},
			}
		}
	}
	s.rememberOriginSeq(tc.TraceID, tc.Tool, string(resolveBytes(ctx, tc.Args)), tc.SeqNo)
	wv = renderVerdict(v, resultMeta(r))
	if r != nil {
		env = &ResultEnvelope{
			Status:  statusName(r.Status),
			Content: string(resolveBytes(ctx, r.Payload)),
			Meta:    r.Meta,
		}
		// Record the account binding this call resolved through (#2528): account id,
		// provider kind, upstream model, the account-resolved engine route, and the
		// credential env NAME — no secret values. No-op without a roster.
		s.recordRouteAccount(env, tc.Tool, readOnly, tc.Meta)
	}
	// Witness the subagent fold on the scripted RPC execute path (#2858, epic #2834
	// Track F): a "zero-context-cost" scripted call that spawns a subagent and folds its
	// return still crosses the SAME fold-witness the model-facing admit path applies, so
	// an unwitnessed done-claim is demoted RESIDUAL/LOOP_DONE_UNWITNESSED and the envelope
	// carries the demotion breadcrumb the parent script checks before folding.
	wv = witnessScriptedFold(tc.Tool, env, wv)
	return wv, env, nil
}

// strictInitialWorkspaceExecution retains the pre-existing strict mutation fence.
func (s *Server) strictInitialWorkspaceExecution(ctx context.Context, call *abi.ToolCall, readOnly bool) (WireVerdict, bool) {
	_, _, mutating, _ := adjudicateRawWriteTargets(call.Tool, string(resolveBytes(ctx, call.Args)), readOnly)
	if mutating && !s.workspaceAdmissionPermissive {
		return leaseAdmissionRefusal(abi.ReasonDefaultDeny, leaseFailureAtomicGuardUnavailable, "atomic workspace lease guard is unavailable for mutating syscall"), true
	}
	return WireVerdict{}, false
}

// prepareWorkspaceExecution inspects the final call already bound by Submit (or
// a native Decide that refuses transforms), without performing another fold.
func (s *Server) prepareWorkspaceExecution(ctx context.Context, call *abi.ToolCall, readOnly bool) (map[string]string, WireVerdict, bool) {
	rawArgs := string(resolveBytes(ctx, call.Args))
	_, _, mutating, _ := adjudicateRawWriteTargets(call.Tool, rawArgs, readOnly)
	if !mutating {
		return nil, WireVerdict{}, false
	}
	atomicRefusal := leaseAdmissionRefusal(abi.ReasonDefaultDeny, leaseFailureAtomicGuardUnavailable, "atomic workspace lease guard is unavailable for mutating syscall")
	if !s.workspaceAdmissionPermissive {
		return nil, atomicRefusal, true
	}
	details := map[string]string{
		"workspace_admission":    "permissive",
		"would_deny":             atomicRefusal.Reason,
		"would_deny_by":          atomicRefusal.By,
		"would_deny_claim":       atomicRefusal.Detail["claim"],
		"atomic_workspace_guard": "unavailable",
	}
	if snapshot, refused := s.adjudicateLeaseAdmission(ctx, call.Tool, rawArgs, readOnly); refused {
		if snapshot.Reason != abi.ReasonName(abi.ReasonDefaultDeny) {
			return nil, snapshot, true
		}
		details["snapshot_would_deny"] = snapshot.Reason
		details["snapshot_would_deny_claim"] = snapshot.Detail["claim"]
	}
	return details, WireVerdict{}, false
}

// workspaceKernelSyscall checks the very submission that Reap will dispatch.
func (s *Server) workspaceKernelSyscall(ctx context.Context, call *abi.ToolCall, readOnly bool) (*abi.Result, abi.Verdict, map[string]string, *WireVerdict) {
	handle, verdict := s.k.Submit(ctx, call)
	if isExecCommandRequest(call) {
		if verdict.Kind != abi.VerdictAllow {
			s.k.DiscardSubmission(handle)
			if verdict.Kind == abi.VerdictDeny || verdict.Kind == abi.VerdictRequireWitness {
				refusal := renderVerdict(verdict, nil)
				return nil, verdict, nil, &refusal
			}
			refusal := execCommandRefusal("exec-command-policy", "the command request was not explicitly allowed")
			return nil, verdict, nil, &refusal
		}
		s.k.DiscardSubmission(handle)
		adapted, err := adaptedExecCommandCall(ctx, call)
		if err != nil {
			refusal := execCommandRefusal("exec-command-arguments", "the command arguments do not match the supported contract")
			return nil, verdict, nil, &refusal
		}
		call = adapted
		handle, verdict = s.k.Submit(ctx, call)
		if verdict.Kind != abi.VerdictAllow {
			s.k.DiscardSubmission(handle)
			if verdict.Kind == abi.VerdictDeny || verdict.Kind == abi.VerdictRequireWitness {
				refusal := renderVerdict(verdict, nil)
				return nil, verdict, nil, &refusal
			}
			refusal := execCommandRefusal("exec-command-policy", "the fenced command request was not explicitly allowed")
			return nil, verdict, nil, &refusal
		}
		if call.Engine != codetools.EngineBash {
			s.k.DiscardSubmission(handle)
			refusal := execCommandRefusal("exec-command-route", "the fenced command engine did not claim the request")
			return nil, verdict, nil, &refusal
		}
	}
	var details map[string]string
	if verdict.Kind == abi.VerdictAllow || verdict.Kind == abi.VerdictTransform || verdict.Kind == abi.VerdictDefer {
		var refusal WireVerdict
		var refused bool
		details, refusal, refused = s.prepareWorkspaceExecution(ctx, call, readOnly)
		if refused {
			s.k.DiscardSubmission(handle)
			return nil, verdict, nil, &refusal
		}
	}
	result, err := s.k.Reap(ctx, handle)
	if err != nil {
		result = &abi.Result{Call: call, Status: abi.StatusError, Meta: map[string]string{"error": err.Error()}}
	}
	if isAdaptedExecCommand(call) && result != nil && result.Status == abi.StatusOK {
		if !validExecCommandResult(ctx, result) {
			refusal := execCommandRefusal("exec-command-result", "the command engine returned a mismatched result contract")
			return nil, verdict, details, &refusal
		}
		meta := make(map[string]string, len(result.Meta)+3)
		for key, value := range result.Meta {
			meta[key] = value
		}
		meta["requested_tool"] = call.Meta["requested_tool"]
		meta["requested_route"] = call.Meta["requested_tool"]
		meta["executed_route"] = codetools.EngineBash
		meta["admitted_result_type"] = execCommandResultType
		result.Meta = meta
		s.recordExecCommandRoute(call)
	}
	return result, verdict, details, nil
}

const execCommandResultType = "codetools.bash-result/1"

func execCommandRefusal(by, claim string) WireVerdict {
	return renderVerdict(abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  abi.ReasonPolicyBlock,
		By:      by,
		Payload: abi.WitnessPayload{Claim: claim},
	}, nil)
}

func isAdaptedExecCommand(call *abi.ToolCall) bool {
	return call != nil && call.Tool == codetools.ToolBash && call.Meta != nil && call.Meta["exec_command_contract"] == execCommandResultType
}

func isExecCommandRequest(call *abi.ToolCall) bool {
	return call != nil && call.Tool == "exec_command" && call.Meta != nil && call.Meta["exec_command_contract"] == execCommandResultType
}

func validExecCommandResult(ctx context.Context, result *abi.Result) bool {
	if result == nil || result.Meta["engine"] != codetools.EngineBash {
		return false
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(resolveBytes(ctx, result.Payload), &fields); err != nil {
		return false
	}
	for key := range fields {
		switch key {
		case "stdout", "stderr", "exit_code", "timed_out", "stdout_truncated", "stderr_truncated":
		default:
			return false
		}
	}
	var stdout, stderr string
	var exitCode int
	if !decodeExecCommandScalar(fields, "stdout", &stdout) ||
		!decodeExecCommandScalar(fields, "stderr", &stderr) ||
		!decodeExecCommandScalar(fields, "exit_code", &exitCode) {
		return false
	}
	for _, key := range []string{"timed_out", "stdout_truncated", "stderr_truncated"} {
		if raw, ok := fields[key]; ok {
			var value bool
			if !decodeExecCommandRawScalar(raw, &value) {
				return false
			}
		}
	}
	return true
}

func decodeExecCommandScalar(fields map[string]json.RawMessage, key string, dst any) bool {
	raw, ok := fields[key]
	return ok && decodeExecCommandRawScalar(raw, dst)
}

func decodeExecCommandRawScalar(raw json.RawMessage, dst any) bool {
	return len(raw) != 0 && string(raw) != "null" && json.Unmarshal(raw, dst) == nil
}

func (s *Server) recordExecCommandRoute(call *abi.ToolCall) {
	if s == nil || s.logf == nil || call == nil {
		return
	}
	event := struct {
		Event              string `json:"event"`
		TraceID            string `json:"trace_id"`
		RequestedRoute     string `json:"requested_route"`
		ExecutedRoute      string `json:"executed_route"`
		AdmittedResultType string `json:"admitted_result_type"`
	}{
		Event:              "gateway_syscall_route",
		TraceID:            strings.TrimSpace(call.TraceID),
		RequestedRoute:     call.Meta["requested_tool"],
		ExecutedRoute:      codetools.EngineBash,
		AdmittedResultType: execCommandResultType,
	}
	encoded, err := json.Marshal(event)
	if err == nil {
		s.logf("%s", encoded)
	}
}

func annotateWorkspaceExecution(verdict *WireVerdict, details map[string]string) {
	if len(details) == 0 || (verdict.Kind != "ALLOW" && verdict.Kind != "TRANSFORM") {
		return
	}
	if verdict.Detail == nil {
		verdict.Detail = make(map[string]string)
	}
	verdict.Detail["kernel_by"] = verdict.By
	verdict.By = "lease-admission-permissive"
	for key, value := range details {
		verdict.Detail[key] = value
	}
}

func isCapabilitiesTool(tool string) bool {
	return tool == "fak_capabilities" ||
		tool == "mcp__fak__fak_capabilities" ||
		tool == "mcp__fak_guard__fak_capabilities"
}

// nativeToolName identifies gateway-owned operations before any model route is selected.
func nativeToolName(tool string) string {
	for _, prefix := range []string{"functions.", "mcp__fak__", "mcp__fak_guard__"} {
		tool = strings.TrimPrefix(tool, prefix)
	}
	switch tool {
	case "fak_context_restore", "fak_memory_run", remoteMaintenanceTool:
		return tool
	default:
		return ""
	}
}

// syscallNative keeps gateway-owned tools on their native implementation even
// when called through fak_syscall. The call and result cross the kernel floors,
// but neither a model route nor a cached mock result can replace the operation.
func (s *Server) syscallNative(ctx context.Context, tc *abi.ToolCall, readOnly bool) (wv WireVerdict, env *ResultEnvelope, handled bool, err error) {
	tool := nativeToolName(tc.Tool)
	if tool == "" {
		return WireVerdict{}, nil, false, nil
	}
	if tool == remoteMaintenanceTool {
		wv, env, err = s.syscallRemoteMaintenance(ctx, tc)
		return wv, env, true, err
	}
	if tc.SeqNo == 0 {
		tc.SeqNo = s.nextOriginSeq()
	}
	v := s.k.Decide(ctx, tc)
	if v.Kind != abi.VerdictAllow {
		// Transforms and witness requests remain a client-visible preview. Native
		// execution requires a fresh affirmative admission of the actual arguments.
		return renderVerdict(v, nil), nil, true, nil
	}
	admissionDetails, refusal, refused := s.prepareWorkspaceExecution(ctx, tc, readOnly)
	if refused {
		return refusal, nil, true, nil
	}
	defer func() { annotateWorkspaceExecution(&wv, admissionDetails) }()
	args := resolveBytes(ctx, tc.Args)
	var value any
	switch tool {
	case "fak_context_restore":
		var req ContextRestoreRequest
		if err = json.Unmarshal(args, &req); err == nil {
			if req.TraceID == "" {
				req.TraceID = tc.TraceID
			}
			value, err = s.restoreContext(principalFromContext(ctx), req)
		}
	case "fak_memory_run":
		var req MemoryRequest
		if err = json.Unmarshal(args, &req); err == nil {
			if readOnly && req.Apply {
				err = errors.New("fak_memory_run apply requires a write-capable call")
			} else {
				value, err = s.memoryRun(ctx, req)
			}
		}
	}
	if err != nil {
		return renderVerdict(v, nil), nil, true, err
	}
	body, err := json.Marshal(value)
	if err != nil {
		return renderVerdict(v, nil), nil, true, err
	}
	r := &abi.Result{Call: tc, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent},
		Meta:    map[string]string{"engine": "fak-mcp", "served_by": "native", "tool": tool}}
	admitted := s.k.AdmitResult(ctx, tc, r)
	s.rememberOriginSeq(tc.TraceID, tc.Tool, string(args), tc.SeqNo)
	if abi.FoldRank(admitted.Kind) > abi.FoldRank(v.Kind) {
		v = admitted
	}
	env = &ResultEnvelope{Status: statusName(r.Status), Content: string(resolveBytes(ctx, r.Payload)), Meta: r.Meta}
	return renderVerdict(v, resultMeta(r)), env, true, nil
}

// dispatchEnsemble executes a multi-member routing Plan (issue #597): it runs each
// member as its OWN independently-adjudicated kernel call — carrying THAT member's
// model in abi.ToolCall.Engine (the same pre-submit residency contract a single-model
// route obeys) — gathers the ALLOWED members' outputs in Plan.Members order, and folds
// them with modelroute.Combine(plan.Reduce, votes). The contract this honors, point by
// point (see the internal/modelroute package doc):
//
//   - N INDEPENDENTLY-ADJUDICATED CALLS, never one fan-out that bypasses the floor.
//     Each member is a full k.Syscall, so a vote member bound for a REMOTE model still
//     crosses the residency/policy floor and is DENIED for a tenant/sensitive payload.
//   - MEMBER ORDER PRESERVED INTO THE FOLD. votes are appended in Plan.Members order,
//     so ReduceConcat / ReduceVote tie-breaks stay deterministic. A member the kernel
//     refused (or that errored at dispatch) contributes NO vote; the survivors keep
//     their relative order.
//   - FAIL CLOSED on a wipeout. If EVERY member was refused, there is no silent empty
//     success — the last member's refusal verdict is surfaced (so a residency/policy
//     reason reaches the wire) and the result Status is ERROR.
//
// vDSO interaction: a member contributes a vote iff its result Status is OK (a refused
// member's Reap yields a Status=Error deny-as-value). For the canonical write-shaped
// ensemble (a guard quorum over a destructive tool) the vDSO never dedups, so every
// member's engine actually runs. A read-only idempotent ensemble may have later members
// served from an earlier member's tier-2 fill — consistent with fak's engine-independence
// model for idempotent reads (the same bytes regardless of which engine), where an
// ensemble adds nothing anyway.
func (s *Server) dispatchEnsemble(ctx context.Context, base *abi.ToolCall, plan modelroute.Plan) (WireVerdict, *ResultEnvelope, error) {
	return s.dispatchEnsembleWithReadOnly(ctx, base, plan, false)
}

func (s *Server) dispatchEnsembleWithReadOnly(ctx context.Context, base *abi.ToolCall, plan modelroute.Plan, readOnly bool) (WireVerdict, *ResultEnvelope, error) {
	votes := make([]modelroute.Vote, 0, len(plan.Members))
	var lastRefused abi.Verdict
	refused := 0
	var admissionDetails map[string]string
	for _, mem := range plan.Members {
		// Bind THIS member through the account roster before its own kernel call (#2528):
		// each ensemble member resolves independently to its account-resolved EngineRoute,
		// so a member bound for a remote account still crosses the residency floor for a
		// sensitive payload, and the fold below stays in Plan.Members order. An unresolvable
		// member fails LOUD for the whole ensemble — never a silently-dropped or
		// default-routed vote. Without a roster the member model id is the route (pre-#2528).
		// Each member is adjudicated for the SAME principal the parent call presented, so
		// an ensemble can never reach an account the tenant is not provisioned for by
		// spreading across members. A refused member fails the whole ensemble LOUD rather
		// than silently dropping a vote — a residency denial is not a missing opinion.
		route, rerr := s.resolveRoute(mem.Model, base.Meta[vdso.MetaPrincipal])
		if rerr != nil {
			return WireVerdict{}, nil, rerr
		}
		r, v, details, refusal := s.workspaceKernelSyscall(ctx, memberCall(base, route), readOnly)
		if refusal != nil {
			return *refusal, nil, nil
		}
		if len(details) != 0 {
			admissionDetails = details
		}
		if r == nil || r.Status != abi.StatusOK {
			lastRefused = v
			refused++
			continue
		}
		votes = append(votes, modelroute.Vote{Member: mem, Output: string(resolveBytes(ctx, r.Payload))})
	}
	if len(votes) == 0 {
		// Every member was refused or errored at dispatch — fail closed (never a silent
		// empty success). Surface the last refusal verdict so the residency/policy reason
		// reaches the wire; default to a plain deny if the verdict was somehow non-refusing.
		wv := renderVerdict(lastRefused, nil)
		if wv.Kind == "ALLOW" {
			wv = WireVerdict{Kind: "DENY", Reason: abi.ReasonName(abi.ReasonPolicyBlock), By: "modelroute-ensemble", Disposition: "TERMINAL"}
		}
		env := &ResultEnvelope{Status: "ERROR", Content: "", Meta: map[string]string{
			"served_by":        "modelroute-ensemble",
			"ensemble_refused": itoa(uint64(refused)),
		}}
		return wv, env, nil
	}
	folded, ferr := modelroute.Combine(plan.Reduce, votes)
	if ferr != nil {
		// A misconfigured reduce over incompatible outputs (e.g. all_reduce over
		// non-numeric tool results) is a fail-loud error, never a silent guess.
		return WireVerdict{}, nil, fmt.Errorf("gateway: ensemble combine: %w", ferr)
	}
	meta := map[string]string{
		"served_by":        "modelroute-ensemble",
		"reduce":           string(folded.Reduce),
		"ensemble_members": itoa(uint64(folded.Members)),
	}
	if refused > 0 {
		meta["ensemble_refused"] = itoa(uint64(refused))
	}
	if folded.Winner != "" {
		meta["winner"] = folded.Winner
	}
	wv := WireVerdict{Kind: "ALLOW", By: "modelroute-ensemble"}
	annotateWorkspaceExecution(&wv, admissionDetails)
	return wv,
		&ResultEnvelope{Status: "OK", Content: folded.Output, Meta: meta}, nil
}

// memberCall clones a base ToolCall for one ensemble member, binding THAT member's
// model to Engine before submission (the pre-submit residency contract) and giving the
// call a fresh identity (SeqNo unset, an independent Meta copy) so the kernel
// adjudicates and dispatches each member on its own. The content-addressed Args Ref is
// shared — every member sees the same input — while the Meta map is copied so a
// per-call kernel annotation can never leak across members.
func memberCall(base *abi.ToolCall, model string) *abi.ToolCall {
	meta := make(map[string]string, len(base.Meta))
	for k, v := range base.Meta {
		meta[k] = v
	}
	return fusedturn.Tag(&abi.ToolCall{
		Tool:    base.Tool,
		Args:    base.Args,
		TraceID: base.TraceID,
		Meta:    meta,
		Engine:  model,
	}, fusedturn.ClassWeight)
}

// canonicalToolName normalizes incoming tool namespace wrappers across diverse
// harness dialects (OpenAI/Codex "functions.<tool>", Claude Code "mcp__<server>__<tool>",
// and OpenCode "<server>_<server>_<tool>" or "<server>_<tool>") to the canonical tool name.
// When policy is non-nil, a prefix is stripped if the candidate exists in policy's allowed
// tools, prefixes, or arg rules.
func canonicalToolName(tool string, pol *policy.Runtime) string {
	tool = strings.TrimSpace(tool)
	if tool == "" {
		return ""
	}

	// If the tool is already recognized as-is by the active policy, no normalization is needed.
	if pol != nil && isToolRecognized(tool, pol) {
		return tool
	}

	// 1. OpenAI / Codex prefix: "functions."
	if strings.HasPrefix(tool, "functions.") {
		candidate := strings.TrimPrefix(tool, "functions.")
		if isToolRecognized(candidate, pol) {
			return candidate
		}
	}

	// 2. Claude Code MCP prefix: "mcp__<server>__<tool>"
	if strings.HasPrefix(tool, "mcp__") {
		rest := strings.TrimPrefix(tool, "mcp__")
		if idx := strings.Index(rest, "__"); idx > 0 && idx+2 < len(rest) {
			candidate := rest[idx+2:]
			if isToolRecognized(candidate, pol) {
				return candidate
			}
		}
	}

	// 3. OpenCode double prefix: "<server>_<server>_<tool>"
	// e.g. "fak_fak_read" -> "fak_read" (or "read")
	if idx1 := strings.Index(tool, "_"); idx1 > 0 {
		server := tool[:idx1]
		rest := tool[idx1+1:]
		if strings.HasPrefix(rest, server+"_") {
			// Stripping one <server>_ yields "<server>_<tool>"
			if isToolRecognized(rest, pol) {
				return rest
			}
			// Stripping second <server>_ yields "<tool>"
			subTool := strings.TrimPrefix(rest, server+"_")
			if isToolRecognized(subTool, pol) {
				return subTool
			}
		}
	}

	// 4. OpenCode single prefix: "<server>_<tool>"
	if idx := strings.Index(tool, "_"); idx > 0 && idx+1 < len(tool) {
		candidate := tool[idx+1:]
		if pol != nil && isToolRecognized(candidate, pol) {
			return candidate
		}
	}

	return tool
}

func isToolRecognized(candidate string, pol *policy.Runtime) bool {
	if pol == nil {
		return true
	}
	if pol.Adjudicator.Allow != nil && pol.Adjudicator.Allow[candidate] {
		return true
	}
	if pol.Adjudicator.Deny != nil {
		if _, ok := pol.Adjudicator.Deny[candidate]; ok {
			return true
		}
	}
	for _, pre := range pol.Adjudicator.AllowPrefix {
		if strings.HasPrefix(candidate, pre) {
			return true
		}
	}
	for _, pred := range pol.Adjudicator.ArgPredicates {
		if pred.Tool == candidate {
			return true
		}
	}
	return false
}

// buildCall converts untrusted wire input into an abi.ToolCall. The raw argument
// bytes are Put into a tainted, agent-scoped Ref (the fail-closed default the IFC
// sink-gate relies on) — the wire NEVER carries a Ref. Empty args normalize to
// "{}" so a zero Ref is never submitted.
func (s *Server) buildCall(ctx context.Context, tool, rawArgs string, readOnly bool, witness, traceID string) (*abi.ToolCall, error) {
	if strings.TrimSpace(tool) == "" {
		return nil, errors.New("missing tool name")
	}
	requestedTool := strings.TrimSpace(tool)
	canonical := canonicalToolName(requestedTool, s.policyRuntime)
	execRequest := canonical == "exec_command"
	args := []byte(rawArgs)
	if len(args) == 0 {
		args = []byte("{}")
	}
	ref, err := abi.ActiveResolver().Put(ctx, args)
	if err != nil {
		return nil, fmt.Errorf("resolver: %w", err)
	}
	meta := metaFor(canonical, readOnly)
	if execRequest {
		meta["requested_tool"] = requestedTool
		meta["canonical_tool"] = "exec_command"
		meta["exec_command_contract"] = execCommandResultType
	} else if canonical != requestedTool {
		meta["wire_tool"] = requestedTool
		meta["canonical_tool"] = canonical
	}
	if label := journal.ArgsLabelForBytes(args); label != "" {
		meta[journal.MetaArgsLabel] = label
	}
	// The external world-state witness (a git commit / blob / lease epoch the caller is
	// reading at) keys this read for cross-agent dedup AND for causal revocation: a later
	// fak_revoke of this witness evicts every pooled entry admitted under it.
	if witness != "" {
		meta["witness"] = witness
	}
	// Lower the request's isolation principal (a tenant / user / auth subject, carried
	// request-scoped on ctx from the X-Fak-Principal header or the request's principal
	// field) onto the call so the vDSO scopes its tier-2 cache entry PER PRINCIPAL: a
	// different principal can neither be served nor fill the same (tool,args) entry,
	// closing the cross-tenant cache leak + the hit/miss timing oracle. Empty =>
	// single-tenant (every caller shares, the v0.1 behavior).
	if p := principalFromContext(ctx); p != "" {
		meta[vdso.MetaPrincipal] = p
	}
	// Thread a TraceID end-to-end: the IFC ledger + plan-CFI key their per-session
	// state on it, so a served call MUST carry one. The wire supplies it for
	// cross-call correlation; absent, we mint a fresh non-empty id rather than fall
	// back to the empty shared-default trace (which would pool every served session
	// onto one taint high-water mark).
	tc := fusedturn.Tag(&abi.ToolCall{Tool: canonical, Args: ref, TraceID: s.traceFor(traceID), Meta: meta}, fusedturn.ClassClassical)
	if execRequest {
		// The external exec_command policy sees the original tool and arguments
		// first. workspaceKernelSyscall discards that admitted submission without
		// dispatch, then submits the adapted Bash call through the codetools rung.
		return tc, nil
	}
	if nativeToolName(canonical) != "" {
		// Local gateway operations retain the same policy/principal metadata but
		// do not borrow a model account's route or remote residency constraints.
		tc.Engine = "local:fak-mcp"
		return tc, nil
	}
	// Per-call model routing (opt-in): classify this tool call into a routing Subject
	// and, for a single-model PICK, bind the chosen model to Engine HERE — before the
	// caller hands tc to k.Syscall. That is the load-bearing residency contract: the
	// residency PDP reads c.Engine INSIDE the adjudication fold, so a route written
	// any later (at Reap/dispatch) would adjudicate an empty Engine and fail open on a
	// tenant payload bound for a remote model. nil manifest => Engine "" => kernel
	// default (byte-for-byte the pre-routing path). With an account roster configured
	// (#2528) the routed model id is BOUND through the roster to its account-resolved
	// EngineRoute here, and an unresolvable id (unknown account, no binding + no default)
	// fails LOUD — the call never dispatches on a silent default.
	route, rerr := s.routeEngineWithContext(ctx, canonical, readOnly, meta)
	if rerr != nil {
		return nil, rerr
	}
	tc.Engine = route
	return tc, nil
}

func adaptedExecCommandCall(ctx context.Context, original *abi.ToolCall) (*abi.ToolCall, error) {
	args, err := adaptExecCommandArgs(string(resolveBytes(ctx, original.Args)))
	if err != nil {
		return nil, err
	}
	ref, err := abi.ActiveResolver().Put(ctx, []byte(args))
	if err != nil {
		return nil, fmt.Errorf("resolver: %w", err)
	}
	meta := codetools.CallMeta(codetools.ToolBash, "")
	for key, value := range original.Meta {
		switch key {
		case "readOnlyHint", "idempotentHint", "destructive":
		default:
			meta[key] = value
		}
	}
	return fusedturn.Tag(&abi.ToolCall{
		Tool:    codetools.ToolBash,
		Args:    ref,
		TraceID: original.TraceID,
		SeqNo:   original.SeqNo,
		Meta:    meta,
	}, fusedturn.ClassClassical), nil
}

func adaptExecCommandArgs(raw string) (string, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &fields); err != nil {
		return "", fmt.Errorf("exec_command arguments: %w", err)
	}
	if fields == nil {
		return "", errors.New("exec_command arguments must be an object")
	}
	for key := range fields {
		switch key {
		case "cmd", "workdir", "command", "cwd", "timeout_ms":
		default:
			return "", fmt.Errorf("exec_command arguments: unsupported field %q", key)
		}
	}
	decodeString := func(primary, alias string) (string, error) {
		var first, second string
		if value := fields[primary]; len(value) > 0 {
			if string(value) == "null" {
				return "", fmt.Errorf("exec_command arguments: %s must be a string", primary)
			}
			if err := json.Unmarshal(value, &first); err != nil {
				return "", fmt.Errorf("exec_command arguments: %s must be a string", primary)
			}
		}
		if value := fields[alias]; len(value) > 0 {
			if string(value) == "null" {
				return "", fmt.Errorf("exec_command arguments: %s must be a string", alias)
			}
			if err := json.Unmarshal(value, &second); err != nil {
				return "", fmt.Errorf("exec_command arguments: %s must be a string", alias)
			}
		}
		if first != "" && second != "" && first != second {
			return "", fmt.Errorf("exec_command arguments: conflicting %s and %s", primary, alias)
		}
		if first != "" {
			return first, nil
		}
		return second, nil
	}
	command, err := decodeString("cmd", "command")
	if err != nil {
		return "", err
	}
	cwd, err := decodeString("workdir", "cwd")
	if err != nil {
		return "", err
	}
	args := codetools.BashArgs{Command: command, Cwd: cwd}
	if value := fields["timeout_ms"]; len(value) > 0 {
		if string(value) == "null" {
			return "", errors.New("exec_command arguments: timeout_ms must be an integer")
		}
		if err := json.Unmarshal(value, &args.TimeoutMS); err != nil {
			return "", fmt.Errorf("exec_command arguments: timeout_ms must be an integer")
		}
	}
	encoded, err := json.Marshal(args)
	if err != nil {
		return "", fmt.Errorf("exec_command arguments: %w", err)
	}
	return string(encoded), nil
}

// traceFor returns the caller's TraceID, or mints a fresh, process-unique non-empty
// one so the result-side IFC ledger + plan-CFI never collapse distinct served
// sessions onto the empty-string default trace.
func (s *Server) traceFor(traceID string) string {
	traceID = strings.TrimSpace(traceID)
	if traceID != "" {
		return traceID
	}
	s.defaultTraceMu.RLock()
	defaultTraceID := s.defaultTraceID
	s.defaultTraceMu.RUnlock()
	if defaultTraceID != "" && defaultTraceID != UnkeyedPerConnectionTraceID {
		return defaultTraceID
	}
	return "gw-" + itoa(atomic.AddUint64(&s.traceSeq, 1))
}

// SetDefaultTraceID changes the trace used for callers that omit X-Trace-Id /
// trace_id. Guard's budget-restart supervisor uses this when it relaunches a child
// under a continuation id; a blank value restores the historical minted gw-N default.
func (s *Server) SetDefaultTraceID(traceID string) {
	if s == nil {
		return
	}
	s.defaultTraceMu.Lock()
	s.defaultTraceID = strings.TrimSpace(traceID)
	s.defaultTraceMu.Unlock()
}

// DefaultTraceID returns the trace currently used for callers that omit an
// explicit trace. Lifecycle adapters use the read before atomically replacing it
// at an external provider session boundary.
func (s *Server) DefaultTraceID() string {
	if s == nil {
		return ""
	}
	s.defaultTraceMu.RLock()
	defer s.defaultTraceMu.RUnlock()
	return s.defaultTraceID
}

// principalCtxKey is the context key carrying a request's isolation principal.
type principalCtxKey struct{}

// WithPrincipal returns a context carrying the caller's isolation principal (a tenant /
// user / auth subject). buildCall lowers it onto ToolCall.Meta[vdso.MetaPrincipal] so
// the vDSO scopes tier-2 cache entries per principal — a different principal can neither
// read nor fill the same (tool,args) entry, closing the cross-tenant cache leak + the
// hit/miss timing oracle. An empty principal returns ctx unchanged (single-tenant: every
// caller shares, the v0.1 behavior). Exported so a host embedding the gateway can set the
// principal from its own auth context before calling Syscall.
func WithPrincipal(ctx context.Context, principal string) context.Context {
	principal = strings.TrimSpace(principal)
	if principal == "" {
		return ctx
	}
	return context.WithValue(ctx, principalCtxKey{}, principal)
}

// principalFromContext returns the request-scoped isolation principal, or "" if none.
func principalFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	p, _ := ctx.Value(principalCtxKey{}).(string)
	return p
}

// contextChange applies a requester-initiated context-control mutation to a
// persisted recall core image. This is intentionally narrower than general file
// mutation: the only shipped operation is a tombstone, which makes a page absent
// from future model-visible recall while leaving the original page row and CAS
// bytes available for audit.
func (s *Server) contextChange(ctx context.Context, req ContextChangeRequest) (ContextChangeResponse, error) {
	select {
	case <-ctx.Done():
		return ContextChangeResponse{}, ctx.Err()
	default:
	}
	imageDir := strings.TrimSpace(req.ImageDir)
	if imageDir == "" {
		return ContextChangeResponse{}, errors.New("context change requires image_dir")
	}
	sess, err := recall.Load(imageDir)
	if err != nil {
		return ContextChangeResponse{}, fmt.Errorf("load core image: %w", err)
	}
	ch, err := sess.RequestContextChange(recall.ContextChangeRequest{
		Action:      contextAction(req.Action),
		Step:        req.Step,
		Digest:      strings.TrimSpace(req.Digest),
		Reason:      req.Reason,
		RequestedBy: req.RequestedBy,
		Witness:     req.Witness,
	})
	if err != nil {
		return ContextChangeResponse{}, err
	}
	if err := sess.Persist(imageDir); err != nil {
		return ContextChangeResponse{}, fmt.Errorf("persist core image: %w", err)
	}
	s.logf("gateway: context change %s step=%d image=%s requested_by=%s", ch.Action, ch.Step, imageDir, ch.RequestedBy)
	// Trust-propagate the operator tombstone onto the restore stash: a span the operator just
	// suppressed in the persisted recall image must NOT stay resurrectable through its
	// content-addressed restore handle, or fak_context_restore would defeat the very suppression this
	// action recorded. This is the shipped SETTER the restore trust gate was waiting for (see
	// ctxrestore_gate.go); restoreContext already refuses a set flag. We key on ch.Digest — the
	// APPLIED change's resolved content-address (always populated on an applied tombstone, unlike the
	// caller-supplied req.Digest which may be blank on a step-only request) — so a step-only tombstone
	// still suppresses the matching handle. The digest is a content-address, so tombstoneRestore flips
	// it across ALL traces that stashed it; 0 is a valid no-op (this digest was never a live restore
	// handle). This does not change the response contract or error behavior.
	if ch.Action == recall.ContextActionTombstone {
		if suppressed := s.tombstoneRestore(ch.Digest); suppressed > 0 {
			s.logf("gateway: context change tombstone digest=%s suppressed %d restore handle(s) on the wire", ch.Digest, suppressed)
		}
	}
	return contextChangeResponse(imageDir, ch, sess.Tombstoned(ch.Step)), nil
}

func contextAction(action string) recall.ContextAction {
	switch strings.TrimSpace(action) {
	case "", "tombstone", string(recall.ContextActionTombstone):
		return recall.ContextActionTombstone
	default:
		return recall.ContextAction(strings.TrimSpace(action))
	}
}

func contextChangeResponse(imageDir string, ch recall.ContextChange, tombstoned bool) ContextChangeResponse {
	return ContextChangeResponse{
		ImageDir:    imageDir,
		ID:          ch.ID,
		Action:      string(ch.Action),
		Step:        ch.Step,
		Digest:      ch.Digest,
		Reason:      ch.Reason,
		RequestedBy: ch.RequestedBy,
		Witness:     ch.Witness,
		TrustEpoch:  ch.TrustEpoch,
		Applied:     ch.Applied,
		Tombstoned:  tombstoned,
	}
}

// metaFor derives the kernel call hints. A caller may explicitly mark a call
// read-only (enabling vDSO dedup of duplicate reads); otherwise the gateway uses
// the read-only NAME prefix family (the same convention as DefaultPolicy's
// AllowPrefix and agent.metaFor) and FAILS CLOSED to destructive for anything
// else, so the vDSO never serves a stale write.
func metaFor(tool string, readOnly bool) map[string]string {
	if readOnly || readOnlyPrefix(tool) {
		return map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}
	}
	return map[string]string{"readOnlyHint": "false", "idempotentHint": "false", "destructive": "true"}
}

func readOnlyPrefix(tool string) bool {
	for _, p := range []string{"get_", "read_", "search_", "list_", "lookup_", "find_", "calc"} {
		if strings.HasPrefix(tool, p) {
			return true
		}
	}
	return false
}

// resolveBytes materializes a Ref's bytes through the active resolver (mirrors
// agent.refBytes). An inline Ref carries its own bytes; a blob/region Ref is
// resolved on demand.
func resolveBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

func resultMeta(r *abi.Result) map[string]string {
	if r == nil {
		return nil
	}
	return r.Meta
}

// loopbackOnly reports whether addr binds ONLY the loopback interface — used to
// warn loudly when a no-auth gateway is exposed beyond localhost. It classifies by
// IP VALUE (net.ParseIP + IsLoopback), not by string prefix: "127.0.0.1.evil.com"
// is NOT loopback, and an empty host (the ":port" wildcard, which net.Listen binds
// to ALL interfaces) is NOT loopback either. A non-IP host (a DNS name) cannot be
// proven loopback at bind time, so it is treated as exposed.
func LoopbackOnly(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr // no port present
	}
	host = strings.Trim(host, "[]")
	if host == "" {
		return false // ":port" => all interfaces
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func loopbackOnly(addr string) bool { return LoopbackOnly(addr) }

// RotationEvidenceSnapshot returns cumulative provider-side failures that are safe to
// use as positive account-rotation evidence. Local child exit codes are intentionally
// absent: only errors observed at the upstream boundary enter these counters.
func (s *Server) RotationEvidenceSnapshot() map[string]uint64 {
	out := map[string]uint64{}
	if s == nil || s.metrics == nil {
		return out
	}
	s.metrics.upstreamErrMu.Lock()
	defer s.metrics.upstreamErrMu.Unlock()
	for _, kind := range []string{"auth", "rate_limited"} {
		if n := s.metrics.upstreamErrors[kind]; n > 0 {
			out[kind] = n
		}
	}
	return out
}

// TransientWireErrorSnapshot returns the cumulative count of TRANSIENT upstream transport
// failures observed this session (the "transport" upstream-error kind: a mid-flight
// connection drop/reset, truncated read, or I/O timeout that exhausted the planner's
// in-handler retry and surfaced to the wrapped agent). `fak guard`'s supervisor snapshots
// this before a child starts and again after it exits: a positive delta is the evidence a
// transient wire crash — not a systematic failure — drove a non-zero exit, so a single
// bounded relaunch is warranted (#3514). Deterministic dial failures are intentionally
// absent: those land in the "unreachable" kind and a relaunch would only re-trip them. A nil
// server or metrics returns 0, so a caller may snapshot unconditionally.
func (s *Server) TransientWireErrorSnapshot() uint64 {
	if s == nil || s.metrics == nil {
		return 0
	}
	s.metrics.upstreamErrMu.Lock()
	defer s.metrics.upstreamErrMu.Unlock()
	return s.metrics.upstreamErrors["transport"]
}

// UpstreamErrorKindsSnapshot returns a copy of the FULL cumulative upstream-error tally by
// kind — every label upstreamErrorKind mints ("stalled", "oom", "unreachable",
// "rate_limited", "auth", "forbidden", "overloaded", "status_4xx", "status_5xx",
// "transport", "other"), not the two-kind rotation subset (RotationEvidenceSnapshot) or the
// single transport scalar (TransientWireErrorSnapshot). It is the accessor the durable
// gateway-usage ledger reads at session teardown (#5487): the kind the classifier already
// computes was process-local — an in-memory /metrics counter plus a stderr FAILED line —
// so under `fak guard`, where the gateway is a per-invocation process, a stall left no
// trace once the wrapped command exited.
//
// Returns nil (not an empty map) when nothing failed, so the caller's omitempty ledger
// field stays ABSENT rather than writing an empty object: absent reads NOT INSTRUMENTED,
// and a pre-field row must stay byte-identical. A nil server or metrics likewise returns
// nil, so a caller may snapshot unconditionally.
func (s *Server) UpstreamErrorKindsSnapshot() map[string]uint64 {
	if s == nil || s.metrics == nil {
		return nil
	}
	s.metrics.upstreamErrMu.Lock()
	defer s.metrics.upstreamErrMu.Unlock()
	if len(s.metrics.upstreamErrors) == 0 {
		return nil
	}
	out := make(map[string]uint64, len(s.metrics.upstreamErrors))
	for k, v := range s.metrics.upstreamErrors {
		out[k] = v
	}
	return out
}
