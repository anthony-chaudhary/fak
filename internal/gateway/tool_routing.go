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
	case *ReplicaRouter:
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
	tc, err := s.buildCall(ctx, tool, rawArgs, readOnly, witness, traceID)
	if err != nil {
		return WireVerdict{}, "", err
	}
	if callSeq != 0 {
		tc.SeqNo = callSeq
	}
	opTrace, opTool = tc.TraceID, tc.Tool
	v := s.k.Decide(ctx, tc)
	wv = renderVerdict(v, nil)
	if v.Kind == abi.VerdictTransform {
		if tp, ok := v.Payload.(abi.TransformPayload); ok {
			repaired = string(resolveBytes(ctx, tp.NewArgs))
		}
	}
	return wv, repaired, nil
}

// syscall runs a (tool, rawArgs) pair through the FULL syscall boundary
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
	tc, err := s.buildCall(ctx, tool, rawArgs, readOnly, witness, traceID)
	if err != nil {
		return WireVerdict{}, nil, err
	}
	opTrace, opTool = tc.TraceID, tc.Tool
	// Ensemble fan-out (issue #597): a multi-member routing Plan runs each member as
	// its OWN adjudicated kernel call and folds the outputs. buildCall left tc.Engine
	// unset for an ensemble (routeEngine returns "" rather than collapse to one member);
	// dispatchEnsemble re-reads the same routing decision and submits N independent
	// calls. The single-model PICK below is byte-for-byte the pre-#597 path.
	if plan, ok := s.ensemblePlan(tc.Tool, readOnly, tc.Meta); ok {
		wv, env, err = s.dispatchEnsemble(ctx, tc, plan)
		if err != nil {
			return wv, env, err
		}
		// The scripted-fold witness rides the SAME execute floor as a single-model call
		// (#2858): an ensemble is still a scripted RPC result folded into the parent.
		wv = witnessScriptedFold(tc.Tool, env, wv)
		return wv, env, nil
	}
	r, v := s.k.Syscall(ctx, tc)
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

func isCapabilitiesTool(tool string) bool {
	return tool == "fak_capabilities" ||
		tool == "mcp__fak__fak_capabilities" ||
		tool == "mcp__fak_guard__fak_capabilities"
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
	votes := make([]modelroute.Vote, 0, len(plan.Members))
	var lastRefused abi.Verdict
	refused := 0
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
		r, v := s.k.Syscall(ctx, memberCall(base, route))
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
	return WireVerdict{Kind: "ALLOW", By: "modelroute-ensemble"},
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
	args := []byte(rawArgs)
	if len(args) == 0 {
		args = []byte("{}")
	}
	ref, err := abi.ActiveResolver().Put(ctx, args)
	if err != nil {
		return nil, fmt.Errorf("resolver: %w", err)
	}
	canonical := canonicalToolName(tool, s.policyRuntime)
	meta := metaFor(canonical, readOnly)
	if canonical != tool {
		meta["wire_tool"] = tool
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
	route, rerr := s.routeEngine(canonical, readOnly, meta)
	if rerr != nil {
		return nil, rerr
	}
	tc.Engine = route
	return tc, nil
}

// routeDecision classifies a tool call into a modelroute.Subject (aspect=tool_call,
// the tool name, and the read-only / sensitivity / tenant signals the gateway already
// attests) and returns the manifest's routing Decision. The second return is false
// when no manifest is configured (the kernel-default path). routeEngine and
// ensemblePlan share this single classification so the single-model and ensemble
// paths can never diverge on what a call routes to.
func (s *Server) routeDecision(tool string, readOnly bool, meta map[string]string) (modelroute.Decision, bool) {
	if s.route == nil {
		return modelroute.Decision{}, false
	}
	return s.route.Route(modelroute.Subject{
		Aspect: modelroute.AspectToolCall,
		Tool:   tool,
		Labels: routeLabels(readOnly, meta),
	}), true
}

// routeEngine consults the optional per-call routing policy and returns the engine
// route to bind to abi.ToolCall.Engine, or "" for the kernel default. It returns
// Decision.Plan.Primary() for a single-model PICK. An ENSEMBLE plan is left to the
// kernel default here (route ""): the N-submit fan-out happens at dispatch time in
// dispatchEnsemble (the syscall path), and collapsing an ensemble to one member here
// would be a silent wrong route. The returned route is the model id verbatim
// (Plan.Primary()'s documented destination), NOT collapsed to a registered engine id —
// the string must keep the model's remote-ness so the residency gate can deny a
// tenant/sensitive payload bound for a remote model. A route to a model with no
// registered engine driver fails LOUD at dispatch ("no engine registered for route"),
// never silently runs elsewhere.
func (s *Server) routeEngine(tool string, readOnly bool, meta map[string]string) (string, error) {
	began := time.Now()
	d, ok := s.routeDecision(tool, readOnly, meta)
	if !ok {
		// No manifest: the kernel-default path, never reached when routing is off — record
		// nothing so the family honestly reads 0 until routing is actually live.
		return "", nil
	}
	// Routing is LIVE for this call: fold the per-aspect Decision into the observability
	// journal (#603) so it reaches /metrics AND the audit trail. This is the ONE fold per
	// served tool call — routeEngine runs on every buildCall (single-model and ensemble
	// alike); ensemblePlan re-routes the same Subject at dispatch but does not re-record, so
	// a call is counted exactly once. The overhead is the wall-clock the decision itself cost
	// (pure-function routing, so tiny). nil metrics / nil routing accumulator => no-op.
	s.metrics.observeRouteDecision(s.routeManifestVersion(), d, time.Since(began))
	if d.Plan.IsEnsemble() {
		return "", nil
	}
	// meta already carries the request's isolation principal (buildCall lowered it from
	// ctx onto vdso.MetaPrincipal), so the residency arm reads the SAME principal the
	// vDSO scopes its cache by — one identity, not two that could disagree.
	return s.resolveRoute(d.Plan.Primary(), meta[vdso.MetaPrincipal])
}

// resolveRoute maps a routed model id to the engine route bound to abi.ToolCall.Engine
// (#2528). With an account roster configured it BINDS the abstract id through the
// roster to the account-resolved Target.EngineRoute() ("openai:acct/gpt-5.5") — the
// load-bearing residency contract, since the residency PDP reads the route INSIDE the
// adjudication fold, so the account-resolved remote/local route must be visible BEFORE
// Submit. Without a roster it returns the id verbatim (byte-for-byte the pre-#2528
// path). A model id that cannot resolve (unknown account, no binding + no default) is a
// FAIL-LOUD error carrying the recovery hint from the pure resolver — never a silent
// fallback to the default engine. An empty id (no primary member) resolves to "" (the
// kernel default), never through the roster.
//
// principal is the caller's tenant ISOLATION principal (the org/project a keyset key
// authenticated as, #5332) and gates WHICH account this call may resolve through — the
// residency arm of the keyset. The check runs HERE, at the same pre-Submit seam that
// binds Engine, because that is the last point before the call reaches the kernel: an
// account the principal is not provisioned for must never become a bound route. It is
// fail-CLOSED in both directions — an account naming principals admits only its listed
// tenants, and the EMPTY principal (an unattributed caller: no keyset, or the single
// --require-key-env bearer) is refused by a restricted account rather than inheriting
// its credential. A roster whose accounts name NO principals admits everyone, so a
// pre-#5332 roster routes byte-for-byte as before.
func (s *Server) resolveRoute(modelID, principal string) (string, error) {
	if s.roster == nil || modelID == "" {
		return modelID, nil
	}
	t, err := s.roster.Resolve(modelID)
	if err != nil {
		return "", fmt.Errorf("gateway: route accounts: %w (fix the roster binding for %q or set a default account; no silent fallback)", err, modelID)
	}
	if !t.Admits(principal) {
		// Name the principal and the account, never the credential: the operator needs to
		// see WHICH tenancy was refused to fix the roster, and Target carries only the
		// credential env NAME anyway. An empty principal is reported as such so an operator
		// can tell "wrong tenant" apart from "unattributed caller".
		who := principal
		if strings.TrimSpace(who) == "" {
			who = "<unattributed>"
		}
		return "", fmt.Errorf("gateway: route accounts: principal %s is not admitted to account %q (routed model %q): that account's principals allowlist scopes it to another tenant (#5332) — add this principal to the account, or bind its key to an account it is provisioned for", who, t.Account, modelID)
	}
	return t.EngineRoute(), nil
}

// routeAccount resolves the account binding for a SINGLE-MODEL routed call so the served
// path can record it (#2528 observability). ok is false when routing is off, no roster is
// configured, the plan is an ensemble, or the id cannot resolve (the fail-loud already
// surfaced at buildCall — observability never re-raises it). The returned Target carries
// only non-secret fields (account id, provider kind, upstream model, credential env NAME),
// so it is safe to fold into a report; the credential VALUE never enters a Target. Pure
// (re-runs the cheap classification), consistent with ensemblePlan re-routing the same
// Subject at dispatch.
func (s *Server) routeAccount(tool string, readOnly bool, meta map[string]string) (modelroute.Target, bool) {
	if s.roster == nil {
		return modelroute.Target{}, false
	}
	d, ok := s.routeDecision(tool, readOnly, meta)
	if !ok || d.Plan.IsEnsemble() {
		return modelroute.Target{}, false
	}
	prim := d.Plan.Primary()
	if prim == "" {
		return modelroute.Target{}, false
	}
	t, err := s.roster.Resolve(prim)
	if err != nil {
		return modelroute.Target{}, false
	}
	return t, true
}

// recordRouteAccount folds the non-secret account binding of a single-model routed call
// into the result envelope Meta (#2528 acceptance: "records route decision plus account
// id/provider kind/upstream model with no secret values"). It writes the account id,
// provider kind, upstream wire model, the account-resolved engine route, and the
// credential env NAME (a name, never the secret — the ticket explicitly permits the env
// name in reports). No-op when no roster resolved the call, so the pre-#2528 meta is
// byte-for-byte unchanged.
func (s *Server) recordRouteAccount(env *ResultEnvelope, tool string, readOnly bool, meta map[string]string) {
	if env == nil {
		return
	}
	t, ok := s.routeAccount(tool, readOnly, meta)
	if !ok {
		return
	}
	if env.Meta == nil {
		env.Meta = map[string]string{}
	}
	env.Meta["route_account"] = t.Account
	env.Meta["route_kind"] = string(t.Kind)
	env.Meta["route_upstream"] = t.UpstreamModel
	env.Meta["route_engine"] = t.EngineRoute()
	if t.CredEnv != "" {
		env.Meta["route_cred_env"] = t.CredEnv // the env-var NAME, never its value
	}
}

// routeManifestVersion returns the installed routing manifest's schema version (for the
// decision digest), defaulting to the current modelroute.Version when the manifest omits
// it or no manifest is installed.
func (s *Server) routeManifestVersion() string {
	if s.route != nil {
		if mf := s.route.Manifest(); mf != nil && mf.Version != "" {
			return mf.Version
		}
	}
	return modelroute.Version
}

// ensemblePlan returns the routing Plan for this call WHEN it is a multi-member
// ensemble, so the syscall path can fan it out (issue #597). A single-model PICK, or
// no manifest, returns ok=false (the call dispatches once on the route routeEngine
// already bound to Engine). The classification is identical to routeEngine's — same
// Subject, same routeDecision — so the two never disagree on whether a call is an
// ensemble.
func (s *Server) ensemblePlan(tool string, readOnly bool, meta map[string]string) (modelroute.Plan, bool) {
	d, ok := s.routeDecision(tool, readOnly, meta)
	if !ok || !d.Plan.IsEnsemble() {
		return modelroute.Plan{}, false
	}
	return d.Plan, true
}

// routeLabels lowers the call signals the gateway honestly knows into the OPEN
// Subject.Labels a manifest Match can route on: read_only (read- vs write-shaped),
// and the sensitivity / tenant tags the residency floor also reads. Per-call prompt
// token estimation and richer classification are a later signal-enrichment child
// (#599 scout classification); the gateway routes on what it can attest today.
func routeLabels(readOnly bool, meta map[string]string) map[string]string {
	labels := map[string]string{"read_only": boolLabel(readOnly)}
	if meta != nil {
		sens := meta["sensitivity"]
		if sens == "" {
			sens = meta["data_sensitivity"]
		}
		if sens != "" {
			labels["sensitivity"] = sens
		}
		if p := meta[vdso.MetaPrincipal]; p != "" {
			labels["tenant"] = p
		}
	}
	return labels
}

// boolLabel renders a bool as a routing-label string ("true"/"false") without
// pulling strconv into this file (it formats ints via the local itoa).
func boolLabel(b bool) string {
	if b {
		return "true"
	}
	return "false"
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
	if defaultTraceID != "" {
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
