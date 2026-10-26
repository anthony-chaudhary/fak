// Package adjudicator is the in-process DOS reference monitor — the v0.1
// realization of the Adjudicator seam. It is the fused, zero-spawn dual of the
// dos-preflake hook: the SAME decide logic that a spawned hook runs, but called
// in-process so the tool-call boundary costs tens of nanoseconds to a few
// microseconds instead of a process spawn (5-14 ms Go / 232-262 ms Python).
//
// Decision discipline (mirrors dos-preflake/go decide.go):
//   - a PROVABLE refusal returns VerdictDeny with a structured ReasonCode and,
//     where the reason warrants it, a BOUNDED-DISCLOSURE witness (the offending
//     set only — the SMT-unsat-core move);
//   - a TRANSFORM rewrites args (e.g. redacts a secret-shaped field) before
//     dispatch;
//   - an UNPROVABLE / not-applicable case returns VerdictDefer (fail-to-abstain),
//     letting the kernel's fold resolve it (default-deny if nothing allowed it).
//
// It registers itself as the rank-100 (authoritative) link so cheaper pre-flight
// rungs (lower rank) run first; the kernel folds the chain by the restrictiveness
// lattice, so order does not change the verdict, only the work done.
package adjudicator

import (
	"context"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

// ReasonDoomLoop is the closed refusal reason code emitted when an in-kernel circuit breaker trips (#12523).
const ReasonDoomLoop abi.ReasonCode = 1140

// ReasonDoomLoopName is the stable name registered for ReasonDoomLoop.
const ReasonDoomLoopName = "DOOM_LOOP"

// ReasonCircuitBreakerTripped is the alternative reason code name for ReasonDoomLoop (#12523).
const ReasonCircuitBreakerTripped abi.ReasonCode = 1141

// ReasonCircuitBreakerTrippedName is the stable name registered for ReasonCircuitBreakerTripped.
const ReasonCircuitBreakerTrippedName = "CIRCUIT_BREAKER_TRIPPED"

type sessionContextKey struct{}

// ContextWithSessionID returns a derived context carrying the designated session ID (#12523).
func ContextWithSessionID(ctx context.Context, sessionID string) context.Context {
	return context.WithValue(ctx, sessionContextKey{}, sessionID)
}

func sessionIDFromContextOrCall(ctx context.Context, c *abi.ToolCall) string {
	if c != nil {
		if c.TraceID != "" {
			return c.TraceID
		}
		if c.Meta != nil {
			if s := c.Meta["session_id"]; s != "" {
				return s
			}
			if s := c.Meta["trace_id"]; s != "" {
				return s
			}
		}
	}
	if ctx != nil {
		if s, ok := ctx.Value(sessionContextKey{}).(string); ok && s != "" {
			return s
		}
	}
	return ""
}

type policyState struct {
	policy    Policy
	argByTool map[string][]ArgPredicate
	// egressList is the policy's egress rules compiled once at install time (nil when
	// the policy configures none, which Decides None for every host at zero cost).
	egressList *egresslist.List
}

// Adjudicator is the reference monitor. Construct with New; the default instance
// registers itself in init().
type Adjudicator struct {
	state           atomic.Pointer[policyState]
	authored        sync.Map
	devEdit         atomic.Pointer[devEditAttestationState]
	missionContract atomic.Pointer[MissionContract]
	receiptRoot     string
	recovery        *RecoveryAuditLedger
	circuitBreaker  atomic.Pointer[circuitBreakerTracker]
}

// New builds an adjudicator with the given policy.
func New(p Policy) *Adjudicator {
	p.Profile = sanitizeProfile(p.Profile)                         // a profile may only narrow the floor
	p.AdvisoryReasons = sanitizeAdvisoryReasons(p.AdvisoryReasons) // only heuristic reasons may soften
	a := &Adjudicator{
		receiptRoot: receiptWorkspaceRoot(),
		recovery:    NewRecoveryAuditLedger(),
	}
	a.circuitBreaker.Store(newCircuitBreakerTracker(DefaultCircuitBreakerThreshold))
	a.state.Store(&policyState{
		policy:     p,
		argByTool:  indexArgPredicates(p.ArgPredicates),
		egressList: compileEgressList(p),
	})
	return a
}

// RecoveryLedger returns the failure-recovery audit ledger for refusal effectiveness.
func (a *Adjudicator) RecoveryLedger() *RecoveryAuditLedger {
	return a.recovery
}

// SetPolicy swaps the policy (used by tests + the bench harness).
func (a *Adjudicator) SetPolicy(p Policy) {
	p.Profile = sanitizeProfile(p.Profile)                         // a profile may only narrow the floor
	p.AdvisoryReasons = sanitizeAdvisoryReasons(p.AdvisoryReasons) // only heuristic reasons may soften
	// Build the immutable predicate index before excluding readers. The lock then
	// protects only the atomic policy+index pair swap, not O(predicate-count) work.
	argByTool := indexArgPredicates(p.ArgPredicates)
	egressList := compileEgressList(p)
	a.state.Store(&policyState{policy: p, argByTool: argByTool, egressList: egressList})
}

// ResetRun clears the per-run synthesized-tool ledger (#543). The authored-script
// set is scoped to ONE agent run: a long-lived adjudicator (the registered Default
// singleton, shared across runs) calls this at a task boundary so a script authored
// in a prior run does not carry over and tighten an unrelated later exec. A
// per-run adjudicator built with New(policy) starts with an empty ledger, so it
// needs no reset. Over-retention is fail-safe (it only ever tightens), so a missed
// reset never opens the floor — it is hygiene, not a security boundary.
func (a *Adjudicator) ResetRun() {
	a.authored.Range(func(k, _ any) bool {
		a.authored.Delete(k)
		return true
	})
}

// DeniedTools returns the names of tools the policy provably refuses BY NAME (the
// Deny map keys), sorted. The static tool linter folds this against the vDSO
// fast-path registries: a denied tool that is ALSO registered pure/static would be
// served Allow by vdso.Lookup (which kernel.Submit consults BEFORE the adjudicator
// chain) and the policy Deny would never fire. Returns a fresh slice; reads under
// the RLock so a concurrent SetPolicy cannot tear the map.
func (a *Adjudicator) DeniedTools() []string {
	state := a.state.Load()
	out := make([]string, 0, len(state.policy.Deny))
	for t := range state.policy.Deny {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// indexArgPredicates groups arg predicates by their Tool, preserving slice order
// within each tool (so first-violation-wins is unchanged). Returns nil for the
// empty set (the common case), preserving Adjudicate's len()==0 fast path. Built
// once per SetPolicy (rare), never per call.
//
// Keys are lower-cased: tool names are case-variant across agents (Claude Code's
// "Bash" / "Edit" vs OpenCode's "bash" / "edit"), so a floor authored against one
// casing must still gate the other — a deny rule that silently stopped applying to a
// differently-cased tool would be a fail-OPEN hole. Lower-casing the index (and the
// lookup) broadens deny coverage to every casing; it never narrows it.
func indexArgPredicates(preds []ArgPredicate) map[string][]ArgPredicate {
	if len(preds) == 0 {
		return nil
	}
	m := make(map[string][]ArgPredicate, len(preds))
	for _, pr := range preds {
		k := strings.ToLower(pr.Tool)
		m[k] = append(m[k], pr)
	}
	return m
}

// Caps advertises no special capabilities (the baseline monitor).
func (a *Adjudicator) Caps() []abi.Capability { return nil }

// writeShaped reports whether a tool name looks like it mutates state (used to
// scope the self-modify check to write-shaped calls). It folds case itself for
// cold callers; Adjudicate folds once and uses writeShapedLower (#4007).
func writeShaped(tool string) bool { return writeShapedLower(strings.ToLower(tool)) }

// writeShapedLower is writeShaped for an ALREADY lower-cased name. Capitalized
// tool names ("Bash", "Edit") made the old per-probe ToLower allocate on every
// prefix, on every adjudication; the fold pays it once per decision.
func writeShapedLower(lowerTool string) bool {
	if lowerTool == "fak_remote_maintenance" {
		return true
	}
	for _, p := range []string{"write", "edit", "delete", "patch", "put", "exec", "run", "modify", "create"} {
		if strings.Contains(lowerTool, p) {
			return true
		}
	}
	return false
}

// lowRiskReadShaped reports whether a default-denied call is safe for the
// admit-and-log posture. It is intentionally name-based and conservative: caller
// Meta is model-controlled and cannot widen authority. Folds case itself for
// cold callers; Adjudicate-path callers use lowRiskReadShapedLower.
func lowRiskReadShaped(tool string) bool {
	return lowRiskReadShapedLower(strings.ToLower(tool))
}

// lowRiskReadShapedLower is lowRiskReadShaped for an ALREADY lower-cased name.
func lowRiskReadShapedLower(lowerTool string) bool {
	if lowerTool == "" || writeShapedLower(lowerTool) {
		return false
	}
	for _, p := range []string{"read_", "get_", "search_", "list_", "lookup_", "find_", "calc"} {
		if strings.HasPrefix(lowerTool, p) {
			return true
		}
	}
	return lowerTool == "calculate"
}

// targetPath best-effort extracts a path-like argument from the call for the
// self-modify check, without trusting the model: it scans the decoded args.
func targetPath(args map[string]any) string {
	// Both snake_case (Claude Code: file_path) and camelCase (OpenCode / AI-SDK tools:
	// filePath) path args are scanned, so the self-modify glob check is not silently
	// bypassed by an agent that names the argument differently.
	for _, k := range []string{"path", "file_path", "filePath", "filepath", "file", "target", "filename", "dir"} {
		if v, ok := args[k]; ok {
			if s, ok := v.(string); ok {
				return s
			}
		}
	}
	return ""
}

// AdjudicateWithFSM evaluates the tool call through the decision table and tracks
// its lifecycle state through a LifecycleFSM instance.
func (a *Adjudicator) AdjudicateWithFSM(ctx context.Context, c *abi.ToolCall) (abi.Verdict, *LifecycleFSM) {
	fsm := NewLifecycleFSM()
	_, _ = fsm.Transition(EventEvaluate)
	v := a.Adjudicate(ctx, c)
	switch v.Kind {
	case abi.VerdictDeny:
		_, _ = fsm.Transition(EventDeny)
	case abi.VerdictQuarantine:
		_, _ = fsm.Transition(EventQuarantine)
	default:
		_, _ = fsm.Transition(EventAllow)
		_, _ = fsm.Transition(EventExecute)
	}
	_, _ = fsm.Transition(EventFinish)
	return v, fsm
}

// Adjudicate is the decision. It is pure and allocation-light on the deny/allow
// paths; only the (rare) TRANSFORM path resolves + re-stores args.
func (a *Adjudicator) Adjudicate(ctx context.Context, c *abi.ToolCall) (verdict abi.Verdict) {
	if a.recovery != nil && c != nil {
		defer func() { a.recordRecovery(ctx, c, verdict) }()
	}
	if c != nil {
		defer func() { a.evaluateCircuitBreaker(ctx, c, &verdict) }()
	}

	lowerTool := strings.ToLower(c.Tool) // folded once for every case-insensitive rung (#4007)

	state := a.state.Load()
	p := state.policy
	var argPreds []ArgPredicate
	if state.argByTool != nil {
		argPreds = state.argByTool[lowerTool]
	}

	// A name-level deny softens only when the operator declared its cited reason
	// advisory; POLICY_BLOCK and SECRET_EXFIL are clamped out of that set.
	// A per-call operator approval may lift a POLICY_BLOCK name deny; the args are
	// decoded only on that path so the no-deny fast path is unchanged.
	var args map[string]any
	var approval *operatorApprovalState
	if r, ok := p.Deny[c.Tool]; ok {
		v := p.soften(abi.Verdict{Kind: abi.VerdictDeny, Reason: r, By: "monitor"}, nil)
		if v.Kind != abi.VerdictDeny {
			return v
		}
		args = decodeArgs(ctx, c)
		approval = newOperatorApprovalState(p, c.Tool, args)
		v, lifted := approval.lift(v)
		if !lifted {
			return v
		}
	} else {
		args = decodeArgs(ctx, c)
		approval = newOperatorApprovalState(p, c.Tool, args)
	}
	defer func() { verdict = approval.finish(verdict) }()

	// The risk class comes from the decoded args, never model-controlled Meta, and
	// only when a profile is installed: a nil profile runs every rung (#666).
	pr := p.Profile
	var cl class
	if pr != nil {
		cl = riskClassLower(lowerTool, args)
	}

	// The host Read/grep/glob rewrite (#11150, #11499) is chosen here but applied
	// last: every rung still judges the ORIGINAL call and only its Allow is
	// rewritten, so the rewrite cannot launder a refused call into a dispatch.
	if rw, ok := transparentRewriteFor(lowerTool, args); ok {
		if pr.runs(cl, rungArgPredicate) {
			rw.tool, rw.preds = c.Tool, argPreds
		}
		defer func() { verdict = rw.admit(ctx, p, verdict) }()
	}

	if v, ok := a.selfModifyPathVerdict(ctx, p, c, lowerTool, args, cl, pr); ok {
		return v
	}

	// A shell tool carries its write target inside the command string and its
	// name has no write-word, so the path rung never fires for it (#172 Hole 1).
	// A read of a guarded file is not a self-modify. The architest wiring gates
	// pin this call and the synth-tool call below inside Adjudicate's body.
	if pr.runs(cl, rungCmdSelfModify) {
		if g := commandSelfModifyWithSpecs(args, p.BlockedPathGlobs, p.InlineEval); g != "" {
			return credentialBlockVerdict(g)
		}
		if g := commandSelfModifyWithSpecs(args, p.SelfModifyGlobs, p.InlineEval); g != "" {
			return p.soften(selfModifyDeny(g, abi.DenyRuleSelfModifyCommand), nil)
		}
	}

	// An exec of a script this agent authored earlier in the run (#543) carries no
	// write verb, so it is bound to the agent's envelope: it can reach no more than
	// the equivalent direct write. The ledger note follows the deny so a write into
	// a guarded tree never enters the ledger.
	if pr.runs(cl, rungSynthTool) {
		if g := synthToolSelfModify(args, &a.authored, p.SelfModifyGlobs); g != "" {
			return p.soften(selfModifyDeny(g, abi.DenyRuleSelfModifySynthTool), nil)
		}
		noteAuthoredScript(args, c.Tool, &a.authored, p.SelfModifyGlobs)
	}

	if v, ok := a.testImmunityVerdict(ctx, p, c, lowerTool, args); ok {
		return v
	}
	if v, ok := a.missionWriteSetVerdict(ctx, p, c, lowerTool, args); ok {
		return v
	}

	// advisoryNotes are bounded records of violated advisory arg rules; they ride
	// whatever admit this call resolves to, while a later hard deny stands alone.
	var advisoryNotes []string
	if cmd, ok := commandArg(args); ok {
		if nudge := CheckShellEditNudge(cmd, false); nudge.IsShellEdit {
			advisoryNotes = append(advisoryNotes, nudge.Suggestion)
		}
	}
	if pr.runs(cl, rungArgPredicate) && len(argPreds) > 0 {
		v, denied, notes := evalArgPredicates(argPreds, c.Tool, args)
		advisoryNotes = append(advisoryNotes, notes...)
		if denied {
			sv, lifted := approval.lift(p.soften(v, advisoryNotes))
			if !lifted {
				return sv
			}
		}
	}

	if v, ok := preAdmitVerdict(p, state.egressList, c, args, advisoryNotes, cl, pr); ok {
		// Only the dangerous-gotcha floor, the last pre-admit rung, is liftable here.
		sv, lifted := approval.lift(v)
		if !lifted {
			return sv
		}
	}

	redactedArgs, redacted := applyRedactionTransforms(p, argPreds, args, cl, pr)
	rev := a.runReversibilityRung(ctx, p, c, args, redactedArgs, redacted, lowerTool, cl, pr)
	if rev.handled {
		held, confirmed := approval.confirmHold(rev.verdict)
		if !confirmed {
			return held
		}
		rev = reversibilityRungResult{args: argsWithoutConfirmation(args), redactedArgs: redactedArgs, redacted: redacted}
	}
	v := a.admitOrDefaultDeny(ctx, p, c, lowerTool, rev, advisoryNotes)
	if v.Reason != abi.ReasonDefaultDeny {
		return v
	}
	v, lifted := approval.lift(v)
	if lifted {
		return allowWithNotes("monitor", advisoryNotes)
	}
	return v
}

func (a *Adjudicator) evaluateCircuitBreaker(ctx context.Context, c *abi.ToolCall, v *abi.Verdict) {
	if a == nil || c == nil || v == nil {
		return
	}
	cb := a.getCircuitBreaker()
	if cb == nil {
		return
	}

	sessionID := sessionIDFromContextOrCall(ctx, c)
	if sessionID == "" {
		return
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	// If the call was permitted or transformed, reset the counter for this session.
	if v.Kind != abi.VerdictDeny {
		delete(cb.sessions, sessionID)
		return
	}

	// The verdict is VerdictDeny.
	argHash := callArgFingerprint(c.Tool, c)
	entry, exists := cb.sessions[sessionID]
	if !exists || entry.tool != c.Tool || entry.reason != v.Reason || entry.argHash != argHash {
		// Non-repeating tool call (different tool, reason, or adapted arguments): reset counter to 1
		cb.sessions[sessionID] = &sessionRefusalState{
			tool:    c.Tool,
			reason:  v.Reason,
			argHash: argHash,
			count:   1,
			tripped: false,
		}
		return
	}

	// Consecutive identical refusal
	entry.count++
	if entry.count >= cb.threshold {
		entry.tripped = true
		originalReason := entry.reason
		v.Kind = abi.VerdictDeny
		v.Reason = ReasonDoomLoop
		v.Disposition = "DOOM_LOOP"
		if v.By == "" {
			v.By = "adjudicator"
		}
		if v.Meta == nil {
			v.Meta = make(map[string]string)
		}
		v.Meta["circuit_breaker"] = "tripped"
		v.Meta["remedy"] = "pivot to a different tool, adapt arguments, or emit a structured refusal"
		v.Meta["fix"] = "consecutive identical refusals detected; stop repeating this tool call and reconsider approach"
		v.Meta["streak"] = strconv.Itoa(entry.count)
		v.Meta["tool"] = c.Tool
		v.Meta["original_reason"] = abi.ReasonName(originalReason)
	}
}

// recordRecovery feeds this call's outcome to the refusal-effectiveness ledger.
func (a *Adjudicator) recordRecovery(ctx context.Context, c *abi.ToolCall, verdict abi.Verdict) {
	sessionID := sessionIDFromContextOrCall(ctx, c)
	turn := 0
	if c.Meta != nil && c.Meta["turn"] != "" {
		if t, err := strconv.Atoi(c.Meta["turn"]); err == nil {
			turn = t
		}
	}
	switch verdict.Kind {
	case abi.VerdictDeny:
		nextAction := ""
		if verdict.Meta != nil {
			nextAction = verdict.Meta["fix"]
			if nextAction == "" {
				nextAction = verdict.Meta["remedy"]
			}
		}
		a.recovery.RecordRefusal(sessionID, turn, abi.ReasonName(verdict.Reason), nextAction)
	case abi.VerdictQuarantine:
	default:
		a.recovery.RecordOutcome(sessionID, turn, c.Tool, OutcomeRecovered)
	}
}

func init() {
	// Rank 100: the authoritative monitor runs after cheaper pre-flight rungs but
	// the fold takes the most-restrictive verdict regardless of order.
	abi.RegisterAdjudicator(100, Default)
	abi.RegisterCapability("adjudicate.v1")
	// Register the egress rung's OUT-OF-TREE refusal name (the closed core vocabulary in
	// internal/abi is human-owned; RegisterReason is the sanctioned additive extension).
	// The adjudicator owns the call because it is already a wired, self-registering leaf —
	// so egressfloor stays a pure, init-free classifier and needs no defconfig entry.
	abi.RegisterReason(egressfloor.ReasonEgressBlock, egressfloor.ReasonEgressBlockName)
	abi.RegisterReason(ReasonMissionWriteSetViolation, ReasonMissionWriteSetViolationName)
	abi.RegisterReason(ReasonDoomLoop, ReasonDoomLoopName)
	abi.RegisterReason(ReasonCircuitBreakerTripped, ReasonCircuitBreakerTrippedName)
}
