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
	"encoding/json"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

// Policy is the decision table. A zero Policy is the fail-closed empty policy:
// nothing is affirmatively allowed, so every call resolves to DEFAULT_DENY.
type Policy struct {
	// Posture controls what happens after every provable refusal check has passed
	// but no affirmative allow matched. The zero value is fail-closed.
	Posture Posture
	// Allow lists tool names that are affirmatively permitted.
	Allow map[string]bool
	// AllowPrefix permits any tool whose name starts with one of these (e.g.
	// "read_", "get_", "search_", "list_" — the read-only family).
	AllowPrefix []string
	// Deny maps a tool name to the reason it is provably refused.
	Deny map[string]abi.ReasonCode
	// SelfModifyGlobs are path fragments that, if present in a write-shaped call's
	// target, prove a SELF_MODIFY attempt (the agent editing its own kernel).
	SelfModifyGlobs []string
	// BlockedPathGlobs are path fragments for credentials and host configuration that,
	// if present in a write-shaped call's target or shell command, are denied with POLICY_BLOCK.
	BlockedPathGlobs []string
	// RedactFields are arg keys whose presence triggers a TRANSFORM that strips the
	// value before dispatch (secret hygiene at the call boundary).
	RedactFields []string
	// ArgPredicates are per-tool ARGUMENT-VALUE constraints (issue #9). Every other
	// field keys on the tool NAME; these gate the tool's argument VALUES, turning
	// the floor from "which tool" into "which tool with which arguments". They are
	// RESTRICT-ONLY: a violated predicate turns an otherwise-allow into a Deny, but
	// a satisfied predicate never grants an allow on its own (a tool nothing else
	// allowed still falls to DEFAULT_DENY). Evaluated after the name-level Deny /
	// SelfModify checks and BEFORE the affirmative allow, so an allow-listed tool
	// invoked with a malicious argument is refused AT THE FLOOR, not waved through
	// to the (evadable) detection layer.
	ArgPredicates []ArgPredicate
	// LintWrites (opt-in, issue #536) turns on the in-process code-lint rung for
	// whole-file writes: a write of unparseable Go/JSON is refused with MALFORMED
	// before it lands — the in-kernel dual of codelint's advisory write-lint. Off
	// by default, so an existing floor is byte-for-byte unchanged unless an
	// operator asks for it. Only the Go and JSON grammars are consulted (they
	// parse in-process via the stdlib, so architest's TestHotPathHasNoExec stays
	// green); languages whose only checkers shell out (Python/CUDA) DEFER — fail
	// open, never denying over a quality signal the decide path cannot produce.
	LintWrites bool
	// Profile, when non-nil, ELIDES sub-rungs Adjudicate would otherwise always run,
	// per risk class (#665/#666, see riskClass + RungProfile). It NARROWS the floor
	// only — SetPolicy clamps any profile that tries to drop a mandatory write-class
	// refusal rung (sanitizeProfile / mustRun), so the write/self-modify floor can
	// never be widened. A nil Profile (the zero Policy and DefaultPolicy) runs the
	// fixed HEAD sequence byte-for-byte.
	Profile *RungProfile
	// Complain is the per-tool admit-and-log set (#670): a tool named here has its
	// DEFAULT_DENY downgraded to an admit-and-log Allow — admitted, with forensic
	// metadata — even when it is NOT read-shaped, so an operator can put a tool on a
	// logged trial without flipping the global Posture. It downgrades ONLY the
	// default-deny rung; the hard-refusal rungs (explicit Deny, self-modify, arg
	// violations) already returned before defaultDeny, so they still fail closed. An
	// empty/nil Complain set is byte-identical to HEAD (no tool is in complain mode).
	Complain map[string]bool
	// AdvisoryReasons is the operator-declared per-reason advisory (warn) posture —
	// the false-positive escape hatch for the HEURISTIC rungs. A monitor refusal
	// citing a reason in this set is downgraded to an admit-and-log Allow carrying
	// the would-deny record in Meta (posture=advisory, would_deny, the bounded
	// claim), so the decision journal keeps every would-deny. CLAMPED to
	// AdvisoryEligible (SELF_MODIFY, MALFORMED, DEFAULT_DENY) by New/SetPolicy —
	// the genuine-danger reasons (POLICY_BLOCK, SECRET_EXFIL, EGRESS_BLOCK) can
	// never be blanket-softened; use ArgPredicate.Advisory for one FP-prone rule.
	// Empty/nil keeps every rung enforcing (byte-identical to HEAD). See advisory.go.
	AdvisoryReasons map[abi.ReasonCode]bool
	// SecretPosture selects what the on-discovery secret rung (internal/secretgate,
	// #884/#885) does when a tool RESULT bears a credential: quarantine (the zero
	// value = today's behavior), fail_closed (deny), or admit_and_log (admit a
	// read-shaped result + record the would-deny). Additive — the zero value is
	// quarantine, so an unset posture is byte-for-byte the pre-#885 path. See
	// secretposture.go for the verdict mapping.
	SecretPosture SecretPosture
	// SecretPatterns are policy-declared EXTRA secret shapes, compiled at policy
	// load (a bad regex fails LOUD at load, not at runtime), UNIONED with the
	// canon.SecretPatterns floor at the gate — extend, never replace. Empty by
	// default (floor patterns only), so this too is additive.
	SecretPatterns []*regexp.Regexp
	// InlineEval extends the compiled inline-interpreter write floor. Entries are
	// validated by policy.Manifest.ToPolicy and unioned with the built-in specs.
	InlineEval []InlineEvalSpec
	// EgressExtraDenyHosts are operator-declared host names/IPs the egress rung refuses
	// IN ADDITION to the hardwired cloud-metadata / link-local class (manifest
	// egress.deny_hosts). It only ever TIGHTENS the floor — the hardwired metadata set
	// can never be disabled — so a deployment blocks its own sensitive endpoints (an
	// internal secrets service, a corp metadata mirror) without a code change. Empty by
	// default (hardwired set only), so this is additive.
	EgressExtraDenyHosts []string
	// ResearchEgressAllowHosts is the positive WebFetch allowlist for research
	// sub-agents. When non-empty, WebFetch is allowed only for URL hosts matching
	// one of these exact hosts or subdomains; every other WebFetch host is refused
	// with POLICY_BLOCK and a bounded host witness. Fetched bytes still flow
	// through the result-admission chain as untrusted data.
	ResearchEgressAllowHosts []string
	// EgressBlockHosts are operator-declared hosts the egress LIST layer refuses, with
	// subdomain coverage (a rule on "tracker.example" also refuses "cdn.tracker.example").
	// Unlike EgressExtraDenyHosts — which tightens the hardwired floor and cites its class
	// — these compose with EgressAllowHosts and EgressBlockLists under adblock precedence.
	// Empty by default, so this is additive. See egresslist.go.
	EgressBlockHosts []string
	// EgressAllowHosts are EXCEPTIONS: hosts carved back open even when a block rule (an
	// operator rule or a subscribed community list) would refuse them — adblock `@@`
	// semantics, resolved per host. Under EgressRestrict they become the total allowlist
	// rather than a set of exceptions. Empty by default.
	EgressAllowHosts []string
	// EgressBlockLists names bundled community block lists this policy subscribes to
	// (egresslist.BundledListNames). Names are validated LOUDLY at policy load, so an
	// unknown list is a load error, never a silently-empty subscription. Empty by default.
	EgressBlockLists []string
	// EgressRestrict inverts the egress posture from "reachable unless listed" to
	// "unreachable unless listed": a destination no EgressAllowHosts rule names is refused
	// POLICY_BLOCK even when no block rule matched it, and an empty allowlist under
	// restrict closes egress entirely. False by default (the additive posture), and it only
	// ever tightens.
	EgressRestrict bool

	// AutoRepairSidestep opts the reversibility rung into IN-FLIGHT REPAIR of a
	// sanctioned compiled sidestep: when a held call's family offers a machine-
	// applicable safe-subset substitution (today only a bare `git push` -> `fak sync
	// push`), emit that as a TRANSFORM instead of the preview-confirm hold. Default
	// false -- the hold is preserved, so every existing deployment is unchanged.
	// Only the SAFE subset is ever substituted: reversibility.go attaches a
	// RewriteCommand ONLY after its own safe-subset gate passes, so a --force /
	// --delete / refspec push carries none and still holds. Turning this on
	// therefore cannot launder a dangerous push into a weaker one -- the safe-subset
	// test lives at the producer, not here. Operators set it with
	// FAK_GUARD_AUTOREPAIR=sidestep (internal/policy, AutoRepairEnv).
	AutoRepairSidestep bool

	// TestLanes lists lane names explicitly designated as test lanes.
	TestLanes []string
	// ExemptLanes lists additional lanes exempt from test immunity.
	ExemptLanes []string
	// DisableTestImmunity disables the test-immunity gate entirely.
	DisableTestImmunity bool
	// Lane specifies the active policy lane name.
	Lane string
}

// Posture selects the policy's default-deny behavior after all provable refusal
// checks have passed.
type Posture = abi.Posture

const (
	// PostureFailClosed keeps the v0.1 floor: anything not affirmatively allowed
	// is denied with DEFAULT_DENY.
	PostureFailClosed = abi.PostureFailClosed
	// PostureAdmitAndLog downgrades low-risk read-shaped DEFAULT_DENY decisions
	// to ALLOW while carrying forensic metadata that records the would-have-denied
	// reason. Explicit denies, self-modify, arg-rule violations, and write-shaped
	// default denies still fail closed.
	PostureAdmitAndLog = abi.PostureAdmitAndLog
	// PostureDefaultOpen permits tools by default after all provable refusal
	// checks have passed, even when no affirmative allow matched.
	PostureDefaultOpen = abi.PostureDefaultOpen
)

// ArgKind selects which argument-value matcher an ArgPredicate applies.
type ArgKind uint8

const (
	// ArgAllowGlob is a POSITIVE requirement: the arg value must be a path UNDER
	// Glob (containment; a "../" escape fails). A missing arg is a violation —
	// the floor cannot prove containment of a value that is not there, so it
	// fails closed.
	ArgAllowGlob ArgKind = iota + 1
	// ArgDenyRegex is a NEGATIVE guard: an arg value matching Re is denied. A
	// missing arg matches nothing, so it is not a violation.
	ArgDenyRegex
	// ArgMaxBytes is a NEGATIVE guard: a string arg longer than N bytes is denied.
	ArgMaxBytes
	// ArgCLIReadOnly validates one gh/git invocation against a positive grammar and may attenuate gh search scope qualifiers.
	ArgCLIReadOnly
)

// ArgPredicate is the compiled, hot-path form of a policy arg rule (issue #9) —
// a constraint on one ARGUMENT of one tool. Construct these from the policy
// manifest (internal/policy), not by hand: the manifest validates exactly-one
// matcher, a closed-vocabulary Reason, and compiles Re once at load.
type ArgPredicate struct {
	Tool   string         // the tool name this constrains (exact match)
	Arg    string         // the argument key whose value is inspected
	Kind   ArgKind        // which matcher below is active
	Glob   string         // ArgAllowGlob: containment glob, e.g. "./out/**"
	Re     *regexp.Regexp // ArgDenyRegex: precompiled RE2 (nil for other kinds)
	N      int            // ArgMaxBytes: byte cap
	Reason abi.ReasonCode // refusal code cited on violation (manifest default: POLICY_BLOCK)
	// Advisory puts THIS rule on logged trial (manifest arg_rules[].advisory): a
	// violation no longer denies — it is noted (bounded, never the arg value) and
	// carried on the eventual verdict's Meta as advisory_violations, so an operator
	// can soften ONE false-positive-prone rule without uncoupling the rest of the
	// floor. The rule-granular dual of Policy.AdvisoryReasons.
	Advisory bool
	// Fix is the operator-authored sanctioned alternative (manifest
	// arg_rules[].fix), carried on the deny verdict's Meta["fix"] so every wire
	// can recommend it in the same breath as the refusal. Empty when the rule
	// declares none. Static manifest text only — never the arg value.
	Fix string
}

// Adjudicator is the reference monitor. Construct with New; the default instance
// registers itself in init().
type policyState struct {
	policy    Policy
	argByTool map[string][]ArgPredicate
	// egressList is the policy's egress rules compiled once at install time (nil when
	// the policy configures none, which Decides None for every host at zero cost).
	egressList *egresslist.List
}

type Adjudicator struct {
	state       atomic.Pointer[policyState]
	authored    sync.Map
	devEdit     atomic.Pointer[devEditAttestationState]
	receiptRoot string
	recovery    *RecoveryAuditLedger
}

// New builds an adjudicator with the given policy.
func New(p Policy) *Adjudicator {
	p.Profile = sanitizeProfile(p.Profile)                         // floor invariant: a profile may narrow only
	p.AdvisoryReasons = sanitizeAdvisoryReasons(p.AdvisoryReasons) // floor invariant: only heuristic reasons soften
	a := &Adjudicator{
		receiptRoot: receiptWorkspaceRoot(),
		recovery:    NewRecoveryAuditLedger(),
	}
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
	p.Profile = sanitizeProfile(p.Profile)                         // floor invariant: a profile may narrow only
	p.AdvisoryReasons = sanitizeAdvisoryReasons(p.AdvisoryReasons) // floor invariant: only heuristic reasons soften
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
		defer func() {
			sessionID := c.TraceID
			if sessionID == "" && c.Meta != nil {
				sessionID = c.Meta["session_id"]
			}
			turn := 0
			if c.Meta != nil && c.Meta["turn"] != "" {
				if t, err := strconv.Atoi(c.Meta["turn"]); err == nil {
					turn = t
				}
			}
			switch verdict.Kind {
			case abi.VerdictDeny:
				reason := abi.ReasonName(verdict.Reason)
				nextAction := ""
				if verdict.Meta != nil {
					nextAction = verdict.Meta["fix"]
					if nextAction == "" {
						nextAction = verdict.Meta["remedy"]
					}
				}
				a.recovery.RecordRefusal(sessionID, turn, reason, nextAction)
			case abi.VerdictQuarantine:
				// no recovery recorded
			default:
				a.recovery.RecordOutcome(sessionID, turn, c.Tool, OutcomeRecovered)
			}
		}()
	}

	lowerTool := strings.ToLower(c.Tool) // folded ONCE; every case-insensitive rung below reuses it (#4007)

	state := a.state.Load()
	p := state.policy
	var argPreds []ArgPredicate
	if state.argByTool != nil { // nil index (no arg predicates, the default floor) pays nothing
		argPreds = state.argByTool[lowerTool] // predicates targeting THIS tool (case-insensitive)
	}

	// Explicit provable refusal. Routed through soften like every monitor deny:
	// a name-level deny only downgrades when the operator declared its CITED
	// reason advisory (never POLICY_BLOCK / SECRET_EXFIL — those are clamped).
	if r, ok := p.Deny[c.Tool]; ok {
		return p.soften(abi.Verdict{Kind: abi.VerdictDeny, Reason: r, By: "monitor"}, nil)
	}

	// Decode args once for the structural checks.
	args := decodeArgs(ctx, c)

	// In-syscall transparent transform from Read to fak_read (#11150).
	if lowerTool == "read" && args != nil {
		var pathVal any
		found := false
		for _, key := range []string{"file_path", "filePath", "path"} {
			if v, ok := args[key]; ok && v != nil {
				pathVal = v
				found = true
				break
			}
		}
		if found {
			normalizedArgs := make(map[string]any, len(args))
			for k, v := range args {
				if k == "filePath" || k == "path" {
					continue
				}
				normalizedArgs[k] = v
			}
			normalizedArgs["file_path"] = pathVal
			if ref, ok := putJSON(ctx, normalizedArgs); ok {
				return abi.Verdict{
					Kind:    abi.VerdictTransform,
					By:      "monitor/read_to_fak_read",
					Payload: abi.TransformPayload{NewTool: "fak_read", NewArgs: ref},
					Meta:    map[string]string{"reversibility_autorepair": "read_to_fak_read"},
				}
			}
		}
	}

	// Coarse risk class for the RungProfile (#666). Computed ONCE from the DECODED
	// args (never model-controlled Meta), and ONLY when a profile is installed — a
	// nil profile runs every rung regardless (pr.runs == true), so the default floor
	// pays zero classification cost and stays byte-for-byte identical to HEAD.
	pr := p.Profile
	var cl class
	if pr != nil {
		cl = riskClassLower(lowerTool, args)
	}

	// SELF_MODIFY: a write-shaped call whose target matches a protected glob is a
	// PROVABLE refusal. Bounded disclosure: the witness carries ONLY the offending
	// glob, never the whole policy (deny channel is not a policy oracle).
	if v, ok := a.selfModifyPathVerdict(ctx, p, c, lowerTool, args, cl, pr); ok {
		return v
	}

	// SELF_MODIFY via the SHELL path (#172 Hole 1): a Bash/exec tool carries its
	// target INSIDE the command string, not a `path` arg, and its tool NAME
	// ("Bash") contains no write-word — so the file-write guard above never fires
	// for it. A shell write into a guarded tree (`sed -i internal/abi/x.go`,
	// `tee internal/kernel/y.go`, `> .dos/state`, `git apply`, `git checkout --`)
	// would otherwise launder a self-edit past the floor. Gate on a write-shaped
	// shell VERB targeting a guarded glob; a read of a guarded file (`cat`/`grep`
	// internal/abi/x.go) is NOT a self-modify, so it stays allowed. Bounded
	// disclosure: the witness names only the offending glob.
	if pr.runs(cl, rungCmdSelfModify) {
		if g := commandSelfModifyWithSpecs(args, p.BlockedPathGlobs, p.InlineEval); g != "" {
			return abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  abi.ReasonPolicyBlock,
				By:      "monitor/credential-block",
				Payload: abi.WitnessPayload{Claim: g},
				Meta:    denyRule(abi.DenyRuleCredentialPathBlock),
			}
		}
		if g := commandSelfModifyWithSpecs(args, p.SelfModifyGlobs, p.InlineEval); g != "" {
			return p.soften(abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  abi.ReasonSelfModify,
				By:      "monitor",
				Payload: abi.WitnessPayload{Claim: g},
				Meta:    denyRule(abi.DenyRuleSelfModifyCommand),
			}, nil)
		}
	}

	// SYNTHESIZED-TOOL invocation (#543): an exec of a script THIS agent authored
	// earlier this run is a self-synthesized tool, not an opaque binary. A
	// `python helper.py` carries no write verb / no inline-eval flag / no `>`, so
	// commandSelfModify above waves it through even when its command line reaches a
	// guarded tree. Bind the synth-tool to the agent's envelope: treat it as
	// write-shaped so one that reaches a guarded glob is denied SELF_MODIFY exactly
	// as the equivalent direct write is — a self-written tool can do no more than a
	// direct adjudicated call. Bounded disclosure: the witness names only the glob.
	// The deny half and the ledger half ride the SAME rung gate, so a profile that
	// elides the synth-tool rung skips both (the ledger note is inert for a call that
	// is not exec/command-shaped anyway).
	if pr.runs(cl, rungSynthTool) {
		if g := synthToolSelfModify(args, &a.authored, p.SelfModifyGlobs); g != "" {
			return p.soften(abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  abi.ReasonSelfModify,
				By:      "monitor",
				Payload: abi.WitnessPayload{Claim: g},
				Meta:    denyRule(abi.DenyRuleSelfModifySynthTool),
			}, nil)
		}
		// Record agent-authored scripts (the ledger half) so the NEXT exec is
		// recognized as a synth-tool. Placed AFTER the self-modify deny rung, so a
		// write into a guarded tree (already denied above) never lands in the ledger.
		noteAuthoredScript(args, c.Tool, &a.authored, p.SelfModifyGlobs)
	}

	// TEST_IMMUNITY: a write, edit, or delete targeting a gating test suite
	// (*_test.go, testdata/**, etc.) under an implementation lane is refused.
	if v, ok := a.testImmunityVerdict(ctx, p, c, lowerTool, args); ok {
		return v
	}

	// ARG-LEVEL value predicates (issue #9): the floor gates argument VALUES, not
	// just the tool name. A constrained arg that fails its predicate is a PROVABLE
	// refusal even for an otherwise allow-listed tool — denied here, never passed
	// to detection. Bounded disclosure: the witness names only the offending
	// tool.arg + the bound it broke, never the whole policy nor the arg value.
	// advisoryNotes accumulates the bounded records of violated ADVISORY arg rules
	// (ArgPredicate.Advisory): noted, never denied, attached to whatever verdict
	// this call ultimately resolves to — if it is an admit. A later hard deny
	// stands on its own (the deny already names its offense).
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
			return p.soften(v, advisoryNotes)
		}
	}

	// EGRESS: a tool call that reaches a blocked NETWORK DESTINATION — above all the
	// cloud-instance METADATA endpoint (169.254.169.254 & peers), whose only purpose
	// from inside a VM is to hand out the box's cloud IAM credentials — is a PROVABLE
	// refusal. This is the structural floor that makes `fak guard` useful on a random
	// cloud VM: a (possibly prompt-injected) agent cannot reach the metadata service to
	// steal the instance's credentials, with no human in the loop. The blocked set is
	// HARDWIRED in egressfloor (these addresses are never a legitimate destination), so
	// the rung needs no policy field and fires under the zero Policy too; it is mandatory
	// for every class (mustRun's default), so a RungProfile can never elide it. Bounded
	// disclosure: the witness names only the offending host + class, never the policy.
	if v, ok := egressRungVerdict(p, state.egressList, c.Tool, args, cl, pr); ok {
		return v
	}
	if v, ok := researchEgressVerdict(c.Tool, args, p.ResearchEgressAllowHosts); ok {
		return v
	}

	// LINT-WRITES (opt-in, #536): a whole-file write of unparseable code is a
	// PROVABLE refusal — Deny(MALFORMED) with a bounded file:line:col witness,
	// the in-kernel dual of codelint's advisory write-lint. Scoped to whole-file
	// writes so a partial edit (a fragment that would never parse standalone) is
	// never false-denied. The Go/JSON grammars parse in-process (stdlib, no exec,
	// so the decide path stays subprocess-free); any other language has no
	// in-process checker here and DEFERs (fail open — lint is a quality signal,
	// not a security gate). Bounded disclosure: the witness names only the first
	// finding, never the file content.
	if v, ok := lintWriteRungVerdict(p, c, args, advisoryNotes, cl, pr); ok {
		return v
	}

	// DANGEROUS GOTCHAS (#11258): fail-closed protection against destructive deletions,
	// raw disk operations, host evasion, privilege escalation, infrastructure teardown,
	// and critical system disruption across default-open posture.
	// Run before allow so dangerous gotchas are fail-closed.
	if p.Posture == PostureDefaultOpen {
		if v, ok := EvalDangerousGotchas(c.Tool, args); ok {
			return v
		}
	}

	redactedArgs, redacted := applyRedactionTransforms(p, argPreds, args, cl, pr)

	// REVERSIBILITY (#2156): before any path that would otherwise dispatch the
	// call (redact-transform, affirmative allow, or admit-and-log default allow),
	// hold irreversible/outward-facing calls behind a deterministic preview token.
	// Hard refusals above still win first, and a normally denied call is not handed
	// a preview token it cannot use.
	rev := a.runReversibilityRung(ctx, p, c, args, redactedArgs, redacted, lowerTool, cl, pr)
	if rev.handled {
		return rev.verdict
	}
	args, redactedArgs, redacted = rev.args, rev.redactedArgs, rev.redacted
	confirmedWithToken := rev.confirmedWithToken

	// TRANSFORM: redact a secret-shaped arg field before dispatch. An advisory
	// arg-rule note rides the transform's Meta so the logged trial is not lost
	// when the call is repaired-and-dispatched rather than plainly allowed.
	if redacted {
		if ref, ok := putJSON(ctx, redactedArgs); ok {
			v := abi.Verdict{Kind: abi.VerdictTransform, By: "monitor",
				Payload: abi.TransformPayload{NewArgs: ref}}
			if len(advisoryNotes) > 0 {
				v.Meta = map[string]string{"advisory_violations": strings.Join(advisoryNotes, "; ")}
			}
			return v
		}
	}

	// Affirmative allow.
	if v, ok := affirmativeAllowVerdict(ctx, p, c.Tool, args, confirmedWithToken, advisoryNotes); ok {
		return v
	}

	// exec_command's readOnlyHint is a classification request, not authority.
	// Structural refusals and explicit policy decisions above keep precedence;
	// only the narrow positive grammar may replace the default deny below.
	if v, ok := a.execCommandReadOnlyVerdict(c, args); ok {
		return v
	}

	if p.Posture == PostureDefaultOpen {
		if confirmedWithToken {
			if v, ok := stripConfirmationTransform(ctx, args, advisoryNotes); ok {
				return v
			}
		}
		meta := map[string]string{
			"posture": "default_open",
		}
		if len(advisoryNotes) > 0 {
			meta["advisory_violations"] = strings.Join(advisoryNotes, "; ")
		}
		return abi.Verdict{
			Kind: abi.VerdictAllow,
			By:   "monitor",
			Meta: meta,
		}
	}

	// Nothing affirmatively allowed it — fail-closed default deny.
	if confirmedWithToken && (p.admitAndLogLower(c.Tool, lowerTool) || p.AdvisoryReasons[abi.ReasonDefaultDeny]) {
		if v, ok := stripConfirmationTransform(ctx, args, advisoryNotes); ok {
			return v
		}
	}
	return defaultDeny(p, c.Tool, lowerTool, advisoryNotes)
}

// selfModifyPathVerdict is the SELF_MODIFY rung for write-shaped calls carrying a
// path ARG: a target matching a protected glob is a PROVABLE refusal. Bounded
// disclosure: the witness carries ONLY the offending glob, never the whole policy
// (deny channel is not a policy oracle). The shell-command and synth-tool halves
// of the rung stay inline in Adjudicate, where the architest wiring gates pin
// their calls. Returns the resolved verdict and whether it decided this call.
func (a *Adjudicator) selfModifyPathVerdict(ctx context.Context, p Policy, c *abi.ToolCall, lowerTool string, args map[string]any, cl class, pr *RungProfile) (abi.Verdict, bool) {
	if !pr.runs(cl, rungSelfModify) || !writeShapedLower(lowerTool) {
		return abi.Verdict{}, false
	}
	target := targetPath(args)
	if g := matchGlob(target, p.BlockedPathGlobs); g != "" {
		return abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonPolicyBlock,
			By:      "monitor/credential-block",
			Payload: abi.WitnessPayload{Claim: g},
			Meta:    denyRule(abi.DenyRuleCredentialPathBlock),
		}, true
	}
	if g := matchGlob(target, p.SelfModifyGlobs); g != "" {
		if directDevEditTool(c.Tool) && a.devEditAttested(ctx, c, target) {
			return abi.Verdict{Kind: abi.VerdictAllow, By: "monitor/dev-lease", Meta: map[string]string{"dev_attested": "true"}}, true
		}
		return p.soften(abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonSelfModify,
			By:      "monitor",
			Payload: abi.WitnessPayload{Claim: g},
			// Three different rungs cite SELF_MODIFY and disclose only the
			// glob, so on the wire they are one undifferentiated bucket. The rule
			// id separates them (#5863) — and it is what decides whether a
			// SELF_MODIFY on a `cd fak && …` compound came in through a path ARG or
			// through the shell command line, the hypothesis the journal could not
			// confirm.
			Meta: denyRule(abi.DenyRuleSelfModifyPath),
		}, nil), true
	}
	return abi.Verdict{}, false
}

// egressRungVerdict is the EGRESS rung: a tool call that reaches a blocked
// NETWORK DESTINATION is refused with the egressfloor class witness. The blocked
// set is HARDWIRED in egressfloor (these addresses are never a legitimate
// destination), so the rung needs no policy field and fires under the zero
// Policy too; it is mandatory for every class (mustRun's default), so a
// RungProfile can never elide it. Bounded disclosure: the witness names only
// the offending host + class, never the policy. Returns the verdict and whether
// it decided this call.
func egressRungVerdict(p Policy, egressList *egresslist.List, tool string, args map[string]any, cl class, pr *RungProfile) (abi.Verdict, bool) {
	if !pr.runs(cl, rungEgress) {
		return abi.Verdict{}, false
	}
	if host, label := egressfloor.Classify(tool, args, p.EgressExtraDenyHosts...); host != "" {
		return abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  egressfloor.ReasonEgressBlock,
			By:      "monitor",
			Payload: abi.WitnessPayload{Claim: label + ": " + host},
		}, true
	}
	// The operator-configurable band of the SAME rung, deliberately AFTER the
	// hardwired check above: block/allow lists and the restrict posture (egresslist.go).
	// Running second is what makes the floor un-openable — an allow rule naming the
	// metadata host never gets asked, because Classify has already returned.
	if v, ok := egressListVerdict(egressList, p, tool, args); ok {
		return v, true
	}
	return abi.Verdict{}, false
}

// lintWriteRungVerdict is the LINT-WRITES rung (opt-in, #536): a whole-file write
// of unparseable code is refused MALFORMED with a bounded file:line:col witness.
// The Go/JSON grammars parse in-process (stdlib, no exec); any other language has
// no in-process checker here and DEFERs (fail open). Bounded disclosure: the
// witness names only the first finding, never the file content. Returns the
// verdict and whether it decided this call.
func lintWriteRungVerdict(p Policy, c *abi.ToolCall, args map[string]any, advisoryNotes []string, cl class, pr *RungProfile) (abi.Verdict, bool) {
	if !pr.runs(cl, rungLintWrite) || !p.LintWrites || !wholeFileWrite(c.Tool) {
		return abi.Verdict{}, false
	}
	if w := lintWriteMalformed(targetPath(args), args); w != "" {
		return p.soften(abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonMalformed,
			By:      "monitor",
			Payload: abi.WitnessPayload{Claim: w},
		}, advisoryNotes), true
	}
	return abi.Verdict{}, false
}

// applyRedactionTransforms prepares the dispatch-time argument transform: the CLI
// grammar attenuation (an argument transform, not a textual advisory — a valid
// read-only command with forbidden gh search scope qualifiers is rewritten before
// dispatch; a non-read-only form was denied above) and the rungTransform
// secret-field redaction. Returns the transformed args (nil when nothing was
// rewritten) and whether any transform is pending.
func applyRedactionTransforms(p Policy, argPreds []ArgPredicate, args map[string]any, cl class, pr *RungProfile) (map[string]any, bool) {
	redactedArgs, redacted := map[string]any(nil), false
	for _, pred := range argPreds {
		if pred.Kind != ArgCLIReadOnly {
			continue
		}
		command, ok := argString(args, pred.Arg)
		if !ok {
			continue
		}
		rewritten, changed, _ := attenuateCLIGrammar(command)
		if changed {
			redactedArgs = cloneArgs(args)
			redactedArgs[pred.Arg] = rewritten
			redacted = true
		}
	}
	if pr.runs(cl, rungTransform) && len(p.RedactFields) > 0 && args != nil {
		if more, did := redact(selectArgs(redactedArgs, args), p.RedactFields); did {
			redactedArgs, redacted = more, true
		}
	}
	return redactedArgs, redacted
}

// reversibilityRungResult carries the reversibility rung's state mutations back
// to Adjudicate: the rung may replace the dispatch args (stripping the
// confirmation arg) and re-run the redaction over them, so Adjudicate must adopt
// the post-rung values on its transform/allow paths. handled/verdict carry an
// early verdict (the untracked-removal allow, the autorepair transform, or the
// hold); when handled is false the caller continues with the returned state.
type reversibilityRungResult struct {
	args               map[string]any
	redactedArgs       map[string]any
	redacted           bool
	confirmedWithToken bool
	verdict            abi.Verdict
	handled            bool
}

// runReversibilityRung is the REVERSIBILITY rung (#2156): before any path that
// would otherwise dispatch the call, hold irreversible/outward-facing calls
// behind a deterministic preview token. Hard refusals before it still win first,
// and a normally denied call is not handed a preview token it cannot use.
func (a *Adjudicator) runReversibilityRung(ctx context.Context, p Policy, c *abi.ToolCall, args map[string]any, redactedArgs map[string]any, redacted bool, lowerTool string, cl class, pr *RungProfile) reversibilityRungResult {
	confirmedWithToken := false
	if pr.runs(cl, rungReversibility) && (redacted || wouldAdmit(p, c.Tool, lowerTool)) {
		if a.selfAuthoredUntrackedRemoval(c, args) {
			return reversibilityRungResult{handled: true, verdict: abi.Verdict{Kind: abi.VerdictAllow, By: "monitor/reversibility", Meta: map[string]string{
				"witness": "trace-authored-git-untracked",
			}}}
		}
		env, ok := ReversibilityConfirmed(c.Tool, args)
		if !ok {
			// In-flight repair of a sanctioned compiled sidestep: when the operator opts
			// in (FAK_GUARD_AUTOREPAIR=sidestep) and the matched family offered a machine-
			// applicable substitution for THIS call, substitute the sanctioned verb instead
			// of holding. env carries a RewriteCommand only when the producer's safe-subset
			// gate passed (a bare `git push`, never a --force/--delete/refspec push), so the
			// dangerous variants never reach here with one and still take the hold below.
			if p.AutoRepairSidestep && env.RewriteCommand != "" {
				// Preserve every non-confirmation arg (workdir, timeout, description, ...) and
				// swap only the effect-bearing command, so the sanctioned verb still runs in
				// the same working directory the operator targeted -- dropping workdir here
				// would silently push a different repo.
				na := argsWithoutConfirmation(args)
				na["command"] = env.RewriteCommand
				if ref, ok := putJSON(ctx, na); ok {
					newTool := ""
					if !strings.EqualFold(c.Tool, env.RewriteTool) {
						newTool = env.RewriteTool // cross-tool sidestep (e.g. MCP git_push -> Bash)
					}
					return reversibilityRungResult{handled: true, verdict: abi.Verdict{
						Kind:    abi.VerdictTransform,
						By:      "monitor/reversibility",
						Payload: abi.TransformPayload{NewArgs: ref, NewTool: newTool},
						Meta: map[string]string{
							"reversibility_autorepair":   "sidestep",
							"reversibility_class":        string(env.Class),
							"reversibility_substitution": env.RewriteCommand,
						},
					}}
				}
			}
			return reversibilityRungResult{handled: true, verdict: reversibilityGateVerdict(env)}
		}
		confirmedWithToken = env.Class != ReversibilityReversible && hasConfirmationArg(args)
		if confirmedWithToken {
			args = argsWithoutConfirmation(args)
			if redacted {
				if more, did := redact(selectArgs(redactedArgs, args), p.RedactFields); did {
					redactedArgs, redacted = more, true
				}
			}
		}
	}
	return reversibilityRungResult{args: args, redactedArgs: redactedArgs, redacted: redacted, confirmedWithToken: confirmedWithToken}
}

// affirmativeAllowVerdict resolves the affirmative-allow rung: an explicit Allow
// entry or an AllowPrefix match admits the call (stripping a spent confirmation
// arg first when the reversibility rung confirmed with a token). Returns the
// verdict and whether an affirmative allow matched; a false result falls through
// to the read-only-hint and default-deny rungs.
func affirmativeAllowVerdict(ctx context.Context, p Policy, tool string, args map[string]any, confirmedWithToken bool, advisoryNotes []string) (abi.Verdict, bool) {
	if p.Allow[tool] {
		if confirmedWithToken {
			if v, ok := stripConfirmationTransform(ctx, args, advisoryNotes); ok {
				return v, true
			}
		}
		return allowWithNotes("monitor", advisoryNotes), true
	}
	for _, pre := range p.AllowPrefix {
		if strings.HasPrefix(tool, pre) {
			if confirmedWithToken {
				if v, ok := stripConfirmationTransform(ctx, args, advisoryNotes); ok {
					return v, true
				}
			}
			return allowWithNotes("monitor", advisoryNotes), true
		}
	}
	return abi.Verdict{}, false
}

// complainFor reports whether a tool is in the per-tool complain set (#670). A nil
// set admits nothing, so the zero Policy is unaffected.
func (p *Policy) complainFor(tool string) bool {
	return p.Complain[tool]
}

// admitAndLog reports whether a DEFAULT-denied tool should be downgraded to an
// admit-and-log Allow: the global read-shaped posture, OR the per-tool complain set
// (#670). It gates ONLY the default-deny rung — the hard-refusal rungs (explicit Deny,
// self-modify, arg violations) return before defaultDeny, so neither path can admit one.
func (p *Policy) admitAndLog(tool string) bool {
	return p.admitAndLogLower(tool, strings.ToLower(tool))
}

// admitAndLogLower is admitAndLog with the case fold already paid (#4007). The
// complain set stays keyed on the RAW name as authored; only the read-shape
// predicate is case-insensitive.
func (p *Policy) admitAndLogLower(tool, lowerTool string) bool {
	return (p.Posture == PostureAdmitAndLog && lowRiskReadShapedLower(lowerTool)) || p.complainFor(tool)
}

func wouldAdmit(p Policy, tool, lowerTool string) bool {
	if p.Posture == PostureDefaultOpen {
		return true
	}
	if p.Allow[tool] || p.admitAndLogLower(tool, lowerTool) || p.AdvisoryReasons[abi.ReasonDefaultDeny] {
		return true
	}
	for _, pre := range p.AllowPrefix {
		if strings.HasPrefix(tool, pre) {
			return true
		}
	}
	return false
}

// NeverAdmits (on the live Adjudicator) is the locked read of the installed floor's
// Policy.NeverAdmits — the args-independent "this name can never be Allowed" query the
// inbound tool-def compactor asks. Reads the current policy under the lock so it is safe
// to call per request from the serving path (where the floor lives in adjudicator.Default
// rather than a host-held Policy value). A pure read: it never mutates run-state.
func (a *Adjudicator) NeverAdmits(tool string) bool {
	p := a.state.Load().policy
	return p.NeverAdmits(tool)
}

// NeverAdmits reports whether the floor can NEVER produce an Allow for this tool
// NAME, for ANY argument value — the pure, args-independent question the inbound
// tool-def compactor (promptmmu) asks before it may safely drop a tool DEFINITION.
//
// True ⇔ the name is not affirmatively allowed (absent from Allow, matching no
// AllowPrefix) AND it would not be admitted-and-logged (so a read-shaped name under
// PostureAdmitAndLog, or a complain-set name, is NOT droppable — it can still be
// Allowed). Arg predicates can only RESTRICT an otherwise-allow, never grant one, so
// a never-allowed name stays never-allowed under every argument: dropping its
// advertisement is behavior-preserving. A pure read — no run-state mutation, no lock,
// safe to call per request — so the gateway can build its drop set without folding a
// real adjudication. Hard-refusal names (explicit Deny / self-modify globs) are ALSO
// never admitted, so they report true too; the inbound compactor only ever needs the
// "model can't reach it" guarantee, which both classes satisfy.
func (p Policy) NeverAdmits(tool string) bool {
	if r, denied := p.Deny[tool]; denied {
		// A name-deny whose cited reason is advisory can still be admitted (the
		// soften downgrade), so its tool-def must not be pruned.
		return !p.AdvisoryReasons[r]
	}
	if p.Posture == PostureDefaultOpen {
		return false
	}
	// Fail-safe against an UNCONFIGURED floor: a Policy with no affirmative-allow
	// surface at all (empty Allow, empty AllowPrefix, fail-closed posture) denies
	// EVERY tool — true by the rule below, but as a DROP signal that is almost always
	// "the floor was never installed" rather than "deliberately deny all advertised
	// tools." Pruning every tool-def against a zero floor would be a catastrophic
	// over-drop, so we refuse to prune anything when there is nothing to admit. A real
	// floor (any Allow entry, AllowPrefix, or research WebFetch allowlist) re-enables
	// pruning of the names it genuinely never admits.
	if len(p.Allow) == 0 && len(p.AllowPrefix) == 0 && len(p.ResearchEgressAllowHosts) == 0 {
		return false
	}
	if p.Allow[tool] {
		return false
	}
	for _, pre := range p.AllowPrefix {
		if strings.HasPrefix(tool, pre) {
			return false
		}
	}
	if strings.EqualFold(tool, "WebFetch") && len(p.ResearchEgressAllowHosts) > 0 {
		return false
	}
	// Advisory DEFAULT_DENY admits any name (with the would-deny record), so
	// nothing is never-admitted under it.
	return !p.admitAndLog(tool) && !p.AdvisoryReasons[abi.ReasonDefaultDeny]
}

func defaultDeny(p Policy, tool, lowerTool string, advisoryNotes []string) abi.Verdict {
	if p.Posture == PostureDefaultOpen {
		meta := map[string]string{
			"posture": "default_open",
		}
		if len(advisoryNotes) > 0 {
			meta["advisory_violations"] = strings.Join(advisoryNotes, "; ")
		}
		return abi.Verdict{
			Kind: abi.VerdictAllow,
			By:   "monitor",
			Meta: meta,
		}
	}
	if p.admitAndLogLower(tool, lowerTool) {
		// Admit-and-log record (#671): the default-deny rung is the refusal being
		// suppressed, so the record carries would_deny = its reason name via
		// abi.ReasonName — the forensic field the promotion ledger (#672) folds. Both
		// the complain-set and the global read-shaped path carry it identically.
		meta := map[string]string{
			"posture":    "admit_and_log",
			"would_deny": abi.ReasonName(abi.ReasonDefaultDeny),
		}
		if len(advisoryNotes) > 0 {
			meta["advisory_violations"] = strings.Join(advisoryNotes, "; ")
		}
		return abi.Verdict{
			Kind: abi.VerdictAllow,
			By:   "monitor",
			Meta: meta,
		}
	}
	// Advisory DEFAULT_DENY (Policy.AdvisoryReasons) admits ANY tool with the
	// would-deny record — the strictly-wider dev dual of admit_and_log, carrying
	// posture=advisory so the promotion ledger's admit_and_log fold is untouched.
	return p.soften(abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "monitor"}, advisoryNotes)
}

func reversibilityGateVerdict(env ReversibilityEnvelope) abi.Verdict {
	claim, err := json.Marshal(env)
	if err != nil {
		claim = []byte(`{"class":"` + string(env.Class) + `"}`)
	}
	meta := map[string]string{
		"reversibility_class": string(env.Class),
		"preview":             env.Preview,
		"confirm_token":       env.ConfirmToken,
	}
	if env.DryRunHint != "" {
		meta["dry_run_hint"] = env.DryRunHint
	}
	return abi.Verdict{
		Kind:    abi.VerdictRequireWitness,
		By:      "monitor/reversibility",
		Payload: abi.WitnessPayload{Claim: string(claim)},
		Meta:    meta,
	}
}

func stripConfirmationTransform(ctx context.Context, args map[string]any, advisoryNotes []string) (abi.Verdict, bool) {
	ref, ok := putJSON(ctx, args)
	if !ok {
		return abi.Verdict{}, false
	}
	v := abi.Verdict{
		Kind:    abi.VerdictTransform,
		By:      "monitor",
		Payload: abi.TransformPayload{NewArgs: ref},
		Meta:    map[string]string{"reversibility_confirmed": "true"},
	}
	if len(advisoryNotes) > 0 {
		v.Meta["advisory_violations"] = strings.Join(advisoryNotes, "; ")
	}
	return v, true
}

func researchEgressVerdict(tool string, args map[string]any, allowHosts []string) (abi.Verdict, bool) {
	if !strings.EqualFold(tool, "WebFetch") || len(allowHosts) == 0 {
		return abi.Verdict{}, false
	}
	raw, _ := args["url"].(string)
	scheme, host := webURLParts(raw)
	if host == "" {
		return abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonMalformed,
			By:      "monitor/research-egress",
			Payload: abi.WitnessPayload{Claim: "research egress missing WebFetch.url"},
		}, true
	}
	if scheme != "http" && scheme != "https" {
		return abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonPolicyBlock,
			By:      "monitor/research-egress",
			Payload: abi.WitnessPayload{Claim: "research egress unsupported WebFetch scheme: " + scheme},
		}, true
	}
	if researchHostAllowed(host, allowHosts) {
		return abi.Verdict{
			Kind: abi.VerdictAllow,
			By:   "monitor/research-egress",
			Meta: map[string]string{
				"research_egress": "allowlisted",
				"host":            host,
			},
		}, true
	}
	return abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  abi.ReasonPolicyBlock,
		By:      "monitor/research-egress",
		Payload: abi.WitnessPayload{Claim: "research egress host not allowlisted: " + host},
	}, true
}

func webURLParts(raw string) (scheme, host string) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Scheme == "" || u.Host == "" {
		return "", ""
	}
	return strings.ToLower(u.Scheme), strings.ToLower(strings.Trim(u.Hostname(), "[]"))
}

func researchHostAllowed(host string, allowHosts []string) bool {
	host = strings.ToLower(strings.Trim(strings.TrimSpace(host), "[]"))
	for _, allowed := range allowHosts {
		a := strings.ToLower(strings.Trim(strings.TrimSpace(allowed), "[]"))
		if a == "" {
			continue
		}
		if host == a || strings.HasSuffix(host, "."+a) {
			return true
		}
	}
	return false
}

func decodeArgs(ctx context.Context, c *abi.ToolCall) map[string]any {
	b := refBytes(ctx, c.Args)
	if len(b) == 0 {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil
	}
	return m
}

func refBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	res := abi.ActiveResolver()
	if res == nil {
		return nil
	}
	b, err := res.Resolve(ctx, r)
	if err != nil {
		return nil
	}
	return b
}

func putJSON(ctx context.Context, m map[string]any) (abi.Ref, bool) {
	b, err := json.Marshal(m)
	if err != nil {
		return abi.Ref{}, false
	}
	res := abi.ActiveResolver()
	if res == nil {
		return abi.Ref{}, false
	}
	ref, err := res.Put(ctx, b)
	if err != nil {
		return abi.Ref{}, false
	}
	return ref, true
}

func cloneArgs(args map[string]any) map[string]any {
	out := make(map[string]any, len(args))
	for k, v := range args {
		out[k] = v
	}
	return out
}
func selectArgs(preferred, fallback map[string]any) map[string]any {
	if preferred != nil {
		return preferred
	}
	return fallback
}

func redact(args map[string]any, fields []string) (map[string]any, bool) {
	changed := false
	out := make(map[string]any, len(args))
	for k, v := range args {
		out[k] = v
	}
	for _, f := range fields {
		if _, ok := out[f]; ok {
			out[f] = "[REDACTED]"
			changed = true
		}
	}
	return out, changed
}

// matchGlob returns the first glob fragment contained in path, or "".
//
// A fragment that names a DOTFILE (leading '.', e.g. ".env", ".aws/") must
// additionally occur at a token boundary: the byte before the match, if any,
// may not be a word byte. Bare substring matching turned the most ordinary
// env READ there is — `python -c "… os.environ …"` — into a SELF_MODIFY deny,
// because the inline-eval floor hands the whole opaque segment to this matcher
// as the write target and "os.environ" contains ".env" (witnessed live
// 2026-07-01: `python -c "import os; print(len(os.environ))"` denied with
// witness ".env"). A genuine dotfile target always follows a separator —
// `> .env`, `src/.env`, `--env-file=.env`, `~/.aws/credentials`, a leading
// `.env` — so every real deny is kept; only the mid-identifier '.' of a dotted
// name ("os.environ", "config.env", "repo.git/") stops counting as the dotfile.
func matchGlob(path string, globs []string) string {
	if path == "" {
		return ""
	}
	for _, g := range globs {
		if g == "" {
			continue
		}
		if g[0] != '.' {
			if strings.Contains(path, g) {
				return g
			}
			if strings.HasSuffix(g, "/") && treeDirFragmentIn(path, strings.TrimSuffix(g, "/")) {
				return g
			}
			continue
		}
		if dotfileFragmentIn(path, g) {
			return g
		}
	}
	return ""
}

// dotfileFragmentIn reports whether the dotfile fragment g occurs in path at a
// token boundary — the start of the string, or preceded by a non-word byte
// (a path separator, whitespace, a quote, '=', shell punctuation, …).
func dotfileFragmentIn(path, g string) bool {
	return fragmentAt(path, g, func(at, _ int) bool {
		return at == 0 || !wordByte(path[at-1])
	})
}

// treeDirFragmentIn reports whether a slash-bearing guarded-tree fragment occurs
// as the tree itself, not only as a parent prefix ending in '/'. This preserves
// the "internal/abi/" glob for descendants while also catching destructive calls
// that name the directory exactly, such as rmtree("internal/abi").
func treeDirFragmentIn(path, g string) bool {
	if g == "" || !strings.Contains(g, "/") {
		return false
	}
	return fragmentAt(path, g, func(at, end int) bool {
		leftOK := at == 0 || !wordByte(path[at-1])
		rightOK := end == len(path) || !wordByte(path[end])
		return leftOK && rightOK
	})
}

func fragmentAt(path, fragment string, accept func(at, end int) bool) bool {
	for from := 0; ; {
		i := strings.Index(path[from:], fragment)
		if i < 0 {
			return false
		}
		at := from + i
		if accept(at, at+len(fragment)) {
			return true
		}
		from = at + 1
	}
}

// wordByte reports whether b continues an identifier/word — a byte that, when
// it precedes a '.', makes that '.' the dot OF a dotted name rather than the
// start of a dotfile. Everything else is a boundary.
func wordByte(b byte) bool {
	return b == '_' || b == '-' ||
		(b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
}

// interpreterEvalSpec pairs a general-purpose interpreter with the inline-program flags
// whose presence as a TOKEN means it runs code from an opaque string argument able to
// write a file directly.
type InlineEvalSpec struct {
	Interp string
	Flags  []string
}

type interpreterEvalSpec struct {
	interp string   // the interpreter named as a command word (`ruby`, `node`, …)
	flags  []string // its inline-eval flags (`-e`, `--eval`, `-c`, `-p`/`--print`)
}

// interpreterEvalFlags is the adjudicator's inline-eval write floor (#172 Hole 1
// residual): the general-purpose interpreters most likely on a coding agent's PATH —
// python, node, and ruby — each paired with the inline-program flags that run an opaque
// program string. It is the interpreter analogue of shellWriteVerbs; commandWrites ranges
// over it and treats such a command as write-shaped, routing it through the SAME
// commandSelfModify guard the shell floor uses.
//
// ruby joins python/node to close an asymmetry the rulesynth RSI loop (internal/rsiloop,
// run with -harness rulesynth) mined from the near-miss corpus and the keep-bit KEPT:
// `ruby -i` (an in-place edit) is already caught by shellWriteVerbs, but `ruby -e
// 'File.write("internal/adjudicator/decide.go", …)'` — the EVAL flag doing the same
// self-edit — slipped every rung. `perl -e`, `php -r`, and `lua -e` are the same shape and
// remain the residual queue the loop's corpus drives next, so they are deliberately not
// listed here yet.
//
// Detection is by TOKEN, not a fixed `<interp> <flag> ` prefix (see interpreterEvalMatch):
// the interpreter need only appear as a word and the flag as its own argument, so the
// idiomatic no-space, quoted, and `=`-joined spellings (`ruby -e'…'`, `node --eval=…`) and
// intervening flags (`ruby -rjson -e …`) are all caught — closing the porous-prefix gap a
// fixed-spelling table leaves open. The identifier name `interpreterEvalFlags` is pinned by
// architest (TestInlineEvalFloorWiredInCommandWrites); rename only with that gate's constant.
var interpreterEvalFlags = []interpreterEvalSpec{
	{"python3", []string{"-c"}},
	{"python", []string{"-c"}},
	{"node", []string{"-e", "--eval", "-p", "--print"}},
	{"ruby", []string{"-e", "--eval"}},
}

// DefaultPolicy is the v0.1 baseline: allow the read-only tool family + the
// frozen tau2 trace tools, deny a self-modify glob set, redact common secret arg
// fields. Tuned to be permissive enough to drive the bench yet fail-closed on
// unknown + self-modifying calls.
func DefaultPolicy() Policy {
	return Policy{
		Allow: map[string]bool{
			"search_flights": true, "get_reservation_details": true,
			"get_user_details": true, "list_all_airports": true,
			"calculate": true, "search_direct_flight": true,
			"transfer_to_human_agents": true, "send_certificate": true,
			"book_reservation": true, "update_reservation_flights": true,
		},
		AllowPrefix: []string{"read_", "get_", "search_", "list_", "lookup_", "find_", "calc"},
		Deny: map[string]abi.ReasonCode{
			"shell_rm_rf": abi.ReasonPolicyBlock,
			"exfiltrate":  abi.ReasonSecretExfil,
		},
		// Witness machinery (#172 Hole 2): internal/architest + internal/shipgate
		// are covered by the fak/internal/ catch-all, but dos.toml (the lane
		// taxonomy + stamp grammar the referee binds to) is not — name it so a
		// self-improving loop cannot quietly rewrite the gate that witnesses it.
		SelfModifyGlobs: []string{"internal/abi/", "internal/kernel/", "internal/adjudicator/", "internal/architest/", "internal/shipgate/", "dos.toml", ".dos/", "fak/internal/"},
		RedactFields:    []string{"password", "secret", "api_key", "token", "authorization"},
	}
}

// DevAgentPolicy is the deployable floor for a CODING agent that runs against this
// repo (the dogfood case — fak passing its own CICD test). It differs from the
// permissive bench DefaultPolicy in three load-bearing ways:
//
//   - it DENIES the shared-history git mutations (push / merge / tag): a dev agent
//     adapts code, it never moves the branch on its own say-so;
//   - it bounds writes AWAY from the spine — a write whose target touches the ABI,
//     the kernel, the adjudicator, the policy loader, the defconfig, or the .git
//     dir is a SELF_MODIFY (ESCALATE), not a silent edit;
//   - it ALLOWS a single high-level `ship_release` action that is NOT free: the
//     shipgate adjudicator holds it behind the require-witness rung, so it proceeds
//     only when a claimed ship is corroborated by git evidence the agent did not
//     author (an unwitnessed ship is refused).
//
// Mirrored on disk as examples/dev-agent-policy.json (round-trips through the
// policy manifest loader), so an adopter selects it with `--policy` — no fork.
func DevAgentPolicy() Policy {
	return Policy{
		Posture: PostureDefaultOpen,
		Allow: map[string]bool{
			// safe inspect / build / test tools a coding agent drives
			"Read":       true,
			"fak_read":   true,
			"git_status": true, "git_diff": true, "git_log": true,
			"go_build": true, "go_test": true, "run_tests": true,
			// the high-level ship action: allowed at the floor, but witness-gated by
			// the shipgate adjudicator (a claimed ship must be git-corroborated).
			"ship_release": true,
		},
		AllowPrefix: []string{"read_", "get_", "search_", "list_", "lookup_", "find_", "calc"},
		Deny: map[string]abi.ReasonCode{
			"git_push":   abi.ReasonPolicyBlock,
			"git_merge":  abi.ReasonPolicyBlock,
			"git_tag":    abi.ReasonPolicyBlock,
			"exfiltrate": abi.ReasonSecretExfil,
		},
		SelfModifyGlobs: []string{
			"internal/abi/", "internal/kernel/", "internal/adjudicator/",
			"internal/policy/", "internal/registrations/",
			// Witness machinery (#172 Hole 2): the trees that, if self-edited,
			// would let an RSI loop grade its own homework — the architest gates,
			// the shipgate require-witness rung, and the workspace's dos.toml
			// (lane taxonomy + stamp grammar the referee binds to).
			"internal/architest/", "internal/shipgate/", "dos.toml",
			".git/", ".dos/", "policy.json", "VERSION", "id_rsa",
		},
		RedactFields: []string{"password", "secret", "api_key", "token", "authorization"},
	}
}

// Default is the registered instance.
var Default = New(DefaultPolicy())

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
}
