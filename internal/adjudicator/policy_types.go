package adjudicator

import (
	"regexp"

	"github.com/anthony-chaudhary/fak/internal/abi"
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
	// OperatorApprovals is the set of per-call operator approval ids
	// (OperatorApprovalID). An approved call may pass a POLICY_BLOCK arg rule, a
	// POLICY_BLOCK name deny, or the default deny, and answers the reversibility
	// confirm hold, for that exact call only. It never lifts the credential block,
	// self-modify, SECRET_EXFIL or egress. Empty/nil changes nothing. See
	// operator_approval.go.
	OperatorApprovals map[string]bool
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
	// ArgAllowExact requires the original string to equal Glob byte-for-byte.
	// It performs no path, quoting, whitespace, or environment normalization.
	ArgAllowExact
)

// ArgPredicate is the compiled, hot-path form of a policy arg rule (issue #9) —
// a constraint on one ARGUMENT of one tool. Construct these from the policy
// manifest (internal/policy), not by hand: the manifest validates exactly-one
// matcher, a closed-vocabulary Reason, and compiles Re once at load.
type ArgPredicate struct {
	Tool   string         // the tool name this constrains (exact match)
	Arg    string         // the argument key whose value is inspected
	Kind   ArgKind        // which matcher below is active
	Glob   string         // ArgAllowGlob: containment glob; ArgAllowExact: literal string
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
