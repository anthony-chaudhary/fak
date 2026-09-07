// Package policy loads the adjudicator's capability floor from a declarative,
// version-tagged JSON manifest instead of a compiled-in Go literal — so an
// adopter configures WHICH tools the agent may call by editing a file the
// operator can read and a reviewer can diff, never by forking the kernel and
// recompiling.
//
// This is the deployable form of the project's "permissions as the floor"
// thesis. The manifest IS the allow-list: anything not affirmatively allowed
// resolves to the fail-closed DEFAULT_DENY, and every explicit deny cites a code
// from the CLOSED refusal vocabulary (internal/abi/reasons.go), so a policy is
// verifiable and lintable, not free text.
//
// Zero new dependencies: the manifest is stdlib JSON. The schema maps 1:1 to
// adjudicator.Policy, with deny reasons named by their stable string and
// validated against abi.ReasonByName at load time. Unknown JSON fields are
// REJECTED (DisallowUnknownFields) so a typo in a hand-authored manifest —
// "allows" for "allow" — fails loudly at the boundary instead of silently
// widening or narrowing the floor.
package policy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/egresslist"
	"github.com/anthony-chaudhary/fak/internal/maputil"
	"github.com/anthony-chaudhary/fak/internal/provenance"
)

// Version is the current manifest schema tag. A manifest MAY omit it (treated as
// the current version); a manifest naming a different MAJOR is refused. Minor
// bumps (fak-policy/v1.x) are forward-accepted so an older binary tolerates a
// newer minor manifest.
const Version = "fak-policy/v1"

const (
	postureFailClosed  = "fail_closed"
	postureAdmitAndLog = "admit_and_log"
	postureDefaultOpen = "default_open"
)

// Exported posture string constants matching manifest schema.
const (
	PostureDefaultOpen = "default_open"
	PostureFailClosed  = "fail_closed"
	PostureAdmitAndLog = "admit_and_log"
)

// Manifest is the on-disk schema. It maps 1:1 to adjudicator.Policy, but names
// deny reasons by their stable refusal string (validated against the closed
// vocabulary) so the file is human-readable and reviewable.
//
// Field semantics mirror adjudicator.Policy exactly:
//   - Allow / AllowPrefix: a call is affirmatively permitted if its tool name is
//     in Allow or starts with one of AllowPrefix. Everything else is DEFAULT_DENY.
//   - Deny: an explicit provable refusal, value = a closed-vocabulary reason name.
//   - SelfModifyGlobs: path fragments that, in a write-shaped call's target, prove
//     a SELF_MODIFY attempt.
//   - RedactFields: arg keys whose value is stripped (TRANSFORM) before dispatch.
//   - Posture: default-deny behavior; omitted means fail_closed. admit_and_log
//     admits low-risk read-shaped DEFAULT_DENY calls with forensic metadata, but
//     does not soften explicit denies, self-modify, arg-rule violations, or writes.
//   - SafeSinks / Authorize / Sources: IFC config for post-read egress precision
//     and host-authored source classes.
type Manifest struct {
	Version             string            `json:"version,omitempty"`
	Profile             string            `json:"profile,omitempty"`
	Posture             string            `json:"posture,omitempty"`
	Allow               []string          `json:"allow,omitempty"`
	AllowPrefix         []string          `json:"allow_prefix,omitempty"`
	Complain            []string          `json:"complain,omitempty"`
	Deny                map[string]string `json:"deny,omitempty"`
	SelfModifyGlobs     []string          `json:"self_modify_globs,omitempty"`
	BlockedPathGlobs    []string          `json:"blocked_path_globs,omitempty"`
	CredentialPathGlobs []string          `json:"credential_path_globs,omitempty"`
	RedactFields        []string          `json:"redact_fields,omitempty"`
	// AdvisoryReasons is the per-reason advisory (warn) posture — the operator's
	// false-positive escape hatch for the HEURISTIC rungs. A monitor refusal citing
	// a listed reason is admitted with the full would-deny record in its verdict
	// Meta (posture=advisory, would_deny, the bounded claim), journaled like every
	// verdict. Only the heuristic reasons are accepted (adjudicator.AdvisoryEligible:
	// SELF_MODIFY, MALFORMED, DEFAULT_DENY); naming a genuine-danger reason
	// (POLICY_BLOCK, SECRET_EXFIL, EGRESS_BLOCK) fails loud at load — soften one
	// FP-prone rule with arg_rules[].advisory instead. The FAK_ADVISORY_REASONS env
	// var (comma-separated reasons, or "dev" for all three) UNIONS into this set at
	// load — the one-line dev-session override that needs no manifest edit; it is
	// read in the fak process that enforces the floor, so a guarded child agent
	// cannot flip it. Maps to adjudicator.Policy.AdvisoryReasons.
	AdvisoryReasons []string `json:"advisory_reasons,omitempty"`
	// SecretPosture (issue #885) selects what the on-discovery secret rung does when
	// a tool result bears a credential: "quarantine" (default, omitted), "fail_closed",
	// or "admit_and_log". An unknown token is refused at load. Maps to
	// adjudicator.Policy.SecretPosture.
	SecretPosture string `json:"secret_posture,omitempty"`
	// SecretPatterns are RE2 strings for EXTRA secret shapes, compiled at load (a bad
	// pattern fails loud) and unioned with the canon floor at the gate. Maps to
	// adjudicator.Policy.SecretPatterns.
	SecretPatterns []string          `json:"secret_patterns,omitempty"`
	InlineEval     []InlineEvalSpec  `json:"inline_eval,omitempty"`
	SafeSinks      []string          `json:"safe_sinks,omitempty"`
	Authorize      []AuthorizeRule   `json:"authorize,omitempty"`
	Sources        map[string]string `json:"sources,omitempty"`
	// ArgRules are per-tool ARGUMENT-VALUE constraints (issue #9) — the manifest
	// form of adjudicator.ArgPredicate. They extend the floor from "which tool"
	// to "which tool with which argument values". See ArgRule for the matchers.
	ArgRules []ArgRule `json:"arg_rules,omitempty"`
	// LintWrites (opt-in, issue #536) turns on the in-process code-lint rung for
	// whole-file writes: a write of unparseable Go/JSON is refused with MALFORMED
	// before it lands. Off by default; languages whose only checkers shell out
	// (Python/CUDA) defer (fail open). Maps 1:1 to adjudicator.Policy.LintWrites.
	LintWrites bool `json:"lint_writes,omitempty"`
	// RateLimit (issue #699, Epic 8) is the declarative throughput/cost cap applied
	// to ratelimit.Default at boot and on --policy hot-reload. Absent (or an empty
	// manifest) leaves the limiter inert (it Defers on every call); a present block
	// installs the cap and is authoritative over the FAK_RATELIMIT_* env fallback.
	// See RateLimitRule. This is manifest/runtime-only — NOT an adjudicator.Policy
	// field (rate config is separate from the name-level allow/deny floor).
	RateLimit *RateLimitRule `json:"rate_limit,omitempty"`
	// Egress (optional) extends the hardwired cloud-metadata / link-local egress floor.
	// deny_hosts only TIGHTENS the floor; research_allow_hosts is a positive,
	// WebFetch-only allowlist for research sub-agents. The hardwired metadata block
	// (169.254.169.254 & peers) is always on and cannot be disabled here.
	Egress *EgressRule `json:"egress,omitempty"`
	// Isolation (issue #2013, epic #2000 M13) is the declarative trust-level →
	// ToolExec-backend dial. Absent leaves the dial unset (resolution fails
	// closed); a present block is validated at load — see IsolationRule. Like
	// RateLimit, this is manifest/runtime-only, NOT an adjudicator.Policy field.
	Isolation *IsolationRule `json:"isolation,omitempty"`
	// ToolRuntime (seam 5 of the tool-process table) grants each tool its
	// runtime envelope — how long it may run and at what heartbeat cadence it
	// must report — in the same manifest that grants the capability: "you may
	// run this tool" and "you may run it for this long" are one grant. Rows
	// are validated at load (see ToolRuntimeRule); embedders of the toolproc
	// supervisor resolve per-spawn envelopes via Runtime.ToolRuntime. Like
	// RateLimit, manifest/runtime-only, NOT an adjudicator.Policy field.
	ToolRuntime []ToolRuntimeRule `json:"tool_runtime,omitempty"`
	// InheritedCapabilities declares the capability subset a brokered child
	// launch may inherit from its parent. Absent means default-deny: no ambient
	// env, secret value, cwd, writable/persistence scope, or egress reference is
	// passed. Like ToolRuntime, this is manifest/runtime-only; launch adapters
	// resolve it through Runtime.InheritedCapabilities before exec.
	InheritedCapabilities []InheritedRule `json:"inherited_capabilities,omitempty"`
	// SubagentDepth (issue #2603, epic #2000) caps how deep a subagent fan-out
	// tree may recurse — the policy-bound form of the harness's depth bound so an
	// unbounded child-spawns-child tree cannot run away. Absent does NOT mean
	// "no cap": it falls back to DefaultMaxSubagentDepth (fail-closed). Like
	// ToolRuntime, manifest/runtime-only; launch adapters resolve it through
	// Runtime.SubagentDepth.AdmitDepth before brokering a child. See
	// SubagentDepthRule.
	SubagentDepth *SubagentDepthRule `json:"subagent_depth,omitempty"`
	// MountView (issue #2577) is the T1 mount view — the declarative per-session
	// namespace of what file tree EXISTS to the agent at all. Each MountRule names a
	// repo-relative subtree that is in view and whether it is read-only or read-write;
	// a path matching no rule is OUTSIDE the view and does not exist (fail-closed
	// DEFAULT_DENY), the same deny-by-default shape the tool floor has over tool names.
	// An empty/omitted MountView means "no view configured" — every path is visible
	// (the feature is off, backward-compatible). Enforced offline by the reference
	// monitor MountViewRefusal (mountview.go), driven purely by the manifest with no
	// model in the loop. Like RateLimit/Isolation/ToolRuntime this is manifest/runtime-
	// only — NOT an adjudicator.Policy field; wiring it into the live pathscope read
	// hot path is the named promotion step (see the doc fence, #2358 owns inheritance).
	MountView []MountRule `json:"mount_view,omitempty"`
}

// EgressRule is the manifest's network-egress block.
// deny_hosts is a list of exact host names / IP literals refused IN ADDITION to the
// hardwired cloud-metadata / link-local class, so a deployment blocks its own sensitive
// endpoints without a code change. research_allow_hosts is a positive WebFetch-only
// allowlist for research agents; matching hosts are admitted after deny_hosts/metadata
// checks, and non-matching WebFetch URLs are refused with POLICY_BLOCK.
type EgressRule struct {
	DenyHosts          []string `json:"deny_hosts,omitempty"`
	ResearchAllowHosts []string `json:"research_allow_hosts,omitempty"`
	// AllowHosts / BlockHosts / BlockLists are the adblock-style site allow/block layer
	// (internal/egresslist): block_hosts refuse a host and every subdomain; block_lists
	// subscribe to a bundled community filter list by name (an unknown name is a HARD
	// load error naming the available lists — a dropped block list can never silently
	// become an all-permissive no-op); allow_hosts are adblock '@@' exceptions that carve
	// a host back open. restrict flips the WebFetch default posture from default-allowed
	// to a strict allowlist. These map to adjudicator.Policy.Egress{Allow,Block}Hosts /
	// EgressBlockLists / EgressRestrict.
	AllowHosts []string `json:"allow_hosts,omitempty"`
	BlockHosts []string `json:"block_hosts,omitempty"`
	BlockLists []string `json:"block_lists,omitempty"`
	Restrict   bool     `json:"restrict,omitempty"`
}

// AuthorizeRule releases a tainted flow into one exact sink tool/class. It is
// intentionally narrow: a rule authorizes one named tool and one sink class.
type AuthorizeRule struct {
	Tool string `json:"tool"`
	Sink string `json:"sink"`
}

// ArgRule is one per-tool argument-value constraint in the manifest. It narrows
// the floor: a tool that clears the name-level allow is still DENIED here when a
// constrained argument fails its predicate. A rule can only RESTRICT, never widen.
//
// Exactly ONE matcher must be set (fail-loud otherwise):
//   - allow_glob: the arg value MUST be a path under this glob ("./out/**"),
//     else DENY. A "../" escape fails; a MISSING required arg fails closed.
//   - deny_regex: the arg value matching this RE2 pattern is DENIED. A missing
//     arg is not a match.
//   - max_bytes: a string arg longer than this many bytes is DENIED.
//
// reason (optional) is the closed-vocabulary refusal code cited on a violation;
// it defaults to POLICY_BLOCK when omitted.
// InlineEvalSpec declares an additional inline-program interpreter spelling.
// It only broadens the write-shaped floor; it cannot make a command less restricted.
type InlineEvalSpec struct {
	Interp string   `json:"interp"`
	Flags  []string `json:"flags"`
}

type ArgRule struct {
	Tool        string `json:"tool"`
	Arg         string `json:"arg"`
	AllowGlob   string `json:"allow_glob,omitempty"`
	DenyRegex   string `json:"deny_regex,omitempty"`
	MaxBytes    int    `json:"max_bytes,omitempty"`
	CLIReadOnly bool   `json:"cli_read_only,omitempty"`
	Reason      string `json:"reason,omitempty"`
	// Fix (optional) is the operator-authored SANCTIONED ALTERNATIVE recommended
	// in the same breath as the refusal: one imperative line naming what the
	// caller should do instead (e.g. "hand the exact elevated command to the
	// operator to run"). It rides the deny verdict's Meta to every wire, so a
	// refusal is a redirect, not a dead end. Rules with no sensible alternative
	// (a fork bomb) simply omit it. Never interpolate the offending arg value —
	// the fix is static manifest text, part of the bounded-disclosure budget.
	Fix string `json:"fix,omitempty"`
	// Advisory puts THIS rule on logged trial: a violation is noted on the
	// admitted verdict's Meta (advisory_violations) instead of denying — the
	// rule-granular false-positive softener, so one noisy rule can warn while
	// the rest of the floor stays enforcing. Maps to ArgPredicate.Advisory.
	Advisory bool `json:"advisory,omitempty"`
}

// RateLimitRule is the declarative throughput/cost cap (Epic 8, issue #699): the
// fak-policy/v1 form of internal/ratelimit's governor. It makes the env-only
// limiter reachable from the manifest an operator edits — a per-key call quota
// and/or cumulative-cost budget, bucketed by a key dimension, with an optional
// advisory retry-after (ms) surfaced on the WAIT the over-cap deny becomes. The
// resolved rule is applied to ratelimit.Default at boot and on --policy hot-reload
// (cmd/fak applyRuntime), mirroring how SafeSinks/Authorize reach ifc.
//
// At least one cap (max_calls / max_cost) must be declared; BOTH may be set
// together — the underlying limiter enforces each independently (check-before-
// consume), exactly as the FAK_RATELIMIT_* env seam does, so the manifest is never
// strictly less capable than env config. Key defaults to "trace" when omitted; an
// unknown key-mode, a negative value, or an all-zero block fails loud at load.
type RateLimitRule struct {
	MaxCalls     int    `json:"max_calls,omitempty"`      // per-key admitted-call quota; 0 = no call cap
	MaxCost      int64  `json:"max_cost,omitempty"`       // per-key cumulative cost budget (arg bytes ~ tokens); 0 = no cost cap
	Key          string `json:"key,omitempty"`            // trace|tool|global (default trace)
	RetryAfterMS int    `json:"retry_after_ms,omitempty"` // advisory back-off (ms) on the over-cap WAIT; 0 = limiter default
}

// Runtime is the full manifest resolved for boot: the existing name-level
// adjudicator policy, plus IFC policy, host-authored source registrations, and the
// declared rate-limit cap (issue #699) pushed into ratelimit.Default by applyRuntime.
type Runtime struct {
	Adjudicator    adjudicator.Policy
	Sources        map[string]provenance.Source
	SafeSinks      []string
	AuthorizeRules []AuthorizeRule
	RateLimit      *RateLimitRule
	// Isolation is the compiled trust-level → ToolExec-backend dial (#2013);
	// nil when the manifest declares none (BackendFor then fails closed).
	Isolation *IsolationRule
	// ToolRuntime is the compiled per-tool runtime-envelope table (seam 5 of
	// the tool-process table); nil when the manifest declares none
	// (EnvelopeFor then resolves nothing and the fold defaults apply).
	ToolRuntime *ToolRuntimeTable
	// InheritedCapabilities is the compiled child-launch inheritance table; nil
	// when the manifest declares none, so Resolve returns an empty envelope.
	InheritedCapabilities *InheritedTable
	// SubagentDepth is the compiled subagent fan-out depth cap (#2603); nil when
	// the manifest declares none, in which case AdmitDepth still enforces the
	// DefaultMaxSubagentDepth conservative default on the nil receiver.
	SubagentDepth    *SubagentDepthRule
	PolicyContext    abi.PolicyContext
	StrictGatedSinks bool
	GatedSinks       map[string]bool
	ContentDigest    string
	Generation       uint64
}

// Load reads, parses, validates, and resolves a manifest file into a Policy.
func Load(path string) (adjudicator.Policy, error) {
	rt, err := LoadRuntime(path)
	if err != nil {
		return adjudicator.Policy{}, err
	}
	return rt.Adjudicator, nil
}

// LoadOp distinguishes the two ways loading a floor fails: the file could not be
// READ at all, or it was read and did not PARSE. They need different next steps —
// a wrong path versus a wrong manifest — and the message alone cannot be
// dispatched on.
type LoadOp string

const (
	LoadOpRead  LoadOp = "read"
	LoadOpParse LoadOp = "parse"
)

// LoadError is the typed failure LoadRuntime returns. It exists so a CALLER can
// react to a floor that would not load rather than only print it: the operator's
// most common fatal misconfiguration used to surface as a bare
// `fak: policy floor.json: ...` with nowhere to go next, and a caller that wants
// to say more has to be able to recognize the error first.
//
// Error() is byte-identical to the strings this function returned before the type
// existed, so anything matching on the message is unaffected.
type LoadError struct {
	Op   LoadOp
	Path string
	Err  error
}

func (e *LoadError) Error() string {
	if e.Op == LoadOpRead {
		return fmt.Sprintf("policy: %v", e.Err)
	}
	return fmt.Sprintf("policy %s: %v", e.Path, e.Err)
}

func (e *LoadError) Unwrap() error { return e.Err }

// LoadRuntime reads, parses, validates, and resolves a manifest file into the
// full boot-time policy set. A failure is always a *LoadError, so a caller can
// errors.As it and report the path and which half failed.
func LoadRuntime(path string) (Runtime, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Runtime{}, &LoadError{Op: LoadOpRead, Path: path, Err: err}
	}
	rt, err := ParseRuntime(b)
	if err != nil {
		return Runtime{}, &LoadError{Op: LoadOpParse, Path: path, Err: err}
	}
	return rt, nil
}

// Parse resolves manifest bytes into a Policy (the byte-level Load core; exported
// for tests and in-memory callers).
func Parse(b []byte) (adjudicator.Policy, error) {
	rt, err := ParseRuntime(b)
	if err != nil {
		return adjudicator.Policy{}, err
	}
	return rt.Adjudicator, nil
}

// ParseRuntime resolves manifest bytes into the full boot-time policy set.
func ParseRuntime(b []byte) (Runtime, error) {
	m, err := ParseManifest(b)
	if err != nil {
		return Runtime{}, err
	}
	rt, err := m.ToRuntime()
	if err != nil {
		return Runtime{}, err
	}
	rt.ContentDigest = ComputeContentDigest(b)
	rt.Generation = 1
	return rt, nil
}

// ParseManifest decodes manifest bytes WITHOUT resolving to a Policy, rejecting
// unknown fields so a misspelled key is a hard error rather than a silent drop.
func ParseManifest(b []byte) (Manifest, error) {
	var m Manifest
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return Manifest{}, fmt.Errorf("invalid manifest: %w", err)
	}
	return m, nil
}

// ToPolicy validates the manifest (version + closed-vocabulary deny reasons) and
// builds the runtime Policy. An unknown deny reason is a hard error listing the
// offending entries and the valid vocabulary — the policy never loads with a
// reason the kernel cannot cite.
func (m Manifest) ToPolicy() (adjudicator.Policy, error) {
	rt, err := m.ToRuntime()
	if err != nil {
		return adjudicator.Policy{}, err
	}
	return rt.Adjudicator, nil
}

// NewTieredEvaluator constructs a TieredEvaluator pre-configured with the Manifest's allowed tools.
func (m Manifest) NewTieredEvaluator() *TieredEvaluator {
	eval := NewTieredEvaluator()
	for _, tool := range m.Allow {
		eval.AllowTool(tool)
	}
	return eval
}

// EvaluateAgainstTiers evaluates a tool invocation against the two-tier decoupled safety floor model
// using this manifest's allowed tools on the convenience surface.
func (m Manifest) EvaluateAgainstTiers(tool string, args any) TierDecision {
	eval := m.NewTieredEvaluator()
	return eval.Evaluate(tool, args)
}

// DeniesToolUnconditionally reports whether the manifest denies tool for EVERY
// argument value — i.e. no ArgRule could make it ALLOW — the args-independent
// "is this a blanket block?" question promptmmu (#752) asks before it may safely
// drop a tool DEFINITION from the advertised surface.
//
// It resolves the manifest to its runtime adjudicator.Policy and delegates to
// Policy.NeverAdmits, so the manifest-level predicate and the live floor can never
// drift: an explicit name-level Deny (or self-modify glob) reports true; a tool the
// floor affirmatively allows — even one narrowed by an arg-conditional ArgRule —
// reports false (arg rules only RESTRICT an otherwise-allow, never grant one, so a
// never-allowed name stays never-allowed under every argument). A manifest that does
// not resolve (malformed, unknown deny reason) reports false: never prune a tool-def
// against a floor that did not load.
func (m Manifest) DeniesToolUnconditionally(tool string) bool {
	p, err := m.ToPolicy()
	if err != nil {
		return false
	}
	return p.NeverAdmits(tool)
}

// ToRuntime validates the manifest and builds the complete runtime policy.
func (m Manifest) ToRuntime() (Runtime, error) {
	if err := m.validateVersion(); err != nil {
		return Runtime{}, err
	}
	var prof Profile
	if m.Profile != "" {
		p, err := ParseProfile(m.Profile)
		if err != nil {
			return Runtime{}, err
		}
		prof = p
	}
	posture, err := parsePosture(m.Posture)
	if err != nil {
		return Runtime{}, err
	}
	var blockedGlobs []string
	if len(m.BlockedPathGlobs) > 0 {
		blockedGlobs = append(blockedGlobs, m.BlockedPathGlobs...)
	}
	if len(m.CredentialPathGlobs) > 0 {
		blockedGlobs = append(blockedGlobs, m.CredentialPathGlobs...)
	}
	p := adjudicator.Policy{
		Posture:          posture,
		AllowPrefix:      cloneSlice(m.AllowPrefix),
		Complain:         make(map[string]bool, len(m.Complain)),
		SelfModifyGlobs:  cloneSlice(m.SelfModifyGlobs),
		BlockedPathGlobs: cloneSlice(blockedGlobs),
		RedactFields:     cloneSlice(m.RedactFields),
	}
	for i, raw := range m.Complain {
		tool := strings.TrimSpace(raw)
		if tool == "" {
			return Runtime{}, fmt.Errorf("complain[%d]: tool name must not be blank", i)
		}
		p.Complain[tool] = true
	}
	if len(p.Complain) == 0 {
		p.Complain = nil
	}
	if m.Egress != nil && len(m.Egress.DenyHosts) > 0 {
		p.EgressExtraDenyHosts = cloneSlice(m.Egress.DenyHosts)
	}
	if m.Egress != nil && len(m.Egress.ResearchAllowHosts) > 0 {
		p.ResearchEgressAllowHosts = cloneSlice(m.Egress.ResearchAllowHosts)
	}
	if m.Egress != nil {
		p.EgressAllowHosts = cloneSlice(m.Egress.AllowHosts)
		p.EgressBlockHosts = cloneSlice(m.Egress.BlockHosts)
		p.EgressRestrict = m.Egress.Restrict
		// Every block_lists name must resolve to a bundled community list. An unknown
		// name fails LOUD here, naming the bad list and the available ones — a dropped
		// block list is a security regression, never a silent all-permissive no-op.
		for i, name := range m.Egress.BlockLists {
			if _, ok := egresslist.BundledList(name); !ok {
				return Runtime{}, fmt.Errorf(
					"egress.block_lists[%d]: unknown list %q; available: %s",
					i, name, strings.Join(egresslist.BundledListNames(), ", "))
			}
		}
		p.EgressBlockLists = cloneSlice(m.Egress.BlockLists)
	}
	if len(m.Allow) > 0 {
		p.Allow = make(map[string]bool, len(m.Allow))
		for _, t := range m.Allow {
			p.Allow[t] = true
		}
	}
	if len(m.Deny) > 0 {
		p.Deny = make(map[string]abi.ReasonCode, len(m.Deny))
		var bad []string
		for tool, reason := range m.Deny {
			code, ok := abi.ReasonByName(reason)
			if !ok {
				bad = append(bad, fmt.Sprintf("%s=%q", tool, reason))
				continue
			}
			p.Deny[tool] = code
		}
		if len(bad) > 0 {
			sort.Strings(bad)
			return Runtime{}, fmt.Errorf(
				"unknown deny reason(s): %s; valid reasons: %s",
				strings.Join(bad, ", "), strings.Join(abi.ReasonNames(), ", "))
		}
	}
	argPreds, err := compileArgRules(m.ArgRules)
	if err != nil {
		return Runtime{}, err
	}
	p.ArgPredicates = argPreds
	p.LintWrites = m.LintWrites
	adv, err := compileAdvisoryReasons(m.AdvisoryReasons)
	if err != nil {
		return Runtime{}, err
	}
	p.AdvisoryReasons = adv
	secretPosture, ok := adjudicator.ParseSecretPosture(m.SecretPosture)
	if !ok {
		return Runtime{}, fmt.Errorf("unknown secret_posture %q; valid: quarantine, fail_closed, admit_and_log", m.SecretPosture)
	}
	p.SecretPosture = secretPosture
	secretPats, err := compileSecretPatterns(m.SecretPatterns)
	if err != nil {
		return Runtime{}, err
	}
	p.SecretPatterns = secretPats
	inlineEval, err := compileInlineEval(m.InlineEval)
	if err != nil {
		return Runtime{}, err
	}
	p.InlineEval = inlineEval
	sources, err := compileSources(m.Sources)
	if err != nil {
		return Runtime{}, err
	}
	auth, err := normalizeAuthorizeRules(m.Authorize)
	if err != nil {
		return Runtime{}, err
	}
	safe, err := normalizeSafeSinks(m.SafeSinks)
	if err != nil {
		return Runtime{}, err
	}
	rl, err := compileRateLimit(m.RateLimit)
	if err != nil {
		return Runtime{}, err
	}
	iso, err := compileIsolation(m.Isolation)
	if err != nil {
		return Runtime{}, err
	}
	tr, err := compileToolRuntime(m.ToolRuntime)
	if err != nil {
		return Runtime{}, err
	}
	ic, err := compileInheritedCapabilities(m.InheritedCapabilities)
	if err != nil {
		return Runtime{}, err
	}
	sd, err := compileSubagentDepth(m.SubagentDepth)
	if err != nil {
		return Runtime{}, err
	}
	// Session-scoped, env-only (see AutoRepairEnv): applied after the manifest so a
	// repo-shipped policy never carries one operator supervision preference to every
	// other clone. A bad mode refuses the whole load rather than silently staying off.
	if p.AutoRepairSidestep, err = autoRepairSidestepFromEnv(os.Getenv(AutoRepairEnv)); err != nil {
		return Runtime{}, err
	}
	rt := Runtime{
		Adjudicator:           p,
		Sources:               sources,
		SafeSinks:             safe,
		AuthorizeRules:        auth,
		RateLimit:             rl,
		Isolation:             iso,
		ToolRuntime:           tr,
		InheritedCapabilities: ic,
		SubagentDepth:         sd,
	}
	if prof != "" {
		prof.Apply(&rt)
		if m.Posture != "" {
			rt.Adjudicator.Posture = posture
			rt.PolicyContext.Posture = posture
		}
	}
	return rt, nil
}

func (m Manifest) validateVersion() error {
	switch {
	case m.Version == "", m.Version == Version:
		return nil
	case strings.HasPrefix(m.Version, "fak-policy/v1"):
		return nil // forward-accept a newer v1 minor
	default:
		return fmt.Errorf("unsupported manifest version %q (this binary speaks %s)", m.Version, Version)
	}
}

// ParsePosture parses a posture string into an adjudicator.Posture.
// Accept "default_open", "", "fail_closed", "admit_and_log".
func ParsePosture(s string) (adjudicator.Posture, error) {
	switch strings.TrimSpace(s) {
	case "", postureFailClosed:
		return adjudicator.PostureFailClosed, nil
	case postureAdmitAndLog:
		return adjudicator.PostureAdmitAndLog, nil
	case postureDefaultOpen:
		return adjudicator.PostureDefaultOpen, nil
	default:
		return adjudicator.PostureFailClosed, fmt.Errorf(
			"unknown posture %q (want %s|%s|%s)", s, postureFailClosed, postureAdmitAndLog, postureDefaultOpen)
	}
}

func parsePosture(s string) (adjudicator.Posture, error) {
	return ParsePosture(s)
}

// FromPolicy renders a runtime Policy back into a manifest — the basis of
// `fak policy --dump`, which emits the built-in DefaultPolicy as a starting
// point an adopter can edit. Round-trips: FromPolicy(p).ToPolicy() == p for any
// p built from a manifest.
func FromPolicy(p adjudicator.Policy) Manifest {
	m := Manifest{
		Version:          Version,
		AllowPrefix:      cloneSlice(p.AllowPrefix),
		SelfModifyGlobs:  cloneSlice(p.SelfModifyGlobs),
		BlockedPathGlobs: cloneSlice(p.BlockedPathGlobs),
		RedactFields:     cloneSlice(p.RedactFields),
	}
	if p.Posture == adjudicator.PostureAdmitAndLog {
		m.Posture = postureAdmitAndLog
	} else if p.Posture == adjudicator.PostureDefaultOpen {
		m.Posture = postureDefaultOpen
	}
	if len(p.AdvisoryReasons) > 0 {
		m.AdvisoryReasons = make([]string, 0, len(p.AdvisoryReasons))
		for r := range p.AdvisoryReasons {
			m.AdvisoryReasons = append(m.AdvisoryReasons, abi.ReasonName(r))
		}
		sort.Strings(m.AdvisoryReasons) // deterministic dump (map iteration is unordered)
	}
	if len(p.Allow) > 0 {
		m.Allow = make([]string, 0, len(p.Allow))
		for t := range p.Allow {
			m.Allow = append(m.Allow, t)
		}
		sort.Strings(m.Allow) // deterministic dump (map iteration is unordered)
	}
	if len(p.Deny) > 0 {
		m.Deny = make(map[string]string, len(p.Deny))
		for t, c := range p.Deny {
			m.Deny[t] = abi.ReasonName(c)
		}
	}
	if len(p.EgressExtraDenyHosts) > 0 || len(p.ResearchEgressAllowHosts) > 0 ||
		len(p.EgressAllowHosts) > 0 || len(p.EgressBlockHosts) > 0 ||
		len(p.EgressBlockLists) > 0 || p.EgressRestrict {
		m.Egress = &EgressRule{
			DenyHosts:          cloneSlice(p.EgressExtraDenyHosts),
			ResearchAllowHosts: cloneSlice(p.ResearchEgressAllowHosts),
			AllowHosts:         cloneSlice(p.EgressAllowHosts),
			BlockHosts:         cloneSlice(p.EgressBlockHosts),
			BlockLists:         cloneSlice(p.EgressBlockLists),
			Restrict:           p.EgressRestrict,
		}
	}
	if len(p.ArgPredicates) > 0 {
		m.ArgRules = make([]ArgRule, 0, len(p.ArgPredicates))
		for _, pred := range p.ArgPredicates {
			r := ArgRule{Tool: pred.Tool, Arg: pred.Arg, Reason: abi.ReasonName(pred.Reason), Advisory: pred.Advisory,
				Fix: pred.Fix}
			switch pred.Kind {
			case adjudicator.ArgAllowGlob:
				r.AllowGlob = pred.Glob
			case adjudicator.ArgDenyRegex:
				if pred.Re != nil {
					r.DenyRegex = pred.Re.String()
				}
			case adjudicator.ArgMaxBytes:
				r.MaxBytes = pred.N
			}
			m.ArgRules = append(m.ArgRules, r)
		}
	}
	if p.SecretPosture != adjudicator.SecretQuarantine {
		m.SecretPosture = p.SecretPosture.String() // quarantine is the default -> omitted
	}
	if len(p.SecretPatterns) > 0 {
		m.SecretPatterns = make([]string, 0, len(p.SecretPatterns))
		for _, re := range p.SecretPatterns {
			if re != nil {
				m.SecretPatterns = append(m.SecretPatterns, re.String())
			}
		}
	}
	if len(p.InlineEval) > 0 {
		m.InlineEval = make([]InlineEvalSpec, 0, len(p.InlineEval))
		for _, spec := range p.InlineEval {
			m.InlineEval = append(m.InlineEval, InlineEvalSpec{Interp: spec.Interp, Flags: cloneSlice(spec.Flags)})
		}
	}
	m.LintWrites = p.LintWrites
	return m
}

// FromRuntime renders a Runtime back into a Manifest, preserving profile and IFC metadata.
func FromRuntime(rt Runtime) Manifest {
	m := FromPolicy(rt.Adjudicator)
	if rt.PolicyContext.Profile != "" {
		m.Profile = rt.PolicyContext.Profile
	}
	return m
}

// JSON renders the manifest as indented, newline-terminated JSON for --dump.
func (m Manifest) JSON() []byte {
	b, _ := json.MarshalIndent(m, "", "  ")
	return append(b, '\n')
}

// Summary renders a human-readable description of what a Policy admits — used by
// `fak policy --check` so an operator can eyeball the floor before deploying it.
// It calls out the fail-closed case (nothing affirmatively allowed) explicitly,
// since an empty allow-list is VALID but means every call resolves to
// DEFAULT_DENY.
func Summary(p adjudicator.Policy) string {
	var b strings.Builder
	fmt.Fprintf(&b, "posture            : %s\n", postureName(p.Posture))
	if len(p.AdvisoryReasons) > 0 {
		names := make([]string, 0, len(p.AdvisoryReasons))
		for r := range p.AdvisoryReasons {
			names = append(names, abi.ReasonName(r))
		}
		sort.Strings(names)
		fmt.Fprintf(&b, "advisory reasons   : %s (would-deny admits, journaled)\n", strings.Join(names, ", "))
	}
	allowN := len(p.Allow)
	fmt.Fprintf(&b, "allow (exact)      : %d tool(s)\n", allowN)
	fmt.Fprintf(&b, "allow (prefix)     : %s\n", joinOrNone(p.AllowPrefix))
	fmt.Fprintf(&b, "deny (explicit)    : %d tool(s)\n", len(p.Deny))
	for _, t := range maputil.SortedKeys(p.Deny) {
		fmt.Fprintf(&b, "                     %s -> %s\n", t, abi.ReasonName(p.Deny[t]))
	}
	fmt.Fprintf(&b, "egress deny hosts  : %s\n", joinOrNone(p.EgressExtraDenyHosts))
	fmt.Fprintf(&b, "research egress    : %s\n", joinOrNone(p.ResearchEgressAllowHosts))
	fmt.Fprintf(&b, "egress allow hosts : %s\n", joinOrNone(p.EgressAllowHosts))
	fmt.Fprintf(&b, "egress block hosts : %s\n", joinOrNone(p.EgressBlockHosts))
	fmt.Fprintf(&b, "egress block lists : %s\n", joinOrNone(p.EgressBlockLists))
	egressPosture := "default-allowed"
	if p.EgressRestrict {
		egressPosture = "restrict (strict WebFetch allowlist)"
	}
	fmt.Fprintf(&b, "egress posture     : %s\n", egressPosture)
	fmt.Fprintf(&b, "self-modify globs  : %s\n", joinOrNone(p.SelfModifyGlobs))
	fmt.Fprintf(&b, "redact arg fields  : %s\n", joinOrNone(p.RedactFields))
	fmt.Fprintf(&b, "arg rules          : %d rule(s)\n", len(p.ArgPredicates))
	for _, pred := range p.ArgPredicates {
		fmt.Fprintf(&b, "                     %s\n", describeArgPredicate(pred))
	}
	if allowN == 0 && len(p.AllowPrefix) == 0 && len(p.ResearchEgressAllowHosts) == 0 {
		if p.Posture == adjudicator.PostureAdmitAndLog {
			b.WriteString("\nNOTE: nothing is affirmatively allowed; read-shaped DEFAULT_DENY\n" +
				"calls are admitted with posture=admit_and_log/would_deny=DEFAULT_DENY,\n" +
				"while explicit denies and write-shaped calls still fail closed.\n")
		} else {
			b.WriteString("\nNOTE: nothing is affirmatively allowed — this is the fail-closed\n" +
				"empty floor; EVERY call resolves to DEFAULT_DENY.\n")
		}
	}
	return b.String()
}

func postureName(p adjudicator.Posture) string {
	switch p {
	case adjudicator.PostureAdmitAndLog:
		return postureAdmitAndLog
	case adjudicator.PostureDefaultOpen:
		return postureDefaultOpen
	default:
		return postureFailClosed
	}
}

// SummaryRuntime renders the complete manifest effect, including IFC config.
func SummaryRuntime(rt Runtime) string {
	var b strings.Builder
	b.WriteString(Summary(rt.Adjudicator))
	if rt.ContentDigest != "" {
		fmt.Fprintf(&b, "content digest     : %s\n", rt.ContentDigest)
	}
	if rt.Generation > 0 {
		fmt.Fprintf(&b, "generation         : %d\n", rt.Generation)
	}
	fmt.Fprintf(&b, "ifc safe sinks     : %s\n", joinOrNone(rt.SafeSinks))
	fmt.Fprintf(&b, "ifc authorize      : %d rule(s)\n", len(rt.AuthorizeRules))
	for _, r := range rt.AuthorizeRules {
		fmt.Fprintf(&b, "                     %s -> %s\n", r.Tool, strings.ToUpper(r.Sink))
	}
	fmt.Fprintf(&b, "ifc sources        : %d tool(s)\n", len(rt.Sources))
	for _, tool := range maputil.SortedKeys(rt.Sources) {
		fmt.Fprintf(&b, "                     %s -> %s\n", tool, rt.Sources[tool])
	}
	if rt.RateLimit != nil {
		key := strings.ToLower(strings.TrimSpace(rt.RateLimit.Key))
		if key == "" {
			key = "trace"
		}
		fmt.Fprintf(&b, "rate limit         : %d call(s) / %d cost per %s (retry_after_ms=%d)\n",
			rt.RateLimit.MaxCalls, rt.RateLimit.MaxCost, key, rt.RateLimit.RetryAfterMS)
	} else {
		fmt.Fprintf(&b, "rate limit         : (none — inert)\n")
	}
	if rt.Isolation != nil {
		fmt.Fprintf(&b, "isolation backends : %s (strongest: %s)\n",
			strings.Join(rt.Isolation.Backends, ", "), rt.Isolation.strongest())
		for _, lv := range maputil.SortedKeys(rt.Isolation.Trust) {
			fmt.Fprintf(&b, "isolation trust    : %s -> %s\n", lv, rt.Isolation.Trust[lv])
		}
	} else {
		fmt.Fprintf(&b, "isolation          : (none — dial unset; placement fails closed)\n")
	}
	if rows := rt.ToolRuntime.Rules(); len(rows) > 0 {
		fmt.Fprintf(&b, "tool runtime       : %d envelope(s)\n", len(rows))
		for _, r := range rows {
			fmt.Fprintf(&b, "                     %s -> deadline_ms=%d heartbeat_every_ms=%d\n",
				r.Tool, r.DeadlineMS, r.HeartbeatEveryMS)
		}
	} else {
		fmt.Fprintf(&b, "tool runtime       : (none — fold defaults apply)\n")
	}
	if rows := rt.InheritedCapabilities.Rules(); len(rows) > 0 {
		fmt.Fprintf(&b, "inherited launch   : %d envelope(s)\n", len(rows))
		for _, r := range rows {
			fmt.Fprintf(&b, "                     %s -> env=%d secret_refs=%d cwd=%t writable=%d persistence=%d egress=%d\n",
				r.Tool, len(r.Env), len(r.SecretRefs), r.CWD != "", len(r.WritablePaths), len(r.PersistencePaths), len(r.EgressRefs))
		}
	} else {
		fmt.Fprintf(&b, "inherited launch   : (none — child inherits nothing)\n")
	}
	if rt.SubagentDepth != nil {
		fmt.Fprintf(&b, "subagent depth     : max_depth=%d\n", rt.SubagentDepth.MaxDepth)
	} else {
		fmt.Fprintf(&b, "subagent depth     : max_depth=%d (default — fail-closed)\n", DefaultMaxSubagentDepth)
	}
	return b.String()
}

// ApplySources installs the host-authored source classes from a runtime manifest.
func ApplySources(rt Runtime) {
	for tool, src := range rt.Sources {
		provenance.RegisterSource(tool, src)
	}
}

func normalizeSafeSinks(safeSinks []string) ([]string, error) {
	if len(safeSinks) == 0 {
		return nil, nil
	}
	out := make([]string, 0, len(safeSinks))
	for i, tool := range safeSinks {
		tool = strings.TrimSpace(tool)
		if tool == "" {
			return nil, fmt.Errorf("safe_sinks[%d]: tool is required", i)
		}
		out = append(out, tool)
	}
	return out, nil
}

// compileRateLimit validates a declared rate_limit block (absent => inert nil) and
// returns it for Runtime. A present block must name a known key-mode, hold
// non-negative values, and declare at least one meaningful cap — so a typo'd or
// empty block fails loud at load rather than silently installing no cap. The rule
// is returned as-is (key defaulting to trace is resolved at the ratelimit side).
func compileRateLimit(r *RateLimitRule) (*RateLimitRule, error) {
	if r == nil {
		return nil, nil // absent => inert limiter
	}
	switch strings.ToLower(strings.TrimSpace(r.Key)) {
	case "", "trace", "tool", "global":
	default:
		return nil, fmt.Errorf("rate_limit.key: unknown mode %q (want trace|tool|global)", r.Key)
	}
	if r.MaxCalls < 0 || r.MaxCost < 0 || r.RetryAfterMS < 0 {
		return nil, fmt.Errorf("rate_limit: max_calls/max_cost/retry_after_ms must be non-negative (got calls=%d cost=%d retry_ms=%d)",
			r.MaxCalls, r.MaxCost, r.RetryAfterMS)
	}
	if r.MaxCalls == 0 && r.MaxCost == 0 {
		return nil, fmt.Errorf("rate_limit: declare at least one of max_calls / max_cost (an all-zero block installs no cap)")
	}
	return r, nil
}

func normalizeAuthorizeRules(rules []AuthorizeRule) ([]AuthorizeRule, error) {
	if len(rules) == 0 {
		return nil, nil
	}
	out := make([]AuthorizeRule, 0, len(rules))
	for i, r := range rules {
		tool := strings.TrimSpace(r.Tool)
		if tool == "" {
			return nil, fmt.Errorf("authorize[%d]: tool is required", i)
		}
		sink, err := normalizeSinkName(r.Sink)
		if err != nil {
			return nil, fmt.Errorf("authorize[%d]: %w", i, err)
		}
		out = append(out, AuthorizeRule{Tool: tool, Sink: sink})
	}
	return out, nil
}

func compileSources(src map[string]string) (map[string]provenance.Source, error) {
	if len(src) == 0 {
		return nil, nil
	}
	out := make(map[string]provenance.Source, len(src))
	for tool, name := range src {
		if strings.TrimSpace(tool) == "" {
			return nil, fmt.Errorf("sources: tool name is required")
		}
		s, err := parseSource(name)
		if err != nil {
			return nil, fmt.Errorf("sources[%s]: %w", tool, err)
		}
		out[tool] = s
	}
	return out, nil
}

func normalizeSinkName(s string) (string, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "EGRESS":
		return "EGRESS", nil
	case "EXEC":
		return "EXEC", nil
	case "DESTRUCTIVE":
		return "DESTRUCTIVE", nil
	default:
		return "", fmt.Errorf("unknown sink %q (want EGRESS|EXEC|DESTRUCTIVE)", s)
	}
}

func parseSource(s string) (provenance.Source, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "trusted_local":
		return provenance.TrustedLocal, nil
	case "untrusted":
		return provenance.Untrusted, nil
	default:
		return provenance.Untrusted, fmt.Errorf("unknown source %q (want trusted_local|untrusted)", s)
	}
}

func compileInlineEval(specs []InlineEvalSpec) ([]adjudicator.InlineEvalSpec, error) {
	if len(specs) == 0 {
		return nil, nil
	}
	out := make([]adjudicator.InlineEvalSpec, 0, len(specs))
	for i, spec := range specs {
		interp := strings.ToLower(strings.TrimSpace(spec.Interp))
		if interp == "" {
			return nil, fmt.Errorf("inline_eval[%d].interp must not be empty", i)
		}
		if len(spec.Flags) == 0 {
			return nil, fmt.Errorf("inline_eval[%d].flags must not be empty", i)
		}
		flags := make([]string, 0, len(spec.Flags))
		for j, flag := range spec.Flags {
			flag = strings.TrimSpace(flag)
			if flag == "" {
				return nil, fmt.Errorf("inline_eval[%d].flags[%d] must not be empty", i, j)
			}
			flags = append(flags, flag)
		}
		out = append(out, adjudicator.InlineEvalSpec{Interp: interp, Flags: flags})
	}
	return out, nil
}

// compileSecretPatterns compiles the manifest's declared EXTRA secret RE2 strings
// (issue #885) at policy LOAD, so a bad pattern fails loud here, never at runtime.
// The compiled set is unioned with the canon floor at the gate (extend, never
// replace). An empty list compiles to nil (floor patterns only).
func compileSecretPatterns(pats []string) ([]*regexp.Regexp, error) {
	if len(pats) == 0 {
		return nil, nil
	}
	out := make([]*regexp.Regexp, 0, len(pats))
	for i, p := range pats {
		if err := ValidateRegexSafety(p); err != nil {
			return nil, fmt.Errorf("secret_patterns[%d] %q: %w", i, p, err)
		}
		re, err := regexp.Compile(p)
		if err != nil {
			return nil, fmt.Errorf("secret_patterns[%d] %q: %w", i, p, err)
		}
		out = append(out, re)
	}
	return out, nil
}

// AdvisoryEnv is the dev-session override: reasons named here UNION into the
// manifest's advisory_reasons at every policy load (boot AND --policy
// hot-reload). Comma/space-separated closed-vocabulary reason names, case-
// insensitive, plus the alias "dev" = every advisory-eligible reason
// (SELF_MODIFY, MALFORMED, DEFAULT_DENY). It is read by the fak PROCESS that
// enforces the floor — the operator's shell — so a guarded child agent cannot
// flip it for the session that judges it. A token that is unknown, or names a
// reason that is not advisory-eligible, fails the load LOUDLY (same discipline
// as a manifest typo): a dev who believes they softened the floor must never
// silently still be enforcing, and vice versa.
const AdvisoryEnv = "FAK_ADVISORY_REASONS"

// compileAdvisoryReasons resolves the manifest advisory_reasons list unioned
// with the AdvisoryEnv overlay into the adjudicator's clamped reason set.
func compileAdvisoryReasons(names []string) (map[abi.ReasonCode]bool, error) {
	set := map[abi.ReasonCode]bool{}
	add := func(name, src string) error {
		tok := strings.ToUpper(strings.TrimSpace(name))
		if tok == "" {
			return fmt.Errorf("%s: empty advisory reason", src)
		}
		code, ok := abi.ReasonByName(tok)
		if !ok {
			return fmt.Errorf("%s: unknown advisory reason %q; advisory-eligible reasons: %s",
				src, name, strings.Join(adjudicator.AdvisoryEligibleNames(), ", "))
		}
		if !adjudicator.AdvisoryEligible(code) {
			return fmt.Errorf("%s: reason %s cannot be advisory — it guards the genuine-danger floor; "+
				"advisory-eligible reasons: %s (to soften ONE false-positive-prone rule, set arg_rules[].advisory)",
				src, tok, strings.Join(adjudicator.AdvisoryEligibleNames(), ", "))
		}
		set[code] = true
		return nil
	}
	for i, n := range names {
		if err := add(n, fmt.Sprintf("advisory_reasons[%d]", i)); err != nil {
			return nil, err
		}
	}
	if env := strings.TrimSpace(os.Getenv(AdvisoryEnv)); env != "" {
		toks := strings.FieldsFunc(env, func(r rune) bool { return r == ',' || r == ';' || r == ' ' || r == '\t' })
		for _, t := range toks {
			if strings.EqualFold(strings.TrimSpace(t), "dev") {
				for _, n := range adjudicator.AdvisoryEligibleNames() {
					if err := add(n, AdvisoryEnv); err != nil {
						return nil, err
					}
				}
				continue
			}
			if err := add(t, AdvisoryEnv); err != nil {
				return nil, err
			}
		}
	}
	if len(set) == 0 {
		return nil, nil
	}
	return set, nil
}

func compileArgRules(rules []ArgRule) ([]adjudicator.ArgPredicate, error) {
	if len(rules) == 0 {
		return nil, nil
	}
	out := make([]adjudicator.ArgPredicate, 0, len(rules))
	for i, r := range rules {
		if strings.TrimSpace(r.Tool) == "" {
			return nil, fmt.Errorf("arg_rules[%d]: tool is required", i)
		}
		if strings.TrimSpace(r.Arg) == "" {
			return nil, fmt.Errorf("arg_rules[%d]: arg is required", i)
		}
		matchers := 0
		if r.AllowGlob != "" {
			matchers++
		}
		if r.DenyRegex != "" {
			matchers++
		}
		if r.MaxBytes > 0 {
			matchers++
		}
		if r.CLIReadOnly {
			matchers++
		}
		if matchers != 1 {
			return nil, fmt.Errorf("arg_rules[%d]: set exactly one of allow_glob, deny_regex, max_bytes, cli_read_only", i)
		}
		reason := abi.ReasonPolicyBlock
		if r.Reason != "" {
			code, ok := abi.ReasonByName(r.Reason)
			if !ok {
				return nil, fmt.Errorf("arg_rules[%d]: unknown reason %q; valid reasons: %s",
					i, r.Reason, strings.Join(abi.ReasonNames(), ", "))
			}
			reason = code
		}
		pred := adjudicator.ArgPredicate{Tool: r.Tool, Arg: r.Arg, Reason: reason, Advisory: r.Advisory,
			Fix: strings.TrimSpace(r.Fix)}
		switch {
		case r.AllowGlob != "":
			pred.Kind = adjudicator.ArgAllowGlob
			pred.Glob = r.AllowGlob
		case r.DenyRegex != "":
			if err := ValidateRegexSafety(r.DenyRegex); err != nil {
				return nil, fmt.Errorf("arg_rules[%d]: invalid deny_regex: %w", i, err)
			}
			re, err := regexp.Compile(r.DenyRegex)
			if err != nil {
				return nil, fmt.Errorf("arg_rules[%d]: invalid deny_regex: %w", i, err)
			}
			pred.Kind = adjudicator.ArgDenyRegex
			pred.Re = re
		case r.MaxBytes > 0:
			pred.Kind = adjudicator.ArgMaxBytes
			pred.N = r.MaxBytes
		case r.CLIReadOnly:
			pred.Kind = adjudicator.ArgCLIReadOnly
		}
		out = append(out, pred)
	}
	return out, nil
}

func describeArgPredicate(p adjudicator.ArgPredicate) string {
	reason := abi.ReasonName(p.Reason)
	if p.Advisory {
		reason += " (advisory)"
	}
	switch p.Kind {
	case adjudicator.ArgAllowGlob:
		return fmt.Sprintf("%s.%s allow_glob %s -> %s", p.Tool, p.Arg, p.Glob, reason)
	case adjudicator.ArgDenyRegex:
		re := ""
		if p.Re != nil {
			re = p.Re.String()
		}
		return fmt.Sprintf("%s.%s deny_regex %s -> %s", p.Tool, p.Arg, re, reason)
	case adjudicator.ArgMaxBytes:
		return fmt.Sprintf("%s.%s max_bytes %d -> %s", p.Tool, p.Arg, p.N, reason)
	case adjudicator.ArgCLIReadOnly:
		return fmt.Sprintf("%s.%s cli_read_only -> %s", p.Tool, p.Arg, reason)
	default:
		return fmt.Sprintf("%s.%s unknown -> %s", p.Tool, p.Arg, reason)
	}
}

func joinOrNone(s []string) string {
	if len(s) == 0 {
		return "(none)"
	}
	return strings.Join(s, ", ")
}

func cloneSlice(s []string) []string {
	if len(s) == 0 {
		return nil
	}
	return append([]string(nil), s...)
}
