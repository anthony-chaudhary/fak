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
	Version         string            `json:"version,omitempty"`
	Posture         string            `json:"posture,omitempty"`
	Allow           []string          `json:"allow,omitempty"`
	AllowPrefix     []string          `json:"allow_prefix,omitempty"`
	Deny            map[string]string `json:"deny,omitempty"`
	SelfModifyGlobs []string          `json:"self_modify_globs,omitempty"`
	RedactFields    []string          `json:"redact_fields,omitempty"`
	// SecretPosture (issue #885) selects what the on-discovery secret rung does when
	// a tool result bears a credential: "quarantine" (default, omitted), "fail_closed",
	// or "admit_and_log". An unknown token is refused at load. Maps to
	// adjudicator.Policy.SecretPosture.
	SecretPosture string `json:"secret_posture,omitempty"`
	// SecretPatterns are RE2 strings for EXTRA secret shapes, compiled at load (a bad
	// pattern fails loud) and unioned with the canon floor at the gate. Maps to
	// adjudicator.Policy.SecretPatterns.
	SecretPatterns []string          `json:"secret_patterns,omitempty"`
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
	// Egress (optional) extends the hardwired cloud-metadata / link-local egress floor
	// with operator-declared destinations. It only TIGHTENS the floor — the hardwired
	// metadata block (169.254.169.254 & peers) is always on and cannot be disabled here.
	// Absent leaves the floor at the hardwired set. Maps to adjudicator.Policy.EgressExtraDenyHosts.
	Egress *EgressRule `json:"egress,omitempty"`
}

// EgressRule is the manifest's network-egress block (issue: cloud-metadata SSRF floor).
// deny_hosts is a list of exact host names / IP literals refused IN ADDITION to the
// hardwired cloud-metadata / link-local class, so a deployment blocks its own sensitive
// endpoints (an internal secrets service, a corp metadata mirror) without a code change.
type EgressRule struct {
	DenyHosts []string `json:"deny_hosts,omitempty"`
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
type ArgRule struct {
	Tool      string `json:"tool"`
	Arg       string `json:"arg"`
	AllowGlob string `json:"allow_glob,omitempty"`
	DenyRegex string `json:"deny_regex,omitempty"`
	MaxBytes  int    `json:"max_bytes,omitempty"`
	Reason    string `json:"reason,omitempty"`
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
}

// Load reads, parses, validates, and resolves a manifest file into a Policy.
func Load(path string) (adjudicator.Policy, error) {
	rt, err := LoadRuntime(path)
	if err != nil {
		return adjudicator.Policy{}, err
	}
	return rt.Adjudicator, nil
}

// LoadRuntime reads, parses, validates, and resolves a manifest file into the
// full boot-time policy set.
func LoadRuntime(path string) (Runtime, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Runtime{}, fmt.Errorf("policy: %w", err)
	}
	rt, err := ParseRuntime(b)
	if err != nil {
		return Runtime{}, fmt.Errorf("policy %s: %w", path, err)
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
	return m.ToRuntime()
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
	posture, err := parsePosture(m.Posture)
	if err != nil {
		return Runtime{}, err
	}
	p := adjudicator.Policy{
		Posture:         posture,
		AllowPrefix:     cloneSlice(m.AllowPrefix),
		SelfModifyGlobs: cloneSlice(m.SelfModifyGlobs),
		RedactFields:    cloneSlice(m.RedactFields),
	}
	if m.Egress != nil && len(m.Egress.DenyHosts) > 0 {
		p.EgressExtraDenyHosts = cloneSlice(m.Egress.DenyHosts)
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
	return Runtime{
		Adjudicator:    p,
		Sources:        sources,
		SafeSinks:      safe,
		AuthorizeRules: auth,
		RateLimit:      rl,
	}, nil
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

func parsePosture(s string) (adjudicator.Posture, error) {
	switch strings.TrimSpace(s) {
	case "", postureFailClosed:
		return adjudicator.PostureFailClosed, nil
	case postureAdmitAndLog:
		return adjudicator.PostureAdmitAndLog, nil
	default:
		return adjudicator.PostureFailClosed, fmt.Errorf(
			"unknown posture %q (want %s|%s)", s, postureFailClosed, postureAdmitAndLog)
	}
}

// FromPolicy renders a runtime Policy back into a manifest — the basis of
// `fak policy --dump`, which emits the built-in DefaultPolicy as a starting
// point an adopter can edit. Round-trips: FromPolicy(p).ToPolicy() == p for any
// p built from a manifest.
func FromPolicy(p adjudicator.Policy) Manifest {
	m := Manifest{
		Version:         Version,
		AllowPrefix:     cloneSlice(p.AllowPrefix),
		SelfModifyGlobs: cloneSlice(p.SelfModifyGlobs),
		RedactFields:    cloneSlice(p.RedactFields),
	}
	if p.Posture == adjudicator.PostureAdmitAndLog {
		m.Posture = postureAdmitAndLog
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
	if len(p.ArgPredicates) > 0 {
		m.ArgRules = make([]ArgRule, 0, len(p.ArgPredicates))
		for _, pred := range p.ArgPredicates {
			r := ArgRule{Tool: pred.Tool, Arg: pred.Arg, Reason: abi.ReasonName(pred.Reason)}
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
	m.LintWrites = p.LintWrites
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
	allowN := len(p.Allow)
	fmt.Fprintf(&b, "allow (exact)      : %d tool(s)\n", allowN)
	fmt.Fprintf(&b, "allow (prefix)     : %s\n", joinOrNone(p.AllowPrefix))
	fmt.Fprintf(&b, "deny (explicit)    : %d tool(s)\n", len(p.Deny))
	for _, t := range maputil.SortedKeys(p.Deny) {
		fmt.Fprintf(&b, "                     %s -> %s\n", t, abi.ReasonName(p.Deny[t]))
	}
	fmt.Fprintf(&b, "self-modify globs  : %s\n", joinOrNone(p.SelfModifyGlobs))
	fmt.Fprintf(&b, "redact arg fields  : %s\n", joinOrNone(p.RedactFields))
	fmt.Fprintf(&b, "arg rules          : %d rule(s)\n", len(p.ArgPredicates))
	for _, pred := range p.ArgPredicates {
		fmt.Fprintf(&b, "                     %s\n", describeArgPredicate(pred))
	}
	if allowN == 0 && len(p.AllowPrefix) == 0 {
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
	default:
		return postureFailClosed
	}
}

// SummaryRuntime renders the complete manifest effect, including IFC config.
func SummaryRuntime(rt Runtime) string {
	var b strings.Builder
	b.WriteString(Summary(rt.Adjudicator))
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
		re, err := regexp.Compile(p)
		if err != nil {
			return nil, fmt.Errorf("secret_patterns[%d] %q: %w", i, p, err)
		}
		out = append(out, re)
	}
	return out, nil
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
		if matchers != 1 {
			return nil, fmt.Errorf("arg_rules[%d]: set exactly one of allow_glob, deny_regex, max_bytes", i)
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
		pred := adjudicator.ArgPredicate{Tool: r.Tool, Arg: r.Arg, Reason: reason}
		switch {
		case r.AllowGlob != "":
			pred.Kind = adjudicator.ArgAllowGlob
			pred.Glob = r.AllowGlob
		case r.DenyRegex != "":
			re, err := regexp.Compile(r.DenyRegex)
			if err != nil {
				return nil, fmt.Errorf("arg_rules[%d]: invalid deny_regex: %w", i, err)
			}
			pred.Kind = adjudicator.ArgDenyRegex
			pred.Re = re
		case r.MaxBytes > 0:
			pred.Kind = adjudicator.ArgMaxBytes
			pred.N = r.MaxBytes
		}
		out = append(out, pred)
	}
	return out, nil
}

func describeArgPredicate(p adjudicator.ArgPredicate) string {
	reason := abi.ReasonName(p.Reason)
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
