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
	SafeSinks       []string          `json:"safe_sinks,omitempty"`
	Authorize       []AuthorizeRule   `json:"authorize,omitempty"`
	Sources         map[string]string `json:"sources,omitempty"`
	// ArgRules are per-tool ARGUMENT-VALUE constraints (issue #9) — the manifest
	// form of adjudicator.ArgPredicate. They extend the floor from "which tool"
	// to "which tool with which argument values". See ArgRule for the matchers.
	ArgRules []ArgRule `json:"arg_rules,omitempty"`
	// LintWrites (opt-in, issue #536) turns on the in-process code-lint rung for
	// whole-file writes: a write of unparseable Go/JSON is refused with MALFORMED
	// before it lands. Off by default; languages whose only checkers shell out
	// (Python/CUDA) defer (fail open). Maps 1:1 to adjudicator.Policy.LintWrites.
	LintWrites bool `json:"lint_writes,omitempty"`
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

// Runtime is the full manifest resolved for boot: the existing name-level
// adjudicator policy, plus IFC policy and host-authored source registrations.
type Runtime struct {
	Adjudicator    adjudicator.Policy
	Sources        map[string]provenance.Source
	SafeSinks      []string
	AuthorizeRules []AuthorizeRule
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
	return Runtime{
		Adjudicator:    p,
		Sources:        sources,
		SafeSinks:      safe,
		AuthorizeRules: auth,
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
	for _, t := range sortedKeys(p.Deny) {
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
	for _, tool := range sortedSourceKeys(rt.Sources) {
		fmt.Fprintf(&b, "                     %s -> %s\n", tool, rt.Sources[tool])
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

func sortedKeys(m map[string]abi.ReasonCode) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedSourceKeys(m map[string]provenance.Source) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func cloneSlice(s []string) []string {
	if len(s) == 0 {
		return nil
	}
	return append([]string(nil), s...)
}
