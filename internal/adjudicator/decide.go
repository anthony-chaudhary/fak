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
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

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
}

// Posture selects the policy's default-deny behavior after all provable refusal
// checks have passed.
type Posture uint8

const (
	// PostureFailClosed keeps the v0.1 floor: anything not affirmatively allowed
	// is denied with DEFAULT_DENY.
	PostureFailClosed Posture = iota
	// PostureAdmitAndLog downgrades low-risk read-shaped DEFAULT_DENY decisions
	// to ALLOW while carrying forensic metadata that records the would-have-denied
	// reason. Explicit denies, self-modify, arg-rule violations, and write-shaped
	// default denies still fail closed.
	PostureAdmitAndLog
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
}

// Adjudicator is the reference monitor. Construct with New; the default instance
// registers itself in init().
type Adjudicator struct {
	mu     sync.RWMutex
	policy Policy
	// argByTool is p.ArgPredicates grouped by Tool, rebuilt on every SetPolicy. It
	// is the hot-path index for issue #9: Adjudicate evaluates only the predicates
	// that target the call's tool — O(predicates-for-this-tool) — instead of
	// scanning every predicate in the policy on every call. Without it, a policy
	// with N per-tool arg rules costs O(N) per call even for tools with no rules,
	// so the floor's per-call cost grew with total policy size, not the call's own
	// rule count.
	argByTool map[string][]ArgPredicate
	// authored is the per-run ledger of agent-authored script paths (#543): the set
	// of scripts THIS agent wrote earlier in the run, used to recognize a later
	// `python helper.py` as a self-synthesized-tool invocation rather than an opaque
	// binary exec (see synthtool.go). A sync.Map gives a lock-free concurrent set
	// alongside the policy RWMutex; its zero value is a ready empty ledger, so New
	// need not initialize it. Cleared at a task boundary via ResetRun.
	authored sync.Map
}

// New builds an adjudicator with the given policy.
func New(p Policy) *Adjudicator {
	p.Profile = sanitizeProfile(p.Profile) // floor invariant: a profile may narrow only
	return &Adjudicator{policy: p, argByTool: indexArgPredicates(p.ArgPredicates)}
}

// SetPolicy swaps the policy (used by tests + the bench harness).
func (a *Adjudicator) SetPolicy(p Policy) {
	p.Profile = sanitizeProfile(p.Profile) // floor invariant: a profile may narrow only
	a.mu.Lock()
	a.policy = p
	a.argByTool = indexArgPredicates(p.ArgPredicates)
	a.mu.Unlock()
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
	a.mu.RLock()
	out := make([]string, 0, len(a.policy.Deny))
	for t := range a.policy.Deny {
		out = append(out, t)
	}
	a.mu.RUnlock()
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
// scope the self-modify check to write-shaped calls).
func writeShaped(tool string) bool {
	for _, p := range []string{"write", "edit", "delete", "patch", "put", "exec", "run", "modify", "create"} {
		if strings.Contains(strings.ToLower(tool), p) {
			return true
		}
	}
	return false
}

// lowRiskReadShaped reports whether a default-denied call is safe for the
// admit-and-log posture. It is intentionally name-based and conservative: caller
// Meta is model-controlled and cannot widen authority.
func lowRiskReadShaped(tool string) bool {
	tool = strings.ToLower(tool)
	if tool == "" || writeShaped(tool) {
		return false
	}
	for _, p := range []string{"read_", "get_", "search_", "list_", "lookup_", "find_", "calc"} {
		if strings.HasPrefix(tool, p) {
			return true
		}
	}
	return tool == "calculate"
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

// Adjudicate is the decision. It is pure and allocation-light on the deny/allow
// paths; only the (rare) TRANSFORM path resolves + re-stores args.
func (a *Adjudicator) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	a.mu.RLock()
	p := a.policy
	argPreds := a.argByTool[strings.ToLower(c.Tool)] // predicates targeting THIS tool (case-insensitive)
	a.mu.RUnlock()

	// Explicit provable refusal.
	if r, ok := p.Deny[c.Tool]; ok {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: r, By: "monitor"}
	}

	// Decode args once for the structural checks.
	args := decodeArgs(ctx, c)

	// Coarse risk class for the RungProfile (#666). Computed ONCE from the DECODED
	// args (never model-controlled Meta), and ONLY when a profile is installed — a
	// nil profile runs every rung regardless (pr.runs == true), so the default floor
	// pays zero classification cost and stays byte-for-byte identical to HEAD.
	pr := p.Profile
	var cl class
	if pr != nil {
		cl = riskClass(c.Tool, args)
	}

	// SELF_MODIFY: a write-shaped call whose target matches a protected glob is a
	// PROVABLE refusal. Bounded disclosure: the witness carries ONLY the offending
	// glob, never the whole policy (deny channel is not a policy oracle).
	if pr.runs(cl, rungSelfModify) && writeShaped(c.Tool) {
		if g := matchGlob(targetPath(args), p.SelfModifyGlobs); g != "" {
			return abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  abi.ReasonSelfModify,
				By:      "monitor",
				Payload: abi.WitnessPayload{Claim: g},
			}
		}
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
		if g := commandSelfModify(args, p.SelfModifyGlobs); g != "" {
			return abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  abi.ReasonSelfModify,
				By:      "monitor",
				Payload: abi.WitnessPayload{Claim: g},
			}
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
			return abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  abi.ReasonSelfModify,
				By:      "monitor",
				Payload: abi.WitnessPayload{Claim: g},
			}
		}
		// Record agent-authored scripts (the ledger half) so the NEXT exec is
		// recognized as a synth-tool. Placed AFTER the self-modify deny rung, so a
		// write into a guarded tree (already denied above) never lands in the ledger.
		noteAuthoredScript(args, c.Tool, &a.authored, p.SelfModifyGlobs)
	}

	// ARG-LEVEL value predicates (issue #9): the floor gates argument VALUES, not
	// just the tool name. A constrained arg that fails its predicate is a PROVABLE
	// refusal even for an otherwise allow-listed tool — denied here, never passed
	// to detection. Bounded disclosure: the witness names only the offending
	// tool.arg + the bound it broke, never the whole policy nor the arg value.
	if pr.runs(cl, rungArgPredicate) && len(argPreds) > 0 {
		if v, denied := evalArgPredicates(argPreds, c.Tool, args); denied {
			return v
		}
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
	if pr.runs(cl, rungLintWrite) && p.LintWrites && wholeFileWrite(c.Tool) {
		if w := lintWriteMalformed(targetPath(args), args); w != "" {
			return abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  abi.ReasonMalformed,
				By:      "monitor",
				Payload: abi.WitnessPayload{Claim: w},
			}
		}
	}

	// TRANSFORM: redact a secret-shaped arg field before dispatch.
	if pr.runs(cl, rungTransform) && len(p.RedactFields) > 0 && args != nil {
		if newArgs, changed := redact(args, p.RedactFields); changed {
			if ref, ok := putJSON(ctx, newArgs); ok {
				return abi.Verdict{Kind: abi.VerdictTransform, By: "monitor",
					Payload: abi.TransformPayload{NewArgs: ref}}
			}
		}
	}

	// Affirmative allow.
	if p.Allow[c.Tool] {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "monitor"}
	}
	for _, pre := range p.AllowPrefix {
		if strings.HasPrefix(c.Tool, pre) {
			return abi.Verdict{Kind: abi.VerdictAllow, By: "monitor"}
		}
	}

	// Nothing affirmatively allowed it — fail-closed default deny.
	return defaultDeny(p, c.Tool)
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
	return (p.Posture == PostureAdmitAndLog && lowRiskReadShaped(tool)) || p.complainFor(tool)
}

// NeverAdmits (on the live Adjudicator) is the locked read of the installed floor's
// Policy.NeverAdmits — the args-independent "this name can never be Allowed" query the
// inbound tool-def compactor asks. Reads the current policy under the lock so it is safe
// to call per request from the serving path (where the floor lives in adjudicator.Default
// rather than a host-held Policy value). A pure read: it never mutates run-state.
func (a *Adjudicator) NeverAdmits(tool string) bool {
	a.mu.RLock()
	p := a.policy
	a.mu.RUnlock()
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
	// Fail-safe against an UNCONFIGURED floor: a Policy with no affirmative-allow
	// surface at all (empty Allow, empty AllowPrefix, fail-closed posture) denies
	// EVERY tool — true by the rule below, but as a DROP signal that is almost always
	// "the floor was never installed" rather than "deliberately deny all advertised
	// tools." Pruning every tool-def against a zero floor would be a catastrophic
	// over-drop, so we refuse to prune anything when there is nothing to admit. A real
	// floor (any Allow entry or any AllowPrefix) re-enables pruning of the names it
	// genuinely never admits.
	if len(p.Allow) == 0 && len(p.AllowPrefix) == 0 {
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
	return !p.admitAndLog(tool)
}

func defaultDeny(p Policy, tool string) abi.Verdict {
	if p.admitAndLog(tool) {
		// Admit-and-log record (#671): the default-deny rung is the refusal being
		// suppressed, so the record carries would_deny = its reason name via
		// abi.ReasonName — the forensic field the promotion ledger (#672) folds. Both
		// the complain-set and the global read-shaped path carry it identically.
		return abi.Verdict{
			Kind: abi.VerdictAllow,
			By:   "monitor",
			Meta: map[string]string{
				"posture":    "admit_and_log",
				"would_deny": abi.ReasonName(abi.ReasonDefaultDeny),
			},
		}
	}
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "monitor"}
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
func matchGlob(path string, globs []string) string {
	if path == "" {
		return ""
	}
	for _, g := range globs {
		if g != "" && strings.Contains(path, g) {
			return g
		}
	}
	return ""
}

// interpreterEvalSpec pairs a general-purpose interpreter with the inline-program flags
// whose presence as a TOKEN means it runs code from an opaque string argument able to
// write a file directly.
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

// evalArgPredicates runs every predicate that targets tool against the decoded
// args, returning the first violation's Deny verdict (and true) or a zero verdict
// (and false) if all pass. It is pure and allocation-light: it only iterates the
// (typically tiny) predicate slice and reads scalar arg values.
func evalArgPredicates(preds []ArgPredicate, tool string, args map[string]any) (abi.Verdict, bool) {
	for i := range preds {
		pr := &preds[i]
		if !strings.EqualFold(pr.Tool, tool) {
			continue // case-insensitive: a "Bash" rule still gates a "bash" call (matches the index)
		}
		val, present := argString(args, pr.Arg)
		switch pr.Kind {
		case ArgAllowGlob:
			// Positive requirement: a missing OR out-of-bounds value fails closed.
			if !present || !pathUnderGlob(pr.Glob, val) {
				return argDeny(pr, "allow_glob "+pr.Glob), true
			}
		case ArgDenyRegex:
			if present && pr.Re != nil && pr.Re.MatchString(val) {
				return argDeny(pr, "deny_regex /"+pr.Re.String()+"/"), true
			}
		case ArgMaxBytes:
			if present && len(val) > pr.N {
				return argDeny(pr, "max_bytes "+strconv.Itoa(pr.N)), true
			}
		}
	}
	return abi.Verdict{}, false
}

// argDeny builds the bounded-disclosure Deny for a violated arg predicate. The
// witness Claim names the offending tool.arg and the bound it broke — never the
// arg value (which may be sensitive) nor the rest of the policy.
func argDeny(pr *ArgPredicate, detail string) abi.Verdict {
	reason := pr.Reason
	if reason == abi.ReasonNone {
		reason = abi.ReasonPolicyBlock
	}
	return abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  reason,
		By:      "monitor",
		Payload: abi.WitnessPayload{Claim: pr.Tool + "." + pr.Arg + " " + detail},
	}
}

// argString returns the string form of args[key] and whether the key was present.
// A scalar (string / number / bool) renders to its natural string; an
// object / array / null is treated as absent — arg predicates target scalar
// values, and a non-scalar where a path or command is expected fails the
// positive (ArgAllowGlob) requirement, which is the fail-closed outcome.
func argString(args map[string]any, key string) (string, bool) {
	v, ok := args[key]
	if !ok {
		return "", false
	}
	switch s := v.(type) {
	case string:
		return s, true
	case float64:
		return strconv.FormatFloat(s, 'g', -1, 64), true
	case bool:
		if s {
			return "true", true
		}
		return "false", true
	default:
		return "", false
	}
}

// pathUnderGlob reports whether value is a path admitted by an allow-glob. Two
// forms, slash-normalized and path-cleaned first so a backslash or "./" cannot
// dodge the check:
//   - "<dir>/**": CONTAINMENT — value must resolve to <dir> itself or inside it;
//     a "../" escape (which path.Clean folds out) fails. This is the
//     "allow write_file only under ./out/**" form. A bare "**" / "./**" admits
//     any relative path that does not escape the working root.
//   - otherwise: a single path.Match (single-segment '*'/'?' wildcards).
func pathUnderGlob(glob, value string) bool {
	norm := func(s string) string { return path.Clean(strings.ReplaceAll(s, `\`, "/")) }
	v := norm(value)
	if strings.HasSuffix(glob, "/**") || glob == "**" {
		dir := norm(strings.TrimSuffix(glob, "**")) // "./out/**" -> "out"; "**" -> "."
		if dir == "." || dir == "/" {               // "**", "./**": any non-escaping relative path
			return v != ".." && !strings.HasPrefix(v, "../") && !strings.HasPrefix(v, "/")
		}
		return v == dir || strings.HasPrefix(v, dir+"/")
	}
	ok, err := path.Match(norm(glob), v)
	return err == nil && ok
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
		Allow: map[string]bool{
			// safe inspect / build / test tools a coding agent drives
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
}
