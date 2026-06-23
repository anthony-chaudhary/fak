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
	return &Adjudicator{policy: p, argByTool: indexArgPredicates(p.ArgPredicates)}
}

// SetPolicy swaps the policy (used by tests + the bench harness).
func (a *Adjudicator) SetPolicy(p Policy) {
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

	// SELF_MODIFY: a write-shaped call whose target matches a protected glob is a
	// PROVABLE refusal. Bounded disclosure: the witness carries ONLY the offending
	// glob, never the whole policy (deny channel is not a policy oracle).
	if writeShaped(c.Tool) {
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
	if g := commandSelfModify(args, p.SelfModifyGlobs); g != "" {
		return abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonSelfModify,
			By:      "monitor",
			Payload: abi.WitnessPayload{Claim: g},
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
	if g := synthToolSelfModify(args, &a.authored, p.SelfModifyGlobs); g != "" {
		return abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  abi.ReasonSelfModify,
			By:      "monitor",
			Payload: abi.WitnessPayload{Claim: g},
		}
	}
	// Record agent-authored scripts (the ledger half) so the NEXT exec is recognized
	// as a synth-tool. Placed AFTER the self-modify deny rungs, so a write into a
	// guarded tree (already denied above) never lands in the ledger.
	noteAuthoredScript(args, c.Tool, &a.authored, p.SelfModifyGlobs)

	// ARG-LEVEL value predicates (issue #9): the floor gates argument VALUES, not
	// just the tool name. A constrained arg that fails its predicate is a PROVABLE
	// refusal even for an otherwise allow-listed tool — denied here, never passed
	// to detection. Bounded disclosure: the witness names only the offending
	// tool.arg + the bound it broke, never the whole policy nor the arg value.
	if len(argPreds) > 0 {
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
	if p.LintWrites && wholeFileWrite(c.Tool) {
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
	if len(p.RedactFields) > 0 && args != nil {
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

func defaultDeny(p Policy, tool string) abi.Verdict {
	if p.Posture == PostureAdmitAndLog && lowRiskReadShaped(tool) {
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

// shellWriteVerbs are the shell idioms that WRITE to a path operand (#172 Hole 1).
// Detection is by substring on the command string and is deliberately conservative:
// these are write operations, so a self-modify guard that fires on them cannot
// block a read (`cat`/`grep`/`head` of a guarded file). A redirect (`>`/`>>`) is
// handled separately since it is punctuation, not a word. `shred` joins the
// destructive-overwrite family (`dd`/`truncate`): it overwrites then unlinks the
// named file and has NO read mode, so a substring match cannot block a read.
var shellWriteVerbs = []string{
	"sed -i", "tee ", "dd ", "truncate ", "shred ",
	"git apply", "git checkout", "git restore", "git stash",
	"cp ", "mv ", "install ", "patch ", "chmod ", "chown ", "ln ", "rm ",
	// In-place interpreter edits — the sed -i family across other interpreters
	// (#172 Hole 1, the porous-denylist residual). `perl -i`/`ruby -i` rewrite a
	// file in place exactly as `sed -i` does, but carry none of the leading tokens
	// above, so `perl -i -pe 's/x/y/' internal/abi/kernel.go` would otherwise
	// launder a self-edit past the floor that `sed -i` is caught on.
	"perl -i", "ruby -i",
}

// commandSelfModify reports the guarded glob a shell command would WRITE into, or
// "" if the command is absent / read-only / writes nowhere guarded. It is the
// shell-path dual of the targetPath self-modify check: it gates a Bash/exec call
// whose write target lives in the `command`/`cmd` string rather than a path arg.
//
// It fires only when BOTH hold: (1) the command string contains a write-shaped
// shell verb or an output redirect, AND (2) the command string contains a guarded
// glob fragment. Requiring a write verb is what keeps a read of a guarded file
// (`cat internal/abi/x.go`) allowed — only a write into the tree is refused. This
// is a substring floor, not a shell parser: it is intentionally over-broad (a
// guarded path mentioned anywhere alongside a write verb is refused) because the
// guarded trees are the kernel's own witness machinery, where a false refusal is
// cheap and a false allow is the self-grading-homework failure the floor exists to
// stop.
func commandSelfModify(args map[string]any, globs []string) string {
	cmd, ok := argString(args, "command")
	if !ok || cmd == "" {
		if cmd, ok = argString(args, "cmd"); !ok || cmd == "" {
			return ""
		}
	}
	g := matchGlob(cmd, globs)
	if g == "" {
		return "" // touches no guarded tree — nothing to guard
	}
	if !commandWrites(cmd) {
		return "" // a read of a guarded file is allowed; only writes are refused
	}
	return g
}

// commandWrites reports whether a shell command string contains a write-shaped
// verb or an output redirect (`>`/`>>`). It is substring/byte level, matching the
// conservative floor commandSelfModify documents.
func commandWrites(cmd string) bool {
	lc := strings.ToLower(cmd)
	for _, v := range shellWriteVerbs {
		if strings.Contains(lc, v) {
			return true
		}
	}
	// `find … -delete` / `find … -exec <writer>`: a delete/exec-write idiom whose
	// effect is a WRITE (it removes or rewrites the matched files) but which carries
	// none of shellWriteVerbs' leading tokens — `find internal/abi -delete` would
	// otherwise launder a deletion of a guarded tree past this floor (#172 Hole 1,
	// the porous-denylist `find -delete` gap the audit named). The `-delete` action
	// and an `-exec`/`-execdir` running anything are both treated as writes; a plain
	// read-only `find` (no `-delete`/`-exec`) stays allowed.
	if strings.Contains(lc, "find ") &&
		(strings.Contains(lc, "-delete") ||
			strings.Contains(lc, "-exec")) {
		return true
	}
	// `git -C <dir> <mutating-subcommand>`: the `-C <dir>` flag runs git AS IF in
	// <dir>, so a guarded tree can be the write TARGET via the directory operand
	// even when the only mutating verbs (`reset`, `commit`, `clean`, …) are not in
	// shellWriteVerbs (#172 Hole 1 residual — `git -C internal/abi reset --hard`
	// rewrites the guarded tree but carries none of the `git apply`/`git checkout`
	// tokens caught above). Fire when a `git -C` invocation also names a
	// tree-mutating subcommand; a `git -C <dir> status`/`log`/`diff` read does not.
	// The `-C` flag is CASE-SENSITIVE (lowercase `-c` selects a config override, not
	// a directory), so the flag is matched against the ORIGINAL command; only the
	// subcommand below is matched case-insensitively against lc.
	if strings.Contains(cmd, "git -C ") || strings.Contains(lc, "git --git-dir") {
		for _, sub := range []string{"reset", "commit", "clean", " rm ", " mv ", "checkout", "restore", "stash", "apply"} {
			if strings.Contains(lc, sub) {
				return true
			}
		}
	}
	// Interpreter inline-program write: a general-purpose interpreter invoked with
	// an inline-program flag (`python -c '…'`, `node -e '…'`, …) can open and rewrite
	// a guarded file from inside the program string with NO shellWriteVerb token and
	// NO `>` redirect — `python3 -c "open('internal/abi/x.go','w').write(...)"` would
	// otherwise launder a self-edit past every rung above (#172 Hole 1 residual, the
	// `perl -i`/`ruby -i` gap one interpreter further out — python/node are the ones
	// most likely on a coding agent's PATH). The program string is opaque to a
	// substring floor, so — matching this floor's documented "a guarded tree named
	// alongside a writer is refused; a false refusal is cheap" stance — an inline
	// interpreter program is treated as write-shaped. commandSelfModify only reaches
	// here once the command already names a guarded glob, so a `python -c` that
	// touches nothing guarded is unaffected; only an inline program that mentions a
	// guarded tree is refused. A read-shaped interpreter call WITHOUT an inline-
	// program flag (`python score.py`, `node app.js`) is not matched here.
	for _, ev := range interpreterEvalFlags {
		if strings.Contains(lc, ev) {
			return true
		}
	}
	// In-place / line-editor writes that carry none of the tokens above (#172 Hole 1
	// residual, the sed -i family two tools further out):
	//
	//   - `awk -i inplace` / `gawk -i inplace`: GNU awk's in-place edit flag rewrites
	//     the named file exactly as `sed -i`/`perl -i`/`ruby -i` (already guarded) do,
	//     but `awk` carries no `-i` leading token in shellWriteVerbs. The flag is the
	//     `inplace` extension specifically — a read-only `awk '{print}' file` has no
	//     `-i inplace` and stays allowed. Match the `-i inplace` token sequence.
	//   - `ed` / `ex`: scripted line editors that WRITE their file operand with no
	//     redirect and no caught verb (`ex -s -c wq file`, `ed -s file`). They are
	//     short tokens, so they are matched only in COMMAND POSITION (start of the
	//     command or right after a shell separator), never as a substring of `sed`,
	//     `indexed`, `fixed`, `expr`, etc. commandSelfModify only reaches here once a
	//     guarded glob is already named, so a benign `ed`/`ex` over an unguarded path
	//     is unaffected; only an editor invocation alongside a guarded tree is refused.
	if strings.Contains(lc, "-i inplace") {
		return true
	}
	if hasCommandLeadingToken(lc, "ed") || hasCommandLeadingToken(lc, "ex") {
		return true
	}
	// Archive EXTRACTION / sync INTO a guarded tree (#172 Hole 1 residual): an
	// extractor writes files into a destination directory with none of the tokens
	// above — `tar -xf a.tar -C internal/abi`, `unzip -d internal/abi a.zip`,
	// `rsync src/ internal/abi/` overwrite the guarded tree via the extract/sync
	// DESTINATION. commandSelfModify only reaches here once a guarded glob is named,
	// so the job is to fire on the WRITE (extract/sync) idioms while leaving the
	// READ idioms over the same tree allowed (tar LIST/CREATE, unzip LIST/test).
	if archiveExtractsOrSyncs(lc) {
		return true
	}
	// Output redirect: a bare '>' or '>>' (not the '2>&1'/'>&' fd-dup forms, which
	// are not a file write per se, but conservatively a '>' anywhere is a write).
	return strings.Contains(cmd, ">")
}

// archiveExtractsOrSyncs reports whether lc (the lowercased command) is an archive
// EXTRACTION or a sync that WRITES files into its destination. It deliberately fires
// only on the write idioms, so a read over the same guarded tree stays allowed
// (commandSelfModify has already confirmed a guarded glob is named in the command):
//
//   - `tar` writes only in EXTRACT mode (`-x` / a leading `x` in the bundled-flag
//     form `tar xf`); LIST (`-t`) and CREATE-archive (`-c`, which READS the tree into
//     an archive written elsewhere) are not writes into the tree. So a `tar` command
//     is a write here iff it carries an extract flag.
//   - `unzip` extracts by default; it is a READ only in LIST/test mode (`-l` / `-v` /
//     `-t` / `-z`). So an `unzip` command is a write here unless it carries a
//     list/test flag.
//   - `rsync` writes its DESTINATION operand; once a guarded glob is named anywhere in
//     an rsync command, conservatively treat it as a write into the tree (the floor's
//     documented "a guarded tree named alongside a writer is refused" stance).
func archiveExtractsOrSyncs(lc string) bool {
	if hasCommandLeadingToken(lc, "rsync") {
		return true
	}
	if hasCommandLeadingToken(lc, "tar") && tarExtracts(lc) {
		return true
	}
	if hasCommandLeadingToken(lc, "unzip") && !unzipListsOnly(lc) {
		return true
	}
	return false
}

// tarExtracts reports whether a tar command is in EXTRACT mode (a write), via either
// the `-x` flag or the classic bundled form (`tar xf …`, `tar xzf …`). A tar command
// with no extract flag (LIST `-t`, CREATE `-c`) is not a write into the named tree.
func tarExtracts(lc string) bool {
	if strings.Contains(lc, "-x") || strings.Contains(lc, "--extract") {
		return true
	}
	// Bundled flags: the first word after `tar ` whose run of letters includes 'x'
	// (e.g. `xf`, `xzf`, `xvf`). Restrict to a leading flag cluster so a path or
	// archive name containing 'x' cannot trip it.
	i := strings.Index(lc, "tar ")
	if i < 0 {
		return false
	}
	rest := strings.TrimLeft(lc[i+len("tar "):], " ")
	// Inspect only the first whitespace-delimited token, and only if it is a bare
	// flag cluster (letters, no '/' or '.', i.e. not a path/archive operand).
	end := strings.IndexAny(rest, " \t")
	if end < 0 {
		end = len(rest)
	}
	tok := rest[:end]
	if tok == "" || strings.ContainsAny(tok, "/.-") {
		return false // a path/archive operand or a `-`-prefixed flag handled above
	}
	return strings.Contains(tok, "x")
}

// unzipListsOnly reports whether an unzip command is a LIST/test (a read), not an
// extraction. unzip extracts by default, so only the list/test flags make it a read.
func unzipListsOnly(lc string) bool {
	for _, f := range []string{"-l", "-v", "-t", "-z"} {
		if strings.Contains(lc, f+" ") || strings.HasSuffix(lc, f) {
			return true
		}
	}
	return false
}

// hasCommandLeadingToken reports whether tok appears in lc as a whole word in
// COMMAND POSITION — at the very start of the command (after any leading whitespace),
// or immediately after a shell command separator (`;`, `|`, `&`, `(`, newline, or the
// `&&`/`||` forms, all of which end in one of those bytes) followed by optional
// whitespace — and is itself followed by a space (an argument follows). This pins a
// short, false-positive-prone editor name like `ed`/`ex` to the slot where it names a
// program to RUN, so it never trips on `sed`, `indexed`, `fixed`, `expr`, a path
// fragment, or an ARGUMENT that happens to be the word (`grep ed file` is a read, not
// an `ed` invocation — the `ed` there follows an argument space, not a separator).
// lc is the lowercased command.
func hasCommandLeadingToken(lc, tok string) bool {
	from := 0
	for {
		i := strings.Index(lc[from:], tok+" ")
		if i < 0 {
			return false
		}
		at := from + i
		// Walk back over run-of-whitespace to the boundary byte (or string start).
		j := at
		for j > 0 && (lc[j-1] == ' ' || lc[j-1] == '\t') {
			j--
		}
		if j == 0 || isShellSep(lc[j-1]) {
			return true
		}
		from = at + 1
	}
}

// isShellSep reports whether b is a shell command separator, so a token following it
// (modulo whitespace) is in command position. A plain space is deliberately NOT a
// separator: a bare argument space does not start a new command, so `grep ed file`
// (the word `ed` as a search argument) is not an `ed` invocation. The `&&`/`||`
// chains end in `&`/`|`, which are covered.
func isShellSep(b byte) bool {
	switch b {
	case ';', '|', '&', '(', '\n':
		return true
	}
	return false
}

// interpreterEvalFlags are the `<interpreter> <inline-program-flag>` prefixes that
// run code from an opaque string argument able to write a file directly (#172
// Hole 1 residual). They are the interpreter analogue of shellWriteVerbs: python
// and node are the general-purpose runtimes most likely on a coding agent's PATH,
// and each carries an inline-eval flag (`-c` / `-e` / `--eval` / `-p`/`--print`,
// which evaluates AND can have side effects). Matched case-insensitively against
// the lowercased command; the trailing space pins the flag as its own token so a
// path like `mynode-eval.txt` cannot trip it.
var interpreterEvalFlags = []string{
	"python -c ", "python3 -c ", "python -c\"", "python3 -c\"",
	"node -e ", "node --eval ", "node -e\"", "node --eval\"",
	"node -p ", "node --print ",
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
