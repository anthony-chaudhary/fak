// Package witness is the in-process realization of the require-witness rung — the
// DOS dos_verify effect-verify, brought inside the kernel.
//
// THE PRINCIPLE (the same one provenance enforces for trust): a worker's claim
// that it DID something is a self-report. "I shipped the test phase", "I created
// the artifact", "the prerequisite landed" — a derailed or lying agent asserts
// these as readily as an honest one. The require-witness gate refuses to take the
// claim on faith: it corroborates the claimed EFFECT against evidence the agent did
// not author — git ancestry, object existence, a tracked path, the filesystem —
// before the gated call is allowed. CONFIRMED opens the gate; REFUTED or
// uncorroborated keeps it closed (fail-closed).
//
// This is the in-process dual of the DOS dos_verify MCP tool ("confirm a claim
// landed from git evidence rather than a worker's self-report"): the SAME
// git-evidence check, run at the tool-call boundary with no process spawn, so the
// kernel itself — not a downstream reviewer — refuses an unwitnessed effect.
//
// CLAIM GRAMMAR (the WitnessPayload.Claim string an adjudicator attaches to a
// VerdictRequireWitness):
//
//	ancestor:<ref>    the ref is an ancestor of HEAD     (a phase actually shipped)
//	commit:<ref>      the ref resolves to a commit object (the commit exists)
//	committed:<path>  the path is tracked in git          (the file was really added)
//	path:<path>       the path exists on disk             (the artifact is present)
//	grep:<pattern>    some commit message in history matches
//	clean:<pathspec>  the working tree is clean there      (a green-tree ship)
//	notests:<ref>     the commit did NOT edit its own gating tests (reward-hack guard)
//
// The notests rung is the dual of the others: where the rest CONFIRM that a
// claimed effect is present, notests REFUTES when a ship-commit modified the very
// *_test.go files that gate it — editing the tests you must pass is the canonical
// reward-hack. CONFIRMED means the commit touched no test file (clean); REFUTED
// names the suspicious commit for a human; a bad/unknown ref or missing git
// abstains (never a false CONFIRM that lets a test-rewriting commit through).
//
// An unrecognized or empty claim, or any environment where git is unavailable,
// resolves to ABSTAIN (fail-to-abstain) — the witness never blocks on its own
// uncertainty; the kernel's fail-closed default turns an abstain into a deny.
package witness

import (
	"context"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Runner executes a git subcommand in dir and returns (stdout, exitCode, err). It
// is injectable so tests drive deterministic evidence without a real repo; the
// default shells out to git. err is non-nil only for a failure to RUN git (git
// missing); a non-zero exit with git present is reported via code, not err.
type Runner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// gitRunner is the default: it runs the real git binary. A non-zero git exit is
// returned in code (not err); err signals git could not be executed at all.
func gitRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = nil
	err := cmd.Run()
	if err == nil {
		return out.String(), 0, nil
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return out.String(), ee.ExitCode(), nil // git ran, returned non-zero
	}
	return "", -1, err // git could not be executed
}

// Resolver is the git-backed WitnessResolver. Construct with New (real git) or
// NewWithRunner (injected evidence). dir is the repo to verify against ("" = git's
// own default discovery from the process cwd).
type Resolver struct {
	run Runner
	dir string
}

// New is the real-git resolver, registered as "dos_verify".
func New() *Resolver { return &Resolver{run: gitRunner} }

// NewWithRunner injects a Runner + dir (tests, or an alternate evidence source).
func NewWithRunner(r Runner, dir string) *Resolver { return &Resolver{run: r, dir: dir} }

// Resolve corroborates the claim against independent evidence and returns the
// outcome. Parse-failures and a missing git both yield Abstain (never a false
// Confirm, never a crash).
func (r *Resolver) Resolve(ctx context.Context, c *abi.ToolCall, claim string) abi.WitnessOutcome {
	kind, arg, ok := splitClaim(claim)
	if !ok {
		return abi.WitnessAbstain
	}
	switch kind {
	case "ancestor":
		// exit 0 => arg is an ancestor of HEAD (shipped); 1 => not; other => abstain.
		_, code, err := r.run(ctx, r.dir, "merge-base", "--is-ancestor", arg, "HEAD")
		if err != nil {
			return abi.WitnessAbstain
		}
		switch code {
		case 0:
			return abi.WitnessConfirmed
		case 1:
			return abi.WitnessRefuted
		default:
			return abi.WitnessAbstain // a bad/unknown ref is not evidence of absence
		}
	case "commit":
		_, code, err := r.run(ctx, r.dir, "cat-file", "-e", arg+"^{commit}")
		if err != nil {
			return abi.WitnessAbstain
		}
		if code == 0 {
			return abi.WitnessConfirmed
		}
		return abi.WitnessRefuted
	case "committed":
		// ":/"+arg is git's repo-root-anchored magic pathspec, so a committed claim
		// is cwd-INDEPENDENT (it means "tracked at this repo-relative path", not
		// "tracked relative to wherever the kernel happens to run").
		_, code, err := r.run(ctx, r.dir, "ls-files", "--error-unmatch", "--", ":/"+arg)
		if err != nil {
			return abi.WitnessAbstain
		}
		if code == 0 {
			return abi.WitnessConfirmed
		}
		return abi.WitnessRefuted // git ran and the path is not tracked
	case "path":
		// filesystem evidence; respects r.dir as a relative base.
		p := arg
		if r.dir != "" && !isAbs(arg) {
			p = r.dir + string(os.PathSeparator) + arg
		}
		if _, err := os.Stat(p); err == nil {
			return abi.WitnessConfirmed
		} else if os.IsNotExist(err) {
			return abi.WitnessRefuted
		}
		return abi.WitnessAbstain
	case "grep":
		out, code, err := r.run(ctx, r.dir, "log", "--grep", arg, "-1", "--format=%H")
		if err != nil || code != 0 {
			return abi.WitnessAbstain
		}
		if strings.TrimSpace(out) != "" {
			return abi.WitnessConfirmed
		}
		return abi.WitnessRefuted
	case "clean":
		// A clean working tree corroborates a "green-tree" ship: no uncommitted
		// changes under the pathspec. `git status --porcelain <arg>` empty => clean
		// (confirmed); any output => dirty (refuted); git unavailable => abstain.
		out, code, err := r.run(ctx, r.dir, "status", "--porcelain", "--", arg)
		if err != nil || code != 0 {
			return abi.WitnessAbstain
		}
		if strings.TrimSpace(out) == "" {
			return abi.WitnessConfirmed
		}
		return abi.WitnessRefuted
	case "notests":
		// The reward-hack guard: list the files <ref> touched and REFUTE if any is a
		// gating test file. `git show --name-only --format=` (empty format) prints the
		// commit's file list with no header; a merge or empty commit prints nothing.
		// CONFIRMED = touched no test (clean ship); REFUTED = edited a gating test;
		// a failure to run (bad ref / git missing) abstains, never a false CONFIRM.
		out, code, err := r.run(ctx, r.dir, "show", "--name-only", "--format=", arg)
		if err != nil || code != 0 {
			return abi.WitnessAbstain
		}
		for _, line := range strings.Split(out, "\n") {
			if isGatingTestPath(strings.TrimSpace(line)) {
				return abi.WitnessRefuted // the commit edited a test it must pass
			}
		}
		return abi.WitnessConfirmed
	}
	return abi.WitnessAbstain
}

// isGatingTestPath reports whether a repo path is a test file whose edit by a
// ship-commit is a reward-hack signal. It matches the Go convention (a basename
// ending in "_test.go") — the language's own definition of a gating test — and is
// deliberately narrow: a non-test source edit, or a fixture/testdata file, is not
// flagged, so the guard refuses only the unambiguous "rewrote the assertions" case.
func isGatingTestPath(p string) bool {
	if p == "" {
		return false
	}
	// normalize the trailing path segment without importing path (keep this file's
	// minimal-deps shape); both separators appear in git output across platforms.
	if i := strings.LastIndexAny(p, "/\\"); i >= 0 {
		p = p[i+1:]
	}
	return strings.HasSuffix(p, "_test.go")
}

// splitClaim parses "kind:arg". Returns ok=false for an empty/colon-less claim.
func splitClaim(claim string) (kind, arg string, ok bool) {
	claim = strings.TrimSpace(claim)
	i := strings.IndexByte(claim, ':')
	if i <= 0 || i == len(claim)-1 {
		return "", "", false
	}
	return strings.ToLower(claim[:i]), strings.TrimSpace(claim[i+1:]), true
}

func isAbs(p string) bool {
	return strings.HasPrefix(p, "/") || (len(p) > 1 && p[1] == ':') // unix or windows drive
}

// Default is the registered resolver.
var Default = New()

func init() {
	// "dos_verify": the in-process git-evidence effect-verify backing the
	// require-witness gate. The kernel consults abi.Witnesses() on a RequireWitness
	// verdict; this turns a claimed effect into a corroborated (or refused) one.
	abi.RegisterWitnessResolver("dos_verify", Default)
	abi.RegisterCapability("witness.dos_verify")
}
