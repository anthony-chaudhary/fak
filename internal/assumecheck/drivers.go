package assumecheck

// drivers.go — the four name-resolved witness drivers (#3821, epic #3818 C3) for
// the probe kinds the C1 kernel froze: git-ancestry, worktree-grep,
// command-probe, and config-flag. Each driver GATHERS evidence and hands it to
// the pure kernel; none judges. The probe bodies are the proven ones from
// internal/recall/reverify.go (verifyGitSHA / verifyFlag / verifyPath) and the
// runner seams + exit mapping are internal/witness/witness.go's — copied as a
// TEMPLATE, not imported: witness.go's resolver lives on the abi require-witness
// plane (Abstain/Confirmed/Refuted) and its "dos_verify" id is reserved to it.
//
// THE TRI-STATE MAPPING (load-bearing, shared by every driver): a probe exit of
// 0 is a witnessed "holds" (Witnessed=true, Holds=true); an exit of 1 is a
// witnessed "does not hold" (Witnessed=true, Holds=false); any other exit, a
// probe that could not start, or git missing entirely is NOT a decision —
// Evidence{Witnessed:false}, which the kernel verdicts UNVERIFIABLE, never a
// guess in either direction.
//
// win32 discipline: every exec.Command a driver builds goes through
// windowgate.ConfigureBackgroundCommand (no console-window flash), and the
// command-probe runner — the one that may spawn a process TREE — additionally
// gets procguard.ConfigureProcessTreeCancel + a bounded WaitDelay so a
// cancelled probe cannot orphan its descendants (#3106).

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Runner executes a git subcommand in dir and returns (stdout, exitCode, err).
// Injectable so driver tests script deterministic evidence without a repo; the
// default shells out to git. err is non-nil only for a failure to RUN git (git
// missing); a non-zero exit with git present is reported via code, not err.
type Runner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// CommandRunner executes an argv vector in dir and returns (stdout+stderr,
// exitCode, err). err is non-nil only when the command could not be started; a
// non-zero process exit is reported in code, not err.
type CommandRunner func(ctx context.Context, dir string, argv ...string) (stdout string, code int, err error)

// defaultGitRunner runs the real git binary (the witness.go gitRunner shape).
func defaultGitRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
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

// defaultCommandRunner runs an arbitrary probe argv (the witness.go
// commandRunner shape). Probes may fork their own descendant tree, and bare
// CommandContext cancel is single-PID — tree-kill on cancel and bound the reap
// so a hung probe's subtree is never orphaned (#3106).
func defaultCommandRunner(ctx context.Context, dir string, argv ...string) (string, int, error) {
	if len(argv) == 0 || strings.TrimSpace(argv[0]) == "" {
		return "", -1, exec.ErrNotFound
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	windowgate.ConfigureBackgroundCommand(cmd)
	procguard.ConfigureProcessTreeCancel(cmd)
	cmd.WaitDelay = 10 * time.Second
	if dir != "" {
		cmd.Dir = dir
	}
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &out
	err := cmd.Run()
	if err == nil {
		return out.String(), 0, nil
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return out.String(), ee.ExitCode(), nil
	}
	return out.String(), -1, err
}

func init() {
	RegisterDriver(NewGitAncestryDriver())
	RegisterDriver(NewWorktreeGrepDriver())
	RegisterDriver(NewCommandProbeDriver())
	RegisterDriver(NewConfigFlagDriver())
}

// ---------------------------------------------------------------------------
// git-ancestry
// ---------------------------------------------------------------------------

// GitAncestryDriver witnesses WitnessGitAncestry targets: is Target.Ref a real
// commit that SHIPPED? The probe chain is recall/reverify.go's verifyGitSHA
// verbatim — `git cat-file -e <ref>^{commit}` (does it resolve), `git rev-parse
// --verify` (pin the full SHA), `git merge-base --is-ancestor <full> HEAD`
// (is it reachable) — plus its revertedBy scan. HOLDS therefore means
// REACHABLE-AND-NOT-REVERTED, the stronger of the two candidate readings: a
// commit whose effect a later revert removed is exactly the stale claim this
// witness kind exists to catch, so mere reachability must not confirm it.
type GitAncestryDriver struct {
	run Runner
	dir string
}

// NewGitAncestryDriver is the real-git driver registered for git-ancestry.
func NewGitAncestryDriver() *GitAncestryDriver {
	return &GitAncestryDriver{run: defaultGitRunner}
}

// NewGitAncestryDriverWithRunner injects a Runner + dir (tests, or an alternate
// evidence source).
func NewGitAncestryDriverWithRunner(r Runner, dir string) *GitAncestryDriver {
	return &GitAncestryDriver{run: r, dir: dir}
}

func (d *GitAncestryDriver) Kind() WitnessKind { return WitnessGitAncestry }

func (d *GitAncestryDriver) Gather(ctx context.Context, t Target) Evidence {
	ev := Evidence{Kind: WitnessGitAncestry}
	ref := strings.TrimSpace(t.Ref)
	if ref == "" {
		ev.Detail = "no git ref to witness (Target.Ref is empty)"
		return ev
	}
	_, code, err := d.run(ctx, resolveDir(t.Dir, d.dir), "cat-file", "-e", ref+"^{commit}")
	if out, done := gitStepOutcome(ev, code, err,
		fmt.Sprintf("ref %s does not resolve to a commit", ref),
		func(code int) string { return fmt.Sprintf("git cat-file exited %d for ref %s", code, ref) }); done {
		return out
	}
	full := ref
	if out, rpCode, rpErr := d.run(ctx, resolveDir(t.Dir, d.dir), "rev-parse", "--verify", ref+"^{commit}"); rpErr == nil && rpCode == 0 {
		if resolved := strings.TrimSpace(out); resolved != "" {
			full = resolved
		}
	}
	_, code, err = d.run(ctx, resolveDir(t.Dir, d.dir), "merge-base", "--is-ancestor", full, "HEAD")
	if out, done := gitStepOutcome(ev, code, err,
		fmt.Sprintf("commit %s resolves but is not reachable from HEAD", shortSHA(full)),
		func(code int) string { return fmt.Sprintf("git merge-base exited %d for %s", code, shortSHA(full)) }); done {
		return out
	}
	if revertSHA, reverted := d.revertedBy(ctx, resolveDir(t.Dir, d.dir), full); reverted {
		ev.Witnessed = true
		ev.Detail = fmt.Sprintf("commit %s is reachable from HEAD but later reverted by %s", shortSHA(full), shortSHA(revertSHA))
		return ev
	}
	ev.Witnessed = true
	ev.Holds = true
	ev.Detail = fmt.Sprintf("commit %s resolves, is reachable from HEAD, and is not reverted", shortSHA(full))
	return ev
}

// gitStepOutcome triages one git probe the way every rung of the ancestry witness does:
// a run failure is INCONCLUSIVE (not witnessed), exit 1 is a witnessed NEGATIVE — the
// probe ran and answered "no" — and any other non-zero exit is inconclusive. done=true
// means the returned Evidence is final and the caller should stop; false means the rung
// passed and the caller continues. `negative` and `oddExit` carry each rung's own
// wording, which is all the two call sites differ on.
func gitStepOutcome(ev Evidence, code int, err error, negative string, oddExit func(code int) string) (Evidence, bool) {
	switch {
	case err != nil:
		ev.Detail = "git could not run: " + err.Error()
		return ev, true
	case code == 1:
		ev.Witnessed = true
		ev.Detail = negative
		return ev, true
	case code != 0:
		ev.Detail = oddExit(code)
		return ev, true
	}
	return ev, false
}

// revertedBy scans fullSHA..HEAD for a `git revert` trailer naming fullSHA —
// reverify.go's revertedBy verbatim, including its posture that an unreadable
// scan reads as not-reverted (the reachability rung above already witnessed).
func (d *GitAncestryDriver) revertedBy(ctx context.Context, dir, fullSHA string) (string, bool) {
	out, code, err := d.run(ctx, dir, "log", "--format=%H%x00%B%x00", fullSHA+"..HEAD")
	if err != nil || code != 0 {
		return "", false
	}
	parts := strings.Split(out, "\x00")
	needle := "this reverts commit " + strings.ToLower(fullSHA)
	for i := 0; i+1 < len(parts); i += 2 {
		if strings.Contains(strings.ToLower(parts[i+1]), needle) {
			return strings.TrimSpace(parts[i]), true
		}
	}
	return "", false
}

// ---------------------------------------------------------------------------
// worktree-grep
// ---------------------------------------------------------------------------

// WorktreeGrepDriver witnesses WitnessWorktreeGrep targets: does the literal
// token Target.Pattern appear in the tracked checkout? The probe is
// reverify.go's verifyFlag verbatim: `git grep -F -- <pattern> -- .` (exit 0
// present, 1 absent, anything else cannot witness). This is a PLAIN literal
// grep — there is no comment-aware grep primitive in this tree, so a token that
// survives only inside a comment still counts as present; a driver that needs
// comment-awareness must build it, and this one does not claim to.
type WorktreeGrepDriver struct {
	run Runner
	dir string
}

// NewWorktreeGrepDriver is the real-git driver registered for worktree-grep.
func NewWorktreeGrepDriver() *WorktreeGrepDriver {
	return &WorktreeGrepDriver{run: defaultGitRunner}
}

// NewWorktreeGrepDriverWithRunner injects a Runner + dir.
func NewWorktreeGrepDriverWithRunner(r Runner, dir string) *WorktreeGrepDriver {
	return &WorktreeGrepDriver{run: r, dir: dir}
}

func (d *WorktreeGrepDriver) Kind() WitnessKind { return WitnessWorktreeGrep }

func (d *WorktreeGrepDriver) Gather(ctx context.Context, t Target) Evidence {
	ev := Evidence{Kind: WitnessWorktreeGrep}
	if strings.TrimSpace(t.Pattern) == "" {
		ev.Detail = "no token to witness (Target.Pattern is empty)"
		return ev
	}
	_, code, err := d.run(ctx, resolveDir(t.Dir, d.dir), "grep", "-F", "--", t.Pattern, "--", ".")
	switch {
	case err != nil:
		ev.Detail = "git could not run: " + err.Error()
	case code == 0:
		ev.Witnessed = true
		ev.Holds = true
		ev.Detail = fmt.Sprintf("token %q appears in the tracked checkout (plain literal git grep -F, not comment-aware)", t.Pattern)
	case code == 1:
		ev.Witnessed = true
		ev.Detail = fmt.Sprintf("token %q is absent from the tracked checkout", t.Pattern)
	default:
		ev.Detail = fmt.Sprintf("git grep exited %d for token %q", code, t.Pattern)
	}
	return ev
}

// ---------------------------------------------------------------------------
// command-probe
// ---------------------------------------------------------------------------

// CommandProbeDriver witnesses WitnessCommandProbe targets: a live command's
// observed exit. It runs Target.Argv through the tree-cancel-safe runner — or,
// when Target.Probe is set, the caller's in-process probe returning the same
// exit-like tri-state — and maps 0/1/other onto holds/does-not-hold/cannot-
// witness. The probe's output tail rides along in Detail as the operator trace.
type CommandProbeDriver struct {
	execRun CommandRunner
}

// NewCommandProbeDriver is the real-exec driver registered for command-probe.
func NewCommandProbeDriver() *CommandProbeDriver {
	return &CommandProbeDriver{execRun: defaultCommandRunner}
}

// NewCommandProbeDriverWithRunner injects a CommandRunner (tests, or an
// alternate probe transport).
func NewCommandProbeDriverWithRunner(r CommandRunner) *CommandProbeDriver {
	return &CommandProbeDriver{execRun: r}
}

func (d *CommandProbeDriver) Kind() WitnessKind { return WitnessCommandProbe }

func (d *CommandProbeDriver) Gather(ctx context.Context, t Target) Evidence {
	ev := Evidence{Kind: WitnessCommandProbe}
	var detail string
	var code int
	var err error
	switch {
	case t.Probe != nil:
		detail, code, err = t.Probe(ctx)
	case len(t.Argv) > 0:
		var out string
		out, code, err = d.execRun(ctx, t.Dir, t.Argv...)
		detail = probeTail(out)
	default:
		ev.Detail = "no command to probe (Target.Argv and Target.Probe are both empty)"
		return ev
	}
	switch {
	case err != nil:
		ev.Detail = joinDetail("probe could not run: "+err.Error(), detail)
	case code == 0:
		ev.Witnessed = true
		ev.Holds = true
		ev.Detail = joinDetail("probe exited 0", detail)
	case code == 1:
		ev.Witnessed = true
		ev.Detail = joinDetail("probe exited 1", detail)
	default:
		ev.Detail = joinDetail(fmt.Sprintf("probe exited %d (neither the holds nor the does-not-hold exit)", code), detail)
	}
	return ev
}

// ---------------------------------------------------------------------------
// config-flag
// ---------------------------------------------------------------------------

// ConfigFlagDriver witnesses WitnessConfigFlag targets whose source of truth is
// path presence: does Target.Path exist on disk? The probe is reverify.go's
// verifyPath os.Stat idiom — exists is a witnessed holds, os.IsNotExist a
// witnessed does-not-hold, any other stat error cannot witness (an unreadable
// source of truth is not evidence of absence).
type ConfigFlagDriver struct {
	stat func(string) (os.FileInfo, error)
}

// NewConfigFlagDriver is the real-filesystem driver registered for config-flag.
func NewConfigFlagDriver() *ConfigFlagDriver {
	return &ConfigFlagDriver{stat: os.Stat}
}

// NewConfigFlagDriverWithStat injects the stat probe (tests).
func NewConfigFlagDriverWithStat(stat func(string) (os.FileInfo, error)) *ConfigFlagDriver {
	return &ConfigFlagDriver{stat: stat}
}

func (d *ConfigFlagDriver) Kind() WitnessKind { return WitnessConfigFlag }

func (d *ConfigFlagDriver) Gather(_ context.Context, t Target) Evidence {
	ev := Evidence{Kind: WitnessConfigFlag}
	path := strings.TrimSpace(t.Path)
	if path == "" {
		ev.Detail = "no path to witness (Target.Path is empty)"
		return ev
	}
	if t.Dir != "" && !filepath.IsAbs(path) {
		path = filepath.Join(t.Dir, path)
	}
	_, err := d.stat(path)
	switch {
	case err == nil:
		ev.Witnessed = true
		ev.Holds = true
		ev.Detail = fmt.Sprintf("path %s exists on disk", path)
	case os.IsNotExist(err):
		ev.Witnessed = true
		ev.Detail = fmt.Sprintf("path %s is missing from disk", path)
	default:
		ev.Detail = fmt.Sprintf("path %s could not be checked: %v", path, err)
	}
	return ev
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

// resolveDir picks the per-call dir override when the target names one, else
// the driver's constructed dir ("" = process default).
func resolveDir(targetDir, driverDir string) string {
	if targetDir != "" {
		return targetDir
	}
	return driverDir
}

// shortSHA trims a resolved SHA for operator-readable detail lines.
func shortSHA(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

// probeTail folds a probe's combined output into a bounded one-line trace so
// Evidence.Detail stays a sentence, not a log dump.
func probeTail(out string) string {
	s := strings.Join(strings.Fields(out), " ")
	const tailCap = 160
	if len(s) > tailCap {
		s = s[:tailCap] + "…"
	}
	return s
}

// joinDetail renders "verdict — trace" when the probe left a trace.
func joinDetail(head, trace string) string {
	if trace == "" {
		return head
	}
	return head + " — " + trace
}
