package main

// superloop_drive_exec.go — the EXECUTION rung of `fak superloop drive` (issue #2224,
// the named follow-on to the admit-and-surface DRIVE). With `--execute`, once a member
// is admitted under the shared region lease, the drive RUNS the member's OWN front door
// behind that held lease and lands a real witness: the member command's exit code IS the
// member's witnessed_done (exit 0) or failure. Without `--execute` the drive stays
// surface-only, byte-for-byte the historical behavior.
//
// The honest fence holds. The super loop still gets NO private spawn path: it runs the
// member's OWN declared front door (the same command an operator would type), behind the
// SAME lease the admit gate fenced — the mutation happens at the MEMBER's altitude, not
// the super loop's. A front door the drive cannot run headless — a Claude skill
// ("/slop-score", needs an agent), a container (descend a subtree), or a member with no
// command declared — is SURFACED, never faked: [superloop.FrontDoorFor] classifies it
// and the drive records only what actually ran.

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// superloopDriveExec records how `--execute` handled a driven member's own front door:
// whether execution was requested, how the front door classified, whether it actually
// ran, and — when it ran — its exit code and whether that exit landed a witnessed_done
// (exit 0). A non-runnable front door (skill / descend / none) is Requested but not Ran:
// the Note says why, and the drive never fabricates a witness it did not earn.
type superloopDriveExec struct {
	Requested bool                    `json:"requested"`
	Kind      superloop.FrontDoorKind `json:"kind"`
	Command   string                  `json:"command,omitempty"`
	Ran       bool                    `json:"ran"`
	ExitCode  int                     `json:"exit_code"`
	Witnessed bool                    `json:"witnessed"`
	// TimeoutMinutes is the wall-clock ceiling actually applied to this member's front-door
	// run, in minutes (0 = no timeout). It is the tighter of the operator's --exec-timeout
	// and the member's declared Time allocation, so the walk's per-member reservation
	// becomes a real deadline (#2224). TimeoutSource names which bound won ("allocation" =
	// the member's budget share was tighter, "operator" = the flag was, "none" = unbounded).
	TimeoutMinutes int    `json:"timeout_minutes,omitempty"`
	TimeoutSource  string `json:"timeout_source,omitempty"`
	Note           string `json:"note"`
}

// superloopFrontDoorRunner executes a runnable front-door command line behind the held
// lease and returns its exit code. It is a package var so a test forces an exit code
// without a shell or any side effect; the live implementation runs the line through the
// platform shell (see [superloopRunFrontDoorLive]).
var superloopFrontDoorRunner = superloopRunFrontDoorLive

// defaultSuperloopExecTimeout bounds a single member front-door run so a headless drive
// cannot hang forever on a member that never exits. It is generous (a dispatch/scorecard
// run can legitimately take many minutes) and operator-overridable via --exec-timeout.
const defaultSuperloopExecTimeout = 30 * time.Minute

// superloopEffectiveTimeout folds the operator's --exec-timeout ceiling with the driven
// member's declared Time allocation into the real deadline the front-door run gets, and
// names which bound won. This is the enforcement half of #2224: the walk hands each
// worklist member a divided share of the intent's declared budget (a reservation), and the
// exec rung turns the Time share into a live ceiling here. The rule is monotone — a bound
// can only ever TIGHTEN the run, never grant more time:
//
//   - operator > 0 and allocMinutes > 0 → the smaller of the two (whichever is stricter).
//   - only one positive → that one binds (an unbudgeted Time dimension, allocMinutes 0, is a
//     HOLD, not "unlimited": it leaves the operator ceiling in force; an operator 0 means
//     "no flag ceiling", so a positive allocation still binds).
//   - both 0 → unbounded (no deadline), byte-for-byte the historical behavior.
//
// The returned source is "allocation" when the member's budget share is the binding (or
// co-binding) ceiling, "operator" when only the flag binds, and "none" when unbounded.
func superloopEffectiveTimeout(operator time.Duration, allocMinutes int) (time.Duration, string) {
	alloc := time.Duration(allocMinutes) * time.Minute
	switch {
	case operator > 0 && alloc > 0:
		if alloc <= operator {
			return alloc, "allocation"
		}
		return operator, "operator"
	case alloc > 0:
		return alloc, "allocation"
	case operator > 0:
		return operator, "operator"
	default:
		return 0, "none"
	}
}

// superloopTimeoutNote renders the human tail for the "running ..." line so the operator
// sees the ceiling actually applied and where it came from — an allocation-bound run reads
// differently from an operator-flag one, and an unbounded run says so plainly.
func superloopTimeoutNote(effTimeout time.Duration, src string) string {
	if effTimeout <= 0 {
		return " (no timeout)"
	}
	if src == "allocation" {
		return fmt.Sprintf(" (deadline %s, bound by the member's time allocation)", effTimeout)
	}
	return fmt.Sprintf(" (deadline %s, --exec-timeout)", effTimeout)
}

// superloopExecuteMember runs one admitted member's front door under `--execute`, behind
// the already-held lease, and returns the execution record. It classifies the front door
// first: a non-runnable one (skill / container / none) is surfaced honestly and NOT run;
// a runnable one is executed through [superloopFrontDoorRunner], with a StatusRunning
// witness before and a StatusWitnessedDone (exit 0) / StatusFailed (non-zero or launch
// error) witness after — the member's own witness landed on the loop ledger, keyed on the
// same lease the admit gate fenced. Output streams to `out` so the operator sees the
// member's own logs while the drive's JSON (if any) stays on stdout.
func superloopExecuteMember(out io.Writer, ledger, intent string, d superloop.DriveDecision, lease string, timeout time.Duration) *superloopDriveExec {
	fd := superloop.FrontDoorFor(d)
	ex := &superloopDriveExec{Requested: true, Kind: fd.Kind, Command: fd.Command, Note: fd.Note}
	if !fd.Runnable() {
		// Honest non-execution: the front door needs an agent (skill), is a subtree to
		// descend (container), or declares no command. Surface it; record nothing as run.
		fmt.Fprintf(out, "fak superloop drive: --execute: %s %s front door is %s — surfaced, not run (%s)\n",
			d.Member.Kind, d.Member.Ref, fd.Kind, fd.Note)
		return ex
	}

	// BIND — the member's declared Time allocation becomes a real ceiling on this run: the
	// tighter of the operator's --exec-timeout and the member's budget share bounds the
	// front-door deadline (#2224). The walk reserved the share; the exec rung enforces it,
	// so a driven member can never outrun the time the intent divided down to it.
	effTimeout, src := superloopEffectiveTimeout(timeout, d.Allocation.MaxMinutes)
	ex.TimeoutMinutes = int(effTimeout / time.Minute)
	ex.TimeoutSource = src

	recordSuperloopDriveEvent(ledger, intent, d, loopmgr.EventStart, loopmgr.StatusRunning, "EXECUTING",
		"executing "+string(d.Member.Kind)+" "+d.Member.Ref+": "+fd.Command, lease)
	fmt.Fprintf(out, "fak superloop drive: --execute: running `%s` behind lease %s%s\n", fd.Command, lease, superloopTimeoutNote(effTimeout, src))

	code, err := superloopFrontDoorRunner(fd.Command, effTimeout, out)
	ex.Ran = true
	if err != nil {
		ex.ExitCode = -1
		ex.Note = "front door could not be launched: " + err.Error()
		recordSuperloopDriveEvent(ledger, intent, d, loopmgr.EventEnd, loopmgr.StatusFailed, "EXEC_ERROR", ex.Note, lease)
		fmt.Fprintf(out, "fak superloop drive: --execute: %s\n", ex.Note)
		return ex
	}
	ex.ExitCode = code
	if code == 0 {
		ex.Witnessed = true
		ex.Note = "front door `" + fd.Command + "` exited 0 — the member's witnessed_done"
		recordSuperloopDriveEvent(ledger, intent, d, loopmgr.EventEnd, loopmgr.StatusWitnessedDone, "WITNESSED_DONE", ex.Note, lease)
		return ex
	}
	ex.Note = fmt.Sprintf("front door `%s` exited %d — member run failed", fd.Command, code)
	recordSuperloopDriveEvent(ledger, intent, d, loopmgr.EventEnd, loopmgr.StatusFailed, "EXEC_FAILED", ex.Note, lease)
	fmt.Fprintf(out, "fak superloop drive: --execute: %s\n", ex.Note)
	return ex
}

// superloopRunFrontDoorLive runs a (possibly compound, e.g. "a && b") front-door command
// line through the platform shell so shell operators work, bounding it with timeout when
// timeout > 0. It returns the command's exit code and a nil error for any clean run OR
// clean non-zero exit — a non-zero exit is a member FAILURE the caller records, not an
// exec-layer error. A non-nil error means the command could not even be launched (missing
// shell) or the context deadline fired. Member output streams to `out`.
func superloopRunFrontDoorLive(command string, timeout time.Duration, out io.Writer) (int, error) {
	name, pre := superloopShell()
	c := ctx()
	if timeout > 0 {
		var cancel context.CancelFunc
		c, cancel = context.WithTimeout(c, timeout)
		defer cancel()
	}
	cmd := exec.CommandContext(c, name, append(pre, command)...)
	cmd.Stdout = out
	cmd.Stderr = out
	err := cmd.Run()
	if err == nil {
		return 0, nil
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		// A clean non-zero exit: the member ran and failed. Report the code, not an error.
		return ee.ExitCode(), nil
	}
	// Could not launch (or the deadline fired before the process reported an exit).
	if derr := c.Err(); derr != nil {
		return -1, fmt.Errorf("front door timed out after %s: %w", timeout, derr)
	}
	return -1, err
}

// superloopShell picks the platform shell used to run a compound member front-door line.
// cmd /c on Windows, sh -c elsewhere — the same split the rest of the fleet's shell-outs
// assume, so a member's declared "a && b" runs as one shell command line.
func superloopShell() (string, []string) {
	if runtime.GOOS == "windows" {
		return "cmd", []string{"/c"}
	}
	return "sh", []string{"-c"}
}
