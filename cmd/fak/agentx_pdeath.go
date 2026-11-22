// Adapted from ai-dynamo/aiperf, src/aiperf/common/bootstrap.py:231
// (_install_parent_death_signal) at ab610add4e8876fca223197596c728e17365f381.
// Apache-2.0; re-expressed as a pure ppid verdict plus a per-OS arm step, so a
// supervised `fak agentx` dies with its supervisor instead of orphaning.

package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
)

// agentxParentPIDEnv names the supervisor PID a launcher hands `fak agentx`. The
// PID captured at launch is the ground truth for "who is our parent": a getppid
// snapshot taken after the supervisor already died would name the subreaper.
const agentxParentPIDEnv = "FAK_AGENTX_PARENT_PID"

type parentDeathAction int

const (
	parentDeathSkip     parentDeathAction = iota // no supervisor declared; guard off
	parentDeathContinue                          // supervisor still our parent
	parentDeathExit                              // reparented before the guard armed
)

func (a parentDeathAction) String() string {
	switch a {
	case parentDeathContinue:
		return "continue"
	case parentDeathExit:
		return "exit"
	default:
		return "skip"
	}
}

// parentDeathVerdict closes the arm race: the kernel death signal only fires
// for a parent death after it is armed, so a child whose live ppid no longer
// matches the expected supervisor has already been orphaned and must exit.
func parentDeathVerdict(expectedPPID, observedPPID int) parentDeathAction {
	if expectedPPID <= 0 {
		return parentDeathSkip
	}
	if observedPPID != expectedPPID {
		return parentDeathExit
	}
	return parentDeathContinue
}

func agentxDefaultParentPID() int {
	pid, err := strconv.Atoi(os.Getenv(agentxParentPIDEnv))
	if err != nil || pid < 0 {
		return 0
	}
	return pid
}

// installAgentXParentDeathGuard arms the OS death signal (best-effort: a failed
// arm keeps the run going) and then applies the verdict against the live ppid.
// It reports false when the caller must exit because the supervisor is gone.
func installAgentXParentDeathGuard(stderr io.Writer, expectedPPID int) bool {
	if expectedPPID <= 0 {
		return true
	}
	if err := armParentDeathSignal(); err != nil {
		fmt.Fprintf(stderr, "fak agentx: parent-death guard not armed: %v\n", err)
	}
	if parentDeathVerdict(expectedPPID, os.Getppid()) == parentDeathExit {
		fmt.Fprintf(stderr, "fak agentx: supervisor pid %d is no longer the parent (ppid %d); exiting rather than orphan\n", expectedPPID, os.Getppid())
		return false
	}
	return true
}
