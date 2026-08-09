package superloop

// frontdoor.go — the pure classifier that decides whether a drive can ACTUALLY RUN a
// member's front door headless, or must only surface it (issue #2224, the named
// follow-on to the DRIVE rung). WALK selects the worst-first member; DRIVE admits it
// under a lease; this classifier answers the last question before execution: is the
// member's own front door (its [Member].Enter) a shell command a headless drive can
// execute, or a Claude skill that needs an agent, or a container to descend?
//
// It is PURE — a fold over the already-selected [DriveDecision], reading no files, no
// clock, no shell. The impure shell (cmd/fak) executes a FrontRunnable command behind
// the held lease and surfaces the rest honestly; keeping the classification here means
// the "runnable vs. agent-only" decision is witnessed by a unit test, not buried in the
// exec path, and the drive can never fake having run a skill it cannot run headless.

import "strings"

const (
	gitDailyDispatchPrefix = "go run ./cmd/fak git-daily --root . && "
	dispatchAutoCommand    = "go run ./cmd/fak dispatch auto"
)

// FrontDoorKind classifies a member's front door by whether a headless drive can run it.
type FrontDoorKind string

const (
	// FrontRunnable is a shell command a headless drive CAN execute behind the lease
	// (e.g. "go run ./cmd/fak dispatch auto --goal throughput",
	// "python tools/tooling_quality_scorecard.py --json", or a compound "a && b" line).
	// Its exit code is the member's own witness: exit 0 lands a witnessed_done.
	FrontRunnable FrontDoorKind = "runnable"
	// FrontSkill is a Claude skill front door ("/slop-score"): it needs an AGENT, not a
	// shell, so a headless drive SURFACES it and never claims to have run it. Executing
	// it is the operator's/agent's move, not the meta-loop's.
	FrontSkill FrontDoorKind = "skill"
	// FrontDescend is a container member (a sub-super-loop or the garden bundle): running
	// it would mean cascading a whole subtree, which the drive must not do implicitly. The
	// drive surfaces the descend pointer (`fak superloop walk <ref>`) instead.
	FrontDescend FrontDoorKind = "descend"
	// FrontNone is a member with no concrete Enter command declared (e.g. a plain loop
	// whose only front door is the generic "drive the loop" pointer). There is nothing a
	// headless drive can run, so it is surfaced, not executed.
	FrontNone FrontDoorKind = "none"
)

// FrontDoor is the classification of a driven member's front door: its kind, the exact
// shell command to run when FrontRunnable (empty otherwise), and a one-line note the
// drive surfaces so a reader knows WHY a member was run or only surfaced.
type FrontDoor struct {
	Kind    FrontDoorKind `json:"kind"`
	Command string        `json:"command,omitempty"`
	Note    string        `json:"note"`
}

// Runnable reports whether the front door is a shell command a headless drive executes.
func (f FrontDoor) Runnable() bool { return f.Kind == FrontRunnable }

// withDailyGitHygiene makes the dispatch front door carry its own once-per-day
// repository maintenance. git-daily is self-deduplicating, but keeping the hook at this
// single execution seam also prevents one super-loop turn from spelling it twice.
func withDailyGitHygiene(command string) string {
	if !strings.Contains(command, dispatchAutoCommand) || strings.Contains(command, "git-daily") {
		return command
	}
	return gitDailyDispatchPrefix + command
}

// FrontDoorFor classifies the front door of one selected [DriveDecision]. The order is
// load-bearing: a CONTAINER is a descend pointer regardless of any Enter string (the
// drive never executes a subtree), so it is checked first; then a skill ("/x"), which
// needs an agent; then an empty Enter (nothing runnable); everything else is a concrete
// shell command the drive can run behind the lease. Pure and total over the decision.
func FrontDoorFor(d DriveDecision) FrontDoor {
	if d.Container {
		return FrontDoor{
			Kind: FrontDescend,
			Note: "container member (sub-super-loop / garden): descend and drive it (`fak superloop walk " + d.Member.Ref + "`), never execute a subtree here",
		}
	}
	enter := strings.TrimSpace(d.Member.Enter)
	if enter == "" {
		return FrontDoor{
			Kind: FrontNone,
			Note: "no concrete front-door command declared for " + string(d.Member.Kind) + " " + d.Member.Ref + "; surfaced for an operator/agent to enter",
		}
	}
	if strings.HasPrefix(enter, "/") {
		return FrontDoor{
			Kind:    FrontSkill,
			Command: enter,
			Note:    "a Claude skill (`" + enter + "`): needs an agent, not a headless shell — surfaced, not run",
		}
	}
	return FrontDoor{
		Kind:    FrontRunnable,
		Command: withDailyGitHygiene(enter),
		Note:    "a shell command a headless drive runs behind the lease; its exit code is the member's witness",
	}
}
