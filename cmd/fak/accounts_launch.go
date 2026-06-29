package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// `fak accounts launch` — the account-switcher LAUNCHER. It resolves a seat (the active
// role by default, or --name <seat>), rehoming a dead/unservable seat forward exactly as
// `resolve` does, then starts the agent UNDER `fak guard` with that seat's
// CLAUDE_CONFIG_DIR — so the launch is, by default:
//
//   - cache/vCache ON — guard's prompt-cache-preserving compaction + the kernel vDSO are
//     on, and guard sources THAT seat's Claude Pro/Max subscription OAuth upstream (it reads
//     CLAUDE_CONFIG_DIR), so each switched account bills its own subscription;
//   - fak permissions, not Claude's prompts — the agent is launched with
//     --dangerously-skip-permissions so Claude does NOT prompt per tool, while every tool
//     call still crosses fak's capability floor first. The kernel is the permission system.
//
// Both defaults are opt-out: --guard=false launches the agent directly (still seat-switched,
// no kernel/cache hop), and --skip-permissions=false lets Claude prompt normally.
//
// This is the durable Go front door the shell shortcuts (`c` / `claude-as <name>`) call,
// replacing a hand-rolled `CLAUDE_CONFIG_DIR=… claude` line with one that defaults to the
// guarded, cache-on, kernel-adjudicated path.

// launchOpts captures the launch knobs after the seat is resolved.
type launchOpts struct {
	command         string   // agent command to start (default "claude")
	useGuard        bool     // wrap the agent in `fak guard` (kernel adjudication + vCache)
	skipPermissions bool     // pass --dangerously-skip-permissions to the agent
	passthrough     []string // extra args appended to the agent command (everything after `--`)
}

// buildLaunchArgv constructs the process argv for an account-switched launch. With useGuard
// the agent runs UNDER `fak guard` (this same binary), so the kernel adjudicates every tool
// call and the prompt-cache/compaction (vCache) layer is on; the agent itself is started with
// --dangerously-skip-permissions (when skipPermissions) so fak's capability floor — not
// Claude's own prompts — is the permission system. fakBin is the path to this binary
// (os.Executable), used only for the guard wrap. It is pure (no I/O) so the wiring is
// unit-tested without spawning anything.
func buildLaunchArgv(fakBin string, o launchOpts) []string {
	agentCmd := []string{o.command}
	if o.skipPermissions {
		agentCmd = append(agentCmd, "--dangerously-skip-permissions")
	}
	agentCmd = append(agentCmd, o.passthrough...)
	if !o.useGuard {
		return agentCmd
	}
	// `fak guard -- <agent ...>`: guard binds the in-process gateway, installs the capability
	// floor, and execs the agent with the gateway URL injected into the child only.
	argv := []string{fakBin, "guard", "--"}
	return append(argv, agentCmd...)
}

// launchParams are the resolved inputs to runAccountsLaunch.
type launchParams struct {
	name    string // seat to launch (empty => the active role / a sensible default)
	command string // agent command (default "claude")
	// rotate launches the NEXT account in the rotation instead of the active/named seat —
	// the round-robin that lets an operator hop off a walled account onto a fresh bucket.
	// after is the anchor it rotates OFF of (empty => the named seat, else the active seat).
	rotate       bool
	after        string
	useGuard     bool // default true
	skipPerms    bool // default true
	dryRun       bool // print the plan, do not exec
	passthrough  []string
	registryPath string
	homeDir      string
}

// accountsLaunchRun is the exec seam: it spawns the resolved argv with the seat's
// CLAUDE_CONFIG_DIR in the environment and returns the child's exit code. A test overrides
// it to capture the plan without spawning a real agent. Production uses execLaunchChild.
var accountsLaunchRun = execLaunchChild

// runAccountsLaunch resolves the seat, builds the (guard-wrapped, skip-permissions) launch
// argv, and execs it under that seat's CLAUDE_CONFIG_DIR. With dryRun it prints the plan and
// returns without launching.
func runAccountsLaunch(stdout, stderr io.Writer, p launchParams) int {
	reg, err := loadOrDiscover(p.registryPath, p.homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	// Serve needs disk-derived identity (a seat that can't serve falls forward), so refresh.
	reg = reg.Refresh()

	name := strings.TrimSpace(p.name)
	if p.rotate {
		// Rotate onto the next DISTINCT account bucket after the anchor (an explicit --after,
		// else the named seat, else the active seat), so a walled account is hopped off of
		// rather than re-launched. NextInRotation skips the anchor's own bucket and every
		// reserved/disabled/tombstoned/duplicate seat.
		anchor := firstNonEmpty(strings.TrimSpace(p.after), name)
		if anchor == "" {
			if picked, ok := activeLaunchSeatName(reg); ok {
				anchor = picked
			}
		}
		seat, ok := reg.NextInRotation(anchor)
		if !ok {
			plan := reg.RotationPlan()
			if len(plan.Pool) == 0 {
				fmt.Fprintln(stderr, "fak accounts launch --rotate: no eligible accounts in rotation "+
					"(every seat is reserved, disabled, tombstoned, or has no live credentials)")
			} else {
				fmt.Fprintf(stderr, "fak accounts launch --rotate: only one account bucket in rotation (%s) — "+
					"nowhere else to rotate; enroll another with `fak accounts add`\n", plan.Pool[0].Name)
			}
			return 1
		}
		if anchor != "" {
			fmt.Fprintf(stderr, "fak accounts launch: rotating off %q -> %q\n", anchor, seat.Name)
		}
		name = seat.Name
	} else if name == "" {
		picked, ok := activeLaunchSeatName(reg)
		if !ok {
			fmt.Fprintln(stderr, "fak accounts launch: no --name and no active seat to default to — "+
				"set one with `fak accounts set-default --name <seat>`, or pass --name <seat>")
			return 2
		}
		name = picked
	}

	// Rehome by default (a tombstoned/unservable seat falls forward to a live one), exactly
	// as `fak accounts resolve` does without --pin.
	home, chain, err := reg.Serve(name)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts launch: %v\n", err)
		return 1
	}
	id := accountsReportHome(stderr, home, chain)

	command := strings.TrimSpace(p.command)
	if command == "" {
		command = "claude"
	}
	fakBin, err := os.Executable()
	if err != nil || strings.TrimSpace(fakBin) == "" {
		fakBin = "fak" // fall back to PATH resolution if the binary path can't be read
	}
	argv := buildLaunchArgv(fakBin, launchOpts{
		command:         command,
		useGuard:        p.useGuard,
		skipPermissions: p.skipPerms,
		passthrough:     p.passthrough,
	})
	env := append(os.Environ(), "CLAUDE_CONFIG_DIR="+home.Dir)

	guardWord := "off (--guard=false; launching the agent directly, no kernel/cache hop)"
	if p.useGuard {
		guardWord = "on (fak guard — kernel adjudicates every tool call; prompt-cache/compaction vCache layer on)"
	}
	permWord := "claude prompts per tool (--skip-permissions=false)"
	if p.skipPerms {
		permWord = "fak floor is the permission system (--dangerously-skip-permissions passed to claude)"
	}
	fmt.Fprintf(stderr, "fak accounts launch — seat %q\n", home.Name)
	fmt.Fprintf(stderr, "  CLAUDE_CONFIG_DIR = %s\n", home.Dir)
	if id.Email != "" {
		fmt.Fprintf(stderr, "  identity          = %s\n", id.Email)
	}
	fmt.Fprintf(stderr, "  guard             = %s\n", guardWord)
	fmt.Fprintf(stderr, "  permissions       = %s\n", permWord)
	fmt.Fprintf(stderr, "  command           = %s\n", strings.Join(argv, " "))

	if p.dryRun {
		fmt.Fprintln(stderr, "  (dry-run — not launching)")
		// Also echo the launch command to stdout so it is scriptable (eval/wrappers).
		fmt.Fprintln(stdout, strings.Join(argv, " "))
		return 0
	}
	return accountsLaunchRun(stdout, stderr, argv, env)
}

// activeLaunchSeatName picks the seat a bare `fak accounts launch` (no --name) starts: the
// active-role seat if set, else a seat literally named "default" (the bare ~/.claude home a
// discovered registry surfaces), else the sole active seat when there is exactly one. ok is
// false when none of those uniquely identify a seat, so the caller can fail loud with a hint
// instead of guessing.
func activeLaunchSeatName(reg accounts.Registry) (string, bool) {
	if h, ok := reg.Role(accounts.RoleActive); ok {
		return h.Name, true
	}
	for _, h := range reg.Homes {
		if h.Active() && strings.EqualFold(h.Name, "default") {
			return h.Name, true
		}
	}
	only, n := "", 0
	for _, h := range reg.Homes {
		if h.Active() {
			only, n = h.Name, n+1
		}
	}
	if n == 1 {
		return only, true
	}
	return "", false
}

// execLaunchChild spawns argv[0] with argv[1:] under env, wiring the child to the real
// terminal (an interactive agent owns stdin/stdout/stderr), and returns its exit code.
// A non-exec error (binary not found, etc.) is surfaced and mapped to 1.
func execLaunchChild(_, stderr io.Writer, argv, env []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak accounts launch: empty command")
		return 2
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak accounts launch: %v\n", err)
		return 1
	}
	return 0
}
