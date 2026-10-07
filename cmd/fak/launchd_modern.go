package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// launchdAgent drives a per-user LaunchAgent through the modern launchctl
// domain verbs (bootout / enable / bootstrap / print). The legacy
// `launchctl load -w` / `unload` pair is deliberately avoided: on macOS 11+
// it routinely prints "Load failed: 5: Input/output error" yet exits 0, so a
// plist sits in ~/Library/LaunchAgents and never runs while the installer
// reports success. bootstrap returns a real non-zero exit on failure, and
// every failure here is surfaced with launchctl's combined output.
//
// The struct carries no build tag and takes an injectable runner so the exact
// argv sequence is testable on Linux.
type launchdAgent struct {
	// run executes launchctl with args and returns its combined output.
	run func(ctx context.Context, args ...string) ([]byte, error)
	// uid is the GUI domain owner (os.Getuid on a real host).
	uid int
	// sleep waits between a bootout and a retried bootstrap.
	sleep func(time.Duration)
}

// newLaunchdAgent returns a launchdAgent bound to the real launchctl binary
// and the current user's gui/<uid> domain.
func newLaunchdAgent() launchdAgent {
	return launchdAgent{
		run: func(ctx context.Context, args ...string) ([]byte, error) {
			return exec.CommandContext(ctx, "launchctl", args...).CombinedOutput()
		},
		uid:   os.Getuid(),
		sleep: time.Sleep,
	}
}

func (a launchdAgent) domain() string { return "gui/" + strconv.Itoa(a.uid) }

func (a launchdAgent) target(label string) string { return a.domain() + "/" + label }

func launchdErr(verb string, args []string, err error, out []byte) error {
	return fmt.Errorf("launchctl %s %s: %w: %s", verb, strings.Join(args, " "), err, strings.TrimSpace(string(out)))
}

// Bootout removes label from the user's GUI domain. A not-loaded service is
// not an error (the desired end state already holds).
func (a launchdAgent) Bootout(ctx context.Context, label string) error {
	t := a.target(label)
	out, err := a.run(ctx, "bootout", t)
	if err != nil && !upLaunchctlNotLoaded(out) {
		return launchdErr("bootout", []string{t}, err, out)
	}
	return nil
}

// Bootstrap (re)loads the agent at plistPath under label: bootout any loaded
// copy, enable the label (clearing a prior `disable` / `unload -w` override),
// then bootstrap the plist into gui/<uid>. A bootstrap that fails right after
// a bootout is retried once, since launchd may still be tearing the old
// instance down (the classic transient "5: Input/output error"). With
// kickstart set, the job is started immediately via `kickstart -k`.
func (a launchdAgent) Bootstrap(ctx context.Context, label, plistPath string, kickstart bool) error {
	if err := a.Bootout(ctx, label); err != nil {
		return err
	}
	t := a.target(label)
	if out, err := a.run(ctx, "enable", t); err != nil {
		return launchdErr("enable", []string{t}, err, out)
	}
	var out []byte
	var err error
	for attempt := 0; attempt < 2; attempt++ {
		if attempt > 0 && a.sleep != nil {
			a.sleep(500 * time.Millisecond)
		}
		if out, err = a.run(ctx, "bootstrap", a.domain(), plistPath); err == nil {
			break
		}
	}
	if err != nil {
		return launchdErr("bootstrap", []string{a.domain(), plistPath}, err, out)
	}
	if kickstart {
		if out, err := a.run(ctx, "kickstart", "-k", t); err != nil {
			return launchdErr("kickstart", []string{"-k", t}, err, out)
		}
	}
	return nil
}

// Loaded reports whether label is loaded in the user's GUI domain, by
// `launchctl print gui/<uid>/<label>` (the modern, exit-status-honest probe).
func (a launchdAgent) Loaded(ctx context.Context, label string) (bool, error) {
	t := a.target(label)
	out, err := a.run(ctx, "print", t)
	if err == nil {
		return true, nil
	}
	if upLaunchctlNotLoaded(out) {
		return false, nil
	}
	return false, launchdErr("print", []string{t}, err, out)
}
