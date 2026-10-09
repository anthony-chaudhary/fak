package main

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// launchdAgent observes one user's GUI domain through an injected command runner.
// It deliberately exposes no lifecycle operations: loading, clearing disable
// overrides, or stopping a service needs a separate opt-in and session-safety gate.
type launchdAgent struct {
	run func(context.Context, ...string) ([]byte, error)
	uid int
}

func (a launchdAgent) domain() string { return "gui/" + strconv.Itoa(a.uid) }

func (a launchdAgent) target(label string) string { return a.domain() + "/" + label }

// Loaded observes whether the exact service is loaded, not whether its process is
// healthy. An unavailable GUI domain or command is an observation failure, not
// evidence that restarting a service is safe.
func (a launchdAgent) Loaded(ctx context.Context, label string) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if a.run == nil {
		return false, errors.New("launchd observation: command runner is missing")
	}
	target := a.target(label)
	out, err := a.run(ctx, "print", target)
	if ctxErr := ctx.Err(); ctxErr != nil {
		return false, ctxErr
	}
	if err == nil {
		return true, nil
	}
	if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
		// Match only the expected service's absence diagnostic. The broader
		// upLaunchctlNotLoaded bootout classifier also accepts "could not find"
		// a domain, which must never turn a failed probe into restart permission.
		domain := " in domain for user gui: " + strconv.Itoa(a.uid)
		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			if line == "Could not find service \""+label+"\""+domain ||
				line == "Could not find service "+label+domain {
				return false, nil
			}
		}
	}
	return false, fmt.Errorf("launchctl print %s: %w: %s", target, err, strings.TrimSpace(string(out)))
}
