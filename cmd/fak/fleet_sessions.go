package main

// `fak fleet sessions` — the cross-account stopped-session index (the in-process port of
// tools/fleet_sessions.py, see internal/fleetmon/sessionreg):
//
//	fak fleet sessions [summary|json|registry|plan|resume] [--window H] [--max-age MIN]
//
// registry writes sessions.json / resume_plan.json / decisions.log / transitions.log into
// the host registry dir and ends with one JSON summary line; the other modes only read.
// Active account probing (--probe) spends real provider turns and stays with the Python
// tool the resume watchdog drives.

import (
	"context"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetmon/sessionreg"
)

// fleetSessionsLiveSIDs is the live-driver census the decision pass screens against.
var fleetSessionsLiveSIDs = sessionreg.ProcessLiveSIDs

// fleetSessionsOptions resolves a pass rooted at the checkout containing root.
func fleetSessionsOptions(root string, now time.Time) sessionreg.Options {
	opts := sessionreg.FromEnv(root, now)
	opts.LiveSIDs = fleetSessionsLiveSIDs
	return opts
}

func runFleetSessions(stdout, stderr io.Writer, argv []string) int {
	mode := sessionreg.ModeSummary
	for _, a := range argv {
		if !strings.HasPrefix(a, "-") {
			mode = a
			break
		}
	}
	flagValue := func(name string) (string, bool, error) {
		for i, a := range argv {
			if a == name {
				if i+1 >= len(argv) {
					return "", true, fmt.Errorf("%s needs a value", name)
				}
				return argv[i+1], true, nil
			}
		}
		return "", false, nil
	}
	if sel := fleetSessionsProbeSelector(argv); sel != "none" {
		fmt.Fprintf(stderr, "fak fleet sessions: --probe=%s runs live account probes; use tools/fleet_sessions.py for probing passes\n", sel)
		return 2
	}
	cwd, _ := os.Getwd()
	opts := fleetSessionsOptions(findRepoRoot(cwd), time.Now())
	for _, f := range []struct {
		name string
		dst  *float64
	}{{"--window", &opts.WindowH}, {"--max-age", &opts.MaxAgeMin}} {
		raw, ok, err := flagValue(f.name)
		if err != nil {
			fmt.Fprintf(stderr, "fak fleet sessions: %v\n", err)
			return 2
		}
		if !ok {
			continue
		}
		v, perr := strconv.ParseFloat(strings.TrimSpace(raw), 64)
		if perr != nil {
			fmt.Fprintf(stderr, "fak fleet sessions: %s %q is not a number\n", f.name, raw)
			return 2
		}
		*f.dst = v
	}
	if _, err := sessionreg.Run(context.Background(), opts, mode, stdout); err != nil {
		fmt.Fprintf(stderr, "fak fleet sessions: %v\n", err)
		return 1
	}
	return 0
}

// fleetSessionsProbeSelector reads --probe / --probe=<sel> / --probe <sel> the way the
// Python tool does: a bare flag means "blocked".
func fleetSessionsProbeSelector(argv []string) string {
	known := map[string]bool{"blocked": true, "stale": true, "all": true, "none": true}
	for _, a := range argv {
		if strings.HasPrefix(a, "--probe=") {
			v := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(a, "--probe=")))
			if known[v] {
				return v
			}
			return "blocked"
		}
	}
	for i, a := range argv {
		if a == "--probe" {
			if i+1 < len(argv) {
				if v := strings.ToLower(strings.TrimSpace(argv[i+1])); known[v] {
					return v
				}
			}
			return "blocked"
		}
	}
	return "none"
}

// dispatchSessionRegistry refreshes the fleet session registry for the dispatch tick,
// returning the registry summary object. It returns at timeout even when a step that cannot
// be interrupted (a slow roster walk) is still running, as the killed subprocess did; the
// abandoned pass sees its ended ctx before writing, so the previous registry stays in place.
var dispatchSessionRegistry = func(root string, timeout time.Duration) (map[string]any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	type result struct {
		sum sessionreg.RegistrySummary
		err error
	}
	done := make(chan result, 1)
	go func() {
		pass, evalErr := sessionreg.Evaluate(ctx, fleetSessionsOptions(root, time.Now()))
		if evalErr != nil {
			done <- result{err: evalErr}
			return
		}
		sum, writeErr := pass.WriteRegistry(ctx)
		done <- result{sum: sum, err: writeErr}
	}()
	select {
	case r := <-done:
		if r.err != nil {
			return nil, fmt.Errorf("fleet session registry: %w", r.err)
		}
		return r.sum.Map(), nil
	case <-ctx.Done():
		return nil, fmt.Errorf("fleet session registry: %w", ctx.Err())
	}
}
