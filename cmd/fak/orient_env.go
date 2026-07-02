package main

// `fak orient env` — the runtime-environment preflight verb (#2079). One call
// that tells an agent landing cold what this host already knows: the test
// route that actually executes here, the shell git/gh must run under, the
// interactive invocations that hang a headless harness, and any live lease
// overlapping the paths it is about to edit. The fold itself is the pure
// preflight.PlanEnvPreflight; this file only gathers the host evidence.

import (
	"flag"
	"fmt"
	"io"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/preflight"
	"github.com/anthony-chaudhary/fak/internal/testroute"
)

func runOrientEnv(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak orient env", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: search upward for dos.toml)")
	asJSON := fs.Bool("json", false, "emit JSON")
	readLeases := fs.Bool("leases", true, "read live refs/fak/locks leases")
	var paths orientPathFlags
	fs.Var(&paths, "paths", "path or glob you intend to edit; repeat or comma-separate")
	fs.Var(&paths, "path", "alias for --paths")
	fs.Usage = func() {
		fmt.Fprint(stderr, `fak orient env -- machine-readable runtime environment constraints (#2079)

  fak orient env                               host facts: test route, git shell, hazards, live leases
  fak orient env --paths internal/foo --json   plus the leases overlapping the paths you intend to edit
`)
	}
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	for _, arg := range fs.Args() {
		_ = paths.Set(arg)
	}

	probe := preflight.EnvProbe{
		GOOS: runtime.GOOS,
		// Same host routing facts `fak test` encodes: on Windows native go
		// test binaries are OS-policy-blocked and the suite runs via WSL.
		Test: testroute.Probe{
			GOOS:              runtime.GOOS,
			NativeTestAllowed: runtime.GOOS != "windows",
			WSLPresent:        runtime.GOOS == "windows",
		},
		Paths: paths,
	}
	if *readLeases {
		rootDir := pathutil.ExpandTilde(*root)
		if rootDir == "" {
			rootDir = devindex.FindRoot(".")
		}
		leases, err := orientLiveLeases(rootDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak orient env: live leases unavailable: %v\n", err)
		}
		for _, l := range leases {
			probe.LiveLeases = append(probe.LiveLeases, preflight.LeaseObservation{
				ID:     l.ID,
				Holder: l.Holder,
				Tree:   l.Tree,
			})
		}
	}

	rep := preflight.PlanEnvPreflight(probe)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak orient env")
	}
	renderOrientEnv(stdout, rep)
	return 0
}

func renderOrientEnv(stdout io.Writer, rep preflight.EnvReport) {
	fmt.Fprintf(stdout, "verdict     %s\n", rep.Verdict)
	fmt.Fprintf(stdout, "test_route  %s  (%s)\n", rep.TestRoute.Kind, rep.TestRoute.Reason)
	if len(rep.TestRoute.CommandTemplate) > 0 {
		fmt.Fprintf(stdout, "            %s\n", strings.Join(rep.TestRoute.CommandTemplate, " "))
	}
	fmt.Fprintf(stdout, "git_shell   %s  (%s)\n", rep.GitShell.Shell, rep.GitShell.Reason)
	for _, h := range rep.InteractiveHazards {
		fmt.Fprintf(stdout, "hazard      %s: %s\n            fix: %s\n", h.Kind, h.Why, h.Fix)
	}
	if len(rep.LiveLeases) == 0 {
		if len(rep.Paths) > 0 {
			fmt.Fprintf(stdout, "leases      none overlap %s\n", strings.Join(rep.Paths, ","))
		} else {
			fmt.Fprintln(stdout, "leases      none live")
		}
		return
	}
	for _, l := range rep.LiveLeases {
		fmt.Fprintf(stdout, "lease       %s@%s [%s]\n", l.ID, l.Holder, strings.Join(l.Tree, ","))
	}
}
