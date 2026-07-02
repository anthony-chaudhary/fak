package main

// resume_resolve.go — `fak resume resolve <sid>`, the Go port of the interactive
// resume-resolver (tools/resume_resolver.py). It decides which account
// `claude --resume <sid>` should run under and re-homes the transcript onto a healthy
// account when the owner is throttled, printing the CLAUDE_CONFIG_DIR to pin to.
//
// When the owner is blocked but its reset is imminent (<= 15 min), the verdict is
// WAIT_RESET — waiting for the owner beats copying the transcript onto another loaded
// seat — and `-wait` turns that verdict into behavior: sleep out the reset (narrated to
// stderr as a countdown), re-resolve over the refreshed roster, and print the pin dir,
// so one command self-heals the account wall:
//
//	CLAUDE_CONFIG_DIR="$(fak resume resolve -wait <sid>)" claude --resume <sid>
//
// The decision is the pure internal/resume/rehome.Resolve; this shell does only the I/O
// the leaf forbids: it builds the live roster (internal/fleetaccounts) into the
// availability + owner-status the decision consumes, uses the account-probe ledger
// (internal/fleetaccounts.FreshProbeFromLedger) as the probe source, and copies with
// rehome.RehomeTranscript. Output contract mirrors resume_resolver.main: stdout is the
// one pin dir (or the full record with --json), stderr the human diagnostic, exit 0
// resolved / 1 not found / 2 usage.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/resume/rehome"
)

// runResumeResolve resolves the pin dir for `claude --resume <sid>`, re-homing off a
// throttled owner onto a healthy account. It returns the process exit code.
func runResumeResolve(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume resolve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	home := fs.String("home", "", "user home holding the .claude* account dirs (default: discovered)")
	cwd := fs.String("cwd", "", "directory `claude --resume` will run from (default: the process cwd); its slug is added to the re-home copy so a resume works from a different folder than the session's birth dir")
	dryRun := fs.Bool("dry-run", false, "decide and report but do NOT copy the transcript (stdout still shows the intended pin dir)")
	noProbe := fs.Bool("no-probe", false, "trust the carried throttle; do NOT consult the probe ledger before re-homing")
	noWait := fs.Bool("no-wait", false, "never answer WAIT_RESET: re-home off a blocked owner immediately even when its reset is minutes away")
	wait := fs.Bool("wait", false, "self-healing mode: when the verdict is WAIT_RESET, sleep out the owner's reset (narrating to stderr), then re-resolve — so `CLAUDE_CONFIG_DIR=\"$(fak resume resolve -wait <sid>)\" claude --resume <sid>` heals the account wall on its own")
	asJSON := fs.Bool("json", false, "emit the full decision record instead of the bare pin dir")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "fak resume resolve: need exactly one <session-id>")
		return 2
	}
	sid := fs.Arg(0)

	paths, homeDir := discoverFleetPaths(*home)

	rec := rehome.Resolve(buildResolveInput(paths, homeDir, sid, *cwd, *dryRun, !*noProbe, *noWait))
	// Self-healing mode: a WAIT_RESET verdict carries a bounded, machine-checkable wait —
	// sleep it out (narrating to stderr so the operator sees a countdown, not a hang), then
	// re-resolve over a FRESH roster. Bounded: each wait is <= the horizon, and after two
	// rounds the resolver is forced to land on whatever is healthy (NoWait), so -wait can
	// never spin forever.
	for round := 0; *wait && rec.OK && rec.Action == "WAIT_RESET" && !*dryRun; round++ {
		until := time.Unix(rec.ResetUnix, 0)
		fmt.Fprintf(stderr, "[resume-resolve] %s\n", rec.Reason)
		fmt.Fprintf(stderr, "[resume-resolve] waiting %s until the owner's reset at %s (+30s slack)…\n",
			(time.Duration(rec.WaitSeconds) * time.Second).String(), until.Format("3:04pm"))
		time.Sleep(time.Until(until.Add(30 * time.Second)))
		// Re-resolve over a FRESH roster: availability derives from the reset strings, so
		// the pre-wait roster still reads blocked after the reset. Two waits without the
		// owner freeing up force a landing on whatever is healthy (NoWait).
		rec = rehome.Resolve(buildResolveInput(paths, homeDir, sid, *cwd, *dryRun, !*noProbe, *noWait || round >= 1))
	}
	if !rec.OK {
		fmt.Fprintf(stderr, "[resume-resolve] %s\n", rec.Reason)
		if *asJSON {
			_ = encodeJSONOrFail(stdout, stderr, rec, "fak resume resolve")
		}
		return 1
	}
	fmt.Fprintf(stderr, "[resume-resolve] %s: %s\n", rec.Action, rec.Reason)
	if rec.DupCount > 1 {
		fmt.Fprintf(stderr, "[resume-resolve] session in %d accounts (%s)\n", rec.DupCount, strings.Join(rec.AllAccounts, ", "))
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rec, "fak resume resolve")
	}
	fmt.Fprintln(stdout, rec.PinConfigDir)
	return 0
}

// discoverFleetPaths resolves the fleet-account discovery roots (the same discovery
// `fak accounts` headroom uses) and the home dir holding the .claude* account dirs,
// honoring an explicit --home override.
func discoverFleetPaths(homeFlag string) (fleetaccounts.Paths, string) {
	cwdNow, _ := os.Getwd()
	toolsDir := filepath.Join(findRepoRoot(cwdNow), "tools")
	paths := fleetaccounts.ResolvePaths(toolsDir)
	homeDir := homeFlag
	if homeDir == "" {
		homeDir = paths.Home
	}
	return paths, homeDir
}

// buildResolveInput reads the live roster (paths -> policy -> registry -> annotated
// roster) into the ResolveInput rehome.Resolve decides over. Shared by `resume resolve`
// (which may re-run it per -wait round — availability derives from the reset strings, so
// a roster read BEFORE the owner's reset still says blocked after it) and `resume why`
// (which dry-runs the same decision so the narrative can never disagree with the verb).
func buildResolveInput(paths fleetaccounts.Paths, homeDir, sid, cwd string, dryRun, probeOwner, noWait bool) rehome.ResolveInput {
	pol := fleetaccounts.LoadPolicy(paths)
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	roster := fleetaccounts.AnnotatedRoster(homeDir, paths.ConfigHome, pol, reg)

	statusByAccount := make(map[string]fleetaccounts.Account, len(roster))
	var avail []rehome.Target
	for _, r := range roster {
		statusByAccount[r.Account] = r
		if !fleetaccounts.RoutableWorker(r) {
			continue
		}
		avail = append(avail, rehome.Target{
			Account:        r.Account,
			Available:      r.Available != nil && *r.Available,
			LiveSessions:   rrDerefInt(r.LiveSessions),
			ActiveSessions: rrDerefInt(r.ActiveSessions),
			Tag:            r.Tag,
			ConfigDir:      r.Dir,
			VerdictSource:  rrDerefStr(r.StatusSource),
		})
	}

	return rehome.ResolveInput{
		SID:          sid,
		Home:         homeDir,
		CWD:          cwd,
		DryRun:       dryRun,
		ProbeOwner:   probeOwner,
		NoWait:       noWait,
		Availability: avail,
		OwnerStatusFn: func(account string) rehome.OwnerStatus {
			r, ok := statusByAccount[account]
			if !ok {
				return rehome.OwnerStatus{Available: true}
			}
			return rehome.OwnerStatus{
				Available:    r.Available == nil || *r.Available,
				BlockReason:  rrDerefStr(r.BlockReason),
				BlockKind:    rrDerefStr(r.BlockKind),
				StatusSource: rrDerefStr(r.StatusSource),
				ResetUnix:    resetStringUnix(rrDerefStr(r.Reset)),
			}
		},
		RehomeFn: rehome.RehomeTranscript,
		// No Go active prober exists; use the freshest RECORDED probe verdict from the
		// account-probe ledger as the probe source (nil == unprobeable -> trust the
		// ranking, matching resume_resolver's probe_fn None semantics).
		ProbeFn: func(account, _ string) *rehome.ProbeResult {
			fp := fleetaccounts.FreshProbeFromLedger(account, "", time.Now().UTC(), 0)
			if fp == nil {
				return nil
			}
			return &rehome.ProbeResult{
				Available:    fp.Available,
				BlockReason:  fp.BlockReason,
				BlockKind:    fp.BlockKind,
				StatusSource: "probe",
				ResetUnix:    resetStringUnix(fp.Reset),
			}
		},
	}
}

// resetStringUnix parses a carried "resets" string ("7:10pm (America/Los_Angeles)",
// "Dec 31, 1pm") into the unix instant it names, 0 when absent or unparseable — the
// shape rehome.Resolve's WAIT_RESET verdict compares against now.
func resetStringUnix(reset string) int64 {
	t, ok := fleetaccounts.ResetInstant(reset, time.Now())
	if !ok {
		return 0
	}
	return t.Unix()
}

func rrDerefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func rrDerefStr(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}
