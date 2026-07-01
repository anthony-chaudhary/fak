package main

// resume_resolve.go — `fak resume resolve <sid>`, the Go port of the interactive
// resume-resolver (tools/resume_resolver.py). It decides which account
// `claude --resume <sid>` should run under and re-homes the transcript onto a healthy
// account when the owner is throttled, printing the CLAUDE_CONFIG_DIR to pin to.
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
	asJSON := fs.Bool("json", false, "emit the full decision record instead of the bare pin dir")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "fak resume resolve: need exactly one <session-id>")
		return 2
	}
	sid := fs.Arg(0)

	// Build the live roster once (paths -> policy -> registry -> annotated roster), the
	// same discovery `fak accounts` headroom uses.
	cwdNow, _ := os.Getwd()
	toolsDir := filepath.Join(findRepoRoot(cwdNow), "tools")
	paths := fleetaccounts.ResolvePaths(toolsDir)
	homeDir := *home
	if homeDir == "" {
		homeDir = paths.Home
	}
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

	in := rehome.ResolveInput{
		SID:          sid,
		Home:         homeDir,
		CWD:          *cwd,
		DryRun:       *dryRun,
		ProbeOwner:   !*noProbe,
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
			}
		},
	}

	rec := rehome.Resolve(in)
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
