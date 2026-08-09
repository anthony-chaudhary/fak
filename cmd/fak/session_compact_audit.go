package main

// session_compact_audit.go — `fak session compact-audit` (#4763): the operator front end
// to internal/session's Codex-rollout compaction miner.
//
//	fak session compact-audit                          # the local corpus, human report
//	fak session compact-audit --since 2026-06-15       # only rollouts touched since then
//	fak session compact-audit --cwd fak --json         # this repo's sessions, machine form
//	fak session compact-audit --json --scrub > a.json  # the checkinable aggregate
//	fak session compact-audit --guarded-only           # only sessions fak actually routed
//
// It is an OFFLINE verb: it reads rollout JSONL and dials no gateway. All logic lives in
// internal/session (compactaudit.go); this file is flags, defaults, and exit codes.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// defaultCodexSessionsRoot is where the Codex CLI writes rollouts.
func defaultCodexSessionsRoot() string {
	if v := os.Getenv("FAK_CODEX_SESSIONS_ROOT"); v != "" {
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".codex", "sessions")
}

// defaultGuardWitnessDir is where `fak guard` records the sessions it routed. It shares
// resolvedCodexLoopHome with the writer (sessions_codex_loop.go) so the reader can never
// drift onto a different Codex home than the one the witnesses were written under.
func defaultGuardWitnessDir() string {
	home, err := resolvedCodexLoopHome("")
	if err != nil {
		return ""
	}
	return filepath.Join(home, session.GuardWitnessDirName)
}

func runSessionCompactAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session compact-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", defaultCodexSessionsRoot(), "Codex rollout corpus root")
	since := fs.String("since", "", "only rollouts modified at/after this date (YYYY-MM-DD or RFC3339)")
	cwd := fs.String("cwd", "", "only sessions whose workspace path contains this substring")
	limit := fs.Int("limit", 0, "scan at most N rollouts (0 = all)")
	asJSON := fs.Bool("json", false, "emit the JSON document instead of the human report")
	scrub := fs.Bool("scrub", false, "--json: drop filesystem paths and cwd so the output is checkinable")
	aggregateOnly := fs.Bool("aggregate-only", false, "--json: emit only the roll-up, not per-session rows")
	top := fs.Int("top", 10, "human report: show the N highest-ranked sessions (0 = none)")
	topBy := fs.String("top-by", session.CompactRankFires, "human report ranking: fires, peak-resident, or cumulative-input")
	// No backquotes in these usage strings: flag.UnquoteUsage reads backquoted text as
	// the value placeholder, so "`fak guard`" would render as the flag's argument name.
	guardedOnly := fs.Bool("guarded-only", false, "keep only sessions present in the fak guard witness ledger — the traffic fak actually routed")
	guardDir := fs.String("guard-witness-dir", defaultGuardWitnessDir(), "--guarded-only: the guard witness ledger directory")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session compact-audit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *root == "" {
		fmt.Fprintln(stderr, "fak session compact-audit: no corpus root (pass --root or set FAK_CODEX_SESSIONS_ROOT)")
		return 2
	}
	if _, err := os.Stat(*root); err != nil {
		fmt.Fprintf(stderr, "fak session compact-audit: corpus root %s: %v\n", *root, err)
		return 1
	}

	opts := session.CompactAuditOptions{
		Root: *root, Cwd: *cwd, Limit: *limit,
		GuardedOnly: *guardedOnly, GuardWitnessDir: *guardDir,
	}
	if *since != "" {
		t, err := parseCompactAuditSince(*since)
		if err != nil {
			fmt.Fprintf(stderr, "fak session compact-audit: --since %q: %v\n", *since, err)
			return 2
		}
		opts.Since = t
	}

	res, err := session.AuditCompactCorpus(opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak session compact-audit: %v\n", err)
		return 1
	}

	if *topBy != session.CompactRankFires && *topBy != session.CompactRankPeakResident && *topBy != session.CompactRankCumulativeInput {
		fmt.Fprintf(stderr, "fak session compact-audit: --top-by %q: want fires, peak-resident, or cumulative-input\n", *topBy)
		return 2
	}

	if !*asJSON {
		if *topBy == session.CompactRankFires {
			session.RenderCompactAudit(stdout, res, *top)
		} else {
			session.RenderCompactAudit(stdout, res, 0)
			if err := session.WriteCompactTrajectoryRanking(stdout, res.Sessions, *top, *topBy); err != nil {
				fmt.Fprintf(stderr, "fak session compact-audit: %v\n", err)
				return 2
			}
		}
		return 0
	}
	if *scrub {
		res = session.ScrubCompactResult(res)
	}
	if *aggregateOnly {
		res.Sessions = nil
	}
	if err := session.WriteCompactAuditJSON(stdout, res); err != nil {
		fmt.Fprintf(stderr, "fak session compact-audit: %v\n", err)
		return 1
	}
	return 0
}

// parseCompactAuditSince accepts the date form an operator actually types, and the
// RFC3339 form a script emits.
func parseCompactAuditSince(s string) (time.Time, error) {
	if t, err := time.Parse("2006-01-02", s); err == nil {
		return t, nil
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Time{}, fmt.Errorf("want YYYY-MM-DD or RFC3339")
	}
	return t, nil
}
