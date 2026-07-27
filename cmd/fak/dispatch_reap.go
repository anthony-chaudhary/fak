package main

// dispatch_reap.go — `fak dispatch reap`, the age reaper for the .dispatch-runs/
// per-run sidecar tree. Every dispatch writes a cluster of disposable traces
// (.log/.txt/.witness/.json/.wave) under .dispatch-runs/ and NOTHING ever collects
// them: on the reference tree this had grown to ~5000 files / 81 MB — PURE COUNT
// growth. growthgate budgets PER-FILE SIZE, so it can never collect thousands of
// small files; that count leak is exactly what this age reaper is for (retention/GC
// census #5345, epic #3873; toward #5352).
//
//	fak dispatch reap                # dry-run: list the COLD sidecars it WOULD reap
//	fak dispatch reap --apply        # actually delete them
//	fak dispatch reap --floor-hours 48 --json
//
// SAFETY (why LOW blast). Dry-run is the DEFAULT: it lists the would-reap set and
// deletes nothing unless --apply (or FAK_DISPATCH_REAP=apply). It only ever touches
// DISPOSABLE classes (ClassifyPath -> a ClassDispatchLog-family trace whose
// Class.Disposable() is true — never a WAL or a hash-chained ledger), only COLD
// sidecars older than a generous grace floor (default 24h, far above any live run's
// turn cadence), and it always KEEPS the newest run's sidecars, so a .witness a live
// progress scan is still reading is never taken. A floor <= 0 falls back to the 24h
// default so a mis-wired zero can never sweep a live file.

import (
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/growthgate"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// DefaultDispatchSidecarFloor is the grace age below which a .dispatch-runs/ sidecar
// is NEVER reaped. It sits far above any live dispatch cadence (a run writes its
// sidecars in seconds and a progress scan reads the newest within minutes), so only
// a sidecar untouched for a full day — necessarily from a run long finished — is
// eligible. Mirrors stepbatoncapture.DefaultStaleFloor.
const DefaultDispatchSidecarFloor = 24 * time.Hour

// dispatchRunSpread is how far back from the freshest sidecar's mtime a file still
// counts as part of "the newest run". One dispatch writes its whole sidecar cluster
// within a short burst, so anything within this window of the newest file is KEPT
// even when the tree is otherwise cold — belt-and-suspenders over the grace floor so
// the newest run's .witness is never taken.
const dispatchRunSpread = 30 * time.Minute

// dispatchReapApplyEnv, when set to "apply", opts an unattended caller into deletion
// without the --apply flag (mirrors FAK_GARDEN_GROWTH_COLLECT=apply).
const dispatchReapApplyEnv = "FAK_DISPATCH_REAP"

// coldSidecar is one reap-eligible dispatch sidecar: a disposable trace older than
// the grace floor and not part of the newest run.
type coldSidecar struct {
	Path   string
	Size   int64
	AgeSec float64
}

// surveyColdDispatchSidecars walks dir and returns the COLD, disposable sidecars a
// reaper may take, plus how many files it kept. It is PURE over (dir, floor, now):
// it deletes nothing. A sidecar is reap-eligible only when it is (a) of a disposable
// class (ClassDispatchLog and its trace siblings — never a WAL/chained ledger a live
// scan reads back), (b) older than floor, and (c) not part of the newest run (within
// dispatchRunSpread of the freshest sidecar). floor <= 0 falls back to
// DefaultDispatchSidecarFloor so a mis-wired zero can never sweep a live run's
// current sidecar. A missing dir is not an error (nothing to reap).
func surveyColdDispatchSidecars(dir string, floor time.Duration, now time.Time) (reap []coldSidecar, kept int, err error) {
	if floor <= 0 {
		floor = DefaultDispatchSidecarFloor
	}
	type entry struct {
		path string
		size int64
		mod  time.Time
	}
	var all []entry
	var newest time.Time
	walkErr := filepath.WalkDir(dir, func(path string, d fs.DirEntry, werr error) error {
		if werr != nil {
			return nil // skip unreadable entries, keep walking
		}
		if d.IsDir() {
			return nil
		}
		if !growthgate.ClassifyPath(path).Disposable() {
			return nil // never a WAL/chained ledger a live scan reads back
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil // races a peer write — a later pass retries
		}
		mod := info.ModTime()
		all = append(all, entry{path: path, size: info.Size(), mod: mod})
		if mod.After(newest) {
			newest = mod
		}
		return nil
	})
	if walkErr != nil {
		if os.IsNotExist(walkErr) {
			return nil, 0, nil // no dir yet — nothing to reap
		}
		return nil, 0, walkErr
	}
	graceCutoff := now.Add(-floor)
	runCutoff := newest.Add(-dispatchRunSpread)
	for _, e := range all {
		if e.mod.After(graceCutoff) { // inside the grace window — recent, keep
			kept++
			continue
		}
		if !e.mod.Before(runCutoff) { // part of the newest run — keep
			kept++
			continue
		}
		reap = append(reap, coldSidecar{Path: e.path, Size: e.size, AgeSec: now.Sub(e.mod).Seconds()})
	}
	return reap, kept, nil
}

// runDispatchReap is the testable core of `fak dispatch reap`: it surveys the cold
// dispatch sidecars, lists them (dry-run) or deletes them (--apply), optionally
// ledgers every decision, and returns the process exit code (0 ok, 1 a runtime
// error, 2 a usage error). Best-effort deletion: one un-removable file (a Windows
// sharing lock, a permission error) is recorded and the sweep continues.
func runDispatchReap(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch reap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	apply := fs.Bool("apply", false, "actually delete the cold sidecars (default: dry-run — list only, delete nothing; also honored: FAK_DISPATCH_REAP=apply)")
	floorHours := fs.Float64("floor-hours", 24, "grace age in hours below which a sidecar is never reaped (<=0 falls back to the 24h default)")
	dir := fs.String("dir", ".dispatch-runs", "the dispatch-runs directory to reap")
	ledgerPath := fs.String("ledger", "", "append one JSON decision line per sidecar to this path")
	topN := fs.Int("top", 20, "how many would-reap sidecars to list (JSON always emits all)")
	asJSON := fs.Bool("json", false, "emit the reap survey as one JSON record")
	if !parseFlags(fs, argv) {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir) // a leading ~ is never expanded by Go; do it so --dir ~/repo/.dispatch-runs works

	applyDelete := *apply || strings.EqualFold(strings.TrimSpace(os.Getenv(dispatchReapApplyEnv)), "apply")
	floor := time.Duration(*floorHours * float64(time.Hour))
	reap, kept, err := surveyColdDispatchSidecars(*dir, floor, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch reap: survey %s: %v\n", *dir, err)
		return 1
	}

	var decisions []growthReapDecision
	var reclaimed, wouldReclaim int64
	var fails []error
	for _, s := range reap {
		wouldReclaim += s.Size
		d := growthReapDecision{Path: s.Path, Class: string(growthgate.ClassDispatchLog), Size: s.Size, Action: "would-reap"}
		if applyDelete {
			if rerr := os.Remove(s.Path); rerr != nil {
				d.Action, d.Err = "reap-failed", rerr.Error()
				fails = append(fails, rerr)
			} else {
				d.Action = "reaped"
				reclaimed += s.Size
			}
		}
		decisions = append(decisions, d)
	}

	if *ledgerPath != "" {
		if lerr := appendGrowthgateReapLedger(*ledgerPath, decisions); lerr != nil {
			fmt.Fprintf(stderr, "fak dispatch reap: ledger %s: %v\n", *ledgerPath, lerr)
		}
	}

	if *asJSON {
		payload := map[string]any{
			"schema":              "fak.dispatch.reap.v1",
			"ts":                  time.Now().UTC().Format(time.RFC3339Nano),
			"dir":                 *dir,
			"applied":             applyDelete,
			"floor_hours":         floor.Hours(),
			"reapable":            len(reap),
			"kept":                kept,
			"would_reclaim_bytes": wouldReclaim,
			"reclaimed_bytes":     reclaimed,
			"decisions":           decisions,
		}
		return encodeJSONOrFail(stdout, stderr, payload, "fak dispatch reap")
	}

	mode := "DRY-RUN (re-run with --apply to delete)"
	if applyDelete {
		mode = "APPLIED"
	}
	fmt.Fprintf(stdout, "dispatch reap: %s  (dir=%s, kept %d sidecar(s) — newest run + inside the %s grace)\n", mode, *dir, kept, floor)
	if len(reap) == 0 {
		fmt.Fprintf(stdout, "nothing reapable (no COLD dispatch sidecars past the %s floor).\n", floor)
		return 0
	}
	if applyDelete {
		fmt.Fprintf(stdout, "reaped %d sidecar(s), reclaimed %s:\n", len(reap)-len(fails), humanBytes(reclaimed))
	} else {
		fmt.Fprintf(stdout, "would reap %d sidecar(s) (%s), 0 deleted (dry-run):\n", len(reap), humanBytes(wouldReclaim))
	}
	shown := decisions
	if *topN > 0 && len(shown) > *topN {
		shown = shown[:*topN]
	}
	for _, d := range shown {
		tag := d.Action
		if d.Err != "" {
			tag += ": " + d.Err
		}
		fmt.Fprintf(stdout, "  %-12s %9s  %s\n", tag, humanBytes(d.Size), d.Path)
	}
	if len(decisions) > len(shown) {
		fmt.Fprintf(stdout, "  ... and %d more (use --top or --json)\n", len(decisions)-len(shown))
	}
	if len(fails) > 0 {
		fmt.Fprintf(stderr, "fak dispatch reap: %d sidecar(s) could not be removed\n", len(fails))
		return 1
	}
	return 0
}
