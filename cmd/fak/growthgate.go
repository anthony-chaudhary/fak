package main

// growthgate.go — `fak growthgate`, the durable answer to "where did the disk go
// and what's driving the I/O thrash." It is the growth twin of `fak stallscan`:
// stallscan classifies the live CHURN of a stall; growthgate classifies the
// standing BLOAT of unbounded append-only ledgers and run logs — the
// `.dos/metrics/observations.jsonl` the dos hook fsyncs one line to per call, the
// `.dos/lane-journal.jsonl` WAL, `.dispatch-runs/*.log`, `.fak/loops.jsonl`,
// `fleet-runs/**` — none of which anything rotates.
//
//	fak growthgate                 # census the current workspace (human)
//	fak growthgate --json          # that census as one JSON record
//	fak growthgate C:/work/fak C:/work/dos C:/work/fak-private   # audit several repos
//	fak growthgate --top 30        # show more offenders
//
// WHY IT EXISTS. On this box these ledgers had grown to ~1.1 GB in the working
// tree (a single 119 MB observations.jsonl) with no cap. Unbounded growth is an
// I/O-thrash source in its own right — an ever-larger fsync'd appendee, a WAL
// folded on every read — and it is invisible to a usage dashboard until the disk
// is gone. This verb makes the audit ONE command and one repeatable verdict:
// internal/growthgate classifies the census against fixed per-class byte budgets
// and names, per offender, WHERE the missing cap belongs.
//
// COST DISCIPLINE. The walk only Lstat()s files whose name ends in a
// growth-prone suffix (.jsonl/.log/.err) and prunes the heavy non-target dirs
// (.git, node_modules, vendor, .venv), so it never becomes the load it measures.
// It reads no file contents.

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/growthgate"
)

func cmdGrowthgate(argv []string) { os.Exit(runGrowthgate(os.Stdout, os.Stderr, argv)) }

// runGrowthgate is the testable core. Exit 0 ok/watch, 1 runtime error, 2 usage,
// 3 an ACTION-level offender was found (so a script/CI gate can fail on it).
func runGrowthgate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("growthgate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the census + verdict as one JSON record")
	topN := fs.Int("top", 20, "how many offenders to list (JSON always emits all)")
	reap := fs.Bool("reap", false, "reap COLD, over-budget, disposable logs/telemetry (dry-run unless --apply)")
	apply := fs.Bool("apply", false, "with --reap, actually delete the files (default is a dry-run listing)")
	ledger := fs.String("ledger", "", "with --reap, append one JSON decision line per file to this path")
	if !parseFlags(fs, argv) {
		return 2
	}
	roots := fs.Args()
	if len(roots) == 0 {
		roots = []string{"."}
	}

	var arts []growthgate.Artifact
	now := time.Now()
	for _, root := range roots {
		info, err := os.Stat(root)
		if err != nil || !info.IsDir() {
			fmt.Fprintf(stderr, "fak growthgate: skip %s (not a directory: %v)\n", root, err)
			continue
		}
		gathered, gerr := gatherGrowthArtifacts(root, now)
		if gerr != nil {
			fmt.Fprintf(stderr, "fak growthgate: walk %s: %v\n", root, gerr)
		}
		arts = append(arts, gathered...)
	}

	rep := growthgate.Classify(arts, growthgate.DefaultBudget())

	if *reap {
		return runGrowthgateReap(stdout, stderr, rep, *apply, *ledger, *asJSON)
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, growthReport(roots, rep), "fak growthgate")
	}
	renderGrowthReport(stdout, roots, rep, *topN)
	if rep.Verdict == growthgate.SevAction {
		return 3
	}
	return 0
}

// growthReport is the JSON record: a garden-style top-level envelope (so
// gardenbundle.Interpret can read ok/verdict/reason directly) plus the full
// classification under "report". The envelope mirrors windowgate's payload so the
// two ratchets fold uniformly.
func growthReport(roots []string, rep growthgate.Report) map[string]any {
	ok := rep.Verdict != growthgate.SevAction
	verdict, finding, next := "OK", "unbounded_growth_clear", "keep append-only ledgers bounded; reap COLD logs with `fak growthgate --reap --apply`"
	action := 0
	for _, f := range rep.Findings {
		if f.Severity == growthgate.SevAction {
			action++
		}
	}
	reason := fmt.Sprintf("%d append-only file(s), %s total; %d over the ACTION budget", rep.Scanned, humanBytes(rep.TotalBytes), action)
	if !ok {
		verdict, finding = "ACTION", "unbounded_growth_action"
		next = "cap the write sites (rotate/compact) and reap COLD disposable logs with `fak growthgate --reap --apply`"
	}
	return map[string]any{
		"schema":      "fak.growthgate.v1",
		"ts":          time.Now().UTC().Format(time.RFC3339Nano),
		"ok":          ok,
		"verdict":     verdict,
		"finding":     finding,
		"reason":      reason,
		"next_action": next,
		"roots":       roots,
		"report":      rep,
	}
}

// runGrowthgateReap lists (dry-run) or deletes (--apply) the reapable set: COLD,
// over-budget files of a disposable class (pure logs + advisory telemetry — never
// a WAL or chained ledger; those are reported as protected and left in place). It
// is best-effort: one un-removable file (a Windows sharing lock, a permission
// error) is recorded and the sweep continues. Returns 0 always (a clean census
// with nothing to reap is success, not a gate failure — that is what --json is for).
func runGrowthgateReap(stdout, stderr io.Writer, rep growthgate.Report, apply bool, ledgerPath string, asJSON bool) int {
	reap, protected := growthgate.ReapPlan(rep)

	var decisions []growthReapDecision
	var reclaimed int64
	var fails []error

	for _, f := range reap {
		d := growthReapDecision{Path: f.Path, Class: string(f.Class), Size: f.Size, Action: "would-reap"}
		if apply {
			if err := os.Remove(f.Path); err != nil {
				d.Action, d.Err = "reap-failed", err.Error()
				fails = append(fails, err)
			} else {
				d.Action = "reaped"
				reclaimed += f.Size
			}
		}
		decisions = append(decisions, d)
	}

	if ledgerPath != "" {
		if err := appendGrowthgateReapLedger(ledgerPath, decisions); err != nil {
			fmt.Fprintf(stderr, "fak growthgate: ledger %s: %v\n", ledgerPath, err)
		}
	}

	if asJSON {
		payload := map[string]any{
			"schema":          "fak.growthgate.reap.v1",
			"ts":              time.Now().UTC().Format(time.RFC3339Nano),
			"applied":         apply,
			"reapable":        len(reap),
			"reclaimed_bytes": reclaimed,
			"protected":       len(protected),
			"decisions":       decisions,
		}
		return encodeJSONOrFail(stdout, stderr, payload, "fak growthgate")
	}

	mode := "DRY-RUN (re-run with --apply to delete)"
	if apply {
		mode = "APPLIED"
	}
	fmt.Fprintf(stdout, "growthgate reap: %s\n", mode)
	if len(reap) == 0 {
		fmt.Fprintf(stdout, "nothing reapable (no COLD, over-budget, disposable logs).\n")
	} else {
		var total int64
		for _, f := range reap {
			total += f.Size
		}
		if apply {
			fmt.Fprintf(stdout, "%d file(s), reclaimed %s:\n", len(reap), humanBytes(reclaimed))
		} else {
			fmt.Fprintf(stdout, "%d file(s), would reclaim %s:\n", len(reap), humanBytes(total))
		}
		for _, d := range decisions {
			tag := d.Action
			if d.Err != "" {
				tag += ": " + d.Err
			}
			fmt.Fprintf(stdout, "  %-12s %9s  %s\n", tag, humanBytes(d.Size), d.Path)
		}
	}
	if len(protected) > 0 {
		fmt.Fprintf(stdout, "\nprotected (over budget but NOT reaped — HOT or a WAL/chained ledger): %d\n", len(protected))
		for _, f := range protected {
			why := "HOT (still being written)"
			if !f.Class.Disposable() {
				why = "non-disposable " + string(f.Class) + " — bound at the write site"
			}
			fmt.Fprintf(stdout, "  %9s  %s  [%s]\n", humanBytes(f.Size), f.Path, why)
		}
	}
	if len(fails) > 0 {
		fmt.Fprintf(stderr, "fak growthgate: %d file(s) could not be removed\n", len(fails))
		return 1
	}
	return 0
}

// growthReapDecision is one reap outcome: what a dry-run WOULD delete, or what an
// --apply run reaped / failed to reap. The JSON-tagged accountability record.
type growthReapDecision struct {
	Path   string `json:"path"`
	Class  string `json:"class"`
	Size   int64  `json:"size_bytes"`
	Action string `json:"action"` // would-reap | reaped | reap-failed
	Err    string `json:"error,omitempty"`
}

// appendGrowthgateReapLedger appends one JSON line per decision — the accountability
// trail for a reap (both dry-run and apply), mirroring the fleet-janitor ledger.
func appendGrowthgateReapLedger(path string, decisions []growthReapDecision) error {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	var errs []error
	ts := time.Now().UTC().Format(time.RFC3339Nano)
	for _, d := range decisions {
		row := map[string]any{"schema": "fak.growthgate.reap-decision.v1", "ts": ts,
			"path": d.Path, "class": d.Class, "size_bytes": d.Size, "action": d.Action}
		if d.Err != "" {
			row["error"] = d.Err
		}
		b, mErr := json.Marshal(row)
		if mErr != nil {
			errs = append(errs, mErr)
			continue
		}
		if _, wErr := f.Write(append(b, '\n')); wErr != nil {
			errs = append(errs, wErr)
		}
	}
	return errors.Join(errs...)
}

// growthSkipDirs are pruned wholesale — heavy and never a growth-ledger home.
var growthSkipDirs = map[string]bool{
	".git": true, "node_modules": true, "vendor": true, ".venv": true, "venv": true,
}

// gatherGrowthArtifacts walks root and returns a census of the growth-prone
// files (by suffix), each with its size and modification age. It Lstat()s only
// the candidate files (via d.Info(), already cached by WalkDir) and never opens
// them, so the scan itself is cheap.
func gatherGrowthArtifacts(root string, now time.Time) ([]growthgate.Artifact, error) {
	var out []growthgate.Artifact
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries, keep walking
		}
		if d.IsDir() {
			if growthSkipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if !isGrowthCandidate(d.Name()) {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil
		}
		out = append(out, growthgate.Artifact{
			Path:      path,
			Size:      info.Size(),
			ModAgeSec: now.Sub(info.ModTime()).Seconds(),
		})
		return nil
	})
	return out, err
}

// isGrowthCandidate is the cheap suffix pre-filter: only append-only classes.
func isGrowthCandidate(name string) bool {
	n := strings.ToLower(name)
	return strings.HasSuffix(n, ".jsonl") || strings.HasSuffix(n, ".log") || strings.HasSuffix(n, ".err")
}

// renderGrowthReport prints the human census: verdict, per-class totals, and the
// worst offenders with their remedy.
func renderGrowthReport(w io.Writer, roots []string, rep growthgate.Report, topN int) {
	fmt.Fprintf(w, "growthgate: %s\n", strings.Join(roots, " "))
	fmt.Fprintf(w, "verdict : %s   (%d files, %s total append-only)\n",
		strings.ToUpper(string(rep.Verdict)), rep.Scanned, humanBytes(rep.TotalBytes))

	if len(rep.ByClass) > 0 {
		fmt.Fprintf(w, "\nby class (bytes desc):\n")
		for _, ct := range rep.ByClass {
			flagged := ""
			if ct.Flagged > 0 {
				flagged = fmt.Sprintf("  [%d over budget]", ct.Flagged)
			}
			fmt.Fprintf(w, "  %-14s %9s  across %d file(s), largest %s%s\n",
				ct.Class, humanBytes(ct.Bytes), ct.Count, humanBytes(ct.MaxBytes), flagged)
		}
	}

	if len(rep.Findings) == 0 {
		fmt.Fprintf(w, "\nno offenders over budget.\n")
		return
	}
	shown := rep.Findings
	if topN > 0 && len(shown) > topN {
		shown = shown[:topN]
	}
	fmt.Fprintf(w, "\noffenders (worst first):\n")
	for _, f := range shown {
		heat := "COLD"
		if f.Hot {
			heat = "HOT "
		}
		fmt.Fprintf(w, "  %-6s %-4s %9s  %s\n", strings.ToUpper(string(f.Severity)), heat, humanBytes(f.Size), f.Path)
		fmt.Fprintf(w, "         fix: %s\n", f.Remedy)
	}
	if len(rep.Findings) > len(shown) {
		fmt.Fprintf(w, "  ... and %d more (use --top or --json)\n", len(rep.Findings)-len(shown))
	}
	if rep.Verdict == growthgate.SevAction {
		fmt.Fprintf(w, "\nVERDICT: unbounded growth over the ACTION budget — cap the write sites above.\n")
	}
}

// (byte rendering reuses the shared humanBytes helper in debug.go.)
