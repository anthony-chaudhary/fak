package treedoctor

// Orphaned go-build WORK-dir reaper (#6207) — the retention half of the in-repo GOTMPDIR.
//
// WHY THIS EXISTS. fak points GOCACHE/GOTMPDIR INSIDE the worker's tree on purpose
// (internal/workerworktree.WorktreeEnv): that redirection is the build-isolation contract
// and stays. What was missing is the other half — nothing ever COLLECTED it. `go` removes
// its own `go-build*` WORK dir on a clean exit, so a surviving one means that process was
// KILLED (timeout, cancellation, worker reap), which is routine for a fleet harness that
// runs builds under deadlines. Measured on the shared checkout 2026-08-10: `_scratch/go-tmp`
// held 6,616 MB over 144 entries, of which 44 orphaned `go-build*` dirs were 5,915 MB —
// 89% of the tree, untouched for 15-17 hours while 138 compilers were live.
//
// WHY THE LIVENESS KEY IS "NEWEST FILE ANYWHERE INSIDE". A go build writes into
// subdirectories of its WORK dir for its whole run, and on most filesystems that does NOT
// bump the top-level dir's mtime. A naive top-level-mtime sweep therefore reads a long,
// still-running build as stale and deletes the WORK dir out from under it, breaking a live
// build. So the age of an entry here is the age of the NEWEST file found anywhere beneath
// it. Anything the walk could not resolve (an error, or a tree so large the bounded walk
// was truncated) is INDETERMINATE and always kept: a false keep costs disk, a false reap
// breaks a peer's running build.
//
// WHY THE REPORT IS AGE-SPLIT. This tree is volatile — 11GB -> 4.4GB -> 5.9GB across two
// days — and a single-sample `du` taken mid-sweep reads as a catastrophic leak when the
// mass is really in-flight churn. An earlier pass of the audit that motivated this file
// sampled while 3,244 MB was under two hours old and wrongly concluded the tree
// self-collects. The bands make "in-flight" and "stale" separately visible in every
// report, so no caller can repeat that mistake from this output.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// DefaultGoTmpMinAge is the grace floor under which a WORK dir is never reclaimed,
// measured against the newest file anywhere inside it. Process references remain the
// authoritative liveness witness after this quiet-period screen.
const DefaultGoTmpMinAge = 20 * time.Minute

// GoTmpColdAgeSec is the band boundary between "stale" and "cold" in the age split. It is
// only a reporting cut — both bands are equally reapable — kept so a reader can tell a
// day's worth of killed builds from residue that has survived a full day.
const GoTmpColdAgeSec = 24 * 60 * 60

// DefaultGoTmpMaxWalkEntries bounds one entry's walk so a pathological tree cannot stall
// an unattended tick. Exceeding it makes the entry INDETERMINATE (kept), never reapable.
const DefaultGoTmpMaxWalkEntries = 500000

// goTmpDirEntryLimit bounds one maintenance pass before any recursive scan or
// process query. Crossing the bound fails closed instead of turning routine hygiene into
// an unbounded filesystem walk.
const goTmpDirEntryLimit = 4096

// GoTmpSchema is the stable machine-readable contract emitted by the maintenance rung.
const GoTmpSchema = "fak-go-tmp-maintenance/1"

// GoTmpDirEnv is the environment variable that names the directory this sweep collects.
const GoTmpDirEnv = "GOTMPDIR"

// DefaultGoTmpPrefixes is the closed set of top-level names the sweep will reap: only the
// WORK dirs `go` itself creates. Deliberately narrow — the same directory also collects
// `t.TempDir()` leftovers named after the test that leaked them (41 MB of the 6,616 MB
// measured), and those belong to whichever package leaked them, not to this reaper.
var DefaultGoTmpPrefixes = []string{"go-build"}

// GoTmpVerdict is the closed vocabulary for one entry's decision. Exactly one of these is
// assigned per entry, and only GoTmpReap ever removes anything.
type GoTmpVerdict string

const (
	// GoTmpReap: a go-build WORK dir whose newest file anywhere inside is older than
	// MinAge. `go` deletes its own WORK dir on a clean exit, so this is an orphan.
	GoTmpReap GoTmpVerdict = "reap"
	// GoTmpKeepLive: something inside was written within MinAge — a build may still be
	// running here.
	GoTmpKeepLive GoTmpVerdict = "live"
	// GoTmpKeepIndeterminate: the walk errored or was truncated, so liveness is unproven.
	// Fail-safe: keep.
	GoTmpKeepIndeterminate GoTmpVerdict = "indeterminate"
	// GoTmpKeepForeign: the name is not one of the swept prefixes (a leaked t.TempDir, a
	// tool's scratch dir). Not this reaper's garbage.
	GoTmpKeepForeign GoTmpVerdict = "foreign"
)

const (
	GoTmpReasonEligible                  = "eligible"
	GoTmpReasonFresh                     = "fresh"
	GoTmpReasonReferenced                = "process-referenced"
	GoTmpReasonProcessEnumerationFailed  = "process-enumeration-failed"
	GoTmpReasonPostMoveReferenced        = "post-move-referenced"
	GoTmpReasonPostMoveEnumerationFailed = "post-move-enumeration-failed"
	GoTmpReasonNestedRepository          = "nested-repository"
	GoTmpReasonReparsePoint              = "reparse-point"
	GoTmpReasonOutsideRoot               = "outside-root"
	GoTmpReasonRoot                      = "scratch-root"
	GoTmpReasonNotDirectory              = "not-directory"
	GoTmpReasonForeign                   = "foreign"
	GoTmpReasonScanFailed                = "scan-failed"
	GoTmpReasonWalkLimit                 = "walk-limit"
	GoTmpReasonReclaimed                 = "reclaimed"
	GoTmpReasonQuarantined               = "quarantined"
)

// GoTmpOptions configures the sweep. The zero value sweeps nothing (empty Root).
type GoTmpOptions struct {
	// Root is the GOTMPDIR to collect. Empty => the sweep is a no-op, which is what makes
	// it safe to wire into a caller that may not have a redirected GOTMPDIR at all.
	Root string
	// RepoRoot, when set, constrains Root to the repository's _scratch subtree. The CLI
	// always sets it; leaving it blank preserves the package API for isolated callers.
	RepoRoot string
	// MinAge is the quiet period an entry must clear before it is reapable, measured on
	// the newest file anywhere inside. Zero => DefaultGoTmpMinAge.
	MinAge time.Duration
	// Now is the reference time for every age (injectable). Zero => time.Now() at call.
	Now time.Time
	// Prefixes overrides the reapable top-level name prefixes. Nil => DefaultGoTmpPrefixes.
	Prefixes []string
	// MaxWalkEntries bounds one entry's walk. Zero => DefaultGoTmpMaxWalkEntries.
	MaxWalkEntries int
	// MaxCandidates bounds the number of immediate children considered. Zero uses the
	// default. Exceeding it fails closed before any candidate is moved.
	MaxCandidates int
	// Processes supplies one host process snapshot. Nil selects the platform witness.
	Processes func() ([]GoTmpProcess, error)
	// QuarantineRoot overrides the OS-temp quarantine parent for deterministic tests.
	QuarantineRoot string
}

// GoTmpEntry is one top-level child of Root, with the liveness evidence the decision rests
// on. NewestAgeSec is the age of the NEWEST file found anywhere beneath the entry (the dir's
// own mtime when it holds no files at all) — never the top-level mtime alone.
type GoTmpEntry struct {
	Name           string       `json:"name"`
	Path           string       `json:"path"`
	Bytes          int64        `json:"bytes"`
	Files          int          `json:"files"`
	Directories    int          `json:"directories"`
	NewestAgeSec   float64      `json:"newest_age_sec"`
	Truncated      bool         `json:"truncated,omitempty"` // walk hit MaxWalkEntries — liveness unproven
	ScanErr        string       `json:"scan_err,omitempty"`
	Verdict        GoTmpVerdict `json:"verdict,omitempty"`
	Reason         string       `json:"reason"`
	ReferencedBy   []int        `json:"referenced_by,omitempty"`
	QuarantinePath string       `json:"quarantine_path,omitempty"`
	Removed        bool         `json:"removed,omitempty"`
	RemoveErr      string       `json:"remove_err,omitempty"`
}

// GoTmpBand is one row of the age split: how much mass sits in a given age range. See the
// file header for why every report carries this.
type GoTmpBand struct {
	Name    string `json:"name"` // in_flight | stale | cold
	Entries int    `json:"entries"`
	Bytes   int64  `json:"bytes"`
}

// GoTmpReport is the full outcome: every entry with its verdict, the age split, and the
// reaped totals. DryRun true means Reaped names what WOULD have been removed.
type GoTmpReport struct {
	Schema           string       `json:"schema"`
	Root             string       `json:"root"`
	DryRun           bool         `json:"dry_run"`
	MinAgeSec        float64      `json:"min_age_sec"`
	Entries          []GoTmpEntry `json:"entries,omitempty"`
	Bands            []GoTmpBand  `json:"bands,omitempty"`
	TotalBytes       int64        `json:"total_bytes"`
	ReapedBytes      int64        `json:"reaped_bytes"`
	ProcessSnapshots int          `json:"process_snapshots"`
	ProcessErr       string       `json:"process_err,omitempty"`
	Quarantine       string       `json:"quarantine,omitempty"`
	// Reaped are the paths removed (or, in a dry run, that would be removed).
	Reaped []string `json:"reaped,omitempty"`
	// Err is set when Root itself could not be read. The sweep then does nothing.
	Err string `json:"err,omitempty"`
}

// ReapCount reports how many entries the sweep reaped (or would reap in a dry run).
func (r GoTmpReport) ReapCount() int { return len(r.Reaped) }

// Failed reports whether any entry the sweep chose to reap could not actually be removed.
// A caller that grades its own maintenance tick must surface this: a WORK dir that resists
// deletion means the space is not coming back on the next tick either.
func (r GoTmpReport) Failed() bool {
	if r.Err != "" || r.ProcessErr != "" {
		return true
	}
	for _, e := range r.Entries {
		if e.RemoveErr != "" {
			return true
		}
	}
	return false
}

// Summary is the one-line operator rendering: what was reclaimed, and — because the whole
// point of the age split is that a raw total misleads — how much mass was deliberately kept
// and why. Lives here rather than at the CLI edge so the sentence is testable and cannot
// drift between the two callers that print it.
func (r GoTmpReport) Summary() string {
	if r.Root == "" {
		return "go-build WORK dirs: rung disabled (no GOTMPDIR configured)"
	}
	if r.Err != "" {
		return "go-build WORK dirs: could not read " + r.Root + ": " + r.Err
	}
	var live, indeterminate, foreign int
	for _, e := range r.Entries {
		switch e.Verdict {
		case GoTmpReap:
			// Reap totals are derived by ReapCount and ReapedBytes below.
		case GoTmpKeepLive:
			live++
		case GoTmpKeepIndeterminate:
			indeterminate++
		case GoTmpKeepForeign:
			foreign++
		}
	}
	verb := "reaped"
	if r.DryRun {
		verb = "would reap"
	}
	return fmt.Sprintf("go-build WORK dirs in %s: %s %d (%s), kept %d live, %d indeterminate, %d foreign; %s on disk",
		r.Root, verb, r.ReapCount(), goTmpMB(r.ReapedBytes), live, indeterminate, foreign, goTmpMB(r.TotalBytes))
}

// goTmpMB renders a byte count in MB at one decimal — the unit the audit that filed this
// reported in, so a report and the ticket read in the same numbers.
func goTmpMB(b int64) string {
	return fmt.Sprintf("%.1f MB", float64(b)/(1<<20))
}

// GoTmpRootFromEnv resolves the directory to sweep from the environment, using the injected
// lookup so the resolution is testable without mutating the process env. A nil lookup or an
// unset/blank GOTMPDIR yields "", which makes the sweep a no-op.
func GoTmpRootFromEnv(lookup func(string) string) string {
	if lookup == nil {
		return ""
	}
	return strings.TrimSpace(lookup(GoTmpDirEnv))
}

// PlanGoTmp assigns a verdict to every scanned entry and folds the age split. It is the
// pure half: no clock, no filesystem — the ages were computed by ScanGoTmp against
// opts.Now — so the whole decision table is testable from hand-built entries.
func PlanGoTmp(entries []GoTmpEntry, opts GoTmpOptions) GoTmpReport {
	minAge := opts.MinAge
	if minAge <= 0 {
		minAge = DefaultGoTmpMinAge
	}
	prefixes := opts.Prefixes
	if prefixes == nil {
		prefixes = DefaultGoTmpPrefixes
	}
	minAgeSec := minAge.Seconds()

	rep := GoTmpReport{Schema: GoTmpSchema, Root: opts.Root, MinAgeSec: minAgeSec}
	bands := map[string]*GoTmpBand{
		"in_flight": {Name: "in_flight"},
		"stale":     {Name: "stale"},
		"cold":      {Name: "cold"},
	}

	out := make([]GoTmpEntry, 0, len(entries))
	for _, e := range entries {
		e.Verdict = goTmpVerdict(e, minAgeSec, prefixes)
		if e.Reason == "" {
			switch e.Verdict {
			case GoTmpReap:
				e.Reason = GoTmpReasonEligible
			case GoTmpKeepLive:
				e.Reason = GoTmpReasonFresh
			case GoTmpKeepForeign:
				e.Reason = GoTmpReasonForeign
			case GoTmpKeepIndeterminate:
				if e.Truncated {
					e.Reason = GoTmpReasonWalkLimit
				} else {
					e.Reason = GoTmpReasonScanFailed
				}
			}
		}
		rep.TotalBytes += e.Bytes
		band := bands["in_flight"]
		switch {
		case e.NewestAgeSec >= GoTmpColdAgeSec:
			band = bands["cold"]
		case e.NewestAgeSec >= minAgeSec:
			band = bands["stale"]
		}
		band.Entries++
		band.Bytes += e.Bytes
		if e.Verdict == GoTmpReap {
			rep.Reaped = append(rep.Reaped, e.Path)
			rep.ReapedBytes += e.Bytes
		}
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	sort.Strings(rep.Reaped)
	rep.Entries = out
	// Fixed band order (youngest first) so two runs over the same tree render identically.
	for _, name := range []string{"in_flight", "stale", "cold"} {
		rep.Bands = append(rep.Bands, *bands[name])
	}
	return rep
}

// goTmpVerdict is the decision table. The keep cases are checked FIRST and in order of how
// little they trust the evidence, so a reap is only ever reached by an entry that is both
// in the swept name set and provably quiet.
func goTmpVerdict(e GoTmpEntry, minAgeSec float64, prefixes []string) GoTmpVerdict {
	if !goTmpNameMatches(e.Name, prefixes) {
		return GoTmpKeepForeign
	}
	if e.Reason != "" {
		return GoTmpKeepIndeterminate
	}
	if e.ScanErr != "" || e.Truncated {
		return GoTmpKeepIndeterminate
	}
	if e.NewestAgeSec < minAgeSec {
		return GoTmpKeepLive
	}
	return GoTmpReap
}

// goTmpNameMatches reports whether a top-level entry name is in the swept set. `go` names a
// WORK dir `go-build<random>`, so the match is a prefix, not equality.
func goTmpNameMatches(name string, prefixes []string) bool {
	for _, p := range prefixes {
		if p != "" && strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

// ScanGoTmp reads Root's immediate children and measures each one: total bytes, file count,
// and the age of the newest file anywhere inside. It never removes anything. Non-directory
// children remain in the inventory with a typed keep reason.
func ScanGoTmp(opts GoTmpOptions) ([]GoTmpEntry, error) {
	root, err := canonicalGoTmpRoot(opts)
	if err != nil {
		return nil, err
	}
	if root == "" {
		return nil, nil
	}
	now := opts.Now
	if now.IsZero() {
		now = time.Now()
	}
	maxWalk := opts.MaxWalkEntries
	if maxWalk <= 0 {
		maxWalk = DefaultGoTmpMaxWalkEntries
	}

	children, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	maxCandidates := opts.MaxCandidates
	if maxCandidates <= 0 {
		maxCandidates = goTmpDirEntryLimit
	}
	if len(children) > maxCandidates {
		return nil, fmt.Errorf("candidate bound exceeded: %d immediate children > %d", len(children), maxCandidates)
	}
	entries := make([]GoTmpEntry, 0, len(children))
	for _, c := range children {
		entry := scanGoTmpEntry(filepath.Join(root, c.Name()), c.Name(), now, maxWalk)
		resolveGoTmpChild(root, &entry)
		entries = append(entries, entry)
	}
	return entries, nil
}

// scanGoTmpEntry measures ONE top-level entry. The newest mtime is taken over files only:
// a directory's mtime moves when its immediate children are created or removed, which
// makes it a noisy proxy for "a build is writing here" in both directions. When the entry
// holds no files at all, the entry dir's own mtime is the only evidence there is.
func scanGoTmpEntry(path, name string, now time.Time, maxWalk int) GoTmpEntry {
	e := GoTmpEntry{Name: name, Path: path}
	info, err := os.Lstat(path)
	if err != nil {
		e.ScanErr = err.Error()
		e.Reason = GoTmpReasonScanFailed
		return e
	}
	reparse, err := goTmpIsReparse(path, info)
	if err != nil {
		e.ScanErr = err.Error()
		e.Reason = GoTmpReasonScanFailed
		return e
	}
	if reparse {
		e.Reason = GoTmpReasonReparsePoint
		return e
	}
	if !info.IsDir() {
		e.Bytes = info.Size()
		e.Reason = GoTmpReasonNotDirectory
		return e
	}
	var newest time.Time
	seen := 0
	walkErr := filepath.WalkDir(path, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			// A vanished child is normal here (a live build is deleting as we walk) and is
			// not evidence of anything; record it so the entry stays INDETERMINATE.
			if e.ScanErr == "" {
				e.ScanErr = err.Error()
			}
			return nil
		}
		seen++
		if seen > maxWalk {
			e.Truncated = true
			return filepath.SkipAll
		}
		if p != path && d.Name() == ".git" {
			e.Reason = GoTmpReasonNestedRepository
			return filepath.SkipAll
		}
		if d.IsDir() {
			e.Directories++
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			if e.ScanErr == "" {
				e.ScanErr = ierr.Error()
			}
			return nil
		}
		reparse, rerr := goTmpIsReparse(p, info)
		if rerr != nil {
			if e.ScanErr == "" {
				e.ScanErr = rerr.Error()
			}
			return nil
		}
		if reparse {
			e.Reason = GoTmpReasonReparsePoint
			return filepath.SkipAll
		}
		e.Files++
		e.Bytes += info.Size()
		if info.ModTime().After(newest) {
			newest = info.ModTime()
		}
		return nil
	})
	if walkErr != nil && e.ScanErr == "" {
		e.ScanErr = walkErr.Error()
	}
	if newest.IsZero() {
		if info, ierr := os.Stat(path); ierr == nil {
			newest = info.ModTime()
		} else if e.ScanErr == "" {
			e.ScanErr = ierr.Error()
		}
	}
	if !newest.IsZero() {
		e.NewestAgeSec = now.Sub(newest).Seconds()
	}
	return e
}

// SweepGoTmp is the whole rung: scan Root, plan, and — when apply is true — remove every
// entry the plan reaped. With apply false it is a pure diagnosis that mutates nothing while
// reporting exactly what it would have removed, through the same decision path.
//
// A removal failure is recorded on the entry (RemoveErr) and surfaced by Report.Failed(); it
// never aborts the rest of the sweep, because one undeletable WORK dir must not strand the
// other 43.
func SweepGoTmp(opts GoTmpOptions, apply bool) GoTmpReport {
	entries, err := ScanGoTmp(opts)
	if err != nil {
		return GoTmpReport{Schema: GoTmpSchema, Root: opts.Root, DryRun: !apply, Err: err.Error()}
	}
	root, err := canonicalGoTmpRoot(opts)
	if err != nil {
		return GoTmpReport{Schema: GoTmpSchema, Root: opts.Root, DryRun: !apply, Err: err.Error()}
	}
	opts.Root = root
	rep := PlanGoTmp(entries, opts)
	rep.DryRun = !apply
	if root == "" {
		return rep
	}
	foundEntry := false
	for _, entry := range rep.Entries {
		if entry.Verdict == GoTmpReap || entry.Verdict == GoTmpKeepLive {
			foundEntry = true
			break
		}
	}
	if !foundEntry {
		return rep
	}
	processes, processErr := goTmpProcessSnapshot(opts)
	rep.ProcessSnapshots = 1
	if processErr != nil {
		rep.ProcessErr = processErr.Error()
	}
	for i := range rep.Entries {
		e := &rep.Entries[i]
		if e.Verdict != GoTmpReap && e.Verdict != GoTmpKeepLive {
			continue
		}
		if processErr != nil {
			if e.Verdict == GoTmpReap {
				e.Verdict = GoTmpKeepIndeterminate
				e.Reason = GoTmpReasonProcessEnumerationFailed
			}
			continue
		}
		if pids := goTmpReferencingPIDs(processes, e.Path); len(pids) > 0 {
			e.Verdict = GoTmpKeepLive
			e.Reason = GoTmpReasonReferenced
			e.ReferencedBy = pids
		}
	}
	rebuildGoTmpReaped(&rep)
	if !apply {
		return rep
	}
	return applyGoTmpQuarantine(opts, rep)
}
