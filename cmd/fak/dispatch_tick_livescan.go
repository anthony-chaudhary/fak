package main

// Live-resolution scanning: discover in-flight issue-resolution workers by reading
// the runs directory. These helpers turn the resolve-*.log / .pid / .lease-tree.json
// sidecars a spawned worker leaves behind into the live view the tick picker needs —
// which issues are already being worked, which lanes are held, which file-trees are
// in flight, and which lanes are pinned only by a dead banner-noop worker (#1275,
// #1398) and can be reclaimed. Split out of dispatch_tick.go along this concern seam
// so the dispatch surface stays steerable as new verbs land (steerability
// dispatch_god_file).
//
// The live-lane / live-scope / cooldown / attempt-budget / tree-collision views are
// now thin projections over a single per-tick runsSnapshot (dispatch_tick_snapshot.go):
// each public helper here builds a one-shot snapshot and projects from it, so a caller
// that needs several views scans the runs directory once instead of once per view
// (#3593). The projections are byte-identical to the prior per-loop scans. All sidecar
// I/O routes through the fsGlob/fsStat/fsReadFile seam below so the scan is countable.

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// fsGlob/fsStat/fsReadFile/fsOpen are the runs-directory I/O seam. Every sidecar glob,
// stat, read and open in this file (and the snapshot scan) routes through them, so a test
// can swap in counting wrappers to prove a tick's discovery cost is one pass -- the
// runsSnapshot projections read only captured state and do zero further I/O (#3593) --
// and that the spawn-header lane probe reads a bounded prefix, never the whole streaming
// transcript (#3466).
var (
	fsGlob     = filepath.Glob
	fsStat     = os.Stat
	fsReadFile = os.ReadFile
	fsOpen     = func(path string) (io.ReadCloser, error) { return os.Open(path) }
)

type dispatchLiveScope struct {
	Issue   int
	Lane    string
	Tree    []string
	Log     string
	PID     int
	Worker  string
	LeaseID string
}

var dispatchResolveLogRE = regexp.MustCompile(`^resolve-(\d+)-.*\.log$`)

// dispatchResolveAttemptRE extracts the issue number from ANY resolve worker attempt
// artifact -- a resolve-<N>-<stamp>.log transcript OR its durable resolve-<N>-<stamp>.witness
// audit sidecar. Unlike dispatchResolveLogRE it is NOT anchored to the .log extension, so it
// mirrors the Python dispatcher's extension-agnostic _LOG_ISSUE_RE (r"resolve-(\d+)-") that
// recently_attempted_issues uses (tools/issue_resolve_dispatch.py).
var dispatchResolveAttemptRE = regexp.MustCompile(`^resolve-(\d+)-`)

// liveResolutionIssues / liveResolutionIssueDetails / liveResolutionLanes /
// liveResolutionTreeCollision / liveResolutionScopes each build a one-shot runsSnapshot
// and project the requested view from it (#3593). The banner-noop lane-reclaim rationale
// (#1275/#1398) now lives on runsSnapshot.liveLanes / the scan's live gate. A caller that
// needs several of these views should build ONE snapshot (scanRunsSnapshot) and project
// from it directly -- the tick picker and wave pricer do, to scan the runs dir once.

func liveResolutionIssues(runsDir string) map[int]bool {
	return scanRunsSnapshot(runsDir, time.Now()).liveIssues()
}

func liveResolutionIssueDetails(runsDir string) map[int]dispatchLiveScope {
	return scanRunsSnapshot(runsDir, time.Now()).liveIssueDetails()
}

func liveResolutionLanes(runsDir string) map[string]bool {
	return scanRunsSnapshot(runsDir, time.Now()).liveLanes()
}

func liveResolutionTreeCollision(runsDir string, requested []string) (dispatchLiveScope, bool) {
	return scanRunsSnapshot(runsDir, time.Now()).treeCollision(requested)
}

func liveResolutionScopes(runsDir string) []dispatchLiveScope {
	return scanRunsSnapshot(runsDir, time.Now()).liveScopes()
}

func readResolveLeaseTree(path string) []string {
	b, err := fsReadFile(path)
	if err != nil {
		return nil
	}
	var tree []string
	if err := json.Unmarshal(b, &tree); err != nil {
		return nil
	}
	return dispatchTrimTree(tree)
}

func readResolveLeaseID(stem, lane string) string {
	b, err := fsReadFile(stem + dispatchLeaseIDSidecarSuffix)
	if err == nil {
		if id := strings.TrimSpace(string(b)); id != "" {
			return id
		}
	}
	if lane != "" {
		return dispatchLaneLeaseID(lane)
	}
	return ""
}

func dispatchTrimTree(tree []string) []string {
	out := make([]string, 0, len(tree))
	for _, item := range tree {
		if s := strings.TrimSpace(item); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// dispatchResolveLogStubFloorBytes mirrors the Python dispatcher's _STUB_LOG_MAX_BYTES
// (tools/issue_resolve_dispatch.py): a genuinely live worker streams kilobytes within
// seconds, so a log at or under this floor that carries only the opencode/glm startup
// banner is a terminal banner no-op (#1275), never live work.
const dispatchResolveLogStubFloorBytes = 512

// dispatchNoopBannerRE matches the opencode/glm startup banner ("> build · glm-…"),
// the documented banner-only no-op signature (#1275). Mirrors the Python
// _NOOP_BANNER_RE so the Go tick classifies a dead no-op the same way the legacy
// helper does.
var dispatchNoopBannerRE = regexp.MustCompile(`(?i)>\s*build\s*[·:]`)

// classifyBannerNoop reports whether a worker log is a terminal banner no-op: it is
// at/under the stub floor AND carries only the opencode/glm startup banner. Used to reap
// a lane held by a dead no-op worker whose recycled pid still passes the liveness gate
// (#1398). The snapshot scan classifies a log off the single stat it already took
// (dispatch_tick_snapshot.go) instead of re-statting. An over-floor log is never a no-op;
// a read error fails closed to false so an unclassifiable log is never falsely reaped.
func classifyBannerNoop(st os.FileInfo, path string) bool {
	if st.Size() > dispatchResolveLogStubFloorBytes {
		return false
	}
	b, err := fsReadFile(path)
	if err != nil {
		return false
	}
	return dispatchNoopBannerRE.Match(b)
}

func recentlyAttemptedIssuesAt(runsDir string, cooldownMin int, now time.Time) map[int]bool {
	return scanRunsSnapshot(runsDir, now).recentlyAttempted(cooldownMin)
}

// dispatchAttemptBudgetDefault is the number of recorded worker attempts on ONE
// still-open issue past which the wave stops auto-dispatching it. A shipped issue
// closes and leaves the candidate set, so an OPEN issue that has burned this many
// ~100k-token worker sessions has not converged — it is poison, not progress, and
// the account budget it keeps consuming every cooldown window is better spent on
// issues that can land. Eight is deliberately generous (a well-scoped leaf ships in
// 1–3 attempts, so 8 unshipped tries means the issue is mis-scoped and needs human
// triage, not another worker); tune or disable via FAK_DISPATCH_ATTEMPT_BUDGET
// (0 disables the cap).
const dispatchAttemptBudgetDefault = 8

// dispatchAttemptBudget resolves the poison-issue attempt cap, honoring the
// FAK_DISPATCH_ATTEMPT_BUDGET override (a non-negative integer; 0 disables).
func dispatchAttemptBudget() int {
	if v := strings.TrimSpace(os.Getenv("FAK_DISPATCH_ATTEMPT_BUDGET")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return dispatchAttemptBudgetDefault
}

// attemptExhaustedIssues returns the issues whose recorded worker attempts have
// reached the budget — the poison-issue cap the time cooldown never enforced (an
// issue cools ~2h then re-enters the pool, so one that can never ship burns a
// worker every window indefinitely; #1419 alone logged 19 attempts, and ~64% of
// all dispatch sessions ended WASTED_SPAWN concentrated on a handful of such
// issues). Each spawned worker writes one resolve log per attempt; some legacy
// workers split it into .out/.err, so an attempt is keyed by its <issue>-<stamp>
// base and counted once. An at-or-over-budget issue is held OUT of the wave so
// its account spend stops; a human (or /dos-replan) then decomposes or closes it.
// Reversible: the hold clears itself once the issue closes (it leaves the
// candidate set), or an operator can raise FAK_DISPATCH_ATTEMPT_BUDGET or clear
// the run-dir logs. A budget <= 0 disables the cap (empty set = legacy behavior).
func attemptExhaustedIssues(runsDir string, budget int) map[int]bool {
	return scanRunsSnapshot(runsDir, time.Now()).attemptExhausted(budget)
}

type dispatchCooldownRow struct {
	Issue                    int
	LastAttemptUnix          int64
	LastAttemptAgeSeconds    int
	CooldownRemainingSeconds int
	NextEligibleUnix         int64
	Cooling                  bool
}

func cooldownIssueRowsAt(runsDir string, cooldownMin int, now time.Time) []dispatchCooldownRow {
	return scanRunsSnapshot(runsDir, now).cooldownRows(cooldownMin)
}

func (r dispatchCooldownRow) Map() map[string]any {
	return map[string]any{
		"issue":                      r.Issue,
		"last_attempt_unix":          r.LastAttemptUnix,
		"last_attempt_utc":           time.Unix(r.LastAttemptUnix, 0).UTC().Format(time.RFC3339),
		"last_attempt_age_seconds":   r.LastAttemptAgeSeconds,
		"cooldown_remaining_seconds": r.CooldownRemainingSeconds,
		"next_eligible_unix":         r.NextEligibleUnix,
		"next_eligible_utc":          time.Unix(r.NextEligibleUnix, 0).UTC().Format(time.RFC3339),
		"cooling":                    r.Cooling,
	}
}

func resolveLogs(runsDir string) []string {
	matches, _ := fsGlob(filepath.Join(runsDir, "resolve-*.log"))
	sort.Strings(matches)
	return matches
}

// resolveWitnessFiles returns the durable resolve-*.witness audit sidecars under runsDir,
// sorted. The cooldown fold reads them so a witnessed dead slot still cools its issue even
// after its .log is gone -- the .witness is the durable cooldown evidence prune_dead_sidecars
// deliberately retains, written post-mortem by the witness sweep so its mtime carries the
// most-recent attempt touch. Together with resolveLogs this mirrors the two-pattern glob
// recently_attempted_issues uses in tools/issue_resolve_dispatch.py.
func resolveWitnessFiles(runsDir string) []string {
	matches, _ := fsGlob(filepath.Join(runsDir, "resolve-*"+dispatchtick.WitnessSidecarSuffix))
	sort.Strings(matches)
	return matches
}

// issueFromResolveArtifact reads the issue number re captures out of a resolve artifact's
// basename. ok is false when the name does not match at all, and when the captured text is
// not a number -- the one extraction rule both named accessors below apply.
func issueFromResolveArtifact(re *regexp.Regexp, name string) (int, bool) {
	m := re.FindStringSubmatch(name)
	if m == nil {
		return 0, false
	}
	n, err := strconv.Atoi(m[1])
	return n, err == nil
}

func issueFromResolveLog(name string) (int, bool) {
	return issueFromResolveArtifact(dispatchResolveLogRE, name)
}

// issueFromResolveAttempt extracts the issue number from a resolve attempt artifact's
// basename -- a .log OR a .witness -- via the extension-agnostic dispatchResolveAttemptRE,
// so the cooldown scan keys off either artifact the same way Python's _LOG_ISSUE_RE does.
func issueFromResolveAttempt(name string) (int, bool) {
	return issueFromResolveArtifact(dispatchResolveAttemptRE, name)
}

// laneHeaderReadCapBytes bounds how much of a worker log laneFromSpawnHeader reads.
// The `lane=` field lives on the spawn-header FIRST line, but the file itself is a
// live worker's streaming transcript (tens of MB and growing), so the probe must
// never pull the whole file into memory (#3466). 4 KB comfortably covers any real
// `# fak-spawn ...` header; mirrors the cheap stat-gate style of
// dispatchLogIsBannerNoop above (dispatchResolveLogStubFloorBytes).
const laneHeaderReadCapBytes = 4096

// laneFromSpawnHeader extracts the `lane=` field from a worker log's first line,
// reading at most laneHeaderReadCapBytes of the file instead of the entire growing
// transcript (#3466). Parsing of the first line is identical to the legacy
// whole-file read (bytes up to the first '\n', whitespace-split fields, `lane=`
// prefix); only how much of the file is read changed. A header line longer than
// the cap is parsed from its readable prefix; any open/read error fails closed to
// "" exactly like the legacy read did.
func laneFromSpawnHeader(path string) string {
	f, err := fsOpen(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, laneHeaderReadCapBytes)
	line, err := r.ReadSlice('\n')
	// io.EOF: a header-only log with no trailing newline; bufio.ErrBufferFull: a
	// first line longer than the cap. Both still yield the readable prefix.
	if err != nil && err != io.EOF && err != bufio.ErrBufferFull {
		return ""
	}
	for _, field := range strings.Fields(strings.TrimSuffix(string(line), "\n")) {
		if strings.HasPrefix(field, "lane=") {
			return strings.TrimPrefix(field, "lane=")
		}
	}
	return ""
}

func readPID(path string) (int, bool) {
	b, err := fsReadFile(path)
	if err != nil {
		return 0, false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(b)))
	return pid, err == nil && pid > 0
}
