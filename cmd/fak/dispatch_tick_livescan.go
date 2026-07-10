package main

// Live-resolution scanning: discover in-flight issue-resolution workers by reading
// the runs directory. These helpers turn the resolve-*.log / .pid / .lease-tree.json
// sidecars a spawned worker leaves behind into the live view the tick picker needs —
// which issues are already being worked, which lanes are held, which file-trees are
// in flight, and which lanes are pinned only by a dead banner-noop worker (#1275,
// #1398) and can be reclaimed. Split out of dispatch_tick.go along this concern seam
// so the dispatch surface stays steerable as new verbs land (steerability
// dispatch_god_file). Behavior-preserving code motion — same package, no logic change.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
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

func liveResolutionIssues(runsDir string) map[int]bool {
	out := map[int]bool{}
	for issue := range liveResolutionIssueDetails(runsDir) {
		out[issue] = true
	}
	return out
}

func liveResolutionIssueDetails(runsDir string) map[int]dispatchLiveScope {
	out := map[int]dispatchLiveScope{}
	for _, live := range liveResolutionScopes(runsDir) {
		if _, exists := out[live.Issue]; !exists {
			out[live.Issue] = live
		}
	}
	return out
}

func liveResolutionLanes(runsDir string) map[string]bool {
	out := map[string]bool{}
	for _, log := range resolveLogs(runsDir) {
		pid, ok := readPID(strings.TrimSuffix(log, filepath.Ext(log)) + ".pid")
		if !ok || !dispatchPIDAlive(pid) {
			continue
		}
		// A worker whose log is a terminal banner no-op (#1275: it printed only its
		// startup banner -- "> build · glm-…" -- and produced nothing) holds no real
		// work even when its pid still passes the liveness gate above. An opencode
		// worker runs as a `node` image, so AFTER it exits a recycled `node` pid that
		// lands in the spawn window passes dispatchPIDAlive and would otherwise pin
		// the lane FOREVER (#1398: `docs` stayed LANE_BUSY behind dead 122-byte no-ops
		// while real docs work could not dispatch). Drop such a lane so a lane held
		// ONLY by dead no-op workers reports FREE and `fak dispatch tick --lane docs`
		// returns WOULD_SPAWN. Safe: a genuinely live worker streams kilobytes past
		// the stub floor within seconds so it never classifies as a banner no-op, and
		// on a LIVE tick the fenced git-ref lease (acquireDispatchLaneLease) still
		// serializes a just-started worker across hosts.
		if dispatchLogIsBannerNoop(log) {
			continue
		}
		if lane := laneFromSpawnHeader(log); lane != "" {
			out[lane] = true
		}
	}
	return out
}

func liveResolutionTreeCollision(runsDir string, requested []string) (dispatchLiveScope, bool) {
	requested = dispatchTrimTree(requested)
	if len(requested) == 0 {
		return dispatchLiveScope{}, false
	}
	for _, live := range liveResolutionScopes(runsDir) {
		if len(live.Tree) == 0 {
			continue
		}
		if dispatchorder.TreesOverlap(requested, live.Tree) {
			return live, true
		}
	}
	return dispatchLiveScope{}, false
}

func liveResolutionScopes(runsDir string) []dispatchLiveScope {
	var out []dispatchLiveScope
	for _, log := range resolveLogs(runsDir) {
		issue, ok := issueFromResolveLog(filepath.Base(log))
		if !ok {
			continue
		}
		stem := strings.TrimSuffix(log, filepath.Ext(log))
		pid, ok := readPID(stem + ".pid")
		if !ok || !dispatchPIDAlive(pid) || dispatchLogIsBannerNoop(log) {
			continue
		}
		lane := laneFromSpawnHeader(log)
		tree := readResolveLeaseTree(stem + dispatchLeaseTreeSidecarSuffix)
		out = append(out, dispatchLiveScope{
			Issue:   issue,
			Lane:    lane,
			Tree:    tree,
			Log:     log,
			PID:     pid,
			Worker:  filepath.Base(stem),
			LeaseID: readResolveLeaseID(stem, lane),
		})
	}
	return out
}

func readResolveLeaseTree(path string) []string {
	b, err := os.ReadFile(path)
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
	b, err := os.ReadFile(stem + dispatchLeaseIDSidecarSuffix)
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

// dispatchLogIsBannerNoop reports whether a worker log is a terminal banner no-op: it
// is at/under the stub floor AND carries only the opencode/glm startup banner. Used to
// reap a lane held by a dead no-op worker whose recycled pid still passes the liveness
// gate (#1398). FAIL-CLOSED to false on any stat/read error or an over-floor log so a
// log we cannot classify -- or one with real streamed work -- is never falsely reaped.
func dispatchLogIsBannerNoop(path string) bool {
	st, err := os.Stat(path)
	if err != nil || st.Size() > dispatchResolveLogStubFloorBytes {
		return false
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	return dispatchNoopBannerRE.Match(b)
}

func recentlyAttemptedIssues(runsDir string, cooldownMin int) map[int]bool {
	return recentlyAttemptedIssuesAt(runsDir, cooldownMin, time.Now())
}

func recentlyAttemptedIssuesAt(runsDir string, cooldownMin int, now time.Time) map[int]bool {
	out := map[int]bool{}
	if cooldownMin <= 0 {
		return out
	}
	for _, row := range cooldownIssueRowsAt(runsDir, cooldownMin, now) {
		if row.Cooling {
			out[row.Issue] = true
		}
	}
	return out
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
	out := map[int]bool{}
	if budget <= 0 {
		return out
	}
	attempts := map[int]map[string]bool{}
	for _, log := range resolveLogs(runsDir) {
		base := filepath.Base(log)
		issue, ok := issueFromResolveLog(base)
		if !ok {
			continue
		}
		key := strings.TrimSuffix(base, ".log")
		key = strings.TrimSuffix(key, ".out")
		key = strings.TrimSuffix(key, ".err")
		if attempts[issue] == nil {
			attempts[issue] = map[string]bool{}
		}
		attempts[issue][key] = true
	}
	for issue, keys := range attempts {
		if len(keys) >= budget {
			out[issue] = true
		}
	}
	return out
}

type dispatchCooldownRow struct {
	Issue                    int
	LastAttemptUnix          int64
	LastAttemptAgeSeconds    int
	CooldownRemainingSeconds int
	NextEligibleUnix         int64
	Cooling                  bool
}

func cooldownIssueRows(runsDir string, cooldownMin int) []map[string]any {
	rows := cooldownIssueRowsAt(runsDir, cooldownMin, time.Now())
	out := make([]map[string]any, 0, len(rows))
	for _, row := range rows {
		out = append(out, row.Map())
	}
	return out
}

func cooldownIssueRowsAt(runsDir string, cooldownMin int, now time.Time) []dispatchCooldownRow {
	if cooldownMin <= 0 {
		return nil
	}
	cooldown := time.Duration(cooldownMin) * time.Minute
	latest := map[int]time.Time{}
	for _, attempt := range resolveAttemptFiles(runsDir) {
		st, err := os.Stat(attempt)
		if err != nil {
			continue
		}
		issue, ok := issueFromResolveAttempt(filepath.Base(attempt))
		if !ok {
			continue
		}
		if prev, exists := latest[issue]; !exists || st.ModTime().After(prev) {
			latest[issue] = st.ModTime()
		}
	}
	issues := make([]int, 0, len(latest))
	for issue := range latest {
		issues = append(issues, issue)
	}
	sort.Ints(issues)
	out := make([]dispatchCooldownRow, 0, len(issues))
	for _, issue := range issues {
		last := latest[issue]
		if last.After(now) {
			last = now
		}
		next := last.Add(cooldown)
		remaining := int(next.Sub(now).Seconds())
		if remaining < 0 {
			remaining = 0
		}
		age := int(now.Sub(last).Seconds())
		if age < 0 {
			age = 0
		}
		out = append(out, dispatchCooldownRow{
			Issue:                    issue,
			LastAttemptUnix:          last.Unix(),
			LastAttemptAgeSeconds:    age,
			CooldownRemainingSeconds: remaining,
			NextEligibleUnix:         next.Unix(),
			Cooling:                  remaining > 0,
		})
	}
	return out
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
	matches, _ := filepath.Glob(filepath.Join(runsDir, "resolve-*.log"))
	sort.Strings(matches)
	return matches
}

// resolveAttemptFiles returns every resolve worker attempt artifact under runsDir: the
// .log transcripts AND the durable .witness audit sidecars. The cooldown scan reads both so
// a witnessed dead slot still cools its issue even after its .log is gone -- the .witness is
// the durable cooldown evidence prune_dead_sidecars deliberately retains, and it is written
// post-mortem by the witness sweep so its mtime carries the most-recent attempt touch. This
// mirrors recently_attempted_issues in tools/issue_resolve_dispatch.py, which globs
// resolve-*.log AND resolve-*.witness. Sorted for a stable, deterministic scan order.
func resolveAttemptFiles(runsDir string) []string {
	var out []string
	for _, pattern := range []string{"resolve-*.log", "resolve-*" + dispatchtick.WitnessSidecarSuffix} {
		matches, _ := filepath.Glob(filepath.Join(runsDir, pattern))
		out = append(out, matches...)
	}
	sort.Strings(out)
	return out
}

func issueFromResolveLog(name string) (int, bool) {
	m := dispatchResolveLogRE.FindStringSubmatch(name)
	if m == nil {
		return 0, false
	}
	n, err := strconv.Atoi(m[1])
	return n, err == nil
}

// issueFromResolveAttempt extracts the issue number from a resolve attempt artifact's
// basename -- a .log OR a .witness -- via the extension-agnostic dispatchResolveAttemptRE,
// so the cooldown scan keys off either artifact the same way Python's _LOG_ISSUE_RE does.
func issueFromResolveAttempt(name string) (int, bool) {
	m := dispatchResolveAttemptRE.FindStringSubmatch(name)
	if m == nil {
		return 0, false
	}
	n, err := strconv.Atoi(m[1])
	return n, err == nil
}

func laneFromSpawnHeader(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	line := strings.SplitN(string(b), "\n", 2)[0]
	for _, field := range strings.Fields(line) {
		if strings.HasPrefix(field, "lane=") {
			return strings.TrimPrefix(field, "lane=")
		}
	}
	return ""
}

func readPID(path string) (int, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(b)))
	return pid, err == nil && pid > 0
}
