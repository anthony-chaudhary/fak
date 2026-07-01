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
)

type dispatchLiveScope struct {
	Issue int
	Lane  string
	Tree  []string
	Log   string
}

var dispatchResolveLogRE = regexp.MustCompile(`^resolve-(\d+)-.*\.log$`)

func liveResolutionIssues(runsDir string) map[int]bool {
	out := map[int]bool{}
	for _, log := range resolveLogs(runsDir) {
		issue, ok := issueFromResolveLog(filepath.Base(log))
		if !ok {
			continue
		}
		pid, ok := readPID(strings.TrimSuffix(log, filepath.Ext(log)) + ".pid")
		if ok && dispatchPIDAlive(pid) {
			out[issue] = true
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
		pid, ok := readPID(strings.TrimSuffix(log, filepath.Ext(log)) + ".pid")
		if !ok || !dispatchPIDAlive(pid) || dispatchLogIsBannerNoop(log) {
			continue
		}
		stem := strings.TrimSuffix(log, filepath.Ext(log))
		tree := readResolveLeaseTree(stem + dispatchLeaseTreeSidecarSuffix)
		if len(tree) == 0 {
			continue
		}
		out = append(out, dispatchLiveScope{
			Issue: issue,
			Lane:  laneFromSpawnHeader(log),
			Tree:  tree,
			Log:   log,
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
	out := map[int]bool{}
	if cooldownMin <= 0 {
		return out
	}
	cutoff := time.Now().Add(-time.Duration(cooldownMin) * time.Minute)
	for _, log := range resolveLogs(runsDir) {
		st, err := os.Stat(log)
		if err != nil || st.ModTime().Before(cutoff) {
			continue
		}
		issue, ok := issueFromResolveLog(filepath.Base(log))
		if ok {
			out[issue] = true
		}
	}
	return out
}

func resolveLogs(runsDir string) []string {
	matches, _ := filepath.Glob(filepath.Join(runsDir, "resolve-*.log"))
	sort.Strings(matches)
	return matches
}

func issueFromResolveLog(name string) (int, bool) {
	m := dispatchResolveLogRE.FindStringSubmatch(name)
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
