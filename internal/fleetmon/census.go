package fleetmon

// census.go is the cross-AGENT session census (#2213, epic #2209 — the B1 fold
// over B2 descriptors). Today the health surfaces walk the Claude transcript
// namespace ONLY: cmd/fak's discoverWorkers globs ~/.claude/projects/*/*.jsonl,
// and tools/fleet_sessions.py the same. Codex / OpenCode / Aider sessions are
// invisible to every monitor, even though `fak guard` fronts all of them. This
// fold closes that gap: ONE discovery pass that enumerates the live/recent
// sessions of EVERY harnessprofile agent, each row carrying correct agent
// attribution.
//
// The mapping is DATA-DRIVEN off internal/harnessprofile: a profile already
// declares each harness's config-home convention (ConfigHomeGlob — ".claude*",
// ".codex*", "" for the env-key harnesses), so the census reads that registry
// and only adds the one fact a profile does not carry — WHERE, under a resolved
// config home, that harness lays out its per-session transcript/log files
// (nsLayouts). Adding a harness stays a registry entry (+ one layout row if its
// on-disk shape is new), never an edit to every monitor.
//
// Honesty fences, matched to the issue's witness:
//   - Claude rows are BYTE-COMPATIBLE with discoverWorkers: the same grouped
//     projects/<namespace>/<session>.jsonl walk, the same session id (the bare
//     file basename), so the existing `fak fleet monitor` classes see identical
//     identities — no regression.
//   - Codex is first-class: its rollout files live under sessions/ (nested by
//     date), and the session id is the uuid pulled out of the rollout filename.
//   - An agent with NO discoverable namespace (an env-key harness with no config
//     home, or a harness whose config home / transcript root is absent on this
//     host) yields a TYPED KindNoNamespace row — never silence.
//   - Liveness is a RECENCY witness read from the transcript mtime (the
//     last-turn-age rung), not a process-table pid check: a session whose file
//     advanced within RecentWindow reads LIVE, older reads IDLE, and an agent
//     with nothing to age (NO_NAMESPACE, or an unreadable file) reads UNKNOWN —
//     the honest UNKNOWN rung the issue names for the best-effort harnesses. True
//     pid liveness needs a run-plan/process join, a documented follow-on.

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

// CensusSchema tags the census payload so a reader can version the format.
const CensusSchema = "fak-fleet-census/1"

// RecentWindow is the transcript-idle floor under which a session's last turn
// reads LIVE. It aligns with DefaultThresholds().StaleTranscript so a census
// LIVE row lines up with a monitor non-stale-transcript verdict.
const RecentWindow = 20 * time.Minute

// RowKind separates a real session row from the typed NO_NAMESPACE sentinel that
// keeps an un-discoverable agent visible instead of silently absent.
type RowKind string

const (
	// KindSession is one discovered session (a transcript/log file on disk).
	KindSession RowKind = "SESSION"
	// KindNoNamespace marks an agent whose transcript namespace could not be
	// located at all — an env-key harness with no config home, or a harness whose
	// config home / transcript root is absent on this host. The row carries the
	// agent and a reason, never nothing.
	KindNoNamespace RowKind = "NO_NAMESPACE"
)

// Liveness is the honest liveness rung of a census row, derived from the
// transcript's last-turn recency (its mtime), NOT from a process-table pid check.
type Liveness string

const (
	// LivenessLive: the session's transcript advanced within RecentWindow — a
	// recency witness that it is actively advancing (the same rung the monitor and
	// fleet_sessions call live).
	LivenessLive Liveness = "LIVE"
	// LivenessIdle: discoverable, but last advanced longer ago than RecentWindow.
	LivenessIdle Liveness = "IDLE"
	// LivenessUnknown: nothing to age — an unreadable transcript, or a
	// NO_NAMESPACE agent. Pid liveness is not knowable here, so it is not claimed.
	LivenessUnknown Liveness = "UNKNOWN"
)

// CensusRow is one per-agent census row: {agent, session id, last-turn age, pid
// liveness where knowable}. For a KindNoNamespace row Session/Path are empty and
// Note carries why the namespace was not discoverable.
type CensusRow struct {
	Agent       string        `json:"agent"`                 // harnessprofile Name (claude, codex, openai-generic)
	Kind        RowKind       `json:"kind"`                  // SESSION | NO_NAMESPACE
	Session     string        `json:"session,omitempty"`     // session id (claude/codex file id)
	Namespace   string        `json:"namespace,omitempty"`   // per-session grouping dir (claude project dir); "" when flat/nested
	Path        string        `json:"path,omitempty"`        // transcript/log file path
	LastTurnAge time.Duration `json:"-"`                     // now - file mtime; meaningful only when HasAge
	HasAge      bool          `json:"has_age"`               // whether LastTurnAge/AgeSeconds is a real reading
	AgeSeconds  int64         `json:"age_seconds,omitempty"` // LastTurnAge in whole seconds, for JSON consumers
	Liveness    Liveness      `json:"liveness"`              // recency rung (see Liveness)
	Note        string        `json:"note,omitempty"`        // NO_NAMESPACE reason, or a source note
}

// nsLayout declares, per harnessprofile Name, HOW that harness lays out its
// per-session transcript/log files UNDER the resolved config home. The config
// home itself comes from the profile's ConfigHomeGlob (data); this table adds
// only the intra-home shape, the one fact a profile does not carry.
type nsLayout struct {
	// subdir is the transcript root relative to the config home ("projects" for
	// claude, "sessions" for codex). Empty means the config home is the root.
	subdir string
	// grouped is true when sessions sit exactly one directory below subdir
	// (claude's projects/<namespace>/<session>.jsonl); that dir becomes Namespace.
	// false means a recursive walk under subdir (codex rollouts nest by date).
	grouped bool
	// ext is the session-file extension.
	ext string
}

// nsLayouts maps a harnessprofile Name to its on-disk transcript shape. A profile
// with a config home but no entry here has no KNOWN transcript namespace, so it
// yields a NO_NAMESPACE row rather than a wrong guess.
var nsLayouts = map[string]nsLayout{
	"claude": {subdir: "projects", grouped: true, ext: ".jsonl"},
	"codex":  {subdir: "sessions", grouped: false, ext: ".jsonl"},
}

// uuidRE pulls a session uuid out of a codex rollout filename
// (rollout-<timestamp>-<uuid>.jsonl). Claude ids are already bare uuids and are
// taken verbatim (grouped layout), so this only fires on nested/rollout files.
var uuidRE = regexp.MustCompile(`[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}`)

// Census enumerates every harnessprofile agent's discoverable sessions under the
// given home directory, plus a typed NO_NAMESPACE row for each agent whose
// transcript namespace is not discoverable. now is the clock the last-turn-age
// and liveness rung are measured against (injected so the fold is deterministic).
// The result is stably sorted by (agent, namespace, session).
func Census(home string, now time.Time) []CensusRow {
	var rows []CensusRow
	for _, p := range harnessprofile.Profiles() {
		rows = append(rows, censusForProfile(home, now, p)...)
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Agent != rows[j].Agent {
			return rows[i].Agent < rows[j].Agent
		}
		if rows[i].Namespace != rows[j].Namespace {
			return rows[i].Namespace < rows[j].Namespace
		}
		return rows[i].Session < rows[j].Session
	})
	return rows
}

// censusForProfile returns one profile's session rows, or a single NO_NAMESPACE
// row when the profile's transcript namespace cannot be located.
func censusForProfile(home string, now time.Time, p harnessprofile.HarnessProfile) []CensusRow {
	layout, known := nsLayouts[p.Name]
	if home == "" || p.ConfigHomeGlob == "" || !known {
		return []CensusRow{noNamespaceRow(p, known)}
	}
	homes, _ := filepath.Glob(filepath.Join(home, p.ConfigHomeGlob))
	var roots []string
	for _, hd := range homes {
		if !isDir(hd) {
			continue
		}
		root := hd
		if layout.subdir != "" {
			root = filepath.Join(hd, layout.subdir)
		}
		if isDir(root) {
			roots = append(roots, root)
		}
	}
	if len(roots) == 0 {
		return []CensusRow{{
			Agent:    p.Name,
			Kind:     KindNoNamespace,
			Liveness: LivenessUnknown,
			Note:     "no transcript namespace on this host (looked for " + p.ConfigHomeGlob + "/" + layout.subdir + ")",
		}}
	}
	var out []CensusRow
	for _, root := range roots {
		out = append(out, sessionsUnder(root, layout, now, p.Name)...)
	}
	return out
}

// noNamespaceRow builds the typed sentinel for an agent with no discoverable
// namespace, with a reason that distinguishes an env-key harness (no config
// home) from one whose on-disk transcript layout is simply not yet mapped.
func noNamespaceRow(p harnessprofile.HarnessProfile, layoutKnown bool) CensusRow {
	note := "no config home (env-key harness: " + strings.Join(p.Names, "/") + ")"
	if p.ConfigHomeGlob != "" && !layoutKnown {
		note = "config home " + p.ConfigHomeGlob + " has no mapped transcript layout"
	}
	return CensusRow{Agent: p.Name, Kind: KindNoNamespace, Liveness: LivenessUnknown, Note: note}
}

// sessionsUnder enumerates the session files under one resolved transcript root.
func sessionsUnder(root string, layout nsLayout, now time.Time, agent string) []CensusRow {
	var out []CensusRow
	if layout.grouped {
		groups, _ := os.ReadDir(root)
		for _, g := range groups {
			if !g.IsDir() {
				continue
			}
			gdir := filepath.Join(root, g.Name())
			files, _ := filepath.Glob(filepath.Join(gdir, "*"+layout.ext))
			sort.Strings(files)
			for _, f := range files {
				out = append(out, sessionRow(agent, g.Name(), f, layout, now))
			}
		}
		return out
	}
	// Nested/flat: walk the whole root and collect every matching file (codex
	// rollouts sit under sessions/YYYY/MM/DD/). Walk errors are skipped, never
	// fatal — a census is best-effort, never a panic.
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, layout.ext) {
			out = append(out, sessionRow(agent, "", path, layout, now))
		}
		return nil
	})
	return out
}

// sessionRow builds one KindSession row, folding the file's mtime into the
// last-turn age and the recency liveness rung.
func sessionRow(agent, namespace, path string, layout nsLayout, now time.Time) CensusRow {
	r := CensusRow{
		Agent:     agent,
		Kind:      KindSession,
		Session:   sessionID(path, layout),
		Namespace: namespace,
		Path:      path,
		Liveness:  LivenessUnknown,
	}
	if info, err := os.Stat(path); err == nil {
		age := now.Sub(info.ModTime())
		if age < 0 {
			age = 0
		}
		r.LastTurnAge = age
		r.HasAge = true
		r.AgeSeconds = int64(age / time.Second)
		if age <= RecentWindow {
			r.Liveness = LivenessLive
		} else {
			r.Liveness = LivenessIdle
		}
	}
	return r
}

// sessionID derives a session id from a transcript path. A grouped (claude)
// layout takes the bare file basename verbatim, byte-compatible with
// discoverWorkers; a nested (codex) rollout file yields the uuid embedded in its
// name, falling back to the basename when none is present.
func sessionID(path string, layout nsLayout) string {
	base := strings.TrimSuffix(filepath.Base(path), layout.ext)
	if layout.grouped {
		return base
	}
	if m := uuidRE.FindString(base); m != "" {
		return m
	}
	return base
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}
