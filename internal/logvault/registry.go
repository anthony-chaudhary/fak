package logvault

import (
	"os"
	"path/filepath"
	"strings"
)

// Source is one root the vault captures. Every source is optional: a missing
// root is a valid empty source, never an error (the `fak audit usage` posture —
// a box that never ran the dispatcher simply has no .dispatch-runs).
type Source struct {
	ID       string   // vault subdirectory name, e.g. "guard-audit"
	Root     string   // absolute root on this box
	Includes []string // when non-empty, ONLY matching files are captured (same syntax as Excludes)
	Excludes []string // rel-path prefixes ("tmp/") or base-name globs ("*.exe") to skip
	MaxBytes int64    // when >0, an aggressive cap: the walk stops admitting files once cumulative admitted size reaches it (bounds an opt-in tier like the scratchpad)
	Note     string   // why this source matters (shown by plan/du)
}

// credentialExcludes are never captured regardless of source. The vault holds
// forensic logs, not secrets; a token co-located with a log store must not ride
// along. (usage.salt is NOT a credential — it salts digests and must travel
// with usage.jsonl for correlation.)
var credentialExcludes = []string{
	".oauth-token*", "*.token", "*credentials*", ".credentials*", "*.pem", "*.key",
}

// commonExcludes are re-derivable or transient artifacts excluded everywhere.
var commonExcludes = []string{"*.exe", "*.lock", "*.pid", "*.part"}

// DefaultSources is the registry of every durable log store discovered on a fak
// box (writer inventory 2026-07-03): the hash-chained guard/usage/loop journals,
// the DOS trust-kernel state, dispatch run logs, the Claude Code harness store,
// and the per-user fak state dir. Scratch trees that dominate the on-disk size
// but are re-derivable (.fak/tmp checkouts, .dos/_dos_park) are excluded.
func DefaultSources(repoRoot, home string) []Source {
	rel := func(parts ...string) string {
		return filepath.Join(append([]string{repoRoot}, parts...)...)
	}
	srcs := []Source{
		{
			ID:   "dispatch-runs",
			Root: rel(".dispatch-runs"),
			Note: "dispatched-worker logs + sidecars, guard-audit journals, slack outbox, contract ledgers",
		},
		{
			ID:       "dos-state",
			Root:     rel(".dos"),
			Excludes: []string{"_dos_park/"},
			Note:     "DOS trust-kernel evidence: lane journal, hook observations, session streams, run registry (markers/streams are age-GC'd at 7d — capture must outrun it)",
		},
		{
			ID:       "fak-state",
			Root:     rel(".fak"),
			Excludes: []string{"tmp/", "build/", "verify/", "overlay/", "bench-dx/"},
			Note:     "loop ledger (+.broken-* archives), toolproc journal, repoguard decisions, task decisions, dogfood evidence",
		},
		{
			ID:   "goal-runs",
			Root: rel(".goal-runs"),
			Note: "detached goal-fleet worker logs + their guard-audit journals",
		},
		{
			ID:   "nightrun-ledgers",
			Root: rel("docs", "nightrun"),
			Note: "gateway-usage/cache-value/cache-savings/harness-resources/collected ledgers (cache-value.jsonl is gitignored — vault is its only durability)",
		},
		{
			ID:   "dojo-episodes",
			Root: rel(".dojo"),
			Note: "live dojo episodes — the billed-reality corpus",
		},
		{
			ID:   "fleet-registry",
			Root: rel("tools", "_registry"),
			Note: "fleet session dispositions: transitions.log, resume ledgers, probe ledger, tombstones",
		},
		{
			ID:   "fleet-watchdog",
			Root: rel("tools", "_watchdog"),
			Note: "resume-watchdog logs, proc_guard.log, per-resume child logs (age-GC'd at 7d — capture must outrun it)",
		},
		{
			ID:   "dos-state-tools",
			Root: rel("tools", ".dos"),
			Note: "cwd-forked DOS instance from sessions run under tools/ (the .dos-wherever-cwd-was hazard)",
		},
	}
	if cfg, err := os.UserConfigDir(); err == nil {
		srcs = append(srcs, Source{
			ID:   "user-fak-state",
			Root: filepath.Join(cfg, "fak"),
			Note: "per-user CLI usage journal + salt, watchdog-autoheal, vcache turn snapshots (truncate-replace — vault is the only history), session registry",
		})
	}
	if home != "" {
		srcs = append(srcs, Source{
			ID:   "harness-store",
			Root: filepath.Join(home, ".claude", "projects", harnessProjectSlug(repoRoot)),
			Note: "Claude Code session transcripts, per-session artifacts, auto-memory for this repo",
		})
		srcs = append(srcs, Source{
			ID:   "harness-user-ledgers",
			Root: filepath.Join(home, ".claude"),
			// Include-listed: ~/.claude also holds credentials, plugins, and other
			// projects' stores — only the fak fleet ledgers and small harness state
			// dirs ride along. (fak-shadow-ledger + memory-cotravel-ledger are
			// cap-rewritten, so the vault history is their only long record.)
			Includes: []string{
				"fak-*.jsonl", "cleanup-routine.log",
				"telemetry/", "sessions/", "session-env/", "backups/", "dos-scratch/", "debug/",
			},
			Excludes: []string{"*.tmp"},
			Note:     "user-level fleet ledgers (shadow switch ledger, memory co-travel) + harness telemetry/session state",
		})
	}
	return srcs
}

// DefaultScratchpadCapBytes bounds the opt-in scratchpad tier. The %TEMP%/claude
// tree is multi-GB (2.7 GB / 77k files witnessed 2026-07-03), new-dir-per-session
// and mostly cold within a day, so the opt-in defaults to an aggressive 256 MiB
// cap — enough to bank a handful of recent sessions' kept artifacts, never the
// whole re-derivable tree.
const DefaultScratchpadCapBytes = 256 << 20

// AllProjectSources returns one harness-store-<slug> source per Claude Code
// project directory under ~/.claude/projects, EXCLUDING this repo's own store —
// DefaultSources already captures that as the canonical "harness-store" source,
// and a second source over the same directory would duplicate the fak project
// store into the vault. This is the `--all-projects` expansion: every other repo
// the harness touched (transcripts + auto-memory) rides along, each under its own
// by-source/harness-store-<slug>/ subtree.
//
// A missing ~/.claude/projects (or an empty home) is the valid-empty posture:
// no extra sources, never an error.
func AllProjectSources(repoRoot, home string) []Source {
	if home == "" {
		return nil
	}
	projectsDir := filepath.Join(home, ".claude", "projects")
	ents, err := os.ReadDir(projectsDir)
	if err != nil {
		return nil
	}
	ownSlug := harnessProjectSlug(repoRoot)
	var out []Source
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		slug := e.Name()
		if slug == ownSlug {
			continue // the fak store is already captured as "harness-store" — no duplicate
		}
		out = append(out, Source{
			ID:   "harness-store-" + slug,
			Root: filepath.Join(projectsDir, slug),
			Note: "Claude Code session store for project " + slug + " (--all-projects)",
		})
	}
	return out
}

// ScratchpadSource returns the opt-in scratchpad tier: the harness's
// per-session working dirs under %TEMP%/claude. It is DEFAULT-EXCLUDED (only
// layered in when the caller opts in) because the tree is re-derivable working
// files, and it is bounded by an aggressive cumulative-size cap (capBytes, or
// DefaultScratchpadCapBytes when non-positive) so the opt-in can never pull the
// whole multi-GB tree into the vault.
func ScratchpadSource(tempDir string, capBytes int64) Source {
	if capBytes <= 0 {
		capBytes = DefaultScratchpadCapBytes
	}
	return Source{
		ID:       "harness-scratchpad",
		Root:     filepath.Join(tempDir, "claude"),
		MaxBytes: capBytes,
		Note:     "opt-in %TEMP%/claude scratchpad (re-derivable; excluded by default, bounded by an aggressive size cap)",
	}
}

// harnessProjectSlug maps a repo path to the Claude Code projects-dir name:
// every ':', '\', '/' becomes '-' (C:\work\fak -> C--work-fak).
func harnessProjectSlug(repoRoot string) string {
	return strings.Map(func(r rune) rune {
		switch r {
		case ':', '\\', '/':
			return '-'
		}
		return r
	}, repoRoot)
}

// matchPattern tests relPath (forward-slash form) against one pattern: a
// trailing-'/' pattern is a subtree prefix, anything else a base-name glob.
func matchPattern(pat, relPath string) bool {
	if strings.HasSuffix(pat, "/") {
		return strings.HasPrefix(relPath, pat)
	}
	base := relPath
	if i := strings.LastIndexByte(relPath, '/'); i >= 0 {
		base = relPath[i+1:]
	}
	ok, _ := filepath.Match(pat, base)
	return ok
}

// excluded reports whether relPath is skipped for the source: per-source
// excludes, then the common and credential lists.
func excluded(src Source, relPath string) bool {
	for _, pat := range src.Excludes {
		if matchPattern(pat, relPath) {
			return true
		}
	}
	for _, pat := range commonExcludes {
		if matchPattern(pat, relPath) {
			return true
		}
	}
	for _, pat := range credentialExcludes {
		if matchPattern(pat, relPath) {
			return true
		}
	}
	return false
}

// admitted reports whether a FILE passes the source's include list (an empty
// list admits everything). Excludes are checked separately and always win.
func admitted(src Source, relPath string) bool {
	if len(src.Includes) == 0 {
		return true
	}
	for _, pat := range src.Includes {
		if matchPattern(pat, relPath) {
			return true
		}
	}
	return false
}

// includesCouldReach reports whether a DIRECTORY (relDir, trailing slash) may
// contain admitted files, so the walk can prune subtrees an include-listed
// source can never capture from. Any bare-glob include forces full descent
// (a base name can match at any depth).
func includesCouldReach(src Source, relDir string) bool {
	if len(src.Includes) == 0 {
		return true
	}
	for _, pat := range src.Includes {
		if !strings.HasSuffix(pat, "/") {
			return true
		}
		if strings.HasPrefix(pat, relDir) || strings.HasPrefix(relDir, pat) {
			return true
		}
	}
	return false
}
